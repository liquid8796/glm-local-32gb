"""Synthetic NVFP4 metadata -> selected local/HTTP reader -> CPU/CUDA projection."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glm_local import execution, nvfp4_execution
from glm_local.catalogue_reader import SelectedCatalogueReader
from glm_local.checkpoint_http import HttpMetadataSource
from glm_local.execution import ShardDescriptor, TensorDescriptor, build_projection_descriptor
from glm_local.nvfp4_execution import (NVFP4ProjectionDescriptor, execute_nvfp4_projection,
                                      reference_nvfp4_projection)
from glm_local.nvfp4_kernels import CudaNVFP4TileBackend, NativeNVFP4CpuBackend
from glm_local.projection_remote import RemoteProjectionReader
from glm_local.residency import BudgetExceededError, ReservationLedger
from checkpoint_test_helpers import MODEL, REVISION, make_header
from test_nvfp4_kernels import oracle
from test_nvfp4_schema import prepare_nvfp4_fixture
from test_projection_remote import ShardOpener


WEIGHT = "model.layers.1.mlp.experts.0.gate_proj.weight"


def payload(name, dtype, shape):
    count = math.prod(shape)
    if dtype == "U8":
        rows, bytecols = shape
        return bytes(((row * 3 + col * 7) % 16) | (((row * 11 + col * 5 + 1) % 16) << 4)
                     for row in range(rows) for col in range(bytecols))
    if dtype == "F8_E4M3":
        rows, blockcols = shape
        return bytes((row * 5 + col * 11) % 127 for row in range(rows) for col in range(blockcols))
    if dtype == "F32":
        value = 0.0137 if name.endswith(".weight_scale_2") else 3.25 if name.endswith(".input_scale") else 0.03125
        return struct.pack("<f", value) * count
    if dtype == "BF16":
        return b"\x80\x3f" * count
    raise AssertionError(dtype)


def ragged_fixture(root, rows=129, cols=144):
    """Explicit generic matrix fixture; does not claim eligibility as a full GLM graph."""
    root = Path(root)
    directory = root / "ragged"
    directory.mkdir()
    stem = WEIGHT[:-len(".weight")]
    specs = [(WEIGHT, "U8", (rows, cols // 2)),
             (stem + ".weight_scale", "F8_E4M3", (rows, cols // 16)),
             (stem + ".weight_scale_2", "F32", ()), (stem + ".input_scale", "F32", ())]
    tensors, shards = [], []
    for number, (name, dtype, shape) in enumerate(specs):
        filename = f"ragged-{number}.safetensors"
        raw, size, nbytes = make_header([(name, dtype, shape)])
        (directory / filename).write_bytes(raw + payload(name, dtype, shape))
        tensors.append(TensorDescriptor(name, dtype, shape, nbytes, (0, nbytes), filename))
        shards.append(ShardDescriptor(filename, size, len(raw), hashlib.sha256(raw).hexdigest()))
    descriptor = NVFP4ProjectionDescriptor(tensors[0], tensors[1], tuple(shards), MODEL, REVISION,
        "synthetic fixture", "0" * 64, "synthetic catalogue", "1" * 64,
        "synthetic snapshot", "2" * 64, False, "SYNTHETIC", tensors[2], tensors[3])
    return directory, descriptor


class Kernel:
    """Independent mathematical tile oracle, injected as a bounded test backend."""
    max_operations = 256

    def __init__(self):
        self.operations, self.calls = 0, []

    def matvec_nvfp4_tile(self, packed, rows, cols, scales, vector, global_scale):
        self.operations += 1
        self.calls.append((rows, cols))
        return oracle(packed, rows, cols, scales, vector, global_scale)


class Gate:
    def __init__(self):
        self.calls = 0

    def before_submit(self):
        self.calls += 1


class NVFP4ExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory, data = prepare_nvfp4_fixture(self.root,
            config_overrides={"hidden_size": 144, "moe_intermediate_size": 144}, payload_factory=payload)
        self.settings = data["settings"]
        self.descriptor = build_projection_descriptor(self.root, self.settings, WEIGHT)
        self.vector = [((index * 7) % 23 - 11) * 0.137 for index in range(144)]

    def source(self, directory=None, change=None):
        opener = ShardOpener(directory or self.directory, change)
        return HttpMetadataSource(MODEL, REVISION, opener=opener), opener

    def test_complete_metadata_selects_four_bound_tensors_with_logical_shape(self):
        descriptor = self.descriptor
        self.assertIsInstance(descriptor, NVFP4ProjectionDescriptor)
        self.assertEqual(descriptor.quant_format, "nvfp4")
        self.assertEqual(descriptor.logical_shape, (144, 144))
        self.assertEqual(descriptor.weight.shape, (144, 72))
        self.assertEqual(descriptor.scale.shape, (144, 9))
        self.assertEqual((descriptor.global_scale.shape, descriptor.input_scale.shape), ((), ()))
        self.assertEqual(len(descriptor.tensors), 4)
        self.assertEqual(len(descriptor.shards), 3)
        self.assertNotEqual(descriptor.weight.shard, descriptor.global_scale.shard)
        self.assertTrue(descriptor.architecture_metadata_verified)
        document = descriptor.to_dict()
        self.assertEqual(document["activation_quantization"], "none")
        self.assertEqual(document["logical_shape"], [144, 144])
        self.assertFalse(document["native_w4a4_parity_verified"])
        for flag in execution.FULL_MODEL_FLAGS:
            self.assertFalse(document[flag])

    def test_cpu_hybrid_reference_use_logical_tiles_and_four_tensor_two_fd_lru(self):
        for backend in ("cpu", "hybrid"):
            cpu, gpu, gate = Kernel(), Kernel(), Gate()
            with self.subTest(backend=backend), SelectedCatalogueReader(self.directory, self.descriptor) as reader:
                output, counts = execute_nvfp4_projection(reader, self.descriptor, self.vector,
                    backend=backend, cpu=cpu, gpu=gpu, gate=gate)
                with patch.object(nvfp4_execution.NVFP4BlockMatrix, "read_block",
                                  side_effect=AssertionError("reference cannot reuse tile adapter")):
                    reference = reference_nvfp4_projection(reader, self.descriptor, self.vector)
                self.assertEqual(output, reference)
                self.assertEqual((counts["cpu_tiles"], counts["gpu_tiles"]), (2, 2) if backend == "hybrid" else (4, 0))
                self.assertEqual(gate.calls, counts["gpu_tiles"])
                self.assertEqual(counts["packed_weight_bytes"], 144 * 72)
                self.assertEqual(counts["block_scale_bytes"], 144 * 9)
                self.assertEqual(counts["max_packed_tile_bytes"], 8192)
                self.assertEqual(counts["activation_quantization"], "none")
                self.assertFalse(counts["native_w4a4_parity_verified"])
                self.assertIn((16, 16), cpu.calls + gpu.calls)
                stats = reader.stats()
                self.assertEqual(stats["selected_tensors"], 4)
                self.assertEqual(stats["selected_shards"], 3)
                self.assertLessEqual(stats["peak_open_shards"], 2)
                self.assertGreater(stats["lru_evictions"], 2)
                self.assertLessEqual(stats["max_actual_read_bytes"], 65536)

    def test_generic_ragged129_by144_across_four_shards_keeps_both_paths(self):
        directory, descriptor = ragged_fixture(self.root)
        cpu, gpu, gate = Kernel(), Kernel(), Gate()
        with SelectedCatalogueReader(directory, descriptor) as reader:
            actual, counts = execute_nvfp4_projection(reader, descriptor, self.vector,
                backend="hybrid", cpu=cpu, gpu=gpu, gate=gate)
            self.assertEqual(actual, reference_nvfp4_projection(reader, descriptor, self.vector))
            self.assertEqual(len(actual), 129)
            self.assertEqual(cpu.calls, [(128, 128), (128, 16)])
            self.assertEqual(gpu.calls, [(1, 128), (1, 16)])
            self.assertEqual((counts["cpu_tiles"], counts["gpu_tiles"], gate.calls), (2, 2, 2))
            self.assertEqual(reader.stats()["selected_shards"], 4)
            self.assertEqual(reader.stats()["peak_open_shards"], 2)

    def test_remote_download_requests_all_four_exact_payloads_and_enforces_two_open_files(self):
        source, opener = self.source()
        destination = self.root / "remote"
        descriptor = self.descriptor
        with RemoteProjectionReader(descriptor, destination, _transport=source) as reader:
            actual, _ = execute_nvfp4_projection(reader, descriptor, self.vector, cpu=Kernel())
            self.assertEqual(actual, reference_nvfp4_projection(reader, descriptor, self.vector))
            self.assertEqual(len(opener.requests), len(descriptor.shards) + 4)
            for request, tensor in zip(opener.requests[len(descriptor.shards):], descriptor.tensors):
                shard = next(item for item in descriptor.shards if item.name == tensor.shard)
                start = shard.header_bytes + tensor.data_offsets[0]
                self.assertEqual(request.get_header("Range"), f"bytes={start}-{start + tensor.nbytes - 1}")
                self.assertEqual(request.get_header("If-match"), '"invented"')
            stats = reader.stats()
            self.assertEqual(stats["max_open_shards_observed"], 2)
            self.assertLessEqual(stats["max_actual_read_bytes"], 65536)
            self.assertEqual(stats["network"]["tensor_payload_bytes_requested"], sum(item.nbytes for item in descriptor.tensors))
            self.assertEqual(stats["network"]["body_bytes_read"],
                sum(item.nbytes for item in descriptor.tensors) + sum(item.header_bytes for item in descriptor.shards))
            receipt = json.loads((destination / "receipt.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertEqual(len(receipt["tensors"]), 4)
            self.assertFalse(receipt["full_checkpoint_downloaded"])
            for tensor in receipt["tensors"]:
                self.assertEqual(hashlib.sha256((destination / tensor["file"]).read_bytes()).hexdigest(), tensor["sha256"])
        self.assertEqual(reader._open, {})

    def test_remote_budget_includes_global_input_scales_before_requests(self):
        needed = sum(tensor.nbytes for tensor in self.descriptor.tensors) + sum(shard.header_bytes for shard in self.descriptor.shards)
        source, opener = self.source()
        destination = self.root / "too-small"
        with self.assertRaisesRegex(ValueError, "budget"):
            RemoteProjectionReader(self.descriptor, destination, budget_bytes=needed - 1, _transport=source)
        self.assertEqual(opener.requests, [])
        self.assertFalse(destination.exists())

    def test_remote_global_or_input_etag_change_and_slice_mutation_fail(self):
        header_requests = len(self.descriptor.shards)
        for tensor_index in (2, 3):
            def change(number, response):
                if number == header_requests + tensor_index + 1:
                    response.headers.replace_header("ETag", '"changed"')
            source, opener = self.source(change=change)
            destination = self.root / f"etag-{tensor_index}"
            with self.subTest(tensor_index=tensor_index), self.assertRaises(ValueError):
                RemoteProjectionReader(self.descriptor, destination, _transport=source)
            self.assertEqual(opener.responses[-1].requested_reads, [])
            self.assertFalse((destination / "receipt.json").exists())
        source, _ = self.source()
        with RemoteProjectionReader(self.descriptor, self.root / "modified", _transport=source) as reader:
            target = reader.directory / "tensor-3.bin"
            target.write_bytes(b"\0" * 3)
            with self.assertRaisesRegex(ValueError, "changed"):
                execute_nvfp4_projection(reader, self.descriptor, self.vector, cpu=Kernel())

    def test_global_and_input_scale_payload_validation_precedes_submission(self):
        for tensor in (self.descriptor.global_scale, self.descriptor.input_scale):
            path = self.directory / tensor.shard
            original = path.read_bytes()
            proof = next(item for item in self.descriptor.shards if item.name == tensor.shard)
            offset = proof.header_bytes + tensor.data_offsets[0]
            for value in (0, -1, math.nan, math.inf):
                path.write_bytes(original[:offset] + struct.pack("<f", value) + original[offset + 4:])
                with self.subTest(name=tensor.name, value=value), SelectedCatalogueReader(self.directory, self.descriptor) as reader:
                    cpu = Kernel()
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        execute_nvfp4_projection(reader, self.descriptor, self.vector, cpu=cpu)
                    self.assertEqual(cpu.operations, 0)
                    with self.assertRaises(ValueError):
                        reference_nvfp4_projection(reader, self.descriptor, self.vector)
            path.write_bytes(original)

    def test_changing_valid_input_scale_never_changes_weight_only_result(self):
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            expected = reference_nvfp4_projection(reader, self.descriptor, self.vector)
        tensor = self.descriptor.input_scale
        proof = next(shard for shard in self.descriptor.shards if shard.name == tensor.shard)
        path = self.directory / tensor.shard
        raw, offset = path.read_bytes(), proof.header_bytes + tensor.data_offsets[0]
        path.write_bytes(raw[:offset] + struct.pack("<f", 123) + raw[offset + 4:])
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            actual, _ = execute_nvfp4_projection(reader, self.descriptor, self.vector, cpu=Kernel())
            self.assertEqual(actual, expected)

    def test_launch_budget_and_gate_are_enforced_and_logical_vector_length_is_required(self):
        cpu, gpu, gate = Kernel(), Kernel(), Gate()
        gpu.operations = 255
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            with self.assertRaisesRegex(ValueError, "CUDA operation budget"):
                execute_nvfp4_projection(reader, self.descriptor, self.vector, backend="hybrid", cpu=cpu, gpu=gpu, gate=gate)
            self.assertEqual(reader.stats()["tensor_read_bytes"], 0)
            self.assertEqual(cpu.operations, 0)
            gpu.operations = 0
            gate.before_submit = lambda: (_ for _ in ()).throw(RuntimeError("telemetry unavailable"))
            with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
                execute_nvfp4_projection(reader, self.descriptor, self.vector, backend="hybrid", cpu=cpu, gpu=gpu, gate=gate)
            self.assertEqual((cpu.operations, gpu.operations), (2, 0))
            for vector in ([0] * 72, [math.nan] * 144, [True] * 144):
                cpu = Kernel()
                with self.assertRaises(ValueError):
                    execute_nvfp4_projection(reader, self.descriptor, vector, cpu=cpu)
                self.assertEqual(cpu.operations, 0)

    def test_ledger_reserves_before_payload_and_releases_after_gpu_gate_failure(self):
        required = 3 * (128 * 128 // 2 + 128 * 128 // 16) + 4 * (144 + 144 + 4 * 128) + 8
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            small = ReservationLedger(required - 1)
            with self.assertRaises(BudgetExceededError):
                execute_nvfp4_projection(reader, self.descriptor, self.vector, cpu=Kernel(), ledger=small)
            self.assertEqual(reader.stats()["tensor_read_bytes"], 0)
            ledger = ReservationLedger(required, 10240)
            execute_nvfp4_projection(reader, self.descriptor, self.vector, backend="hybrid",
                cpu=Kernel(), gpu=Kernel(), gate=Gate(), ledger=ledger)
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
            self.assertEqual(ledger.snapshot()["cpu"]["peak_bytes"], required)
            # GPU receives the last16rows, so its actual largest tile is16x128.
            self.assertEqual(ledger.snapshot()["cuda"]["peak_bytes"], 16 * 64 + 16 * 8 + 4 * (16 + 128))
            gate = Gate()
            gate.before_submit = lambda: (_ for _ in ()).throw(RuntimeError("denied"))
            with self.assertRaisesRegex(RuntimeError, "denied"):
                execute_nvfp4_projection(reader, self.descriptor, self.vector, backend="hybrid",
                    cpu=Kernel(), gpu=Kernel(), gate=gate, ledger=ledger)
            self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_selection_preflights_nvfp4_vector_and_tile_policy(self):
        for rows, cols, block_rows, block_cols, expected in (
            (65537, 16, 513, 1, "vector"), (32768, 32784, 256, 257, "tile")):
            bounds = SimpleNamespace(rows=rows, cols=cols, block_rows=block_rows, block_cols=block_cols)
            with patch("glm_local.nvfp4_blocks.NVFP4BlockMatrix", return_value=bounds):
                with self.subTest(rows=rows, cols=cols), self.assertRaisesRegex(ValueError, expected):
                    build_projection_descriptor(self.root, self.settings, WEIGHT)

    def test_execution_preflights_forged_metadata_bounds_before_payload(self):
        for rows, cols, expected in ((65537, 144, "vector"), (32768, 32784, "tile")):
            weight = replace(self.descriptor.weight, shape=(rows, cols // 2), nbytes=rows * cols // 2)
            scale = replace(self.descriptor.scale, shape=(rows, cols // 16), nbytes=rows * cols // 16)
            descriptor = replace(self.descriptor, weight=weight, scale=scale)
            reader = SimpleNamespace(tensors={item.name: item.info() for item in descriptor.tensors})
            with self.subTest(rows=rows, cols=cols), self.assertRaisesRegex(ValueError, expected):
                execute_nvfp4_projection(reader, descriptor, [], cpu=Kernel())

    def test_all_four_descriptor_metadata_entries_are_bound_and_outputs_checked(self):
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            for role in ("weight", "scale", "global_scale", "input_scale"):
                item = getattr(self.descriptor, role)
                descriptor = replace(self.descriptor, **{role: replace(item, data_offsets=(1, item.nbytes + 1))})
                with self.subTest(role=role), self.assertRaisesRegex(ValueError, "metadata differs"):
                    execute_nvfp4_projection(reader, descriptor, self.vector, cpu=Kernel())
            self.assertEqual(reader.stats()["tensor_read_bytes"], 0)
            for output in (None, [], [math.inf] * 128, [True] * 128):
                cpu = Kernel()
                cpu.matvec_nvfp4_tile = lambda *_args, result=output: result
                with self.subTest(output=str(output)[:30]), self.assertRaises(ValueError):
                    execute_nvfp4_projection(reader, self.descriptor, self.vector, cpu=cpu)


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "set GLM_TEST_NATIVE=1")
class ActualNVFP4ProjectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory, data = prepare_nvfp4_fixture(self.root,
            config_overrides={"hidden_size": 144, "moe_intermediate_size": 144}, payload_factory=payload)
        self.descriptor = build_projection_descriptor(self.root, data["settings"], WEIGHT)
        self.vector = [(index % 17 - 8) * 0.0137 for index in range(144)]
        self.cpu = NativeNVFP4CpuBackend()
        self.addCleanup(self.cpu.close)

    def test_actual_cpu_complete_metadata_and_http_slices_match_independent_reference(self):
        opener = ShardOpener(self.directory)
        source = HttpMetadataSource(MODEL, REVISION, opener=opener)
        with RemoteProjectionReader(self.descriptor, self.root / "actual-remote", _transport=source) as reader:
            actual, counts = execute_nvfp4_projection(reader, self.descriptor, self.vector, cpu=self.cpu)
            self.assertEqual(actual, reference_nvfp4_projection(reader, self.descriptor, self.vector))
            self.assertEqual(counts["cpu_tiles"], 4)

    @unittest.skipUnless(os.environ.get("GLM_TEST_CUDA") == "1", "set GLM_TEST_CUDA=1")
    def test_actual_cpu_cuda_hybrid_complete_and_generic_ragged_projections(self):
        generic_directory, generic_descriptor = ragged_fixture(self.root)
        with CudaNVFP4TileBackend(max_operations=4) as gpu:
            for directory, descriptor in ((self.directory, self.descriptor), (generic_directory, generic_descriptor)):
                gate = Gate()
                with self.subTest(rows=descriptor.logical_shape[0]), SelectedCatalogueReader(directory, descriptor) as reader:
                    actual, counts = execute_nvfp4_projection(reader, descriptor, self.vector,
                        backend="hybrid", cpu=self.cpu, gpu=gpu, gate=gate)
                    self.assertEqual(actual, reference_nvfp4_projection(reader, descriptor, self.vector))
                    self.assertEqual((counts["cpu_tiles"], counts["gpu_tiles"], gate.calls), (2, 2, 2))
                    self.assertFalse(counts["native_w4a4_parity_verified"])
            self.assertEqual(gpu.operations, 4)


if __name__ == "__main__":
    unittest.main()
