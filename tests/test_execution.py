"""Invented multi-tile projections; no remote checkpoint or native dependency."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from glm_local import execution
from glm_local.catalogue_reader import SelectedCatalogueReader
from glm_local.checkpoint_check import execute_metadata, publish_report
from glm_local.checkpoint_snapshot import write_json
from glm_local.cuda_probe import CudaProbeError, CudaTileBackend
from checkpoint_test_helpers import MODEL, REVISION, NAME, SCALE, MemorySource, checkpoint, make_header, settings
import test_cuda_probe as cuda_tests


def projection_fixture(root, *, rows=257, cols=259, extra_tensors=0, scale_value=None):
    """Capture genuine producer artifacts, then materialize small invented payloads."""
    root = Path(root)
    config = deepcopy(checkpoint()["config"])
    config.update(hidden_size=cols, intermediate_size=rows)
    specs = {"weights.safetensors": [(NAME, "F8_E4M3", [rows, cols]), ("bf16.aux", "BF16", [1])],
             "scales.safetensors": [(SCALE, "F32", [(rows + 127)//128, (cols + 127)//128])]}
    for index in range(extra_tensors):
        specs.setdefault(f"extra-{index//3000}.safetensors", []).append(
            (f"aux.{index:05d}." + "x" * 160, "F32", [0]))
    headers, mapping, siblings, total = {}, {}, [], 0
    directory = root / "model"
    directory.mkdir(parents=True)
    for filename, entries in specs.items():
        raw, size, payload_count = make_header(entries)
        headers[filename] = raw, size
        siblings.append({"rfilename": filename, "size": size})
        mapping.update({name: filename for name, _, _ in entries})
        total += payload_count
        payload = bytearray(payload_count)
        header = json.loads(raw[8:])
        for name, dtype, shape in entries:
            begin, end = header[name]["data_offsets"]
            if name == NAME:
                codes = (0x10, 0x21, 0x35, 0x38, 0x42, 0x91, 0xA4, 0xB8, 0xC1)
                payload[begin:end] = bytes(codes[(row * 7 + col * 3) % len(codes)]
                    for row in range(rows) for col in range(cols))
            elif name == SCALE:
                count = (end-begin)//4
                payload[begin:end] = b"".join(struct.pack("<f", scale_value if scale_value is not None
                                                        else 0.125 * (index+1)) for index in range(count))
        (directory / filename).write_bytes(raw + payload)
    expected = {"model_id": MODEL, "revision": REVISION,
                "weights": [{"name": item["rfilename"], "bytes": item["size"]} for item in siblings],
                "architecture": {**{key: value for key, value in config.items() if key != "quantization_config"},
                                 "quantization": deepcopy(config["quantization_config"])}}
    data = {"model": {"id": MODEL, "sha": REVISION, "siblings": siblings}, "config": config,
            "index": {"metadata": {"total_size": total}, "weight_map": mapping},
            "headers": headers, "expected": expected}
    write_json(root / "docs/model-metadata.json", expected)
    run = root / "reports/metadata/invented"
    run.mkdir(parents=True)
    with redirect_stdout(io.StringIO()):
        report, code = execute_metadata(root, settings(), {"max_shards": 512, "budget_mib": 64, "offline": None},
                                        run, source=MemorySource(data))
        publish_report(root, run, report, code)
    if code not in (0, 3):
        raise AssertionError(report)
    return directory, report, run


class Kernel:
    def __init__(self):
        self.calls = []
        self.operations = 0

    def matvec_tile(self, weights, rows, cols, vector, scale):
        self.calls.append((rows, cols))
        self.operations += 1
        return cuda_tests.matvec_oracle(weights, rows, cols, vector, scale)


class Gate:
    def __init__(self):
        self.calls = 0

    def before_submit(self):
        self.calls += 1


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directory, self.report, self.run = projection_fixture(self.root)
        self.descriptor = execution.build_projection_descriptor(self.root, settings(), NAME)
        self.vector = [((index * 11) % 23 - 11) / 32 for index in range(259)]

    def test_selected_descriptor_does_not_require_complete_architecture(self):
        value = self.descriptor.to_dict()
        self.assertTrue(value["selected_projection_eligible"])
        self.assertFalse(value["architecture_metadata_verified"])
        self.assertEqual(value["weight"]["dtype"], "F8_E4M3")
        self.assertEqual(value["weight"]["shape"], (257, 259))
        self.assertEqual(value["scale"]["shape"], (3, 3))
        self.assertNotEqual(value["weight"]["shard"], value["scale"]["shard"])
        self.assertTrue(all(value[key] is False for key in execution.FULL_MODEL_FLAGS))
        self.assertTrue(all(len(shard["header_sha256"]) == 64 for shard in value["shards"]))
        self.assertEqual(value["scale_semantics"], "decoded_e4m3fn_times_stored_scale")

    def test_ragged_cross_shard_cpu_and_hybrid_match_independent_row_reference(self):
        for backend in ("cpu", "hybrid"):
            cpu, gpu, gate = Kernel(), Kernel(), Gate()
            with self.subTest(backend=backend), SelectedCatalogueReader(self.directory, self.descriptor) as reader:
                actual, counts = execution.execute_projection(reader, self.descriptor, self.vector,
                    backend=backend, cpu=cpu, gpu=gpu, gate=gate)
                with patch.object(execution.FP8BlockMatrix, "read_block", side_effect=AssertionError("oracle cannot use block path")):
                    expected = execution.reference_projection(reader, self.descriptor, self.vector)
                self.assertTrue(execution.compare_projection(actual, expected)["passed"])
                self.assertEqual(counts["cpu_tiles"] + counts["gpu_tiles"], 9)
                self.assertEqual(counts["gpu_tiles"], 3 if backend == "hybrid" else 0)
                self.assertEqual(gate.calls, counts["gpu_tiles"])
                self.assertEqual(counts["fp8_bytes"], 257 * 259)
                self.assertEqual(counts["scale_bytes"], 36)
                self.assertEqual(counts["max_packed_tile_bytes"], 16384)
                self.assertEqual(counts["retained_decoded_weight_bytes"], 0)
                self.assertIn((1, 3), cpu.calls)

    def test_malformed_scales_fail_before_kernel_submission(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            path = self.directory / self.descriptor.scale.shard
            original = path.read_bytes()
            proof = next(item for item in self.descriptor.shards if item.name == path.name)
            path.write_bytes(original[:proof.header_bytes] + struct.pack("<f", value) + original[proof.header_bytes+4:])
            with self.subTest(value=value), SelectedCatalogueReader(self.directory, self.descriptor) as reader:
                cpu = Kernel()
                with self.assertRaisesRegex(ValueError, "scale"):
                    execution.execute_projection(reader, self.descriptor, self.vector, cpu=cpu)
                self.assertEqual(cpu.operations, 0)
                with self.assertRaisesRegex(ValueError, "scale"):
                    execution.reference_projection(reader, self.descriptor, self.vector)
            path.write_bytes(original)

    def test_invalid_vectors_and_kernel_outputs_are_rejected(self):
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            for vector in ([0]*258, [float("nan")]*259, [True]*259, "x"*259):
                cpu = Kernel()
                with self.assertRaises(ValueError):
                    execution.execute_projection(reader, self.descriptor, vector, cpu=cpu)
                self.assertEqual(cpu.operations, 0)
            cpu = Kernel()
            cpu.matvec_tile = lambda w, r, c, v, s: [float("inf")] * r
            with self.assertRaisesRegex(ValueError, "kernel output"):
                execution.execute_projection(reader, self.descriptor, self.vector, cpu=cpu)

    def test_gpu_gate_error_prevents_gpu_launch_and_budget_is_preflighted(self):
        cpu, gpu, gate = Kernel(), Kernel(), Gate()
        gate.before_submit = lambda: (_ for _ in ()).throw(RuntimeError("no telemetry"))
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            with self.assertRaisesRegex(RuntimeError, "telemetry"):
                execution.execute_projection(reader, self.descriptor, self.vector, backend="hybrid", cpu=cpu, gpu=gpu, gate=gate)
            self.assertEqual(gpu.operations, 0)
            gpu.operations = 255
            before = reader.stats()["tensor_read_bytes"]
            with self.assertRaisesRegex(ValueError, "CUDA launch"):
                execution.execute_projection(reader, self.descriptor, self.vector, backend="hybrid", cpu=cpu, gpu=gpu, gate=Gate())
            self.assertEqual(reader.stats()["tensor_read_bytes"], before)
            gpu.max_operations = 512
            execution.execute_projection(reader, self.descriptor, self.vector, backend="hybrid", cpu=cpu, gpu=gpu, gate=Gate())
            self.assertEqual(gpu.operations, 258)

    def test_tampered_report_catalogue_snapshot_and_index_fail_closed(self):
        for path in (self.root / "reports/metadata-latest.json", self.run / "tensor-catalogue.jsonl",
                     self.run / "evidence/config.json", self.run / "evidence/model.safetensors.index.json"):
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(path=path.name):
                # A report whitespace change is allowed but produces a fresh source digest.
                if path.name == "metadata-latest.json":
                    updated = execution.build_projection_descriptor(self.root, settings(), NAME)
                    self.assertNotEqual(updated.source_sha256, self.descriptor.source_sha256)
                else:
                    with self.assertRaises(ValueError):
                        execution.build_projection_descriptor(self.root, settings(), NAME)
            path.write_bytes(original)

    def test_bad_dtype_shape_layout_and_missing_scale_are_rejected(self):
        for weight, scale in ((replace(self.descriptor.weight, dtype="F32"), self.descriptor.scale),
                              (self.descriptor.weight, replace(self.descriptor.scale, dtype="BF16")),
                              (self.descriptor.weight, replace(self.descriptor.scale, shape=(1, 1)))):
            with self.assertRaises(ValueError):
                execution._validate_pair(weight, scale)
        with self.assertRaisesRegex(ValueError, "absent"):
            execution.build_projection_descriptor(self.root, settings(), NAME, "missing")
        report = deepcopy(self.report)
        report["config"]["quantization_config"]["weight_block_size"] = [64, 128]
        write_json(self.root / "reports/metadata-latest.json", report)
        with self.assertRaisesRegex(ValueError, "128x128"):
            execution.build_projection_descriptor(self.root, settings(), NAME)

    def test_deceptive_huge_metadata_is_rejected_before_vector_or_payload_allocation(self):
        weight = replace(self.descriptor.weight, shape=(65537, 259), nbytes=65537*259, data_offsets=(0, 65537*259))
        scale = replace(self.descriptor.scale, shape=(513, 3), nbytes=513*3*4, data_offsets=(0, 513*3*4))
        with self.assertRaisesRegex(ValueError, "vector bound"):
            execution._validate_pair(weight, scale)

    def test_ledger_preflights_before_payload_and_releases_after_success_or_failure(self):
        from glm_local.residency import ReservationLedger, BudgetExceededError
        required = 3 * 128 * 128 + 4 * (257 + 259 + 4 * 128)
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            small = ReservationLedger(required - 1)
            with self.assertRaises(BudgetExceededError):
                execution.execute_projection(reader, self.descriptor, self.vector, cpu=Kernel(), ledger=small)
            self.assertEqual(reader.stats()["tensor_read_bytes"], 0)
            self.assertEqual(small.snapshot()["active_leases"], 0)
            ledger = ReservationLedger(required, 17408)
            execution.execute_projection(reader, self.descriptor, self.vector, backend="hybrid",
                cpu=Kernel(), gpu=Kernel(), gate=Gate(), ledger=ledger)
            snapshot = ledger.snapshot()
            self.assertEqual(snapshot["active_leases"], 0)
            self.assertEqual(snapshot["cpu"]["peak_bytes"], required)
            self.assertEqual(snapshot["cuda"]["peak_bytes"], 17408)
            gate = Gate()
            gate.before_submit = lambda: (_ for _ in ()).throw(RuntimeError("denied"))
            with self.assertRaisesRegex(RuntimeError, "denied"):
                execution.execute_projection(reader, self.descriptor, self.vector, backend="hybrid",
                    cpu=Kernel(), gpu=Kernel(), gate=gate, ledger=ledger)
            self.assertEqual(ledger.snapshot()["active_leases"], 0)


class CudaExecutionBudgetTests(unittest.TestCase):
    def test_invalid_opt_in_budget_fails_before_driver_load(self):
        with patch("glm_local.cuda_probe._load_driver") as load:
            for value in (0, True, 65537, -1, "512"):
                with self.assertRaises(ValueError):
                    CudaTileBackend(max_operations=value)
            load.assert_not_called()

    def test_opt_in_extends_default_but_remains_enforced(self):
        gpu, events = cuda_tests.CudaFailureCleanupTests().backend()
        gpu.max_operations = 512
        gpu.operations = 256
        gpu.matvec_tile(b"\x38", 1, 1, [1.0], 1.0)
        self.assertEqual(gpu.operations, 257)
        gpu.operations = 512
        events.clear()
        with self.assertRaisesRegex(CudaProbeError, "512"):
            gpu.matvec_tile(b"\x38", 1, 1, [1.0], 1.0)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
