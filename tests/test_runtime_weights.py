"""Small complete metadata -> local payload -> runtime weight workflow."""
from array import array
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from checkpoint_test_helpers import MODEL, REVISION, MemorySource, encode, make_header, settings
from test_architecture_mapper import fixture_catalogue, fixture_config
from glm_local.checkpoint_check import execute_metadata, publish_report
from glm_local.checkpoint_snapshot import write_json
from glm_local.execution import _reference_decode
from glm_local.residency import ReservationLedger
from glm_local.runtime_weights import FullCatalogueReader, RuntimeWeights
from glm_local.safetensor_reader import SafeTensorError
from glm_local.sharded_safetensors import MAX_INDEX_BYTES, MAX_INDEX_TENSORS


class CpuKernel:
    def matvec_tile(self, weights, rows, cols, vector, scale):
        return [sum(_reference_decode(weights[r * cols + c]) * scale * vector[c]
                    for c in range(cols)) for r in range(rows)]


class Gate:
    def __init__(self):
        self.calls = 0

    def before_submit(self):
        self.calls += 1


class GpuKernel(CpuKernel):
    operations = 0
    max_operations = 65536

    def matvec_tile(self, *args):
        self.operations += 1
        return super().matvec_tile(*args)


def prepare_runtime_fixture(root, *, output_dtype="BF16", wide=False, config_overrides=None, payload_factory=None):
    config, records = fixture_config(), fixture_catalogue()
    config.update(config_overrides or {})
    config["vocab_size"] = (config_overrides or {}).get("vocab_size", 257)
    if wide:
        config["intermediate_size"] = 129
    for record in records:
        name = record["name"]
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            record["shape"] = [config["vocab_size"], 16]
        if name == "lm_head.weight":
            record["dtype"] = output_dtype
        if wide and name.startswith("model.layers.0.mlp."):
            down, scale = ".down_proj." in name, name.endswith("_scale_inv")
            record["shape"] = list(((1, 2) if down else (2, 1)) if scale
                                   else ((16, 129) if down else (129, 16)))
    filenames = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
                 "model-00003-of-00003.safetensors"]
    groups = {name: [] for name in filenames}
    for n, record in enumerate(records):
        groups[filenames[n % 3]].append((record["name"], record["dtype"], record["shape"]))
    headers, mapping, siblings, total, payloads = {}, {}, [], 0, {}
    for filename, tensors in groups.items():
        raw, size, payload_size = make_header(tensors)
        headers[filename] = raw, size
        mapping.update({name: filename for name, _, _ in tensors})
        siblings.append({"rfilename": filename, "size": size})
        total += payload_size
        payload = bytearray()
        for name, dtype, shape in tensors:
            count = 1
            for dim in shape:
                count *= dim
            if payload_factory is not None:
                raw_tensor = payload_factory(name, dtype, tuple(shape))
                itemsize = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}[dtype]
                if not isinstance(raw_tensor, bytes) or len(raw_tensor) != count * itemsize:
                    raise AssertionError("Payload factory must supply exactly the declared tensor bytes")
                payload.extend(raw_tensor)
                continue
            value = 0.5 if name.endswith("_scale_inv") else 1.0
            unit = {"BF16": b"\x80\x3f", "F16": struct.pack("<e", value),
                    "F32": struct.pack("<f", value), "F8_E4M3": b"\x38"}[dtype]
            payload.extend(unit * count)
        payloads[filename] = bytes(payload)
    expected = {"model_id": MODEL, "revision": REVISION,
                "weights": [{"name": s["rfilename"], "bytes": s["size"]} for s in siblings],
                "architecture": {**{k: v for k, v in config.items() if k != "quantization_config"},
                                 "quantization": deepcopy(config["quantization_config"])}}
    data = {"model": {"id": MODEL, "sha": REVISION, "siblings": siblings}, "config": config,
            "index": {"metadata": {"total_size": total}, "weight_map": mapping},
            "headers": headers, "expected": expected}
    (root / "docs").mkdir(exist_ok=True)
    write_json(root / "docs/model-metadata.json", expected)
    run = root / "reports/metadata/run"
    run.mkdir(parents=True)
    with redirect_stdout(io.StringIO()):
        report, code = execute_metadata(root, settings(), {"max_shards": 512, "budget_mib": 64, "offline": None},
                                        run, source=MemorySource(data))
        publish_report(root, run, report, code)
    if code != 0:
        raise AssertionError(report)
    model_dir = root / settings()["model_directory"]
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_bytes(encode(config))
    (model_dir / "model.safetensors.index.json").write_bytes(encode(data["index"]))
    for name, payload in payloads.items():
        (model_dir / name).write_bytes(headers[name][0] + payload)
    return model_dir, data


class RuntimeWeightsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def weights(self, **kwargs):
        return RuntimeWeights(self.root, settings(), cpu=CpuKernel(), **kwargs)

    def test_complete_reader_validates_all_shards_without_payload_reads(self):
        prepare_runtime_fixture(self.root)
        with FullCatalogueReader(self.root, settings()) as reader:
            stats = reader.stats()
            self.assertTrue(stats["local_all_shard_headers_verified"])
            self.assertEqual(stats["tensor_read_bytes"], 0)
            self.assertEqual(stats["selected_shards"], 3)
            self.assertLessEqual(stats["peak_open_shards"], 2)
            self.assertEqual(MAX_INDEX_BYTES, 1024**2)
            self.assertEqual(MAX_INDEX_TENSORS, 8192)

    def test_vectors_embedding_dense_and_fp8_use_official_names(self):
        prepare_runtime_fixture(self.root)
        with self.weights() as weights:
            self.assertEqual(weights.vector("model.norm.weight"), array("f", [1.0] * 16))
            self.assertEqual(weights.embedding(256), array("f", [1.0] * 16))
            self.assertEqual(weights.linear("lm_head.weight", [1.0] * 16), array("f", [16.0] * 257))
            self.assertEqual(weights.linear("model.layers.0.mlp.gate_proj.weight", [1.0] * 16), array("f", [8.0] * 24))
            stats = weights.stats()
            self.assertLessEqual(stats["peak_open_shards"], 2)
            self.assertLessEqual(stats["max_actual_read_bytes"], 65536)
            self.assertLessEqual(stats["max_decoded_dense_tile_bytes"], 128 * 128 * 4)
            self.assertLessEqual(stats["retained_weight_payload_bytes"], 8 * 1024**2)
            for name in ("full_model_loaded", "inference_verified", "payload_values_verified", "real_checkpoint_compatible"):
                self.assertFalse(stats[name])

    def test_dense_fp16_and_fp32_read_without_a_whole_matrix(self):
        for dtype in ("F16", "F32"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                prepare_runtime_fixture(root, output_dtype=dtype)
                with RuntimeWeights(root, settings(), cpu=CpuKernel()) as weights:
                    result = weights.linear("lm_head.weight", [0.5] * 16)
                    self.assertEqual(result, array("f", [8.0] * 257))

    def test_ragged_fp8_hybrid_routes_rows_and_admits_each_gpu_tile(self):
        prepare_runtime_fixture(self.root, wide=True)
        gate, gpu = Gate(), GpuKernel()
        with self.weights(backend="hybrid", gpu=gpu, gate=gate) as weights:
            result = weights.linear("model.layers.0.mlp.gate_proj.weight", [1.0] * 16)
            self.assertEqual(result, array("f", [8.0] * 129))
            self.assertEqual(weights.stats()["cpu_tiles"], 1)
            self.assertEqual(weights.stats()["gpu_tiles"], 1)
            self.assertEqual(gate.calls, 1)
            # A one-row-block FP8 projection has a deterministic CPU placement.
            weights.linear("model.layers.1.mlp.shared_experts.gate_proj.weight", [1.0] * 16)
            self.assertEqual(gate.calls, 1)

    def test_owned_gpu_context_has_exact_projection_budget_and_closes_each_call(self):
        prepare_runtime_fixture(self.root, wide=True)
        contexts = []
        class ScopedGpu(GpuKernel):
            def __init__(self, device_index, *, max_operations):
                self.operations, self.max_operations, self.closed = 0, max_operations, False
                contexts.append(self)
            def __enter__(self):
                return self
            def __exit__(self, *_):
                self.closed = True
        gate = Gate()
        with patch("glm_local.cuda_probe.CudaTileBackend", ScopedGpu), self.weights(backend="hybrid", gate=gate) as weights:
            for _ in range(2):
                weights.linear("model.layers.0.mlp.gate_proj.weight", [1.0] * 16)
            self.assertEqual(weights.stats()["scoped_cuda_contexts"], 2)
            self.assertEqual(weights.stats()["maximum_context_launch_budget"], 1)
        self.assertEqual(len(contexts), 2)
        self.assertTrue(all(context.closed and context.operations == context.max_operations == 1 for context in contexts))

    def test_architecture_review_is_not_sufficient_for_decoder_access(self):
        prepare_runtime_fixture(self.root)
        path = self.root / "reports/metadata-latest.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["status"] = "REVIEW_REQUIRED"
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, "eligible architecture"):
            self.weights()

    def test_missing_shard_rejected_before_any_weight_access(self):
        directory, data = prepare_runtime_fixture(self.root)
        (directory / next(iter(data["headers"]))).unlink()
        with self.assertRaises((SafeTensorError, OSError)):
            self.weights()

    def test_local_config_or_index_must_match_captured_exact_document(self):
        for filename in ("config.json", "model.safetensors.index.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                directory, _ = prepare_runtime_fixture(root)
                path = directory / filename
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(SafeTensorError, "length differs"):
                    RuntimeWeights(root, settings(), cpu=CpuKernel())

    def test_local_header_same_size_tampering_is_rejected(self):
        directory, data = prepare_runtime_fixture(self.root)
        path = directory / next(iter(data["headers"]))
        content = path.read_bytes()
        self.assertIn(b'"format":"pt"', content)
        path.write_bytes(content.replace(b'"format":"pt"', b'"format":"xx"', 1))
        with self.assertRaisesRegex(SafeTensorError, "SHA-256"):
            self.weights()

    def test_local_metadata_change_after_open_fails_closed(self):
        directory, _ = prepare_runtime_fixture(self.root)
        with self.weights() as weights:
            with (directory / "config.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(SafeTensorError, "changed since validation"):
                weights.embedding(0)

    def test_nonfinite_bf16_payload_is_rejected_when_read(self):
        directory, data = prepare_runtime_fixture(self.root)
        name = "model.norm.weight"
        shard = data["index"]["weight_map"][name]
        raw, _ = data["headers"][shard]
        tensor = json.loads(raw[8:])[name]
        with (directory / shard).open("r+b") as stream:
            stream.seek(len(raw) + tensor["data_offsets"][0])
            stream.write(b"\xc0\x7f")
        with self.weights() as weights:
            with self.assertRaisesRegex(SafeTensorError, "nonfinite"):
                weights.vector(name)

    def test_budget_failure_precedes_payload_and_releases_leases(self):
        prepare_runtime_fixture(self.root)
        ledger = ReservationLedger(100)
        with self.weights(ledger=ledger) as weights:
            before = weights.stats()["tensor_read_bytes"]
            with self.assertRaises(ValueError):
                weights.linear("lm_head.weight", [1.0] * 16)
            self.assertEqual(weights.stats()["tensor_read_bytes"], before)
            self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_invalid_tokens_lengths_and_closed_adapter_are_rejected(self):
        prepare_runtime_fixture(self.root)
        with self.weights() as weights:
            for token in (-1, 257, True):
                with self.assertRaises(ValueError):
                    weights.embedding(token)
            with self.assertRaises(ValueError):
                weights.linear("lm_head.weight", [1.0])
        with self.assertRaises(SafeTensorError):
            weights.vector("model.norm.weight")


if __name__ == "__main__":
    unittest.main()
