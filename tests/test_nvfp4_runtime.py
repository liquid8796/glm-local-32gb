"""Complete tiny NVFP4 shards through the streamed GLM backbone.

The independent oracle decodes bytes into float matrices and uses the pinned
unquantized Transformers graph. This tests decoded-weight FP32 fallback only;
it is not W4A4 activation quantization or full-checkpoint inference evidence.
"""
from array import array
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import math
import json
import os
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

from glm_local.architecture.mapper import NVFP4_MODEL_ID, NVFP4_REVISION
from glm_local.residency import PlannerSettings, build_plan
from glm_local.runtime_weights import RuntimeWeights
from glm_local.streaming_decoder import StreamingDecoder
from test_nvfp4_schema import nvfp4_config, prepare_nvfp4_fixture
from test_streaming_decoder import decoder_config, official_state


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def decode_e2m1(code):
    value = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[code & 7]
    return -value if code & 8 else value


def decode_scale(code):
    magnitude = code & 127
    if code & 128 or magnitude == 127:
        raise ValueError("Invalid independent scale")
    exponent, mantissa = magnitude >> 3, magnitude & 7
    return math.ldexp(mantissa, -9) if not exponent else math.ldexp(1 + mantissa / 8, exponent - 7)


class NvCpuKernel:
    """Portable test kernel, independently defining E2M1 and scale arithmetic."""
    def matvec_nvfp4_tile(self, packed, rows, cols, scales, vector, global_scale):
        output = []
        for row in range(rows):
            total = 0.0
            for col in range(cols):
                nibble = packed[row * (cols // 2) + col // 2] >> (4 * (col % 2)) & 15
                combined = f32(decode_scale(scales[row * (cols // 16) + col // 16]) * global_scale)
                weight = f32(decode_e2m1(nibble) * combined)
                total = f32(total + f32(weight * vector[col]))
            output.append(total)
        return output


class NvGpuKernel(NvCpuKernel):
    def __init__(self, device_index=0, *, max_operations=65536):
        self.operations, self.max_operations, self.closed = 0, max_operations, False

    def matvec_nvfp4_tile(self, *args):
        if self.operations >= self.max_operations:
            raise RuntimeError("Test GPU exceeded its launch budget")
        self.operations += 1
        return super().matvec_nvfp4_tile(*args)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True


class Gate:
    def __init__(self):
        self.calls = 0

    def before_submit(self):
        self.calls += 1


class NvPayloads:
    """Retain tiny test bytes and independently decode them for graph oracles."""
    def __init__(self):
        self.raw = {}
        self.matrices, self.vectors = {}, {}

    def __call__(self, name, dtype, shape):
        generator = random.Random(sum((index + 1) * ord(char) for index, char in enumerate(name)))
        raw = bytearray()
        for index in range(math.prod(shape)):
            if dtype == "U8":
                raw.append(generator.choice((1, 2, 3, 9, 10, 11)) | generator.choice((1, 2, 3, 9, 10, 11)) << 4)
            elif dtype == "F8_E4M3":
                raw.append(generator.choice((0x30, 0x38, 0x40)))
            else:
                if name.endswith(".weight_scale_2"):
                    value = 0.0625
                elif name.endswith(".input_scale"):
                    value = 3.25
                elif name.endswith("e_score_correction_bias"):
                    value = 0.125 * index
                elif len(shape) == 1:
                    value = generator.uniform(-0.02, 0.02) + (0 if name.endswith("bias") else 1)
                else:
                    value = generator.uniform(-0.07, 0.07)
                if dtype == "BF16":
                    raw.extend(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16))
                else:
                    raw.extend(struct.pack("<f", value))
        self.raw[name] = dtype, shape, bytes(raw)
        return bytes(raw)

    def finish(self):
        for name, (dtype, shape, raw) in self.raw.items():
            if dtype == "U8":
                rows, packed_cols = shape
                cols, stem = 2 * packed_cols, name[:-len(".weight")]
                scale_bytes = self.raw[stem + ".weight_scale"][2]
                global_scale = struct.unpack("<f", self.raw[stem + ".weight_scale_2"][2])[0]
                values = []
                for row in range(rows):
                    for col in range(cols):
                        code = raw[row * packed_cols + col // 2] >> (4 * (col % 2)) & 15
                        # Calibration input_scale intentionally does not enter this equation.
                        combined = f32(decode_scale(scale_bytes[row * (cols // 16) + col // 16]) * global_scale)
                        values.append(f32(decode_e2m1(code) * combined))
                self.matrices[name] = [values[start:start + cols] for start in range(0, len(values), cols)]
            elif len(shape) and not name.endswith(".weight_scale"):
                values = ([struct.unpack("<f", struct.pack("<I", value[0] << 16))[0] for value in struct.iter_unpack("<H", raw)]
                          if dtype == "BF16" else [value[0] for value in struct.iter_unpack("<f", raw)])
                if len(shape) == 1:
                    self.vectors[name] = values
                elif len(shape) == 2:
                    self.matrices[name] = [values[start:start + shape[1]] for start in range(0, len(values), shape[1])]


class Nvfp4RuntimeTests(unittest.TestCase):
    def fixture(self, *, hybrid=False, mtp=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = decoder_config()
        config.update(quantization_config=deepcopy(nvfp4_config()["quantization_config"]), moe_intermediate_size=144)
        identity = {}
        if mtp:
            config.update(num_nextn_predict_layers=1, architectures=["GlmMoeDsaForCausalLM"], transformers_version="5.15.0")
            config["quantization_config"]["ignore"].append("model.layers.2*")
            identity = {"model_id": NVFP4_MODEL_ID, "revision": NVFP4_REVISION}
        payloads = NvPayloads()
        directory, data = prepare_nvfp4_fixture(root, config_overrides=config, payload_factory=payloads, **identity)
        payloads.finish()
        options = PlannerSettings(context_tokens=16, max_new_tokens=4, runtime_headroom_bytes=1024**2,
                                  device="hybrid" if hybrid else "cpu", vram_budget_bytes=1024**3 if hybrid else 0)
        initial = build_plan(data["config"], options, **identity)
        plan = build_plan(data["config"], replace(options, ram_budget_bytes=initial.ram_required_bytes), **identity)
        return root, directory, data, payloads, plan, identity

    def test_logical_input_shape_and_plan_are_used_for_packed_weight(self):
        root, _, data, payloads, plan, _ = self.fixture()
        ledger = plan.allocator(include_cache=False)
        with RuntimeWeights(root, data["settings"], cpu=NvCpuKernel(), ledger=ledger, plan=plan) as weights:
            name = "model.layers.1.mlp.experts.0.gate_proj.weight"
            self.assertEqual(weights._reader.tensors[name].shape, (144, 8))
            before = weights.stats()["tensor_read_bytes"]
            with self.assertRaises(ValueError):
                weights.linear(name, [0.125] * 8)
            self.assertEqual(weights.stats()["tensor_read_bytes"], before)
            actual = weights.linear(name, [0.125] * 16)
            expected = [sum(value * 0.125 for value in row) for row in payloads.matrices[name]]
            self.assertEqual(len(actual), 144)
            for left, right in zip(actual, expected):
                self.assertAlmostEqual(left, right, delta=1e-6)
        self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_full_tiny_generation_preserves_bf16_and_exact_shared_plan_budget(self):
        root, _, data, _, plan, _ = self.fixture()
        ledger = plan.allocator(include_cache=False)
        with RuntimeWeights(root, data["settings"], cpu=NvCpuKernel(), ledger=ledger, plan=plan) as weights:
            for name in ("lm_head.weight", "model.embed_tokens.weight", "model.layers.0.mlp.gate_proj.weight",
                         "model.layers.1.self_attn.q_a_proj.weight", "model.layers.1.mlp.shared_experts.gate_proj.weight"):
                self.assertEqual(weights._reader.tensors[name].dtype, "BF16")
            with StreamingDecoder(data["config"], weights, plan, ledger=ledger) as decoder:
                generated = decoder.generate([1, 4, 2], 3, eos_token_ids=[])
                self.assertEqual(len(generated), 3)
                self.assertEqual(decoder.cache_bytes, plan.cache_bytes)
                self.assertEqual(decoder.position, 5)
                stats = weights.stats()
                self.assertGreater(stats["cpu_tiles"], 0)
                self.assertLessEqual(stats["peak_open_shards"], 2)
                self.assertLessEqual(stats["max_actual_read_bytes"], 65536)
                self.assertEqual(stats["activation_quantization"], "none")
                self.assertFalse(stats["native_w4a4_parity_verified"])
                for flag in ("full_model_loaded", "full_model_limits_verified", "inference_verified", "real_checkpoint_compatible"):
                    self.assertIs(stats[flag], False)
            self.assertEqual(ledger.snapshot()["active_leases"], weights.stats()["row_band_cache"]["cached_bands"])
            self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"],
                             plan.settings.runtime_headroom_bytes + weights.stats()["retained_weight_payload_bytes"])
        self.assertEqual(ledger.snapshot()["active_leases"], 0)
        self.assertLessEqual(ledger.snapshot()["cpu"]["peak_bytes"], plan.ram_required_bytes)

    def test_hybrid_uses_scoped_nvfp4_contexts_and_gate_with_bounded_operations(self):
        root, _, data, _, plan, _ = self.fixture(hybrid=True)
        ledger, contexts, gate = plan.allocator(include_cache=False), [], Gate()
        class ScopedGpu(NvGpuKernel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                contexts.append(self)
        with patch("glm_local.nvfp4_kernels.CudaNVFP4TileBackend", ScopedGpu):
            with RuntimeWeights(root, data["settings"], backend="hybrid", cpu=NvCpuKernel(), gate=gate,
                                ledger=ledger, plan=plan) as weights:
                with StreamingDecoder(data["config"], weights, plan, ledger=ledger) as decoder:
                    decoder.generate([1], 2, eos_token_ids=[])
                stats = weights.stats()
                self.assertEqual(stats["scoped_cuda_contexts"], len(contexts))
                self.assertEqual(stats["gpu_tiles"], gate.calls)
                self.assertGreater(gate.calls, 0)
        self.assertTrue(all(context.closed and context.operations == context.max_operations for context in contexts))
        self.assertEqual(ledger.snapshot()["active_leases"], 0)
        self.assertLessEqual(ledger.snapshot()["cpu"]["peak_bytes"], plan.ram_required_bytes)

    def test_declared_bf16_mtp_headers_are_checked_but_never_executed(self):
        root, _, data, _, plan, identity = self.fixture(mtp=True)
        ledger = plan.allocator(include_cache=False)
        with RuntimeWeights(root, data["settings"], cpu=NvCpuKernel(), ledger=ledger, plan=plan) as weights:
            self.assertEqual(weights._reader.tensors["model.layers.2.mlp.experts.0.gate_proj.weight"].dtype, "BF16")
            with patch.object(weights, "linear", wraps=weights.linear) as linear, patch.object(weights, "vector", wraps=weights.vector) as vector:
                with StreamingDecoder(data["config"], weights, plan, ledger=ledger, **identity) as decoder:
                    decoder.generate([1], 2, eos_token_ids=[])
                names = [call.args[0] for call in linear.call_args_list + vector.call_args_list]
                self.assertFalse(any(name.startswith("model.layers.2.") for name in names))
            self.assertEqual(plan.mtp_layers_excluded, 1)

    def test_input_calibration_scalar_is_validated_but_not_folded_into_weights(self):
        root, directory, data, _, plan, _ = self.fixture()
        name = "model.layers.1.mlp.experts.0.gate_proj"
        with RuntimeWeights(root, data["settings"], cpu=NvCpuKernel(), plan=plan) as weights:
            original = weights.linear(name + ".weight", [0.125] * 16)
        scalar = name + ".input_scale"
        shard = data["index"]["weight_map"][scalar]
        raw, _ = data["headers"][shard]
        offset = json.loads(raw[8:])[scalar]["data_offsets"][0]
        with (directory / shard).open("r+b") as stream:
            stream.seek(len(raw) + offset)
            stream.write(struct.pack("<f", 19.0))
        with RuntimeWeights(root, data["settings"], cpu=NvCpuKernel(), plan=plan) as weights:
            self.assertEqual(weights.linear(name + ".weight", [0.125] * 16), original)
        with (directory / shard).open("r+b") as stream:
            stream.seek(len(raw) + offset)
            stream.write(struct.pack("<f", float("nan")))
        with RuntimeWeights(root, data["settings"], cpu=NvCpuKernel(), plan=plan) as weights:
            with self.assertRaises(ValueError):
                weights.linear(name + ".weight", [0.125] * 16)

    @unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1" and os.environ.get("GLM_TEST_OFFICIAL") == "1",
                         "Requires native NVFP4 kernels and pinned Transformers reference")
    def test_native_cpu_and_optional_cuda_match_independently_dequantized_official_graph(self):
        import torch
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from glm_local.nvfp4_kernels import NativeNVFP4CpuBackend, CudaNVFP4TileBackend
        from glm_local.gpu_gate import GpuBoundaryGate
        torch.set_num_threads(1)
        measurements = []
        for backend in (["cpu", "hybrid"] if os.environ.get("GLM_TEST_CUDA") == "1" else ["cpu"]):
            root, _, data, payloads, plan, _ = self.fixture(hybrid=backend == "hybrid")
            official_config = deepcopy(data["config"])
            del official_config["quantization_config"]
            official_config = GlmMoeDsaConfig(**official_config)
            official_config._attn_implementation = "eager"
            model = GlmMoeDsaForCausalLM(official_config).float().eval()
            model.load_state_dict(official_state(model, payloads.matrices, payloads.vectors, torch), strict=True)
            ledger = plan.allocator(include_cache=False)
            with ExitStack() as stack:
                cpu = stack.enter_context(NativeNVFP4CpuBackend())
                gpu = gate = None
                if backend == "hybrid":
                    gpu = stack.enter_context(CudaNVFP4TileBackend(0, max_operations=256))
                    gate = GpuBoundaryGate(device_index=0, target=0.6)
                weights = stack.enter_context(RuntimeWeights(root, data["settings"], backend=backend, cpu=cpu,
                    gpu=gpu, gate=gate, ledger=ledger, plan=plan))
                def official_topk(scores, count):
                    return torch.topk(torch.tensor(scores, dtype=torch.float32), count).indices.tolist()
                decoder = stack.enter_context(StreamingDecoder(data["config"], weights, plan, ledger=ledger,
                                                                attention_topk=official_topk))
                tokens = [1, 4, 2, 9, 3, 7]
                max_error = 0.0
                for position, token in enumerate(tokens):
                    actual = decoder.step(token)
                    with torch.inference_mode():
                        expected = model(input_ids=torch.tensor([tokens[:position + 1]]), use_cache=False).logits[0, -1]
                    with self.subTest(backend=backend, position=position):
                        torch.testing.assert_close(torch.tensor(actual), expected, rtol=0, atol=5e-6)
                    max_error = max(max_error, float(torch.max(torch.abs(torch.tensor(actual) - expected))))
                self.assertGreater(weights.stats()["cpu_tiles"], 0)
                if gate:
                    self.assertGreater(weights.stats()["gpu_tiles"], 0)
                    gate.finish()
                stats = weights.stats()
                measurements.append({"backend": backend, "tokens_compared": len(tokens), "maximum_absolute_logit_error": max_error,
                                     "absolute_tolerance": 5e-6, "cpu_tiles": stats["cpu_tiles"], "gpu_tiles": stats["gpu_tiles"],
                                     "peak_open_shards": stats["peak_open_shards"], "max_actual_read_bytes": stats["max_actual_read_bytes"],
                                     "ledger_cpu_peak_bytes": ledger.snapshot()["cpu"]["peak_bytes"],
                                     "planned_ram_bytes": plan.ram_required_bytes})
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
        if os.environ.get("GLM_NVFP4_PARITY_REPORT"):
            destination = Path(os.environ["GLM_NVFP4_PARITY_REPORT"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"status": "PASS", "fixture": "small_synthetic_nvfp4_2_layer_4_expert_moe144",
                "reference": "Independent byte dequantization into pinned Transformers FP32 eager graph",
                "activation_quantization": "none", "native_w4a4_parity_verified": False,
                "real_checkpoint_compatible": False, "full_model_limits_verified": False,
                "measurements": measurements}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
