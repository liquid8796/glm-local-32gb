"""Dynamic tiny backbone checks; official parity requires the pinned extras."""
from array import array
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
import math
import os
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

from glm_local.checkpoint_schema import known_shape
from glm_local.residency import PlannerSettings, ReservationLedger, build_plan
from glm_local.streaming_decoder import DecoderError, StreamingDecoder


def decoder_config(*, layers=2, hidden=16, heads=2, experts=4):
    return {
        "model_type": "glm_moe_dsa", "num_hidden_layers": layers, "hidden_size": hidden,
        "vocab_size": 32, "num_attention_heads": heads, "num_key_value_heads": heads,
        "q_lora_rank": 8, "kv_lora_rank": 4, "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4, "v_head_dim": 4, "index_n_heads": 2, "index_head_dim": 8,
        "intermediate_size": 24, "moe_intermediate_size": 24,
        "n_routed_experts": experts, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "max_position_embeddings": 32, "index_topk": 4,
        "layer_types": ["deepseek_sparse_attention"] * layers,
        "mlp_layer_types": ["dense"] + ["sparse"] * (layers - 1),
        "indexer_types": ["full"] + ["shared"] * (layers - 1),
        "attention_bias": False, "mlp_bias": False, "tie_word_embeddings": False,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
        "hidden_act": "silu", "norm_topk_prob": True, "use_cache": True,
        "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.5,
        "rms_norm_eps": 1e-5, "attention_dropout": 0.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    }


class ProceduralWeights:
    """Small, repeatable values; the decoder never supplies these formulas."""
    def __init__(self, config):
        self.config = deepcopy(config)
        self.calls = []
        self.fail_name = None

    def vector(self, name):
        self.calls.append(name)
        shape = known_shape(name, self.config)
        if not shape or len(shape) != 1:
            raise ValueError(name)
        if "e_score_correction_bias" in name:
            return array("f", (0.013 * i for i in range(shape[0])))
        if name.endswith("bias"):
            return array("f", (0.01 * math.sin(i + 1) for i in range(shape[0])))
        return array("f", (1.0 + 0.02 * math.sin(i + 1) for i in range(shape[0])))

    def matrix(self, name):
        rows, columns = known_shape(name, self.config)
        phase = sum((index + 1) * ord(char) for index, char in enumerate(name))
        return [array("f", (0.07 * math.sin((row + 1) * (col + 1) * 0.31 + phase * 0.01)
                             for col in range(columns))) for row in range(rows)]

    def embedding(self, token):
        return self.matrix("model.embed_tokens.weight")[token]

    def linear(self, name, values):
        self.calls.append(name)
        if name == self.fail_name:
            raise RuntimeError("injected projection failure")
        return array("f", (math.fsum(weight * value for weight, value in zip(row, values))
                            for row in self.matrix(name)))


class StoredPayloads:
    """Encode actual tiny BF16/FP8 payloads and independently retain oracle values."""
    def __init__(self):
        self.matrices, self.vectors = {}, {}

    def __call__(self, name, dtype, shape):
        generator = random.Random(sum((i + 1) * ord(char) for i, char in enumerate(name)))
        raw, values = bytearray(), []
        for index in range(math.prod(shape)):
            if dtype == "F8_E4M3":
                code = generator.choice((0x18, 0x20, 0x28, 0x30, 0x38, 0x98, 0xA0, 0xA8, 0xB0, 0xB8))
                raw.append(code)
                exponent, mantissa = (code & 127) >> 3, code & 7
                value = math.ldexp(1 + mantissa / 8, exponent - 7) * 0.0625
                values.append(-value if code & 128 else value)
                continue
            if name.endswith("_scale_inv"):
                value = 0.0625
            elif "e_score_correction_bias" in name:
                value = 0.05 * index
            elif len(shape) == 1:
                value = generator.uniform(-0.02, 0.02) + (0 if name.endswith("bias") else 1)
            else:
                value = generator.uniform(-0.07, 0.07)
            if dtype == "BF16":
                bits = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
                raw.extend(struct.pack("<H", bits))
                value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            else:
                code = "<e" if dtype == "F16" else "<f"
                encoded = struct.pack(code, value)
                raw.extend(encoded)
                value = struct.unpack(code, encoded)[0]
            values.append(value)
        if not name.endswith("_scale_inv"):
            if len(shape) == 1:
                self.vectors[name] = values
            else:
                self.matrices[name] = [values[start:start + shape[1]]
                                       for start in range(0, len(values), shape[1])]
        return bytes(raw)


def official_state(model, matrices, vectors, torch):
    state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name[:-len("gate_up_proj")]
            state[name] = torch.tensor([matrices[prefix + f"{e}.gate_proj.weight"]
                                       + matrices[prefix + f"{e}.up_proj.weight"]
                                       for e in range(tensor.shape[0])])
        elif name.endswith(".experts.down_proj"):
            prefix = name[:-len("down_proj")]
            state[name] = torch.tensor([matrices[prefix + f"{e}.down_proj.weight"]
                                       for e in range(tensor.shape[0])])
        else:
            state[name] = torch.tensor(vectors[name] if tensor.ndim == 1 else matrices[name])
    return state


def fixture(config=None, **options):
    config = decoder_config() if config is None else config
    plan = build_plan(config, PlannerSettings(context_tokens=32, max_new_tokens=4,
                                               runtime_headroom_bytes=1024 * 1024))
    weights = ProceduralWeights(config)
    return StreamingDecoder(config, weights, plan, **options), weights, plan


class StreamingDecoderTests(unittest.TestCase):
    def test_actual_cache_arrays_equal_plan_and_shared_indexer_is_not_loaded(self):
        decoder, weights, plan = fixture()
        self.addCleanup(decoder.close)
        self.assertEqual(decoder.cache_bytes, plan.cache_bytes)
        self.assertEqual(len(decoder._cache), 2)
        self.assertEqual(set(decoder._index_cache), {0})
        self.assertTrue(all(cache.typecode == "f" for cache in decoder._cache))
        for position, token in enumerate([1, 4, 2, 9, 3, 7, 0]):
            result = decoder.step(token)
            self.assertEqual(len(result), 32)
            self.assertEqual(result.typecode, "f")
            self.assertTrue(all(math.isfinite(value) for value in result))
            self.assertEqual(decoder.position, position + 1)
            selected = decoder.last_trace["last_selected_indices"]
            self.assertEqual(len(selected), min(4, position + 1))
            self.assertTrue(all(index <= position for index in selected))
        self.assertFalse(any("layers.1.self_attn.indexer" in name for name in weights.calls))
        self.assertEqual(decoder.cache_used_bytes, plan.cache_bytes // 32 * 7)

    def test_distinct_dynamic_dimensions_and_nonprefix_full_indexer_schedule(self):
        config = decoder_config(layers=3, hidden=12, heads=3, experts=8)
        config.update(indexer_types=["full", "shared", "full"], n_group=2, topk_group=1,
                      n_shared_experts=2, moe_intermediate_size=12)
        decoder, _, plan = fixture(config)
        with decoder:
            self.assertEqual(plan.indexer_owners, (0, 0, 2))
            self.assertEqual(set(decoder._index_cache), {0, 2})
            self.assertEqual(len(decoder.step(5)), 32)
            self.assertEqual(len(decoder.step(7)), 32)

    def test_reset_replays_identically_and_preserves_prior_prefix(self):
        decoder, _, _ = fixture()
        with decoder:
            first = decoder.step(2)
            prefix = [cache[:8] for cache in decoder._cache]
            for token in [5, 8, 11]:
                decoder.step(token)
            self.assertEqual([cache[:8] for cache in decoder._cache], prefix)
            decoder.reset()
            self.assertEqual(decoder.position, 0)
            self.assertEqual(decoder.cache_used_bytes, 0)
            self.assertEqual(decoder.step(2), first)

    def test_generate_context_and_eos_are_validated_before_any_projection(self):
        decoder, weights, _ = fixture()
        with decoder:
            for prompt, count in (([], 1), ([32], 1), ([True], 1), ([1] * 30, 4), ([1], 5)):
                with self.assertRaises(DecoderError):
                    decoder.generate(prompt, count)
                self.assertEqual(weights.calls, [])
            with self.assertRaises(DecoderError):
                decoder.generate([1], 2, [False])
            self.assertEqual(weights.calls, [])
            generated = decoder.generate([1, 2], 4, eos_token_ids=list(range(32)))
            self.assertEqual(len(generated), 1)
            self.assertEqual(decoder.position, 2)
            with self.assertRaises(DecoderError):
                decoder.generate([1], 1)

    def test_generation_uses_previous_logits_and_zero_generation_prefills(self):
        decoder, _, _ = fixture()
        with decoder:
            generated = decoder.generate([1, 2], 4, eos_token_ids=[])
            self.assertEqual(len(generated), 4)
            self.assertEqual(decoder.position, 5)
            decoder.reset()
            self.assertEqual(decoder.generate([1, 2], 0), [])
            self.assertEqual(decoder.position, 2)

    def test_failed_step_releases_scratch_and_retry_rewrites_uncommitted_slots(self):
        decoder, weights, _ = fixture()
        fresh, _, _ = fixture()
        with decoder, fresh:
            decoder.step(1)
            fresh.step(1)
            before = decoder.ledger.snapshot()
            weights.fail_name = "model.layers.1.mlp.experts.3.up_proj.weight"
            with self.assertRaisesRegex(RuntimeError, "injected projection failure"):
                decoder.step(2)
            self.assertEqual(decoder.position, 1)
            self.assertEqual(decoder.ledger.snapshot()["cpu"]["used_bytes"], before["cpu"]["used_bytes"])
            self.assertEqual(decoder.ledger.snapshot()["active_leases"], 1)
            weights.fail_name = None
            self.assertEqual(decoder.step(2), fresh.step(2))

    def test_bad_projection_and_selection_outputs_fail_without_position_commit(self):
        decoder, weights, _ = fixture()
        with decoder:
            for output in ([0.0], [float("nan")] * 8, [1e99] * 8, iter([0.0] * 8)):
                with patch.object(weights, "linear", return_value=output):
                    # Bound methods were captured at construction; inject a
                    # deliberately bad adapter at the declared callback boundary.
                    original, decoder._linear = decoder._linear, weights.linear
                    try:
                        with self.assertRaises(DecoderError):
                            decoder.step(1)
                    finally:
                        decoder._linear = original
                self.assertEqual(decoder.position, 0)
            for selected in ([1], [0, 0], [True], None):
                decoder._attention_topk = lambda scores, count, result=selected: result
                with self.assertRaises(DecoderError):
                    decoder.step(1)
                self.assertEqual(decoder.position, 0)

    def test_invalid_semantics_fail_before_weights_or_cache_allocation(self):
        for field, value in (("hidden_act", "gelu"), ("n_group", 3), ("topk_group", 2),
                             ("num_key_value_heads", 1), ("rms_norm_eps", 0),
                             ("norm_topk_prob", 1), ("routed_scaling_factor", float("nan")),
                             ("use_cache", False), ("attention_dropout", 0.1),
                             ("rope_parameters", {"rope_type": "yarn", "rope_theta": 10000}),
                             ("rope_interleave", False), ("scoring_func", "softmax")):
            config = decoder_config()
            config[field] = value
            plan = build_plan(config, PlannerSettings(context_tokens=32, max_new_tokens=4))
            weights = ProceduralWeights(config)
            with self.subTest(field=field):
                with self.assertRaises(DecoderError):
                    StreamingDecoder(config, weights, plan)
                self.assertEqual(weights.calls, [])

    def test_plan_mismatch_is_rejected_before_allocating_cache(self):
        config = decoder_config()
        plan = build_plan(config, PlannerSettings(context_tokens=32, max_new_tokens=4))
        config["hidden_size"] = 24
        with self.assertRaises(DecoderError):
            StreamingDecoder(config, ProceduralWeights(config), plan)

    def test_weights_numeric_config_cannot_drift_even_with_identical_shapes(self):
        config = decoder_config()
        plan = build_plan(config, PlannerSettings(context_tokens=32, max_new_tokens=4))
        weights = ProceduralWeights(config)
        weights.config["rope_parameters"]["rope_theta"] = 500000
        with self.assertRaisesRegex(DecoderError, "Weight source configuration differs"):
            StreamingDecoder(config, weights, plan)
        self.assertEqual(weights.calls, [])

    def test_real_cache_allocation_failure_releases_lifetime_reservation(self):
        config = decoder_config()
        plan = build_plan(config, PlannerSettings(context_tokens=32, max_new_tokens=4))
        ledger = plan.allocator(include_cache=False)
        before = ledger.snapshot()
        with patch("glm_local.streaming_decoder.array", side_effect=MemoryError("test allocation")):
            with self.assertRaises(MemoryError):
                StreamingDecoder(config, ProceduralWeights(config), plan, ledger=ledger)
        self.assertEqual(ledger.snapshot()["active_leases"], 0)
        self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"], before["cpu"]["used_bytes"])

    def test_close_releases_cache_and_refuses_use_or_insufficient_headroom(self):
        decoder, _, _ = fixture()
        decoder.step(1)
        decoder.close()
        self.assertEqual(decoder.cache_bytes, 0)
        self.assertEqual(decoder.ledger.snapshot()["active_leases"], 0)
        decoder.close()
        for operation in (lambda: decoder.step(1), decoder.reset,
                          lambda: decoder.generate([1], 1), decoder.__enter__):
            with self.assertRaises(DecoderError):
                operation()
        config = decoder_config()
        plan = build_plan(config, PlannerSettings(context_tokens=32, max_new_tokens=4))
        with self.assertRaises(DecoderError):
            StreamingDecoder(config, ProceduralWeights(config), plan,
                             ledger=ReservationLedger(plan.settings.ram_budget_bytes))

    def test_context_full_does_not_read_weights(self):
        config = decoder_config()
        plan = build_plan(config, PlannerSettings(context_tokens=1, max_new_tokens=0))
        weights = ProceduralWeights(config)
        with StreamingDecoder(config, weights, plan) as decoder:
            decoder.step(1)
            before = list(weights.calls)
            with self.assertRaises(DecoderError):
                decoder.step(2)
            self.assertEqual(weights.calls, before)


@unittest.skipUnless(os.environ.get("GLM_TEST_OFFICIAL") == "1", "Requires opt-in pinned official reference")
class StreamingOfficialParityTests(unittest.TestCase):
    def test_dynamic_graph_logits_match_official_full_prefix(self):
        import torch
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

        torch.set_num_threads(1)
        for layers, hidden, heads, experts in ((2, 16, 2, 4), (3, 12, 3, 8)):
            config = decoder_config(layers=layers, hidden=hidden, heads=heads, experts=experts)
            if layers == 3:
                config.update(indexer_types=["full", "shared", "full"], n_group=2,
                              topk_group=1, n_shared_experts=2, moe_intermediate_size=12)
            weights = ProceduralWeights(config)
            official_config = deepcopy(config)
            del official_config["quantization_config"]
            official_config = GlmMoeDsaConfig(**official_config)
            official_config._attn_implementation = "eager"
            model = GlmMoeDsaForCausalLM(official_config).float().eval()
            state = {}
            for name, tensor in model.state_dict().items():
                if name.endswith(".experts.gate_up_proj"):
                    prefix = name[:-len("gate_up_proj")]
                    state[name] = torch.tensor([weights.matrix(prefix + f"{e}.gate_proj.weight")
                                                + weights.matrix(prefix + f"{e}.up_proj.weight")
                                                for e in range(experts)])
                elif name.endswith(".experts.down_proj"):
                    prefix = name[:-len("down_proj")]
                    state[name] = torch.tensor([weights.matrix(prefix + f"{e}.down_proj.weight")
                                                for e in range(experts)])
                elif tensor.ndim == 1:
                    state[name] = torch.tensor(weights.vector(name))
                else:
                    state[name] = torch.tensor(weights.matrix(name))
            model.load_state_dict(state, strict=True)
            plan = build_plan(config, PlannerSettings(context_tokens=32, max_new_tokens=4))
            def official_topk(scores, count):
                return torch.topk(torch.tensor(scores, dtype=torch.float32), count).indices.tolist()
            with StreamingDecoder(config, weights, plan, attention_topk=official_topk) as decoder:
                tokens = [1, 4, 2, 9, 3, 7, 0, 11]
                for position, token in enumerate(tokens):
                    actual = decoder.step(token)
                    with torch.inference_mode():
                        expected = model(input_ids=torch.tensor([tokens[:position + 1]]),
                                         use_cache=False).logits[0, -1]
                    with self.subTest(layers=layers, position=position):
                        torch.testing.assert_close(torch.tensor(actual), expected, rtol=0, atol=3e-6)


class StreamingLocalShardTests(unittest.TestCase):
    def _setup_local(self, *, hybrid=False):
        from test_runtime_weights import prepare_runtime_fixture
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        payloads = StoredPayloads()
        _, data = prepare_runtime_fixture(root, wide=True, config_overrides=decoder_config(),
                                           payload_factory=payloads)
        config = data["config"]
        options = PlannerSettings(context_tokens=32, max_new_tokens=4,
                                  runtime_headroom_bytes=1024**2,
                                  device="hybrid" if hybrid else "cpu",
                                  vram_budget_bytes=1024**3 if hybrid else 0)
        initial = build_plan(config, options)
        # An exact budget is intentional: tests must catch a plan that omits
        # real fixed-capacity adapter leases, even when default RAM is roomy.
        plan = build_plan(config, replace(options, ram_budget_bytes=initial.ram_required_bytes))
        return root, config, payloads, plan

    def test_complete_local_shards_generate_with_shared_exact_budget_ledger(self):
        from checkpoint_test_helpers import settings
        from test_runtime_weights import CpuKernel
        from glm_local.runtime_weights import RuntimeWeights
        root, config, _, plan = self._setup_local()
        ledger = plan.allocator(include_cache=False)
        with RuntimeWeights(root, settings(), cpu=CpuKernel(), ledger=ledger, plan=plan) as weights:
            with StreamingDecoder(config, weights, plan, ledger=ledger) as decoder:
                generated = decoder.generate([1, 4, 2], 3, eos_token_ids=[])
                self.assertEqual(len(generated), 3)
                self.assertEqual(decoder.position, 5)
                self.assertEqual(decoder.cache_bytes, plan.cache_bytes)
                stats = weights.stats()
                self.assertTrue(stats["local_all_shard_headers_verified"])
                self.assertGreater(stats["tensor_read_bytes"], 0)
                self.assertGreater(stats["cpu_tiles"], 0)
                self.assertLessEqual(stats["peak_open_shards"], 2)
                self.assertLessEqual(stats["max_actual_read_bytes"], 65536)
                self.assertEqual(stats["retained_weight_payload_bytes"], 0)
                self.assertFalse(stats["full_model_limits_verified"])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)
        self.assertLessEqual(ledger.snapshot()["cpu"]["peak_bytes"], plan.ram_required_bytes)

    @unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1"
                         and os.environ.get("GLM_TEST_OFFICIAL") == "1",
                         "Requires native kernels and pinned official reference")
    def test_native_local_cpu_and_optional_cuda_match_official_graph(self):
        import torch
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from checkpoint_test_helpers import settings
        from glm_local.cpu_probe import NativeCpuBackend
        from glm_local.cuda_probe import CudaTileBackend
        from glm_local.gpu_gate import GpuBoundaryGate
        from glm_local.runtime_weights import RuntimeWeights

        torch.set_num_threads(1)
        backends = ["cpu", "hybrid"] if os.environ.get("GLM_TEST_CUDA") == "1" else ["cpu"]
        for backend in backends:
            root, config, payloads, plan = self._setup_local(hybrid=backend == "hybrid")
            official_config = deepcopy(config)
            del official_config["quantization_config"]
            official_config = GlmMoeDsaConfig(**official_config)
            official_config._attn_implementation = "eager"
            model = GlmMoeDsaForCausalLM(official_config).float().eval()
            model.load_state_dict(official_state(model, payloads.matrices, payloads.vectors, torch), strict=True)
            ledger = plan.allocator(include_cache=False)
            with ExitStack() as stack:
                cpu = stack.enter_context(NativeCpuBackend())
                gpu = gate = None
                if backend == "hybrid":
                    gpu = stack.enter_context(CudaTileBackend(0, max_operations=128))
                    gate = GpuBoundaryGate(device_index=0, target=0.6)
                weights = stack.enter_context(RuntimeWeights(root, settings(), cpu=cpu, gpu=gpu,
                    gate=gate, backend=backend, ledger=ledger, plan=plan))
                def official_topk(scores, count):
                    return torch.topk(torch.tensor(scores, dtype=torch.float32), count).indices.tolist()
                decoder = stack.enter_context(StreamingDecoder(config, weights, plan, ledger=ledger,
                                                attention_topk=official_topk))
                tokens = [1, 4, 2, 9, 3, 7, 0, 11]
                for position, token in enumerate(tokens):
                    actual = decoder.step(token)
                    with torch.inference_mode():
                        expected = model(input_ids=torch.tensor([tokens[:position + 1]]),
                                         use_cache=False).logits[0, -1]
                    with self.subTest(backend=backend, position=position):
                        torch.testing.assert_close(torch.tensor(actual), expected, rtol=0, atol=4e-6)
                self.assertGreater(weights.stats()["cpu_tiles"], 0)
                if gate:
                    gate.finish()
                    self.assertGreater(weights.stats()["gpu_tiles"], 0)
                    self.assertEqual(gate.summary()["admitted_submissions"], weights.stats()["gpu_tiles"])
                self.assertLessEqual(ledger.snapshot()["cpu"]["peak_bytes"], plan.ram_required_bytes)
            self.assertEqual(ledger.snapshot()["active_leases"], 0)


if __name__ == "__main__":
    unittest.main()
