"""Bounded expanded-MLA reuse preserves scalar graph math and source identity."""
from array import array
from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from glm_local.expanded_cache import ExpandedKvCache
from glm_local.residency import PlannerSettings, ReservationLedger, build_plan
from glm_local.streaming_decoder import StreamingDecoder
from test_streaming_decoder import ProceduralWeights, StoredPayloads, decoder_config
from test_batched_prefill import active_cache


class ExpandedCacheUnitTests(unittest.TestCase):
    def setUp(self):
        self.ledger = ReservationLedger(1024**2)
        self.cache = ExpandedKvCache(2, 2, 4, self.ledger)
        self.addCleanup(self.cache.clear)

    def test_exact_latent_bytes_layer_and_causal_position_determine_hit(self):
        compute = Mock(side_effect=lambda: array("f", [1, 2, 3, 4]))
        validate = Mock()
        first = self.cache.get(0, 0, array("f", [0.0, 1.0]), compute, validate)
        self.assertIs(self.cache.get(0, 0, array("f", [0.0, 1.0]), compute, validate), first)
        self.assertEqual((compute.call_count, validate.call_count), (1, 1))
        # Negative zero has different FP32 bytes and cannot reuse the old key.
        self.cache.get(0, 0, array("f", [-0.0, 1.0]), compute, validate)
        self.cache.get(0, 1, array("f", [-0.0, 1.0]), compute, validate)
        self.cache.get(1, 0, array("f", [-0.0, 1.0]), compute, validate)
        self.assertEqual(compute.call_count, 4)
        self.assertEqual(self.cache.stats()["hits"], 1)

    def test_per_layer_lru_truncate_clear_and_accounted_capacity(self):
        compute = Mock(side_effect=lambda: array("f", [1, 2, 3, 4]))
        for layer in (0, 1):
            for token in (0, 1):
                self.cache.get(layer, token, array("f", [token]), compute)
        self.cache.get(0, 0, array("f", [0]), compute)
        self.cache.get(0, 2, array("f", [2]), compute)
        self.assertEqual(set(self.cache.layers[0]), {0, 2})
        self.assertEqual(set(self.cache.layers[1]), {0, 1})
        maximum = 2 * 2 * (4 * 4 + 32)
        self.assertLessEqual(self.cache.stats()["bytes"], maximum)
        self.assertLessEqual(self.ledger.snapshot()["cpu"]["peak_bytes"], maximum)
        self.cache.truncate(1)
        self.assertEqual([set(layer) for layer in self.cache.layers], [{0}, {0}])
        self.cache.clear(); self.cache.clear()
        self.assertEqual(self.cache.stats()["bytes"], 0)
        self.assertEqual(self.ledger.snapshot()["active_leases"], 0)
        self.assertEqual(self.ledger.snapshot()["cpu"]["used_bytes"], 0)

    def test_disabled_cache_does_not_retain_or_reserve(self):
        disabled = ExpandedKvCache(2, 0, 4, ReservationLedger(0))
        compute = Mock(side_effect=lambda: array("f", [1, 2, 3, 4]))
        for _ in range(3):
            disabled.get(0, 0, array("f", [1]), compute)
        self.assertEqual(compute.call_count, 3)
        self.assertEqual(disabled.stats()["bytes"], 0)
        self.assertEqual(disabled.ledger.snapshot()["active_leases"], 0)

    def test_full_real_profile_token_layer_count_does_not_exhaust_ledger_lease_limit(self):
        #4992/19968logical entries; less than1MiB of tiny fake outputs.
        for tokens in (64, 256):
            ledger = ReservationLedger(1024**2)
            cache = ExpandedKvCache(78, tokens, 4, ledger)
            try:
                for layer in range(78):
                    for position in range(tokens):
                        cache.get(layer, position, array("f", [position]), lambda: array("f", [1, 2, 3, 4]))
                self.assertEqual(sum(len(layer) for layer in cache.layers), 78 * tokens)
                self.assertEqual(ledger.snapshot()["active_leases"], 78)
                self.assertLessEqual(ledger.snapshot()["cpu"]["peak_bytes"], 78 * tokens * (16 + 32))
            finally:
                cache.clear()
            self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_constructor_bounds_and_plan_capacity_include_layer_width_and_context_cap(self):
        for layers, tokens, width in ((0, 1, 4), (257, 1, 4), (True, 1, 4), (1, -1, 4),
                                     (1, 257, 4), (1, True, 4), (1, 1, 0), (1, 1, 1048577)):
            with self.assertRaises(ValueError):
                ExpandedKvCache(layers, tokens, width, self.ledger)
        config = decoder_config()
        options = PlannerSettings(context_tokens=16, max_new_tokens=4, expanded_cache_tokens=256)
        enabled = build_plan(config, options)
        disabled = build_plan(config, replace(options, expanded_cache_tokens=0))
        expected = 2 * 16 * (2 * (4 + 4) * 4 + 32)
        self.assertEqual(dict(enabled.cpu_components)["expanded_mla_cache"], expected)
        self.assertEqual(enabled.ram_required_bytes - disabled.ram_required_bytes, expected)

    def test_invalid_keys_or_unbounded_latents_fail_before_compute_and_leave_no_leases(self):
        compute = Mock(return_value=array("f", [1, 2, 3, 4]))
        for layer, position in ((-1, 0), (2, 0), (True, 0), (0, -1), (0, True), (0, 2**63)):
            with self.subTest(layer=layer, position=position), self.assertRaises(ValueError):
                self.cache.get(layer, position, array("f", [1]), compute)
        def never_consume():
            raise AssertionError("Unbounded latent iterator must not be drained")
            yield 1
        for latent in (never_consume(), "text", [], [math.nan], [math.inf], [True]):
            with self.subTest(latent=str(latent)[:50]), self.assertRaises(ValueError):
                self.cache.get(0, 0, latent, compute)
        compute.assert_not_called()
        self.assertEqual(self.ledger.snapshot()["active_leases"], 0)

    def test_compute_failure_shape_failure_and_budget_failure_release_reservations(self):
        for result in ([1, 2, 3, 4], array("d", [1, 2, 3, 4]), array("f", [1])):
            with self.assertRaises(ValueError):
                self.cache.get(0, 0, array("f", [1]), lambda: result)
            self.assertEqual(self.cache.stats()["bytes"], 0)
            self.assertEqual(self.ledger.snapshot()["active_leases"], 0)
        with self.assertRaises(KeyboardInterrupt):
            self.cache.get(0, 0, array("f", [1]), Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(self.ledger.snapshot()["active_leases"], 0)
        limited = ExpandedKvCache(1, 1, 4, ReservationLedger(47))
        compute = Mock()
        with self.assertRaises(ValueError):
            limited.get(0, 0, array("f", [1]), compute)
        compute.assert_not_called()
        self.assertEqual(limited.ledger.snapshot()["active_leases"], 0)

    def test_weight_validation_runs_on_every_hit_and_never_returns_stale_value(self):
        compute = Mock(return_value=array("f", [1, 2, 3, 4]))
        self.cache.get(0, 0, array("f", [1]), compute)
        validator = Mock(side_effect=RuntimeError("weight changed"))
        with self.assertRaisesRegex(RuntimeError, "weight changed"):
            self.cache.get(0, 0, array("f", [1]), compute, validator)
        validator.assert_called_once()
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(self.cache.stats()["hits"], 0)


class ValidatedWeights(ProceduralWeights):
    def __init__(self, config):
        super().__init__(config)
        self.validations = []
        self.changed = False
    def assert_weight_unchanged(self, name):
        self.validations.append(name)
        if self.changed:
            raise RuntimeError("retained projection source changed")


def make_decoder(tokens, *, config=None, weights=None):
    config = decoder_config() if config is None else config
    options = PlannerSettings(context_tokens=32, max_new_tokens=4, expanded_cache_tokens=tokens,
                              runtime_headroom_bytes=1024**2)
    estimate = build_plan(config, options)
    plan = build_plan(config, replace(options, ram_budget_bytes=estimate.ram_required_bytes))
    weights = ValidatedWeights(config) if weights is None else weights
    return StreamingDecoder(config, weights, plan), weights


class ExpandedDecoderTests(unittest.TestCase):
    def test_enabled_disabled_eviction_and_multi_indexer_logits_caches_generation_are_identical(self):
        config = decoder_config(layers=3, hidden=12, heads=3, experts=8)
        config.update(indexer_types=["full", "shared", "full"], index_topk=32, n_group=2, topk_group=1)
        prompt = [1, 4, 2, 9, 1, 3, 7, 8]
        oracle, _ = make_decoder(0, config=config)
        with oracle:
            logits, caches = [], []
            for token in prompt:
                logits.append(oracle.step(token).tobytes()); caches.append(active_cache(oracle))
            oracle.reset()
            generated = oracle.generate(prompt, 4, [])
            final_cache = active_cache(oracle)
        for capacity in (1, 2, 64):
            decoder, weights = make_decoder(capacity, config=config)
            with decoder:
                for index, token in enumerate(prompt):
                    self.assertEqual(decoder.step(token).tobytes(), logits[index])
                    self.assertEqual(active_cache(decoder), caches[index])
                if capacity == 64:
                    expansions = [name for name in weights.calls if name.endswith(".kv_b_proj.weight")]
                    self.assertEqual(len(expansions), len(prompt) * config["num_hidden_layers"])
                    self.assertGreater(len(weights.validations), len(expansions))
                decoder.reset()
                self.assertEqual(decoder._expanded_cache.bytes, 0)
                self.assertEqual(decoder.generate(prompt, 4, []), generated)
                self.assertEqual(active_cache(decoder), final_cache)
                self.assertLessEqual(decoder.ledger.snapshot()["cpu"]["peak_bytes"], decoder.plan.ram_required_bytes)
            self.assertEqual(decoder.ledger.snapshot()["active_leases"], 0)

    def test_scalar_failure_discards_uncommitted_expansions_and_retry_matches_fresh(self):
        decoder, weights = make_decoder(64)
        oracle, _ = make_decoder(0)
        with decoder, oracle:
            decoder.step(1); oracle.step(1)
            before = decoder.ledger.snapshot()["cpu"]["used_bytes"]
            weights.fail_name = "model.layers.1.self_attn.q_a_proj.weight"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                decoder.step(4)
            self.assertEqual(decoder.position, 1)
            self.assertTrue(all(all(key < 1 for key in layer) for layer in decoder._expanded_cache.layers))
            self.assertEqual(decoder.ledger.snapshot()["cpu"]["used_bytes"], before)
            weights.fail_name = None
            self.assertEqual(decoder.step(9), oracle.step(9))
            self.assertEqual(active_cache(decoder), active_cache(oracle))

    def test_batched_failure_truncates_expansions_and_retry_remains_exact(self):
        decoder, weights = make_decoder(64)
        oracle, _ = make_decoder(0)
        with decoder, oracle:
            decoder.step(1); oracle.step(1)
            weights.fail_name = "lm_head.weight"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                decoder.prefill([4, 2, 9], batch_size=3)
            self.assertEqual(decoder.position, 1)
            self.assertTrue(all(all(key < 1 for key in layer) for layer in decoder._expanded_cache.layers))
            weights.fail_name = None
            for token in (3, 7, 8):
                expected = oracle.step(token)
            self.assertEqual(decoder.prefill([3, 7, 8], batch_size=3), expected)
            self.assertEqual(active_cache(decoder), active_cache(oracle))

    def test_changed_source_rejects_cache_hit_before_reusing_expanded_values(self):
        decoder, weights = make_decoder(64)
        with decoder:
            decoder.step(1)
            weights.changed = True
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                decoder.step(4)
            self.assertEqual(decoder.position, 1)

    def test_fp8_scale_in_other_evicted_shard_is_a_cache_validation_dependency(self):
        from checkpoint_test_helpers import settings
        from glm_local.runtime_weights import RuntimeWeights
        from test_runtime_weights import CpuKernel, prepare_runtime_fixture
        from test_architecture_mapper import fixture_catalogue, header
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = "model.layers.0.self_attn.kv_b_proj.weight"
            scale = name + "_scale_inv"
            records = fixture_catalogue()
            index = next(i for i, record in enumerate(records) if record["name"] == name)
            records[index] = header(name, (16, 4), "F8_E4M3")
            records.insert(index + 1, header(scale, (1, 1), "F32"))
            with patch("test_runtime_weights.fixture_catalogue", return_value=records):
                directory, data = prepare_runtime_fixture(root, config_overrides=decoder_config(), payload_factory=StoredPayloads())
            weight_shard, scale_shard = (data["index"]["weight_map"][key] for key in (name, scale))
            self.assertNotEqual(weight_shard, scale_shard)
            with RuntimeWeights(root, settings(), cpu=CpuKernel(), max_open_shards=1) as weights:
                weights.linear(name, [0.125] * data["config"]["kv_lora_rank"])
                source = weights._reader._source
                source._reader(weight_shard)  # Evict the scale handle before the independent file mutation.
                self.assertNotIn(scale_shard, source._open)
                with (directory / scale_shard).open("ab") as stream:
                    stream.write(b" ")
                with self.assertRaisesRegex(ValueError, "changed|size"):
                    weights.assert_weight_unchanged(name)

    @unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "Explicit tiny native CPU FP8/NVFP4 fixture")
    def test_native_fp8_and_nvfp4_enabled_cache_match_disabled_under_exact_budgets(self):
        from test_batched_prefill import NativeBatchedPrefillTests
        from glm_local.cpu_probe import NativeCpuBackend
        from glm_local.runtime_weights import RuntimeWeights
        measurements = []
        for nvfp4 in (False, True):
            root, data, original_plan = NativeBatchedPrefillTests.fixture(self, nvfp4)
            results, counts = [], []
            for capacity in (0, 64):
                options = replace(original_plan.settings, expanded_cache_tokens=capacity, ram_budget_bytes=32_000_000_000)
                estimate = build_plan(data["config"], options)
                plan = build_plan(data["config"], replace(options, ram_budget_bytes=estimate.ram_required_bytes))
                ledger = plan.allocator(include_cache=False)
                with NativeCpuBackend(row_band_threads=2) as cpu, RuntimeWeights(root, data["settings"], cpu=cpu,
                        plan=plan, ledger=ledger) as weights, StreamingDecoder(data["config"], weights, plan, ledger=ledger) as decoder:
                    output = [decoder.step(token).tobytes() for token in (1, 4, 2, 9, 3, 7)]
                    results.append((output, active_cache(decoder)))
                    counts.append(weights.stats()["linear_calls"])
                    self.assertLessEqual(ledger.snapshot()["cpu"]["peak_bytes"], plan.ram_required_bytes)
                self.assertEqual(ledger.snapshot()["active_leases"], 0)
            self.assertEqual(results[0], results[1])
            self.assertLess(counts[1], counts[0])
            measurements.append({"quant_format": "nvfp4" if nvfp4 else "fp8", "tokens": 6,
                "bit_identical_logits_and_caches": True, "linear_calls_disabled_enabled": counts})
        if os.environ.get("GLM_EXPANDED_CACHE_REPORT"):
            import json
            from glm_local.execution import FULL_MODEL_FLAGS
            path = Path(os.environ["GLM_EXPANDED_CACHE_REPORT"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"status": "PASS", "scope": "tiny_native_cpu_expanded_mla_cache",
                "measurements": measurements, **FULL_MODEL_FLAGS}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
