"""Independent scalar-versus-batched prompt math, failure and native I/O tests."""
from array import array
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from glm_local.batched_prefill import prefill_chunk, _Projections
from glm_local.residency import PlannerSettings, build_plan
from glm_local.streaming_decoder import DecoderError, StreamingDecoder
from test_streaming_decoder import ProceduralWeights, StoredPayloads, decoder_config


class BatchWeights(ProceduralWeights):
    def __init__(self, config):
        super().__init__(config)
        self.batches = []

    def linear_many(self, name, vectors):
        self.batches.append((name, len(vectors)))
        return [self.linear(name, vector) for vector in vectors]

    def alternate_linear(self, name, values):
        return array("f", (value * 0.5 for value in self.linear(name, values)))


def make_decoder(config=None, *, custom=None, progress=None):
    config = decoder_config() if config is None else config
    initial = PlannerSettings(context_tokens=32, max_new_tokens=4, runtime_headroom_bytes=1024**2)
    estimate = build_plan(config, initial)
    plan = build_plan(config, replace(initial, ram_budget_bytes=estimate.ram_required_bytes))
    weights = BatchWeights(config)
    options = {} if custom is None else {"linear": custom(weights)}
    return StreamingDecoder(config, weights, plan, progress=progress, **options), weights


def active_cache(decoder):
    return ([cache[:decoder.position * decoder._width].tobytes() for cache in decoder._cache],
            {layer: cache[:decoder.position * decoder.config["index_head_dim"]].tobytes()
             for layer, cache in decoder._index_cache.items()})


class BatchedPrefillTests(unittest.TestCase):
    def test_direct_dense_moe_outputs_do_not_replay_scalar_routing_or_ffn(self):
        config = decoder_config(layers=3, hidden=12, heads=3, experts=8)
        config.update(indexer_types=["full", "shared", "full"], n_group=2, topk_group=1)
        tokens = list(range(1, 14))
        scalar, _ = make_decoder(config)
        batched, weights = make_decoder(config)
        with scalar, batched:
            for token in tokens:
                expected = scalar.step(token)
            with patch.object(batched, "_ffn", side_effect=AssertionError("scalar FFN replay")) as scalar_ffn, \
                    patch.object(batched, "_moe", side_effect=AssertionError("scalar MoE replay")) as scalar_moe:
                actual = batched.prefill(tokens, batch_size=16)
            self.assertEqual(actual, expected)
            self.assertEqual(active_cache(batched), active_cache(scalar))
            scalar_ffn.assert_not_called(); scalar_moe.assert_not_called()
            for layer in (1, 2):
                self.assertEqual(weights.calls.count(f"model.layers.{layer}.mlp.gate.e_score_correction_bias"), 1)
            self.assertEqual(batched.last_trace["projection_cache_misses"], 0)

    def test_fresh_prepared_results_are_not_recomputed_or_revalidated_after_lru_eviction(self):
        decoder, weights = make_decoder()
        vectors = [array("f", [value] * 16) for value in (0.125, 0.25, 0.5)]
        name = "model.layers.0.mlp.gate_proj.weight"
        expected = [weights.linear(name, vector) for vector in vectors]
        weights.calls.clear()
        weights.assert_weight_unchanged = Mock()
        cache = _Projections(decoder, weights.linear)
        with decoder, patch("glm_local.batched_prefill._CACHE_ENTRIES", 1):
            self.assertEqual(cache.prepare(name, vectors), expected)
            self.assertEqual(weights.calls, [name] * len(vectors))
            self.assertEqual(cache.misses, 0)
            weights.assert_weight_unchanged.assert_not_called()
            # A later retained memo hit still validates its source before reuse.
            self.assertEqual(cache.prepare(name, [vectors[-1]]), [expected[-1]])
            weights.assert_weight_unchanged.assert_called_once_with(name)
            self.assertEqual(weights.calls, [name] * len(vectors))

    def test_logits_and_both_caches_equal_scalar_across_prompt_and_chunk_boundaries(self):
        grouped = decoder_config(layers=3, hidden=12, heads=3, experts=8)
        grouped.update(indexer_types=["full", "shared", "full"], n_group=2, topk_group=1,
                       norm_topk_prob=False)
        for config in (decoder_config(), grouped):
            for length in (1, 2, 3, 16, 17):
                tokens = [(index * 7 + 1) % 32 for index in range(length)]
                scalar, _ = make_decoder(config)
                with scalar:
                    for token in tokens:
                        expected = scalar.step(token)
                    cache = active_cache(scalar)
                    selected = scalar.last_trace["last_selected_indices"]
                    expected_next = scalar.step(9)
                for batch_size in (1, 2, 3, 16):
                    with self.subTest(layers=config["num_hidden_layers"], length=length, batch=batch_size):
                        batched, weights = make_decoder(config)
                        with batched:
                            self.assertEqual(batched.prefill(tokens, batch_size=batch_size).tobytes(), expected.tobytes())
                            self.assertEqual(batched.position, length)
                            self.assertEqual(active_cache(batched), cache)
                            self.assertEqual(batched.last_trace["last_selected_indices"], selected)
                            self.assertEqual(batched.step(9).tobytes(), expected_next.tobytes())
                            self.assertTrue(weights.batches)
                            self.assertTrue(all(1 <= count <= batch_size for _, count in weights.batches))
                            self.assertLessEqual(batched.ledger.snapshot()["cpu"]["peak_bytes"], batched.plan.ram_required_bytes)
                        self.assertEqual(batched.ledger.snapshot()["active_leases"], 0)

    def test_generation_and_observers_equal_legacy_including_eos_and_prefill_only(self):
        tokens = [(i * 3 + 2) % 32 for i in range(17)]
        for count, eos in ((4, []), (4, list(range(32))), (0, [])):
            scalar, _ = make_decoder()
            with scalar:
                expected = scalar.generate(tokens, count, eos)
                cache, position = active_cache(scalar), scalar.position
            for size in (2, 3, 16):
                batched, _ = make_decoder()
                events = []
                with batched:
                    actual = batched.generate(tokens, count, eos, prefill_batch_size=size,
                        on_token=lambda token, index: events.append((token, index)))
                    self.assertEqual(actual, expected)
                    self.assertEqual(events, [(token, index) for index, token in enumerate(expected)])
                    self.assertEqual(batched.position, position)
                    self.assertEqual(active_cache(batched), cache)

    def test_duplicate_tokens_and_projection_eviction_preserve_causal_rotations(self):
        tokens = [1] * 17
        scalar, _ = make_decoder()
        with scalar:
            for token in tokens:
                expected = scalar.step(token)
            expected_cache = active_cache(scalar)
        for budget, entries in ((96 * 1024**2, 4096), (256, 2)):
            batched, _ = make_decoder()
            with batched, patch("glm_local.batched_prefill._CACHE_BYTES", budget), \
                    patch("glm_local.batched_prefill._CACHE_ENTRIES", entries):
                self.assertEqual(batched.prefill(tokens, batch_size=16), expected)
                self.assertEqual(active_cache(batched), expected_cache)

    def test_append_prefill_reuses_prior_committed_context_and_skips_intermediate_heads(self):
        scalar, _ = make_decoder()
        batched, weights = make_decoder()
        with scalar, batched:
            for token in (1, 4, 2):
                scalar.step(token); batched.step(token)
            old = active_cache(batched)
            for token in (9, 3, 7, 5, 8):
                expected = scalar.step(token)
            weights.calls.clear()
            actual = batched.prefill([9, 3, 7, 5, 8], batch_size=2)
            self.assertEqual(actual, expected)
            self.assertEqual(active_cache(batched), active_cache(scalar))
            self.assertEqual(sum(name == "lm_head.weight" for name in weights.calls), 1)
            for original, current in zip(old[0], active_cache(batched)[0]):
                self.assertTrue(current.startswith(original))

    def test_partial_layer_failure_restores_position_linear_trace_leases_and_retry(self):
        for failure in ("model.layers.1.self_attn.o_proj.weight", "model.layers.1.mlp.shared_experts.down_proj.weight", "lm_head.weight"):
            decoder, weights = make_decoder()
            oracle, _ = make_decoder()
            with decoder, oracle:
                decoder.step(1); oracle.step(1)
                committed, trace = active_cache(decoder), deepcopy(decoder.last_trace)
                original, allocations = decoder._linear, decoder.ledger.snapshot()
                weights.fail_name = failure
                with self.subTest(failure=failure), self.assertRaisesRegex(RuntimeError, "injected"):
                    decoder.prefill([4, 2, 9], batch_size=3)
                self.assertEqual(decoder.position, 1)
                self.assertEqual(active_cache(decoder), committed)
                self.assertEqual(decoder.last_trace, trace)
                self.assertEqual(decoder._linear, original)
                self.assertEqual(decoder.ledger.snapshot()["active_leases"], allocations["active_leases"])
                self.assertEqual(decoder.ledger.snapshot()["cpu"]["used_bytes"], allocations["cpu"]["used_bytes"])
                weights.fail_name = None
                for token in (4, 2, 9):
                    expected = oracle.step(token)
                self.assertEqual(decoder.prefill([4, 2, 9], batch_size=3), expected)
                self.assertEqual(active_cache(decoder), active_cache(oracle))

    def test_failure_in_second_chunk_keeps_first_commit_and_can_retry_only_tail(self):
        fired = False
        def observer(stage, **fields):
            nonlocal fired
            if stage == "prefill_layer_complete" and fields["layer_index"] == 1 and decoder.position >= 3 and not fired:
                fired = True
                raise RuntimeError("second chunk failure")
        decoder, _ = make_decoder(progress=observer)
        oracle, _ = make_decoder()
        with decoder, oracle:
            with self.assertRaisesRegex(RuntimeError, "second chunk"):
                decoder.prefill([1, 4, 2, 9, 3], batch_size=2)
            self.assertTrue(fired)
            self.assertEqual(decoder.position, 2)
            for token in (1, 4):
                oracle.step(token)
            self.assertEqual(active_cache(decoder), active_cache(oracle))
            for token in (2, 9, 3):
                expected = oracle.step(token)
            self.assertEqual(decoder.prefill([2, 9, 3], batch_size=2), expected)
            self.assertEqual(active_cache(decoder), active_cache(oracle))

    def test_invalid_input_and_closed_decoder_never_read_weights(self):
        decoder, weights = make_decoder()
        with decoder:
            for ids, batch in (([], 2), ([1], True), ([1], 0), ([1], 17), ([1, False], 2), ([1] * 33, 2)):
                with self.assertRaises(ValueError):
                    decoder.prefill(ids, batch_size=batch)
            for batch in (True, 0, 17):
                with self.assertRaises(ValueError):
                    decoder.generate([1], 1, [], prefill_batch_size=batch)
            self.assertEqual(weights.calls, [])
        with self.assertRaisesRegex(DecoderError, "closed"):
            decoder.prefill([1], batch_size=2)
        self.assertEqual(weights.calls, [])

    def test_oversized_accepted_generic_profile_fails_before_any_batch_weight_read(self):
        config = decoder_config()
        config["intermediate_size"] = 2 * 1024**2
        decoder, weights = make_decoder(config)
        with decoder, patch.object(weights, "embedding", side_effect=AssertionError("must not allocate/read")) as embedding:
            baseline = decoder.ledger.snapshot()
            with self.assertRaisesRegex(ValueError, "128-MiB workspace"):
                decoder.prefill([1] * 16, batch_size=16)
            embedding.assert_not_called()
            self.assertEqual(weights.calls, [])
            self.assertEqual(weights.batches, [])
            self.assertEqual(decoder.position, 0)
            self.assertEqual(decoder.ledger.snapshot(), baseline)

    def test_keyboard_interrupt_after_cache_write_restores_chunk_and_allows_retry(self):
        interrupt = True
        def observer(stage, **fields):
            nonlocal interrupt
            if stage == "projection_complete" and fields.get("tensor_name", "").endswith("indexer.weights_proj.weight") and interrupt:
                interrupt = False
                raise KeyboardInterrupt()
        decoder, _ = make_decoder(progress=observer)
        oracle, _ = make_decoder()
        with decoder, oracle:
            original = decoder._linear
            with self.assertRaises(KeyboardInterrupt):
                decoder.prefill([1, 4, 2], batch_size=3)
            self.assertEqual(decoder.position, 0)
            self.assertEqual(decoder._linear, original)
            self.assertEqual(decoder.ledger.snapshot()["active_leases"], 1)
            for token in (1, 4, 2):
                expected = oracle.step(token)
            self.assertEqual(decoder.prefill([1, 4, 2], batch_size=3), expected)
            self.assertEqual(active_cache(decoder), active_cache(oracle))

    def test_batch_lease_includes_context_dependent_scalar_attention_scratch(self):
        from glm_local.batched_prefill import PREFILL_SCRATCH_BYTES
        observed = []
        def observer(stage, **fields):
            if stage == "prefill_layer":
                observed.append(decoder.ledger.snapshot()["cpu"]["used_bytes"] - decoder._expanded_cache.reserved_bytes)
        decoder, _ = make_decoder(progress=observer)
        with decoder:
            baseline = decoder.ledger.snapshot()["cpu"]["used_bytes"]
            decoder.prefill([1, 4, 2], batch_size=3)
            self.assertEqual(observed, [baseline + PREFILL_SCRATCH_BYTES + decoder._scratch_bytes] * decoder.plan.layer_count)
            self.assertEqual(decoder.ledger.snapshot()["cpu"]["used_bytes"], baseline + decoder._expanded_cache.reserved_bytes)

    def test_custom_unbound_linear_override_never_uses_weights_default_batch_method(self):
        custom = lambda weights: lambda name, values: weights.alternate_linear(name, values)
        scalar, _ = make_decoder(custom=custom)
        batched, weights = make_decoder(custom=custom)
        with scalar, batched:
            for token in (1, 4, 2):
                expected = scalar.step(token)
            self.assertEqual(batched.prefill([1, 4, 2], batch_size=3), expected)
            self.assertEqual(active_cache(batched), active_cache(scalar))
            self.assertEqual(weights.batches, [])

    def test_custom_bound_linear_override_never_uses_weights_default_batch_method(self):
        custom = lambda weights: weights.alternate_linear
        scalar, _ = make_decoder(custom=custom)
        batched, weights = make_decoder(custom=custom)
        with scalar, batched:
            for token in (1, 4, 2):
                expected = scalar.step(token)
            self.assertEqual(batched.prefill([1, 4, 2], batch_size=3), expected)
            self.assertEqual(active_cache(batched), active_cache(scalar))
            self.assertEqual(weights.batches, [])


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "Explicit bounded native CPU fixture test")
class NativeBatchedPrefillTests(unittest.TestCase):
    def fixture(self, nvfp4):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = decoder_config()
        if nvfp4:
            from test_nvfp4_schema import nvfp4_config, prepare_nvfp4_fixture
            from test_nvfp4_runtime import NvPayloads
            config.update(quantization_config=deepcopy(nvfp4_config()["quantization_config"]), moe_intermediate_size=144)
            _, data = prepare_nvfp4_fixture(root, config_overrides=config, payload_factory=NvPayloads())
        else:
            from test_runtime_weights import prepare_runtime_fixture
            from checkpoint_test_helpers import settings
            _, data = prepare_runtime_fixture(root, config_overrides=config, payload_factory=StoredPayloads())
            data["settings"] = settings()
        options = PlannerSettings(context_tokens=32, max_new_tokens=4, runtime_headroom_bytes=1024**2)
        estimate = build_plan(data["config"], options)
        plan = build_plan(data["config"], replace(options, ram_budget_bytes=estimate.ram_required_bytes))
        return root, data, plan

    def test_real_native_cpu_fp8_and_nvfp4_batches_equal_scalar_and_release_exact_plan_leases(self):
        from glm_local.cpu_probe import NativeCpuBackend
        from glm_local.runtime_weights import RuntimeWeights
        measurements = []
        for nvfp4 in (False, True):
            root, data, plan = self.fixture(nvfp4)
            outputs, caches, stats = [], [], []
            for batch in (1, 3, 16):
                ledger = plan.allocator(include_cache=False)
                with NativeCpuBackend(row_band_threads=2) as cpu:
                    with RuntimeWeights(root, data["settings"], cpu=cpu, plan=plan, ledger=ledger) as weights:
                        with StreamingDecoder(data["config"], weights, plan, ledger=ledger) as decoder:
                            if batch == 1:
                                for token in range(1, 18):
                                    logits = decoder.step(token)
                            else:
                                logits = decoder.prefill(list(range(1, 18)), batch_size=batch)
                            outputs.append(logits.tobytes())
                            caches.append(active_cache(decoder))
                            stats.append(weights.stats())
                            self.assertLessEqual(ledger.snapshot()["cpu"]["peak_bytes"], plan.ram_required_bytes)
                            self.assertLessEqual(weights.stats()["peak_open_shards"], 2)
                self.assertEqual(ledger.snapshot()["active_leases"], 0)
            with self.subTest(nvfp4=nvfp4):
                self.assertEqual(outputs[1:], outputs[:1] * 2)
                self.assertEqual(caches[1:], caches[:1] * 2)
                self.assertGreater(stats[-1]["projection_batch_vectors"], stats[-1]["projection_batch_calls"])
                self.assertLess(stats[-1]["encoded_band_reads"], stats[0]["encoded_band_reads"])
                measurements.append({"quant_format": "nvfp4" if nvfp4 else "fp8", "prompt_tokens": 17,
                    "batches": [1, 3, 16], "logits_and_active_caches_bit_identical": True,
                    "maximum_absolute_logit_error": 0.0,
                    "encoded_band_reads": [value["encoded_band_reads"] for value in stats],
                    "batch_calls": [value["projection_batch_calls"] for value in stats],
                    "batch_vectors": [value["projection_batch_vectors"] for value in stats],
                    "planned_ram_bytes": plan.ram_required_bytes})
        if os.environ.get("GLM_BATCH_PREFILL_REPORT"):
            from glm_local.execution import FULL_MODEL_FLAGS
            path = Path(os.environ["GLM_BATCH_PREFILL_REPORT"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"status": "PASS", "scope": "synthetic_native_cpu_prefill",
                "measurements": measurements, **FULL_MODEL_FLAGS}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
