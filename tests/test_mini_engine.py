"""Isolated miniature decoder invariants; no native libraries or model data."""

from copy import deepcopy
import math
import unittest

from glm_local.mini_engine import MiniDecoder, _layer_norm, _rope, _sigmoid
from glm_local.mini_spec import matrix_shapes, vector_lengths


class InventedWeights:
    def __init__(self):
        self.vectors = {name: [1.0] * length for name, length in vector_lengths().items()}
        self.vectors["layer.0.index_norm_bias"] = [0.0] * 8
        self.vectors["layer.1.router_bias"] = [0.0] * 4
        self.embedding_calls = []

    def vector(self, name):
        return self.vectors[name]

    def embedding(self, token):
        self.embedding_calls.append(token)
        return [math.sin((token + 1) * (index + 1) * 0.19) for index in range(16)]


class InventedLinear:
    def __init__(self, overrides=None):
        self.shapes = matrix_shapes()
        self.calls = []
        self.overrides = overrides or {}

    def __call__(self, name, values):
        self.calls.append(name)
        if name in self.overrides:
            return list(self.overrides[name])
        rows, columns = self.shapes[name]
        phase = sum(ord(character) for character in name)
        return [math.fsum(math.sin((row + 1) * (column + 1) + phase) * values[column] * 0.07
                          for column in range(columns)) for row in range(rows)]


def make_decoder(overrides=None):
    weights, linear = InventedWeights(), InventedLinear(overrides)
    return MiniDecoder(weights, linear), weights, linear


class MiniMathTests(unittest.TestCase):
    def test_rope_adjacent_pairs_produce_split_output_order(self):
        self.assertEqual(_rope([1.0, 2.0, 3.0, 4.0], 0), [1.0, 3.0, 2.0, 4.0])
        result = _rope([1.0, 2.0, 3.0, 4.0], 2)
        expected = [math.cos(2) - 2 * math.sin(2),
                    3 * math.cos(0.02) - 4 * math.sin(0.02),
                    2 * math.cos(2) + math.sin(2),
                    4 * math.cos(0.02) + 3 * math.sin(0.02)]
        for actual, reference in zip(result, expected):
            self.assertAlmostEqual(actual, reference, places=14)

    def test_index_layernorm_subtracts_mean_and_adds_bias(self):
        self.assertEqual(_layer_norm([5.0] * 8, [2.0] * 8, list(range(8)), 1e-6),
                         list(range(8)))

    def test_sigmoid_large_magnitudes_do_not_overflow(self):
        self.assertEqual(_sigmoid(1000), 1.0)
        self.assertEqual(_sigmoid(-1000), 0.0)


class MiniDecoderTests(unittest.TestCase):
    def test_incremental_trace_is_causal_and_sparse(self):
        decoder, _, _ = make_decoder()
        for position, token in enumerate([3, 5, 2, 7, 1, 9, 4]):
            logits = decoder.step(token)
            self.assertEqual(len(logits), 32)
            self.assertTrue(all(math.isfinite(value) for value in logits))
            trace = decoder.trace[-1]
            self.assertEqual(trace["position"], position)
            self.assertEqual(len(trace["selected_indices"]), min(position + 1, 4))
            self.assertTrue(all(0 <= selected <= position for selected in trace["selected_indices"]))
            for layer in trace["layers"]:
                self.assertEqual(layer["selected_indices"], trace["selected_indices"])
                for probabilities in layer["attention_probabilities"]:
                    self.assertAlmostEqual(sum(probabilities), 1.0, places=14)
                    self.assertTrue(all(value >= 0 for value in probabilities))

    def test_future_steps_leave_prior_results_and_cache_prefix_unchanged(self):
        decoder, _, _ = make_decoder()
        initial = decoder.step(7)
        first_trace = deepcopy(decoder.trace)
        initial_keys = [list(cache[:16]) for cache in decoder._keys]
        for token in [2, 4, 8]:
            decoder.step(token)
        self.assertEqual(decoder.trace[:1], first_trace)
        self.assertEqual([list(cache[:16]) for cache in decoder._keys], initial_keys)
        fresh, _, _ = make_decoder()
        self.assertEqual(fresh.step(7), initial)

    def test_selection_bias_does_not_become_combination_weight(self):
        logits = [-4.0, -2.0, 0.0, 2.0]
        decoder, weights, _ = make_decoder({"layer.1.router": logits})
        weights.vectors["layer.1.router_bias"] = [20.0, 10.0, 0.0, 0.0]
        decoder.step(1)
        trace = decoder.trace[-1]
        self.assertEqual(trace["routed_experts"], [0, 1])
        original = [1 / (1 + math.exp(4)), 1 / (1 + math.exp(2))]
        expected = [score / sum(original) * 2.5 for score in original]
        for actual, reference in zip(trace["router_weights"], expected):
            self.assertAlmostEqual(actual, reference, places=14)
        self.assertLess(trace["router_weights"][0], trace["router_weights"][1])

    def test_shared_layer_has_own_key_values_and_no_indexer_calls(self):
        decoder, _, linear = make_decoder({"layer.0.index_weight": [0.0, 0.0],
                                           "layer.0.kv_b": [0.25] * 16,
                                           "layer.1.kv_b": [0.75] * 16})
        for token in range(6):
            decoder.step(token)
        trace = decoder.trace[-1]
        self.assertEqual(trace["selected_indices"], [0, 1, 2, 3])
        self.assertEqual(trace["layers"][1]["selected_indices"], [0, 1, 2, 3])
        self.assertEqual(list(decoder._values[0][:8]), [0.25] * 8)
        self.assertEqual(list(decoder._values[1][:8]), [0.75] * 8)
        self.assertEqual(linear.calls.count("layer.0.index_q"), 6)
        self.assertFalse(any(name.startswith("layer.1.index") for name in linear.calls))

    def test_signed_index_head_weight_can_reverse_sparse_selection(self):
        selections = []
        for multiplier in [1.0, -1.0]:
            decoder, weights, _ = make_decoder({
                "layer.0.index_q": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2,
                "layer.0.index_weight": [multiplier, 0.0],
            })
            weights.vectors["layer.0.index_norm_weight"] = [0.0] * 8
            for position in range(6):
                # A learned LayerNorm bias supplies a distinct positive cached
                # key coordinate; all RoPE coordinates remain zero.
                weights.vectors["layer.0.index_norm_bias"] = [0.0] * 4 + [position + 1.0] + [0.0] * 3
                decoder.step(position)
            selections.append(decoder.trace[-1]["selected_indices"])
        self.assertEqual(selections, [[5, 4, 3, 2], [0, 1, 2, 3]])

    def test_only_selected_routed_experts_run_and_shared_expert_always_runs(self):
        decoder, _, linear = make_decoder({"layer.1.router": [0.0] * 4})
        decoder.step(0)
        self.assertEqual(decoder.trace[-1]["routed_experts"], [0, 1])
        for expert in [0, 1]:
            for projection in ["gate", "up", "down"]:
                self.assertIn(f"layer.1.expert.{expert}.{projection}", linear.calls)
        for expert in [2, 3]:
            self.assertFalse(any(name.startswith(f"layer.1.expert.{expert}.") for name in linear.calls))
        for projection in ["gate", "up", "down"]:
            self.assertIn(f"layer.1.shared.{projection}", linear.calls)

    def test_cache_allocation_is_fixed_and_reset_reproduces_logits(self):
        decoder, _, _ = make_decoder()
        self.assertEqual(decoder.cache_bytes, 57344)
        self.assertEqual(decoder.cache_used_bytes, 0)
        first = decoder.step(4)
        decoder.step(5)
        self.assertEqual(decoder.cache_used_bytes, 896)
        self.assertEqual(decoder.cache_bytes, 57344)
        decoder.reset()
        self.assertEqual(decoder.position, 0)
        self.assertEqual(decoder.trace, [])
        self.assertEqual(decoder.cache_used_bytes, 0)
        self.assertEqual(decoder.step(4), first)

    def test_invalid_tokens_are_rejected_before_adapter_or_cache_mutation(self):
        decoder, weights, linear = make_decoder()
        decoder.step(5)
        before = (decoder.position, deepcopy(decoder.trace), list(linear.calls),
                  list(weights.embedding_calls), [list(cache) for cache in decoder._keys])
        for token in [-1, 32, True, False, 1.0, "1", None, [], {}]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                decoder.step(token)
            after = (decoder.position, decoder.trace, linear.calls,
                     weights.embedding_calls, [list(cache) for cache in decoder._keys])
            self.assertEqual(after, before)

    def test_context_limit_rejects_129th_token_before_work(self):
        decoder, weights, linear = make_decoder()
        for position in range(128):
            decoder.step(position % 32)
        self.assertEqual(decoder.position, 128)
        self.assertEqual(decoder.cache_used_bytes, decoder.cache_bytes)
        call_counts = len(linear.calls), len(weights.embedding_calls)
        with self.assertRaisesRegex(ValueError, "128 tokens"):
            decoder.step(0)
        self.assertEqual((len(linear.calls), len(weights.embedding_calls)), call_counts)
        self.assertEqual(len(decoder.trace), 128)

    def test_bad_adapter_shape_and_nonfinite_output_are_rejected(self):
        for output in ([0.0] * 7, [float("nan")] * 8, [float("inf")] * 8):
            decoder, _, _ = make_decoder({"layer.0.q_a": output})
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "linear output"):
                decoder.step(0)
            self.assertEqual(decoder.position, 0)
            self.assertEqual(decoder.trace, [])


if __name__ == "__main__":
    unittest.main()
