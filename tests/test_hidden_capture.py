"""Optional hidden-state instrumentation of the fixed invented mini decoder."""

from copy import deepcopy
import math
import unittest

from glm_local.mini_engine import MiniDecoder
from glm_local.mini_spec import SPEC, matrix_shapes, vector_lengths


STATE_NAMES = {"embedding", "final_norm"} | {
    f"layer.{layer}.{name}"
    for layer in range(2)
    for name in ("input_norm", "attention_output", "post_attention",
                 "post_attention_norm", "output")
}


class CaptureWeights:
    def __init__(self):
        self.vectors = {name: [1.0] * length for name, length in vector_lengths().items()}
        self.vectors["layer.0.index_norm_bias"] = [0.0] * 8
        self.vectors["layer.1.router_bias"] = [0.0] * 4
        self.embeddings = [[(token + index + 1) / 32.0 for index in range(16)]
                           for token in range(32)]
        self.embedding_calls = []
        self.vector_calls = []

    def embedding(self, token):
        self.embedding_calls.append(token)
        return self.embeddings[token]

    def vector(self, name):
        self.vector_calls.append(name)
        return self.vectors[name]


class CaptureLinear:
    def __init__(self, *, zero_ffn=False):
        self.shapes = matrix_shapes()
        self.calls = []
        self.inputs = []
        self.outputs = []
        self.zero_ffn = zero_ffn

    def __call__(self, name, values):
        self.calls.append(name)
        self.inputs.append(values)
        rows, columns = self.shapes[name]
        if self.zero_ffn:
            if name in ("layer.0.o", "layer.1.o"):
                sign = 1.0 if name == "layer.0.o" else -1.0
                result = [sign * (index + 1) / 64.0 for index in range(rows)]
            elif name == "lm_head":
                result = list(values) + list(values)
            else:
                result = [0.0] * rows
        else:
            phase = sum(ord(character) for character in name)
            result = [math.fsum(math.cos(phase + row * 3 + column) * values[column] / 50.0
                                for column in range(columns)) for row in range(rows)]
        self.outputs.append(result)
        return result


def make_decoder(*, capture_states=False, zero_ffn=False):
    weights = CaptureWeights()
    linear = CaptureLinear(zero_ffn=zero_ffn)
    return MiniDecoder(weights, linear, capture_states=capture_states), weights, linear


def rms_expected(values):
    inverse = 1.0 / math.sqrt(math.fsum(value * value for value in values) / 16 + SPEC["norm_eps"])
    return [value * inverse for value in values]


class HiddenCaptureTests(unittest.TestCase):
    def test_enabled_and_disabled_have_identical_outputs_cache_calls_and_base_trace(self):
        captured, captured_weights, captured_linear = make_decoder(capture_states=True)
        ordinary, ordinary_weights, ordinary_linear = make_decoder()
        for token in (3, 1, 5, 7, 2, 9):
            self.assertEqual(captured.step(token), ordinary.step(token))
            self.assertEqual(captured.position, ordinary.position)
            self.assertEqual(captured.cache_bytes, ordinary.cache_bytes)
            self.assertEqual(captured.cache_used_bytes, ordinary.cache_used_bytes)
            self.assertEqual(captured._keys, ordinary._keys)
            self.assertEqual(captured._values, ordinary._values)
            self.assertEqual(captured._index_keys, ordinary._index_keys)
            record = deepcopy(captured.trace[-1])
            states = record.pop("hidden_states")
            self.assertEqual(record, ordinary.trace[-1])
            self.assertEqual(set(states), STATE_NAMES)
            self.assertNotIn("hidden_states", ordinary.trace[-1])
        self.assertEqual(captured_weights.embedding_calls, ordinary_weights.embedding_calls)
        self.assertEqual(captured_weights.vector_calls, ordinary_weights.vector_calls)
        self.assertEqual(captured_linear.calls, ordinary_linear.calls)

    def test_captures_are_finite_independent_copies_and_cannot_affect_future_steps(self):
        decoder, weights, linear = make_decoder(capture_states=True)
        reference, _, _ = make_decoder(capture_states=True)
        decoder.step(2)
        reference.step(2)
        states = decoder.trace[-1]["hidden_states"]
        self.assertEqual(len({id(values) for values in states.values()}), len(STATE_NAMES))
        live_vectors = weights.embeddings + list(weights.vectors.values())
        adapter_vectors = linear.inputs + linear.outputs
        for name, values in states.items():
            with self.subTest(name=name):
                self.assertIsInstance(values, list)
                self.assertEqual(len(values), 16)
                self.assertTrue(all(math.isfinite(value) for value in values))
                self.assertTrue(all(values is not original for original in live_vectors + adapter_vectors))
                values[0] = 1000.0
        self.assertEqual(decoder.step(6), reference.step(6))
        self.assertEqual(decoder.trace[-1], reference.trace[-1])

    def test_later_steps_and_external_weight_mutation_do_not_modify_prior_captures(self):
        decoder, weights, _ = make_decoder(capture_states=True)
        decoder.step(4)
        first = deepcopy(decoder.trace[0]["hidden_states"])
        weights.embeddings[4][0] = 999.0
        for token in (7, 9, 3):
            decoder.step(token)
        self.assertEqual(decoder.trace[0]["hidden_states"], first)
        all_states = [values for record in decoder.trace for values in record["hidden_states"].values()]
        self.assertEqual(len({id(values) for values in all_states}), len(all_states))

    def test_nodes_match_hand_calculated_residuals_and_normalizations_with_zero_ffn(self):
        decoder, weights, _ = make_decoder(capture_states=True, zero_ffn=True)
        logits = decoder.step(5)
        states = decoder.trace[-1]["hidden_states"]
        hidden = list(weights.embeddings[5])
        self.assertEqual(states["embedding"], hidden)
        for layer in range(2):
            prefix = f"layer.{layer}."
            self.assertEqual(states[prefix + "input_norm"], rms_expected(hidden))
            sign = 1.0 if layer == 0 else -1.0
            attention = [sign * (index + 1) / 64.0 for index in range(16)]
            self.assertEqual(states[prefix + "attention_output"], attention)
            hidden = [base + delta for base, delta in zip(hidden, attention)]
            self.assertEqual(states[prefix + "post_attention"], hidden)
            self.assertEqual(states[prefix + "post_attention_norm"], rms_expected(hidden))
            self.assertEqual(states[prefix + "output"], hidden)
        self.assertEqual(states["final_norm"], rms_expected(hidden))
        self.assertEqual(logits, states["final_norm"] * 2)

    def test_reset_releases_capture_records_and_reproduces_initial_result(self):
        decoder, _, _ = make_decoder(capture_states=True)
        logits = decoder.step(8)
        initial = deepcopy(decoder.trace)
        decoder.step(3)
        decoder.reset()
        self.assertEqual(decoder.trace, [])
        self.assertEqual(decoder.position, 0)
        self.assertEqual(decoder.cache_used_bytes, 0)
        self.assertEqual(decoder.cache_bytes, 57344)
        self.assertEqual(decoder.step(8), logits)
        self.assertEqual(decoder.trace, initial)

    def test_capture_is_bounded_by_existing_context_limit(self):
        decoder, weights, linear = make_decoder(capture_states=True, zero_ffn=True)
        for token in range(128):
            decoder.step(token % 32)
        self.assertEqual(len(decoder.trace), 128)
        self.assertEqual(decoder.cache_used_bytes, decoder.cache_bytes)
        before = deepcopy(decoder.trace)
        calls = len(weights.embedding_calls), len(linear.calls)
        with self.assertRaisesRegex(ValueError, "128 tokens"):
            decoder.step(0)
        self.assertEqual(decoder.trace, before)
        self.assertEqual((len(weights.embedding_calls), len(linear.calls)), calls)

    def test_capture_argument_requires_bool_and_is_keyword_only(self):
        for value in (None, 0, 1, "true", [], {}, 1.0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "capture_states.*bool"):
                MiniDecoder(CaptureWeights(), CaptureLinear(), capture_states=value)
        with self.assertRaises(TypeError):
            MiniDecoder(CaptureWeights(), CaptureLinear(), True)


if __name__ == "__main__":
    unittest.main()
