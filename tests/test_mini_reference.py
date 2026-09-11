"""Independent tiny-oracle tests: hand results, causality, and file boundaries."""

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    if os.environ.get("GLM_TEST_MINI") == "1":
        raise
    raise unittest.SkipTest("Optional miniature reference requires NumPy; install the validation extra")

from glm_local.mini_reference import ReferenceDecoder, _rotate_adjacent
from glm_local.mini_spec import SPEC, matrix_shapes, vector_lengths


def fixture(directory, *, matrices=None, vectors=None, random_seed=None):
    """Write tiny fixtures directly; deliberately do not use the engine's writer."""
    matrices, vectors = matrices or {}, vectors or {}
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    records, payload = {}, bytearray()
    for name, (rows, cols) in matrix_shapes().items():
        raw = np.zeros((rows, cols), dtype=np.uint8)
        if rng is not None:
            # Include both signs and zeros, with modest exact binary scales.
            raw = rng.choice(np.array([0, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8], dtype=np.uint8),
                             size=(rows, cols))
        for row, column, code in matrices.get(name, []):
            raw[row, column] = code
        records[name] = {"offset": len(payload), "rows": rows, "cols": cols,
                         "scale": 0.0625 if rng is not None else 1.0}
        payload.extend(raw.tobytes())
    vector_data = {}
    for name, length in vector_lengths().items():
        value = 0.0 if name.endswith("bias") else 1.0
        vector_data[name] = list(vectors.get(name, [value] * length))
    manifest = {"format": "glm-synthetic-mini-v1", "spec": SPEC, "seed": 0,
                "weight_file": "weights.bin", "weight_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "matrices": records, "vectors": vector_data}
    (directory / "weights.bin").write_bytes(payload)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def rewrite_manifest(directory, manifest):
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class MiniReferenceMathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_zero_checkpoint_has_exact_logits_and_deterministic_ties(self):
        fixture(self.directory)
        result = ReferenceDecoder(self.directory).forward([1, 2, 3, 4, 5, 6])
        self.assertEqual(result["logits"], [[0.0] * 32] * 6)
        for position, trace in enumerate(result["traces"]):
            self.assertEqual(trace["selected_indices"], list(range(min(position + 1, 4))))
            self.assertEqual(trace["routed_experts"], [0, 1])
            self.assertEqual(trace["router_weights"], [1.25, 1.25])

    def test_zero_residual_branches_preserve_identity_embedding(self):
        identity = [(row, row % 16, 0x38) for row in range(32)]
        fixture(self.directory, matrices={"embed": identity, "lm_head": identity})
        result = ReferenceDecoder(self.directory).forward((0, 1, 17))
        amplitude = 1.0 / math.sqrt(1.0 / 16.0 + 1e-5)
        expected = [[amplitude if column % 16 == token % 16 else 0.0
                     for column in range(32)] for token in (0, 1, 17)]
        np.testing.assert_allclose(result["logits"], expected, rtol=0, atol=1e-14)

    def test_uniform_attention_matches_hand_average(self):
        fixture(self.directory, matrices={
            "embed": [(0, 0, 0x38), (1, 0, 0xb8)],
            "lm_head": [(0, 0, 0x38)],
            "layer.0.kv_a": [(0, 0, 0x38)],
            "layer.0.kv_b": [(4, 0, 0x38)],
            "layer.0.o": [(0, 0, 0x38)],
        })
        result = ReferenceDecoder(self.directory).forward([0, 1])
        normalized = 1.0 / math.sqrt(1.0 / 16.0 + 1e-5)
        value = normalized / math.sqrt(normalized**2 / 4.0 + 1e-6)
        # Query/key dots are zero. The first query sees +value; the second
        # sees the mean of opposite values, exactly zero.
        residuals = [1.0 + value, -1.0]
        expected = [[x / math.sqrt(x * x / 16.0 + 1e-5)] + [0.0] * 31 for x in residuals]
        np.testing.assert_allclose(result["logits"], expected, rtol=0, atol=1e-13)

    def test_rotary_uses_adjacent_pairs_and_even_then_odd_layout(self):
        values = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        expected = [[1.0, 3.0, 2.0, 4.0],
                    [math.cos(1.0) - 2 * math.sin(1.0),
                     3 * math.cos(0.01) - 4 * math.sin(0.01),
                     math.sin(1.0) + 2 * math.cos(1.0),
                     3 * math.sin(0.01) + 4 * math.cos(0.01)]]
        np.testing.assert_allclose(_rotate_adjacent(values), expected, rtol=0, atol=1e-15)

    def test_index_key_layernorm_centers_a_constant_projection(self):
        fixture(self.directory, matrices={
            "embed": [(0, 0, 0x38)],
            "layer.0.q_a": [(0, 0, 0x38)],
            "layer.0.index_q": [(0, 0, 0x38)],
            "layer.0.index_k": [(row, 0, 0x38) for row in range(8)],
            "layer.0.index_weight": [(0, 0, 0x38)],
        })
        # Each index key is constant across its channels. LayerNorm subtracts
        # that constant, so all scores tie despite nonzero query/head weights.
        result = ReferenceDecoder(self.directory).forward([0] * 9)
        for position, trace in enumerate(result["traces"]):
            self.assertEqual(trace["selected_indices"], list(range(min(position + 1, 4))))

    def test_router_bias_selects_but_does_not_reweight_original_sigmoids(self):
        fixture(self.directory, matrices={
            "embed": [(0, 0, 0x38)],
            "layer.1.router": [(0, 0, 0x38), (1, 0, 0xb8)],
        }, vectors={"layer.1.router_bias": [0.0, 5.0, 0.0, -1.0]})
        trace = ReferenceDecoder(self.directory).forward([0])["traces"][0]
        normalized = 1.0 / math.sqrt(1.0 / 16.0 + 1e-5)
        probability = 1.0 / (1.0 + math.exp(-normalized))
        self.assertEqual(trace["routed_experts"], [1, 0])
        np.testing.assert_allclose(trace["router_weights"],
                                   [2.5 * (1.0 - probability), 2.5 * probability],
                                   rtol=0, atol=1e-14)

    def test_shared_expert_is_added_without_routing_weight(self):
        matrices = {"embed": [(0, 0, 0x38)],
                    "lm_head": [(0, 0, 0x38), (1, 1, 0x38)]}
        for prefix in ("layer.1.expert.0.", "layer.1.expert.1.", "layer.1.shared."):
            matrices[prefix + "gate"] = [(0, 0, 0x38)]
            matrices[prefix + "up"] = [(0, 0, 0x38)]
            matrices[prefix + "down"] = [(1, 0, 0x38)]
        fixture(self.directory, matrices=matrices)
        logits = ReferenceDecoder(self.directory).forward([0])["logits"][0]
        normalized = 1.0 / math.sqrt(1.0 / 16.0 + 1e-5)
        feedforward = normalized**2 / (1.0 + math.exp(-normalized))
        second = (1.25 + 1.25 + 1.0) * feedforward
        denominator = math.sqrt((1.0 + second**2) / 16.0 + 1e-5)
        np.testing.assert_allclose(logits, [1.0 / denominator, second / denominator] + [0.0] * 30,
                                   rtol=0, atol=1e-13)

    def test_signed_subnormal_and_largest_finite_fp8_have_hand_decoded_logits(self):
        fixture(self.directory, matrices={
            "embed": [(0, 0, 0x01), (0, 1, 0x81), (0, 2, 0x7e), (0, 3, 0xfe)],
            "lm_head": [(0, 0, 0x38), (1, 1, 0x38), (2, 2, 0x38), (3, 3, 0x38)],
        })
        values = [1.0 / 512, -1.0 / 512, 448.0, -448.0]
        denominator = math.sqrt(sum(value**2 for value in values) / 16.0 + 1e-5)
        expected = [value / denominator for value in values] + [0.0] * 28
        logits = ReferenceDecoder(self.directory).forward([0])["logits"][0]
        np.testing.assert_allclose(logits, expected, rtol=0, atol=1e-14)

    def test_random_full_sequence_is_causal_and_calls_are_stateless(self):
        fixture(self.directory, random_seed=654)
        decoder = ReferenceDecoder(self.directory)
        prefix = [1, 2, 3, 4, 5]
        small = decoder.forward(prefix)
        large = decoder.forward(prefix + [31, 0, 17, 2])
        alternate = decoder.forward(prefix + [0, 0, 0, 0])
        np.testing.assert_allclose(small["logits"], large["logits"][:5], rtol=0, atol=1e-13)
        np.testing.assert_allclose(small["logits"], alternate["logits"][:5], rtol=0, atol=1e-13)
        self.assertEqual(small["traces"], large["traces"][:5])
        self.assertEqual(small, decoder.forward(prefix))
        for position, trace in enumerate(large["traces"]):
            self.assertEqual(len(trace["selected_indices"]), min(position + 1, 4))
            self.assertTrue(all(0 <= index <= position for index in trace["selected_indices"]))
            self.assertAlmostEqual(sum(trace["router_weights"]), 2.5, places=14)

    def test_invalid_token_inputs_are_rejected(self):
        fixture(self.directory)
        decoder = ReferenceDecoder(self.directory)
        for tokens in ([], [True], [1.0], [-1], [32], [0] * 129, "1", None, np.array([1])):
            with self.subTest(tokens=str(tokens)[:50]):
                with self.assertRaises(ValueError):
                    decoder.forward(tokens)

    def test_maximum_context_is_supported(self):
        fixture(self.directory)
        self.assertEqual(len(ReferenceDecoder(self.directory).forward([0] * 128)["logits"]), 128)


class MiniReferenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_manifest_validation_rejects_changed_contract(self):
        mutations = {
            "format": lambda manifest: manifest.update(format="real-model"),
            "spec": lambda manifest: manifest.update(spec={**SPEC, "layers": 40}),
            "bool_spec": lambda manifest: manifest.update(spec={**SPEC, "layers": True}),
            "seed": lambda manifest: manifest.update(seed=True),
            "weight_file": lambda manifest: manifest.update(weight_file="../weights.bin"),
            "byte_count": lambda manifest: manifest.update(weight_bytes=True),
            "checksum": lambda manifest: manifest.update(sha256="0" * 64),
            "extra_field": lambda manifest: manifest.update(tokenizer="external"),
            "missing_matrix": lambda manifest: manifest["matrices"].pop("embed"),
            "missing_vector": lambda manifest: manifest["vectors"].pop("final_norm"),
            "shape": lambda manifest: manifest["matrices"]["embed"].update(rows=31),
            "overlap": lambda manifest: manifest["matrices"]["lm_head"].update(offset=0),
            "negative_offset": lambda manifest: manifest["matrices"]["embed"].update(offset=-1),
            "bool_offset": lambda manifest: manifest["matrices"]["embed"].update(offset=False),
            "zero_scale": lambda manifest: manifest["matrices"]["embed"].update(scale=0),
            "bool_scale": lambda manifest: manifest["matrices"]["embed"].update(scale=True),
            "nan_scale": lambda manifest: manifest["matrices"]["embed"].update(scale=math.nan),
            "huge_integer": lambda manifest: manifest["matrices"]["embed"].update(scale=10**1000),
            "short_vector": lambda manifest: manifest["vectors"].update(final_norm=[1.0]),
            "nan_vector": lambda manifest: manifest["vectors"].update(final_norm=[math.nan] * 16),
            "bool_vector": lambda manifest: manifest["vectors"].update(final_norm=[True] * 16),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                manifest = fixture(self.directory)
                mutate(manifest)
                rewrite_manifest(self.directory, manifest)
                with self.assertRaises(ValueError):
                    ReferenceDecoder(self.directory)

    def test_duplicate_json_fields_are_rejected(self):
        fixture(self.directory)
        path = self.directory / "manifest.json"
        content = path.read_text(encoding="utf-8")
        path.write_text(content[:-1] + ', "seed": 1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ReferenceDecoder(self.directory)

    def test_nonfinite_fp8_codes_are_rejected_even_with_valid_hash(self):
        for code in (0x7f, 0xff):
            with self.subTest(code=code):
                manifest = fixture(self.directory)
                payload = bytearray((self.directory / "weights.bin").read_bytes())
                payload[0] = code
                (self.directory / "weights.bin").write_bytes(payload)
                manifest["sha256"] = hashlib.sha256(payload).hexdigest()
                rewrite_manifest(self.directory, manifest)
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    ReferenceDecoder(self.directory)

    def test_truncated_and_extra_payloads_are_rejected(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                fixture(self.directory)
                path = self.directory / "weights.bin"
                payload = path.read_bytes()
                path.write_bytes(payload[:-1] if delta < 0 else payload + b"\0")
                with self.assertRaisesRegex(ValueError, "truncated|trailing"):
                    ReferenceDecoder(self.directory)

    def test_file_size_limits_apply_to_both_files(self):
        for name in ("manifest.json", "weights.bin"):
            with self.subTest(name=name):
                fixture(self.directory)
                (self.directory / name).write_bytes(b" " * (64 * 1024 + 1))
                with self.assertRaisesRegex(ValueError, "64 KiB"):
                    ReferenceDecoder(self.directory)

    def test_nonfinite_forward_arithmetic_raises_a_clear_error(self):
        fixture(self.directory, matrices={"embed": [(0, 0, 0x38)]},
                vectors={"layer.0.in_norm": [1e300] * 16})
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        manifest["matrices"]["layer.0.q_a"]["scale"] = 1e300
        rewrite_manifest(self.directory, manifest)
        # Make q_a nonzero while retaining an independently recomputed hash.
        payload = bytearray((self.directory / "weights.bin").read_bytes())
        payload[manifest["matrices"]["layer.0.q_a"]["offset"]] = 0x38
        (self.directory / "weights.bin").write_bytes(payload)
        manifest["sha256"] = hashlib.sha256(payload).hexdigest()
        rewrite_manifest(self.directory, manifest)
        decoder = ReferenceDecoder(self.directory)
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            decoder.forward([0])


if __name__ == "__main__":
    unittest.main()
