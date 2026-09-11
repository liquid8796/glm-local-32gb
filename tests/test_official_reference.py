"""Opt-in checks of the installed official graph with tiny invented weights."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if os.environ.get("GLM_TEST_OFFICIAL") != "1":
    raise unittest.SkipTest("Official graph tests require GLM_TEST_OFFICIAL=1 and reference extras")

import numpy as np
import torch

from glm_local.mini_reference import ReferenceDecoder
from glm_local.mini_spec import SOURCE_REVISION
from glm_local.mini_weights import write_mini_bundle
from glm_local.official_reference import HIDDEN_STATE_NAMES, OfficialMiniReference, _mapped_state


class OfficialMiniReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        write_mini_bundle(self.directory, seed=7)

    def test_source_metadata_and_every_loaded_value_are_exact(self):
        fixture = ReferenceDecoder(self.directory)
        expected = _mapped_state(fixture, torch)
        with OfficialMiniReference(self.directory) as reference:
            metadata = reference.metadata
            self.assertEqual(metadata["requested_source_revision"], SOURCE_REVISION)
            self.assertEqual(metadata["matrix_count"], 34)
            self.assertEqual(metadata["vector_count"], 12)
            self.assertEqual(metadata["state_dict_count"], 36)
            self.assertTrue(metadata["all_state_values_copied_exactly"])
            self.assertTrue(metadata["synthetic_only"])
            self.assertFalse(metadata["real_checkpoint_compatible"])
            for source in metadata["source_files"].values():
                self.assertEqual(source["sha256"],
                                 hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest())
            actual = reference._model.state_dict()
            self.assertEqual(set(actual), set(expected))
            for name, value in actual.items():
                with self.subTest(name=name):
                    self.assertTrue(torch.equal(value, expected[name]))
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertEqual(value.device.type, "cpu")
            packed = actual["model.layers.1.mlp.experts.gate_up_proj"]
            for expert in range(4):
                np.testing.assert_array_equal(packed[expert, :24].numpy(),
                                              fixture._matrices[f"layer.1.expert.{expert}.gate"])
                np.testing.assert_array_equal(packed[expert, 24:].numpy(),
                                              fixture._matrices[f"layer.1.expert.{expert}.up"])
            metadata["source_files"].clear()
            self.assertEqual(len(reference.metadata["source_files"]), 2)

    def test_short_incremental_sequence_matches_independent_oracle(self):
        tokens = [1, 4, 2, 9, 3, 7, 0, 11]
        expected = ReferenceDecoder(self.directory).forward(tokens)
        with OfficialMiniReference(self.directory) as reference:
            for position, token in enumerate(tokens):
                actual = reference.step(token)
                self.assertEqual(set(actual["hidden_states"]), set(HIDDEN_STATE_NAMES))
                for values in actual["hidden_states"].values():
                    self.assertEqual(len(values), 16)
                    self.assertTrue(np.isfinite(values).all())
                np.testing.assert_allclose(actual["logits"], expected["logits"][position],
                                           rtol=0, atol=2e-6)
                trace = expected["traces"][position]
                self.assertEqual(set(actual["selected_indices"]), set(trace["selected_indices"]))
                self.assertEqual(set(actual["routed_experts"]), set(trace["routed_experts"]))
                expected_weights = dict(zip(trace["routed_experts"], trace["router_weights"]))
                for expert, weight in zip(actual["routed_experts"], actual["router_weights"]):
                    self.assertAlmostEqual(weight, expected_weights[expert], places=5)
            self.assertEqual(reference.position, len(tokens))

    def test_reset_clears_cache_and_repeats_identically(self):
        tokens = [2, 5, 8, 11, 14]
        with OfficialMiniReference(self.directory) as reference:
            first = [reference.step(token) for token in tokens]
            reference.reset()
            self.assertEqual(reference.position, 0)
            self.assertIsNone(reference._cache)
            second = [reference.step(token) for token in tokens]
            self.assertEqual(first, second)

    def test_invalid_token_and_context_boundary_do_not_mutate_cache(self):
        with OfficialMiniReference(self.directory) as reference:
            reference.step(0)
            for token in (True, 1.0, -1, 32, None, "1", [1]):
                with self.subTest(token=token):
                    with self.assertRaises(ValueError):
                        reference.step(token)
                    self.assertEqual(reference.position, 1)
            reference._position = 128
            with self.assertRaisesRegex(ValueError, "128"):
                reference.step(0)

    def test_hooks_removed_and_calls_rejected_after_close(self):
        reference = OfficialMiniReference(self.directory)
        model = reference._model
        self.assertGreater(sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                               for module in model.modules()), 0)
        reference.close()
        self.assertEqual(sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                             for module in model.modules()), 0)
        for operation in (lambda: reference.step(0), reference.reset, reference.__enter__):
            with self.assertRaisesRegex(ValueError, "closed"):
                operation()
        reference.close()

    def test_official_forward_error_invalidates_partial_cache(self):
        with OfficialMiniReference(self.directory) as reference:
            first = reference.step(3)
            with patch.object(reference._model, "forward", side_effect=RuntimeError("fixture error")):
                with self.assertRaisesRegex(RuntimeError, "fixture error"):
                    reference.step(4)
            self.assertEqual(reference.position, 0)
            self.assertIsNone(reference._cache)
            self.assertEqual(first, reference.step(3))

    def test_no_pretrained_entrypoint_used(self):
        from transformers.modeling_utils import PreTrainedModel
        with patch.object(PreTrainedModel, "from_pretrained", side_effect=AssertionError("network path")):
            with OfficialMiniReference(self.directory) as reference:
                self.assertEqual(len(reference.step(1)["logits"]), 32)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
