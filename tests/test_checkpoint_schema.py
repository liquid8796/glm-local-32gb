from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from glm_local.checkpoint_http import MetadataError, MAX_TENSORS
from glm_local.checkpoint_schema import (compare_snapshot, known_shape, review_tensors,
                                        runtime_index_policy, validate_index, validate_manifest)
from glm_local.safetensor_reader import SafeTensorError, parse_header_bytes
from checkpoint_test_helpers import MODEL, REVISION, NAME, SCALE, checkpoint, encode, make_header


class CheckpointSchemaTests(unittest.TestCase):
    def setUp(self):
        self.data = checkpoint()
        self.sizes = validate_manifest(MODEL, REVISION, self.data["model"])

    def test_index_and_manifest_exact_sets(self):
        groups = validate_index(self.data["index"], self.sizes)
        self.assertEqual(set(groups), set(self.sizes))
        self.assertEqual(sum(map(len, groups.values())), 6)

    def test_wrong_model_or_revision_rejected(self):
        for key, value in (("id", "other/model"), ("sha", "b" * 40)):
            data = deepcopy(self.data["model"]); data[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(MetadataError, "identity/revision"):
                validate_manifest(MODEL, REVISION, data)

    def test_manifest_invalid_sizes_duplicates_case_collisions(self):
        for size in (True, 0, 7, "100", 2 * 1024**4 + 1):
            data = deepcopy(self.data["model"]); data["siblings"][0]["size"] = size
            with self.subTest(size=size), self.assertRaises(MetadataError):
                validate_manifest(MODEL, REVISION, data)
        for duplicate in (self.data["model"]["siblings"][0]["rfilename"],
                          self.data["model"]["siblings"][0]["rfilename"].replace("model", "MODEL")):
            data = deepcopy(self.data["model"])
            data["siblings"].append({"rfilename": duplicate, "size": 100})
            with self.subTest(name=duplicate), self.assertRaises(MetadataError):
                validate_manifest(MODEL, REVISION, data)

    def test_index_unknown_and_unreferenced_shard_rejected(self):
        data = deepcopy(self.data["index"])
        data["weight_map"][NAME] = "missing.safetensors"
        with self.assertRaises(MetadataError):
            validate_index(data, self.sizes)
        with self.assertRaises(MetadataError):
            validate_index(self.data["index"], {**self.sizes, "extra.safetensors": 500})

    def test_index_invalid_names_and_sizes(self):
        for name in ("", "a" * 513, "bad\0name", "\ud800"):
            data = deepcopy(self.data["index"])
            data["weight_map"][name] = next(iter(self.sizes))
            with self.subTest(name=repr(name)), self.assertRaises(MetadataError):
                validate_index(data, self.sizes)
        for size in (True, -1, 2**63, "5"):
            data = deepcopy(self.data["index"]); data["metadata"]["total_size"] = size
            with self.subTest(size=size), self.assertRaises(MetadataError):
                validate_index(data, self.sizes)

    def test_audit_supports_index_larger_than_miniature_without_relaxing_runtime(self):
        index = {"metadata": {"total_size": 0}, "weight_map": {
            f"invented.{n}.weight": "part.safetensors" for n in range(9000)}}
        groups = validate_index(index, {"part.safetensors": 100})
        self.assertEqual(len(groups["part.safetensors"]), 9000)
        policy = runtime_index_policy(len(encode(index)), 9000)
        self.assertFalse(policy["fits_current_reader_index_policy"])
        self.assertEqual(policy["current_reader_max_index_tensors"], 8192)
        self.assertFalse(policy["reader_limits_changed"])
        self.assertGreater(MAX_TENSORS, 9000)

    def test_snapshot_comparison_preserves_declared_types(self):
        args = (MODEL, REVISION, self.sizes)
        self.assertTrue(compare_snapshot(*args, self.data["config"], self.data["expected"])["matched"])
        for value in (True, 1.0):
            config = deepcopy(self.data["config"]); config["num_hidden_layers"] = value
            result = compare_snapshot(*args, config, self.data["expected"])
            self.assertFalse(result["matched"])

    def test_snapshot_mismatch_reports_without_mutation(self):
        original = deepcopy(self.data)
        config = deepcopy(self.data["config"]); config["hidden_size"] = 100
        result = compare_snapshot(MODEL, REVISION, self.sizes, config, self.data["expected"])
        self.assertFalse(result["matched"])
        self.assertEqual(result["mismatches"][0]["field"], "hidden_size")
        self.assertEqual(self.data, original)

    def test_shared_header_parser_needs_no_payload_or_sparse_file(self):
        raw, size, _ = make_header([("huge", "F8_E4M3", [1000000, 1000000])])
        self.assertLess(len(raw), 300)
        tensors, _ = parse_header_bytes(raw[8:], size)
        self.assertEqual(tensors["huge"].nbytes, 10**12)
        self.assertEqual(tensors["huge"].shape, (1000000, 1000000))

    def test_header_rejects_duplicate_keys_unknown_dtype_overlap_and_tail(self):
        raw, size, _ = make_header([("x", "F32", [1]), ("y", "F32", [1])])
        root = json.loads(raw[8:])
        bad_dtype = deepcopy(root); bad_dtype["x"]["dtype"] = "F8_UNKNOWN"
        overlap = deepcopy(root); overlap["y"]["data_offsets"] = [0, 4]
        for value in (encode(bad_dtype), encode(overlap), b'{"x":{},"x":{}}'):
            with self.subTest(value=value), self.assertRaises(SafeTensorError):
                parse_header_bytes(value, 8 + len(value) + 8)
        with self.assertRaises(SafeTensorError):
            parse_header_bytes(raw[8:], size + 1)

    def test_shared_header_empty_scalar_and_zero_dimensions(self):
        raw, size, _ = make_header([("empty", "F16", [0, 3]), ("scalar", "F32", [])])
        tensors, _ = parse_header_bytes(raw[8:], size)
        self.assertEqual(tensors["empty"].nbytes, 0)
        self.assertEqual(tensors["scalar"].nbytes, 4)

    def test_known_shapes_do_not_fill_missing_config_from_miniature(self):
        config = self.data["config"]
        self.assertEqual(known_shape(NAME, config), (257, 259))
        self.assertIsNone(known_shape("model.layers.0.self_attn.q_b_proj.weight", config))
        self.assertIsNone(known_shape("model.layers.78.mlp.gate_proj.weight", config))
        self.assertIsNone(known_shape("some.vendor.extra.weight", config))

    def test_explicit_known_attention_expert_indexer_formulas(self):
        config = {**self.data["config"], "hidden_size": 16, "q_lora_rank": 8, "kv_lora_rank": 4,
                  "qk_rope_head_dim": 4, "qk_nope_head_dim": 4, "v_head_dim": 4,
                  "num_attention_heads": 2, "moe_intermediate_size": 24,
                  "n_routed_experts": 4, "n_shared_experts": 2, "index_n_heads": 2,
                  "index_head_dim": 8}
        cases = {"self_attn.q_a_proj.weight": (8, 16), "self_attn.q_b_proj.weight": (16, 8),
                 "self_attn.kv_a_proj_with_mqa.weight": (8, 16), "self_attn.kv_b_proj.weight": (16, 4),
                 "self_attn.o_proj.weight": (16, 8), "self_attn.kv_a_layernorm.weight": (4,),
                 "mlp.gate.weight": (4, 16), "mlp.gate.e_score_correction_bias": (4,),
                 "mlp.experts.0.up_proj.weight": (24, 16), "mlp.experts.3.down_proj.weight": (16, 24),
                 "mlp.shared_experts.gate_proj.weight": (48, 16),
                 "mlp.experts.gate_up_proj": (4, 48, 16), "mlp.experts.down_proj": (4, 16, 24),
                 "self_attn.indexer.wq_b.weight": (16, 8), "self_attn.indexer.wk.weight": (8, 16),
                 "self_attn.indexer.weights_proj.weight": (2, 16), "self_attn.indexer.k_norm.bias": (8,)}
        for suffix, shape in cases.items():
            with self.subTest(suffix=suffix):
                self.assertEqual(known_shape("model.layers.0." + suffix, config), shape)
        self.assertIsNone(known_shape("model.layers.0.mlp.experts.4.up_proj.weight", config))


class CheckpointFp8ReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = checkpoint()
        self.tensors = {}
        for raw, size in self.data["headers"].values():
            self.tensors.update(parse_header_bytes(raw[8:], size)[0])
        self.mapping = self.data["index"]["weight_map"]

    def review(self, *, complete=True):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalogue.jsonl"
            result = review_tensors(self.tensors, self.mapping, self.data["config"],
                                    complete=complete, catalogue_path=path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            return result, records

    def test_cross_shard_ragged_scale_grid_matches_without_claiming_numeric_validity(self):
        result, records = self.review()
        self.assertTrue(result["fp8_adapter_metadata_verified"])
        self.assertEqual(result["fp8_pairs"]["cross_shard_pairs"], 1)
        self.assertFalse(result["architecture_mapping_verified"])
        self.assertEqual(result["findings"]["count"], 0)
        weight = next(r for r in records if r["name"] == NAME)
        self.assertEqual(weight["scale_shape"], [3, 3])
        self.assertEqual(weight["fp8_profile_check"], "match_metadata_only")
        self.assertIn("aux.stat", result["unreviewed_tensor_examples"])

    def test_missing_scale_is_a_finding_not_a_default_value(self):
        del self.tensors[SCALE]; del self.mapping[SCALE]
        result, _ = self.review()
        self.assertFalse(result["fp8_adapter_metadata_verified"])
        self.assertEqual(result["findings"]["by_code"]["MISSING_SCALE"], 1)

    def test_partial_cross_shard_pair_is_deferred_not_passed_or_marked_missing(self):
        del self.tensors[SCALE]
        result, _ = self.review(complete=False)
        self.assertFalse(result["fp8_adapter_metadata_verified"])
        self.assertEqual(result["fp8_pairs"]["deferred_scale_headers"], 1)
        self.assertNotIn("MISSING_SCALE", result["findings"]["by_code"])

    def test_bf16_scales_are_observed_but_not_claimed_supported_by_f32_adapter(self):
        self.tensors[SCALE] = replace(self.tensors[SCALE], dtype="BF16", itemsize=2, nbytes=18)
        result, records = self.review()
        self.assertFalse(result["fp8_adapter_metadata_verified"])
        self.assertIn("CURRENT_ADAPTER_REQUIRES_F32_SCALE", result["findings"]["by_code"])
        self.assertEqual(next(r for r in records if r["name"] == NAME)["scale_dtype"], "BF16")

    def test_wrong_scale_shape_is_not_reshaped(self):
        self.tensors[SCALE] = replace(self.tensors[SCALE], shape=(1, 9))
        result, _ = self.review()
        self.assertIn("SCALE_GRID_MISMATCH", result["findings"]["by_code"])

    def test_orphan_scale_and_nonfp8_weight_pair(self):
        del self.tensors[NAME]; del self.mapping[NAME]
        result, _ = self.review()
        self.assertIn("ORPHAN_SCALE", result["findings"]["by_code"])
        self.setUp()
        self.tensors[NAME] = replace(self.tensors[NAME], dtype="BF16", itemsize=2)
        result, _ = self.review()
        self.assertIn("SCALE_FOR_NON_FP8_WEIGHT", result["findings"]["by_code"])

    def test_shape_disagreement_with_config_is_reported(self):
        self.data["config"]["hidden_size"] = 258
        result, _ = self.review()
        self.assertIn("CONFIG_TENSOR_SHAPE_MISMATCH", result["findings"]["by_code"])

    def test_unsupported_block_or_format_not_silently_treated_as_128(self):
        for field, value in (("weight_block_size", [64, 128]), ("weight_block_size", [128.0, 128]),
                             ("fmt", "e5m2"), ("scale_fmt", "ue8m0")):
            self.setUp(); self.data["config"]["quantization_config"][field] = value
            result, _ = self.review()
            with self.subTest(field=field, value=value):
                self.assertFalse(result["fp8_adapter_metadata_verified"])
                self.assertIn("UNSUPPORTED_QUANTIZATION_PROFILE", result["findings"]["by_code"])

    def test_packed_fp8_is_not_assumed_compatible_with_2d_blocks(self):
        name = "model.layers.0.mlp.experts.gate_up_proj"
        tensor = self.tensors.pop(NAME); self.tensors[name] = replace(tensor, shape=(1, 257, 259))
        self.mapping[name] = self.mapping.pop(NAME)
        result, _ = self.review()
        self.assertIn("UNREVIEWED_FP8_NAME", result["findings"]["by_code"])
        self.assertIn("FP8_REQUIRES_NONEMPTY_2D", result["findings"]["by_code"])
