"""Header-only fixtures for inventory validation; never open model payloads."""
from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from glm_local.architecture.mapper import _classify, analyze_catalogue


def fixture_config():
    return {
        "model_type": "glm_moe_dsa", "num_hidden_layers": 2, "hidden_size": 16,
        "vocab_size": 32, "num_attention_heads": 2, "q_lora_rank": 8,
        "kv_lora_rank": 4, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
        "v_head_dim": 4, "index_n_heads": 2, "index_head_dim": 8,
        "intermediate_size": 24, "moe_intermediate_size": 24,
        "n_routed_experts": 4, "n_shared_experts": 1,
        "layer_types": ["deepseek_sparse_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "sparse"], "indexer_types": ["full", "shared"],
        "attention_bias": False, "mlp_bias": False, "tie_word_embeddings": False,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
    }


def header(name, shape, dtype="BF16"):
    sizes = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "F16": 2}
    return {"name": name, "shape": list(shape), "dtype": dtype,
            "nbytes": math.prod(shape) * sizes[dtype], "shard": "synthetic-header-only.safetensors"}


def fixture_catalogue():
    # Deliberately explicit expected names/shapes, independent of mapper helpers.
    records = [header("model.embed_tokens.weight", (32, 16)),
               header("lm_head.weight", (32, 16)), header("model.norm.weight", (16,))]
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for suffix, shape in (
            ("input_layernorm.weight", (16,)), ("post_attention_layernorm.weight", (16,)),
            ("self_attn.q_a_proj.weight", (8, 16)), ("self_attn.q_b_proj.weight", (16, 8)),
            ("self_attn.kv_a_proj_with_mqa.weight", (8, 16)), ("self_attn.kv_b_proj.weight", (16, 4)),
            ("self_attn.o_proj.weight", (16, 8)), ("self_attn.q_a_layernorm.weight", (8,)),
            ("self_attn.kv_a_layernorm.weight", (4,)),
        ):
            records.append(header(prefix + suffix, shape))
    for suffix, shape in (("wq_b.weight", (16, 8)), ("wk.weight", (8, 16)),
                          ("weights_proj.weight", (2, 16)), ("k_norm.weight", (8,)), ("k_norm.bias", (8,))):
        records.append(header("model.layers.0.self_attn.indexer." + suffix, shape))
    for prefix in ["model.layers.0.mlp.", "model.layers.1.mlp.shared_experts."] + [
            f"model.layers.1.mlp.experts.{expert}." for expert in range(4)]:
        for direction in ("gate", "up", "down"):
            shape = (16, 24) if direction == "down" else (24, 16)
            name = prefix + direction + "_proj.weight"
            records.append(header(name, shape, "F8_E4M3"))
            records.append(header(name + "_scale_inv", (1, 1), "F32"))
    records.append(header("model.layers.1.mlp.gate.weight", (4, 16)))
    records.append(header("model.layers.1.mlp.gate.e_score_correction_bias", (4,), "F32"))
    return records


class ArchitectureMapperTests(unittest.TestCase):
    def setUp(self):
        self.config = fixture_config()
        self.records = fixture_catalogue()

    def review(self, records=None, config=None, complete=True):
        return analyze_catalogue(self.records if records is None else records,
                                 self.config if config is None else config, complete=complete)

    def assert_review(self, result, code):
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["metadata_mapping_verified"])
        self.assertFalse(result["layers"]["verified"])
        self.assertFalse(result["attention"]["verified"])
        self.assertFalse(result["moe"]["verified"])
        self.assertIn(code, result["findings"]["by_code"])

    def test_complete_explicit_inventory_passes_only_metadata(self):
        result = self.review()
        self.assertEqual(result["status"], "PASS", result["findings"])
        self.assertTrue(result["metadata_mapping_verified"])
        self.assertTrue(result["layers"]["verified"])
        self.assertEqual(result["layers"]["count_expected"], 2)
        self.assertEqual(result["layers"]["ids"], [0, 1])
        self.assertEqual(result["shape_checks"]["required_tensor_count"], 46)
        self.assertEqual(result["shape_checks"]["missing_required_count"], 0)
        self.assertEqual(result["fp8"]["valid_metadata_pairs"], 18)
        self.assertTrue(result["metadata_only"])
        for field in ("real_checkpoint_compatible", "inference_verified", "payload_values_verified"):
            self.assertIs(result[field], False)
        json.dumps(result, allow_nan=False)

    def test_partial_inventory_can_never_pass(self):
        self.assert_review(self.review(complete=False), "INCOMPLETE_CATALOGUE")
        self.assert_review(self.review(complete=1), "INCOMPLETE_CATALOGUE")

    def test_config_missing_or_snapshot_incomplete_requires_review(self):
        self.assert_review(analyze_catalogue(self.records, complete=True), "CONFIG_REQUIRED")
        self.assert_review(self.review(config={"model_type": "glm_moe_dsa", "num_hidden_layers": 2}),
                           "CONFIG_DIMENSION_REQUIRED")

    def test_layer_name_detection_is_separate_from_validation(self):
        result = self.review(records=[header("model.layers.0.fake_attention.weight", (2, 2))])
        self.assertEqual(result["layers"]["count_detected"], 1)
        self.assert_review(result, "MISSING_REQUIRED_TENSOR")
        self.assertEqual(result["groups"]["attention_projection"]["count"], 0)

    def test_missing_layer_and_missing_expert_block_pass(self):
        for predicate in (lambda n: not n.startswith("model.layers.1."),
                          lambda n: not n.startswith("model.layers.1.mlp.experts.2.")):
            result = self.review(records=[r for r in self.records if predicate(r["name"])])
            self.assert_review(result, "MISSING_REQUIRED_TENSOR")
            self.assertGreater(result["shape_checks"]["missing_required_count"], 0)

    def test_every_required_role_missing_is_detected(self):
        names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
                 "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_b_proj.weight",
                 "model.layers.0.self_attn.kv_a_layernorm.weight",
                 "model.layers.0.self_attn.indexer.k_norm.bias", "model.layers.0.mlp.up_proj.weight",
                 "model.layers.1.mlp.gate.weight", "model.layers.1.mlp.gate.e_score_correction_bias",
                 "model.layers.1.mlp.shared_experts.down_proj.weight"]
        for name in names:
            with self.subTest(name=name):
                result = self.review(records=[r for r in self.records if r["name"] != name])
                self.assert_review(result, "MISSING_REQUIRED_TENSOR")

    def test_wrong_shape_fails_even_with_consistent_nbytes(self):
        self.records[0] = header("model.embed_tokens.weight", (33, 16))
        result = self.review()
        self.assert_review(result, "CONFIG_TENSOR_SHAPE_MISMATCH")
        self.assertEqual(result["shape_checks"]["invalid_count"], 1)

    def test_invalid_dtypes_and_byte_counts_cannot_pass(self):
        for dtype in ("I32", "NOT_A_DTYPE", {}, True):
            records = deepcopy(self.records)
            records[0]["dtype"] = dtype
            with self.subTest(dtype=dtype):
                self.assert_review(self.review(records=records), "INVALID_DTYPE")
        for count in (True, -1, "1024", 1, 2**63):
            records = deepcopy(self.records)
            records[0]["nbytes"] = count
            with self.subTest(nbytes=count):
                self.assert_review(self.review(records=records), "INVALID_NBYTES")

    def test_shapes_require_bounded_integer_dimensions_and_rank(self):
        for shape in ([], [True, 16], [0, 16], [-1, 16], [2**31, 16], [2] * 9, "32,16", None):
            records = deepcopy(self.records)
            records[0]["shape"] = shape
            with self.subTest(shape=shape):
                result = self.review(records=records)
                self.assert_review(result, "INVALID_SHAPE")
                json.dumps(result, allow_nan=False)

    def test_duplicate_names_are_rejected(self):
        self.assert_review(self.review(records=self.records + [deepcopy(self.records[0])]), "DUPLICATE_TENSOR")

    def test_router_is_not_dense_expert_or_scale(self):
        result = self.review()
        self.assertEqual(result["groups"]["moe_router"]["count"], 1)
        self.assertEqual(result["groups"]["router_bias"]["count"], 1)
        self.assertEqual(result["groups"]["dense_ffn"]["count"], 3)
        self.assertEqual(result["groups"]["shared_expert"]["count"], 3)
        self.assertEqual(result["groups"]["expert_projection"]["count"], 12)
        self.assertEqual(result["groups"]["fp8_scale"]["count"], 18)
        for name in ("model.layers.1.mlp.experts.0.gate_proj.weight",
                     "model.layers.0.mlp.gate_proj.weight",
                     "model.layers.1.mlp.gate.weight_scale_inv"):
            self.assertNotIn("moe_router", _classify(name))

    def test_anchored_names_do_not_accept_substrings_or_aliases(self):
        for name in ("other.model.layers.0.self_attn.q_a_proj.weight",
                     "model.layers.00.self_attn.q_a_proj.weight",
                     "model.layers.0.self_attn.q_a_proj.weight.extra",
                     "some.embedding.weight", "something.router.weight"):
            with self.subTest(name=name):
                self.assertEqual(_classify(name), ["unknown"])
                self.assert_review(self.review(records=self.records + [header(name, (2, 2))]), "UNKNOWN_TENSOR_NAME")

    def test_unexpected_layer_expert_or_shared_indexer_is_not_verified(self):
        for name in ("model.layers.2.input_layernorm.weight",
                     "model.layers.1.mlp.experts.4.gate_proj.weight",
                     "model.layers.1.self_attn.indexer.k_norm.weight"):
            result = self.review(records=self.records + [header(name, (16,))])
            self.assert_review(result, "UNEXPECTED_TENSOR")

    def test_all_attention_indexer_and_mlp_schedules_are_explicit(self):
        for field in ("layer_types", "indexer_types", "mlp_layer_types"):
            config = deepcopy(self.config)
            del config[field]
            code = {"layer_types": "ATTENTION_SCHEDULE_REQUIRED", "indexer_types": "INDEXER_SCHEDULE_REQUIRED",
                    "mlp_layer_types": "MLP_SCHEDULE_REQUIRED"}[field]
            self.assert_review(self.review(config=config), code)

    def test_dense_prefix_schedule_is_supported_and_conflicts_are_rejected(self):
        config = deepcopy(self.config)
        del config["mlp_layer_types"]
        config["first_k_dense_replace"] = 1
        self.assertEqual(self.review(config=config)["status"], "PASS")
        config["mlp_layer_types"] = ["sparse", "dense"]
        self.assert_review(self.review(config=config), "CONFLICTING_MLP_SCHEDULES")

    def test_invalid_profile_flags_and_model_types_require_review(self):
        for key in ("attention_bias", "mlp_bias", "tie_word_embeddings"):
            config = deepcopy(self.config)
            del config[key]
            self.assert_review(self.review(config=config), "UNSUPPORTED_OR_MISSING_CONFIG_FLAG")
        config = deepcopy(self.config)
        config["model_type"] = "unrelated"
        self.assert_review(self.review(config=config), "UNSUPPORTED_MODEL_TYPE")
        config = deepcopy(self.config)
        config["indexer_types"] = ["shared", "full"]
        self.assert_review(self.review(config=config), "INDEXER_SCHEDULE_REQUIRED")

    def test_invalid_quantization_profile_is_not_assumed(self):
        for value in (None, {"quant_method": "int8"}, {"quant_method": "fp8", "fmt": "e4m3",
                     "weight_block_size": [128.0, 128]}):
            config = deepcopy(self.config)
            config["quantization_config"] = value
            self.assert_review(self.review(config=config), "UNSUPPORTED_QUANTIZATION_PROFILE")

    def test_missing_scale_or_orphan_scale_requires_review(self):
        scale = next(r["name"] for r in self.records if r["name"].endswith("_scale_inv"))
        self.assert_review(self.review(records=[r for r in self.records if r["name"] != scale]), "MISSING_FP8_SCALE")
        self.assert_review(self.review(records=[r for r in self.records if r["name"] != scale[:-len("_scale_inv")]]),
                           "ORPHAN_FP8_SCALE")

    def test_scale_dtype_grid_and_base_dtype_are_checked(self):
        index = next(i for i, r in enumerate(self.records) if r["name"].endswith("_scale_inv"))
        original = self.records[index]
        for shape, dtype, code in (((1, 2), "F32", "FP8_SCALE_GRID_MISMATCH"),
                                   ((1, 1), "BF16", "INVALID_FP8_SCALE_DTYPE")):
            records = deepcopy(self.records)
            records[index] = header(original["name"], shape, dtype)
            self.assert_review(self.review(records=records), code)
        records = deepcopy(self.records)
        base = original["name"][:-len("_scale_inv")]
        index = next(i for i, r in enumerate(records) if r["name"] == base)
        records[index] = header(base, records[index]["shape"])
        self.assert_review(self.review(records=records), "SCALE_FOR_NON_FP8_WEIGHT")

    def test_noncanonical_scale_name_does_not_satisfy_weight_pair(self):
        records = deepcopy(self.records)
        scale = next(r for r in records if r["name"].endswith("_scale_inv"))
        scale["name"] = scale["name"].replace(".weight_scale_inv", ".scale")
        result = self.review(records=records)
        self.assert_review(result, "MISSING_FP8_SCALE")
        self.assertIn("UNKNOWN_TENSOR_NAME", result["findings"]["by_code"])

    def test_scale_grid_uses_ceil_for_partial_blocks(self):
        config = deepcopy(self.config)
        config["intermediate_size"] = 129
        records = deepcopy(self.records)
        for index, record in enumerate(records):
            name = record["name"]
            if name.startswith("model.layers.0.mlp."):
                down = ".down_proj." in name
                is_scale = name.endswith("_scale_inv")
                shape = ((1, 2) if down else (2, 1)) if is_scale else ((16, 129) if down else (129, 16))
                records[index] = header(name, shape, "F32" if is_scale else "F8_E4M3")
        self.assertEqual(self.review(records=records, config=config)["status"], "PASS")

    def test_packed_expert_banks_remain_explicitly_unsupported(self):
        records = [r for r in self.records if not r["name"].startswith("model.layers.1.mlp.experts.")]
        records += [header("model.layers.1.mlp.experts.gate_up_proj", (4, 48, 16)),
                    header("model.layers.1.mlp.experts.down_proj", (4, 16, 24))]
        result = self.review(records=records)
        self.assert_review(result, "UNSUPPORTED_PACKED_EXPERT_LAYOUT")
        self.assertEqual(result["shape_checks"]["missing_required_count"], 12)

    def test_findings_and_group_examples_are_bounded(self):
        records = self.records + [header(f"unknown.{n}.weight", (1,)) for n in range(100)]
        result = self.review(records=records)
        self.assertEqual(result["findings"]["by_code"]["UNKNOWN_TENSOR_NAME"], 100)
        self.assertLessEqual(len(result["groups"]["unknown"]["samples"]), 8)
        self.assertTrue(all(len(items) <= 8 for items in result["findings"]["examples"].values()))

    def test_inventory_limit_stops_iterator_without_draining(self):
        seen = []
        def rows():
            for number in range(100):
                seen.append(number)
                yield header(f"unknown.{number}", (1,))
        with patch("glm_local.architecture.mapper.MAX_INVENTORY", 5):
            result = analyze_catalogue(rows(), complete=True)
        self.assertEqual(len(seen), 6)
        self.assert_review(result, "INVENTORY_LIMIT")
        self.assertFalse(result["complete_catalogue"])

    def test_large_requirement_plan_stops_before_expert_expansion(self):
        config = deepcopy(self.config)
        config.update(num_hidden_layers=256, n_routed_experts=1024,
                      layer_types=["deepseek_sparse_attention"] * 256,
                      mlp_layer_types=["sparse"] * 256, indexer_types=["full"] * 256)
        result = self.review(config=config)
        self.assert_review(result, "REQUIRED_INVENTORY_LIMIT")
        self.assertEqual(result["shape_checks"]["required_tensor_count"], 0)

    def test_config_layer_expert_and_dimension_bounds(self):
        for key, value in (("num_hidden_layers", 257), ("n_routed_experts", 1025),
                           ("hidden_size", True), ("q_lora_rank", 0), ("vocab_size", 2**31)):
            config = deepcopy(self.config)
            config[key] = value
            self.assert_review(self.review(config=config), "CONFIG_DIMENSION_REQUIRED")

    def test_invalid_catalogue_and_record_inputs_remain_json_safe(self):
        for records in (None, "not-an-inventory", {}, [None], [{"name": "x" * 513}],
                        [{"name": "bad\u0000name"}], [{"name": "\ud800"}]):
            with self.subTest(records=repr(records)):
                result = self.review(records=records) if records is not None else analyze_catalogue(None)
                self.assertEqual(result["status"], "REVIEW_REQUIRED")
                json.dumps(result, allow_nan=False)

    def test_input_records_and_config_are_not_mutated(self):
        records, config = deepcopy(self.records), deepcopy(self.config)
        self.review()
        self.assertEqual(self.records, records)
        self.assertEqual(self.config, config)


if __name__ == "__main__":
    unittest.main()
