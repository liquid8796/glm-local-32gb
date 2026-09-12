"""Small metadata fixtures for the pinned checkpoint's default and MTP profile.

MTP shape evidence: pinned shard 00270 has eh_proj[6144,12288], enorm/hnorm
[6144]; shard 00274 has shared_head.norm[6144], a full indexer and MoE.
The test scales the same topology down; no checkpoint payload is created/read.
"""
from copy import deepcopy
import unittest

from glm_local.architecture.mapper import (
    CONFIG_RULE_SOURCE, REVIEWED_MODEL_ID, REVIEWED_REVISION, analyze_catalogue,
)
from glm_local.checkpoint_schema import known_shape
from test_architecture_mapper import fixture_catalogue, fixture_config, header


def reviewed_fixture():
    config = fixture_config()
    config.update(architectures=["GlmMoeDsaForCausalLM"], transformers_version="5.15.0",
                  num_nextn_predict_layers=1)
    del config["layer_types"]
    del config["mlp_bias"]
    records = fixture_catalogue()
    # Every attention projection and the two indexer projections are FP8 in
    # this checkpoint; norm/router/indexer head-weight tensors remain BF16.
    for index, record in enumerate(records[:]):
        name = record["name"]
        if ".self_attn." in name and name.endswith(tuple("." + suffix + ".weight" for suffix in (
                "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "wq_b", "wk"))):
            records[index] = header(name, record["shape"], "F8_E4M3")
            records.append(header(name + "_scale_inv", [(n + 127) // 128 for n in record["shape"]], "F32"))
    # MTP uses a full-indexer sparse layer, plus four additional BF16 tensors.
    records += [{**record, "name": record["name"].replace("model.layers.1.", "model.layers.2.")}
                for record in records[:] if record["name"].startswith("model.layers.1.")]
    records += [{**record, "name": record["name"].replace("model.layers.0.", "model.layers.2.")}
                for record in records[:] if record["name"].startswith("model.layers.0.self_attn.indexer.")]
    records += [header("model.layers.2.eh_proj.weight", (16, 32))]
    records += [header("model.layers.2." + name, (16,))
                for name in ("enorm.weight", "hnorm.weight", "shared_head.norm.weight")]
    return config, records


class ReviewedArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.config, self.records = reviewed_fixture()

    def review(self, **kwargs):
        return analyze_catalogue(kwargs.pop("records", self.records), kwargs.pop("config", self.config),
                                 complete=True, model_id=kwargs.pop("model_id", REVIEWED_MODEL_ID),
                                 revision=kwargs.pop("revision", REVIEWED_REVISION), **kwargs)

    def test_reviewed_defaults_and_complete_mtp_inventory_pass_only_metadata(self):
        original = deepcopy(self.config)
        result = self.review()
        self.assertEqual(result["status"], "PASS", result["findings"])
        self.assertEqual(result["configuration_resolution"]["derived"], {
            "mlp_bias": False, "layer_types": ["deepseek_sparse_attention"] * 2})
        self.assertEqual(result["configuration_resolution"]["source"], CONFIG_RULE_SOURCE)
        self.assertEqual(result["layers"]["count_expected"], 3)
        self.assertEqual(result["layers"]["backbone_count_expected"], 2)
        self.assertEqual(result["mtp"]["declared_layers"], 1)
        self.assertTrue(result["mtp"]["metadata_verified"])
        self.assertFalse(result["mtp"]["execution_verified"])
        for key in ("inference_verified", "payload_values_verified", "real_checkpoint_compatible"):
            self.assertFalse(result[key])
        self.assertEqual(self.config, original)

    def test_unreviewed_identity_cannot_inherit_defaults_or_mtp(self):
        for changes in ({"model_id": "other/model"}, {"revision": "a" * 40},
                        {"model_id": None, "revision": None}):
            result = self.review(**changes)
            self.assertEqual(result["status"], "REVIEW_REQUIRED")
            for code in ("ATTENTION_SCHEDULE_REQUIRED", "UNSUPPORTED_OR_MISSING_CONFIG_FLAG", "UNSUPPORTED_MTP_PROFILE"):
                self.assertIn(code, result["findings"]["by_code"])

    def test_changed_config_source_profile_requires_review(self):
        for key, value in (("architectures", ["UnknownModel"]), ("transformers_version", "9.0.0")):
            config = deepcopy(self.config)
            config[key] = value
            self.assertEqual(self.review(config=config)["status"], "REVIEW_REQUIRED")

    def test_explicit_unsupported_values_are_never_replaced_by_defaults(self):
        for key, value in (("mlp_bias", True), ("mlp_bias", None),
                           ("layer_types", ["full_attention"] * 2), ("layer_types", [])):
            config = deepcopy(self.config)
            config[key] = value
            self.assertEqual(self.review(config=config)["status"], "REVIEW_REQUIRED")

    def test_each_mtp_extra_tensor_is_required_and_bf16(self):
        for suffix in ("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight"):
            name = "model.layers.2." + suffix
            records = [r for r in self.records if r["name"] != name]
            result = self.review(records=records)
            self.assertIn("MISSING_REQUIRED_TENSOR", result["findings"]["by_code"])
            records = deepcopy(self.records)
            index = next(i for i, r in enumerate(records) if r["name"] == name)
            records[index] = header(name, records[index]["shape"], "F32")
            self.assertIn("MTP_DTYPE_MISMATCH", self.review(records=records)["findings"]["by_code"])

    def test_mtp_requires_full_indexer_and_all_experts(self):
        for prefix in ("model.layers.2.self_attn.indexer.", "model.layers.2.mlp.experts.3."):
            records = [r for r in self.records if not r["name"].startswith(prefix)]
            self.assertIn("MISSING_REQUIRED_TENSOR", self.review(records=records)["findings"]["by_code"])

    def test_unknown_mtp_count_or_undeclared_layer_cannot_pass(self):
        for count in (-1, True, 2, "1"):
            config = deepcopy(self.config)
            config["num_nextn_predict_layers"] = count
            self.assertIn("UNSUPPORTED_MTP_PROFILE", self.review(config=config)["findings"]["by_code"])
        config = deepcopy(self.config)
        del config["num_nextn_predict_layers"]
        self.assertIn("UNEXPECTED_TENSOR", self.review(config=config)["findings"]["by_code"])

    def test_fp8_projection_cannot_be_replaced_with_an_unscaled_bf16_tensor(self):
        name = "model.layers.0.self_attn.q_a_proj.weight"
        records = [deepcopy(r) for r in self.records if r["name"] != name + "_scale_inv"]
        index = next(i for i, r in enumerate(records) if r["name"] == name)
        records[index] = header(name, records[index]["shape"], "BF16")
        self.assertIn("REVIEWED_CHECKPOINT_DTYPE_MISMATCH", self.review(records=records)["findings"]["by_code"])

    def test_real_mtp_shapes_are_candidate_formulas_only(self):
        config = {"hidden_size": 6144, "num_hidden_layers": 78, "num_nextn_predict_layers": 1}
        shapes = {"eh_proj.weight": (6144, 12288), "enorm.weight": (6144,),
                  "hnorm.weight": (6144,), "shared_head.norm.weight": (6144,)}
        for name, expected in shapes.items():
            self.assertEqual(known_shape("model.layers.78." + name, config), expected)
            self.assertIsNone(known_shape("model.layers.77." + name, config))
            self.assertIsNone(known_shape("model.layers.79." + name, config))
        del config["num_nextn_predict_layers"]
        self.assertIsNone(known_shape("model.layers.78.eh_proj.weight", config))


if __name__ == "__main__":
    unittest.main()
