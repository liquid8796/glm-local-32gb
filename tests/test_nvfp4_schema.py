"""Header-only NVFP4 profile tests, independent of native decoder kernels."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import math
import struct
from pathlib import Path
import tempfile
import unittest

from glm_local.architecture.mapper import (NVFP4_MODEL_ID, NVFP4_REVISION, _classify, _profile, analyze_catalogue)
from glm_local.checkpoint_schema import (Findings, known_shape, stored_shape, quantization_format,
    nvfp4_quantized_weight, nvfp4_weight_name, nvfp4_ancillary_names, validate_nvfp4_config, review_tensors)
from glm_local.safetensor_reader import TensorInfo
from glm_local.architecture.report import run_architecture
from glm_local.checkpoint_check import execute_metadata, publish_report
from glm_local.checkpoint_snapshot import write_json
from checkpoint_test_helpers import MODEL, REVISION, MemorySource, make_header, settings
from test_architecture_mapper import fixture_config, fixture_catalogue


def nvfp4_config():
    config = fixture_config()
    config["moe_intermediate_size"] = 32
    config["quantization_config"] = {
        "quant_method": "modelopt", "quant_algo": "NVFP4",
        "producer": {"name": "modelopt", "version": "0.45.0"},
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "config_groups": {"group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"]}},
        "ignore": ["lm_head", "model.embed_tokens", "model.layers.0*",
                   "model.layers.1.self_attn*", "model.layers.1.mlp.shared_experts*"]}
    return config


def nv_header(name, shape, dtype):
    return {"name": name, "shape": list(shape), "dtype": dtype,
            "nbytes": math.prod(shape) * {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}[dtype],
            "shard": "nvfp4-synthetic.safetensors"}


def nvfp4_catalogue():
    records = []
    for original in fixture_catalogue():
        name, shape = original["name"], original["shape"]
        if name.endswith("_scale_inv"):
            continue
        if "model.layers.1.mlp." in name and "_proj.weight" in name:
            shape = (16, 32) if ".down_proj." in name else (32, 16)
        if ".mlp.experts." in name:
            rows, cols = shape
            prefix = name[:-len(".weight")]
            records.extend([nv_header(name, (rows, cols // 2), "U8"),
                            nv_header(prefix + ".weight_scale", (rows, cols // 16), "F8_E4M3"),
                            nv_header(prefix + ".weight_scale_2", (), "F32"),
                            nv_header(prefix + ".input_scale", (), "F32")])
        else:
            records.append(nv_header(name, shape, "F32" if name.endswith("e_score_correction_bias") else "BF16"))
    return records


def prepare_nvfp4_fixture(root, *, config_overrides=None, payload_factory=None, model_id=MODEL, revision=REVISION):
    """Complete small three-shard fixture for runtime/projection integration tests.

    Canonical two-layer topology is independent of the mapper inventory builder;
    configurable dimensions feed the separately tested logical shape formulas.
    Optional one-layer MTP requires the reviewed identity in the test request.
    """
    root = Path(root)
    config = nvfp4_config()
    config.update(deepcopy(config_overrides or {}))
    if config["num_hidden_layers"] != 2:
        raise ValueError("Shared NV fixture has exactly two backbone layers")
    names = [r["name"] for r in fixture_catalogue() if not r["name"].endswith("_scale_inv")]
    if config.get("num_nextn_predict_layers") == 1:
        names += [name.replace("model.layers.1.", "model.layers.2.") for name in names[:]
                  if name.startswith("model.layers.1.")]
        names += [name.replace("model.layers.0.", "model.layers.2.") for name in names[:]
                  if name.startswith("model.layers.0.self_attn.indexer.")]
        names += ["model.layers.2." + suffix for suffix in ("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight")]
    records = []
    for name in names:
        shape = known_shape(name, config)
        if shape is None:
            raise AssertionError(name)
        if nvfp4_quantized_weight(name, config):
            rows, cols = shape
            stem = name[:-len(".weight")]
            records.extend([nv_header(name, (rows, cols // 2), "U8"),
                            nv_header(stem + ".weight_scale", (rows, cols // 16), "F8_E4M3"),
                            nv_header(stem + ".weight_scale_2", (), "F32"),
                            nv_header(stem + ".input_scale", (), "F32")])
        else:
            records.append(nv_header(name, shape, "F32" if name.endswith("e_score_correction_bias") else "BF16"))
    filenames = [f"model-{number:05d}-of-00003.safetensors" for number in range(1, 4)]
    groups = {name: [] for name in filenames}
    for index, record in enumerate(records):
        groups[filenames[index % 3]].append(record)
    headers, mapping, siblings, payloads, total = {}, {}, [], {}, 0
    for filename, group in groups.items():
        raw, size, payload_size = make_header([(r["name"], r["dtype"], r["shape"]) for r in group])
        headers[filename] = raw, size
        siblings.append({"rfilename": filename, "size": size})
        mapping.update({r["name"]: filename for r in group})
        total += payload_size
        data = bytearray()
        for record in group:
            name, dtype, shape = record["name"], record["dtype"], tuple(record["shape"])
            if payload_factory is None:
                value = 0.125 if name.endswith(".weight_scale_2") else 2.0 if name.endswith(".input_scale") else 0.03125
                unit = (b"\x22" if dtype == "U8" else b"\x38" if dtype == "F8_E4M3" else
                        struct.pack("<f", value) if dtype == "F32" else
                        b"\x80\x3f" if len(shape) == 1 else b"\x00\x3d")
                encoded = unit * math.prod(shape)
            else:
                encoded = payload_factory(name, dtype, shape)
            if not isinstance(encoded, bytes) or len(encoded) != record["nbytes"]:
                raise AssertionError("NV fixture payload must match declared dtype/shape exactly")
            data.extend(encoded)
        payloads[filename] = bytes(data)
    expected = {"model_id": model_id, "revision": revision,
                "weights": [{"name": s["rfilename"], "bytes": s["size"]} for s in siblings],
                "architecture": {**{k: v for k, v in config.items() if k != "quantization_config"},
                                 "quantization": deepcopy(config["quantization_config"])}}
    run_settings = {**settings(), "model_id": model_id, "revision": revision}
    data = {"model": {"id": model_id, "sha": revision, "siblings": siblings}, "config": config,
            "index": {"metadata": {"total_size": total}, "weight_map": mapping}, "headers": headers,
            "expected": expected, "settings": run_settings}
    (root / "docs").mkdir(exist_ok=True)
    write_json(root / "docs/model-metadata.json", expected)
    directory = root / "reports/metadata/run"
    directory.mkdir(parents=True)
    with redirect_stdout(io.StringIO()):
        report, code = execute_metadata(root, run_settings, {"max_shards": 512, "budget_mib": 64, "offline": None},
                                        directory, source=MemorySource(data))
        publish_report(root, directory, report, code)
    if code:
        raise AssertionError(report)
    model_directory = root / run_settings["model_directory"]
    model_directory.mkdir(parents=True)
    for filename, value in (("config.json", config), ("model.safetensors.index.json", data["index"])):
        (model_directory / filename).write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    for filename in filenames:
        (model_directory / filename).write_bytes(headers[filename][0] + payloads[filename])
    return model_directory, data


class Nvfp4SchemaTests(unittest.TestCase):
    def setUp(self):
        self.config, self.records = nvfp4_config(), nvfp4_catalogue()

    def review(self, config=None, records=None, **kwargs):
        return analyze_catalogue(self.records if records is None else records,
                                 self.config if config is None else config, complete=True, **kwargs)

    def check_finding(self, result, code):
        self.assertEqual(result["status"], "REVIEW_REQUIRED", result)
        self.assertIn(code, result["findings"]["by_code"], result["findings"])
        self.assertFalse(result["architecture_mapping_verified"])

    def test_complete_tiny_nvfp4_inventory_passes_only_structural_metadata(self):
        result = self.review()
        self.assertEqual(result["status"], "PASS", result["findings"])
        self.assertEqual(result["quant_format"], "nvfp4")
        self.assertEqual(result["nvfp4"]["weight_tensor_count"], 12)
        self.assertEqual(result["nvfp4"]["valid_metadata_quadruples"], 12)
        self.assertTrue(result["nvfp4"]["metadata_verified"])
        self.assertFalse(result["fp8"]["metadata_verified"])
        self.assertEqual(result["fp8"]["weight_tensor_count"], 0)
        for key in ("payload_values_verified", "real_checkpoint_compatible", "inference_verified"):
            self.assertIs(result[key], False)
        self.assertFalse(result["nvfp4"]["activation_quantization_verified"])

    def test_graph_profile_exposes_quant_format_without_changing_logical_dimensions(self):
        findings = Findings()
        profile = _profile(self.config, findings)
        self.assertEqual(profile["quant_format"], "nvfp4")
        self.assertEqual(profile["mlps"], ["dense", "moe"])
        name = "model.layers.1.mlp.experts.0.up_proj.weight"
        self.assertEqual(known_shape(name, self.config), (32, 16))
        self.assertEqual(stored_shape(name, self.config), (32, 8))
        self.assertEqual(stored_shape(name[:-7] + ".weight_scale", self.config), (32, 1))
        self.assertEqual(stored_shape(name[:-7] + ".weight_scale_2", self.config), ())
        self.assertEqual(stored_shape(name[:-7] + ".input_scale", self.config), ())

    def test_scale_and_activation_names_are_anchored_and_distinct(self):
        stem = "model.layers.1.mlp.experts.0.up_proj"
        expected = {".weight_scale": "nvfp4_block_scale", ".weight_scale_2": "nvfp4_global_scale",
                    ".input_scale": "nvfp4_input_scale"}
        for suffix, role in expected.items():
            self.assertEqual(_classify(stem + suffix), [role])
            self.assertEqual(nvfp4_weight_name(stem + suffix), stem + ".weight")
        self.assertEqual(nvfp4_ancillary_names(stem + ".weight"), tuple(stem + suffix for suffix in expected))
        for name in (stem + ".input_scale_2", "other." + stem + ".weight_scale",
                     stem.replace(".1.", ".01.") + ".weight_scale", "model.layers.1.mlp.gate.weight_scale"):
            self.assertEqual(_classify(name), ["unknown"])

    def test_each_ancillary_is_mandatory(self):
        for suffix in (".weight_scale", ".weight_scale_2", ".input_scale"):
            missing = "model.layers.1.mlp.experts.0.gate_proj" + suffix
            self.check_finding(self.review(records=[r for r in self.records if r["name"] != missing]),
                               "MISSING_REQUIRED_TENSOR")

    def test_block_scale_global_scale_and_input_scale_dtype_shape_rules(self):
        cases = [(".weight", (32, 16), "U8", "CONFIG_TENSOR_SHAPE_MISMATCH"),
                 (".weight_scale", (1, 1), "F8_E4M3", "CONFIG_TENSOR_SHAPE_MISMATCH"),
                 (".weight_scale", (32, 1), "F32", "REVIEWED_CHECKPOINT_DTYPE_MISMATCH"),
                 (".weight_scale_2", (1,), "F32", "CONFIG_TENSOR_SHAPE_MISMATCH"),
                 (".input_scale", (), "BF16", "REVIEWED_CHECKPOINT_DTYPE_MISMATCH")]
        for suffix, shape, dtype, code in cases:
            with self.subTest(suffix=suffix, dtype=dtype):
                name = "model.layers.1.mlp.experts.0.gate_proj" + suffix
                records = [nv_header(name, shape, dtype) if r["name"] == name else r for r in self.records]
                self.check_finding(self.review(records=records), code)

    def test_scalar_shape_is_only_allowed_for_known_nvfp4_global_and_input_scales(self):
        records = [nv_header("model.norm.weight", (), "F32") if r["name"] == "model.norm.weight" else r for r in self.records]
        self.check_finding(self.review(records=records), "INVALID_SHAPE")

    def test_orphan_extra_and_fp8_scale_formats_are_rejected(self):
        weight = "model.layers.1.mlp.experts.0.gate_proj.weight"
        self.check_finding(self.review(records=[r for r in self.records if r["name"] != weight]), "ORPHAN_NVFP4_ANCILLARY")
        self.check_finding(self.review(records=self.records + [nv_header(weight + "_scale_inv", (1, 1), "F32")]),
                           "UNEXPECTED_QUANTIZATION_TENSOR")
        self.check_finding(self.review(records=self.records + [nv_header("model.layers.1.mlp.shared_experts.up_proj.input_scale", (), "F32")]),
                           "UNKNOWN_TENSOR_NAME")

    def test_only_routed_backbone_experts_are_packed(self):
        self.assertTrue(nvfp4_quantized_weight("model.layers.1.mlp.experts.0.up_proj.weight", self.config))
        for name in ("model.layers.0.mlp.gate_proj.weight", "model.layers.1.mlp.shared_experts.up_proj.weight",
                     "model.layers.2.mlp.experts.0.up_proj.weight", "model.layers.1.mlp.experts.4.up_proj.weight"):
            self.assertFalse(nvfp4_quantized_weight(name, self.config))
        records = [nv_header(r["name"], r["shape"], "F8_E4M3") if r["name"] == "model.layers.0.self_attn.q_a_proj.weight"
                   else r for r in self.records]
        self.check_finding(self.review(records=records), "REVIEWED_CHECKPOINT_DTYPE_MISMATCH")

    def test_unknown_producer_group_or_activation_scheme_cannot_pass(self):
        modifications = [lambda q: q["producer"].update(version="0.46.0"),
                         lambda q: q["config_groups"]["group_0"].update(targets=["Conv2d"]),
                         lambda q: q["config_groups"]["group_0"]["weights"].update(group_size=32),
                         lambda q: q["config_groups"]["group_0"]["input_activations"].update(dynamic=True),
                         lambda q: q["config_groups"]["group_0"]["weights"].update(num_bits=4.0),
                         lambda q: q["kv_cache_scheme"].update(num_bits=4)]
        for change in modifications:
            config = deepcopy(self.config)
            change(config["quantization_config"])
            self.check_finding(self.review(config=config), "UNSUPPORTED_NVFP4_CONFIG")
        config = deepcopy(self.config)
        config["quantization_config"]["unreviewed_option"] = True
        self.check_finding(self.review(config=config), "UNSUPPORTED_NVFP4_CONFIG_FIELDS")

    def test_ignore_policy_cannot_silently_change_packing_coverage(self):
        for change in (lambda values: values.pop(), lambda values: values.append("model.layers.1.mlp.experts.0*"),
                       lambda values: values.append(values[0]), lambda values: values.__setitem__(2, "model.layers.*")):
            config = deepcopy(self.config)
            change(config["quantization_config"]["ignore"])
            self.check_finding(self.review(config=config), "UNSUPPORTED_NVFP4_IGNORE_POLICY")

    def test_logical_columns_must_be_divisible_by_16(self):
        for key, value in (("hidden_size", 18), ("moe_intermediate_size", 24), ("hidden_size", True)):
            config = deepcopy(self.config)
            config[key] = value
            self.check_finding(self.review(config=config), "NVFP4_LOGICAL_COLUMNS_ALIGNMENT")

    def test_large_nvfp4_required_inventory_is_bounded_before_expansion(self):
        config = deepcopy(self.config)
        config.update(num_hidden_layers=256, n_routed_experts=1024,
                      layer_types=["deepseek_sparse_attention"] * 256, mlp_layer_types=["sparse"] * 256,
                      indexer_types=["full"] * 256)
        config["quantization_config"]["ignore"] = ["lm_head", "model.embed_tokens"] + [
            f"model.layers.{layer}.{suffix}" for layer in range(256) for suffix in ("self_attn*", "mlp.shared_experts*")]
        result = self.review(config=config)
        self.check_finding(result, "REQUIRED_INVENTORY_LIMIT")
        self.assertEqual(result["shape_checks"]["required_tensor_count"], 0)

    def test_actual_nvfp4_identity_allows_only_reviewed_glm_defaults_and_mtp(self):
        config = deepcopy(self.config)
        config.update(architectures=["GlmMoeDsaForCausalLM"], transformers_version="5.15.0", num_nextn_predict_layers=1)
        del config["layer_types"]
        del config["mlp_bias"]
        config["quantization_config"]["ignore"].append("model.layers.2*")
        findings = Findings()
        profile = _profile(config, findings, model_id=NVFP4_MODEL_ID, revision=NVFP4_REVISION)
        self.assertIsNotNone(profile, findings.report())
        self.assertEqual(profile["mtp_layers"], 1)
        self.assertIn("model.layers.2.eh_proj.weight", profile["required"])
        self.assertNotIn("model.layers.2.mlp.experts.0.gate_proj.weight_scale", profile["required"])
        self.assertFalse(nvfp4_quantized_weight("model.layers.2.mlp.experts.0.gate_proj.weight", config))
        findings = Findings()
        self.assertIsNone(_profile(config, findings, model_id=NVFP4_MODEL_ID, revision="b" * 40))

    def test_metadata_review_writes_stored_and_logical_shapes_without_fp8_confusion(self):
        tensors, mapping, offset = {}, {}, 0
        for record in self.records:
            name, dtype, shape, size = record["name"], record["dtype"], tuple(record["shape"]), record["nbytes"]
            tensors[name] = TensorInfo(dtype, shape, (offset, offset + size), size,
                                       {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}[dtype])
            offset += size
            mapping[name] = record["shard"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalogue.jsonl"
            review = review_tensors(tensors, mapping, self.config, complete=True, catalogue_path=path)
            rows = {r["name"]: r for r in map(json.loads, path.read_text().splitlines())}
        self.assertEqual(review["findings"]["count"], 0, review)
        self.assertTrue(review["nvfp4_adapter_metadata_verified"])
        self.assertFalse(review["fp8_adapter_metadata_verified"])
        name = "model.layers.1.mlp.experts.0.gate_proj.weight"
        self.assertEqual(rows[name]["logical_shape"], [32, 16])
        self.assertEqual(rows[name]["shape"], [32, 8])
        self.assertEqual(rows[name]["input_scale"], name[:-7] + ".input_scale")
        self.assertEqual(review["nvfp4_groups"]["valid_metadata_quadruples"], 12)

    def test_partial_metadata_defers_uninspected_ancillaries_but_never_verifies(self):
        record = next(r for r in self.records if r["dtype"] == "U8")
        tensor = TensorInfo("U8", tuple(record["shape"]), (0, record["nbytes"]), record["nbytes"], 1)
        mapping = {r["name"]: r["shard"] for r in self.records}
        with tempfile.TemporaryDirectory() as directory:
            result = review_tensors({record["name"]: tensor}, mapping, self.config, complete=False,
                                    catalogue_path=Path(directory) / "catalogue.jsonl")
        self.assertEqual(result["nvfp4_groups"]["deferred_ancillary_headers"], 3)
        self.assertFalse(result["nvfp4_adapter_metadata_verified"])

    def test_complete_producer_to_architecture_pipeline_accepts_nvfp4_scalars_and_packed_shapes(self):
        filenames = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        groups = {name: [] for name in filenames}
        for index, record in enumerate(self.records):
            groups[filenames[index % 2]].append((record["name"], record["dtype"], record["shape"]))
        headers, mapping, siblings, total = {}, {}, [], 0
        for filename, records in groups.items():
            raw, size, payload = make_header(records)
            headers[filename] = raw, size
            siblings.append({"rfilename": filename, "size": size})
            mapping.update({name: filename for name, _, _ in records})
            total += payload
        expected = {"model_id": MODEL, "revision": REVISION,
                    "weights": [{"name": s["rfilename"], "bytes": s["size"]} for s in siblings],
                    "architecture": {**{k: v for k, v in self.config.items() if k != "quantization_config"},
                                     "quantization": deepcopy(self.config["quantization_config"])}}
        data = {"config": self.config, "model": {"id": MODEL, "sha": REVISION, "siblings": siblings},
                "index": {"metadata": {"total_size": total}, "weight_map": mapping}, "headers": headers, "expected": expected}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs").mkdir()
            write_json(root / "docs/model-metadata.json", expected)
            directory = root / "reports/metadata/run"
            directory.mkdir(parents=True)
            with redirect_stdout(io.StringIO()):
                report, code = execute_metadata(root, settings(), {"max_shards": 512, "budget_mib": 64, "offline": None},
                                                directory, source=MemorySource(data))
                publish_report(root, directory, report, code)
            self.assertEqual((report["status"], code), ("PASS", 0), report)
            self.assertTrue(report["tensor_review"]["nvfp4_adapter_metadata_verified"])
            architecture = run_architecture(root, settings())
            self.assertEqual(architecture["status"], "PASS", architecture)
            self.assertTrue(architecture["nvfp4"]["metadata_verified"])
            self.assertFalse(architecture["fp8"]["metadata_verified"])
            self.assertFalse(architecture["inference_verified"])

    def test_input_config_and_records_remain_unchanged(self):
        config, records = deepcopy(self.config), deepcopy(self.records)
        self.review()
        self.assertEqual(self.config, config)
        self.assertEqual(self.records, records)
        self.assertEqual(quantization_format(fixture_config()), "fp8")


if __name__ == "__main__":
    unittest.main()
