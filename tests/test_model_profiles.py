"""Checkpoint selection isolates evidence and declares external runtime needs."""
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glm_local.model_profiles import (PROFILE_NAMES, load_profile,
                                      metadata_snapshot_path, profile_config_path,
                                      reports_directory)


PROJECT = Path(__file__).resolve().parent.parent


class BundledProfileTests(unittest.TestCase):
    def test_bundled_profiles_are_pinned_and_preserve_resource_limits(self):
        fp8 = load_profile(PROJECT, "fp8")
        nvfp4 = load_profile(PROJECT, "nvfp4")
        self.assertEqual(PROFILE_NAMES, ("fp8", "nvfp4", "flash-w4a16"))
        self.assertEqual(fp8["model_id"], "dealignai/GLM-5.3-CYBERSECURITY-FP8")
        self.assertEqual(fp8["revision"], "5915c1b88f998a9c1e1a0c83688e285a08ae3ca5")
        self.assertEqual(nvfp4["model_id"], "dealignai/GLM-5.3-ABLITERATED-NVFP4")
        self.assertEqual(nvfp4["revision"], "371bdb985d0124e76348c91e4a8fcf3a9d719d09")
        for key in ("ram_budget_bytes", "cpu_job_percent", "gpu_index", "gpu_average_target",
                    "gpu_window_seconds", "disk_reserve_bytes"):
            self.assertEqual(fp8[key], nvfp4[key], key)
        self.assertNotEqual(fp8["model_directory"], nvfp4["model_directory"])

    def test_bundled_profiles_select_independent_metadata_and_reports(self):
        fp8, nvfp4 = [load_profile(PROJECT, name) for name in ("fp8", "nvfp4")]
        self.assertEqual(metadata_snapshot_path(PROJECT, fp8), PROJECT / "docs/model-metadata.json")
        self.assertEqual(metadata_snapshot_path(PROJECT, nvfp4),
                         PROJECT / "docs/models/abliterated-nvfp4/model-metadata.json")
        self.assertEqual(reports_directory(PROJECT, fp8), PROJECT / "reports")
        self.assertEqual(reports_directory(PROJECT, nvfp4), PROJECT / "reports/nvfp4")

    def test_flash_profile_pins_external_backend_and_isolates_evidence(self):
        flash = load_profile(PROJECT, "flash-w4a16")
        self.assertEqual(flash["model_id"], "dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16")
        self.assertEqual(flash["revision"], "4c2ee763ff75f3ba50652ee0f024b0b630cd54a0")
        self.assertEqual(flash["backend_status"], "external_openai_required")
        self.assertEqual(flash["inference_backend"], "openai")
        self.assertEqual(flash["served_model_name"], flash["model_id"])
        self.assertEqual(metadata_snapshot_path(PROJECT, flash),
                         PROJECT / "docs/models/flash-cybersecurity-w4a16/model-metadata.json")
        self.assertEqual(reports_directory(PROJECT, flash), PROJECT / "reports/flash-w4a16")
        self.assertEqual(flash["model_directory"], "models/GLM-5.3-Flash-CYBERSECURITY-W4A16")

    def test_flash_baseline_preserves_actual_architecture_and_native_rejection(self):
        from glm_local.architecture.mapper import _profile
        from glm_local.checkpoint_schema import Findings, compare_snapshot, quantization_format

        flash = load_profile(PROJECT, "flash-w4a16")
        snapshot = json.loads(metadata_snapshot_path(PROJECT, flash).read_text(encoding="utf-8"))
        architecture = snapshot["architecture"]
        self.assertEqual(architecture["model_type"], "glm5_next")
        self.assertEqual(architecture["architectures"], ["Glm5NextForConditionalGeneration"])
        self.assertEqual(architecture["text_config"]["num_hidden_layers"], 45)
        self.assertTrue(architecture["text_config"]["mhc"])
        self.assertIn("linear_attention", architecture["text_config"]["layer_types"])
        self.assertEqual(architecture["quantization"]["quant_method"], "compressed-tensors")
        self.assertEqual(architecture["quantization"]["format"], "pack-quantized")
        group = architecture["quantization"]["config_groups"]["group_0"]
        self.assertEqual(group["weights"]["num_bits"], 4)
        self.assertEqual(group["weights"]["group_size"], 32)
        self.assertIsNone(group["input_activations"])
        self.assertEqual(snapshot["weight_shards"], 120)
        self.assertEqual(snapshot["weight_bytes"], 194660869128)
        self.assertEqual(sum(weight["bytes"] for weight in snapshot["weights"]), snapshot["weight_bytes"])
        self.assertEqual(len({weight["name"] for weight in snapshot["weights"]}), 120)
        self.assertFalse(snapshot["verification"]["tensor_payload_read"])
        self.assertFalse(snapshot["verification"]["native_inference_supported"])
        self.assertFalse(snapshot["verification"]["external_inference_verified"])
        config = {key: value for key, value in architecture.items() if key != "quantization"}
        config["quantization_config"] = architecture["quantization"]
        sizes = {weight["name"]: weight["bytes"] for weight in snapshot["weights"]}
        self.assertTrue(compare_snapshot(flash["model_id"], flash["revision"], sizes, config, snapshot)["matched"])
        findings = Findings()
        self.assertIsNone(quantization_format(config))
        self.assertIsNone(_profile(config, findings, model_id=flash["model_id"], revision=flash["revision"]))
        self.assertIn("UNSUPPORTED_MODEL_TYPE", findings.counts)
        self.assertIn("UNSUPPORTED_QUANTIZATION_PROFILE", findings.counts)


class ProfilePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_legacy_layout_and_missing_parent_paths_are_supported(self):
        self.assertEqual(metadata_snapshot_path(self.root, {}), self.root / "docs/model-metadata.json")
        self.assertEqual(reports_directory(self.root, {}), self.root / "reports")
        self.assertEqual(profile_config_path(self.root, "nvfp4"),
                         self.root / "config/models/abliterated-nvfp4.json")
        self.assertFalse((self.root / "config").exists())

    def test_profile_names_cannot_select_arbitrary_files(self):
        for name in (None, "", "NVFP4", "main", "../local", "fp8.json", [], 1):
            with self.subTest(name=name), self.assertRaises(ValueError):
                profile_config_path(self.root, name)

    def test_snapshot_paths_are_confined_portable_json_paths(self):
        invalid = (None, "", "model-metadata.json", "reports/model-metadata.json", "docs",
                   "docs/model-metadata.txt", "docs/../model-metadata.json", "../docs/m.json",
                   "/docs/m.json", "C:/docs/m.json", "docs\\m.json", "docs//m.json",
                   "docs/./m.json", "docs/CON.json", "docs/a /m.json", "docs/a*/m.json",
                   "docs/a\x00/m.json", "docs/a?/m.json", "docs/a\"/m.json", "docs/" + "a" * 1024)
        for name in invalid:
            with self.subTest(name=name), self.assertRaises(ValueError):
                metadata_snapshot_path(self.root, {"metadata_snapshot": name})

    def test_report_namespaces_are_single_portable_slugs(self):
        for namespace in (None, "", "../fp8", "fp8/nvfp4", "NVFP4", "nv_fp4", "con",
                          "aux", "nvfp4.", "nv fp4", "-nvfp4", "nvfp4-", "nv--fp4", "a" * 65, 7):
            with self.subTest(namespace=namespace), self.assertRaises(ValueError):
                reports_directory(self.root, {"report_namespace": namespace})
        self.assertEqual(reports_directory(self.root, {"report_namespace": "test-nvfp4"}),
                         self.root / "reports/test-nvfp4")

    def test_snapshot_directory_cannot_be_read_as_file(self):
        (self.root / "docs/model-metadata.json").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "regular file"):
            metadata_snapshot_path(self.root, {})

    def test_report_file_cannot_be_used_as_directory(self):
        (self.root / "reports").write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "regular directory"):
            reports_directory(self.root, {"report_namespace": "nvfp4"})

    def test_link_or_windows_reparse_component_is_rejected(self):
        original = Path.lstat
        docs = self.root / "docs"
        for mode, attributes in ((stat.S_IFLNK | 0o777, 0), (stat.S_IFDIR | 0o755, 0x400)):
            def inspected(path, *args, **kwargs):
                if path == docs:
                    return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
                return original(path, *args, **kwargs)
            with self.subTest(mode=mode, attributes=attributes), patch.object(Path, "lstat", inspected):
                with self.assertRaisesRegex(ValueError, "links or reparse points"):
                    metadata_snapshot_path(self.root, {})

    def test_profile_reads_are_bounded_strict_and_identity_checked(self):
        path = profile_config_path(self.root, "fp8")
        path.parent.mkdir(parents=True)
        settings = load_profile(PROJECT, "fp8")
        invalid = (
            (b" " * (64 * 1024 + 1), "exceeds 64 KiB"),
            (b'{"a": 1, "a": 2}', "Duplicate JSON key"),
            (b"[]", "JSON object"),
            (json.dumps({**settings, "revision": "main"}).encode(), "pinned"),
            (json.dumps({**settings, "model_id": "other/checkpoint"}).encode(), "identity"),
            (json.dumps({**settings, "metadata_snapshot": "../bad.json"}).encode(), "path"),
        )
        for raw, message in invalid:
            path.write_bytes(raw)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                load_profile(self.root, "fp8")
        path.write_bytes(b"\xef\xbb\xbf" + json.dumps(settings).encode())
        self.assertEqual(load_profile(self.root, "fp8"), settings)


if __name__ == "__main__":
    unittest.main()
