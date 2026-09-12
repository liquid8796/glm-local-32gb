"""Two checkpoint profiles share a project without replacing each other's evidence."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local import (__main__ as cli, checkpoint_check as metadata,
                       checkpoint_worker, runtime_commands, runtime_worker, tokenizer)
from glm_local.architecture import report as architecture
from glm_local.checkpoint_snapshot import write_json
from glm_local.model_profiles import load_profile, metadata_snapshot_path, reports_directory
from checkpoint_test_helpers import MemorySource, settings
from test_architecture_report import full_checkpoint
from test_audit import hardware
from test_tokenizer import manifest_for


PROJECT = Path(__file__).resolve().parent.parent


class ProfileWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.settings = {**settings(), "report_namespace": "nvfp4",
                         "metadata_snapshot": "docs/models/test/model-metadata.json"}
        self.parameters = {"max_shards": 512, "budget_mib": 64, "offline": None}
        self.data = full_checkpoint()
        self.data["model"]["siblings"].extend(manifest_for({
            "tokenizer.json": b"{}", "tokenizer_config.json": b"{}"})["siblings"])
        write_json(metadata_snapshot_path(self.root, self.settings), self.data["expected"])

    def publish_metadata(self):
        with patch.object(metadata.sys, "platform", "linux"), \
                patch.object(metadata, "HttpMetadataSource", return_value=MemorySource(self.data)), \
                redirect_stdout(io.StringIO()):
            code = metadata.launch_metadata(self.root, self.settings)
        self.assertEqual(code, 0)
        path = reports_directory(self.root, self.settings) / "metadata-latest.json"
        return json.loads(path.read_text(encoding="utf-8")), path

    def test_metadata_uses_selected_baseline_and_keeps_legacy_reports_unchanged(self):
        write_json(self.root / "docs/model-metadata.json", {"model_id": "wrong/model"})
        legacy = self.root / "reports/metadata-latest.json"
        write_json(legacy, {"sentinel": "existing FP8 report"})
        before = legacy.read_bytes()
        report, path = self.publish_metadata()
        self.assertTrue(report["baseline_comparison"]["matched"])
        self.assertEqual(report["io"]["tensor_payload_bytes_requested"], 0)
        self.assertEqual(Path(report["run_directory"]).parent, path.parent / "metadata")
        self.assertEqual(legacy.read_bytes(), before)
        self.assertEqual(json.loads((Path(report["run_directory"]) / "result.json").read_text()), report)

    def test_architecture_and_tokenizer_consume_only_selected_namespace(self):
        self.publish_metadata()
        write_json(self.root / "reports/metadata-latest.json", {"sentinel": "legacy"})
        legacy = self.root / "reports/architecture-latest.json"
        write_json(legacy, {"sentinel": "legacy architecture"})
        result = architecture.run_architecture(self.root, self.settings)
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue((self.root / "reports/nvfp4/architecture-latest.json").is_file())
        self.assertEqual(json.loads(legacy.read_text()), {"sentinel": "legacy architecture"})
        manifest, provenance = tokenizer._captured_manifest(self.root, self.settings)
        self.assertEqual(manifest["id"], self.settings["model_id"])
        self.assertEqual(len(manifest["siblings"]), 2)
        self.assertEqual(Path(provenance["source"]), self.root / "reports/nvfp4/metadata-latest.json")

    def test_architecture_rejects_metadata_run_outside_selected_namespace(self):
        report, source = self.publish_metadata()
        report["run_directory"] = str(self.root / "reports/metadata/legacy-run")
        write_json(source, report)
        result = architecture.run_architecture(self.root, self.settings)
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("Referenced metadata run", result["error"])

    def test_unmatched_source_baseline_clears_both_quantization_verification_flags(self):
        report, source = self.publish_metadata()
        report["baseline_comparison"]["matched"] = False
        write_json(source, report)
        mapping = {"status": "PASS", "metadata_mapping_verified": True,
                   "architecture_mapping_verified": True,
                   "fp8": {"metadata_verified": True}, "nvfp4": {"metadata_verified": True}}
        with patch.object(architecture, "analyze_catalogue", return_value=mapping):
            result = architecture.run_architecture(self.root, self.settings)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["source_eligibility_verified"])
        self.assertFalse(result["fp8"]["metadata_verified"])
        self.assertFalse(result["nvfp4"]["metadata_verified"])

    def test_metadata_worker_accepts_selected_namespace_and_rejects_cross_namespace(self):
        request = self.root / "reports/nvfp4/metadata/run/request.json"
        write_json(request, {"settings": self.settings, "parameters": self.parameters})
        with patch.object(checkpoint_worker, "ROOT", self.root), \
                patch.object(checkpoint_worker, "execute_metadata", return_value=({"status": "PASS"}, 0)) as execute:
            self.assertEqual(checkpoint_worker.main([str(request)]), 0)
            self.assertEqual(execute.call_args.args, (self.root, self.settings, self.parameters, request.parent))
        self.assertEqual(json.loads((request.parent / "result.json").read_text()), {"status": "PASS"})
        for namespace in (None, "other"):
            configured = deepcopy(self.settings)
            if namespace is None:
                configured.pop("report_namespace")
            else:
                configured["report_namespace"] = namespace
            write_json(request, {"settings": configured, "parameters": self.parameters})
            with self.subTest(namespace=namespace), patch.object(checkpoint_worker, "ROOT", self.root), \
                    patch.object(checkpoint_worker, "execute_metadata") as execute, self.assertRaisesRegex(ValueError, "namespace"):
                checkpoint_worker.main([str(request)])
            execute.assert_not_called()

    def test_metadata_worker_requires_bounded_generated_run_request(self):
        for relative in ("request.json", "reports/nvfp4/request.json", "reports/nvfp4/metadata/request.json",
                         "reports/nvfp4/metadata/run/nested/request.json"):
            request = self.root / relative
            write_json(request, {"settings": self.settings, "parameters": self.parameters})
            with self.subTest(path=relative), patch.object(checkpoint_worker, "ROOT", self.root), \
                    patch.object(checkpoint_worker, "execute_metadata") as execute, self.assertRaises(ValueError):
                checkpoint_worker.main([str(request)])
            execute.assert_not_called()
        request = self.root / "reports/nvfp4/metadata/big/request.json"
        request.parent.mkdir(parents=True)
        request.write_bytes(b" " * 65537)
        with patch.object(checkpoint_worker, "ROOT", self.root), \
                patch.object(checkpoint_worker, "execute_metadata") as execute, self.assertRaisesRegex(ValueError, "size policy"):
            checkpoint_worker.main([str(request)])
        execute.assert_not_called()

    def test_runtime_worker_enforces_settings_namespace(self):
        request = self.root / "reports/nvfp4/plan/run/request.json"
        document = {"root": str(self.root), "settings": self.settings, "action": "plan", "parameters": {}}
        write_json(request, document)
        with patch.object(runtime_worker, "execute_runtime", return_value=({"status": "PASS"}, 0)) as execute:
            self.assertEqual(runtime_worker.main([str(request)]), 0)
            self.assertEqual(execute.call_args.args[-1], request.parent)
        for namespace in ("fp8", "../nvfp4", None):
            configured = deepcopy(self.settings)
            if namespace is None:
                configured.pop("report_namespace")
            else:
                configured["report_namespace"] = namespace
            write_json(request, {**document, "settings": configured})
            with self.subTest(namespace=namespace), patch.object(runtime_worker, "execute_runtime") as execute, \
                    self.assertRaises(ValueError):
                runtime_worker.main([str(request)])
            execute.assert_not_called()

    def test_doctor_uses_selected_baseline_and_cache_without_changing_legacy(self):
        snapshot = {**self.data["expected"],
                    "weight_bytes": sum(item["bytes"] for item in self.data["expected"]["weights"])}
        write_json(metadata_snapshot_path(self.root, self.settings), snapshot)
        legacy = self.root / "reports/model-metadata.json"
        write_json(legacy, {"sentinel": "FP8 cache"})
        before = legacy.read_bytes()
        with patch.object(cli, "ROOT", self.root), patch.object(cli, "detect_hardware", return_value=hardware()), \
                patch("glm_local.audit.disk_free_for", return_value=10**12), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.doctor(self.settings, False), 2)
            with patch.object(cli, "refresh_metadata", return_value=snapshot) as refresh:
                self.assertEqual(cli.doctor(self.settings, True), 2)
                refresh.assert_called_once_with(self.settings["model_id"], self.settings["revision"])
        self.assertEqual(legacy.read_bytes(), before)
        report = json.loads((self.root / "reports/nvfp4/latest.json").read_text())
        self.assertEqual(report["model_id"], self.settings["model_id"])
        self.assertEqual(json.loads((self.root / "reports/nvfp4/model-metadata.json").read_text()), snapshot)

    def test_cli_profiles_and_config_are_exclusive_and_preserve_root(self):
        for name, filename in (("fp8", "cybersecurity-fp8.json"), ("nvfp4", "abliterated-nvfp4.json")):
            write_json(self.root / "config/models" / filename, load_profile(PROJECT, name))
        for profile in ("fp8", "nvfp4"):
            with self.subTest(profile=profile), patch.object(cli, "ROOT", self.root), \
                    patch.object(metadata, "launch_metadata", return_value=0) as launch:
                self.assertEqual(cli.main(["--profile", profile, "metadata-check"]), 0)
            self.assertEqual(launch.call_args.args[0], self.root)
            self.assertEqual(launch.call_args.args[1], load_profile(PROJECT, profile))
        with patch.object(metadata, "launch_metadata") as launch, redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            cli.main(["--profile", "nvfp4", "--config", "other.json", "metadata-check"])
        self.assertEqual(error.exception.code, 2)
        launch.assert_not_called()

    def test_cli_default_and_explicit_config_work_without_forwarding_profile_option(self):
        write_json(self.root / "config/local.json", self.settings)
        explicit = {**self.settings, "report_namespace": "another"}
        write_json(self.root / "config/custom.json", explicit)
        for arguments, configured in (([], self.settings), (["--config", str(self.root / "config/custom.json")], explicit)):
            with self.subTest(arguments=arguments), patch.object(cli, "ROOT", self.root), \
                    patch.object(runtime_commands, "launch_runtime", return_value=0) as launch:
                self.assertEqual(cli.main([*arguments, "projection-check", "--backend", "cpu"]), 0)
            self.assertEqual(launch.call_args.args[1], configured)
            self.assertNotIn("profile", launch.call_args.args[3])
            self.assertNotIn("config", launch.call_args.args[3])
            self.assertIsNone(launch.call_args.args[3]["tensor"])


if __name__ == "__main__":
    unittest.main()
