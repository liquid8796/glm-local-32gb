import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local import __main__ as cli
from glm_local import __version__
from glm_local import checkpoint_check as check
from glm_local import checkpoint_worker as worker
from glm_local.checkpoint_http import HttpMetadataSource, MetadataError
from glm_local.checkpoint_snapshot import write_json
from glm_local.winjob import InstalledLimits
from checkpoint_test_helpers import (MODEL, REVISION, NAME, SCALE, MemorySource, Opener,
                                     Response, checkpoint, encode, make_header, ranged, settings)


class MetadataWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        self.data = checkpoint()
        write_json(self.root / "docs" / "model-metadata.json", self.data["expected"])
        self.parameters = {"max_shards": 512, "budget_mib": 64, "offline": None}
        self.iteration = 0

    def run_audit(self, source=None, parameters=None):
        self.iteration += 1
        directory = self.root / f"run-{self.iteration}"
        directory.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            report, code = check.execute_metadata(self.root, settings(), parameters or self.parameters,
                                                   directory, source=source or MemorySource(self.data))
        return report, code, directory

    def test_full_metadata_workflow_creates_catalogue_snapshot_and_truthful_flags(self):
        report, code, directory = self.run_audit()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["metadata_structure_verified"])
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["headers_checked"], 2)
        self.assertEqual(report["coverage"]["checked_tensors"], 6)
        self.assertTrue((directory / "evidence" / "snapshot.json").is_file())
        self.assertEqual(len((directory / "tensor-catalogue.jsonl").read_text().splitlines()), 6)
        for field in ("inference_verified", "architecture_mapping_verified", "real_checkpoint_compatible",
                      "full_model_loaded", "payload_values_verified", "gpu_used", "full_model_limits_verified"):
            self.assertFalse(report[field], field)
        self.assertEqual(report["io"]["tensor_payload_bytes_requested"], 0)
        accounting = report["tensor_payload_accounting"]
        self.assertEqual(accounting["validation_mode"], "strict_exact_total_after_all_headers")
        self.assertEqual(accounting["validation_status"], "verified")
        self.assertTrue(accounting["exact_match"])
        self.assertTrue(accounting["complete"])
        self.assertEqual(accounting["observed_minus_declared_bytes"], 0)
        self.assertEqual(report["observed_tensor_payload_bytes"], report["declared_tensor_payload_bytes"])

    def test_strict_http_transport_wired_through_complete_metadata_workflow(self):
        responses = [Response(encode(self.data[k]), headers={"Content-Length": len(encode(self.data[k]))})
                     for k in ("model", "config", "index")]
        for raw, size in self.data["headers"].values():
            responses += [ranged(raw[:8], 0, size), ranged(raw[8:], 8, size)]
        opener = Opener(responses)
        source = HttpMetadataSource(MODEL, REVISION, opener=opener)
        report, code, _ = self.run_audit(source)
        self.assertEqual(code, 0)
        self.assertEqual(len(opener.requests), 7)
        self.assertEqual(report["io"]["range_requests"], 4)
        expected_bytes = sum(len(encode(self.data[k])) for k in ("model", "config", "index"))
        expected_bytes += sum(len(raw) for raw, _ in self.data["headers"].values())
        self.assertEqual(report["io"]["body_bytes_read"], expected_bytes)

    def test_max_shards_is_partial_and_does_not_validate_uninspected_scale(self):
        source = MemorySource(self.data)
        report, code, _ = self.run_audit(source, {**self.parameters, "max_shards": 1})
        self.assertEqual((code, report["status"]), (2, "PARTIAL"))
        self.assertFalse(report["metadata_structure_verified"])
        self.assertEqual(len(source.calls), 4)
        self.assertFalse(report["tensor_review"]["fp8_adapter_metadata_verified"])
        self.assertEqual(report["tensor_review"]["fp8_pairs"]["deferred_scale_headers"], 1)
        accounting = report["tensor_payload_accounting"]
        self.assertEqual(accounting["validation_status"], "deferred")
        self.assertFalse(accounting["complete"])
        self.assertFalse(accounting["exact_match"])
        self.assertLess(accounting["observed_minus_declared_bytes"], 0)

    def test_partial_payload_equality_does_not_claim_complete_verification(self):
        source = MemorySource(self.data)
        first = sorted(source.data["headers"])[0]
        raw, size = source.data["headers"][first]
        source.data["index"]["metadata"]["total_size"] = size - len(raw)
        report, code, _ = self.run_audit(source, {**self.parameters, "max_shards": 1})
        self.assertEqual((code, report["status"]), (2, "PARTIAL"))
        self.assertFalse(report["metadata_structure_verified"])
        accounting = report["tensor_payload_accounting"]
        self.assertTrue(accounting["exact_match"])
        self.assertFalse(accounting["complete"])
        self.assertEqual(accounting["validation_status"], "deferred")

    def test_partial_payload_under_declared_total_defers_final_equality(self):
        source = MemorySource(self.data)
        source.data["index"]["metadata"]["total_size"] += 1
        report, code, _ = self.run_audit(source, {**self.parameters, "max_shards": 1})
        self.assertEqual((code, report["status"]), (2, "PARTIAL"))
        self.assertNotIn("error", report)
        self.assertEqual(report["tensor_payload_accounting"]["validation_status"], "deferred")

    def test_offline_replay_matches_catalogue_and_does_not_access_network(self):
        online, _, directory = self.run_audit()
        replay = self.root / "replay"; replay.mkdir()
        with patch.object(check, "HttpMetadataSource", side_effect=AssertionError("No network")), \
                contextlib.redirect_stdout(io.StringIO()):
            report, code = check.execute_metadata(self.root, settings(), {
                **self.parameters, "offline": str(directory / "evidence")}, replay)
        self.assertEqual(code, 0)
        self.assertEqual(report["source_mode"], "offline_replay")
        self.assertEqual(report["io"]["requests"], 0)
        self.assertFalse(report["io"]["remote_provenance_authenticated"])
        self.assertEqual(report["catalogue"]["sha256"], online["catalogue"]["sha256"])
        self.assertEqual(report["tensor_review"], online["tensor_review"])

    def test_partial_snapshot_replays_as_partial(self):
        _, _, directory = self.run_audit(parameters={**self.parameters, "max_shards": 1})
        replay = self.root / "replay"; replay.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            report, code = check.execute_metadata(self.root, settings(), {
                **self.parameters, "offline": str(directory / "evidence")}, replay)
        self.assertEqual(code, 2)
        self.assertEqual(report["coverage"]["checked_shards"], 1)
        self.assertFalse(report["metadata_structure_verified"])

    def test_remote_identity_mismatch_stops_before_config_or_header_fetch(self):
        source = MemorySource(self.data); source.data["model"]["sha"] = "b" * 40
        report, code, directory = self.run_audit(source)
        self.assertEqual(code, 1)
        self.assertEqual(source.calls, ["model"])
        self.assertFalse(report["metadata_structure_verified"])
        self.assertTrue((directory / "evidence" / "model.json").is_file())

    def test_baseline_identity_mismatch_never_opens_network(self):
        expected = deepcopy(self.data["expected"]); expected["revision"] = "b" * 40
        write_json(self.root / "docs" / "model-metadata.json", expected)
        source = MemorySource(self.data)
        report, code, _ = self.run_audit(source)
        self.assertEqual(code, 1)
        self.assertFalse(source.calls)

    def test_index_header_mismatch_retains_evidence_and_active_shard(self):
        source = MemorySource(self.data)
        source.data["index"]["weight_map"]["renamed.weight"] = source.data["index"]["weight_map"].pop(NAME)
        report, code, directory = self.run_audit(source)
        self.assertEqual(code, 1)
        self.assertIn("tensor set mismatch", report["error"])
        self.assertIn("active_shard", report)
        self.assertFalse(report["metadata_structure_verified"])
        self.assertEqual(len(list((directory / "evidence" / "headers").iterdir())), 1)

    def test_total_size_mismatch_fails_after_headers(self):
        source = MemorySource(self.data); source.data["index"]["metadata"]["total_size"] += 1
        report, code, _ = self.run_audit(source)
        self.assertEqual(code, 1)
        self.assertFalse(report["metadata_structure_verified"])
        self.assertIn("total_size", report["error"])
        self.assertEqual(report["stage"], "validate_payload_accounting")
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["headers_checked"], 2)
        accounting = report["tensor_payload_accounting"]
        self.assertTrue(accounting["complete"])
        self.assertFalse(accounting["exact_match"])
        self.assertEqual(accounting["validation_status"], "error")
        self.assertEqual(accounting["observed_minus_declared_bytes"], -1)
        self.assertNotIn("tensor_review", report)

    def test_exact_shard_file_total_is_proved_from_all_received_headers(self):
        source = MemorySource(self.data)
        source.data["index"]["metadata"]["total_size"] = sum(size for _, size in source.data["headers"].values())
        report, code, _ = self.run_audit(source)
        self.assertEqual(code, 0, report)
        accounting = report["tensor_payload_accounting"]
        self.assertEqual(accounting["index_size_convention"], "complete_shard_files")
        self.assertEqual(accounting["validation_status"], "verified")
        self.assertFalse(accounting["exact_match"])
        self.assertEqual(accounting["observed_prefix_and_header_bytes"],
                         sum(len(raw) for raw, _ in source.data["headers"].values()))
        self.assertTrue(accounting["payload_plus_headers_matches_manifest"])

    def test_near_shard_file_total_is_rejected(self):
        for delta in (-1, 1):
            source = MemorySource(self.data)
            source.data["index"]["metadata"]["total_size"] = sum(size for _, size in source.data["headers"].values()) + delta
            report, code, _ = self.run_audit(source)
            self.assertEqual(code, 1)
            self.assertFalse(report["metadata_structure_verified"])

    def test_zero_payload_empty_tensor_satisfies_accounting_without_fp8_claim(self):
        data = deepcopy(self.data)
        filename = "model-00001-of-00001.safetensors"
        raw, size, payload = make_header([("empty", "F32", [0, 3])])
        data["headers"] = {filename: (raw, size)}
        data["model"]["siblings"] = [{"rfilename": filename, "size": size}]
        data["index"] = {"metadata": {"total_size": payload}, "weight_map": {"empty": filename}}
        data["expected"]["weights"] = [{"name": filename, "bytes": size}]
        write_json(self.root / "docs" / "model-metadata.json", data["expected"])
        report, code, _ = self.run_audit(MemorySource(data))
        self.assertEqual((code, report["status"]), (3, "REVIEW_REQUIRED"))
        self.assertTrue(report["metadata_structure_verified"])
        self.assertEqual(report["observed_tensor_payload_bytes"], 0)
        self.assertEqual(report["declared_tensor_payload_bytes"], 0)
        self.assertEqual(report["tensor_payload_accounting"]["validation_status"], "verified")
        self.assertTrue(report["tensor_payload_accounting"]["exact_match"])
        self.assertFalse(report["tensor_review"]["fp8_adapter_metadata_verified"])
        self.assertIn("NO_FP8_WEIGHTS_OBSERVED", report["tensor_review"]["findings"]["by_code"])

    def test_observed_payload_cannot_exceed_index_even_during_partial_read(self):
        source = MemorySource(self.data); source.data["index"]["metadata"]["total_size"] = 1
        report, code, _ = self.run_audit(source, {**self.parameters, "max_shards": 1})
        self.assertEqual(code, 1)
        self.assertIn("already exceed", report["error"])
        self.assertFalse(report["metadata_structure_verified"])
        self.assertEqual(report["headers_checked"], 1)
        accounting = report["tensor_payload_accounting"]
        self.assertEqual(accounting["validation_status"], "error")
        self.assertFalse(accounting["complete"])
        self.assertGreater(accounting["observed_minus_declared_bytes"], 0)
        self.assertGreater(report["observed_tensor_payload_bytes"], report["declared_tensor_payload_bytes"])

    def test_config_mismatch_is_review_required_not_schema_or_runtime_pass(self):
        source = MemorySource(self.data); source.data["config"]["hidden_size"] = 258
        report, code, _ = self.run_audit(source)
        self.assertEqual((code, report["status"]), (3, "REVIEW_REQUIRED"))
        self.assertTrue(report["metadata_structure_verified"])
        self.assertFalse(report["baseline_comparison"]["matched"])
        self.assertFalse(report["real_checkpoint_compatible"])

    def test_invalid_scale_dtype_is_observed_in_full_metadata_review(self):
        data = deepcopy(self.data)
        filename = data["index"]["weight_map"][SCALE]
        old_size = data["headers"][filename][1]
        raw, size, payload = make_header([(SCALE, "BF16", [3, 3]), ("aux.stat", "F32", [])])
        data["headers"][filename] = (raw, size)
        old_payload = 9 * 4 + 4
        data["index"]["metadata"]["total_size"] += payload - old_payload
        for entry in data["model"]["siblings"]:
            if entry["rfilename"] == filename:
                entry["size"] = size
        report, code, _ = self.run_audit(MemorySource(data))
        self.assertEqual(code, 3)
        self.assertTrue(report["metadata_structure_verified"])
        self.assertFalse(report["tensor_review"]["fp8_adapter_metadata_verified"])
        self.assertIn("CURRENT_ADAPTER_REQUIRES_F32_SCALE", report["tensor_review"]["findings"]["by_code"])

    def test_fetch_failure_preserves_stage_traceback_partial_statistics(self):
        source = MemorySource(self.data)
        source.header_bytes = lambda *args: (_ for _ in ()).throw(MetadataError("test connection failure"))
        report, code, directory = self.run_audit(source)
        self.assertEqual(code, 1)
        self.assertEqual(report["stage"], "fetch_header")
        self.assertIn("test connection failure", report["traceback"])
        self.assertGreater(report["io"]["body_bytes_read"], 0)
        self.assertTrue((directory / "evidence" / "config.json").is_file())

    def test_interrupt_never_reports_success(self):
        source = MemorySource(self.data)
        source.json_bytes = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
        report, code, _ = self.run_audit(source)
        self.assertEqual((report["status"], code), ("INTERRUPTED", 130))
        self.assertFalse(report["metadata_structure_verified"])

    def test_invalid_arguments_fail_before_source_io(self):
        for changes in ({"max_shards": 0}, {"max_shards": True}, {"max_shards": 513},
                        {"budget_mib": 129}, {"budget_mib": 0}, {"offline": ""}):
            source = MemorySource(self.data)
            report, code, _ = self.run_audit(source, {**self.parameters, **changes})
            with self.subTest(changes=changes):
                self.assertEqual(code, 1)
                self.assertFalse(source.calls)

    def test_render_includes_error_coverage_scope_and_not_entire_shard_list(self):
        report, _, _ = self.run_audit()
        text = check.render_metadata(report)
        self.assertIn("Metadata only", text)
        self.assertIn("Real checkpoint runtime compatible: **False**", text)
        self.assertNotIn('"checked_shard_names"', text)
        self.assertIn("Tensor review", text)
        self.assertIn("Tensor payload accounting", text)
        self.assertIn('"validation_status": "verified"', text)

    def test_render_retains_payload_accounting_on_both_mismatch_errors(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                source = MemorySource(self.data)
                source.data["index"]["metadata"]["total_size"] += delta
                report, code, _ = self.run_audit(source)
                self.assertEqual(code, 1)
                text = check.render_metadata(report)
                self.assertIn("Status: **ERROR**", text)
                self.assertIn("Metadata structure verified: **False**", text)
                self.assertIn("Tensor payload accounting", text)
                self.assertIn('"validation_status": "error"', text)
                self.assertIn(f'"observed_minus_declared_bytes": {-delta}', text)
                self.assertIn('"exact_match": false', text)
                self.assertNotIn("shared/tied", text)

    def test_linux_launch_does_not_claim_windows_policy_and_publishes_latest(self):
        with patch.object(check.sys, "platform", "linux"), \
                patch.object(check, "HttpMetadataSource", return_value=MemorySource(self.data)), \
                contextlib.redirect_stdout(io.StringIO()):
            code = check.launch_metadata(self.root, settings())
        latest = json.loads((self.root / "reports" / "metadata-latest.json").read_text())
        self.assertEqual(code, 0)
        self.assertFalse(latest["job_policy_verified"])
        self.assertEqual(latest["tool_version"], __version__)
        self.assertEqual(latest, json.loads((Path(latest["run_directory"]) / "result.json").read_text()))

    def test_windows_launcher_passes_unchanged_job_limits_before_worker(self):
        def run(command, *, cwd, limits, on_policy, timeout):
            self.assertEqual(limits.cpu_percent, 70)
            self.assertEqual(limits.committed_memory_bytes, 32000000000)
            self.assertIn("glm_local.checkpoint_worker", command)
            self.assertLessEqual(timeout, 1860)
            on_policy(InstalledLimits(70, 32000000000, True, True, True))
            request = Path(command[-1])
            data = json.loads(request.read_text())
            report, code = check.execute_metadata(cwd, data["settings"], data["parameters"], request.parent,
                                                   source=MemorySource(self.data))
            write_json(request.parent / "result.json", report)
            return code
        with patch.object(check.sys, "platform", "win32"), patch.object(check, "run_local_process", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()):
            code = check.launch_metadata(self.root, settings())
        self.assertEqual(code, 0)
        report = json.loads((self.root / "reports" / "metadata-latest.json").read_text())
        self.assertTrue(report["job_policy_verified"])
        self.assertEqual(report["installed_job_policy"]["cpu_percent"], 70)

    def test_windows_worker_missing_report_is_error_not_pass(self):
        with patch.object(check.sys, "platform", "win32"), patch.object(check, "run_local_process", return_value=0), \
                contextlib.redirect_stdout(io.StringIO()):
            code = check.launch_metadata(self.root, settings())
        self.assertEqual(code, 1)
        report = json.loads((self.root / "reports" / "metadata-latest.json").read_text())
        self.assertEqual(report["status"], "ERROR")
        self.assertIn("bounded result", report["error"])

    def test_cli_wires_metadata_arguments_without_native_or_optional_libraries(self):
        config = self.root / "settings.json"; write_json(config, settings())
        with patch("glm_local.checkpoint_check.launch_metadata", return_value=2) as launch:
            code = cli.main(["--config", str(config), "metadata-check", "--max-shards", "1", "--budget-mib", "32"])
        self.assertEqual(code, 2)
        self.assertEqual(launch.call_args.args[2:], (1, 32, None))

    def test_worker_request_path_guard(self):
        with self.assertRaises(ValueError):
            worker.main([])
        request = self.root / "outside.json"; write_json(request, {})
        with patch.object(worker, "ROOT", self.root), self.assertRaises(ValueError):
            worker.main([str(request)])
