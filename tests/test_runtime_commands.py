"""Runtime command/CLI/worker boundaries, using tiny local data and mocked GPU."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from checkpoint_test_helpers import settings
from test_runtime_weights import CpuKernel, prepare_runtime_fixture
from test_streaming_decoder import decoder_config
from glm_local import __main__ as cli, runtime_commands as commands, runtime_worker as worker
from glm_local.checkpoint_snapshot import write_json
from glm_local.execution import FULL_MODEL_FLAGS
from glm_local.winjob import InstalledLimits


class RuntimeCommandsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parameters = {"backend": "cpu", "context": 8, "generate": 2}
        self.config = decoder_config()
        prepare_runtime_fixture(self.root, config_overrides=self.config)
        self.output = self.root / "reports/runtime-test"
        self.output.mkdir()
        self.config_path = self.root / "local.json"
        write_json(self.config_path, settings())

    def execute(self, action="plan", parameters=None):
        return commands.execute_runtime(self.root, settings(), action,
                                        self.parameters if parameters is None else parameters, self.output)

    def assert_full_unverified(self, report):
        for key in FULL_MODEL_FLAGS:
            self.assertIs(report[key], False, key)

    def latest(self, action="plan"):
        return json.loads((self.root / "reports" / f"{action}-latest.json").read_text(encoding="utf-8"))

    def test_plan_uses_complete_metadata_and_never_starts_kernels(self):
        with patch.object(commands, "_kernels") as kernels, patch.object(commands, "read_gpus") as hardware:
            report, code = self.execute()
        self.assertEqual((report["status"], code), ("ESTIMATE_FITS", 0), report)
        self.assertEqual(report["plan"]["backbone_layers"], 2)
        self.assertEqual(report["plan"]["context_tokens"], 8)
        self.assertTrue(report["plan"]["estimate_only"])
        self.assertIn("sha256", report["source_report"])
        self.assert_full_unverified(report)
        kernels.assert_not_called()
        hardware.assert_not_called()

    def test_hybrid_plan_accepts_finite_float_free_vram_and_truncates_to_bytes(self):
        parameters = {**self.parameters, "backend": "hybrid"}
        with patch.object(commands, "read_gpus", return_value=[{"index": 0, "memory_free_bytes": 4_000_000_000.75}]):
            report, code = self.execute(parameters=parameters)
        self.assertEqual(code, 0, report)
        self.assertEqual(report["plan"]["vram"]["budget_bytes"], 4_000_000_000)
        self.assertIs(type(report["plan"]["vram"]["budget_bytes"]), int)

    def test_hybrid_explicit_budget_never_queries_gpu(self):
        with patch.object(commands, "read_gpus") as gpu:
            report, code = self.execute(parameters={**self.parameters, "backend": "hybrid",
                                                    "vram_budget_bytes": 1024**3})
        self.assertEqual(code, 0, report)
        gpu.assert_not_called()

    def test_hybrid_missing_or_invalid_free_vram_fails_without_execution(self):
        for value in (None, True, 0, -1, float("inf"), float("nan"), "4000000000"):
            with self.subTest(value=value), patch.object(commands, "read_gpus", return_value=[{
                    "index": 0, "memory_free_bytes": value}]), patch.object(commands, "_kernels") as kernels:
                report, code = self.execute(parameters={**self.parameters, "backend": "hybrid"})
                self.assertEqual((report["status"], code), ("ERROR", 1))
                self.assertIn("free-VRAM", report["error"])
                kernels.assert_not_called()
        with patch.object(commands, "read_gpus", return_value=[]):
            self.assertEqual(self.execute(parameters={**self.parameters, "backend": "hybrid"})[1], 1)

    def test_bad_plan_dimensions_and_budgets_fail_before_kernels(self):
        for change in ({"context": 0}, {"context": 33}, {"generate": -1}, {"generate": 9},
                       {"backend": "hybrid", "vram_budget_bytes": -1}):
            with self.subTest(change=change), patch.object(commands, "_kernels") as kernels:
                report, code = self.execute(parameters={**self.parameters, **change})
                self.assertEqual((report["status"], code), ("ERROR", 1), report)
                kernels.assert_not_called()
                self.assert_full_unverified(report)

    def test_malformed_source_and_changed_catalogue_fail_closed(self):
        path = self.root / "reports/metadata-latest.json"
        original = path.read_bytes()
        path.write_text("{bad", encoding="utf-8")
        report, code = self.execute()
        self.assertEqual((report["status"], code), ("ERROR", 1))
        path.write_bytes(original)
        source = json.loads(original)
        catalogue = Path(source["run_directory"]) / "tensor-catalogue.jsonl"
        with catalogue.open("ab") as stream:
            stream.write(b" ")
        report, code = self.execute()
        self.assertEqual((report["status"], code), ("ERROR", 1))

    def test_valid_partial_metadata_cannot_authorize_runtime_plan(self):
        path = self.root / "reports/metadata-latest.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        source["status"] = "REVIEW_REQUIRED"
        source["metadata_structure_verified"] = False
        source["coverage"]["complete"] = False
        source["coverage"]["total_shards"] += 1
        source["coverage"]["total_index_tensors"] += 1
        write_json(path, source)
        report, code = self.execute()
        self.assertEqual((report["status"], code), ("ERROR", 1))
        self.assertIn("must PASS", report["error"])

    def test_generate_tiny_local_payloads_remains_explicitly_unverified(self):
        parameters = {**self.parameters, "tokens": [1, 2]}
        with patch.object(commands, "_kernels", return_value=(CpuKernel(), None, None)), \
                patch("glm_local.tokenizer.load_tokenizer") as loader:
            report, code = self.execute("generate", parameters)
        loader.assert_not_called()
        self.assertEqual((report["status"], code), ("GENERATED_UNVERIFIED", 0), report)
        self.assertEqual(len(report["generated_token_ids"]), 2)
        self.assertTrue(all(type(token) is int and 0 <= token < 32 for token in report["generated_token_ids"]))
        self.assertEqual(report["prompt_tokens"], 2)
        self.assert_full_unverified(report)
        self.assertFalse(report["reader"]["whole_payload_checksum_verified"])

    def test_text_generation_prepares_tokenizer_offline_and_decodes_generated_ids(self):
        with patch.object(commands, "_kernels", return_value=(CpuKernel(), None, None)), \
                patch("glm_local.tokenizer.prepare_tokenizer", return_value={"manifest": {"test": True}}) as prepare, \
                patch("glm_local.tokenizer.load_tokenizer") as loader:
            tokenizer = loader.return_value
            tokenizer.encode.return_value = [1, 2]
            tokenizer.decode.return_value = "decoded continuation"
            report, code = self.execute("generate", {**self.parameters, "tokens": None, "prompt": "tiny prompt"})
        self.assertEqual((report["status"], code), ("GENERATED_UNVERIFIED", 0), report)
        self.assertIs(prepare.call_args.kwargs["online"], False)
        self.assertEqual(loader.call_args.kwargs["model_id"], settings()["model_id"])
        self.assertEqual(loader.call_args.kwargs["revision"], settings()["revision"])
        tokenizer.encode.assert_called_once_with("tiny prompt", add_special_tokens=True)
        tokenizer.decode.assert_called_once_with(report["generated_token_ids"], skip_special_tokens=True)
        self.assertEqual(report["text"], "decoded continuation")
        self.assert_full_unverified(report)

    def test_tokenizer_check_routes_explicit_download_preference_without_kernels(self):
        prepared = {"status": "PASS", "tokenizer_verified": True, "manifest": {"test": True}}
        with patch("glm_local.tokenizer.prepare_tokenizer", return_value=prepared) as prepare, \
                patch.object(commands, "_kernels") as kernels:
            report, code = self.execute("tokenizer", {"online": True, "model_directory": "models/local"})
        self.assertEqual((report["status"], code), ("PASS", 0))
        prepare.assert_called_once_with(self.root, settings(), "models/local", online=True)
        kernels.assert_not_called()
        self.assert_full_unverified(report)

    def test_prompt_budget_failure_precedes_kernel_creation(self):
        with patch.object(commands, "_kernels") as kernels:
            report, code = self.execute("generate", {**self.parameters, "tokens": list(range(7))})
        self.assertEqual((report["status"], code), ("ERROR", 1))
        kernels.assert_not_called()

    def test_invalid_projection_seed_is_rejected_before_descriptor_or_payload(self):
        for seed in (True, -1, 2**32, 0.5):
            with self.subTest(seed=seed), patch.object(commands, "build_projection_descriptor") as descriptor:
                report, code = self.execute("projection", {"seed": seed, "tensor": "not-read"})
                self.assertEqual((report["status"], code), ("ERROR", 1))
                self.assertIn("uint32", report["error"])
                descriptor.assert_not_called()

    def test_interrupt_and_unknown_action_have_defined_exit_codes(self):
        with patch.object(commands, "verified_source", side_effect=KeyboardInterrupt):
            report, code = self.execute()
        self.assertEqual((report["status"], code), ("INTERRUPTED", 130))
        self.assert_full_unverified(report)
        report, code = self.execute("unknown")
        self.assertEqual((report["status"], code), ("ERROR", 1))

    def test_windows_job_policy_callback_precedes_worker_plan(self):
        events = []
        def run(command, *, cwd, limits, on_policy, timeout):
            self.assertIn("glm_local.runtime_worker", command)
            self.assertEqual(cwd, self.root.resolve())
            self.assertEqual((limits.cpu_percent, limits.committed_memory_bytes), (70, 32_000_000_000))
            self.assertEqual(timeout, 1800)
            on_policy(InstalledLimits(70, 32_000_000_000, True, True, True))
            events.append("policy")
            request = Path(command[-1])
            data = json.loads(request.read_text(encoding="utf-8"))
            events.append("worker")
            report, code = commands.execute_runtime(cwd, data["settings"], data["action"], data["parameters"], request.parent)
            write_json(request.parent / "result.json", report)
            return code
        with patch.object(commands.sys, "platform", "win32"), patch.object(commands, "run_local_process", side_effect=run), redirect_stdout(io.StringIO()):
            code = commands.launch_runtime(self.root, settings(), "plan", self.parameters)
        self.assertEqual(code, 0)
        self.assertEqual(events, ["policy", "worker"])
        report = self.latest()
        self.assertTrue(report["job_policy_verified"])
        self.assertEqual(report["installed_job_policy"]["committed_memory_bytes"], 32_000_000_000)
        self.assertEqual(report, json.loads((Path(report["run_directory"]) / "result.json").read_text(encoding="utf-8")))
        self.assert_full_unverified(report)

    def test_windows_worker_exit_status_disagreement_or_absent_policy_is_error(self):
        for report_status, worker_code, policy in (("PASS", 2, True), ("ERROR", 0, True),
                                                    ("PASS", 0, False), ("unknown", 0, True)):
            with self.subTest(status=report_status, code=worker_code, policy=policy):
                def run(command, *, on_policy, **kwargs):
                    if policy:
                        on_policy(InstalledLimits(70, 32_000_000_000, True, True, True))
                    write_json(Path(command[-1]).parent / "result.json", {
                        "status": report_status, "action": "plan", "model_id": settings()["model_id"],
                        "revision": settings()["revision"], **FULL_MODEL_FLAGS})
                    return worker_code
                with patch.object(commands.sys, "platform", "win32"), patch.object(commands, "run_local_process", side_effect=run), redirect_stdout(io.StringIO()):
                    code = commands.launch_runtime(self.root, settings(), "plan", self.parameters)
                self.assertEqual(code, 1)
                self.assertEqual(self.latest()["status"], "ERROR")
                self.assert_full_unverified(self.latest())

    def test_windows_missing_result_and_interrupted_worker_are_published(self):
        for outcome, expected_code in ((0, 1), (KeyboardInterrupt(), 130)):
            with self.subTest(outcome=outcome), patch.object(commands.sys, "platform", "win32"), \
                    patch.object(commands, "run_local_process", side_effect=outcome if isinstance(outcome, BaseException) else None,
                                 return_value=outcome), redirect_stdout(io.StringIO()):
                code = commands.launch_runtime(self.root, settings(), "plan", self.parameters)
                self.assertEqual(code, expected_code)
                self.assert_full_unverified(self.latest())

    def test_worker_report_identity_action_and_capability_claims_must_match_request(self):
        for change in ({"action": "generate"}, {"model_id": "wrong/model"}, {"revision": "b" * 40},
                       *({flag: True} for flag in FULL_MODEL_FLAGS), {"inference_verified": 0}):
            with self.subTest(change=change):
                def run(command, *, on_policy, **kwargs):
                    on_policy(InstalledLimits(70, 32_000_000_000, True, True, True))
                    report = {"status": "ESTIMATE_FITS", "action": "plan", "model_id": settings()["model_id"],
                              "revision": settings()["revision"], **FULL_MODEL_FLAGS, **change}
                    write_json(Path(command[-1]).parent / "result.json", report)
                    return 0
                with patch.object(commands.sys, "platform", "win32"), \
                        patch.object(commands, "run_local_process", side_effect=run), redirect_stdout(io.StringIO()):
                    code = commands.launch_runtime(self.root, settings(), "plan", self.parameters)
                self.assertEqual(code, 1)
                report = self.latest()
                self.assertIn("identity, action or capability", report["error"])
                self.assertEqual(report["model_id"], settings()["model_id"])
                self.assert_full_unverified(report)

    def test_nonwindows_launch_publishes_without_claiming_windows_limits(self):
        with patch.object(commands.sys, "platform", "linux"), patch.object(commands, "run_local_process") as launcher, redirect_stdout(io.StringIO()):
            code = commands.launch_runtime(self.root, settings(), "plan", self.parameters)
        self.assertEqual(code, 0)
        launcher.assert_not_called()
        report = self.latest()
        self.assertFalse(report["job_policy_verified"])
        self.assertIsNone(report["installed_job_policy"])
        self.assertIn("ESTIMATE_FITS", (self.root / "reports/plan-latest.md").read_text(encoding="utf-8"))

    def test_runtime_worker_reads_request_and_publishes_adjacent_result(self):
        directory = self.root / "reports/plan/worker-run"
        directory.mkdir(parents=True)
        request = directory / "request.json"
        write_json(request, {"root": str(self.root), "settings": settings(), "action": "plan", "parameters": self.parameters})
        with patch.object(worker, "execute_runtime", wraps=commands.execute_runtime) as execution:
            self.assertEqual(worker.main([str(request)]), 0)
        self.assertEqual(execution.call_args.args[-1], request.parent)
        self.assertEqual(json.loads((request.parent / "result.json").read_text(encoding="utf-8"))["status"], "ESTIMATE_FITS")
        with self.assertRaises(ValueError):
            worker.main([])

    def test_runtime_worker_rejects_request_path_root_or_action_mismatch_before_work(self):
        directory = self.root / "reports/plan/worker-guard"
        directory.mkdir(parents=True)
        baseline = {"root": str(self.root), "settings": settings(), "action": "plan", "parameters": self.parameters}
        for filename, change in (("other.json", {}), ("request.json", {"root": str(self.root / "wrong-root")}),
                                 ("request.json", {"action": "generate"}), ("request.json", {"action": "unknown"})):
            with self.subTest(filename=filename, change=change):
                request = directory / filename
                write_json(request, {**baseline, **change})
                with patch.object(worker, "execute_runtime") as execution, self.assertRaises(ValueError):
                    worker.main([str(request)])
                execution.assert_not_called()

    def test_cli_maps_plan_projection_and_generation_arguments(self):
        cases = [(["runtime-plan", "--backend", "cpu", "--context", "8", "--generate", "2", "--vram-budget-mib", "512"],
                  "plan", {"backend": "cpu", "context": 8, "generate": 2, "vram_budget_bytes": 512 * 1024**2}),
                 (["projection-check", "--backend", "cpu", "--tensor", "x.weight", "--online", "--budget-mib", "8", "--seed", "3"],
                  "projection", {"tensor": "x.weight", "backend": "cpu", "online": True, "budget_mib": 8, "seed": 3}),
                 (["generate", "--tokens", "1,2,3", "--backend", "cpu", "--context", "8", "--generate", "2", "--timeout", "15"],
                  "generate", {"tokens": [1, 2, 3], "backend": "cpu", "context": 8, "generate": 2, "timeout": 15}),
                 (["tokenizer-check", "--online", "--model-directory", "models/local"],
                  "tokenizer", {"online": True, "model_directory": str(Path("models/local"))})]
        for args, action, subset in cases:
            with self.subTest(action=action), patch.object(commands, "launch_runtime", return_value=7) as launch:
                self.assertEqual(cli.main(["--config", str(self.config_path), *args]), 7)
                self.assertEqual(launch.call_args.args[2], action)
                self.assertEqual({key: launch.call_args.args[3][key] for key in subset}, subset)

    def test_cli_rejects_invalid_transfer_vram_and_timeout_budgets_before_launch(self):
        for args in (["runtime-plan", "--vram-budget-mib", "0"],
                     ["runtime-plan", "--vram-budget-mib", "1048577"],
                     ["projection-check", "--budget-mib", "0"], ["projection-check", "--budget-mib", "129"],
                     ["generate", "--tokens", "1", "--timeout", "0"],
                     ["generate", "--tokens", "1", "--timeout", "86401"], ["generate", "--tokens", "1,no"]):
            with self.subTest(args=args), patch.object(commands, "launch_runtime") as launch, redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--config", str(self.config_path), *args]), 1)
                launch.assert_not_called()

    def test_cli_text_and_token_inputs_are_mutually_exclusive(self):
        with patch.object(commands, "launch_runtime") as launch, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.main(["--config", str(self.config_path), "generate", "--tokens", "1", "--prompt", "hello"])
        self.assertEqual(error.exception.code, 2)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
