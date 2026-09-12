import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glm_local import __version__
from glm_local import parity_worker
from glm_local.parity_run import render_parity


class ParityWorkerDiagnosticTests(unittest.TestCase):
    def run_worker(self, request_content, *, result=None, error=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "reports" / "parity" / "unit-test"
            directory.mkdir(parents=True)
            request = directory / "request.json"
            request.write_text(request_content, encoding="utf-8")
            with patch.object(parity_worker, "ROOT", root), \
                    patch.object(sys, "argv", ["parity_worker", str(request)]), \
                    patch.object(parity_worker, "execute_parity", return_value=result, side_effect=error), \
                    patch.dict(sys.modules, {"safetensors": SimpleNamespace(__version__="0.8.0-test-double")}):
                code = parity_worker.main()
            return code, json.loads((directory / "result.json").read_text(encoding="utf-8"))

    @staticmethod
    def request():
        return json.dumps({"settings": {}, "parameters": {
            "storage": "safetensors", "backend": "hybrid", "lengths": [120], "generate": 8, "seed": 7}})

    def test_fixture_error_retains_parameters_environment_traceback_and_false_flags(self):
        code, report = self.run_worker(self.request(), error=TypeError(
            "argument 'tensor_dict': 'dict' object is not an instance of 'TensorSpec'"))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "ERROR")
        self.assertIn("TensorSpec", report["error"])
        self.assertEqual(report["parameters"]["storage"], "safetensors")
        self.assertEqual(report["parameters"]["lengths"], [120])
        self.assertEqual(report["worker_environment"]["safetensors_version"], "0.8.0-test-double")
        self.assertEqual(report["tool_version"], __version__)
        self.assertIn("execute_parity", report["traceback"])
        for flag in ("synthetic_official_parity_verified", "inference_verified", "full_model_loaded", "job_policy_verified"):
            self.assertFalse(report[flag])
        self.assertNotIn("cases", report)

    def test_error_markdown_contains_parameters_and_traceback(self):
        _, report = self.run_worker(self.request(), error=RuntimeError("fixture failure"))
        rendered = render_parity(report)
        self.assertIn("Status: **ERROR**", rendered)
        self.assertIn('"storage": "safetensors"', rendered)
        self.assertIn("0.8.0-test-double", rendered)
        self.assertIn("## Worker traceback", rendered)
        self.assertIn("RuntimeError: fixture failure", rendered)

    def test_malformed_request_still_writes_diagnostics(self):
        for data in ("{broken", "[]"):
            with self.subTest(data=data):
                code, report = self.run_worker(data)
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "ERROR")
                self.assertEqual(report["parameters"], {})
                self.assertIn("traceback", report)

    def test_traceback_size_is_bounded(self):
        _, report = self.run_worker(self.request(), error=RuntimeError("x" * 20000))
        self.assertLessEqual(len(report["traceback"]), 16000)

    def test_success_and_mismatch_preserve_exit_codes_and_do_not_add_error_traceback(self):
        for status, expected in (("PASS", 0), ("NUMERICAL_MISMATCH", 3)):
            with self.subTest(status=status):
                code, report = self.run_worker(self.request(), result={"status": status})
                self.assertEqual(code, expected)
                self.assertEqual(report["status"], status)
                self.assertEqual(report["tool_version"], __version__)
                self.assertNotIn("traceback", report)
