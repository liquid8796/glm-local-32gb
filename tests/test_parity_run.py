import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local.parity_run import launch_parity
from glm_local.winjob import InstalledLimits


def settings():
    return {"model_directory": "unused", "ram_budget_bytes": 32_000_000_000,
            "cpu_job_percent": 70, "gpu_average_target": .6,
            "gpu_window_seconds": 10, "gpu_index": 0, "disk_reserve_bytes": 0}


class ParityLaunchTests(unittest.TestCase):
    def test_missing_environment_does_not_run_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch("glm_local.parity_run.run_local_process") as runner:
            with self.assertRaisesRegex(RuntimeError, "setup-reference"):
                launch_parity(tmp, settings())
            runner.assert_not_called()

    def test_job_limits_reference_interpreter_and_error_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / ".venv-reference/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            def run(args, **kwargs):
                self.assertEqual(args[0], str(python))
                self.assertEqual(kwargs["limits"].cpu_percent, 70)
                self.assertEqual(kwargs["limits"].committed_memory_bytes, 32_000_000_000)
                kwargs["on_policy"](InstalledLimits(70, 32_000_000_000, True, True, True))
                result = Path(args[-1]).parent / "result.json"
                result.write_text(json.dumps({"status": "PASS"}))
                return 1  # Contradiction must never publish success.
            with patch("glm_local.parity_run.run_local_process", side_effect=run), patch("builtins.print"):
                self.assertEqual(launch_parity(root, settings()), 1)
            report = json.loads((root / "reports/parity-latest.json").read_text())
            self.assertEqual(report["status"], "ERROR")
            self.assertFalse(report["synthetic_official_parity_verified"])
            self.assertTrue(report["job_policy_verified"])

    def test_parameter_budget_rejected_before_directory_or_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch("glm_local.parity_run.run_local_process") as runner:
            with self.assertRaises(ValueError):
                launch_parity(tmp, settings(), lengths=(64, 96, 120), generate=8)
            self.assertFalse((Path(tmp)/"reports").exists())
            runner.assert_not_called()
