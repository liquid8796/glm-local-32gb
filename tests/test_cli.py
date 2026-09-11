import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local import __main__ as cli


class CliTests(unittest.TestCase):
    def test_successful_diagnostic_preserves_blocked_exit_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docs").mkdir()
            (root / "docs" / "model-metadata.json").write_text("{}", encoding="utf-8")
            report = {"status": "BLOCKED", "inference_verified": False}
            with patch.object(cli, "ROOT", root), patch.object(cli, "evaluate", return_value=report), \
                 patch.object(cli, "detect_hardware", return_value={}), \
                 patch.object(cli, "render_report", return_value="Status: BLOCKED\n"), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.doctor({}, False), 2)
            stored = json.loads((root / "reports" / "latest.json").read_text(encoding="utf-8"))
            self.assertFalse(stored["inference_verified"])
            self.assertEqual(stored["status"], "BLOCKED")
            self.assertEqual((root / "reports" / "latest.md").read_text(), "Status: BLOCKED\n")

    def test_invalid_monitor_duration_never_probes_gpu(self):
        for seconds in (0, -1, 61, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), patch.object(cli, "read_gpus") as gpu:
                with self.assertRaises(ValueError):
                    cli.monitor({}, seconds)
                gpu.assert_not_called()

    def test_missing_configuration_returns_explicit_error(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()) as err:
            status = cli.main(["--config", str(Path(temp) / "missing.json"), "doctor"])
        self.assertEqual(status, 1)
        self.assertIn("ERROR:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
