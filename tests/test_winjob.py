"""Small live Windows checks; no model, network, or CPU stress workload."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from glm_local.winjob import JobLimits, WindowsJob, run_local_process


class LimitsValidationTests(unittest.TestCase):
    def test_invalid_limits_are_rejected(self):
        for value in (0, 101, 2.5, True):
            with self.subTest(cpu=value), self.assertRaises(ValueError):
                JobLimits(cpu_percent=value)
        for value in (0, -1, 1.5, True):
            with self.subTest(memory=value), self.assertRaises(ValueError):
                JobLimits(committed_memory_bytes=value)


@unittest.skipUnless(sys.platform == "win32", "requires live Windows Job Objects")
class WindowsJobTests(unittest.TestCase):
    def setUp(self):
        self.limits = JobLimits(cpu_percent=70, committed_memory_bytes=192 * 1024**2)

    def test_installed_policy_is_queried_from_windows(self):
        limits = JobLimits(cpu_percent=35, committed_memory_bytes=192 * 1024**2)
        with WindowsJob(limits) as job:
            policy = job.query_limits()
            self.assertEqual(policy.cpu_percent, 35)
            self.assertEqual(policy.committed_memory_bytes, 192 * 1024**2)
            self.assertTrue(policy.cpu_hard_cap)
            self.assertTrue(policy.memory_limit_enabled)
            self.assertTrue(policy.kill_on_close)
        job.close()  # Closing twice is safe; querying a closed handle is not.
        with self.assertRaises(RuntimeError):
            job.query_limits()

    def test_exit_code_is_propagated(self):
        code = run_local_process([sys.executable, "-c", "raise SystemExit(37)"],
                                 limits=self.limits, timeout=10)
        self.assertEqual(code, 37)

    def test_unicode_paths_arguments_and_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="winjob-") as temp:
            directory = Path(temp) / "kiểm tra Unicode có dấu"
            directory.mkdir()
            script = directory / "đọc tham số.py"
            script.write_text(
                "import json, os, pathlib, sys\n"
                "pathlib.Path('kết quả.json').write_text(\n"
                "    json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}, ensure_ascii=False),\n"
                "    encoding='utf-8')\n", encoding="utf-8")
            arguments = ["xin chào", "", 'a"quoted" value', "trailing slash\\", "a\tb", "日本語"]
            code = run_local_process([sys.executable, script, *arguments], cwd=directory,
                                     limits=self.limits, timeout=10)
            self.assertEqual(code, 0)
            result = json.loads((directory / "kết quả.json").read_text(encoding="utf-8"))
            self.assertEqual(result["args"], arguments)
            self.assertEqual(Path(result["cwd"]), directory)

    def test_allocation_over_small_commit_limit_fails(self):
        script = ("import sys\n"
                  "try:\n"
                  "    allocation = bytearray(128 * 1024**2)\n"
                  "except MemoryError:\n"
                  "    sys.exit(42)\n"
                  "sys.exit(99)\n")
        code = run_local_process([sys.executable, "-c", script],
                                 limits=JobLimits(committed_memory_bytes=64 * 1024**2),
                                 timeout=10)
        self.assertEqual(code, 42, "allocation must be rejected by the job commit quota")

    def test_callback_error_prevents_child_code_from_running(self):
        with tempfile.TemporaryDirectory(prefix="winjob-") as temp:
            marker = Path(temp) / "child-ran.txt"

            def reject_policy(policy):
                self.assertTrue(policy.cpu_hard_cap)
                raise RuntimeError("test callback rejected policy")

            with self.assertRaisesRegex(RuntimeError, "callback rejected"):
                run_local_process(
                    [sys.executable, "-c", "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()",
                     marker], limits=self.limits, on_policy=reject_policy, timeout=10)
            self.assertFalse(marker.exists())

    def test_stdout_and_stderr_follow_parent_redirection(self):
        wrapper = (
            "import sys\n"
            "from glm_local.winjob import JobLimits, run_local_process\n"
            "sys.exit(run_local_process([sys.executable, '-c', "
            "\"import sys; print('child stdout'); print('child stderr', file=sys.stderr)\"], "
            "limits=JobLimits(committed_memory_bytes=192*1024**2), timeout=10))\n"
        )
        result = subprocess.run([sys.executable, "-c", wrapper],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "child stdout")
        self.assertEqual(result.stderr.strip(), "child stderr")

    def test_timeout_terminates_child_before_later_side_effect(self):
        with tempfile.TemporaryDirectory(prefix="winjob-") as temp:
            marker = Path(temp) / "after-timeout.txt"
            with self.assertRaises(subprocess.TimeoutExpired):
                run_local_process(
                    [sys.executable, "-c", "import time, pathlib, sys; time.sleep(2); "
                     "pathlib.Path(sys.argv[1]).touch()", marker],
                    limits=self.limits, timeout=0.2)
            self.assertFalse(marker.exists())

    def test_job_close_terminates_remaining_descendant(self):
        with tempfile.TemporaryDirectory(prefix="winjob-") as temp:
            pid_file = Path(temp) / "descendant.pid"
            script = (
                "import pathlib, subprocess, sys\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')\n"
            )
            code = run_local_process([sys.executable, "-c", script, pid_file],
                                     limits=self.limits, timeout=10)
            self.assertEqual(code, 0)
            pid = int(pid_file.read_text(encoding="ascii"))
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if not handle:
                self.assertEqual(ctypes.get_last_error(), 87)  # PID no longer exists
            else:
                try:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0,
                                     "descendant must terminate when the job is closed")
                finally:
                    kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
