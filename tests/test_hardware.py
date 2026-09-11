"""Hardware probes are mocked; CSV missing data must never become measured zero."""

import json
import subprocess
import unittest
from unittest.mock import patch

from glm_local import hardware


class NvidiaCsvTests(unittest.TestCase):
    def test_multiple_gpus_and_byte_conversion(self):
        rows = ('0,"Mock GPU, board", 8192, 4096, 0, 8.6, 600.00\n'
                '1, Second GPU, 16384, 12000, 45, 9.0, 600.00\n')
        result = hardware.parse_nvidia_csv(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "Mock GPU, board")
        self.assertEqual(result[0]["memory_total_bytes"], 8192 * 1024**2)
        self.assertEqual(result[0]["memory_free_bytes"], 4096 * 1024**2)
        self.assertEqual(result[0]["utilization_percent"], 0)
        self.assertEqual(result[1]["index"], 1)
        self.assertEqual(result[1]["utilization_percent"], 45)

    def test_na_and_nonfinite_telemetry_remain_unknown(self):
        for missing in ("N/A", "[N/A]", "[Not Supported]", "", "nan", "inf", "-inf"):
            row = f"0, Mock GPU, {missing}, {missing}, {missing}, N/A, 600.00\n"
            result = hardware.parse_nvidia_csv(row)[0]
            for key in ("memory_total_bytes", "memory_free_bytes", "utilization_percent"):
                with self.subTest(value=missing, key=key):
                    self.assertIsNone(result[key])

    def test_unexpected_csv_schema_or_invalid_index_fails(self):
        for row in ("0, GPU, 8192", "0, GPU, 8192, 4096, 1, 8.6, 600.00, extra",
                    "N/A, GPU, 8192, 4096, 1, 8.6, 600.00"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                hardware.parse_nvidia_csv(row)

    def test_empty_output_is_no_detected_gpus(self):
        self.assertEqual(hardware.parse_nvidia_csv("\n\n"), [])


class HardwareProbeTests(unittest.TestCase):
    def test_missing_nvidia_tool_does_not_start_a_subprocess(self):
        with patch.object(hardware.shutil, "which", return_value=None), \
                patch.object(hardware.subprocess, "run") as run, \
                self.assertRaisesRegex(RuntimeError, "unavailable"):
            hardware.read_gpus()
        run.assert_not_called()

    def test_gpu_probe_reads_bounded_cli_output(self):
        response = subprocess.CompletedProcess([], 0, stdout="0, Mock GPU, N/A, N/A, N/A, N/A, 600.00\n")
        with patch.object(hardware.shutil, "which", return_value="C:/mock/nvidia-smi.exe"), \
                patch.object(hardware.subprocess, "run", return_value=response) as run:
            result = hardware.read_gpus()
        self.assertIsNone(result[0]["utilization_percent"])
        self.assertGreater(run.call_args.kwargs["timeout"], 0)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_windows_probe_combines_cpu_disk_and_gpu_without_live_calls(self):
        system = {"cpu_name": "Mock CPU", "physical_cores": 4,
                  "physical_memory_bytes": 32 * 1024**3, "available_memory_bytes": 20 * 1024**3,
                  "disks": [{"root": "D:\\", "total_bytes": 1000, "free_bytes": 500}]}
        response = subprocess.CompletedProcess([], 0, stdout=json.dumps(system).encode("utf-8-sig"))
        gpu = [{"index": 0, "name": "Mock GPU", "utilization_percent": None}]
        with patch.object(hardware.os, "name", "nt"), \
                patch.object(hardware.os, "cpu_count", return_value=8), \
                patch.object(hardware.subprocess, "run", return_value=response), \
                patch.object(hardware, "read_gpus", return_value=gpu):
            result = hardware.detect_hardware()
        self.assertEqual(result["logical_processors"], 8)
        self.assertEqual(result["cpu_name"], "Mock CPU")
        self.assertEqual(result["disks"], system["disks"])
        self.assertEqual(result["gpus"], gpu)
        self.assertEqual(result["errors"], [])

    def test_probe_failures_are_errors_not_fabricated_hardware(self):
        with patch.object(hardware.os, "name", "nt"), \
                patch.object(hardware.subprocess, "run", side_effect=subprocess.TimeoutExpired("mock", 1)), \
                patch.object(hardware, "read_gpus", side_effect=RuntimeError("unavailable")):
            result = hardware.detect_hardware()
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(result["gpus"], [])
        self.assertNotIn("physical_memory_bytes", result)

    def test_unsupported_platform_does_not_run_probe(self):
        with patch.object(hardware.os, "name", "posix"), \
                patch.object(hardware.subprocess, "run") as run, \
                patch.object(hardware, "read_gpus") as read:
            result = hardware.detect_hardware()
        self.assertTrue(result["errors"])
        run.assert_not_called()
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
