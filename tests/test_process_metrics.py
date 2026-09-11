"""Validate counter semantics and small live snapshots without a stress workload."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import replace
import math
import os
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glm_local import process_metrics
from glm_local.process_metrics import ProcessSnapshot, average_cpu_percent, sample_process


def snapshot(**changes):
    values = dict(process_id=42, monotonic_seconds=10.0, working_set_bytes=100,
                  peak_working_set_bytes=200, private_commit_bytes=300,
                  peak_private_commit_bytes=400, process_cpu_seconds=2.0,
                  logical_cpu_count=8)
    values.update(changes)
    return ProcessSnapshot(**values)


class ProcessCpuMathTests(unittest.TestCase):
    def test_cpu_percent_is_normalized_across_logical_cpus(self):
        start = snapshot()
        end = replace(start, monotonic_seconds=12.0, process_cpu_seconds=6.0)
        self.assertEqual(average_cpu_percent(start, end), 25.0)

    def test_unchanged_cpu_counter_is_valid_zero(self):
        start = snapshot()
        self.assertEqual(average_cpu_percent(start, replace(start, monotonic_seconds=11)), 0)

    def test_zero_negative_and_overflowing_elapsed_are_rejected(self):
        for start_time, end_time in ((10, 10), (10, 9), (-1e308, 1e308)):
            with self.subTest(times=(start_time, end_time)), self.assertRaises(ValueError):
                average_cpu_percent(snapshot(monotonic_seconds=start_time),
                                    snapshot(monotonic_seconds=end_time))

    def test_decreasing_cpu_counter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not decrease"):
            average_cpu_percent(snapshot(), snapshot(monotonic_seconds=11,
                                                     process_cpu_seconds=1))

    def test_different_process_or_cpu_count_is_rejected(self):
        start = snapshot()
        for changes in ({"process_id": 43}, {"logical_cpu_count": 4}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                average_cpu_percent(start, replace(start, monotonic_seconds=11, **changes))

    def test_short_interval_measurement_is_not_silently_clamped(self):
        start = snapshot()
        self.assertGreater(average_cpu_percent(
            start, replace(start, monotonic_seconds=10.001, process_cpu_seconds=2.02)), 100)

    def test_nonfinite_negative_and_unknown_counters_are_rejected(self):
        invalid = {
            "monotonic_seconds": (math.nan, math.inf, True),
            "process_cpu_seconds": (math.nan, math.inf, -1, True),
            "logical_cpu_count": (None, 0, -1, 2.5, True),
            "process_id": (None, 0, -1, True),
            "working_set_bytes": (-1, 1.5, True),
            "peak_working_set_bytes": (99,),
            "private_commit_bytes": (-1,),
            "peak_private_commit_bytes": (299,),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    snapshot(**{name: value})

    def test_filetime_conversion_keeps_high_bits(self):
        ticks = (7 << 32) + 123
        self.assertEqual(process_metrics._filetime_ticks(wintypes.FILETIME(123, 7)), ticks)


class ProcessSamplingFailureTests(unittest.TestCase):
    def test_unsupported_system_fails_without_loading_apis(self):
        with patch.object(process_metrics.sys, "platform", "linux"), \
                patch.object(process_metrics, "_api") as api:
            with self.assertRaisesRegex(OSError, "require Windows"):
                sample_process()
            api.assert_not_called()

    def test_unknown_cpu_count_fails_instead_of_assuming_one(self):
        for count in (None, 0, -1):
            with self.subTest(count=count), \
                    patch.object(process_metrics.sys, "platform", "win32"), \
                    patch.object(process_metrics.os, "cpu_count", return_value=count), \
                    patch.object(process_metrics, "_api") as api:
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    sample_process()
                api.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "requires Windows error formatting")
    def test_api_failures_are_reported_instead_of_zero_counters(self):
        for failed in ("GetProcessMemoryInfo", "GetProcessTimes"):
            kernel = SimpleNamespace(GetCurrentProcess=lambda: -1,
                                     GetProcessTimes=lambda *args: failed != "GetProcessTimes")
            psapi = SimpleNamespace(
                GetProcessMemoryInfo=lambda *args: failed != "GetProcessMemoryInfo")
            with self.subTest(failed=failed), \
                    patch.object(process_metrics, "_api", return_value=(kernel, psapi)), \
                    patch.object(process_metrics.os, "cpu_count", return_value=8):
                ctypes.set_last_error(5)  # ERROR_ACCESS_DENIED
                with self.assertRaisesRegex(OSError, failed) as raised:
                    sample_process()
                self.assertEqual(raised.exception.winerror, 5)

    def test_snapshot_reads_private_usage_and_os_peak_fields(self):
        def memory_info(process, pointer, size):
            counters = ctypes.cast(pointer, ctypes.POINTER(
                process_metrics._ProcessMemoryCountersEx)).contents
            self.assertEqual(size, ctypes.sizeof(counters))
            self.assertEqual(counters.cb, size)
            counters.WorkingSetSize = 100
            counters.PeakWorkingSetSize = 200
            counters.PagefileUsage = 0  # Must read PrivateUsage, even on older Windows.
            counters.PrivateUsage = 300
            counters.PeakPagefileUsage = 400
            return True

        def process_times(process, creation, exit_time, kernel_pointer, user_pointer):
            for pointer, ticks in ((kernel_pointer, 10_000_000),
                                   (user_pointer, (1 << 32) + 20_000_000)):
                value = ctypes.cast(pointer, ctypes.POINTER(wintypes.FILETIME)).contents
                value.dwLowDateTime = ticks & 0xFFFFFFFF
                value.dwHighDateTime = ticks >> 32
            return True

        kernel = SimpleNamespace(GetCurrentProcess=lambda: -1, GetProcessTimes=process_times)
        psapi = SimpleNamespace(GetProcessMemoryInfo=memory_info)
        with patch.object(process_metrics.sys, "platform", "win32"), \
                patch.object(process_metrics, "_api", return_value=(kernel, psapi)), \
                patch.object(process_metrics.os, "cpu_count", return_value=8), \
                patch.object(process_metrics.os, "getpid", return_value=42), \
                patch.object(process_metrics.time, "monotonic", return_value=10.0):
            self.assertEqual(sample_process(), snapshot(
                process_cpu_seconds=((1 << 32) + 30_000_000) / 10_000_000))


@unittest.skipUnless(sys.platform == "win32", "requires live Windows process counters")
class LiveWindowsSnapshotTests(unittest.TestCase):
    def test_current_process_memory_cpu_and_lifetime_peaks(self):
        start = sample_process()
        time.sleep(0.02)
        end = sample_process()
        self.assertEqual(end.process_id, os.getpid())
        self.assertEqual(end.logical_cpu_count, os.cpu_count())
        self.assertGreater(end.working_set_bytes, 0)
        self.assertGreater(end.private_commit_bytes, 0)
        self.assertGreaterEqual(end.peak_working_set_bytes, end.working_set_bytes)
        self.assertGreaterEqual(end.peak_private_commit_bytes, end.private_commit_bytes)
        self.assertGreaterEqual(end.peak_working_set_bytes, start.peak_working_set_bytes)
        self.assertGreaterEqual(end.peak_private_commit_bytes, start.peak_private_commit_bytes)
        self.assertGreater(end.monotonic_seconds, start.monotonic_seconds)
        self.assertGreaterEqual(end.process_cpu_seconds, start.process_cpu_seconds)
        self.assertGreaterEqual(average_cpu_percent(start, end), 0)


if __name__ == "__main__":
    unittest.main()
