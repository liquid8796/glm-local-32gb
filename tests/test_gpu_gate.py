"""Admission-gate tests use synthetic telemetry and a clock with no wall sleeps."""

import math
import unittest
from unittest.mock import patch

from glm_local.gpu_gate import GpuBoundaryGate, MAX_SAMPLES


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class GateTests(unittest.TestCase):
    def gate(self, reader=None, percent=0, **kwargs):
        clock = FakeClock()
        query_times = []

        def read():
            query_times.append(clock.now)
            return ([{"index": 0, "utilization_percent": percent}] if reader is None
                    else reader(clock))

        gate = GpuBoundaryGate(clock=clock, sleep=clock.sleep, read_gpus=read, **kwargs)
        return gate, clock, query_times

    def assert_bounded_sleep_and_polling(self, clock, query_times):
        self.assertTrue(all(0 < duration <= 0.2 for duration in clock.sleeps))
        self.assertTrue(all(right - left >= 0.2 - 1e-12
                            for left, right in zip(query_times, query_times[1:])))

    def test_zero_is_valid_and_initial_warmup_uses_elapsed_time(self):
        gate, clock, query_times = self.gate()
        gate.before_submit()
        summary = gate.summary()
        self.assertGreaterEqual(clock.now, 2.0)
        self.assertLess(clock.now, 2.5)
        self.assertEqual(summary["admitted_submissions"], 1)
        self.assertEqual(summary["sampled_gpu_peak_percent"], 0.0)
        self.assertEqual(summary["latest_weighted_average_percent"], 0.0)
        self.assertEqual(summary["telemetry_scope"], "whole_device")
        self.assertTrue(summary["heuristic_only"])
        self.assertFalse(summary["gpu_cap_verified"])
        self.assert_bounded_sleep_and_polling(clock, query_times)

    def test_sixty_percent_boundary_is_inclusive_and_short_window_reduces_warmup(self):
        gate, clock, query_times = self.gate(percent=60, window_seconds=0.6)
        gate.before_submit()
        self.assertGreaterEqual(clock.now, 0.6)
        self.assertLess(clock.now, 1.0)
        self.assertAlmostEqual(gate.summary()["latest_weighted_average_percent"], 60)
        self.assertEqual(gate.summary()["admitted_submissions"], 1)
        self.assert_bounded_sleep_and_polling(clock, query_times)

    def test_recent_admission_is_cached_but_finish_forces_new_read(self):
        gate, clock, query_times = self.gate()
        gate.before_submit()
        samples = gate.summary()["telemetry_samples"]
        gate.before_submit()
        self.assertEqual(gate.summary()["telemetry_samples"], samples)
        gate.finish()
        self.assertEqual(gate.summary()["telemetry_samples"], samples + 1)
        self.assertEqual(gate.summary()["finish_calls"], 1)
        self.assert_bounded_sleep_and_polling(clock, query_times)

    def test_busy_device_times_out_without_any_admitted_submission(self):
        gate, clock, query_times = self.gate(percent=90, max_wait_seconds=10)
        with self.assertRaises(TimeoutError):
            gate.before_submit()
        self.assertLessEqual(clock.now, 10)
        self.assertEqual(gate.summary()["admitted_submissions"], 0)
        self.assertEqual(gate.summary()["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "already failed"):
            gate.before_submit()
        self.assert_bounded_sleep_and_polling(clock, query_times)

    def test_gate_rechecks_fresh_readings_instead_of_resuming_after_delay(self):
        def reader(clock):
            return [{"index": 0, "utilization_percent": 100 if clock.now < 3 else 0}]

        gate, clock, query_times = self.gate(reader=reader)
        gate.before_submit()
        self.assertGreaterEqual(clock.now, 5.0)
        self.assertLessEqual(gate.summary()["latest_weighted_average_percent"], 60 + 1e-10)
        self.assertGreater(gate.summary()["telemetry_samples"], 20)
        self.assertEqual(gate.summary()["sampled_gpu_peak_percent"], 100)
        self.assert_bounded_sleep_and_polling(clock, query_times)

    def test_errors_missing_duplicate_and_invalid_readings_fail_closed(self):
        invalid = (None, {}, [], [{"index": 1, "utilization_percent": 0}],
                   [{"index": True, "utilization_percent": 0}],
                   [{"index": 0}], [{"index": 0, "utilization_percent": None}],
                   [{"index": 0, "utilization_percent": 0}] * 2)
        for value in invalid:
            with self.subTest(value=value):
                gate, clock, query_times = self.gate(reader=lambda clock: value)
                with self.assertRaises(RuntimeError):
                    gate.before_submit()
                self.assertEqual(gate.summary()["admitted_submissions"], 0)
                self.assertEqual(gate.summary()["telemetry_errors"], 1)
        for value in (math.nan, math.inf, -1, 101, "0", True):
            with self.subTest(percent=value):
                gate, clock, query_times = self.gate(percent=value)
                with self.assertRaises(RuntimeError):
                    gate.before_submit()
                self.assertEqual(gate.summary()["admitted_submissions"], 0)

        def failed_reader(clock):
            raise OSError("query failed")

        gate, clock, query_times = self.gate(reader=failed_reader)
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            gate.before_submit()
        self.assertEqual(gate.summary()["telemetry_errors"], 1)

    def test_slow_telemetry_is_timestamped_after_read_and_stale_read_fails(self):
        def reader(clock):
            clock.now += 0.1
            return [{"index": 0, "utilization_percent": 0}]

        gate, clock, query_times = self.gate(reader=reader)
        gate.before_submit()
        gate.finish()
        self.assertEqual(gate._last_sample_time, clock.now)
        self.assertGreaterEqual(gate.summary()["observed_seconds"], 2)
        self.assert_bounded_sleep_and_polling(clock, query_times)

        def stale_reader(clock):
            clock.now += 2.1
            return [{"index": 0, "utilization_percent": 0}]

        gate, clock, query_times = self.gate(reader=stale_reader)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            gate.before_submit()
        self.assertEqual(gate.summary()["admitted_submissions"], 0)

    def test_long_gap_restarts_warmup(self):
        gate, clock, query_times = self.gate()
        gate.before_submit()
        clock.now += 5
        restarted_at = clock.now
        gate.before_submit()
        self.assertGreaterEqual(clock.now - restarted_at, 2)
        self.assertEqual(gate.summary()["admitted_submissions"], 2)

    def test_finish_requires_fresh_valid_reading(self):
        def reader(clock):
            return [{"index": 0, "utilization_percent": 0 if clock.now < 2.3 else None}]

        gate, clock, query_times = self.gate(reader=reader)
        gate.before_submit()
        clock.now += 0.5
        with self.assertRaises(RuntimeError):
            gate.finish()
        self.assertEqual(gate.summary()["status"], "failed")
        self.assertEqual(gate.summary()["finish_calls"], 1)

    def test_deadline_and_total_sample_count_are_bounded(self):
        gate, clock, query_times = self.gate(max_wait_seconds=0.5)
        with self.assertRaises(TimeoutError):
            gate.before_submit()
        self.assertLessEqual(clock.now, 0.5)
        self.assertEqual(gate.summary()["admitted_submissions"], 0)
        self.assert_bounded_sleep_and_polling(clock, query_times)
        gate, clock, query_times = self.gate()
        gate._samples = MAX_SAMPLES
        with self.assertRaisesRegex(RuntimeError, "sample limit"):
            gate.before_submit()
        self.assertEqual(len(query_times), 0)

    def test_clock_regression_and_nonadvancing_sleep_fail(self):
        gate, clock, query_times = self.gate()
        gate.before_submit()
        clock.now -= 1
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            gate.before_submit()
        gate, clock, query_times = self.gate()
        gate._sleep = lambda seconds: None
        with self.assertRaisesRegex(RuntimeError, "did not advance"):
            gate.before_submit()

    def test_default_reader_wrapper_propagates_errors_without_admitting_work(self):
        class InlineThread:
            def __init__(self, *, target, name, daemon):
                self.target = target

            def start(self):
                self.target()

        def failed_reader(clock):
            raise OSError("reader unavailable")

        gate, clock, query_times = self.gate(reader=failed_reader)
        gate._threaded_reader = True
        with patch("glm_local.gpu_gate.threading.Thread", InlineThread):
            with self.assertRaisesRegex(RuntimeError, "telemetry query failed"):
                gate.before_submit()
        self.assertEqual(gate.summary()["admitted_submissions"], 0)

    def test_default_reader_wrapper_has_bounded_wait_even_if_query_does_not_finish(self):
        gate, clock, query_times = self.gate(max_wait_seconds=0.5)
        gate._threaded_reader = True

        class PendingEvent:
            def is_set(self):
                return False

            def wait(self, seconds):
                clock.sleep(seconds)

        with patch("glm_local.gpu_gate.threading.Thread"), \
                patch("glm_local.gpu_gate.threading.Event", PendingEvent):
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                gate.before_submit()
        self.assertEqual(clock.now, 0.5)
        self.assertEqual(gate.summary()["admitted_submissions"], 0)
        self.assertEqual(gate.summary()["telemetry_errors"], 1)
        self.assert_bounded_sleep_and_polling(clock, query_times)

    def test_invalid_configuration(self):
        for kwargs in ({"device_index": True}, {"device_index": -1}, {"target": 60},
                       {"target": 0}, {"target": math.nan}, {"window_seconds": 0},
                       {"max_wait_seconds": 31}, {"max_wait_seconds": 0},
                       {"max_wait_seconds": True}, {"clock": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GpuBoundaryGate(**kwargs)


if __name__ == "__main__":
    unittest.main()
