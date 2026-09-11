"""Deterministic synthetic telemetry tests: no host sleeps or GPU activity."""

import math
import unittest

from glm_local.pacing import CooperativeGpuPacer, PacingConfig


class PacingTests(unittest.TestCase):
    def test_nonuniform_samples_use_previous_value_over_elapsed_interval(self):
        pacer = CooperativeGpuPacer(PacingConfig(max_sample_age_seconds=10))
        pacer.add_sample(0, 1.0)
        pacer.add_sample(1, 0.0)
        pacer.add_sample(10, 1.0)
        result = pacer.recommend(10)
        # Busy [0,1), idle [1,10). The sample at t=10 has no elapsed weight.
        self.assertAlmostEqual(result.average_utilization, 0.1)
        self.assertEqual(result.observed_seconds, 10)
        self.assertTrue(result.allow_work)
        self.assertEqual(result.delay_seconds, 0)

    def test_rolling_window_keeps_partial_previous_interval(self):
        pacer = CooperativeGpuPacer(PacingConfig(
            window_seconds=4, warmup_seconds=1, max_sample_age_seconds=10
        ))
        pacer.add_sample(0, 1)
        pacer.add_sample(3, 0)
        pacer.add_sample(6, 1)
        result = pacer.recommend(7)
        # Window [3,7): zero for 3s, one for 1s.
        self.assertAlmostEqual(result.average_utilization, 0.25)
        self.assertEqual(result.observed_seconds, 4)
        self.assertEqual(pacer.sample_count, 2)

    def test_partial_cutoff_uses_anchor_before_window(self):
        pacer = CooperativeGpuPacer(PacingConfig(
            window_seconds=4, warmup_seconds=1, max_sample_age_seconds=10
        ))
        pacer.add_sample(0, 1)
        pacer.add_sample(4, 0)
        pacer.add_sample(5, 0)
        self.assertAlmostEqual(pacer.recommend(6).average_utilization, 0.5)

    def test_busy_average_recommends_bounded_delay(self):
        pacer = CooperativeGpuPacer(PacingConfig(max_delay_seconds=0.25))
        pacer.add_sample(0, 1)
        pacer.add_sample(2, 1)
        result = pacer.recommend(2)
        self.assertEqual(result.status, "throttled")
        self.assertFalse(result.allow_work)
        self.assertEqual(result.delay_seconds, 0.25)

    def test_small_excess_is_not_rounded_to_maximum_delay(self):
        pacer = CooperativeGpuPacer()
        pacer.add_sample(0, 0.61)
        pacer.add_sample(2, 0.61)
        result = pacer.recommend(2)
        self.assertAlmostEqual(result.delay_seconds, 1.22 / 0.6 - 2)

    def test_missing_and_stale_telemetry_fail_closed(self):
        pacer = CooperativeGpuPacer()
        self.assertFalse(pacer.recommend(0).allow_work)
        pacer.add_sample(0, 0)
        result = pacer.recommend(3)
        self.assertEqual(result.status, "stale_telemetry")
        self.assertFalse(result.allow_work)
        self.assertIsNone(result.average_utilization)
        pacer.add_sample(3, None)
        result = pacer.recommend(3)
        self.assertEqual(result.status, "telemetry_unavailable")
        self.assertFalse(result.allow_work)
        self.assertEqual(result.delay_seconds, pacer.config.max_delay_seconds)

    def test_warmup_is_elapsed_time_not_sample_count(self):
        pacer = CooperativeGpuPacer()
        for index in range(100):
            pacer.add_sample(index / 100, 0)
        result = pacer.recommend(1)
        self.assertEqual(result.status, "warming_up")
        self.assertFalse(result.allow_work)
        pacer.add_sample(2, 0)
        self.assertTrue(pacer.recommend(2).allow_work)

    def test_long_gap_restarts_warmup_without_inventing_measurements(self):
        pacer = CooperativeGpuPacer()
        pacer.add_sample(0, 0)
        pacer.add_sample(2, 0)
        self.assertTrue(pacer.recommend(2).allow_work)
        pacer.add_sample(10, 0)
        result = pacer.recommend(10)
        self.assertEqual(pacer.sample_count, 1)
        self.assertEqual(result.status, "warming_up")
        self.assertEqual(result.observed_seconds, 0)
        self.assertFalse(result.allow_work)

    def test_history_is_bounded_even_with_high_frequency_sampling(self):
        pacer = CooperativeGpuPacer(PacingConfig(max_samples=3))
        for index in range(100):
            pacer.add_sample(index / 100, 0)
        self.assertEqual(pacer.sample_count, 3)
        # Evicted history cannot count toward elapsed coverage or warmup.
        result = pacer.recommend(1)
        self.assertAlmostEqual(result.observed_seconds, 0.03)
        self.assertFalse(result.allow_work)

    def test_invalid_samples_and_clock_order_raise(self):
        pacer = CooperativeGpuPacer()
        pacer.add_sample(1, 0.5)
        for value in (-0.1, 1.1, 60, math.nan, math.inf, True, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pacer.add_sample(2, value)
        for timestamp in (0, 1, math.nan, math.inf, True, "2"):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                pacer.add_sample(timestamp, 0)
        self.assertEqual(pacer.sample_count, 1)
        pacer.recommend(2)
        with self.assertRaises(ValueError):
            pacer.recommend(1.5)
        with self.assertRaises(ValueError):
            pacer.add_sample(1.5, 0)

    def test_reset_discards_history_and_allows_new_clock_origin(self):
        pacer = CooperativeGpuPacer()
        pacer.add_sample(100, 1)
        pacer.reset()
        pacer.add_sample(0, 0)
        self.assertEqual(pacer.recommend(0).status, "warming_up")

    def test_config_rejects_invalid_limits(self):
        invalid = (
            {"target_utilization": 0}, {"target_utilization": 60},
            {"window_seconds": 0}, {"warmup_seconds": 0},
            {"warmup_seconds": 11}, {"max_sample_age_seconds": 0},
            {"max_delay_seconds": -1}, {"max_delay_seconds": math.nan},
            {"max_samples": 1}, {"max_samples": 3.5}, {"max_samples": True},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PacingConfig(**values)


if __name__ == "__main__":
    unittest.main()
