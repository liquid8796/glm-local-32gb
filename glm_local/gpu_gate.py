"""Cooperative admission at tiny GPU kernel boundaries using device telemetry.

This is a heuristic based on sampled whole-device utilization, including other
applications. It cannot establish an instantaneous or rolling-average GPU cap.
Short kernels may be missed by the driver's own utilization sampling interval.
"""

import math
from numbers import Real
import threading
import time

from .hardware import read_gpus as _read_gpus
from .pacing import CooperativeGpuPacer, PacingConfig


POLL_SECONDS = 0.2
MAX_SAMPLE_AGE_SECONDS = 2.0
MAX_SAMPLES = 1024


def _positive(name, value, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(value) or value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be positive and at most {maximum}" if maximum is not None
                         else f"{name} must be a finite positive number")
    return value


class GpuBoundaryGate:
    """Wait for recent acceptable telemetry before the caller submits a kernel.

    The default reader uses the existing read-only nvidia-smi probe. A bounded
    daemon thread prevents its subprocess timeout from exceeding this gate's
    deadline; a timed-out read may finish in the background, but cannot admit
    work. Custom test readers run synchronously and must return promptly; their
    elapsed time is checked using the injected monotonic clock after each read.
    """

    def __init__(self, device_index=0, target=0.6, window_seconds=10,
                 max_wait_seconds=10, *, clock=None, sleep=None, read_gpus=None):
        if type(device_index) is not int or device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        self.device_index = device_index
        self.target = _positive("target", target, 1.0)
        self.window_seconds = _positive("window_seconds", window_seconds)
        self.max_wait_seconds = _positive("max_wait_seconds", max_wait_seconds, 30.0)
        self._clock = time.monotonic if clock is None else clock
        self._sleep = time.sleep if sleep is None else sleep
        self._reader = _read_gpus if read_gpus is None else read_gpus
        self._threaded_reader = read_gpus is None
        if not all(callable(value) for value in (self._clock, self._sleep, self._reader)):
            raise ValueError("clock, sleep, and read_gpus must be callable")
        self._pacer = CooperativeGpuPacer(PacingConfig(
            target_utilization=self.target, window_seconds=self.window_seconds,
            warmup_seconds=min(2.0, self.window_seconds),
            max_sample_age_seconds=MAX_SAMPLE_AGE_SECONDS,
            max_delay_seconds=POLL_SECONDS, max_samples=MAX_SAMPLES,
        ))
        self._last_clock = None
        self._last_sample_time = None
        self._last_recommendation = None
        self._failed = None
        self._samples = 0
        self._before_calls = 0
        self._admitted = 0
        self._finish_calls = 0
        self._checks = 0
        self._sleep_seconds = 0.0
        self._sleep_calls = 0
        self._telemetry_errors = 0
        self._peak_percent = None
        self._latest_percent = None

    def _now(self):
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError("Monotonic clock returned an invalid timestamp")
        value = float(value)
        if not math.isfinite(value) or (self._last_clock is not None and value < self._last_clock):
            raise RuntimeError("Monotonic clock moved backwards or became invalid")
        self._last_clock = value
        return value

    def _ensure_healthy(self):
        if self._failed is not None:
            raise RuntimeError(f"GPU admission gate already failed: {self._failed}")

    def _sleep_bounded(self, duration, deadline):
        before = self._now()
        remaining = deadline - before
        if remaining <= 0:
            raise TimeoutError("GPU admission timed out without permission to submit work")
        duration = min(POLL_SECONDS, duration, remaining)
        if duration <= 0:
            return
        self._sleep(duration)
        after = self._now()
        self._sleep_calls += 1
        self._sleep_seconds += after - before
        if after <= before:
            raise RuntimeError("Sleep did not advance the monotonic clock")
        if after > deadline:
            raise TimeoutError("GPU admission deadline expired while waiting")

    def _default_read(self, deadline, read_started):
        done = threading.Event()
        result = []

        def read():
            try:
                result.append((True, self._reader()))
            except BaseException as error:
                result.append((False, error))
            finally:
                done.set()

        threading.Thread(target=read, name="gpu-boundary-telemetry", daemon=True).start()
        while not done.is_set():
            now = self._now()
            remaining = min(deadline - now, MAX_SAMPLE_AGE_SECONDS - (now - read_started))
            if remaining <= 0:
                raise TimeoutError("GPU telemetry read exceeded its freshness or admission deadline")
            done.wait(min(POLL_SECONDS, remaining))
        success, payload = result[0]
        if not success:
            raise RuntimeError("GPU telemetry query failed") from payload
        return payload

    def _sample(self, deadline):
        if self._samples >= MAX_SAMPLES:
            raise RuntimeError(f"GPU telemetry sample limit reached ({MAX_SAMPLES})")
        started = self._now()
        if started >= deadline:
            raise TimeoutError("GPU admission deadline expired before a fresh telemetry read")
        try:
            rows = (self._default_read(deadline, started) if self._threaded_reader
                    else self._reader())
            # The sample receives its completion timestamp. Never add it after a
            # recommendation with a newer timestamp, or pretend read time was idle.
            sampled_at = self._now()
            if sampled_at > deadline:
                raise TimeoutError("GPU admission deadline expired during telemetry read")
            if sampled_at - started > MAX_SAMPLE_AGE_SECONDS:
                raise RuntimeError("GPU telemetry read is stale")
            if not isinstance(rows, (list, tuple)):
                raise RuntimeError("GPU telemetry returned an invalid device collection")
            matches = [row for row in rows if isinstance(row, dict)
                       and type(row.get("index")) is int and row["index"] == self.device_index]
            if len(matches) != 1:
                raise RuntimeError("GPU telemetry is missing the device or contains duplicates")
            percent = matches[0].get("utilization_percent")
            if (isinstance(percent, bool) or not isinstance(percent, Real)
                    or not math.isfinite(float(percent)) or not 0 <= percent <= 100):
                raise RuntimeError("GPU utilization telemetry is missing or invalid")
            percent = float(percent)
            self._pacer.add_sample(sampled_at, percent / 100.0)
        except TimeoutError:
            self._telemetry_errors += 1
            raise
        except Exception as error:
            self._telemetry_errors += 1
            raise RuntimeError(f"GPU telemetry unavailable: {error}") from error
        self._last_sample_time = sampled_at
        self._samples += 1
        self._latest_percent = percent
        self._peak_percent = percent if self._peak_percent is None else max(self._peak_percent,
                                                                          percent)

    def _recommend(self, now):
        self._checks += 1
        recommendation = self._pacer.recommend(now)
        self._last_recommendation = recommendation
        return recommendation

    def before_submit(self):
        """Return only when the recent measured average permits a tiny submission."""
        self._ensure_healthy()
        self._before_calls += 1
        try:
            deadline = self._now() + self.max_wait_seconds
            while True:
                now = self._now()
                if now > deadline:
                    raise TimeoutError("GPU admission timed out without permission to submit work")
                if (self._last_sample_time is None
                        or now - self._last_sample_time >= POLL_SECONDS):
                    self._sample(deadline)
                    now = self._now()
                decision = self._recommend(now)
                # Avoid a floating-point integration roundoff rejecting an exact
                # 60% (or configured-target) boundary. Telemetry is percentage data.
                boundary_ready = (decision.status == "throttled"
                                  and decision.average_utilization is not None
                                  and abs(decision.average_utilization - self.target) <= 1e-12)
                if decision.allow_work or boundary_ready:
                    self._admitted += 1
                    return
                if now >= deadline:
                    raise TimeoutError("GPU remained above target or lacked fresh warmup coverage")
                due_in = (POLL_SECONDS if self._last_sample_time is None else
                          max(0.0, self._last_sample_time + POLL_SECONDS - now))
                self._sleep_bounded(due_in or POLL_SECONDS, deadline)
        except Exception as error:
            self._failed = str(error)
            raise

    def finish(self):
        """Force one new post-kernel telemetry read, respecting the polling interval."""
        self._ensure_healthy()
        self._finish_calls += 1
        try:
            deadline = self._now() + self.max_wait_seconds
            if self._last_sample_time is not None:
                while True:
                    remaining = self._last_sample_time + POLL_SECONDS - self._now()
                    if remaining <= 0:
                        break
                    self._sleep_bounded(remaining, deadline)
            self._sample(deadline)
            self._recommend(self._now())
        except Exception as error:
            self._failed = str(error)
            raise

    def summary(self):
        """Return observation metrics without treating them as GPU cap verification."""
        last = self._last_recommendation
        average = None if last is None else last.average_utilization
        return {
            "device_index": self.device_index,
            "target_utilization_percent": self.target * 100,
            "window_seconds": self.window_seconds,
            "max_wait_seconds": self.max_wait_seconds,
            "warmup_seconds": self._pacer.config.warmup_seconds,
            "before_submit_calls": self._before_calls,
            "admitted_submissions": self._admitted,
            "finish_calls": self._finish_calls,
            "checks": self._checks,
            "telemetry_samples": self._samples,
            "retained_samples": self._pacer.sample_count,
            "telemetry_errors": self._telemetry_errors,
            "sleep_calls": self._sleep_calls,
            "sleep_seconds": self._sleep_seconds,
            "sampled_gpu_peak_percent": self._peak_percent,
            "latest_sample_utilization_percent": self._latest_percent,
            "latest_weighted_average_percent": None if average is None else average * 100,
            "observed_seconds": 0 if last is None else last.observed_seconds,
            "status": "failed" if self._failed else "unstarted" if last is None else last.status,
            "failure": self._failed,
            "telemetry_scope": "whole_device",
            "heuristic_only": True,
            "gpu_cap_verified": False,
            "limitations": "Driver samples may miss short workloads; other GPU users are included.",
        }
