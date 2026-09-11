"""Heuristic pacing recommendations for cooperative GPU backend boundaries.

This module never sleeps, accesses a GPU, suspends a process, or interrupts a
kernel. A backend must honor ``allow_work`` at a safe boundary, collect fresh
telemetry during any delay, and ask again before submitting more work. Pacing
cannot guarantee instantaneous, rolling-average, or system-wide utilization:
other applications and work already submitted to the GPU remain uncontrolled.

Utilization is a fraction in [0, 1], not a percentage. All timestamps must use
the same monotonic clock. Samples are interpreted as a step function: the
*previous* sample applies until the next sample, subject to the freshness limit.
"""

from collections import deque
from dataclasses import dataclass
import math
from numbers import Real
from typing import Deque, Optional, Tuple


def _number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class PacingConfig:
    target_utilization: float = 0.6
    window_seconds: float = 10.0
    warmup_seconds: float = 2.0
    max_sample_age_seconds: float = 2.0
    max_delay_seconds: float = 1.0
    max_samples: int = 4096

    def __post_init__(self) -> None:
        for name in (
            "target_utilization", "window_seconds", "warmup_seconds",
            "max_sample_age_seconds", "max_delay_seconds",
        ):
            value = _number(name, getattr(self, name))
            object.__setattr__(self, name, value)
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.target_utilization > 1:
            raise ValueError("target_utilization must be at most 1")
        if self.warmup_seconds > self.window_seconds:
            raise ValueError("warmup_seconds cannot exceed window_seconds")
        if (isinstance(self.max_samples, bool)
                or not isinstance(self.max_samples, int)
                or self.max_samples < 2):
            raise ValueError("max_samples must be an integer of at least 2")


@dataclass(frozen=True)
class PacingRecommendation:
    """``allow_work=False`` means collect telemetry and recheck before work.

    ``delay_seconds`` is a bounded retry suggestion, never permission to resume
    blindly after that delay. ``average_utilization`` may cover less than the
    full window; ``observed_seconds`` reports the actual measured coverage.
    """

    status: str
    allow_work: bool
    delay_seconds: float
    average_utilization: Optional[float]
    observed_seconds: float
    reason: str


class CooperativeGpuPacer:
    """Keep bounded telemetry history and recommend heuristic idle delays."""

    def __init__(self, config: Optional[PacingConfig] = None) -> None:
        self.config = config if config is not None else PacingConfig()
        if not isinstance(self.config, PacingConfig):
            raise TypeError("config must be PacingConfig")
        self._samples: Deque[Tuple[float, float]] = deque(
            maxlen=self.config.max_samples
        )
        self._last_event_time: Optional[float] = None

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        """Discard history and restart warmup (also permits a new clock origin)."""
        self._samples.clear()
        self._last_event_time = None

    def _event_time(self, timestamp: float) -> float:
        timestamp = _number("timestamp", timestamp)
        if self._last_event_time is not None and timestamp < self._last_event_time:
            raise ValueError("timestamps must not move backwards")
        return timestamp

    def _trim(self, now: float) -> None:
        cutoff = now - self.config.window_seconds
        # Retain one sample at/before the cutoff: its value covers the interval
        # from the cutoff to the first newer sample.
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()

    def add_sample(self, timestamp: float, utilization: Optional[float]) -> None:
        """Record telemetry; None explicitly invalidates all prior telemetry.

        Missing readings and gaps longer than ``max_sample_age_seconds`` restart
        warmup. Invalid numbers raise ValueError without changing controller
        state. Sample timestamps must strictly increase; recommendations may
        share the latest sample's timestamp.
        """
        timestamp = self._event_time(timestamp)
        if self._samples and timestamp <= self._samples[-1][0]:
            raise ValueError("sample timestamps must strictly increase")
        if utilization is None:
            self._samples.clear()
            self._last_event_time = timestamp
            return
        utilization = _number("utilization", utilization)
        if not 0 <= utilization <= 1:
            raise ValueError("utilization must be a fraction between 0 and 1")
        if (self._samples and timestamp - self._samples[-1][0]
                > self.config.max_sample_age_seconds):
            self._samples.clear()
        self._samples.append((timestamp, utilization))
        self._last_event_time = timestamp
        self._trim(timestamp)

    def recommend(self, now: float) -> PacingRecommendation:
        """Return a decision only; no waiting or device/process control occurs."""
        now = self._event_time(now)
        self._last_event_time = now
        if not self._samples:
            return self._blocked(
                "telemetry_unavailable", "No valid telemetry; collect a fresh sample."
            )
        if now - self._samples[-1][0] > self.config.max_sample_age_seconds:
            return self._blocked(
                "stale_telemetry", "Telemetry is stale; collect a fresh sample."
            )
        self._trim(now)
        start = max(now - self.config.window_seconds, self._samples[0][0])
        observed = now - start
        area = 0.0
        # Left-hold integration handles uneven sampling and the partial first
        # interval at the rolling-window boundary, including the latest tail.
        previous_time, previous_value = self._samples[0]
        for sample_time, sample_value in list(self._samples)[1:]:
            area += max(0.0, sample_time - max(start, previous_time)) * previous_value
            previous_time, previous_value = sample_time, sample_value
        area += max(0.0, now - max(start, previous_time)) * previous_value
        average = min(1.0, max(0.0, area / observed)) if observed > 0 else None
        if observed < self.config.warmup_seconds:
            return self._blocked(
                "warming_up", "Collect more elapsed telemetry before submitting work.",
                average, observed,
            )
        assert average is not None
        if average <= self.config.target_utilization:
            return PacingRecommendation(
                "ready", True, 0.0, average, observed,
                "Measured average is within the heuristic target.",
            )
        # Approximate idle time that dilutes excess busy-time at the target.
        # This assumes idle work, does not predict rolling-window eviction, and
        # cannot account for other GPU users. Recheck with new telemetry.
        requested_delay = max(
            0.0, area / self.config.target_utilization - observed
        )
        delay = min(self.config.max_delay_seconds, requested_delay)
        return PacingRecommendation(
            "throttled", False, delay, average, observed,
            "Delay at a cooperative boundary, collect fresh telemetry, then recheck.",
        )

    def _blocked(
        self, status: str, reason: str,
        average: Optional[float] = None, observed: float = 0.0,
    ) -> PacingRecommendation:
        return PacingRecommendation(
            status, False, self.config.max_delay_seconds, average, observed, reason
        )
