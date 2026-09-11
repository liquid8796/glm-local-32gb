"""Read-only memory and CPU counters for this Windows process.

Working set is resident memory (RSS), while private commit includes committed
private memory that need not be resident. Both peak counters are OS-recorded
process-lifetime peaks, including work before the first sample. These counters
do not measure system-wide use, child processes, a Windows job, or GPU use, and
cannot establish that a resource cap was enforced.

Win32 counter definitions:
https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import sys
import time


@dataclass(frozen=True)
class ProcessSnapshot:
    """A current-process snapshot; byte counters use bytes, CPU uses seconds."""

    process_id: int
    monotonic_seconds: float
    working_set_bytes: int
    peak_working_set_bytes: int
    private_commit_bytes: int
    peak_private_commit_bytes: int
    process_cpu_seconds: float
    logical_cpu_count: int

    def __post_init__(self) -> None:
        for name in ("process_id", "logical_cpu_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a known positive integer")
        for name in ("working_set_bytes", "peak_working_set_bytes",
                     "private_commit_bytes", "peak_private_commit_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("monotonic_seconds", "process_cpu_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.process_cpu_seconds < 0:
            raise ValueError("process_cpu_seconds must be nonnegative")
        if self.peak_working_set_bytes < self.working_set_bytes:
            raise ValueError("peak working set must be at least current working set")
        if self.peak_private_commit_bytes < self.private_commit_bytes:
            raise ValueError("peak private commit must be at least current private commit")


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


@lru_cache(maxsize=1)
def _api():
    if sys.platform != "win32":
        raise OSError("Process resource snapshots require Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)
    ] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCountersEx), wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    return kernel, psapi


def _check(success, operation: str) -> None:
    if not success:
        error = ctypes.get_last_error()
        raise OSError(error, f"{operation}: {ctypes.FormatError(error).strip()}", None, error)


def _filetime_ticks(value: wintypes.FILETIME) -> int:
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


def sample_process() -> ProcessSnapshot:
    """Read this process through GetProcessMemoryInfo and GetProcessTimes.

    The monotonic timestamp is taken immediately after reading CPU counters;
    memory and time calls are sequential, not an atomic sample. API failures,
    unsupported systems, and an unknown logical CPU count fail explicitly.
    The current-process pseudo-handle is borrowed and must not be closed.
    """
    if sys.platform != "win32":
        raise OSError("Process resource snapshots require Windows")
    cpu_count = os.cpu_count()
    if type(cpu_count) is not int or cpu_count <= 0:
        raise RuntimeError("Logical CPU count is unavailable; cannot normalize CPU usage")
    kernel, psapi = _api()
    process = kernel.GetCurrentProcess()
    memory = _ProcessMemoryCountersEx()
    memory.cb = ctypes.sizeof(memory)
    _check(psapi.GetProcessMemoryInfo(process, ctypes.byref(memory), memory.cb),
           "GetProcessMemoryInfo")
    creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
    _check(kernel.GetProcessTimes(process, ctypes.byref(creation), ctypes.byref(exit_time),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)),
           "GetProcessTimes")
    sampled_at = time.monotonic()
    return ProcessSnapshot(
        process_id=os.getpid(),
        monotonic_seconds=sampled_at,
        working_set_bytes=memory.WorkingSetSize,
        peak_working_set_bytes=memory.PeakWorkingSetSize,
        private_commit_bytes=memory.PrivateUsage,
        peak_private_commit_bytes=memory.PeakPagefileUsage,
        process_cpu_seconds=(_filetime_ticks(kernel_time) + _filetime_ticks(user_time))
        / 10_000_000,
        logical_cpu_count=cpu_count,
    )


def average_cpu_percent(start: ProcessSnapshot, end: ProcessSnapshot) -> float:
    """Process CPU seconds / elapsed seconds / logical CPU count * 100.

    100 means all reported logical CPUs occupied by this process on average.
    This does not measure utilization of the whole machine or enforce a cap.
    Short intervals are noisy because OS CPU counters have finite resolution;
    results are not clamped, so brief samples can read above 100. Use snapshots
    from the same live process; PID equality cannot detect reuse after an exit.
    """
    if not isinstance(start, ProcessSnapshot) or not isinstance(end, ProcessSnapshot):
        raise TypeError("start and end must be ProcessSnapshot instances")
    if start.process_id != end.process_id:
        raise ValueError("CPU snapshots must belong to the same process")
    if start.logical_cpu_count != end.logical_cpu_count:
        raise ValueError("Logical CPU count changed between snapshots")
    elapsed = end.monotonic_seconds - start.monotonic_seconds
    cpu_delta = end.process_cpu_seconds - start.process_cpu_seconds
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Elapsed monotonic time must be positive and finite")
    if not math.isfinite(cpu_delta) or cpu_delta < 0:
        raise ValueError("Process CPU counter must not decrease or become nonfinite")
    percentage = cpu_delta / elapsed / start.logical_cpu_count * 100
    if not math.isfinite(percentage):
        raise ValueError("CPU percentage is nonfinite; use a longer sampling interval")
    return percentage
