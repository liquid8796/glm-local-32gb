"""Run a local Windows executable under a CPU and committed-memory quota.

The quota belongs to this job and its descendants. It is not a system-wide
CPU limit, a physical RAM/RSS limit, or a GPU limit. Ancestor jobs may impose
additional restrictions; a nested CPU quota is relative to its parent quota.
Closing the job terminates any remaining descendants, including after the
initial process exits. No shell, model, network or download integration exists.

Reference: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Iterator, Sequence


@dataclass(frozen=True)
class JobLimits:
    """Requested job CPU percentage and aggregate committed-memory bytes."""

    cpu_percent: int = 70
    committed_memory_bytes: int = 32_000_000_000

    def __post_init__(self) -> None:
        if type(self.cpu_percent) is not int or not 1 <= self.cpu_percent <= 100:
            raise ValueError("cpu_percent must be an integer from 1 to 100")
        size_max = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1
        if (type(self.committed_memory_bytes) is not int
                or not 1 <= self.committed_memory_bytes <= size_max):
            raise ValueError("committed_memory_bytes must be a positive SIZE_T integer")


@dataclass(frozen=True)
class InstalledLimits:
    """Values queried from this job, excluding any ancestor-job restrictions."""

    cpu_percent: float
    committed_memory_bytes: int
    cpu_hard_cap: bool
    memory_limit_enabled: bool
    kill_on_close: bool


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IOCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _CpuLimits(ctypes.Structure):
    # CpuRate is the DWORD member of the native union when HARD_CAP is set.
    _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD),
    ]


_JOB_EXTENDED_LIMITS = 9
_JOB_CPU_RATE_CONTROL = 15
_JOB_MEMORY = 0x200
_KILL_ON_CLOSE = 0x2000
_CPU_ENABLE_HARD_CAP = 0x1 | 0x4
_CREATE_SUSPENDED = 0x4
_EXTENDED_STARTUPINFO_PRESENT = 0x80000
_HANDLE_LIST_ATTRIBUTE = 0x20002
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_WAIT_FAILED = 0xFFFFFFFF
_INVALID_HANDLE = ctypes.c_void_p(-1).value


@lru_cache(maxsize=1)
def _api():
    if sys.platform != "win32":
        raise OSError("Windows Job Objects require Windows 8 / Server 2012 or later")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, dword, boolean = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    signatures = {
        "CreateJobObjectW": ([ptr, wintypes.LPCWSTR], handle),
        "SetInformationJobObject": ([handle, ctypes.c_int, ptr, dword], boolean),
        "QueryInformationJobObject": ([handle, ctypes.c_int, ptr, dword, ptr], boolean),
        "AssignProcessToJobObject": ([handle, handle], boolean),
        "CloseHandle": ([handle], boolean),
        "GetCurrentProcess": ([], handle),
        "GetStdHandle": ([dword], handle),
        "DuplicateHandle": ([handle, handle, handle, ctypes.POINTER(handle),
                             dword, boolean, dword], boolean),
        "InitializeProcThreadAttributeList": ([ptr, dword, dword, ctypes.POINTER(size)], boolean),
        "UpdateProcThreadAttribute": ([ptr, dword, size, ptr, size, ptr, ptr], boolean),
        "DeleteProcThreadAttributeList": ([ptr], None),
        "CreateProcessW": ([wintypes.LPCWSTR, wintypes.LPWSTR, ptr, ptr, boolean,
                            dword, ptr, wintypes.LPCWSTR, ctypes.POINTER(_StartupInfoEx),
                            ctypes.POINTER(_ProcessInfo)], boolean),
        "ResumeThread": ([handle], dword),
        "WaitForSingleObject": ([handle, dword], dword),
        "GetExitCodeProcess": ([handle, ctypes.POINTER(dword)], boolean),
        "TerminateProcess": ([handle, wintypes.UINT], boolean),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(kernel, name)
        fn.argtypes, fn.restype = args, result
    return kernel


def _check(success, operation: str) -> None:
    if not success:
        error = ctypes.get_last_error()
        raise OSError(error, f"{operation}: {ctypes.FormatError(error).strip()}", None, error)


class WindowsJob:
    """An owned, non-inheritable job handle; use as a context manager."""

    def __init__(self, limits: JobLimits | None = None) -> None:
        self._kernel = _api()
        limits = limits or JobLimits()
        self._handle = self._kernel.CreateJobObjectW(None, None)
        _check(self._handle, "CreateJobObjectW")
        try:
            extended = _ExtendedLimits()
            extended.BasicLimitInformation.LimitFlags = _JOB_MEMORY | _KILL_ON_CLOSE
            extended.JobMemoryLimit = limits.committed_memory_bytes
            _check(self._kernel.SetInformationJobObject(
                self._handle, _JOB_EXTENDED_LIMITS, ctypes.byref(extended), ctypes.sizeof(extended)
            ), "SetInformationJobObject(memory)")
            cpu = _CpuLimits(_CPU_ENABLE_HARD_CAP, limits.cpu_percent * 100)
            _check(self._kernel.SetInformationJobObject(
                self._handle, _JOB_CPU_RATE_CONTROL, ctypes.byref(cpu), ctypes.sizeof(cpu)
            ), "SetInformationJobObject(CPU)")
            installed = self.query_limits()
            if (installed.cpu_percent != limits.cpu_percent
                    or installed.committed_memory_bytes != limits.committed_memory_bytes
                    or not all((installed.cpu_hard_cap, installed.memory_limit_enabled,
                                installed.kill_on_close))):
                raise RuntimeError("Windows did not install the requested job policy")
        except BaseException:
            self.close()
            raise

    def _require_handle(self):
        if not self._handle:
            raise RuntimeError("WindowsJob is closed")
        return self._handle

    def query_limits(self) -> InstalledLimits:
        """Read back the installed policy through QueryInformationJobObject."""
        handle = self._require_handle()
        cpu, extended = _CpuLimits(), _ExtendedLimits()
        for info_class, info in ((_JOB_CPU_RATE_CONTROL, cpu), (_JOB_EXTENDED_LIMITS, extended)):
            _check(self._kernel.QueryInformationJobObject(
                handle, info_class, ctypes.byref(info), ctypes.sizeof(info), None
            ), "QueryInformationJobObject")
        flags = extended.BasicLimitInformation.LimitFlags
        return InstalledLimits(
            cpu_percent=cpu.CpuRate / 100,
            committed_memory_bytes=extended.JobMemoryLimit,
            cpu_hard_cap=(cpu.ControlFlags & _CPU_ENABLE_HARD_CAP) == _CPU_ENABLE_HARD_CAP,
            memory_limit_enabled=bool(flags & _JOB_MEMORY),
            kill_on_close=bool(flags & _KILL_ON_CLOSE),
        )

    def close(self) -> None:
        if self._handle:
            _check(self._kernel.CloseHandle(self._handle), "CloseHandle(job)")
            self._handle = None

    def __enter__(self) -> WindowsJob:
        self._require_handle()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@contextmanager
def _standard_handles(kernel) -> Iterator[_StartupInfoEx]:
    """Duplicate stdio without changing parent handles; inherit only these three."""
    import msvcrt

    duplicates = []
    attribute_list = None
    try:
        current = kernel.GetCurrentProcess()
        for std_number, mode in ((-10, "rb"), (-11, "wb"), (-12, "wb")):
            source = kernel.GetStdHandle(std_number & 0xFFFFFFFF)
            fallback = None
            try:
                if not source or source == _INVALID_HANDLE:
                    fallback = open(os.devnull, mode)
                    source = msvcrt.get_osfhandle(fallback.fileno())
                duplicate = wintypes.HANDLE()
                _check(kernel.DuplicateHandle(
                    current, source, current, ctypes.byref(duplicate), 0, True, 2
                ), "DuplicateHandle(stdio)")
                duplicates.append(duplicate.value)
            finally:
                if fallback is not None:
                    fallback.close()
        size = ctypes.c_size_t()
        kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        if not size.value:
            _check(False, "InitializeProcThreadAttributeList(size)")
        storage = ctypes.create_string_buffer(size.value)
        _check(kernel.InitializeProcThreadAttributeList(
            storage, 1, 0, ctypes.byref(size)
        ), "InitializeProcThreadAttributeList")
        attribute_list = ctypes.cast(storage, ctypes.c_void_p)
        inherited = (wintypes.HANDLE * len(duplicates))(*duplicates)
        _check(kernel.UpdateProcThreadAttribute(
            attribute_list, 0, _HANDLE_LIST_ATTRIBUTE, inherited,
            ctypes.sizeof(inherited), None, None
        ), "UpdateProcThreadAttribute(handle list)")
        startup = _StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = 0x100  # STARTF_USESTDHANDLES
        (startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput,
         startup.StartupInfo.hStdError) = duplicates
        startup.lpAttributeList = attribute_list
        yield startup
    finally:
        if attribute_list:
            kernel.DeleteProcThreadAttributeList(attribute_list)
        for handle in duplicates:
            kernel.CloseHandle(handle)


def _command(args: Sequence[str | os.PathLike[str]], cwd: str | None) -> list[str]:
    if isinstance(args, (str, bytes)) or not args:
        raise ValueError("args must be a non-empty sequence, not a command string")
    command = [os.fspath(arg) for arg in args]
    if any(not isinstance(arg, str) or "\0" in arg for arg in command) or not command[0]:
        raise ValueError("arguments must be Unicode strings without NUL characters")
    executable = command[0]
    if os.path.dirname(executable):
        if not os.path.isabs(executable):
            executable = os.path.join(cwd or os.getcwd(), executable)
    else:
        executable = shutil.which(executable)
    if not executable or not os.path.isfile(executable):
        raise FileNotFoundError(f"Local executable does not exist: {command[0]}")
    if os.path.splitext(executable)[1].lower() in (".bat", ".cmd"):
        raise ValueError("Use a native executable; batch files require a shell")
    command[0] = os.path.abspath(executable)
    return command


def run_local_process(
    args: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    limits: JobLimits | None = None,
    on_policy: Callable[[InstalledLimits], None] | None = None,
    timeout: float | None = None,
) -> int:
    """Run without a shell and return the unsigned Windows process exit code.

    Stdin/stdout/stderr remain connected to the caller's Windows standard handles.
    Policy is installed before creation; the process is created suspended, assigned
    to the job, and resumed only after policy readback and optional on_policy.
    Any exception (including KeyboardInterrupt or timeout) cleans up the process
    tree. A failure to install or assign a job aborts without running uncontained.
    The callback runs before child code and must return promptly. CPU quotas are
    scheduling-interval limits and do not guarantee Task Manager sample readings.
    """
    kernel = _api()
    if timeout is not None and (not isinstance(timeout, (int, float))
                                or not 0 < timeout < float("inf")):
        raise ValueError("timeout must be a positive finite number")
    directory = os.fspath(cwd) if cwd is not None else None
    if directory is not None:
        if not isinstance(directory, str) or "\0" in directory:
            raise ValueError("cwd must be a Unicode path without NUL characters")
        directory = os.path.abspath(directory)
    command = _command(args, directory)
    mutable_command = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
    process = _ProcessInfo()
    finished = False
    job = WindowsJob(limits)
    try:
        with _standard_handles(kernel) as startup:
            _check(kernel.CreateProcessW(
                command[0], mutable_command, None, None, True,
                _CREATE_SUSPENDED | _EXTENDED_STARTUPINFO_PRESENT,
                None, directory, ctypes.byref(startup), ctypes.byref(process),
            ), "CreateProcessW")
        _check(kernel.AssignProcessToJobObject(job._require_handle(), process.hProcess),
               "AssignProcessToJobObject")
        policy = job.query_limits()
        if on_policy is not None:
            on_policy(policy)
        if kernel.ResumeThread(process.hThread) == 0xFFFFFFFF:
            _check(False, "ResumeThread")
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            status = kernel.WaitForSingleObject(process.hProcess, 100)
            if status == _WAIT_OBJECT_0:
                finished = True
                code = wintypes.DWORD()
                _check(kernel.GetExitCodeProcess(process.hProcess, ctypes.byref(code)),
                       "GetExitCodeProcess")
                return code.value
            if status == _WAIT_FAILED:
                _check(False, "WaitForSingleObject")
            if status != _WAIT_TIMEOUT:
                raise RuntimeError(f"Unexpected process wait result: {status}")
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
    finally:
        # TerminateProcess also covers errors before assignment, when kill-on-close
        # cannot yet protect the suspended process. Never leave it running on error.
        if process.hProcess and not finished:
            kernel.TerminateProcess(process.hProcess, 1223)  # ERROR_CANCELLED
        try:
            job.close()
        finally:
            if process.hProcess:
                kernel.WaitForSingleObject(process.hProcess, 5000)
            for handle in (process.hThread, process.hProcess):
                if handle:
                    kernel.CloseHandle(handle)
