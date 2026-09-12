"""Bounded stage diagnostics; never generated-text streaming or model evidence.

A single daemon heartbeat retains only the latest stage. Updates are atomic,
limited to one small JSON file, and throttled independently from stage changes.
The parent can recover the last complete record after its existing Job timeout
has terminated the worker tree. Reporting I/O failures do not change the math.
"""
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import stat
import threading
import time

from .checkpoint_http import strict_json
from .checkpoint_snapshot import write_json
from .execution import FULL_MODEL_FLAGS

MAX_PROGRESS_BYTES = 16 * 1024
_TEXT = {"phase": 32, "tensor_name": 512, "backend": 32}
_COUNTS = {"position", "token_index", "token_count", "prompt_tokens", "generated_tokens",
           "requested_new_tokens", "layer_index", "layer_count", "rows", "cols"}
_TERMINAL = {"ERROR", "INTERRUPTED", "PASS", "ESTIMATE_FITS", "GENERATED_UNVERIFIED", "NUMERICAL_MISMATCH"}


def _text(value, maximum):
    try:
        return isinstance(value, str) and 1 <= len(value.encode("utf-8")) <= maximum and not any(ord(c) < 32 for c in value)
    except UnicodeError:
        return False


class RuntimeProgress:
    def __init__(self, directory, settings, action, *, interval=1.0, stdout_interval=5.0,
                 heartbeat=True, clock=time.monotonic):
        if (type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0
                or type(stdout_interval) not in (int, float) or not math.isfinite(stdout_interval) or stdout_interval <= 0):
            raise ValueError("Progress intervals must be positive finite seconds")
        self.directory = Path(directory).resolve()
        if (not _text(str(self.directory), 4096) or not _text(settings.get("model_id"), 512)
                or not _text(settings.get("revision"), 128) or not _text(action, 32)):
            raise ValueError("Progress identity or run path exceeds its text bounds")
        self.path = self.directory / "progress.json"
        self._clock, self._interval, self._stdout_interval = clock, interval, stdout_interval
        self._started = self._stage_started = clock()
        self._last_write = self._last_stdout = float("-inf")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._state = {"format_version": 1, "scope": "runtime_stage_progress", "diagnostic_only": True,
            "model_id": settings["model_id"], "revision": settings["revision"], "action": action,
            "run_directory": str(self.directory), "pid": os.getpid(), "sequence": 0,
            "stage": "starting", "status": "RUNNING", **FULL_MODEL_FLAGS}
        self._io_error = None
        self._stdout_broken = False
        with self._lock:
            self._publish(force=True)
        if heartbeat:
            self._thread = threading.Thread(target=self._heartbeat, name="runtime-stage-progress", daemon=True)
            self._thread.start()

    def __call__(self, stage, **fields):
        if not _text(stage, 80):
            raise ValueError("Progress stage must be bounded printable text")
        for key, value in fields.items():
            if key in _TEXT:
                if not _text(value, _TEXT[key]):
                    raise ValueError("Progress text field is invalid")
            elif key not in _COUNTS or type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise ValueError("Progress fields must be known bounded counters or labels")
        with self._lock:
            if self._stop.is_set():
                return
            old_phase = self._state.get("phase")
            self._stage_started = self._clock()
            if not stage.startswith("projection"):
                for key in ("tensor_name", "rows", "cols"):
                    self._state.pop(key, None)
            if "phase" in fields:
                for key in ("position", "layer_index", "layer_count", "token_index", "token_count"):
                    self._state.pop(key, None)
            if stage in ("output_head", "token_complete"):
                self._state.pop("layer_index", None)
                self._state.pop("layer_count", None)
            self._state.update(stage=stage, **fields)
            self._state["sequence"] += 1
            # Setup and phase boundaries must survive even before the first
            # heartbeat. High-frequency graph updates retain only the latest.
            force = stage in {"metadata_validation", "memory_planning", "tokenizer_preparing", "tokenizer_loading",
                              "prompt_encoding", "kernel_initializing", "weights_initializing", "weights_ready",
                              "decoder_initializing", "decoder_ready", "generation_complete"}
            force = force or (stage in ("prefill", "decode") and fields.get("phase") != old_phase)
            self._publish(force=force)

    def _snapshot(self):
        now = self._clock()
        result = {**self._state, "elapsed_seconds": max(0.0, now - self._started),
                  "stage_elapsed_seconds": max(0.0, now - self._stage_started),
                  "updated_at": datetime.now(timezone.utc).isoformat()}
        if self._io_error is not None:
            result["diagnostic_io_error"] = self._io_error
        return result

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _publish(self, *, force=False):
        now = self._clock()
        record = self._snapshot()
        if force or now - self._last_write >= self._interval:
            try:
                write_json(self.path, record)
            except OSError as error:
                self._io_error = f"{type(error).__name__}: progress file unavailable"
            self._last_write = now
        if not self._stdout_broken and (force or now - self._last_stdout >= self._stdout_interval):
            text = f"[{record['action']} {record['elapsed_seconds']:.1f}s] {record['stage']}"
            if "phase" in record:
                text += f" phase={record['phase']}"
            if "token_index" in record:
                text += f" token={record['token_index'] + 1}/{record.get('token_count', '?')}"
            if "layer_index" in record:
                text += f" layer={record['layer_index'] + 1}/{record.get('layer_count', '?')}"
            if "tensor_name" in record:
                text += f" tensor={record['tensor_name']}"
            text += f" stage_elapsed={record['stage_elapsed_seconds']:.1f}s"
            try:
                print(text, flush=True)
            except (OSError, UnicodeError, ValueError):
                self._stdout_broken = True
            self._last_stdout = now

    def _heartbeat(self):
        while not self._stop.wait(self._interval):
            with self._lock:
                if not self._stop.is_set():
                    self._publish()

    def finish(self, status):
        if status not in _TERMINAL:
            raise ValueError("Unknown terminal progress status")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self._interval + 1, 2))
        with self._lock:
            self._state["status"] = status
            self._publish(force=True)
            return self._snapshot()


def load_last_progress(directory, settings, action):
    """Recover only a bounded record bound to this run and checkpoint identity."""
    directory = Path(directory).resolve()
    path = directory / "progress.json"
    try:
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or getattr(before, "st_file_attributes", 0) & 0x400
                or not 1 <= before.st_size <= MAX_PROGRESS_BYTES):
            raise ValueError("Progress record is not a bounded regular file")
        with path.open("rb") as stream:
            raw = stream.read(MAX_PROGRESS_BYTES + 1)
        after = path.lstat()
        if (len(raw) != before.st_size or (before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError("Progress record changed during recovery")
        value = strict_json(raw)
        if not isinstance(value, dict):
            raise ValueError("Progress record must be an object")
        required = {"format_version", "scope", "diagnostic_only", "action", "model_id", "revision", "run_directory",
                    "pid", "sequence", "stage", "status", "elapsed_seconds", "stage_elapsed_seconds", "updated_at", *FULL_MODEL_FLAGS}
        if not required <= value.keys() or set(value) - required - _TEXT.keys() - _COUNTS - {"diagnostic_io_error"}:
            raise ValueError("Progress record has invalid fields")
        if (type(value["format_version"]) is not int or value["format_version"] != 1
                or value["scope"] != "runtime_stage_progress" or value["diagnostic_only"] is not True
                or value["model_id"] != settings["model_id"] or value["revision"] != settings["revision"]
                or value["action"] != action or Path(value["run_directory"]).resolve() != directory
                or value["status"] not in (_TERMINAL | {"RUNNING"}) or not _text(value["stage"], 80)
                or any(value[flag] is not False for flag in FULL_MODEL_FLAGS)):
            raise ValueError("Progress record identity, scope or capability flags differ")
        for key in ("elapsed_seconds", "stage_elapsed_seconds"):
            if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not 0 <= value[key] <= 2**53:
                raise ValueError("Progress elapsed time is invalid")
        for key in (_COUNTS | {"pid", "sequence"}) & value.keys():
            if type(value[key]) is not int or not 0 <= value[key] <= 2**63 - 1:
                raise ValueError("Progress counter is invalid")
        for key, maximum in {**_TEXT, "diagnostic_io_error": 256}.items():
            if key in value and not _text(value[key], maximum):
                raise ValueError("Progress label is invalid")
        timestamp = datetime.fromisoformat(value["updated_at"])
        if timestamp.tzinfo is None:
            raise ValueError("Progress timestamp needs a timezone")
        return value, None
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        return None, f"{type(error).__name__}: last progress unavailable or invalid"
