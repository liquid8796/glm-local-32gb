"""Explicit loader for the bounded synthetic C CPU tile probe.

No checkpoints, model graphs, downloads, compilation, or fallback are performed here.
Build the DLL separately with build-native.bat. A tile has a single positive scalar
scale, row-major E4M3FN bytes, and at most 128 rows and 128 columns. Decode and
dequantization use FP32; reduction is sequential FP32 with no fused multiply-add.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
from numbers import Real
from pathlib import Path
from typing import Sequence


MAX_TILE = 128
DEFAULT_DLL = Path(__file__).resolve().parent.parent / "build" / "fp8_cpu.dll"


def _finite_float32(value: Real, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number representable as float32")
    try:
        converted = float(value)
        rounded = struct.unpack("=f", struct.pack("=f", converted))[0]
    except (OverflowError, ValueError, TypeError, struct.error) as exc:
        raise ValueError(f"{label} must be finite and representable as float32") from exc
    if not math.isfinite(converted) or not math.isfinite(rounded):
        raise ValueError(f"{label} must be finite and representable as float32")
    if positive and rounded <= 0.0:
        raise ValueError(f"{label} must remain positive after rounding to float32")
    return rounded


def _validate_tile(
    weights: bytes, rows: int, cols: int, vector: Sequence[float], scale: float
) -> tuple[list[float], float]:
    for label, dimension in (("rows", rows), ("cols", cols)):
        if type(dimension) is not int or not 1 <= dimension <= MAX_TILE:
            raise ValueError(f"{label} must be an integer from 1 to {MAX_TILE}")
    if not isinstance(weights, bytes) or len(weights) != rows * cols:
        raise ValueError("weights must be bytes with exactly rows * cols elements")
    if 0x7F in weights or 0xFF in weights:
        raise ValueError("weights contain an E4M3FN NaN encoding")
    try:
        length = len(vector)
    except TypeError as exc:
        raise ValueError("vector must be a sequence with exactly cols elements") from exc
    if length != cols:
        raise ValueError("vector must have exactly cols elements")
    rounded_vector = [_finite_float32(vector[i], f"vector[{i}]") for i in range(cols)]
    return rounded_vector, _finite_float32(scale, "scale", positive=True)


class NativeCpuBackend:
    """Small, single-threaded C probe. Closing releases wrapper references only.

    The shared library lifetime is managed by ctypes/the process. A closed wrapper
    cannot execute any operation. Construction never builds a missing library.
    """

    def __init__(self, dll_path: str | Path | None = None) -> None:
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("The native CPU probe requires 64-bit Windows Python")
        path = Path(dll_path) if dll_path is not None else DEFAULT_DLL
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Native CPU DLL is missing: {path}. Run build-native.bat")
        self._dll: ctypes.CDLL | None = None
        dll = ctypes.CDLL(str(path))
        dll.fp8_cpu_abi_version.argtypes = []
        dll.fp8_cpu_abi_version.restype = ctypes.c_int
        if dll.fp8_cpu_abi_version() != 1:
            raise RuntimeError("Unsupported native CPU probe ABI (expected version 1)")
        dll.fp8_cpu_build_info.argtypes = []
        dll.fp8_cpu_build_info.restype = ctypes.c_char_p
        dll.fp8_cpu_decode_e4m3fn.argtypes = [ctypes.c_uint8]
        dll.fp8_cpu_decode_e4m3fn.restype = ctypes.c_float
        dll.fp8_cpu_matvec_tile.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]
        dll.fp8_cpu_matvec_tile.restype = ctypes.c_int
        build_info = dll.fp8_cpu_build_info()
        if not build_info:
            raise RuntimeError("Native CPU DLL returned empty build metadata")
        self.metadata = {
            "backend": "native-c-cpu-synthetic-tile",
            "dll_path": str(path),
            "abi_version": 1,
            "build_info": build_info.decode("ascii"),
            "weight_format": "E4M3FN",
            "arithmetic": "FP32 dequantization and sequential FP32 reduction",
            "max_tile_rows": MAX_TILE,
            "max_tile_cols": MAX_TILE,
            "threads": 1,
            "full_model_inference": False,
        }
        self._dll = dll

    def __enter__(self) -> NativeCpuBackend:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._dll = None

    def _require_open(self) -> ctypes.CDLL:
        if self._dll is None:
            raise RuntimeError("Native CPU backend is closed")
        return self._dll

    def decode_e4m3fn(self, code: int) -> float:
        dll = self._require_open()
        if type(code) is not int or not 0 <= code <= 255:
            raise ValueError("code must be an integer from 0 to 255")
        return float(dll.fp8_cpu_decode_e4m3fn(code))

    def matvec_tile(
        self, weights: bytes, rows: int, cols: int, vector: Sequence[float], scale: float
    ) -> list[float]:
        dll = self._require_open()
        rounded_vector, rounded_scale = _validate_tile(weights, rows, cols, vector, scale)
        weight_buffer = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
        vector_buffer = (ctypes.c_float * cols)(*rounded_vector)
        output = (ctypes.c_float * rows)()
        status = dll.fp8_cpu_matvec_tile(
            weight_buffer, len(weights), rows, cols, vector_buffer, cols,
            rounded_scale, output, rows,
        )
        if status == 1:
            raise ValueError("Native CPU backend rejected tile arguments")
        if status == 2:
            raise ValueError("Native CPU calculation encountered a non-finite FP32 value")
        if status != 0:
            raise RuntimeError(f"Native CPU backend returned unknown status {status}")
        result = list(output)
        if not all(math.isfinite(value) for value in result):
            raise ValueError("Native CPU backend returned a non-finite FP32 result")
        return result
