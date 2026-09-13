"""Explicit loader for the bounded synthetic C CPU tile probe.

No checkpoints, model graphs, downloads, compilation, or fallback are performed here.
Build the DLL separately with build-native.bat. A tile has a single positive scalar
scale, row-major E4M3FN bytes, and at most 128 rows and 128 columns. Decode and
dequantization use FP32; reduction is sequential FP32 with no fused multiply-add.
"""

from __future__ import annotations

import ctypes
from array import array
from contextlib import nullcontext
import math
import os
import struct
import threading
from numbers import Real
from pathlib import Path
from typing import Sequence


MAX_TILE = 128
DEFAULT_DLL = Path(__file__).resolve().parent.parent / "build" / "fp8_cpu.dll"
DENSE_DTYPES = {"BF16": (1, 2), "F16": (2, 2), "F32": (3, 4)}
MAX_DENSE_ROW_BAND_COLUMNS = 16384
MAX_ROW_BATCH = 16


class _PreparedRowBatch:
    """Owned exact [columns,batch] FP32 storage; at most 1 MiB."""
    __slots__ = ("buffer", "cols", "batch")

    def __init__(self, cols, batch):
        self.cols, self.batch = cols, batch
        self.buffer = (ctypes.c_float * (cols * batch))()


def _is_native_float32_array(values):
    return type(values) is array and values.typecode == "f" and values.itemsize == 4


def _owned_float32_buffer(values):
    """Copy exact native FP32 arrays before finite checking the owned snapshot."""
    storage = ctypes.c_float * len(values)
    if _is_native_float32_array(values):
        buffer = storage.from_buffer_copy(values)
        if not all(map(math.isfinite, buffer)):
            raise ValueError("Prepared input must contain finite float32 values")
        return buffer
    return storage(*values)


def _owned_float32_result(output, *, as_array, error_message):
    """Copy native output without per-element float conversion for array callers."""
    if as_array:
        result = array("f")
        with memoryview(output).cast("B") as raw:
            result.frombytes(raw)
    else:
        result = list(output)
    if not all(map(math.isfinite, result)):
        raise ValueError(error_message)
    return result


def _prepare_row_batch(vectors, *, multiple=1):
    if not isinstance(vectors, (list, tuple)) or not 1 <= len(vectors) <= MAX_ROW_BATCH:
        raise ValueError("Native row batch must contain 1..16 input vectors")
    try:
        cols = len(vectors[0])
        if not multiple <= cols <= MAX_DENSE_ROW_BAND_COLUMNS or cols % multiple:
            raise ValueError("Native row batch columns exceed the supported bound or block multiple")
        for vector in vectors:
            if isinstance(vector, (str, bytes, bytearray)) or len(vector) != cols:
                raise ValueError("Native row batch vectors must have identical numeric lengths")
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError("Native row batch requires sized numeric input vectors") from error
    prepared = _PreparedRowBatch(cols, len(vectors))
    if all(map(_is_native_float32_array, vectors)):
        # Copy/scatter in C into the same bounded private allocation. The
        # finite scan sees the owned snapshot, so input edits cannot race a
        # check-then-copy boundary. Exported views also prevent input resizing.
        with memoryview(prepared.buffer).cast("B").cast("f") as destination:
            for lane, vector in enumerate(vectors):
                with memoryview(vector) as source:
                    destination[lane::prepared.batch] = source
            if not all(map(math.isfinite, destination)):
                raise ValueError("Prepared batch input must contain finite float32 values")
        return prepared
    try:
        for col in range(cols):
            for lane, vector in enumerate(vectors):
                prepared.buffer[col * prepared.batch + lane] = _finite_float32(vector[col], "batch input")
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError("Native row batch requires indexable numeric input vectors") from error
    return prepared


def _validate_prepared_batch(prepared, cols):
    if (not isinstance(prepared, _PreparedRowBatch) or type(prepared.cols) is not int or prepared.cols != cols
            or type(prepared.batch) is not int or not 1 <= prepared.batch <= MAX_ROW_BATCH
            or not isinstance(prepared.buffer, ctypes.Array) or prepared.buffer._type_ is not ctypes.c_float
            or len(prepared.buffer) != cols * prepared.batch):
        raise ValueError("Prepared row batch must own exactly cols * batch float32 values")
    return prepared


class _PreparedDenseVector:
    __slots__ = ("buffer",)

    def __init__(self, values):
        self.buffer = _owned_float32_buffer(values)

    def __len__(self):
        return len(self.buffer)


class _NativeRowPool:
    """Lazy per-instance pool; its lock serializes FFI use and deterministic close."""
    def __init__(self, dll, prefix, threads):
        self.lock = threading.RLock()
        self._pointer = ctypes.c_void_p()
        self._closed = False
        self._create = getattr(dll, prefix + "_row_pool_create", None)
        self._destroy = getattr(dll, prefix + "_row_pool_destroy", None)
        self.threads = threads if self._create is not None and self._destroy is not None else 1
        if self._create is not None:
            self._create.argtypes, self._create.restype = [ctypes.c_int], ctypes.c_void_p
        if self._destroy is not None:
            self._destroy.argtypes, self._destroy.restype = [ctypes.c_void_p], None

    def pointer(self, parallelize):
        if self._closed:
            raise RuntimeError("Native row pool is closed")
        if parallelize and self.threads > 1 and not self._pointer.value:
            pointer = self._create(self.threads)
            if not pointer:
                raise RuntimeError("Could not create the bounded native CPU row pool")
            self._pointer = ctypes.c_void_p(pointer)
        return self._pointer

    def close(self):
        with self.lock:
            if self._closed:
                return
            if self._pointer.value:
                self._destroy(self._pointer)
            self._pointer = ctypes.c_void_p()
            self._closed = True
            self._create = self._destroy = None


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

    def __init__(self, dll_path: str | Path | None = None, *, row_band_threads: int = 8) -> None:
        if type(row_band_threads) is not int or not 1 <= row_band_threads <= 8:
            raise ValueError("row_band_threads must be an integer from1 to8")
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
        dense = getattr(dll, "fp8_cpu_matvec_dense_tile", None)
        if dense is not None:
            dense.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ]
            dense.restype = ctypes.c_int
        row_band = getattr(dll, "fp8_cpu_matvec_dense_row_band", None)
        self.supports_dense_row_band = row_band is not None
        if row_band is not None:
            row_band.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t, ctypes.c_void_p,
            ]
            row_band.restype = ctypes.c_int
        many = getattr(dll, "fp8_cpu_matvec_dense_row_band_many", None)
        self.supports_dense_row_band_many = many is not None
        if many is not None:
            many.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t, ctypes.c_void_p]
            many.restype = ctypes.c_int
        self._row_pool = _NativeRowPool(dll, "fp8_cpu", row_band_threads)
        self.row_band_threads = self._row_pool.threads if self.supports_dense_row_band else 1
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
            "dense_tile_available": dense is not None,
            "dense_formats": list(DENSE_DTYPES) if dense is not None else [],
            "dense_row_band_available": self.supports_dense_row_band,
            "dense_row_band_many_available": self.supports_dense_row_band_many,
            "max_row_batch": MAX_ROW_BATCH,
            "max_dense_row_band_columns": MAX_DENSE_ROW_BAND_COLUMNS,
            "row_band_threads": self.row_band_threads,
            "thread_pool_scope": "per-instance; caller included; large row bands only",
        }
        self._dll = dll

    def __enter__(self) -> NativeCpuBackend:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        pool = getattr(self, "_row_pool", None)
        with pool.lock if pool is not None else nullcontext():
            if pool is not None:
                pool.close()
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

    def matvec_dense_tile(self, weights: bytes, rows: int, cols: int,
                          vector: Sequence[float], dtype: str) -> list[float]:
        """Decode a <=64-KiB dense tile and reduce with sequential FP32 arithmetic."""
        dll = self._require_open()
        for label, dimension in (("rows", rows), ("cols", cols)):
            if type(dimension) is not int or not 1 <= dimension <= MAX_TILE:
                raise ValueError(f"{label} must be an integer from 1 to {MAX_TILE}")
        if not isinstance(dtype, str) or dtype not in DENSE_DTYPES:
            raise ValueError("Dense tile dtype must be BF16, F16 or F32")
        code, itemsize = DENSE_DTYPES[dtype]
        if type(weights) is not bytes or len(weights) != rows * cols * itemsize:
            raise ValueError("Dense weights must contain exactly rows * cols * itemsize bytes")
        if isinstance(vector, (str, bytes, bytearray)):
            raise ValueError("Dense vector must be an indexable numeric sequence with cols elements")
        try:
            if len(vector) != cols:
                raise ValueError("Dense vector length must equal cols")
            values = [_finite_float32(vector[index], f"vector[{index}]") for index in range(cols)]
        except (TypeError, KeyError, IndexError) as error:
            raise ValueError("Dense vector must be an indexable numeric sequence with cols elements") from error
        function = getattr(dll, "fp8_cpu_matvec_dense_tile", None)
        if function is None:
            raise RuntimeError("Native CPU DLL lacks the dense tile entry point. Run build-native.bat")
        packed = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
        vector_buffer = (ctypes.c_float * cols)(*values)
        output = (ctypes.c_float * rows)()
        status = function(packed, len(weights), rows, cols, code, vector_buffer, cols, output, rows)
        if status == 1:
            raise ValueError("Native dense CPU backend rejected tile arguments")
        if status == 2:
            raise ValueError("Native dense payload or calculation encountered a non-finite FP32 value")
        if status != 0:
            raise RuntimeError(f"Native dense CPU backend returned unknown status {status}")
        result = list(output)
        if not all(math.isfinite(value) for value in result):
            raise ValueError("Native dense CPU backend returned a non-finite FP32 result")
        return result

    def prepare_dense_vector(self, vector):
        self._require_open()
        if isinstance(vector, (str, bytes, bytearray)):
            raise ValueError("Dense vector must be an indexable numeric sequence")
        try:
            count = len(vector)
            if not 1 <= count <= MAX_DENSE_ROW_BAND_COLUMNS:
                raise ValueError("Dense row-band vector must contain1..16384 elements")
            if _is_native_float32_array(vector):
                return _PreparedDenseVector(vector)
            values = [_finite_float32(vector[index], f"vector[{index}]") for index in range(count)]
        except (TypeError, IndexError, KeyError) as error:
            raise ValueError("Dense vector must be an indexable finite numeric sequence") from error
        return _PreparedDenseVector(values)

    def prepare_dense_batch(self, vectors):
        self._require_open()
        return _prepare_row_batch(vectors)

    def matvec_dense_row_band_many(self, weights, rows, cols, prepared, dtype):
        with self._row_pool.lock:
            dll = self._require_open()
            if type(rows) is not int or not 1 <= rows <= MAX_TILE:
                raise ValueError("Dense row batch must contain 1..128 rows")
            if type(cols) is not int or not 1 <= cols <= MAX_DENSE_ROW_BAND_COLUMNS:
                raise ValueError("Dense row batch columns must be 1..16384")
            if not isinstance(dtype, str) or dtype not in DENSE_DTYPES:
                raise ValueError("Dense row batch dtype must be BF16, F16 or F32")
            code, itemsize = DENSE_DTYPES[dtype]
            if type(weights) not in (bytes, bytearray) or len(weights) != rows * cols * itemsize:
                raise ValueError("Dense row batch byte count must match shape and dtype")
            _validate_prepared_batch(prepared, cols)
            function = getattr(dll, "fp8_cpu_matvec_dense_row_band_many", None)
            if function is None:
                raise RuntimeError("Native CPU DLL lacks SIMD input batching. Run build-native.bat")
            pointer = self._row_pool.pointer(rows > 1 and rows * cols * prepared.batch >= 65536)
            storage = ctypes.c_uint8 * len(weights)
            packed = storage.from_buffer(weights) if type(weights) is bytearray else storage.from_buffer_copy(weights)
            count = rows * prepared.batch
            output = (ctypes.c_float * count)()
            status = function(packed, len(weights), rows, cols, prepared.batch, code,
                prepared.buffer, len(prepared.buffer), output, count, pointer)
            if status in (1, 2):
                raise ValueError("Native dense row batch rejected arguments" if status == 1 else
                                 "Native dense row batch encountered non-finite FP32 values")
            if status:
                raise RuntimeError(f"Native dense row batch returned status {status}")
            if not all(math.isfinite(value) for value in output):
                raise ValueError("Native dense row batch returned non-finite FP32 values")
            return [array("f", (output[row * prepared.batch + lane] for row in range(rows)))
                    for lane in range(prepared.batch)]

    def matvec_dense_row_band(self, weights, rows, cols, vector, dtype):
        pool = getattr(self, "_row_pool", None)
        with pool.lock if pool is not None else nullcontext():
            return self._matvec_dense_row_band(weights, rows, cols, vector, dtype, pool)

    def matvec_dense_row_band_array(self, weights, rows, cols, vector, dtype):
        """Return an owned finite FP32 array with the same row-band arithmetic."""
        pool = getattr(self, "_row_pool", None)
        with pool.lock if pool is not None else nullcontext():
            return self._matvec_dense_row_band(weights, rows, cols, vector, dtype, pool, as_array=True)

    def _matvec_dense_row_band(self, weights, rows, cols, vector, dtype, pool, *, as_array=False):
        dll = self._require_open()
        if type(rows) is not int or not 1 <= rows <= MAX_TILE:
            raise ValueError("Dense row band must contain1..128 rows")
        if type(cols) is not int or not 1 <= cols <= MAX_DENSE_ROW_BAND_COLUMNS:
            raise ValueError("Dense row-band columns must be1..16384")
        if not isinstance(dtype, str) or dtype not in DENSE_DTYPES:
            raise ValueError("Dense row-band dtype must be BF16, F16 or F32")
        code, itemsize = DENSE_DTYPES[dtype]
        if type(weights) not in (bytes, bytearray) or len(weights) != rows * cols * itemsize:
            raise ValueError("Dense row-band byte count must match its shape and dtype")
        prepared = vector if isinstance(vector, _PreparedDenseVector) else self.prepare_dense_vector(vector)
        if (not isinstance(prepared.buffer, ctypes.Array) or prepared.buffer._type_ is not ctypes.c_float
                or len(prepared) != cols):
            raise ValueError("Prepared dense vector must own exactly cols float32 values")
        function = getattr(dll, "fp8_cpu_matvec_dense_row_band", None)
        if function is None:
            raise RuntimeError("Native CPU DLL lacks dense row-band batching. Run build-native.bat")
        pointer = pool.pointer(rows > 1 and rows * cols >= 65536) if pool is not None else ctypes.c_void_p()
        storage = ctypes.c_uint8 * len(weights)
        packed = storage.from_buffer(weights) if type(weights) is bytearray else storage.from_buffer_copy(weights)
        output = (ctypes.c_float * rows)()
        status = function(packed, len(weights), rows, cols, code, prepared.buffer, cols, output, rows, pointer)
        if status == 1:
            raise ValueError("Native dense row-band backend rejected arguments")
        if status == 2:
            raise ValueError("Native dense row-band payload or arithmetic encountered non-finite FP32 values")
        if status != 0:
            raise RuntimeError(f"Native dense row-band backend returned status {status}")
        return _owned_float32_result(output, as_array=as_array,
                                    error_message="Native dense row band returned non-finite FP32 results")
