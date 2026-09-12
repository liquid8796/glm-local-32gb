"""Tiny generic FP8 arithmetic through CUDA's Driver API, without a toolkit.

This probe accepts only one <=128x128 E4M3FN tile in memory. It does not read
checkpoint files, infer a model, or promise any GPU utilization cap. Explicit
CUDA requests fail with a diagnostic; there is no CPU fallback.

Driver ABI and parameter layout:
https://docs.nvidia.com/cuda/archive/11.5.2/cuda-driver-api/group__CUDA__CTX.html
https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html
"""

from contextlib import contextmanager
import ctypes as C
import math
from numbers import Real
import os
from pathlib import Path
import sys
import threading
from typing import Sequence


MAX_TILE = 128
MAX_OPERATIONS = 256
MAX_EXECUTION_OPERATIONS = 65536
MAX_EXPLICIT_DEVICE_BYTES = MAX_TILE * MAX_TILE + 2 * MAX_TILE * 4
_PTX_PATH = Path(__file__).resolve().parent.parent / "native" / "fp8_matvec.ptx"


class CudaProbeError(RuntimeError):
    """CUDA is unavailable, a driver operation failed, or probe bounds were reached."""


def _float32(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number representable as float32")
    try:
        number = float(value)
        rounded = C.c_float(number).value
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be represented as float32") from error
    if not math.isfinite(number) or not math.isfinite(rounded):
        raise ValueError(f"{name} must be finite in float32")
    return rounded


def validate_tile(weights: bytes, rows: int, cols: int,
                  vector: Sequence[float], scale: float):
    """Validate all sizes before copying, returning bounded float32 host values."""
    for name, value in (("rows", rows), ("cols", cols)):
        if type(value) is not int or not 1 <= value <= MAX_TILE:
            raise ValueError(f"{name} must be an integer in [1, {MAX_TILE}]")
    if type(weights) is not bytes or len(weights) != rows * cols:
        raise ValueError("weights must be bytes of exactly rows * cols length")
    if any((value & 0x7F) == 0x7F for value in weights):
        raise ValueError("E4M3FN NaN weight bytes (0x7F or 0xFF) are not supported")
    if isinstance(vector, (str, bytes, bytearray)):
        raise ValueError("vector must be a finite numeric sequence of length cols")
    try:
        length = len(vector)
    except TypeError as error:
        raise ValueError("vector must be a finite numeric sequence of length cols") from error
    if length != cols:
        raise ValueError("vector length must equal cols")
    # Indexed access, rather than unbounded iteration, also rejects deceptive iterators.
    try:
        values = tuple(_float32(vector[index], f"vector[{index}]") for index in range(cols))
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("vector must be an indexable sequence of length cols") from error
    block_scale = _float32(scale, "scale")
    if block_scale <= 0:
        raise ValueError("scale must be positive and nonzero after float32 conversion")
    return values, block_scale


def _load_driver():
    if os.name != "nt":
        raise CudaProbeError("The CUDA probe currently requires Windows and nvcuda.dll")
    if C.sizeof(C.c_void_p) != 8:
        raise CudaProbeError("The CUDA probe requires a 64-bit Python interpreter")
    try:
        # LOAD_LIBRARY_SEARCH_SYSTEM32 prevents a workspace DLL from being selected.
        return C.WinDLL("nvcuda.dll", winmode=0x00000800)
    except OSError as error:
        raise CudaProbeError("Cannot load the installed NVIDIA driver (nvcuda.dll)") from error


def _bind(driver):
    pvoid = C.POINTER(C.c_void_p)
    pint = C.POINTER(C.c_int)
    pu64 = C.POINTER(C.c_uint64)
    signatures = {
        "cuInit": [C.c_uint],
        "cuDriverGetVersion": [pint],
        "cuDeviceGet": [pint, C.c_int],
        "cuDeviceGetName": [C.c_char_p, C.c_int, C.c_int],
        "cuDeviceGetAttribute": [pint, C.c_int, C.c_int],
        "cuDeviceTotalMem_v2": [C.POINTER(C.c_size_t), C.c_int],
        "cuCtxCreate_v2": [pvoid, C.c_uint, C.c_int],
        "cuCtxDestroy_v2": [C.c_void_p],
        "cuCtxPushCurrent_v2": [C.c_void_p],
        "cuCtxPopCurrent_v2": [pvoid],
        "cuCtxSynchronize": [],
        "cuModuleLoadData": [pvoid, C.c_void_p],
        "cuModuleGetFunction": [pvoid, C.c_void_p, C.c_char_p],
        "cuModuleUnload": [C.c_void_p],
        "cuMemAlloc_v2": [pu64, C.c_size_t],
        "cuMemFree_v2": [C.c_uint64],
        "cuMemcpyHtoD_v2": [C.c_uint64, C.c_void_p, C.c_size_t],
        "cuMemcpyDtoH_v2": [C.c_void_p, C.c_uint64, C.c_size_t],
        "cuLaunchKernel": [C.c_void_p] + [C.c_uint] * 7
                          + [C.c_void_p, pvoid, pvoid],
        "cuGetErrorName": [C.c_int, C.POINTER(C.c_char_p)],
        "cuGetErrorString": [C.c_int, C.POINTER(C.c_char_p)],
    }
    for name, args in signatures.items():
        try:
            function = getattr(driver, name)
        except AttributeError as error:
            raise CudaProbeError(f"NVIDIA driver is missing required symbol {name}") from error
        function.argtypes = args
        function.restype = C.c_int
    return driver


def _cleanup_errors(errors, primary):
    if not errors:
        return
    description = "; ".join(str(error) for error in errors)
    if primary is None:
        raise CudaProbeError(f"CUDA resource cleanup failed: {description}")
    if hasattr(primary, "add_note"):
        primary.add_note(f"CUDA resource cleanup also failed: {description}")


class CudaTileBackend:
    """Synchronous, single-threaded numerical probe with explicitly owned resources.

    By default at most 256 launches per instance; an explicit execution budget
    may permit up to 65536. Each launch allocates at most 17,408
    application buffer bytes, all freed before returning. Driver/context overhead
    is additional and is not included in that explicit allocation bound.
    """

    def __init__(self, device_index=0, *, max_operations=MAX_OPERATIONS):
        if type(device_index) is not int or device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        if type(max_operations) is not int or not 1 <= max_operations <= MAX_EXECUTION_OPERATIONS:
            raise ValueError(f"max_operations must be an integer in [1, {MAX_EXECUTION_OPERATIONS}]")
        self.max_operations = max_operations
        self._thread_id = threading.get_ident()
        self._driver = _bind(_load_driver())
        self._context = C.c_void_p()
        self._module = C.c_void_p()
        self._function = C.c_void_p()
        self.closed = False
        self.operations = 0
        self.peak_explicit_device_bytes = 0
        self._check("cuInit", 0)
        device = C.c_int()
        self._check("cuDeviceGet", C.byref(device), device_index)
        name = C.create_string_buffer(256)
        self._check("cuDeviceGetName", name, len(name), device)
        major, minor, version, memory = C.c_int(), C.c_int(), C.c_int(), C.c_size_t()
        self._check("cuDeviceGetAttribute", C.byref(major), 75, device)
        self._check("cuDeviceGetAttribute", C.byref(minor), 76, device)
        self._check("cuDriverGetVersion", C.byref(version))
        self._check("cuDeviceTotalMem_v2", C.byref(memory), device)
        if major.value < 8:
            raise CudaProbeError("This PTX probe requires compute capability 8.0 or newer")
        self.device_info = {
            "index": device_index, "name": name.value.decode("utf-8", errors="replace"),
            "compute_capability": f"{major.value}.{minor.value}",
            "cuda_driver_api_version": version.value, "total_memory_bytes": memory.value,
            "native_fp8_instructions": False,
        }
        ptx = _PTX_PATH.read_bytes()
        if len(ptx) > 65536 or b"\0" in ptx:
            raise CudaProbeError("Bundled PTX must be <=64 KiB and contain no NUL bytes")
        ptx_buffer = C.create_string_buffer(ptx)
        # CU_CTX_SCHED_BLOCKING_SYNC: the waiting CPU thread blocks instead of spinning.
        self._check("cuCtxCreate_v2", C.byref(self._context), 4, device)
        try:
            self._check("cuModuleLoadData", C.byref(self._module), ptx_buffer)
            self._check("cuModuleGetFunction", C.byref(self._function), self._module,
                        b"fp8_matvec_tile")
        except BaseException:
            primary = sys.exc_info()[1]
            errors = []
            if self._module.value:
                try:
                    self._check("cuModuleUnload", self._module)
                except CudaProbeError as error:
                    errors.append(error)
            try:
                self._check("cuCtxDestroy_v2", self._context)
            except CudaProbeError as error:
                errors.append(error)
            self.closed = True
            _cleanup_errors(errors, primary)
            raise
        popped = C.c_void_p()
        try:
            self._check("cuCtxPopCurrent_v2", C.byref(popped))
        except BaseException:
            self.close()
            raise

    def _check(self, operation, *args):
        code = getattr(self._driver, operation)(*args)
        if code:
            name, detail = C.c_char_p(), C.c_char_p()
            self._driver.cuGetErrorName(code, C.byref(name))
            self._driver.cuGetErrorString(code, C.byref(detail))
            label = name.value.decode("utf-8", errors="replace") if name.value else str(code)
            explanation = detail.value.decode("utf-8", errors="replace") if detail.value else ""
            raise CudaProbeError(f"{operation}: {label} ({code}): {explanation}")

    def _check_thread(self):
        if threading.get_ident() != self._thread_id:
            raise CudaProbeError("Use and close this CUDA probe only on its creating thread")

    @contextmanager
    def _activate(self):
        self._check_thread()
        self._check("cuCtxPushCurrent_v2", self._context)
        try:
            yield
        finally:
            primary = sys.exc_info()[1]
            popped = C.c_void_p()
            try:
                self._check("cuCtxPopCurrent_v2", C.byref(popped))
                if popped.value != self._context.value:
                    raise CudaProbeError("Unexpected CUDA context stack change")
            except CudaProbeError as error:
                _cleanup_errors([error], primary)

    def matvec_tile(self, weights: bytes, rows: int, cols: int,
                    vector: Sequence[float], scale: float) -> list[float]:
        """Return row dot products, using FP32 scaling, products, and accumulation."""
        self._check_thread()
        if self.closed:
            raise CudaProbeError("CUDA probe is closed")
        values, block_scale = validate_tile(weights, rows, cols, vector, scale)
        maximum = getattr(self, "max_operations", MAX_OPERATIONS)
        if self.operations >= maximum:
            raise CudaProbeError(f"Probe operation limit reached ({maximum})")
        host_weights = (C.c_ubyte * len(weights)).from_buffer_copy(weights)
        host_vector = (C.c_float * cols)(*values)
        host_output = (C.c_float * rows)()
        buffers = []
        with self._activate():
            try:
                for size in (len(weights), C.sizeof(host_vector), C.sizeof(host_output)):
                    pointer = C.c_uint64()
                    self._check("cuMemAlloc_v2", C.byref(pointer), size)
                    buffers.append(pointer)
                allocated = len(weights) + C.sizeof(host_vector) + C.sizeof(host_output)
                self.peak_explicit_device_bytes = max(self.peak_explicit_device_bytes, allocated)
                self._check("cuMemcpyHtoD_v2", buffers[0], host_weights, len(weights))
                self._check("cuMemcpyHtoD_v2", buffers[1], host_vector, C.sizeof(host_vector))
                arguments = (*buffers, C.c_uint32(rows), C.c_uint32(cols), C.c_float(block_scale))
                parameters = (C.c_void_p * len(arguments))(
                    *(C.cast(C.byref(arg), C.c_void_p).value for arg in arguments))
                self.operations += 1
                self._check("cuLaunchKernel", self._function,
                            (rows + 31) // 32, 1, 1, 32, 1, 1, 0, None, parameters, None)
                self._check("cuCtxSynchronize")
                self._check("cuMemcpyDtoH_v2", host_output, buffers[2], C.sizeof(host_output))
                result = list(host_output)
                if not all(math.isfinite(value) for value in result):
                    raise CudaProbeError("FP32 arithmetic overflowed; use smaller vector/scale values")
                return result
            finally:
                primary = sys.exc_info()[1]
                errors = []
                for pointer in reversed(buffers):
                    try:
                        self._check("cuMemFree_v2", pointer)
                    except CudaProbeError as error:
                        errors.append(error)
                _cleanup_errors(errors, primary)

    def close(self):
        self._check_thread()
        if self.closed:
            return
        errors = []
        try:
            with self._activate():
                for operation, args in (("cuCtxSynchronize", ()),
                                        ("cuModuleUnload", (self._module,))):
                    try:
                        self._check(operation, *args)
                    except CudaProbeError as error:
                        errors.append(error)
        except CudaProbeError as error:
            errors.append(error)
        finally:
            try:
                self._check("cuCtxDestroy_v2", self._context)
            except CudaProbeError as error:
                errors.append(error)
            self.closed = True
            self._context = C.c_void_p()
            self._module = C.c_void_p()
            self._function = C.c_void_p()
        _cleanup_errors(errors, None)

    def __enter__(self):
        if self.closed:
            raise CudaProbeError("CUDA probe is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except CudaProbeError as error:
            _cleanup_errors([error], exc)
        return False
