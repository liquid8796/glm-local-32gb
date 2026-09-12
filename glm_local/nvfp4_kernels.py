"""Bounded native CPU/CUDA NVFP4 decoded-weight FP32 matvecs.

These implement a weight-only reference, with activation_quantization='none'.
They do not reproduce W4A4 activation rounding or use native FP4 tensor cores.
Stored unswizzled ModelOpt scales are combined in FP32 before weight decode;
products and sequential sums round to FP32, with no fused multiply-add.

Format source: NVIDIA ModelOpt 51de53e48ccae8804f8fe1198b7cf89475c5c4f4,
modelopt/torch/quantization/qtensor/nvfp4_tensor.py. The CUDA backend reuses
the existing Driver API context ownership and also supports its FP8 method.
"""

import ctypes as C
import math
import os
from pathlib import Path
import sys

from .cpu_probe import _finite_float32
from .cuda_probe import CudaProbeError, CudaTileBackend, MAX_OPERATIONS, _cleanup_errors
from .nvfp4_blocks import BLOCK, TILE, decode_e4m3_scale


DEFAULT_DLL = Path(__file__).resolve().parent.parent / "build/nvfp4_cpu.dll"
_PTX_PATH = Path(__file__).resolve().parent.parent / "native/nvfp4_matvec.ptx"
MAX_EXPLICIT_DEVICE_BYTES = TILE * TILE // 2 + TILE * TILE // BLOCK + 2 * TILE * 4
MAX_ROW_BAND_COLUMNS = 16384


class _PreparedNVFP4Vector:
    """One owned finite FP32 input buffer reused across the projection's row bands."""
    __slots__ = ("buffer",)

    def __init__(self, values):
        self.buffer = (C.c_float * len(values))(*values)

    def __len__(self):
        return len(self.buffer)


def validate_nvfp4_tile(packed, rows, cols, scales, vector, global_scale):
    """Validate dimensions and bounded data before allocating host/device buffers."""
    for name, dimension in (("rows", rows), ("cols", cols)):
        if type(dimension) is not int or not 1 <= dimension <= TILE:
            raise ValueError(f"NVFP4 {name} must be an integer from 1 to {TILE}")
    if cols % BLOCK:
        raise ValueError("NVFP4 columns must be a multiple of 16")
    if type(packed) is not bytes or len(packed) != rows * (cols // 2):
        raise ValueError("NVFP4 packed weights must be exactly rows * cols / 2 bytes")
    if type(scales) is not bytes or len(scales) != rows * (cols // BLOCK):
        raise ValueError("NVFP4 block scales must be exactly rows * cols / 16 bytes")
    for code in scales:
        decode_e4m3_scale(code)
    if isinstance(vector, (str, bytes, bytearray)):
        raise ValueError("NVFP4 vector must be a finite numeric sequence with cols elements")
    try:
        if len(vector) != cols:
            raise ValueError("NVFP4 vector length must equal cols")
        values = tuple(_finite_float32(vector[index], f"vector[{index}]") for index in range(cols))
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError("NVFP4 vector must be an indexable finite numeric sequence") from error
    return values, _finite_float32(global_scale, "NVFP4 global weight scale", positive=True)


class NativeNVFP4CpuBackend:
    """Single-threaded, <=128-square C primitive; DLL compilation is explicit."""

    activation_quantization = "none"

    def __init__(self, dll_path=None):
        self._dll = None
        if os.name != "nt" or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError("The native NVFP4 CPU backend requires 64-bit Windows Python")
        path = Path(dll_path if dll_path is not None else DEFAULT_DLL).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Native NVFP4 CPU DLL is missing: {path}. Run build-native.bat")
        dll = C.CDLL(str(path))
        dll.nvfp4_cpu_abi_version.argtypes = []
        dll.nvfp4_cpu_abi_version.restype = C.c_int
        if dll.nvfp4_cpu_abi_version() != 1:
            raise RuntimeError("Unsupported native NVFP4 CPU ABI (expected version 1)")
        dll.nvfp4_cpu_build_info.argtypes = []
        dll.nvfp4_cpu_build_info.restype = C.c_char_p
        dll.nvfp4_cpu_matvec_tile.argtypes = [
            C.POINTER(C.c_uint8), C.c_size_t, C.c_int, C.c_int,
            C.POINTER(C.c_uint8), C.c_size_t,
            C.POINTER(C.c_float), C.c_size_t, C.c_float,
            C.POINTER(C.c_float), C.c_size_t,
        ]
        dll.nvfp4_cpu_matvec_tile.restype = C.c_int
        row_band = getattr(dll, "nvfp4_cpu_matvec_row_band", None)
        if row_band is not None:
            row_band.argtypes = list(dll.nvfp4_cpu_matvec_tile.argtypes)
            row_band.restype = C.c_int
        self.supports_nvfp4_row_band = row_band is not None
        info = dll.nvfp4_cpu_build_info()
        if not info:
            raise RuntimeError("Native NVFP4 DLL returned empty build metadata")
        self.metadata = dict(backend="native-c-cpu-nvfp4-decoded-weight", dll_path=str(path),
            abi_version=1, build_info=info.decode("ascii"), weight_format="NVFP4_E2M1",
            activation_quantization="none", native_nvfp4_instructions=False,
            arithmetic="FP32 combined scales, weight decode, products and sequential reduction",
            max_tile_rows=TILE, max_tile_cols=TILE, threads=1, full_model_inference=False,
            row_band_available=self.supports_nvfp4_row_band, max_row_band_columns=MAX_ROW_BAND_COLUMNS)
        self._dll = dll

    def _require_open(self):
        if self._dll is None:
            raise RuntimeError("Native NVFP4 CPU backend is closed")
        return self._dll

    def matvec_nvfp4_tile(self, packed, rows, cols, scales, vector, global_scale):
        dll = self._require_open()
        values, global_weight_scale = validate_nvfp4_tile(packed, rows, cols, scales, vector, global_scale)
        weights_buffer = (C.c_uint8 * len(packed)).from_buffer_copy(packed)
        scales_buffer = (C.c_uint8 * len(scales)).from_buffer_copy(scales)
        vector_buffer = (C.c_float * cols)(*values)
        output = (C.c_float * rows)()
        status = dll.nvfp4_cpu_matvec_tile(weights_buffer, len(packed), rows, cols,
            scales_buffer, len(scales), vector_buffer, cols, global_weight_scale, output, rows)
        if status == 1:
            raise ValueError("Native NVFP4 CPU backend rejected tile arguments")
        if status == 2:
            raise ValueError("Native NVFP4 CPU arithmetic encountered a non-finite FP32 value")
        if status != 0:
            raise RuntimeError(f"Native NVFP4 CPU backend returned unknown status {status}")
        result = list(output)
        if not all(math.isfinite(value) for value in result):
            raise ValueError("Native NVFP4 CPU backend returned a non-finite FP32 result")
        return result

    def prepare_nvfp4_vector(self, vector):
        self._require_open()
        if isinstance(vector, (str, bytes, bytearray)):
            raise ValueError("NVFP4 vector must be an indexable numeric sequence")
        try:
            length = len(vector)
            if not 16 <= length <= MAX_ROW_BAND_COLUMNS or length % BLOCK:
                raise ValueError("NVFP4 row-band vector length must be a multiple of 16 up to 16384")
            values = [_finite_float32(vector[index], f"vector[{index}]") for index in range(length)]
        except (TypeError, IndexError, KeyError) as error:
            raise ValueError("NVFP4 vector must be an indexable finite numeric sequence") from error
        return _PreparedNVFP4Vector(values)

    def matvec_nvfp4_row_band(self, packed, rows, cols, scales, vector, global_scale):
        dll = self._require_open()
        if type(rows) is not int or not 1 <= rows <= TILE:
            raise ValueError("NVFP4 row band must contain 1..128 rows")
        if type(cols) is not int or not 16 <= cols <= MAX_ROW_BAND_COLUMNS or cols % BLOCK:
            raise ValueError("NVFP4 row-band columns must be a multiple of 16 up to 16384")
        if type(packed) is not bytes or len(packed) != rows * (cols // 2):
            raise ValueError("NVFP4 row-band packed byte count differs from shape")
        if type(scales) is not bytes or len(scales) != rows * (cols // BLOCK):
            raise ValueError("NVFP4 row-band scale byte count differs from shape")
        # Both byte scans run in C; valid scales are exactly codes0..126.
        # The native boundary also validates every byte independently.
        if not scales.isascii() or b"\x7f" in scales:
            for code in scales:
                decode_e4m3_scale(code)
        prepared = vector if isinstance(vector, _PreparedNVFP4Vector) else self.prepare_nvfp4_vector(vector)
        if not isinstance(prepared.buffer, C.Array) or prepared.buffer._type_ is not C.c_float:
            raise ValueError("Prepared NVFP4 vector must own a float32 native buffer")
        if len(prepared) != cols:
            raise ValueError("NVFP4 vector length must equal row-band columns")
        scale = _finite_float32(global_scale, "NVFP4 global weight scale", positive=True)
        function = getattr(dll, "nvfp4_cpu_matvec_row_band", None)
        if function is None:
            raise RuntimeError("NVFP4 CPU DLL lacks row-band batching. Run build-native.bat")
        weights_buffer = (C.c_uint8 * len(packed)).from_buffer_copy(packed)
        scales_buffer = (C.c_uint8 * len(scales)).from_buffer_copy(scales)
        output = (C.c_float * rows)()
        status = function(weights_buffer, len(packed), rows, cols, scales_buffer, len(scales),
                          prepared.buffer, cols, scale, output, rows)
        if status == 1:
            raise ValueError("Native NVFP4 CPU backend rejected row-band arguments")
        if status == 2:
            raise ValueError("Native NVFP4 row-band arithmetic encountered a non-finite FP32 value")
        if status != 0:
            raise RuntimeError(f"Native NVFP4 row-band backend returned unknown status {status}")
        result = list(output)
        if not all(math.isfinite(value) for value in result):
            raise ValueError("Native NVFP4 row band returned non-finite FP32 results")
        return result

    def close(self):
        self._dll = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()


class CudaNVFP4TileBackend(CudaTileBackend):
    """Driver-JIT NVFP4 FP32 reference with the existing bounded CUDA lifecycle.

    NVFP4 allocates at most 10,240 explicit device-buffer bytes per tile. Driver
    context/module overhead is additional. The inherited FP8 method stays valid
    and shares the same total operation budget (default 256, explicit <=65536).
    """

    activation_quantization = "none"

    def __init__(self, device_index=0, *, max_operations=MAX_OPERATIONS):
        self._nvfp4_module = C.c_void_p()
        self._nvfp4_function = C.c_void_p()
        super().__init__(device_index, max_operations=max_operations)
        try:
            ptx = _PTX_PATH.read_bytes()
            if len(ptx) > 65536 or b"\0" in ptx:
                raise CudaProbeError("Bundled NVFP4 PTX must be <=64 KiB and contain no NUL bytes")
            buffer = C.create_string_buffer(ptx)
            with self._activate():
                self._check("cuModuleLoadData", C.byref(self._nvfp4_module), buffer)
                self._check("cuModuleGetFunction", C.byref(self._nvfp4_function), self._nvfp4_module,
                            b"nvfp4_matvec_tile")
        except BaseException:
            primary = sys.exc_info()[1]
            try:
                self.close()
            except CudaProbeError as error:
                _cleanup_errors([error], primary)
            raise
        self.device_info.update(native_nvfp4_instructions=False, activation_quantization="none")
        self.metadata = dict(backend="cuda-ptx-nvfp4-decoded-weight", weight_format="NVFP4_E2M1",
            activation_quantization="none", native_nvfp4_instructions=False,
            max_tile_rows=TILE, max_tile_cols=TILE, nvfp4_max_explicit_device_bytes=MAX_EXPLICIT_DEVICE_BYTES,
            arithmetic="FP32 combined scales, weight decode, products and sequential reduction",
            full_model_inference=False)

    def matvec_nvfp4_tile(self, packed, rows, cols, scales, vector, global_scale):
        self._check_thread()
        if self.closed:
            raise CudaProbeError("CUDA NVFP4 backend is closed")
        values, global_weight_scale = validate_nvfp4_tile(packed, rows, cols, scales, vector, global_scale)
        maximum = getattr(self, "max_operations", MAX_OPERATIONS)
        if self.operations >= maximum:
            raise CudaProbeError(f"Probe operation limit reached ({maximum})")
        host_weights = (C.c_uint8 * len(packed)).from_buffer_copy(packed)
        host_scales = (C.c_uint8 * len(scales)).from_buffer_copy(scales)
        host_vector = (C.c_float * cols)(*values)
        host_output = (C.c_float * rows)()
        buffers = []
        with self._activate():
            try:
                hosts = (host_weights, host_scales, host_vector, host_output)
                for host in hosts:
                    pointer = C.c_uint64()
                    self._check("cuMemAlloc_v2", C.byref(pointer), C.sizeof(host))
                    buffers.append(pointer)
                allocated = sum(C.sizeof(host) for host in hosts)
                self.peak_explicit_device_bytes = max(self.peak_explicit_device_bytes, allocated)
                for pointer, host in zip(buffers[:3], hosts[:3]):
                    self._check("cuMemcpyHtoD_v2", pointer, host, C.sizeof(host))
                arguments = (*buffers, C.c_uint32(rows), C.c_uint32(cols), C.c_float(global_weight_scale))
                parameters = (C.c_void_p * len(arguments))(
                    *(C.cast(C.byref(argument), C.c_void_p).value for argument in arguments))
                self.operations += 1
                self._check("cuLaunchKernel", self._nvfp4_function,
                            (rows + 31) // 32, 1, 1, 32, 1, 1, 0, None, parameters, None)
                self._check("cuCtxSynchronize")
                self._check("cuMemcpyDtoH_v2", host_output, buffers[3], C.sizeof(host_output))
                result = list(host_output)
                if not all(math.isfinite(value) for value in result):
                    raise CudaProbeError("NVFP4 FP32 arithmetic overflowed; use smaller vector/scale values")
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
            if self._nvfp4_module.value:
                with self._activate():
                    for operation in ("cuCtxSynchronize", "cuModuleUnload"):
                        try:
                            arguments = (self._nvfp4_module,) if operation == "cuModuleUnload" else ()
                            self._check(operation, *arguments)
                        except CudaProbeError as error:
                            errors.append(error)
        except CudaProbeError as error:
            errors.append(error)
        finally:
            self._nvfp4_module = C.c_void_p()
            self._nvfp4_function = C.c_void_p()
            try:
                super().close()
            except CudaProbeError as error:
                errors.append(error)
        _cleanup_errors(errors, None)
