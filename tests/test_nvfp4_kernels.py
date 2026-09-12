"""Independent format arithmetic plus opt-in actual native CPU and CUDA checks."""
import ctypes as C
import math
import os
import random
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from glm_local.cuda_probe import CudaProbeError
from glm_local.nvfp4_kernels import (CudaNVFP4TileBackend, MAX_EXPLICIT_DEVICE_BYTES,
    NativeNVFP4CpuBackend, validate_nvfp4_tile)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fp4_oracle(nibble):
    exponent, mantissa = divmod(nibble & 7, 2)
    magnitude = mantissa / 2 if exponent == 0 else (1 + mantissa / 2) * 2 ** (exponent - 1)
    return math.copysign(magnitude, -1 if nibble & 8 else 1)


def scale_oracle(byte):
    exponent, mantissa = divmod(byte, 8)
    return mantissa / 512 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)


def oracle(packed, rows, cols, scales, vector, global_scale):
    """Use mathematical fields, no product-code lookup table or decoder helper."""
    result = []
    for row in range(rows):
        total = 0.0
        for col in range(cols):
            byte = packed[row * (cols // 2) + col // 2]
            nibble = byte % 16 if col % 2 == 0 else byte // 16
            combined_scale = f32(scale_oracle(scales[row * (cols // 16) + col // 16]) * f32(global_scale))
            weight = f32(fp4_oracle(nibble) * combined_scale)
            total = f32(total + f32(weight * f32(vector[col])))
        result.append(total)
    return result


def cases():
    rng = random.Random(2917)
    # All 256 packed byte values, with independently distinguishable input columns.
    yield bytes(range(256)), 32, 16, bytes([56] * 32), list(range(-7, 9)), 0.125
    # Every nonnegative finite E4M3FN scale including zero and subnormals.
    yield bytes([0x33] * (127 * 8)), 127, 16, bytes(range(127)), [1] + [0] * 15, 0.137
    for rows, cols in ((1, 16), (7, 32), (33, 80), (128, 128)):
        yield (bytes(rng.randrange(256) for _ in range(rows * cols // 2)), rows, cols,
               bytes(rng.randrange(127) for _ in range(rows * cols // 16)),
               [rng.uniform(-1, 1) for _ in range(cols)], 0.0137)
    yield b"\x11" * 8, 1, 16, b"\x01", [1] + [0] * 15, 2 ** -126


class ValidationTests(unittest.TestCase):
    def test_validation_rounds_inputs_and_bounds_explicit_device_memory(self):
        values, scale = validate_nvfp4_tile(b"\0" * 8, 1, 16, b"\0", [0.1] * 16, 0.2)
        self.assertEqual(values, (f32(0.1),) * 16)
        self.assertEqual(scale, f32(0.2))
        self.assertEqual(MAX_EXPLICIT_DEVICE_BYTES, 10240)

    def test_invalid_inputs_fail_before_native_call(self):
        good = (b"\x12" * 8, 1, 16, b"\x38", [1.0] * 16, 0.5)
        bad = [(0, value) for value in (b"", b"\0" * 7, b"\0" * 9, bytearray(8), None)]
        bad += [(1, value) for value in (0, 129, True, 1.5)]
        bad += [(2, value) for value in (0, 15, 17, 144, True)]
        bad += [(3, value) for value in (b"", b"\0\0", b"\x7f", b"\x80", b"\xff", bytearray(1), None)]
        bad += [(4, value) for value in ([], [1] * 17, "1" * 16, b"1" * 16, iter([1] * 16), None)]
        bad += [(4, [value] * 16) for value in (math.inf, math.nan, 1e40, True, "1")]
        bad += [(5, value) for value in (0, -1, math.inf, math.nan, 1e40, 1e-50, True, "1")]
        backend = NativeNVFP4CpuBackend.__new__(NativeNVFP4CpuBackend)
        backend._dll = Mock()
        for index, value in bad:
            arguments = list(good)
            arguments[index] = value
            with self.subTest(index=index, value=str(value)[:50]), self.assertRaises(ValueError):
                backend.matvec_nvfp4_tile(*arguments)
        backend._dll.nvfp4_cpu_matvec_tile.assert_not_called()

    def test_error_status_and_closed_native_wrapper_fail(self):
        backend = NativeNVFP4CpuBackend.__new__(NativeNVFP4CpuBackend)
        backend._dll = Mock()
        for status, kind in ((1, ValueError), (2, ValueError), (99, RuntimeError)):
            backend._dll.nvfp4_cpu_matvec_tile.return_value = status
            with self.subTest(status=status), self.assertRaises(kind):
                backend.matvec_nvfp4_tile(b"\0" * 8, 1, 16, b"\0", [1] * 16, 1)
        backend.close()
        backend.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.__enter__()

    @unittest.skipUnless(os.name == "nt" and C.sizeof(C.c_void_p) == 8, "requires x64 Windows")
    def test_missing_dll_does_not_build_or_download(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "build-native.bat"):
                NativeNVFP4CpuBackend(Path(folder) / "missing.dll")
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_invalid_cuda_device_or_budget_fails_before_driver_load(self):
        with patch("glm_local.cuda_probe._load_driver") as load:
            for arguments in ((-1, 256), (True, 256), (0, 0), (0, 65537), (0, True)):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    CudaNVFP4TileBackend(arguments[0], max_operations=arguments[1])
            load.assert_not_called()


class CudaCleanupTests(unittest.TestCase):
    def backend(self, failure=None, failure_number=1):
        gpu = CudaNVFP4TileBackend.__new__(CudaNVFP4TileBackend)
        gpu._thread_id = threading.get_ident()
        gpu._context, gpu._module, gpu._function = C.c_void_p(100), C.c_void_p(200), C.c_void_p(300)
        gpu._nvfp4_module, gpu._nvfp4_function = C.c_void_p(400), C.c_void_p(500)
        gpu.closed, gpu.operations, gpu.peak_explicit_device_bytes = False, 0, 0
        events, counts = [], {}

        def check(operation, *args):
            events.append((operation, args))
            counts[operation] = counts.get(operation, 0) + 1
            if operation == failure and counts[operation] == failure_number:
                raise CudaProbeError(f"injected {operation}")
            if operation == "cuMemAlloc_v2":
                args[0]._obj.value = counts[operation]
            if operation == "cuModuleLoadData":
                args[0]._obj.value = 400
            if operation == "cuCtxPopCurrent_v2":
                args[0]._obj.value = gpu._context.value

        gpu._check = check
        return gpu, events

    def run_tile(self, gpu):
        return gpu.matvec_nvfp4_tile(b"\x22" * 8, 1, 16, b"\x38", [1] * 16, 0.5)

    def test_allocation_failure_frees_only_already_allocated_buffers(self):
        for failure_number in (1, 2, 3, 4):
            gpu, events = self.backend("cuMemAlloc_v2", failure_number)
            with self.subTest(failure_number=failure_number), self.assertRaises(CudaProbeError):
                self.run_tile(gpu)
            self.assertEqual([args[0].value for op, args in events if op == "cuMemFree_v2"],
                             list(range(failure_number - 1, 0, -1)))
            self.assertEqual(gpu.operations, 0)
            self.assertEqual(events[-1][0], "cuCtxPopCurrent_v2")

    def test_copy_launch_sync_and_readback_errors_free_all_four_buffers(self):
        for failure in ("cuMemcpyHtoD_v2", "cuLaunchKernel", "cuCtxSynchronize", "cuMemcpyDtoH_v2"):
            gpu, events = self.backend(failure)
            with self.subTest(failure=failure), self.assertRaisesRegex(CudaProbeError, failure):
                self.run_tile(gpu)
            self.assertEqual([args[0].value for op, args in events if op == "cuMemFree_v2"], [4, 3, 2, 1])
            self.assertEqual(events[-1][0], "cuCtxPopCurrent_v2")

    def test_free_failure_keeps_cleaning_up_and_restores_context(self):
        gpu, events = self.backend("cuMemFree_v2")
        with self.assertRaisesRegex(CudaProbeError, "cleanup failed"):
            self.run_tile(gpu)
        self.assertEqual([args[0].value for op, args in events if op == "cuMemFree_v2"], [4, 3, 2, 1])
        self.assertEqual(events[-1][0], "cuCtxPopCurrent_v2")

    def test_close_unloads_both_modules_and_destroys_context_even_after_errors(self):
        for failure in (None, "cuCtxSynchronize", "cuModuleUnload", "cuCtxPushCurrent_v2"):
            gpu, events = self.backend(failure)
            if failure is None:
                gpu.close()
            else:
                with self.subTest(failure=failure), self.assertRaises(CudaProbeError):
                    gpu.close()
            self.assertEqual(events[-1][0], "cuCtxDestroy_v2")
            self.assertTrue(gpu.closed)
            count = len(events)
            gpu.close()
            self.assertEqual(len(events), count)

    def test_close_preserves_primary_context_manager_error(self):
        gpu, events = self.backend("cuModuleUnload")
        with self.assertRaisesRegex(ValueError, "body failure"):
            with gpu:
                raise ValueError("body failure")
        self.assertTrue(gpu.closed)
        self.assertEqual(events[-1][0], "cuCtxDestroy_v2")

    def test_operation_budget_and_thread_owner_enforced_before_driver_calls(self):
        gpu, events = self.backend()
        gpu.operations = 256
        with self.assertRaisesRegex(CudaProbeError, "operation limit"):
            self.run_tile(gpu)
        self.assertEqual(events, [])
        with patch("glm_local.cuda_probe.threading.get_ident", return_value=-1):
            with self.assertRaisesRegex(CudaProbeError, "creating thread"):
                gpu.close()
        self.assertEqual(events, [])

    def test_nvfp4_constructor_load_failures_release_base_context(self):
        for failure in ("cuModuleLoadData", "cuModuleGetFunction"):
            template, events = self.backend(failure)

            def initialize(instance, *_args, **_kwargs):
                for name in ("_thread_id", "_context", "_module", "_function", "closed",
                             "operations", "peak_explicit_device_bytes", "_check"):
                    setattr(instance, name, getattr(template, name))

            with patch("glm_local.nvfp4_kernels.CudaTileBackend.__init__", initialize):
                with self.subTest(failure=failure), self.assertRaisesRegex(CudaProbeError, failure):
                    CudaNVFP4TileBackend()
            self.assertEqual(events[-1][0], "cuCtxDestroy_v2")
            unloaded = [args[0].value for operation, args in events if operation == "cuModuleUnload"]
            self.assertEqual(unloaded, [200] if failure == "cuModuleLoadData" else [400, 200])


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "set GLM_TEST_NATIVE=1")
class NativeNVFP4IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.cpu = NativeNVFP4CpuBackend()
        self.addCleanup(self.cpu.close)

    def test_all_nibbles_scales_and_random_tiles_match_independent_fp32_oracle(self):
        for arguments in cases():
            with self.subTest(rows=arguments[1], cols=arguments[2], scale=arguments[-1]):
                self.assertEqual(self.cpu.matvec_nvfp4_tile(*arguments), oracle(*arguments))

    def test_zero_scales_zero_block_without_division(self):
        self.assertEqual(self.cpu.matvec_nvfp4_tile(b"\x77" * 8, 1, 16, b"\0", [1] * 16, 2), [0.0])

    def test_c_abi_rejects_mismatched_lengths_before_buffer_access(self):
        packed, scales, vector, output = (C.c_uint8 * 1)(0), (C.c_uint8 * 1)(0), (C.c_float * 1)(1), (C.c_float * 1)(123)
        status = self.cpu._dll.nvfp4_cpu_matvec_tile(packed, 1, 128, 128, scales, 1, vector, 1, 1, output, 1)
        self.assertEqual(status, 1)
        self.assertEqual(output[0], 123)

    def test_intermediate_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.cpu.matvec_nvfp4_tile(b"\xf7" * 8, 1, 16, b"\x7e", [1] * 16, 3e38)

    def test_metadata_states_weight_only_reference_scope(self):
        self.assertEqual(self.cpu.metadata["activation_quantization"], "none")
        self.assertEqual(self.cpu.metadata["abi_version"], 1)
        self.assertFalse(self.cpu.metadata["native_nvfp4_instructions"])
        self.assertFalse(self.cpu.metadata["full_model_inference"])


@unittest.skipUnless(os.environ.get("GLM_TEST_CUDA") == "1", "set GLM_TEST_CUDA=1")
class RealNVFP4CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = CudaNVFP4TileBackend()

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def test_real_cuda_all_packed_codes_scales_and_random_tiles_match_oracle(self):
        for arguments in cases():
            with self.subTest(rows=arguments[1], cols=arguments[2], scale=arguments[-1]):
                self.assertEqual(self.gpu.matvec_nvfp4_tile(*arguments), oracle(*arguments))
        self.assertLessEqual(self.gpu.peak_explicit_device_bytes, MAX_EXPLICIT_DEVICE_BYTES)

    def test_real_cuda_zero_scale_and_overflow(self):
        self.assertEqual(self.gpu.matvec_nvfp4_tile(b"\x77" * 8, 1, 16, b"\0", [1] * 16, 2), [0.0])
        with self.assertRaisesRegex(CudaProbeError, "overflowed"):
            self.gpu.matvec_nvfp4_tile(b"\xf7" * 8, 1, 16, b"\x7e", [1] * 16, 3e38)

    def test_inherited_fp8_method_uses_its_own_module(self):
        self.assertEqual(self.gpu.matvec_tile(b"\x38", 1, 1, [2.0], 0.5), [1.0])
        self.assertEqual(self.gpu.metadata["activation_quantization"], "none")
        self.assertFalse(self.gpu.device_info["native_nvfp4_instructions"])


if __name__ == "__main__":
    unittest.main()
