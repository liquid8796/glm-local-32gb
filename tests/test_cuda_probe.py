"""Bounded numerical validation; opt-in CUDA tests must really use the GPU."""

import ctypes
import math
import os
import random
import struct
import threading
import unittest
from unittest.mock import patch

from glm_local.cuda_probe import (
    CudaProbeError, CudaTileBackend, MAX_EXPLICIT_DEVICE_BYTES, validate_tile,
)


def decode_oracle(byte):
    """Independent arithmetic interpretation of the E4M3FN bit fields."""
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    magnitude = (math.ldexp(mantissa, -9) if exponent == 0
                 else math.ldexp(1 + mantissa / 8, exponent - 7))
    return math.copysign(magnitude, -1.0 if byte & 128 else 1.0)


def f32(number):
    return struct.unpack("<f", struct.pack("<f", number))[0]


def matvec_oracle(weights, rows, cols, vector, scale):
    result = []
    scale = f32(scale)
    for row in range(rows):
        accumulator = 0.0
        for col in range(cols):
            scaled = f32(decode_oracle(weights[row * cols + col]) * scale)
            product = f32(scaled * f32(vector[col]))
            accumulator = f32(accumulator + product)
        result.append(accumulator)
    return result


class TileValidationTests(unittest.TestCase):
    def test_valid_tile_rounds_inputs_to_float32(self):
        values, scale = validate_tile(b"\x38\xb8", 1, 2, [0.1, -0.2], 0.3)
        self.assertEqual(values, (f32(0.1), f32(-0.2)))
        self.assertEqual(scale, f32(0.3))
        self.assertEqual(MAX_EXPLICIT_DEVICE_BYTES, 17408)

    def test_invalid_sizes_and_packed_data_fail(self):
        for rows, cols in ((0, 1), (1, 0), (129, 1), (1, 129), (True, 1),
                           (1, 1.0), (-1, 1), (1, "1")):
            with self.subTest(rows=rows, cols=cols), self.assertRaises(ValueError):
                validate_tile(b"\0", rows, cols, [1.0], 1.0)
        for weights in (b"", b"\0\0", bytearray(b"\0"), [0], "0", None):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                validate_tile(weights, 1, 1, [1.0], 1.0)

    def test_nonfinite_nan_and_unbounded_values_fail(self):
        for weights in (b"\x7f", b"\xff"):
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, "NaN"):
                validate_tile(weights, 1, 1, [1.0], 1.0)
        for value in (math.inf, -math.inf, math.nan, 1e39, 10 ** 1000, True, "1"):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                validate_tile(b"\0", 1, 1, [value], 1.0)
        for scale in (0, -1, 1e-50, math.inf, math.nan, 1e39, True, "1"):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                validate_tile(b"\0", 1, 1, [1.0], scale)
        for vector in ([], [1, 2], "1", b"1", None, iter([1]), {1: 1}):
            with self.subTest(vector=str(vector)), self.assertRaises(ValueError):
                validate_tile(b"\0", 1, 1, vector, 1.0)

    def test_cuda_unavailable_fails_explicitly(self):
        with patch("glm_local.cuda_probe._load_driver", side_effect=CudaProbeError("no CUDA")):
            with self.assertRaisesRegex(CudaProbeError, "no CUDA"):
                CudaTileBackend()

    def test_invalid_device_does_not_load_driver(self):
        with patch("glm_local.cuda_probe._load_driver") as load:
            for index in (-1, True, "0", 0.5):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    CudaTileBackend(index)
            load.assert_not_called()


class CudaFailureCleanupTests(unittest.TestCase):
    def backend(self, failure=None, failure_number=1):
        """Exercise actual resource orchestration with a deterministic driver double."""
        gpu = object.__new__(CudaTileBackend)
        gpu._thread_id = threading.get_ident()
        gpu._context = ctypes.c_void_p(100)
        gpu._module = ctypes.c_void_p(200)
        gpu._function = ctypes.c_void_p(300)
        gpu.closed = False
        gpu.operations = 0
        gpu.peak_explicit_device_bytes = 0
        events, counts = [], {}

        def check(operation, *args):
            counts[operation] = counts.get(operation, 0) + 1
            events.append((operation, args))
            if operation == failure and counts[operation] == failure_number:
                raise CudaProbeError(f"injected {operation}")
            if operation == "cuMemAlloc_v2":
                args[0]._obj.value = counts[operation]
            if operation == "cuCtxPopCurrent_v2":
                args[0]._obj.value = gpu._context.value

        gpu._check = check
        return gpu, events

    def test_partial_allocation_failure_frees_successful_allocation(self):
        gpu, events = self.backend("cuMemAlloc_v2", 2)
        with self.assertRaisesRegex(CudaProbeError, "cuMemAlloc_v2"):
            gpu.matvec_tile(b"\x38", 1, 1, [1.0], 1.0)
        freed = [args[0].value for name, args in events if name == "cuMemFree_v2"]
        self.assertEqual(freed, [1])
        self.assertEqual(events[-1][0], "cuCtxPopCurrent_v2")
        self.assertEqual(gpu.operations, 0)

    def test_copy_launch_sync_and_readback_failures_free_every_buffer(self):
        for failure in ("cuMemcpyHtoD_v2", "cuLaunchKernel", "cuCtxSynchronize",
                        "cuMemcpyDtoH_v2"):
            with self.subTest(failure=failure):
                gpu, events = self.backend(failure)
                with self.assertRaisesRegex(CudaProbeError, failure):
                    gpu.matvec_tile(b"\x38", 1, 1, [1.0], 1.0)
                freed = [args[0].value for name, args in events if name == "cuMemFree_v2"]
                self.assertEqual(freed, [3, 2, 1])
                self.assertEqual(events[-1][0], "cuCtxPopCurrent_v2")

    def test_free_failure_still_attempts_other_buffers_and_restores_context(self):
        gpu, events = self.backend("cuMemFree_v2")
        with self.assertRaisesRegex(CudaProbeError, "cleanup failed"):
            gpu.matvec_tile(b"\x38", 1, 1, [1.0], 1.0)
        freed = [args[0].value for name, args in events if name == "cuMemFree_v2"]
        self.assertEqual(freed, [3, 2, 1])
        self.assertEqual(events[-1][0], "cuCtxPopCurrent_v2")

    def test_close_destroys_context_even_when_synchronization_or_unload_fails(self):
        for failure in ("cuCtxSynchronize", "cuModuleUnload", "cuCtxPushCurrent_v2"):
            with self.subTest(failure=failure):
                gpu, events = self.backend(failure)
                with self.assertRaisesRegex(CudaProbeError, failure):
                    gpu.close()
                self.assertEqual(events[-1][0], "cuCtxDestroy_v2")
                self.assertTrue(gpu.closed)
                event_count = len(events)
                gpu.close()
                self.assertEqual(len(events), event_count)

    def test_context_manager_preserves_body_error_during_cleanup_failure(self):
        gpu, events = self.backend("cuModuleUnload")
        with self.assertRaisesRegex(ValueError, "body failure"):
            with gpu:
                raise ValueError("body failure")
        self.assertTrue(gpu.closed)
        self.assertEqual(events[-1][0], "cuCtxDestroy_v2")

    def test_operation_limit_and_thread_mismatch_do_not_call_driver(self):
        gpu, events = self.backend()
        gpu.operations = 256
        with self.assertRaisesRegex(CudaProbeError, "operation limit"):
            gpu.matvec_tile(b"\x38", 1, 1, [1.0], 1.0)
        self.assertFalse(events)
        with patch("glm_local.cuda_probe.threading.get_ident", return_value=-1):
            with self.assertRaisesRegex(CudaProbeError, "creating thread"):
                gpu.close()
        self.assertFalse(events)


@unittest.skipUnless(os.environ.get("GLM_TEST_CUDA") == "1", "Set GLM_TEST_CUDA=1 for real CUDA")
class RealCudaNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No availability skip: once explicitly requested, any CUDA failure is a test failure.
        cls.gpu = CudaTileBackend()

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def assert_tile(self, weights, rows, cols, vector, scale):
        expected = matvec_oracle(weights, rows, cols, vector, scale)
        actual = self.gpu.matvec_tile(weights, rows, cols, vector, scale)
        self.assertEqual(actual, expected)
        self.assertLessEqual(self.gpu.peak_explicit_device_bytes, MAX_EXPLICIT_DEVICE_BYTES)

    def test_every_finite_encoding_and_signed_zero(self):
        finite = [value for value in range(256) if (value & 127) != 127]
        # Only two small launches cover all 254 finite encodings independently.
        for offset in range(0, len(finite), 128):
            weights = bytes(finite[offset:offset + 128])
            self.assert_tile(weights, len(weights), 1, [1.0], 1.0)
        self.assertEqual(decode_oracle(0x7E), 448.0)
        self.assertEqual(decode_oracle(0xFE), -448.0)
        self.assertEqual(decode_oracle(0x01), 2 ** -9)
        self.assertEqual(math.copysign(1.0, decode_oracle(0x80)), -1.0)

    def test_rows_columns_scaling_and_fp32_accumulation(self):
        rng = random.Random(2409)
        finite = [value for value in range(256) if (value & 127) != 127]
        for rows, cols, scale in ((1, 1, 0.1), (7, 3, 0.015625), (33, 65, 0.1234567),
                                 (128, 128, 0.03125), (1, 128, 1.0)):
            weights = bytes(rng.choice(finite) for _ in range(rows * cols))
            vector = [rng.uniform(-2, 2) for _ in range(cols)]
            self.assert_tile(weights, rows, cols, vector, scale)

    def test_fp32_subnormal_result_is_preserved(self):
        self.assert_tile(b"\x01\x81", 2, 1, [1.0], 2.0 ** -126)

    def test_rejected_arguments_do_not_launch(self):
        before = self.gpu.operations
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.gpu.matvec_tile(b"\x7f", 1, 1, [1], 1)
        self.assertEqual(self.gpu.operations, before)

    def test_real_device_and_idempotent_close(self):
        self.assertGreaterEqual(float(self.gpu.device_info["compute_capability"]), 8.0)
        self.assertFalse(self.gpu.device_info["native_fp8_instructions"])
        with CudaTileBackend() as other:
            self.assertEqual(other.matvec_tile(b"\x38", 1, 1, [2.0], 0.5), [1.0])
        other.close()
        with self.assertRaisesRegex(CudaProbeError, "closed"):
            other.matvec_tile(b"\x38", 1, 1, [2.0], 0.5)


if __name__ == "__main__":
    unittest.main()
