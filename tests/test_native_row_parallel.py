"""Exact row-band arithmetic, bounded row workers, mutable buffers and close races."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import math
import os
import random
import struct
import threading
import unittest
from unittest.mock import patch

from glm_local.cpu_probe import NativeCpuBackend
from glm_local.nvfp4_kernels import NativeNVFP4CpuBackend
from test_dense_cpu import encode, f32
from test_nvfp4_row_band import old_tile_reduction


def old_dense_tiles(cpu, raw, rows, cols, vector, dtype):
    itemsize = 4 if dtype == "F32" else 2
    output = [0.0] * rows
    for column in range(0, cols, 128):
        count = min(128, cols - column)
        packed = b"".join(raw[(row * cols + column) * itemsize:(row * cols + column + count) * itemsize] for row in range(rows))
        partial = cpu.matvec_dense_tile(packed, rows, count, vector[column:column + count], dtype)
        output = [f32(left + right) for left, right in zip(output, partial)]
    return output


def packed_floats(values):
    return struct.pack(f"<{len(values)}f", *values)


class RowThreadArgumentTests(unittest.TestCase):
    def test_thread_bounds_are_rejected_before_loading_a_library(self):
        with patch("glm_local.cpu_probe.ctypes.CDLL") as load:
            for backend in (NativeCpuBackend, NativeNVFP4CpuBackend):
                for threads in (0, -1, 9, True, 1.0, "8"):
                    with self.subTest(backend=backend, threads=threads), self.assertRaises(ValueError):
                        backend(row_band_threads=threads)
            load.assert_not_called()


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "set GLM_TEST_NATIVE=1")
class ParallelNativeRowTests(unittest.TestCase):
    def setUp(self):
        self.dense1 = NativeCpuBackend(row_band_threads=1)
        self.dense8 = NativeCpuBackend(row_band_threads=8)
        self.nv1 = NativeNVFP4CpuBackend(row_band_threads=1)
        self.nv8 = NativeNVFP4CpuBackend(row_band_threads=8)
        for backend in (self.dense1, self.dense8, self.nv1, self.nv8):
            self.addCleanup(backend.close)

    def test_dense_one_and_eight_workers_match_old_tile_order_for_three_storage_formats(self):
        rng = random.Random(29111)
        for dtype in ("BF16", "F16", "F32"):
            for rows, cols in ((1, 1), (7, 144), (128, 512), (128, 6144), (3, 16384)):
                raw = bytearray(encode([rng.uniform(-2, 2) for _ in range(rows * cols)], dtype))
                before = bytes(raw)
                vector = [rng.uniform(-1, 1) for _ in range(cols)]
                expected = old_dense_tiles(self.dense1, before, rows, cols, vector, dtype)
                for cpu in (self.dense1, self.dense8):
                    actual = cpu.matvec_dense_row_band(raw, rows, cols, cpu.prepare_dense_vector(vector), dtype)
                    with self.subTest(dtype=dtype, rows=rows, cols=cols, threads=cpu.row_band_threads):
                        self.assertEqual(packed_floats(actual), packed_floats(expected))
                self.assertEqual(bytes(raw), before)
        self.assertTrue(self.dense8._row_pool._pointer.value)

    def test_nvfp4_one_and_eight_workers_match_old_tiles_with_zero_copy_buffers(self):
        rng = random.Random(37197)
        for rows, cols in ((7, 144), (128, 512), (128, 6144), (3, 16384)):
            packed = bytearray(rng.randrange(256) for _ in range(rows * cols // 2))
            scales = bytearray(rng.randrange(127) for _ in range(rows * cols // 16))
            vector = [rng.uniform(-2, 2) for _ in range(cols)]
            expected = old_tile_reduction(self.nv1, bytes(packed), rows, cols, bytes(scales), vector, 0.017)
            for cpu in (self.nv1, self.nv8):
                actual = cpu.matvec_nvfp4_row_band(packed, rows, cols, scales, cpu.prepare_nvfp4_vector(vector), 0.017)
                with self.subTest(rows=rows, cols=cols, threads=cpu.row_band_threads):
                    self.assertEqual(packed_floats(actual), packed_floats(expected))
        self.assertTrue(self.nv8._row_pool._pointer.value)

    def test_parallel_dense_subnormal_and_partial_tile_rounding_are_exact(self):
        for dtype, unit, vector in (("BF16", b"\x01\0", [1.0] * 512),
                                   ("F16", b"\x01\0", [2**-125] * 512),
                                   ("F32", b"\x01\0\0\0", [1.0] * 512)):
            raw = unit * (128 * 512)
            expected = self.dense1.matvec_dense_row_band(raw, 128, 512, vector, dtype)
            self.assertEqual(packed_floats(self.dense8.matvec_dense_row_band(raw, 128, 512, vector, dtype)), packed_floats(expected))
        vector = [0.0] * 144
        vector[0], vector[128], vector[129], vector[130] = 1e8, 1, -1e8, 1
        self.assertEqual(self.dense8.matvec_dense_row_band(encode([1.0] * 144, "F32"), 1, 144, vector, "F32"), [0.0])

    def test_nonfinite_in_a_worker_row_and_accumulation_overflow_fail_without_partial_output(self):
        for cpu in (self.dense1, self.dense8):
            raw = bytearray(encode([1.0] * (128 * 512), "F32"))
            raw[-4:] = struct.pack("<f", math.nan)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                cpu.matvec_dense_row_band(raw, 128, 512, [1] * 512, "F32")
            raw = encode([1.0] * (128 * 512), "F32")
            vector = [0.0] * 512
            vector[0] = vector[128] = 2e38
            with self.assertRaisesRegex(ValueError, "non-finite"):
                cpu.matvec_dense_row_band(raw, 128, 512, vector, "F32")

    def test_dense_band_bounds_and_prepared_buffer_validation(self):
        good = [b"\x80\x3f", 1, 1, [1], "BF16"]
        for index, value in ((0, b""), (0, bytearray(1)), (1, 129), (1, True), (2, 0), (2, 16385),
                             (3, []), (3, [math.nan]), (3, [True]), (3, "1"), (4, "U8")):
            args = list(good)
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                self.dense8.matvec_dense_row_band(*args)
        prepared = self.dense8.prepare_dense_vector([1])
        prepared.buffer = (C.c_double * 1)(1)
        with self.assertRaisesRegex(ValueError, "float32"):
            self.dense8.matvec_dense_row_band(*good[:3], prepared, "BF16")
        raw, vector, output = (C.c_uint8 * 2)(0x80, 0x3f), (C.c_float * 1)(1), (C.c_float * 1)(123)
        status = self.dense8._dll.fp8_cpu_matvec_dense_row_band(raw, 2, 128, 16384, 1, vector, 1, output, 1, None)
        self.assertEqual(status, 1)
        self.assertEqual(output[0], 123)

    def test_same_instance_concurrent_calls_are_serialized_and_close_waits_for_ffi(self):
        for cpu, symbol, operation in (
            (self.dense8, "fp8_cpu_matvec_dense_row_band", lambda backend: backend.matvec_dense_row_band(b"\x80\x3f", 1, 1, [1], "BF16")),
            (self.nv8, "nvfp4_cpu_matvec_row_band_parallel", lambda backend: backend.matvec_nvfp4_row_band(b"\x22" * 8, 1, 16, b"\x38", [1] * 16, 1))):
            entered, release, closing = threading.Event(), threading.Event(), threading.Event()
            def blocking(*_):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("test FFI release deadline")
                return 0
            original = getattr(cpu._dll, symbol)
            setattr(cpu._dll, symbol, blocking)
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    call = executor.submit(operation, cpu)
                    self.assertTrue(entered.wait(5))
                    def close():
                        closing.set()
                        cpu.close()
                    close_call = executor.submit(close)
                    self.assertTrue(closing.wait(5))
                    self.assertFalse(close_call.done())
                    release.set()
                    self.assertEqual(call.result(timeout=5), [0.0])
                    close_call.result(timeout=5)
                self.assertFalse(cpu._row_pool._pointer.value)
                cpu.close()
            finally:
                release.set()
                if cpu._dll is not None:
                    setattr(cpu._dll, symbol, original)

    def test_metadata_keeps_thread_limit_and_existing_tile_semantics_explicit(self):
        self.assertEqual(self.dense8.row_band_threads, 8)
        self.assertEqual(self.nv8.row_band_threads, 8)
        self.assertEqual(self.dense1.row_band_threads, 1)
        self.assertEqual(self.nv1.row_band_threads, 1)
        self.assertEqual(self.dense8.metadata["threads"], 1)
        self.assertEqual(self.dense8.metadata["row_band_threads"], 8)
        self.assertEqual(self.nv8.metadata["row_band_threads"], 8)
        self.assertIsNone(self.dense8._dll.fp8_cpu_row_pool_create(9))
        self.assertIsNone(self.nv8._dll.nvfp4_cpu_row_pool_create(0))


if __name__ == "__main__":
    unittest.main()
