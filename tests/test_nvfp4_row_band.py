"""Bounded native NVFP4 bands must preserve the established128-column arithmetic."""
import ctypes as C
import math
import os
import random
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from glm_local.nvfp4_kernels import NativeNVFP4CpuBackend


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def old_tile_reduction(cpu, packed, rows, cols, scales, vector, global_scale):
    output = [0.0] * rows
    for col in range(0, cols, 128):
        width = min(128, cols - col)
        weights = b"".join(packed[row * (cols // 2) + col // 2:row * (cols // 2) + (col + width) // 2] for row in range(rows))
        scale = b"".join(scales[row * (cols // 16) + col // 16:row * (cols // 16) + (col + width) // 16] for row in range(rows))
        partial = cpu.matvec_nvfp4_tile(weights, rows, width, scale, vector[col:col + width], global_scale)
        output = [f32(a + b) for a, b in zip(output, partial)]
    return output


class RowBandBoundaryTests(unittest.TestCase):
    def backend(self):
        cpu = NativeNVFP4CpuBackend.__new__(NativeNVFP4CpuBackend)
        function = Mock(return_value=0)
        cpu._dll = SimpleNamespace(nvfp4_cpu_matvec_row_band=function)
        return cpu, function

    def test_invalid_band_metadata_or_payload_never_calls_c(self):
        good = [b"\x22" * 8, 1, 16, b"\x38", [1] * 16, 1]
        invalid = [(0, b""), (0, bytearray(7)), (1, 0), (1, 129), (1, True), (2, 15), (2, 16385),
                   (2, False), (3, b""), (3, b"\x7f"), (3, b"\xff"), (3, b"\x80"),
                   (4, [1] * 15), (4, [math.nan] * 16), (4, [True] * 16), (4, "1" * 16),
                   (4, None), (5, 0), (5, -1), (5, math.inf), (5, 1e40)]
        cpu, function = self.backend()
        for index, value in invalid:
            args = list(good)
            args[index] = value
            with self.subTest(index=index, value=str(value)[:32]), self.assertRaises(ValueError):
                cpu.matvec_nvfp4_row_band(*args)
        function.assert_not_called()

    def test_prepared_vector_and_old_library_failure_are_explicit(self):
        cpu, function = self.backend()
        prepared = cpu.prepare_nvfp4_vector([0.1] * 16)
        self.assertEqual(len(prepared), 16)
        self.assertEqual(prepared.buffer[0], f32(0.1))
        cpu.matvec_nvfp4_row_band(b"\0" * 8, 1, 16, b"\0", prepared, 1)
        function.assert_called_once()
        cpu._dll = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "build-native.bat"):
            cpu.matvec_nvfp4_row_band(b"\0" * 8, 1, 16, b"\0", prepared, 1)
        cpu.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            cpu.prepare_nvfp4_vector([1] * 16)

    def test_native_status_is_not_returned_as_success(self):
        cpu, function = self.backend()
        for status, expected in ((1, ValueError), (2, ValueError), (99, RuntimeError)):
            function.return_value = status
            with self.subTest(status=status), self.assertRaises(expected):
                cpu.matvec_nvfp4_row_band(b"\0" * 8, 1, 16, b"\0", [1] * 16, 1)


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "set GLM_TEST_NATIVE=1")
class NativeRowBandTests(unittest.TestCase):
    def setUp(self):
        self.cpu = NativeNVFP4CpuBackend()
        self.addCleanup(self.cpu.close)

    def test_row_band_is_bitwise_identical_to_old_tile_reduction_at_edges_and_large_width(self):
        rng = random.Random(24107)
        for rows, cols in ((1, 16), (7, 144), (128, 144), (128, 6144), (3, 16384)):
            packed = bytes(rng.randrange(256) for _ in range(rows * cols // 2))
            scales = bytes(rng.randrange(127) for _ in range(rows * cols // 16))
            vector = [rng.uniform(-2, 2) for _ in range(cols)]
            expected = old_tile_reduction(self.cpu, packed, rows, cols, scales, vector, 0.0137)
            prepared = self.cpu.prepare_nvfp4_vector(vector)
            actual = self.cpu.matvec_nvfp4_row_band(packed, rows, cols, scales, prepared, 0.0137)
            with self.subTest(rows=rows, cols=cols):
                self.assertEqual(struct.pack(f"<{rows}f", *actual), struct.pack(f"<{rows}f", *expected))
        self.assertTrue(self.cpu.supports_nvfp4_row_band)

    def test_tile_boundary_rounding_is_preserved_instead_of_one_long_dot_product(self):
        vector = [0.0] * 144
        vector[0], vector[128], vector[129], vector[130] = 1e8, 1, -1e8, 1
        self.assertEqual(self.cpu.matvec_nvfp4_row_band(b"\x22" * 72, 1, 144, b"\x38" * 9, vector, 1), [0.0])
        # A naive continuous accumulation would produce1.0 for this fixture.

    def test_zero_scales_and_subnormal_results_match_tiles(self):
        for code, global_scale in ((0, 1), (1, 2**-126), (126, 0.03125)):
            packed, scales, vector = b"\x91" * (3 * 72), bytes([code] * 27), [1.0] * 144
            actual = self.cpu.matvec_nvfp4_row_band(packed, 3, 144, scales, vector, global_scale)
            self.assertEqual(actual, old_tile_reduction(self.cpu, packed, 3, 144, scales, vector, global_scale))

    def test_overflow_inside_and_between128_column_subtotals_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.cpu.matvec_nvfp4_row_band(b"\x77" * 72, 1, 144, b"\x7e" * 9, [1] * 144, 3e38)
        vector = [0.0] * 256
        vector[0] = vector[128] = 2e38
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.cpu.matvec_nvfp4_row_band(b"\x22" * 128, 1, 256, b"\x38" * 16, vector, 1)

    def test_c_rechecks_actual_band_buffer_sizes_before_reading(self):
        packed, scales = (C.c_uint8 * 8)(*([0x22] * 8)), (C.c_uint8 * 1)(0x38)
        vector, output = (C.c_float * 16)(*([1] * 16)), (C.c_float * 1)(123)
        status = self.cpu._dll.nvfp4_cpu_matvec_row_band(packed, 8, 128, 16384, scales, 1, vector, 16, 1, output, 1)
        self.assertEqual(status, 1)
        self.assertEqual(output[0], 123)


if __name__ == "__main__":
    unittest.main()
