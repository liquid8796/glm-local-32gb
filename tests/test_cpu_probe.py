"""Boundary checks always run; opt in to the tiny compiled probe with GLM_TEST_NATIVE=1."""

import ctypes
import math
import os
import random
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from glm_local.cpu_probe import NativeCpuBackend


def oracle_decode(code):
    """Independent mathematical E4M3FN oracle, expressed as binary fractions."""
    sign = -1 if code >= 128 else 1
    magnitude = code % 128
    exponent, fraction = divmod(magnitude, 8)
    if magnitude == 127:
        return math.nan
    if exponent == 0:
        return sign * (fraction / 512)
    return sign * (1 + fraction / 8) * 2 ** (exponent - 7)


def f32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


def oracle_matvec(weights, rows, cols, vector, scale):
    result = []
    for row in range(rows):
        total = 0.0
        for column in range(cols):
            scaled = f32(oracle_decode(weights[row * cols + column]) * f32(scale))
            total = f32(total + f32(scaled * f32(vector[column])))
        result.append(total)
    return result


class NativeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.backend = NativeCpuBackend.__new__(NativeCpuBackend)
        self.backend._dll = Mock()

    def test_invalid_inputs_are_rejected_before_native_call(self):
        cases = [
            (b"\x38", 0, 1, [1], 1),
            (b"\x38", 129, 1, [1], 1),
            (b"\x38", 1, 129, [1], 1),
            (b"\x38", True, 1, [1], 1),
            (b"\x38", 1, 1.0, [1], 1),
            (b"", 1, 1, [1], 1),
            (b"\x38\x38", 1, 1, [1], 1),
            (bytearray(b"\x38"), 1, 1, [1], 1),
            (b"\x7f", 1, 1, [1], 1),
            (b"\xff", 1, 1, [1], 1),
            (b"\x38", 1, 1, [], 1),
            (b"\x38", 1, 1, iter([1]), 1),
            (b"\x38", 1, 1, [math.nan], 1),
            (b"\x38", 1, 1, [math.inf], 1),
            (b"\x38", 1, 1, [1e100], 1),
            (b"\x38", 1, 1, [10**1000], 1),
            (b"\x38", 1, 1, ["1"], 1),
            (b"\x38", 1, 1, [True], 1),
            (b"\x38", 1, 1, [1], 0),
            (b"\x38", 1, 1, [1], -1),
            (b"\x38", 1, 1, [1], math.nan),
            (b"\x38", 1, 1, [1], math.inf),
            (b"\x38", 1, 1, [1], 1e100),
            (b"\x38", 1, 1, [1], 1e-100),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.backend.matvec_tile(*arguments)
        self.backend._dll.fp8_cpu_matvec_tile.assert_not_called()

    def test_closed_wrapper_cannot_be_reentered_or_used(self):
        self.backend.close()
        self.backend.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.backend.__enter__()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.backend.matvec_tile(b"\x38", 1, 1, [1], 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.backend.decode_e4m3fn(0)

    def test_decoder_rejects_wrapping_and_noninteger_inputs(self):
        for value in (-1, 256, True, 1.0, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.backend.decode_e4m3fn(value)
        self.backend._dll.fp8_cpu_decode_e4m3fn.assert_not_called()

    def test_native_error_status_is_not_reported_as_a_result(self):
        for status, exception in ((1, ValueError), (2, ValueError), (99, RuntimeError)):
            self.backend._dll.fp8_cpu_matvec_tile.return_value = status
            with self.subTest(status=status), self.assertRaises(exception):
                self.backend.matvec_tile(b"\x38", 1, 1, [1], 1)

    @unittest.skipUnless(os.name == "nt" and ctypes.sizeof(ctypes.c_void_p) == 8,
                         "requires 64-bit Windows")
    def test_missing_dll_fails_without_implicit_build(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "build-native.bat"):
                NativeCpuBackend(Path(folder) / "missing.dll")
            self.assertEqual(list(Path(folder).iterdir()), [])


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "set GLM_TEST_NATIVE=1")
class NativeCpuIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.backend = NativeCpuBackend()
        self.addCleanup(self.backend.close)

    def test_every_encoding_matches_independent_mathematical_oracle(self):
        for code in range(256):
            with self.subTest(code=code):
                actual = self.backend.decode_e4m3fn(code)
                expected = oracle_decode(code)
                if math.isnan(expected):
                    self.assertTrue(math.isnan(actual))
                else:
                    self.assertEqual(actual, expected)
        self.assertEqual(math.copysign(1.0, self.backend.decode_e4m3fn(0)), 1.0)
        self.assertEqual(math.copysign(1.0, self.backend.decode_e4m3fn(128)), -1.0)
        self.assertEqual(self.backend.decode_e4m3fn(126), 448.0)

    def test_tiny_rectangular_tiles_and_float32_rounding(self):
        generator = random.Random(4427)
        codes = [code for code in range(256) if code not in (127, 255)]
        for rows, cols in ((1, 1), (1, 128), (17, 9), (128, 128)):
            weights = bytes(generator.choice(codes) for _ in range(rows * cols))
            vector = [generator.uniform(-3.0, 3.0) for _ in range(cols)]
            for scale in (0.03125, 0.1, 1.5):
                with self.subTest(rows=rows, cols=cols, scale=scale):
                    actual = self.backend.matvec_tile(weights, rows, cols, vector, scale)
                    self.assertEqual(actual, oracle_matvec(weights, rows, cols, vector, scale))

    def test_asymmetric_known_matrix_uses_row_major_order_and_scale(self):
        # [[1, -2, 0.5], [4, 0, -1]] @ [2, 3, -4], with block scale 0.25.
        self.assertEqual(self.backend.matvec_tile(
            bytes([0x38, 0xC0, 0x30, 0x48, 0x00, 0xB8]), 2, 3, [2, 3, -4], 0.25
        ), [-1.5, 3.0])

    def test_intermediate_overflow_fails_even_if_later_values_would_cancel(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.backend.matvec_tile(bytes([0x7E, 0xFE]), 1, 2, [1, 1], 3e38)

    def test_c_boundary_rechecks_lengths_before_accessing_buffers(self):
        weights = (ctypes.c_uint8 * 1)(0x38)
        vector = (ctypes.c_float * 1)(1)
        output = (ctypes.c_float * 1)(123)
        status = self.backend._dll.fp8_cpu_matvec_tile(
            weights, 1, 128, 128, vector, 1, 1.0, output, 1
        )
        self.assertEqual(status, 1)
        self.assertEqual(output[0], 123)

    def test_metadata_identifies_actual_cpu_dll_and_limited_scope(self):
        self.assertEqual(self.backend.metadata["abi_version"], 1)
        self.assertIn("MSVC", self.backend.metadata["build_info"])
        self.assertFalse(self.backend.metadata["full_model_inference"])
        self.assertEqual(self.backend.metadata["threads"], 1)
        with self.backend:
            self.assertEqual(self.backend.decode_e4m3fn(0x38), 1.0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.backend.decode_e4m3fn(0x38)


if __name__ == "__main__":
    unittest.main()
