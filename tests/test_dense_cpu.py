"""Independent dense storage/FP32 oracle and bounded native runtime dispatch checks."""
from array import array
import ctypes as C
import math
import os
from pathlib import Path
import random
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from glm_local.cpu_probe import NativeCpuBackend


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode(values, dtype):
    if dtype == "BF16":
        return b"".join(struct.pack("<f", value)[2:] for value in values)
    return b"".join(struct.pack("<e" if dtype == "F16" else "<f", value) for value in values)


def oracle(raw, rows, cols, vector, dtype):
    if dtype == "BF16":
        decoded = [struct.unpack("<f", b"\0\0" + raw[index:index + 2])[0] for index in range(0, len(raw), 2)]
    else:
        decoded = [item[0] for item in struct.iter_unpack("<e" if dtype == "F16" else "<f", raw)]
    result = []
    for row in range(rows):
        total = 0.0
        for col in range(cols):
            total = f32(total + f32(decoded[row * cols + col] * f32(vector[col])))
        result.append(total)
    return result


class DenseBoundaryTests(unittest.TestCase):
    def backend(self):
        backend = NativeCpuBackend.__new__(NativeCpuBackend)
        function = Mock(return_value=0)
        backend._dll = SimpleNamespace(fp8_cpu_matvec_dense_tile=function)
        return backend, function

    def test_invalid_dense_inputs_are_rejected_before_native_entry(self):
        valid = (b"\x80\x3f", 1, 1, [1], "BF16")
        replacements = [(0, b""), (0, b"\0" * 4), (0, bytearray(2)), (1, True), (1, 0), (1, 129),
                        (2, 129), (2, 1.0), (3, []), (3, "1"), (3, b"1"), (3, [True]),
                        (3, [math.nan]), (3, [math.inf]), (3, [1e100]), (3, iter([1])),
                        (4, "F64"), (4, None), (4, [])]
        backend, function = self.backend()
        for index, value in replacements:
            args = list(valid)
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                backend.matvec_dense_tile(*args)
        function.assert_not_called()

    def test_old_abi1_library_has_an_explicit_rebuild_message_for_dense_only(self):
        backend = NativeCpuBackend.__new__(NativeCpuBackend)
        backend._dll = SimpleNamespace(fp8_cpu_matvec_tile=Mock())
        with self.assertRaisesRegex(RuntimeError, "build-native.bat"):
            backend.matvec_dense_tile(b"\x80\x3f", 1, 1, [1], "BF16")

    def test_native_status_and_closed_backend_do_not_return_partial_results(self):
        backend, function = self.backend()
        for status, expected in ((1, ValueError), (2, ValueError), (9, RuntimeError)):
            function.return_value = status
            with self.subTest(status=status), self.assertRaises(expected):
                backend.matvec_dense_tile(b"\x80\x3f", 1, 1, [1], "BF16")
        backend.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.matvec_dense_tile(b"\x80\x3f", 1, 1, [1], "BF16")


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "set GLM_TEST_NATIVE=1")
class NativeDenseTests(unittest.TestCase):
    def setUp(self):
        self.cpu = NativeCpuBackend()
        self.addCleanup(self.cpu.close)

    def test_three_dtypes_rectangular_edges_and_rounding_match_independent_oracle(self):
        randomizer = random.Random(5731)
        for dtype in ("BF16", "F16", "F32"):
            for rows, cols in ((1, 1), (7, 13), (128, 3), (3, 128), (128, 128)):
                raw = encode([randomizer.uniform(-3, 3) for _ in range(rows * cols)], dtype)
                vector = [randomizer.uniform(-2, 2) for _ in range(cols)]
                with self.subTest(dtype=dtype, shape=(rows, cols)):
                    self.assertEqual(self.cpu.matvec_dense_tile(raw, rows, cols, vector, dtype),
                                     oracle(raw, rows, cols, vector, dtype))
                    self.assertLessEqual(len(raw), 65536)

    def test_every_finite_bf16_and_float16_bit_pattern_is_decoded_exactly(self):
        for dtype, exponent_mask in (("BF16", 0x7f80), ("F16", 0x7c00)):
            codes = [code for code in range(65536) if code & exponent_mask != exponent_mask]
            for start in range(0, len(codes), 128):
                block = codes[start:start + 128]
                raw = b"".join(struct.pack("<H", code) for code in block)
                with self.subTest(dtype=dtype, start=start):
                    self.assertEqual(self.cpu.matvec_dense_tile(raw, len(block), 1, [1], dtype),
                                     oracle(raw, len(block), 1, [1], dtype))

    def test_nonfinite_payloads_and_intermediate_overflow_are_rejected(self):
        for dtype, payloads in (("BF16", (b"\x80\x7f", b"\xc0\x7f", b"\x80\xff")),
                                ("F16", (b"\x00\x7c", b"\x01\x7c", b"\x00\xfc")),
                                ("F32", (struct.pack("<f", math.inf), struct.pack("<f", math.nan)))):
            for raw in payloads:
                with self.subTest(dtype=dtype, raw=raw), self.assertRaisesRegex(ValueError, "non-finite"):
                    self.cpu.matvec_dense_tile(raw, 1, 1, [1], dtype)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.cpu.matvec_dense_tile(encode([3e38, -3e38], "F32"), 1, 2, [2, 2], "F32")

    def test_float32_product_and_sum_do_not_contract_to_fma(self):
        raw = encode([-(1 + 2**-22), 1 + 2**-23], "F32")
        self.assertEqual(self.cpu.matvec_dense_tile(raw, 1, 2, [1, 1 + 2**-23], "F32"), [0.0])

    def test_c_boundary_rechecks_buffer_lengths_and_dtype_before_access(self):
        raw, vector, output = (C.c_uint8 * 2)(0x80, 0x3f), (C.c_float * 1)(1), (C.c_float * 1)(123)
        for rows, cols, dtype in ((128, 128, 1), (1, 1, 99)):
            with self.subTest(rows=rows, cols=cols, dtype=dtype):
                status = self.cpu._dll.fp8_cpu_matvec_dense_tile(raw, 2, rows, cols, dtype, vector, 1, output, 1)
                self.assertEqual(status, 1)
                self.assertEqual(output[0], 123)

    def test_runtime_dispatches_dense_tiles_to_native_without_python_matrix_decode(self):
        from checkpoint_test_helpers import settings
        from glm_local.runtime_weights import RuntimeWeights
        from test_runtime_weights import prepare_runtime_fixture
        for dtype in ("BF16", "F16", "F32"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prepare_runtime_fixture(root, output_dtype=dtype)
                with RuntimeWeights(root, settings(), cpu=self.cpu) as weights:
                    with patch("glm_local.runtime_weights._decode", side_effect=AssertionError("dense path must be native")):
                        result = weights.linear("lm_head.weight", [0.5] * 16)
                    self.assertEqual(result, array("f", [8.0] * 257))
                    self.assertEqual(weights.stats()["native_dense_tiles"], 3)
                    self.assertEqual(weights.stats()["scalar_dense_tiles"], 0)
                    self.assertEqual(weights.stats()["max_decoded_dense_tile_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
