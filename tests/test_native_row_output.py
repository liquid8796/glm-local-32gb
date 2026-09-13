"""Owned FP32 row-band outputs preserve the legacy list ABI and exact bits."""
from array import array
import ctypes as C
import math
import os
import random
import struct
from types import SimpleNamespace
import unittest

from glm_local.cpu_probe import NativeCpuBackend
from glm_local.nvfp4_kernels import NativeNVFP4CpuBackend
from test_dense_cpu import encode


def bits(values):
    return struct.pack(f"={len(values)}f", *values)


class ArrayOutputBoundaryTests(unittest.TestCase):
    def backend(self, kind, values):
        captured = []
        state = {"status": 0}
        def native(*args):
            output = args[7 if kind == "dense" else 9]
            captured.append(output)
            for index, value in enumerate(values):
                output[index] = value
            return state["status"]
        if kind == "dense":
            backend = NativeCpuBackend.__new__(NativeCpuBackend)
            backend._dll = SimpleNamespace(fp8_cpu_matvec_dense_row_band=native)
            args = (b"\0\0" * len(values), len(values), 1, [1], "BF16")
        else:
            backend = NativeNVFP4CpuBackend.__new__(NativeNVFP4CpuBackend)
            backend._dll = SimpleNamespace(nvfp4_cpu_matvec_row_band=native)
            args = (b"\0" * (8 * len(values)), len(values), 16, b"\0" * len(values), [1] * 16, 1)
        return backend, args, captured, state

    def test_array_is_owned_bitwise_copy_and_old_api_still_returns_list(self):
        values = [0.0, -0.0, 2**-149, -2**-149, 1 + 2**-23, -2.5, 3.4028234663852886e38]
        for kind in ("dense", "nvfp4"):
            with self.subTest(kind=kind):
                backend, args, captured, _ = self.backend(kind, values)
                legacy = getattr(backend, f"matvec_{kind}_row_band")(*args)
                actual = getattr(backend, f"matvec_{kind}_row_band_array")(*args)
                self.assertIs(type(legacy), list)
                self.assertIs(type(actual), array)
                self.assertEqual(actual.typecode, "f")
                self.assertEqual(actual.itemsize, 4)
                self.assertEqual(actual.tobytes(), bits(legacy))
                self.assertEqual(actual.tobytes(), bits(values))
                for buffer in captured:
                    C.memset(C.addressof(buffer), 0, C.sizeof(buffer))
                backend.close()
                self.assertEqual(actual.tobytes(), bits(values))
                actual.append(1)  # No exported native view survives the returned copy.

    def test_nonfinite_output_and_native_failures_never_escape_either_api(self):
        for kind in ("dense", "nvfp4"):
            for value in (math.nan, math.inf, -math.inf):
                for suffix in ("", "_array"):
                    with self.subTest(kind=kind, value=value, suffix=suffix):
                        backend, args, _, state = self.backend(kind, [value])
                        method = getattr(backend, f"matvec_{kind}_row_band{suffix}")
                        with self.assertRaisesRegex(ValueError, "non-finite"):
                            method(*args)
                        state["status"] = 2
                        with self.assertRaisesRegex(ValueError, "non-finite"):
                            method(*args)
                        state["status"] = 99
                        with self.assertRaisesRegex(RuntimeError, "99"):
                            method(*args)

    def test_array_path_keeps_shape_input_and_closed_backend_guards(self):
        for kind in ("dense", "nvfp4"):
            backend, good, captured, _ = self.backend(kind, [1])
            method = getattr(backend, f"matvec_{kind}_row_band_array")
            for index, value in ((0, b""), (1, 129), (1, True), (2, 0)):
                args = list(good)
                args[index] = value
                with self.subTest(kind=kind, index=index), self.assertRaises(ValueError):
                    method(*args)
            args = list(good)
            args[3 if kind == "dense" else 4] = [math.nan] * good[2]
            with self.assertRaises(ValueError):
                method(*args)
            self.assertEqual(captured, [])
            backend.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                method(*good)


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "set GLM_TEST_NATIVE=1")
class NativeArrayOutputTests(unittest.TestCase):
    def test_dense_array_preserves_all_storage_types_and_partial_row_band_bits(self):
        rng = random.Random(34119)
        with NativeCpuBackend() as backend:
            for dtype in ("BF16", "F16", "F32"):
                for rows, cols in ((1, 1), (7, 144), (128, 512), (3, 16384)):
                    weights = bytearray(encode([rng.uniform(-2, 2) for _ in range(rows * cols)], dtype))
                    vector = backend.prepare_dense_vector(array("f", (rng.uniform(-1, 1) for _ in range(cols))))
                    args = (weights, rows, cols, vector, dtype)
                    with self.subTest(dtype=dtype, rows=rows, cols=cols):
                        expected = backend.matvec_dense_row_band(*args)
                        actual = backend.matvec_dense_row_band_array(*args)
                        self.assertEqual(actual.tobytes(), bits(expected))
                        self.assertEqual(len(actual), rows)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.matvec_dense_row_band_array(encode([3e38, -3e38], "F32"), 1, 2, [2, 2], "F32")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.matvec_dense_row_band_array(encode([math.nan], "F32"), 1, 1, [1], "F32")

    def test_nvfp4_array_preserves_partial_bands_scales_and_overflow_errors(self):
        rng = random.Random(31147)
        with NativeNVFP4CpuBackend() as backend:
            for rows, cols in ((1, 16), (7, 144), (128, 512), (3, 16384)):
                packed = bytearray(rng.randrange(256) for _ in range(rows * cols // 2))
                scales = bytearray(rng.randrange(127) for _ in range(rows * cols // 16))
                vector = backend.prepare_nvfp4_vector(array("f", (rng.uniform(-1, 1) for _ in range(cols))))
                args = (packed, rows, cols, scales, vector, 0.017)
                with self.subTest(rows=rows, cols=cols):
                    expected = backend.matvec_nvfp4_row_band(*args)
                    actual = backend.matvec_nvfp4_row_band_array(*args)
                    self.assertEqual(actual.tobytes(), bits(expected))
                    self.assertEqual(len(actual), rows)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.matvec_nvfp4_row_band_array(b"\x77" * 72, 1, 144, b"\x7e" * 9, [1] * 144, 3e38)
            with self.assertRaises(ValueError):
                backend.matvec_nvfp4_row_band_array(b"\x22" * 8, 1, 16, b"\x7f", [1] * 16, 1)


if __name__ == "__main__":
    unittest.main()
