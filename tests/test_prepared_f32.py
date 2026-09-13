"""FP32 buffer copies preserve bits, finite guards and caller ownership."""
from array import array
import ctypes
import math
import os
import struct
import unittest
from unittest.mock import patch

from glm_local import cpu_probe, runtime_linear
from glm_local.cpu_probe import NativeCpuBackend, _prepare_row_batch
from glm_local.nvfp4_kernels import NativeNVFP4CpuBackend
from glm_local.residency import ReservationLedger
from glm_local.safetensor_reader import TensorInfo


def samples(count):
    bits = (0, 0x80000000, 1, 0x80000001, 0x00800000, 0x7F7FFFFF, 0x3F812345)
    values = array("f")
    values.frombytes(b"".join(struct.pack("=I", bits[index % len(bits)]) for index in range(count)))
    return values


class PreparedFloat32Tests(unittest.TestCase):
    def test_exact_fp32_batches_scatter_bit_for_bit_without_scalar_reconversion(self):
        for batch in (1, 3, 4, 13, 16):
            vectors = [samples(144) for _ in range(batch)]
            expected = array("f", (vectors[lane][column] for column in range(144) for lane in range(batch)))
            with patch.object(cpu_probe, "_finite_float32", side_effect=AssertionError("FP32 must not be rounded again")):
                prepared = _prepare_row_batch(vectors, multiple=16)
            self.assertEqual(memoryview(prepared.buffer).tobytes(), expected.tobytes())
            self.assertEqual(ctypes.sizeof(prepared.buffer), 144 * batch * 4)

    def test_prepared_batch_owns_snapshot_and_releases_all_source_views(self):
        vectors = [array("f", [1, 2, 3]), array("f", [4, 5, 6])]
        prepared = _prepare_row_batch(vectors)
        before = memoryview(prepared.buffer).tobytes()
        vectors[0][0] = math.nan
        vectors[0].append(7)
        vectors[1] = array("f", [8, 9, 10])
        self.assertEqual(memoryview(prepared.buffer).tobytes(), before)
        prepared.buffer[0] = 11
        self.assertTrue(math.isnan(vectors[0][0]))

    def test_every_lane_finite_guard_survives_fast_copy(self):
        for lane in (0, 6, 12):
            for value in (math.nan, math.inf, -math.inf):
                vectors = [array("f", [0]) * 144 for _ in range(13)]
                vectors[lane][-1] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    _prepare_row_batch(vectors, multiple=16)
                vectors[lane].append(1)  # Failure also releases exported views.

    def test_generic_or_mixed_buffers_still_use_numeric_and_rounding_validation(self):
        vectors = [array("d", [1.00000007, 2]), array("f", [3, 4])]
        with patch.object(cpu_probe, "_finite_float32", wraps=cpu_probe._finite_float32) as checked:
            prepared = _prepare_row_batch(vectors)
        self.assertEqual(checked.call_count, 4)
        self.assertEqual(list(prepared.buffer), [struct.unpack("=f", struct.pack("=f", 1.00000007))[0], 3, 2, 4])
        for invalid in ([[True]], [[1e100]], [[1j]], [["1"]]):
            with self.assertRaises(ValueError):
                _prepare_row_batch(invalid)

    def test_shape_and_maximum_allocation_bounds_are_unchanged(self):
        for invalid in ([array("f")], [array("f", [1])] * 17,
                        [array("f", [1]), array("f", [1, 2])], [array("f", [0]) * 16385]):
            with self.assertRaises(ValueError):
                _prepare_row_batch(invalid)
        with self.assertRaises(ValueError):
            _prepare_row_batch([array("f", [0]) * 17], multiple=16)
        maximum = _prepare_row_batch([array("f", [0]) * 16384] * 16)
        self.assertEqual(ctypes.sizeof(maximum.buffer), 1024**2)

    def test_output_array_fast_path_preserves_bits_and_still_rejects_nonfinite(self):
        values = samples(128)
        with patch.object(runtime_linear, "_finite_float32", side_effect=AssertionError("FP32 output must not be rounded again")):
            self.assertIs(runtime_linear._output_band(values, 128), values)
        for value in (math.nan, math.inf, -math.inf):
            invalid = array("f", [0, value])
            with self.assertRaisesRegex(ValueError, "finite"):
                runtime_linear._output_band(invalid, 2)
        for values, height in (([True], 1), ([1e100], 1), (array("d", [math.inf]), 1), ([0], 2), (0, 1)):
            with self.assertRaises(ValueError):
                runtime_linear._output_band(values, height)

    def test_projection_copies_outputs_and_releases_ledger_on_invalid_native_result(self):
        class Reader:
            def read_span(self, name, offset, count):
                return bytearray(count)
        class Cpu:
            supports_dense_row_band_many = True
            results = []
            invalid = False
            def prepare_dense_vector(self, values):
                return values
            def prepare_dense_batch(self, values):
                return values
            def matvec_dense_row_band_many(self, weights, rows, cols, vectors, dtype):
                output = [array("f", [lane + 0.5]) * rows for lane in range(len(vectors))]
                if self.invalid:
                    output[-1][-1] = math.nan
                self.results.extend(output)
                return output
        info = TensorInfo("BF16", (129, 144), (0, 129 * 144 * 2), 129 * 144 * 2, 2)
        cpu, ledger = Cpu(), ReservationLedger(64 * 1024**2)
        values, metrics = runtime_linear.project_many(Reader(), "w", info, [array("f", [1]) * 144] * 3, cpu, ledger=ledger)
        expected = [value.tobytes() for value in values]
        for band in cpu.results:
            band[0] = 100
        self.assertEqual([value.tobytes() for value in values], expected)
        self.assertEqual(metrics["native_band_calls"], 2)
        self.assertEqual(ledger.snapshot()["active_leases"], 0)
        cpu.invalid = True
        with self.assertRaisesRegex(ValueError, "finite"):
            runtime_linear.project_many(Reader(), "w", info, [array("f", [1]) * 144] * 3, cpu, ledger=ledger)
        self.assertEqual(ledger.snapshot()["active_leases"], 0)


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "Explicit native wrapper preparation check")
class NativePreparedFloat32Tests(unittest.TestCase):
    def test_dense_and_nvfp4_vector_snapshots_preserve_bits_and_guard_finite_and_closed(self):
        for backend_type, method in ((NativeCpuBackend, "prepare_dense_vector"),
                                     (NativeNVFP4CpuBackend, "prepare_nvfp4_vector")):
            backend = backend_type()
            try:
                prepare = getattr(backend, method)
                vector = samples(144)
                expected = vector.tobytes()
                with patch.object(cpu_probe, "_finite_float32", side_effect=AssertionError("already FP32")):
                    prepared = prepare(vector)
                self.assertEqual(memoryview(prepared.buffer).tobytes(), expected)
                vector[0] = math.nan
                vector.append(1)
                self.assertEqual(memoryview(prepared.buffer).tobytes(), expected)
                for value in (math.nan, math.inf, -math.inf):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        prepare(array("f", [value]) * 144)
            finally:
                backend.close()
            with self.assertRaises(RuntimeError):
                prepare(array("f", [0]) * 144)


if __name__ == "__main__":
    unittest.main()
