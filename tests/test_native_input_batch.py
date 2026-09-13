"""Independent SIMD input lanes preserve scalar128-column FP32 reductions."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import threading
import time
import unittest

from glm_local.cpu_probe import NativeCpuBackend, _prepare_row_batch
from glm_local.nvfp4_kernels import NativeNVFP4CpuBackend
from test_dense_cpu import encode
from test_native_row_parallel import packed_floats


class PreparedInputBatchTests(unittest.TestCase):
    def test_interleaved_layout_and_one_mib_maximum_native_storage(self):
        prepared = _prepare_row_batch([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual((prepared.cols, prepared.batch), (3, 3))
        self.assertEqual(list(prepared.buffer), [1, 4, 7, 2, 5, 8, 3, 6, 9])
        maximum = _prepare_row_batch([array("f", [0]) * 16384] * 16)
        self.assertEqual(C.sizeof(maximum.buffer), 1024**2)

    def test_non_sized_ragged_nonfinite_and_out_of_bounds_inputs_fail_before_native(self):
        cases = [[], [[1]] * 17, [[], []], [[1], [2, 3]], [[1] * 16385],
                 [[math.nan]], [[math.inf]], [[True]], [[1e100]], ["abc"], [[1j]], iter([[1]])]
        for vectors in cases:
            with self.subTest(value=str(vectors)[:50]), self.assertRaises(ValueError):
                _prepare_row_batch(vectors)
        with self.assertRaises(ValueError):
            _prepare_row_batch([[1] * 17], multiple=16)


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "Explicit bounded native CPU check")
class NativeInputBatchTests(unittest.TestCase):
    def setUp(self):
        self.dense = [NativeCpuBackend(row_band_threads=threads) for threads in (1, 8)]
        self.nv = [NativeNVFP4CpuBackend(row_band_threads=threads) for threads in (1, 8)]
        for backend in self.dense + self.nv:
            self.addCleanup(backend.close)

    def assert_bits(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        self.assertTrue(all(type(row) is array and row.typecode == "f" for row in actual))
        self.assertEqual([packed_floats(row) for row in actual], [packed_floats(row) for row in expected])

    def test_dense_many_matches_each_single_vector_and_worker_count_for_all_dtypes(self):
        rng = random.Random(933118)
        for dtype in ("BF16", "F16", "F32"):
            for rows, cols in ((7, 144), (128, 512), (1, 16384)):
                raw = bytearray(encode([rng.uniform(-2, 2) for _ in range(rows * cols)], dtype))
                original = bytes(raw)
                for batch in (1, 3, 4, 7, 8, 16):
                    vectors = [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(batch)]
                    expected = [self.dense[0].matvec_dense_row_band(raw, rows, cols,
                                self.dense[0].prepare_dense_vector(vector), dtype) for vector in vectors]
                    prepared = self.dense[0].prepare_dense_batch(vectors)
                    for backend in self.dense:
                        with self.subTest(dtype=dtype, rows=rows, cols=cols, batch=batch, threads=backend.row_band_threads):
                            actual = backend.matvec_dense_row_band_many(raw, rows, cols, prepared, dtype)
                            self.assert_bits(actual, expected)
                    self.assertEqual(bytes(raw), original)
        self.assertTrue(self.dense[1]._row_pool._pointer.value)

    def test_nvfp4_many_matches_single_vectors_with_ragged_final_tiles_and_scales(self):
        rng = random.Random(261293)
        for rows, cols in ((7, 144), (128, 512), (1, 16384)):
            packed = bytearray(rng.randrange(256) for _ in range(rows * cols // 2))
            scales = bytearray(rng.randrange(127) for _ in range(rows * cols // 16))
            originals = bytes(packed), bytes(scales)
            for batch in (1, 3, 4, 7, 8, 16):
                vectors = [[rng.uniform(-2, 2) for _ in range(cols)] for _ in range(batch)]
                expected = [self.nv[0].matvec_nvfp4_row_band(packed, rows, cols, scales,
                            self.nv[0].prepare_nvfp4_vector(vector), 0.017) for vector in vectors]
                prepared = self.nv[0].prepare_nvfp4_batch(vectors)
                for backend in self.nv:
                    with self.subTest(rows=rows, cols=cols, batch=batch, threads=backend.row_band_threads):
                        self.assert_bits(backend.matvec_nvfp4_row_band_many(packed, rows, cols, scales, prepared, 0.017), expected)
                self.assertEqual((bytes(packed), bytes(scales)), originals)
        self.assertTrue(self.nv[1]._row_pool._pointer.value)

    def test_fp32_subnormal_fma_sensitive_and_128_column_subtotal_order_is_preserved(self):
        for dtype, unit, value in (("BF16", b"\x01\0", 1.0), ("F16", b"\x01\0", 2**-125),
                                  ("F32", b"\x01\0\0\0", 1.0)):
            vectors = [[value] * 512] * 16
            raw = unit * (128 * 512)
            expected = [self.dense[0].matvec_dense_row_band(raw, 128, 512, vectors[0], dtype)] * 16
            self.assert_bits(self.dense[1].matvec_dense_row_band_many(raw, 128, 512,
                             self.dense[1].prepare_dense_batch(vectors), dtype), expected)
        vector = [0.0] * 144
        vector[0], vector[128], vector[129], vector[130] = 1e8, 1, -1e8, 1
        self.assert_bits(self.dense[1].matvec_dense_row_band_many(encode([1.0] * 144, "F32"), 1, 144,
                         self.dense[1].prepare_dense_batch([vector] * 16), "F32"), [[0.0]] * 16)
        # A fused multiply-add would retain a residual here; two FP32 operations return exactly zero.
        a, b = 1.0000001192092896, 1.000000238418579
        self.assert_bits(self.dense[1].matvec_dense_row_band_many(encode([-b, a], "F32"), 1, 2,
                         self.dense[1].prepare_dense_batch([[1, a]] * 16), "F32"), [[0.0]] * 16)

    def test_nonfinite_payloads_scales_prepared_mutation_and_overflow_raise(self):
        for backend in self.dense:
            prepared = backend.prepare_dense_batch([[1.0] * 256] * 7)
            raw = bytearray(encode([1.0] * (128 * 256), "F32"))
            raw[-4:] = struct.pack("<f", math.nan)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.matvec_dense_row_band_many(raw, 128, 256, prepared, "F32")
            raw[-4:] = struct.pack("<f", 1)
            prepared.buffer[-1] = math.inf
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.matvec_dense_row_band_many(raw, 128, 256, prepared, "F32")
            vector = [0.0] * 256
            vector[0] = vector[128] = 2e38
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.matvec_dense_row_band_many(raw, 128, 256, backend.prepare_dense_batch([vector] * 16), "F32")
        for backend in self.nv:
            prepared = backend.prepare_nvfp4_batch([[1.0] * 144] * 16)
            for scale in (b"\x7f", b"\xff", b"\x80"):
                with self.assertRaises(ValueError):
                    backend.matvec_nvfp4_row_band_many(b"\x22" * 72, 1, 144, scale * 9, prepared, 1)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.matvec_nvfp4_row_band_many(b"\x77" * 72, 1, 144, b"\x7e" * 9, prepared, 3e38)
            prepared.buffer[0] = math.nan
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.matvec_nvfp4_row_band_many(b"\x22" * 72, 1, 144, b"\x38" * 9, prepared, 1)

    def test_native_and_wrapper_lengths_reject_before_touching_short_buffers(self):
        for backend in self.dense:
            prepared = backend.prepare_dense_batch([[1]])
            arguments = [b"\x80\x3f", 1, 1, prepared, "BF16"]
            for index, bad in ((0, b""), (1, 129), (1, True), (2, 0), (2, 16385), (3, [[1]]), (4, "U8")):
                modified = list(arguments); modified[index] = bad
                with self.assertRaises(ValueError):
                    backend.matvec_dense_row_band_many(*modified)
            prepared.batch = 17
            with self.assertRaises(ValueError):
                backend.matvec_dense_row_band_many(*arguments)
            prepared.batch = 1; prepared.buffer = (C.c_double * 1)(1)
            with self.assertRaises(ValueError):
                backend.matvec_dense_row_band_many(*arguments)
            weight, vector, output = (C.c_uint8 * 2)(0x80, 0x3f), (C.c_float * 1)(1), (C.c_float * 1)(123)
            for batch in (0, 17, 16):
                status = backend._dll.fp8_cpu_matvec_dense_row_band_many(weight, 2, 128, 16384, batch, 1,
                    vector, 1, output, 1, None)
                self.assertEqual(status, 1)
                self.assertEqual(output[0], 123)
        backend = self.nv[0]
        packed, scales, vector, output = (C.c_uint8 * 8)(), (C.c_uint8 * 1)(0x38), (C.c_float * 16)(*([1]*16)), (C.c_float * 1)(123)
        for rows, cols, batch in ((1, 17, 1), (1, 16, 17), (128, 16384, 16)):
            self.assertEqual(backend._dll.nvfp4_cpu_matvec_row_band_many(packed, 8, rows, cols, batch,
                scales, 1, vector, 16, 1, output, 1, None), 1)
            self.assertEqual(output[0], 123)

    def test_close_waits_for_batch_ffi_and_closed_instances_reject_preparation_and_calls(self):
        for backend, symbol, prepare, run in (
            (self.dense[1], "fp8_cpu_matvec_dense_row_band_many", lambda b: b.prepare_dense_batch([[1]] * 4),
             lambda b, p: b.matvec_dense_row_band_many(b"\x80\x3f", 1, 1, p, "BF16")),
            (self.nv[1], "nvfp4_cpu_matvec_row_band_many", lambda b: b.prepare_nvfp4_batch([[1] * 16] * 4),
             lambda b, p: b.matvec_nvfp4_row_band_many(b"\x22" * 8, 1, 16, b"\x38", p, 1))):
            entered, release, closing = threading.Event(), threading.Event(), threading.Event()
            prepared = prepare(backend)
            def blocked(*_):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("bounded test FFI timeout")
                return 0
            setattr(backend._dll, symbol, blocked)
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    work = executor.submit(run, backend, prepared)
                    self.assertTrue(entered.wait(5))
                    def close():
                        closing.set(); backend.close()
                    closed = executor.submit(close)
                    self.assertTrue(closing.wait(5))
                    self.assertFalse(closed.done())
                    release.set()
                    self.assert_bits(work.result(timeout=5), [[0]] * 4)
                    closed.result(timeout=5)
                self.assertFalse(backend._row_pool._pointer.value)
                with self.assertRaises(RuntimeError):
                    prepare(backend)
                with self.assertRaises(RuntimeError):
                    run(backend, prepared)
            finally:
                release.set()

    @unittest.skipUnless(os.environ.get("GLM_INPUT_BATCH_BENCHMARK"), "Explicit bounded synthetic compute benchmark")
    def test_bounded_b16_compute_benchmark_records_measured_speed_without_network_claims(self):
        rng = random.Random(43474)
        rows, cols, batch = 128, 4096, 16
        vectors = [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(batch)]
        dense = bytearray(encode([rng.uniform(-2, 2) for _ in range(rows * cols)], "BF16"))
        packed = bytearray(rng.randrange(256) for _ in range(rows * cols // 2))
        scales = bytearray(rng.randrange(0x20, 0x41) for _ in range(rows * cols // 16))
        records = []
        for kind, backends in (("BF16", self.dense), ("NVFP4", self.nv)):
            for backend in backends:
                if kind == "BF16":
                    singles = [backend.prepare_dense_vector(vector) for vector in vectors]
                    prepared = backend.prepare_dense_batch(vectors)
                    scalar = lambda: [backend.matvec_dense_row_band(dense, rows, cols, vector, "BF16") for vector in singles]
                    many = lambda: backend.matvec_dense_row_band_many(dense, rows, cols, prepared, "BF16")
                else:
                    singles = [backend.prepare_nvfp4_vector(vector) for vector in vectors]
                    prepared = backend.prepare_nvfp4_batch(vectors)
                    scalar = lambda: [backend.matvec_nvfp4_row_band(packed, rows, cols, scales, vector, 0.0625) for vector in singles]
                    many = lambda: backend.matvec_nvfp4_row_band_many(packed, rows, cols, scales, prepared, 0.0625)
                self.assert_bits(many(), scalar())
                elapsed = {"single_calls": [], "simd_batch": []}
                for repeat in range(5):
                    for name, operation in (("single_calls", scalar), ("simd_batch", many)) if repeat % 2 == 0 else (("simd_batch", many), ("single_calls", scalar)):
                        start = time.perf_counter(); operation(); elapsed[name].append(time.perf_counter() - start)
                single, simd = statistics.median(elapsed["single_calls"]), statistics.median(elapsed["simd_batch"])
                records.append({"dtype": kind, "rows": rows, "cols": cols, "batch": batch,
                    "threads": backend.row_band_threads, "single_calls_median_seconds": single,
                    "simd_batch_median_seconds": simd, "speed_ratio": single / simd,
                    "prepared_native_bytes": C.sizeof(prepared.buffer), "bit_identical": True})
        path = Path(os.environ["GLM_INPUT_BATCH_BENCHMARK"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "PASS", "scope": "bounded_synthetic_cpu_band_compute_only",
            "weights_already_in_memory": True, "network_used": False, "full_model_inference": False,
            "measurements": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
