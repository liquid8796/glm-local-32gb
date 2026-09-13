"""Small input batches use independent output rows as SIMD lanes."""
from array import array
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import unittest

from glm_local.cpu_probe import NativeCpuBackend
from glm_local.nvfp4_kernels import NativeNVFP4CpuBackend
from test_dense_cpu import encode
from test_native_row_parallel import old_dense_tiles, packed_floats
from test_nvfp4_row_band import old_tile_reduction


@unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "Explicit bounded native CPU row SIMD check")
class NativeRowSimdTests(unittest.TestCase):
    def setUp(self):
        candidate = os.environ.get("GLM_ROW_SIMD_CANDIDATE")
        dense_path = Path(candidate) / "fp8_cpu.dll" if candidate else None
        nv_path = Path(candidate) / "nvfp4_cpu.dll" if candidate else None
        self.dense = [NativeCpuBackend(dense_path, row_band_threads=count) for count in (1, 8)]
        self.nv = [NativeNVFP4CpuBackend(nv_path, row_band_threads=count) for count in (1, 8)]
        for backend in self.dense + self.nv:
            self.addCleanup(backend.close)

    def test_dense_small_batches_match_unchanged_tile_reduction_across_worker_and_row_tails(self):
        rng = random.Random(93617)
        for dtype in ("BF16", "F16", "F32"):
            for rows, cols in ((1, 144), (3, 144), (4, 144), (5, 144), (7, 144), (31, 512), (33, 512), (127, 512), (128, 512), (8, 16384)):
                raw = bytearray(encode([rng.uniform(-2, 2) for _ in range(rows * cols)], dtype))
                vectors = [array("f", (rng.uniform(-1, 1) for _ in range(cols))) for _ in range(3)]
                expected = [packed_floats(old_dense_tiles(self.dense[0], bytes(raw), rows, cols, vector, dtype)) for vector in vectors]
                for batch in (1, 2, 3):
                    for backend in self.dense:
                        with self.subTest(dtype=dtype, rows=rows, cols=cols, batch=batch, threads=backend.row_band_threads):
                            actual = backend.matvec_dense_row_band_many(raw, rows, cols, backend.prepare_dense_batch(vectors[:batch]), dtype)
                            self.assertEqual([packed_floats(row) for row in actual], expected[:batch])
                            if batch == 1:
                                self.assertEqual(packed_floats(backend.matvec_dense_row_band(raw, rows, cols,
                                    backend.prepare_dense_vector(vectors[0]), dtype)), expected[0])

    def test_nvfp4_small_batches_match_unchanged_tile_reduction_across_worker_and_row_tails(self):
        rng = random.Random(35711)
        for rows, cols in ((1, 144), (3, 144), (4, 144), (5, 144), (7, 144), (31, 512), (33, 512), (127, 512), (128, 512), (8, 16384)):
            raw = bytearray(rng.randrange(256) for _ in range(rows * cols // 2))
            scales = bytearray(rng.randrange(127) for _ in range(rows * cols // 16))
            vectors = [array("f", (rng.uniform(-2, 2) for _ in range(cols))) for _ in range(3)]
            expected = [packed_floats(old_tile_reduction(self.nv[0], bytes(raw), rows, cols, bytes(scales), vector, 0.017)) for vector in vectors]
            for batch in (1, 2, 3):
                for backend in self.nv:
                    with self.subTest(rows=rows, cols=cols, batch=batch, threads=backend.row_band_threads):
                        actual = backend.matvec_nvfp4_row_band_many(raw, rows, cols, scales, backend.prepare_nvfp4_batch(vectors[:batch]), 0.017)
                        self.assertEqual([packed_floats(row) for row in actual], expected[:batch])
                        if batch == 1:
                            self.assertEqual(packed_floats(backend.matvec_nvfp4_row_band(raw, rows, cols, scales,
                                backend.prepare_nvfp4_vector(vectors[0]), 0.017)), expected[0])

    def test_row_lanes_preserve_subnormals_separate_products_and_tile_subtotals(self):
        for backend in self.dense:
            for dtype, unit, value in (("BF16", b"\x01\0", 1.0), ("F16", b"\x01\0", 2**-125),
                                      ("F32", b"\x01\0\0\0", 1.0)):
                raw, vectors = unit * (4 * 144), [[value] * 144] * 3
                expected = old_dense_tiles(backend, raw, 4, 144, vectors[0], dtype)
                actual = backend.matvec_dense_row_band_many(raw, 4, 144, backend.prepare_dense_batch(vectors), dtype)
                self.assertEqual([packed_floats(row) for row in actual], [packed_floats(expected)] * 3)
            a, b = 1.0000001192092896, 1.000000238418579
            actual = backend.matvec_dense_row_band_many(encode([-b, a] * 4, "F32"), 4, 2,
                backend.prepare_dense_batch([[1, a]] * 3), "F32")
            self.assertTrue(all(list(row) == [0.0] * 4 for row in actual))
            vector = [0.0] * 144
            vector[0], vector[128], vector[129], vector[130] = 1e8, 1, -1e8, 1
            actual = backend.matvec_dense_row_band_many(encode([1.0] * (4 * 144), "F32"), 4, 144,
                backend.prepare_dense_batch([vector] * 3), "F32")
            self.assertTrue(all(list(row) == [0.0] * 4 for row in actual))
        for backend in self.nv:
            packed, scales, vector = b"\x77" * (4 * 144 // 2), b"\x01" * (4 * 144 // 16), [1.0] * 144
            expected = old_tile_reduction(backend, packed, 4, 144, scales, vector, 2**-125)
            actual = backend.matvec_nvfp4_row_band_many(packed, 4, 144, scales,
                backend.prepare_nvfp4_batch([vector] * 3), 2**-125)
            self.assertEqual([packed_floats(row) for row in actual], [packed_floats(expected)] * 3)

    @unittest.skipUnless(os.environ.get("GLM_ROW_SIMD_BASELINE") and os.environ.get("GLM_ROW_SIMD_REPORT"),
                         "Explicit saved before-DLL and bounded benchmark report required")
    def test_before_after_small_batch_resident_band_benchmark(self):
        baseline = Path(os.environ["GLM_ROW_SIMD_BASELINE"])
        rng = random.Random(955711)
        rows, cols = 128, 6144
        dense = bytearray(encode([rng.uniform(-2, 2) for _ in range(rows * cols)], "BF16"))
        raw = bytearray(rng.randrange(256) for _ in range(rows * cols // 2))
        scales = bytearray(rng.randrange(0x20, 0x41) for _ in range(rows * cols // 16))
        vectors = [array("f", (rng.uniform(-1, 1) for _ in range(cols))) for _ in range(3)]
        records = []
        for kind, current, previous_path, backend_type in (
            ("BF16", self.dense, baseline / "fp8_cpu_before_rows.dll", NativeCpuBackend),
            ("NVFP4", self.nv, baseline / "nvfp4_cpu_before_rows.dll", NativeNVFP4CpuBackend)):
            for index, count in enumerate((1, 8)):
                with backend_type(previous_path, row_band_threads=count) as previous:
                    for batch in (1, 2, 3):
                        operations = []
                        for backend in (previous, current[index]):
                            if kind == "BF16":
                                prepared = backend.prepare_dense_batch(vectors[:batch])
                                single = backend.prepare_dense_vector(vectors[0]) if batch == 1 else None
                                operation = (lambda b=backend, p=single: [b.matvec_dense_row_band(dense, rows, cols, p, "BF16")]) if batch == 1 else (
                                    lambda b=backend, p=prepared: b.matvec_dense_row_band_many(dense, rows, cols, p, "BF16"))
                            else:
                                prepared = backend.prepare_nvfp4_batch(vectors[:batch])
                                single = backend.prepare_nvfp4_vector(vectors[0]) if batch == 1 else None
                                operation = (lambda b=backend, p=single: [b.matvec_nvfp4_row_band(raw, rows, cols, scales, p, 0.0625)]) if batch == 1 else (
                                    lambda b=backend, p=prepared: b.matvec_nvfp4_row_band_many(raw, rows, cols, scales, p, 0.0625))
                            operations.append(operation)
                        self.assertEqual([packed_floats(row) for row in operations[0]()], [packed_floats(row) for row in operations[1]()])
                        samples = [[], []]
                        for repeat in range(7):
                            for item in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                                started = time.perf_counter(); operations[item](); samples[item].append(time.perf_counter() - started)
                        before, after = map(statistics.median, samples)
                        records.append({"dtype": kind, "rows": rows, "cols": cols, "batch": batch, "threads": count,
                            "before_median_seconds": before, "after_median_seconds": after,
                            "speed_ratio": before / after, "bit_identical": True,
                            "baseline_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                            "candidate_sha256": hashlib.sha256(Path(current[index].metadata["dll_path"]).read_bytes()).hexdigest()})
        path = Path(os.environ["GLM_ROW_SIMD_REPORT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "PASS", "scope": "bounded_synthetic_resident_band_compute_only",
            "input_preparation_timed": False, "file_io_timed": False, "model_inference_executed": False,
            "measurements": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
