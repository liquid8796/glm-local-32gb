"""Projection integration: exclusive reader ownership, exact bands and cleanup."""
from array import array
import json
import os
from pathlib import Path
import struct
import tempfile
from threading import get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glm_local.residency import ReservationLedger
from glm_local.runtime_linear import project_many, DENSE_ROW_BAND_SCRATCH_BYTES
from glm_local.runtime_read_ahead import ReadAheadPool, READ_AHEAD_BUFFER_BYTES
from glm_local.safetensor_reader import SafeTensorReader


class BandReader:
    def __init__(self, rows=257, cols=144, *, nvfp4=False):
        self.rows, self.cols, self.nvfp4 = rows, cols, nvfp4
        self.reads, self.metadata_calls = [], []
        if nvfp4:
            self.info = SimpleNamespace(shape=(rows, cols // 2), itemsize=1, dtype="U8")
            self.data = {"w": bytes((row % 100 + 1) for row in range(rows) for _ in range(cols // 2)),
                         "s": bytes((row % 63 + 1) for row in range(rows) for _ in range(cols // 16)),
                         "g": struct.pack("<f", 0.125), "i": struct.pack("<f", 3.0)}
        else:
            self.info = SimpleNamespace(shape=(rows, cols), itemsize=4, dtype="F32")
            self.data = {"w": b"".join(struct.pack("<f", row * 0.125) * cols for row in range(rows))}

    def read_span(self, name, offset, count):
        self.reads.append((get_ident(), name, offset, count))
        return bytearray(self.data[name][offset:offset + count])

    def read_bytes(self, name, offset, count):
        self.reads.append((get_ident(), name, offset, count))
        return self.data[name][offset:offset + count]

    def projection(self, name):
        self.metadata_calls.append(get_ident())
        return SimpleNamespace(scale=SimpleNamespace(name="s"), global_scale=SimpleNamespace(name="g"),
                               input_scale=SimpleNamespace(name="i"))


class ScalarMatrix:
    def __init__(self, reader):
        self.reader = reader
    def _scalar(self, name, label):
        return struct.unpack("<f", self.reader.read_bytes(name, 0, 4))[0]


class ArrayKernel:
    def __init__(self):
        self.calls = []
        self.fail = False
    def prepare_dense_vector(self, vector):
        return vector
    prepare_nvfp4_vector = prepare_dense_vector
    def matvec_dense_row_band_array(self, raw, rows, cols, vector, dtype):
        self.calls.append((get_ident(), rows, cols))
        if self.fail:
            raise ArithmeticError("injected kernel failure")
        return array("f", (struct.unpack_from("<f", raw, row * cols * 4)[0] * vector[0] for row in range(rows)))
    def matvec_dense_row_band(self, *_):
        raise AssertionError("Owned-array path must be preferred")
    def matvec_nvfp4_row_band_array(self, raw, rows, cols, scales, vector, global_scale):
        self.calls.append((get_ident(), rows, cols))
        return array("f", (raw[row * (cols // 2)] * scales[row * (cols // 16)] * vector[0] * global_scale for row in range(rows)))
    def matvec_nvfp4_row_band(self, *_):
        raise AssertionError("Owned-array NVFP4 path must be preferred")


class RuntimeReadAheadTests(unittest.TestCase):
    def run_projection(self, reader, cpu, pool, *, vectors=None):
        ledger = ReservationLedger(128 * 1024**2)
        vectors = vectors or [array("f", [0.5]) * reader.cols]
        with patch("glm_local.runtime_linear._matrix", side_effect=lambda source, descriptor: ScalarMatrix(source)):
            output, counts = project_many(reader, "w", reader.info, vectors, cpu, nvfp4=reader.nvfp4,
                                         ledger=ledger, read_ahead=pool)
        self.assertEqual(ledger.snapshot()["active_leases"], 0)
        return output, counts, ledger.snapshot()

    def test_dense_ragged_bands_keep_order_and_reader_ownership_on_one_producer(self):
        sequential, parallel = BandReader(), BandReader()
        with ReadAheadPool() as pool:
            expected, _, _ = self.run_projection(sequential, ArrayKernel(), None)
            cpu = ArrayKernel()
            actual, counts, ledger = self.run_projection(parallel, cpu, pool)
            self.assertEqual(actual, expected)
            self.assertEqual([row[1:] for row in parallel.reads], [row[1:] for row in sequential.reads])
            self.assertEqual([call[1] for call in cpu.calls], [128, 128, 1])
            self.assertEqual({call[0] for call in cpu.calls}, {get_ident()})
            self.assertEqual(len({read[0] for read in parallel.reads}), 1)
            self.assertNotIn(get_ident(), {read[0] for read in parallel.reads})
            self.assertTrue(counts["read_ahead"]["enabled"])
            self.assertEqual(counts["read_ahead"]["consumed_bands"], 3)
            self.assertEqual(counts["read_ahead"]["live_bands"], 0)
            self.assertLessEqual(counts["read_ahead"]["peak_live_bands"], 2)
            self.assertLessEqual(ledger["cpu"]["peak_bytes"], counts["declared_buffer_bytes"] + READ_AHEAD_BUFFER_BYTES)

    def test_nvfp4_producer_owns_weight_scale_and_both_scalar_reads(self):
        sequential, parallel = BandReader(nvfp4=True), BandReader(nvfp4=True)
        with ReadAheadPool() as pool:
            expected, _, _ = self.run_projection(sequential, ArrayKernel(), None)
            actual, counts, _ = self.run_projection(parallel, ArrayKernel(), pool)
        self.assertEqual(actual, expected)
        self.assertEqual([row[1:] for row in parallel.reads], [row[1:] for row in sequential.reads])
        self.assertEqual([row[1] for row in parallel.reads], ["w", "s", "g", "i"] * 3)
        self.assertEqual(len({row[0] for row in parallel.reads}), 1)
        self.assertNotIn(get_ident(), {row[0] for row in parallel.reads})
        self.assertEqual(parallel.metadata_calls, [get_ident()])
        self.assertEqual(counts["band_reads"], 6)

    def test_single_band_and_missing_ledger_keep_sequential_fast_path_without_worker(self):
        with ReadAheadPool() as pool:
            reader = BandReader(rows=7)
            _, counts, _ = self.run_projection(reader, ArrayKernel(), pool)
            self.assertFalse(counts["read_ahead"]["enabled"])
            self.assertIsNone(pool._executor)
            larger = BandReader()
            _, counts = project_many(larger, "w", larger.info, [[0.5] * larger.cols], ArrayKernel(), read_ahead=pool)
            self.assertFalse(counts["read_ahead"]["enabled"])
            self.assertIsNone(pool._executor)

    def test_legacy_list_kernel_still_works_when_array_entry_point_is_unavailable(self):
        class LegacyKernel(ArrayKernel):
            matvec_dense_row_band_array = None
            def matvec_dense_row_band(self, *args):
                return list(ArrayKernel.matvec_dense_row_band_array(self, *args))
        with ReadAheadPool() as pool:
            expected, _, _ = self.run_projection(BandReader(), ArrayKernel(), None)
            actual, _, _ = self.run_projection(BandReader(), LegacyKernel(), pool)
        self.assertEqual(actual, expected)

    def test_kernel_and_reader_failures_join_drain_release_and_allow_pool_reuse(self):
        with ReadAheadPool() as pool:
            for kind in ("kernel", "reader"):
                ledger = ReservationLedger(128 * 1024**2)
                reader, cpu = BandReader(), ArrayKernel()
                if kind == "kernel":
                    cpu.fail = True
                else:
                    original = reader.read_span
                    def bad(name, offset, count):
                        if offset:
                            raise OSError("injected reader failure")
                        return original(name, offset, count)
                    reader.read_span = bad
                with self.subTest(kind=kind), self.assertRaisesRegex((ArithmeticError, OSError), "injected"):
                    project_many(reader, "w", reader.info, [[1.0] * reader.cols], cpu, ledger=ledger, read_ahead=pool)
                self.assertIsNone(pool._active)
                self.assertEqual(ledger.snapshot()["active_leases"], 0)
                self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"], 0)
            actual, _, _ = self.run_projection(BandReader(), ArrayKernel(), pool)
            self.assertEqual(len(actual[0]), 257)

    def test_short_payload_and_insufficient_band_budget_fail_without_retention(self):
        with ReadAheadPool() as pool:
            reader = BandReader()
            reader.data["w"] = reader.data["w"][:-1]
            with self.assertRaisesRegex(ValueError, "incorrect byte count"):
                self.run_projection(reader, ArrayKernel(), pool)
            self.assertIsNone(pool._active)
            reader = BandReader()
            declared = DENSE_ROW_BAND_SCRATCH_BYTES + 4 * (reader.rows + reader.cols) + 2 * 65536
            ledger = ReservationLedger(declared)
            with self.assertRaises(ValueError):
                project_many(reader, "w", reader.info, [[1] * reader.cols], ArrayKernel(), ledger=ledger, read_ahead=pool)
            self.assertEqual(reader.reads, [])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)

    @unittest.skipUnless(os.environ.get("GLM_TEST_NATIVE") == "1", "Explicit tiny native CPU safetensors check")
    def test_native_real_reader_outputs_are_bit_identical_with_two_band_read_ahead(self):
        from glm_local.cpu_probe import NativeCpuBackend
        rows, cols = 385, 144
        payload = b"".join(struct.pack("<f", (i % 23 - 11) * 0.03125) for i in range(rows * cols))
        header = json.dumps({"w": {"dtype": "F32", "shape": [rows, cols], "data_offsets": [0, len(payload)]}},
                            separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            with SafeTensorReader(path) as reader, NativeCpuBackend(row_band_threads=2) as cpu, ReadAheadPool() as pool:
                vectors = [array("f", [value]) * cols for value in (0.25, 0.5, -0.25)]
                ledger = ReservationLedger(128 * 1024**2)
                expected, _ = project_many(reader, "w", reader.tensors["w"], vectors, cpu, ledger=ledger)
                actual, counts = project_many(reader, "w", reader.tensors["w"], vectors, cpu, ledger=ledger, read_ahead=pool)
                self.assertEqual([row.tobytes() for row in actual], [row.tobytes() for row in expected])
                self.assertEqual(counts["read_ahead"]["consumed_bands"], 4)
                self.assertLessEqual(reader.stats()["max_actual_read_bytes"], 65536)
                self.assertEqual(ledger.snapshot()["active_leases"], 0)


if __name__ == "__main__":
    unittest.main()
