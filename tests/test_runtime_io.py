"""Row-band I/O accounting with deterministic bytes, never real model payloads."""
from types import MappingProxyType
import unittest
from unittest.mock import patch

from glm_local.residency import BudgetExceededError, ReservationLedger
from glm_local.runtime_io import DEFAULT_ROW_BAND_CACHE_BYTES, RowBandCacheReader
from glm_local.safetensor_reader import (DTYPE_ITEMSIZE, MAX_READ_BYTES, SafeTensorError,
                                         SafeTensorReader, TensorInfo, _uint)


class CountingReader:
    """A bounded source with the original tile gather and controllable identity."""

    def __init__(self, specifications):
        tensors = {}
        for name, (shape, dtype) in specifications.items():
            elements = 1
            for dimension in shape:
                elements *= dimension
            size = elements * DTYPE_ITEMSIZE[dtype]
            tensors[name] = TensorInfo(dtype, tuple(shape), (0, size), size, DTYPE_ITEMSIZE[dtype])
        self._tensors = MappingProxyType(tensors)
        self.config = {"fixture": "row-band"}
        self.closed = False
        self.changed = False
        self.reads = []
        self.tile_calls = 0
        self.identity_checks = 0
        self.mutate_on_check = None
        self.mutate_after_read = None
        self.short_after_read = None
        self.read_observer = None

    @property
    def tensors(self):
        return self._tensors

    _tensor = SafeTensorReader._tensor

    def _assert_unchanged(self):
        self.identity_checks += 1
        if self.mutate_on_check == self.identity_checks:
            self.changed = True
        if self.closed or self.changed:
            raise SafeTensorError("Fixture source is closed or changed")

    def assert_tensor_unchanged(self, name):
        self._tensor(name)
        self._assert_unchanged()

    def expected_bytes(self, name, offset, count):
        shift = (sum(name.encode()) + offset) % 256
        pattern = bytes(range(256)) * ((shift + count + 255) // 256)
        return pattern[shift:shift + count]

    def expected_tile(self, name, row, col, rows, cols):
        info = self.tensors[name]
        return b"".join(self.expected_bytes(name, ((row + index) * info.shape[1] + col) * info.itemsize,
                                             cols * info.itemsize) for index in range(rows))

    def read_bytes(self, name, offset, count):
        info = self._tensor(name)
        offset = _uint(offset, info.nbytes, "Tensor-relative byte offset")
        count = _uint(count, MAX_READ_BYTES, "Read byte count")
        if offset % info.itemsize or count % info.itemsize:
            raise SafeTensorError("Byte offset and count must align with tensor itemsize")
        if count > info.nbytes - offset:
            raise SafeTensorError("Read extends beyond tensor bounds")
        self._assert_unchanged()
        if count == 0:
            return b""
        self.reads.append((name, offset, count))
        if self.read_observer is not None:
            self.read_observer()
        result = self.expected_bytes(name, offset, count)
        if self.mutate_after_read == len(self.reads):
            self.changed = True
        return result[:-1] if self.short_after_read == len(self.reads) else result

    def read_matrix_tile(self, name, row, col, rows, cols):
        self.tile_calls += 1
        return SafeTensorReader.read_matrix_tile(self, name, row, col, rows, cols)

    def projection(self, name):
        return ("descriptor", name)

    def stats(self):
        return {"actual_read_calls": len(self.reads), "tensor_read_calls": len(self.reads),
                "actual_read_bytes": sum(read[2] for read in self.reads),
                "tensor_read_bytes": sum(read[2] for read in self.reads),
                "max_actual_read_bytes": max((read[2] for read in self.reads), default=0),
                "retained_weight_payload_bytes": 0}

    def close(self):
        self.closed = True


class RowBandCacheTests(unittest.TestCase):
    def test_nvfp4_weight_and_scale_columns_use_seven_reads_instead_of_per_row_reads(self):
        source = CountingReader({"weight": ((256, 3072), "U8"), "scale": ((256, 384), "F8_E4M3")})
        ledger = ReservationLedger(DEFAULT_ROW_BAND_CACHE_BYTES)
        with RowBandCacheReader(source, ledger) as reader:
            for column in range(48):
                for name, width in (("weight", 64), ("scale", 8)):
                    actual = reader.read_matrix_tile(name, 0, column * width, 128, width)
                    self.assertEqual(actual, source.expected_tile(name, 0, column * width, 128, width))
            stats = reader.stats()
            self.assertEqual(stats["tensor_read_calls"], 7)
            self.assertEqual(stats["row_band_cache"]["coalesced_read_calls"], 7)
            self.assertEqual(stats["row_band_cache"]["hits"], 94)
            self.assertEqual(stats["row_band_cache"]["misses"], 2)
            self.assertEqual(stats["retained_weight_payload_bytes"], 128 * (3072 + 384))
            self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"], stats["retained_weight_payload_bytes"])
            self.assertLess(stats["tensor_read_calls"], 2 * 128 * 48 // 100)
            self.assertLessEqual(stats["max_actual_read_bytes"], MAX_READ_BYTES)
            self.assertEqual(source.tile_calls, 0)
        self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_unaligned_and_partial_tiles_preserve_encoded_bytes_for_all_item_sizes(self):
        for dtype in ("U8", "F8_E4M3", "BF16", "F16", "F32", "F64"):
            with self.subTest(dtype=dtype):
                source = CountingReader({"matrix": ((257, 259), dtype)})
                with RowBandCacheReader(source) as reader:
                    for row, col, rows, cols in ((3, 5, 17, 19), (3, 231, 17, 28),
                                                  (255, 256, 2, 3), (256, 258, 1, 1)):
                        actual = reader.read_matrix_tile("matrix", row, col, rows, cols)
                        self.assertIsInstance(actual, bytes)
                        self.assertEqual(actual, source.expected_tile("matrix", row, col, rows, cols))
                        self.assertLessEqual(len(actual), MAX_READ_BYTES)

    def test_valid_input_edges_and_rejections_match_original_reader(self):
        source = CountingReader({"matrix": ((256, 259), "F64"), "scalar": ((), "F32")})
        invalid = [("matrix", True, 0, 1, 1), ("matrix", 0, -1, 1, 1),
                   ("matrix", 0, 0, 0, 1), ("matrix", 0, 0, 1, 0),
                   ("matrix", 0, 0, 129, 1), ("matrix", 0, 0, 1, 129),
                   ("matrix", 256, 0, 1, 1), ("matrix", 0, 259, 1, 1),
                   ("matrix", 255, 258, 2, 1), ("matrix", 0, 0, 128, 128),
                   ("scalar", 0, 0, 1, 1), (None, 0, 0, 1, 1), ("missing", 0, 0, 1, 1)]
        with RowBandCacheReader(source) as reader:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    with self.assertRaises((SafeTensorError, KeyError)) as original:
                        source.read_matrix_tile(*arguments)
                    with self.assertRaises(type(original.exception)) as cached:
                        reader.read_matrix_tile(*arguments)
                    self.assertEqual(str(cached.exception), str(original.exception))
            actual = reader.read_matrix_tile("matrix", 0, 0, 64, 128)
            self.assertEqual(len(actual), MAX_READ_BYTES)
        self.assertEqual(len(source.reads), 3)  # 64 full F64 rows: two full chunks and a tail.

    def test_scalar_and_vector_reads_remain_passthrough(self):
        source = CountingReader({"scalar": ((), "F32"), "vector": ((5,), "BF16")})
        with RowBandCacheReader(source) as reader:
            for _ in range(2):
                self.assertEqual(reader.read_bytes("scalar", 0, 4), source.expected_bytes("scalar", 0, 4))
            self.assertEqual(reader.read_bytes("vector", 2, 4), source.expected_bytes("vector", 2, 4))
            self.assertEqual(reader.read_bytes("vector", 10, 0), b"")
            with self.assertRaises(SafeTensorError):
                reader.read_bytes("vector", 1, 2)
            stats = reader.stats()
            self.assertEqual(stats["tensor_read_calls"], 3)
            self.assertEqual(stats["row_band_cache"]["retained_bytes"], 0)

    def test_identity_change_rejects_a_hit_before_returning_cached_data(self):
        source = CountingReader({"matrix": ((16, 16), "U8")})
        ledger = ReservationLedger(1024)
        with RowBandCacheReader(source, ledger, 1024) as reader:
            reader.read_matrix_tile("matrix", 0, 0, 4, 4)
            reads = len(source.reads)
            source.changed = True
            with self.assertRaisesRegex(SafeTensorError, "changed"):
                reader.read_matrix_tile("matrix", 0, 4, 4, 4)
            self.assertEqual(len(source.reads), reads)
            self.assertEqual(reader.stats()["row_band_cache"]["retained_bytes"], 0)
            self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_identity_is_checked_again_after_cached_tile_extraction(self):
        source = CountingReader({"matrix": ((16, 16), "U8")})
        ledger = ReservationLedger(1024)
        with RowBandCacheReader(source, ledger, 1024) as reader:
            reader.read_matrix_tile("matrix", 0, 0, 4, 4)
            source.mutate_on_check = source.identity_checks + 2
            with self.assertRaisesRegex(SafeTensorError, "changed"):
                reader.read_matrix_tile("matrix", 0, 4, 4, 4)
            self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_changed_or_short_fill_releases_its_lease_and_retains_nothing(self):
        for failure in ("mutation", "short"):
            with self.subTest(failure=failure):
                source = CountingReader({"matrix": ((256, 1024), "U8")})
                ledger = ReservationLedger(256 * 1024)
                if failure == "mutation":
                    source.mutate_after_read = 2
                else:
                    source.short_after_read = 2
                with RowBandCacheReader(source, ledger) as reader:
                    with self.assertRaises(SafeTensorError):
                        reader.read_matrix_tile("matrix", 0, 0, 128, 64)
                    self.assertEqual(ledger.snapshot()["active_leases"], 0)
                    self.assertEqual(reader.stats()["row_band_cache"]["retained_bytes"], 0)

    def test_global_lru_evicts_and_releases_before_new_allocation(self):
        source = CountingReader({name: ((8, 8), "U8") for name in ("a", "b", "c")})
        ledger = ReservationLedger(32)
        observed = []
        source.read_observer = lambda: observed.append(ledger.snapshot()["cpu"]["used_bytes"])
        with RowBandCacheReader(source, ledger, 32) as reader:
            for name, column in (("a", 0), ("b", 0), ("a", 2), ("c", 0), ("a", 4), ("b", 2)):
                self.assertEqual(reader.read_matrix_tile(name, 0, column, 2, 2),
                                 source.expected_tile(name, 0, column, 2, 2))
            self.assertEqual([entry[0] for entry in source.reads], ["a", "b", "c", "b"])
            self.assertEqual(observed, [16, 32, 32, 32])
            self.assertEqual(reader.stats()["row_band_cache"]["evictions"], 2)
            self.assertEqual(ledger.snapshot()["cpu"]["peak_bytes"], 32)
        self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"], 0)

    def test_switching_rows_replaces_the_old_band_not_accumulates_a_matrix(self):
        source = CountingReader({"matrix": ((6, 8), "U8")})
        with RowBandCacheReader(source, max_cache_bytes=64) as reader:
            for row in (0, 2, 4):
                reader.read_matrix_tile("matrix", row, 0, 2, 4)
                stats = reader.stats()["row_band_cache"]
                self.assertEqual(stats["cached_bands"], 1)
                self.assertEqual(stats["retained_bytes"], 16)
                self.assertFalse(stats["full_matrix_retained"])

    def test_oversized_disabled_or_complete_matrix_bands_use_original_tile_reader(self):
        for capacity, rows in ((32, 4), (0, 2), (1024, 8)):
            with self.subTest(capacity=capacity, rows=rows):
                source = CountingReader({"matrix": ((8, 16), "U8")})
                ledger = ReservationLedger(1024)
                with RowBandCacheReader(source, ledger, capacity) as reader:
                    actual = reader.read_matrix_tile("matrix", 0, 3, rows, 4)
                    self.assertEqual(actual, source.expected_tile("matrix", 0, 3, rows, 4))
                    self.assertEqual(source.tile_calls, 1)
                    self.assertEqual(len(source.reads), rows)
                    self.assertEqual(reader.stats()["row_band_cache"]["fallback_tiles"], 1)
                    self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_ledger_failure_precedes_payload_reads_and_buffer_allocation(self):
        source = CountingReader({"matrix": ((16, 16), "U8")})
        ledger = ReservationLedger(63)
        with RowBandCacheReader(source, ledger, 1024) as reader:
            with patch("glm_local.runtime_io.bytearray", side_effect=AssertionError("No allocation"), create=True):
                with self.assertRaises(BudgetExceededError):
                    reader.read_matrix_tile("matrix", 0, 0, 4, 4)
            self.assertEqual(source.reads, [])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_allocation_failure_releases_the_lease(self):
        source = CountingReader({"matrix": ((16, 16), "U8")})
        ledger = ReservationLedger(1024)
        with RowBandCacheReader(source, ledger, 1024) as reader:
            with patch("glm_local.runtime_io.bytearray", side_effect=MemoryError("fixture"), create=True):
                with self.assertRaises(MemoryError):
                    reader.read_matrix_tile("matrix", 0, 0, 4, 4)
            self.assertEqual(source.reads, [])
            self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_metadata_entries_are_bounded_even_for_single_byte_bands(self):
        source = CountingReader({f"tiny-{index}": ((2, 1), "U8") for index in range(129)})
        with RowBandCacheReader(source) as reader:
            for name in source.tensors:
                reader.read_matrix_tile(name, 0, 0, 1, 1)
            stats = reader.stats()["row_band_cache"]
            self.assertEqual(stats["cached_bands"], stats["max_cached_bands"])
            self.assertEqual(stats["evictions"], 1)
            self.assertEqual(stats["retained_bytes"], 128)

    def test_close_releases_cache_is_idempotent_and_can_leave_source_owned_by_caller(self):
        for owns_source in (True, False):
            with self.subTest(owns_source=owns_source):
                source = CountingReader({"matrix": ((8, 16), "U8")})
                ledger = ReservationLedger(1024)
                reader = RowBandCacheReader(source, ledger, owns_source=owns_source)
                self.assertIs(reader.tensors, source.tensors)
                self.assertIs(reader.config, source.config)
                self.assertEqual(reader.projection("matrix"), source.projection("matrix"))
                reader.read_matrix_tile("matrix", 0, 0, 4, 4)
                reader.close(); reader.close()
                self.assertEqual(source.closed, owns_source)
                self.assertEqual(ledger.snapshot()["active_leases"], 0)
                self.assertEqual(reader.stats()["retained_weight_payload_bytes"], 0)
                for operation in (lambda: reader.read_bytes("matrix", 0, 0),
                                  lambda: reader.read_matrix_tile("matrix", 0, 0, 1, 1),
                                  lambda: reader.assert_tensor_unchanged("matrix"), lambda: reader.__enter__()):
                    with self.assertRaisesRegex(SafeTensorError, "closed"):
                        operation()

    def test_invalid_capacity_and_missing_identity_contract_fail_before_reading(self):
        source = CountingReader({"matrix": ((8, 16), "U8")})
        for capacity in (-1, True, 1.5, DEFAULT_ROW_BAND_CACHE_BYTES + 1):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                RowBandCacheReader(source, max_cache_bytes=capacity)
        with self.assertRaisesRegex(ValueError, "assert_tensor_unchanged"):
            RowBandCacheReader(object())
        with self.assertRaises(ValueError):
            RowBandCacheReader(source, owns_source=1)
        self.assertEqual(source.reads, [])
        self.assertFalse(source.closed)


if __name__ == "__main__":
    unittest.main()
