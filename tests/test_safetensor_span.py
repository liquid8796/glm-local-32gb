"""Owned span buffers, protected local handles and bounded synthetic file reads."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from glm_local import safetensor_reader as storage
from glm_local.catalogue_reader import SelectedCatalogueReader
from glm_local.execution import build_projection_descriptor
from glm_local.residency import BudgetExceededError, ReservationLedger
from glm_local.runtime_io import RowBandCacheReader
from glm_local.safetensor_reader import (MAX_READ_BYTES, MAX_READ_SPAN_BYTES,
                                         SafeTensorError, SafeTensorReader, _uint)
from checkpoint_test_helpers import NAME, settings
from test_execution import projection_fixture
from test_runtime_io import CountingReader
from test_safetensor_reader import encode, entry


class HookedStream:
    def __init__(self, stream, hook=None):
        self.stream, self.hook = stream, hook
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def readinto(self, target):
        self.calls.append(len(target))
        count = self.stream.readinto(target)
        if self.hook is not None:
            self.hook(len(self.calls), target, count)
        return count


class SpanReaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="glm-span-tests-")
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(temporary.cleanup)
        self.path = self.root / "fixture.safetensors"

    def write(self, size=256 * 1024, *, dtype="U8"):
        itemsize = storage.DTYPE_ITEMSIZE[dtype]
        payload = (bytes(range(256)) * ((size + 255) // 256))[:size]
        self.path.write_bytes(encode({"w": entry(dtype, [size // itemsize], [0, size])}, payload))
        return payload

    def test_owned_partial_span_uses_64k_reads_and_does_not_modify_file(self):
        payload = self.write()
        before = self.path.read_bytes()
        with SafeTensorReader(self.path) as reader:
            result = reader.read_span("w", 14, 170_003)
            self.assertIs(type(result), bytearray)
            self.assertEqual(result, payload[14:170_017])
            stats = reader.stats()
            self.assertEqual(stats["span_read_calls"], 1)
            self.assertEqual(stats["span_inner_read_calls"], 3)
            self.assertEqual(stats["span_read_bytes"], 170_003)
            self.assertEqual(stats["max_actual_read_bytes"], MAX_READ_BYTES)
            self.assertEqual(stats["max_span_bytes"], 170_003)
            result[0] ^= 255
            self.assertEqual(reader.read_bytes("w", 14, 1), payload[14:15])
        self.assertEqual(self.path.read_bytes(), before)

    def test_warmed_span_checks_identity_twice_not_on_every_inner_chunk(self):
        self.write(1024 * 1024)
        with SafeTensorReader(self.path) as reader:
            reader.read_span("w", 0, 1)  # Upgrade the Windows handle outside the measured span.
            wrapped = HookedStream(reader._stream)
            reader._stream = wrapped
            with patch.object(reader, "_assert_unchanged", wraps=reader._assert_unchanged) as unchanged:
                reader.read_span("w", 0, 1024 * 1024)
            self.assertEqual(wrapped.calls, [MAX_READ_BYTES] * 16)
            self.assertEqual(unchanged.call_count, 2)
            with patch.object(reader, "_assert_unchanged", wraps=reader._assert_unchanged) as unchanged:
                for offset in range(0, 1024 * 1024, MAX_READ_BYTES):
                    reader.read_bytes("w", offset, MAX_READ_BYTES)
            self.assertEqual(unchanged.call_count, 48)

    def test_span_cap_alignment_and_invalid_ranges_fail_before_payload_io(self):
        self.write(MAX_READ_SPAN_BYTES + MAX_READ_BYTES, dtype="F32")
        with SafeTensorReader(self.path) as reader:
            for offset, count in ((-1, 4), (True, 4), (0, True), (1, 4), (0, 3),
                                  (0, MAX_READ_SPAN_BYTES + 4), (MAX_READ_SPAN_BYTES + MAX_READ_BYTES, 4)):
                with self.subTest(offset=offset, count=count), self.assertRaises(SafeTensorError):
                    reader.read_span("w", offset, count)
            self.assertEqual(reader.stats()["tensor_read_calls"], 0)
            self.assertEqual(reader.stats()["protected_handle_upgrades"], 0)
            self.assertEqual(reader.read_span("w", 0, 0), bytearray())
            self.assertEqual(reader.stats()["protected_handle_upgrades"], 0)
            self.assertEqual(len(reader.read_span("w", 0, MAX_READ_SPAN_BYTES)), MAX_READ_SPAN_BYTES)
            with self.assertRaises(SafeTensorError):
                reader.read_bytes("w", 0, MAX_READ_BYTES + 4)
        with self.assertRaisesRegex(SafeTensorError, "closed"):
            reader.read_span("w", 0, 0)

    def test_fingerprint_validation_rejects_mutation_during_unlocked_logical_span(self):
        self.write()
        with SafeTensorReader(self.path) as reader:
            def mutate(number, _target, _count):
                if number == 1:
                    with self.path.open("r+b") as writer:
                        writer.seek(reader._payload_start + MAX_READ_BYTES + 5)
                        writer.write(b"changed")
                    # Force an observable timestamp independently of filesystem
                    # granularity/deferred Windows last-write updates.
                    old = self.path.stat()
                    os.utime(self.path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
            reader._stream = HookedStream(reader._stream, mutate)
            # Exercise the portable boundary-check path even on a Windows test host.
            with patch.object(reader, "_protect_span_handle", return_value=None):
                with self.assertRaisesRegex(SafeTensorError, "changed"):
                    reader.read_span("w", 0, 2 * MAX_READ_BYTES)

    def test_short_invalid_and_failing_inner_reads_never_return_a_partial_span(self):
        self.write()
        for failure in ("short", "invalid", "io"):
            with self.subTest(failure=failure), SafeTensorReader(self.path) as reader:
                reader.read_span("w", 0, 1)
                stream = reader._stream
                class Broken:
                    def __getattr__(self, name):
                        return getattr(stream, name)

                    def readinto(self, target):
                        if failure == "short":
                            return stream.readinto(target[:-1])
                        if failure == "invalid":
                            return len(target) + 1
                        raise OSError("fixture I/O failure")
                reader._stream = Broken()
                with self.assertRaises(SafeTensorError):
                    reader.read_span("w", 0, 2 * MAX_READ_BYTES)

    def test_preexisting_mutation_and_closed_stream_reject_span(self):
        self.write()
        with SafeTensorReader(self.path) as reader:
            with self.path.open("ab") as writer:
                writer.write(b"x")
            with self.assertRaisesRegex(SafeTensorError, "changed"):
                reader.read_span("w", 0, 1)
        self.write()
        reader = SafeTensorReader(self.path); reader.close()
        with self.assertRaisesRegex(SafeTensorError, "closed"):
            reader.read_span("w", 0, 1)

    @unittest.skipUnless(os.name == "nt", "Windows sharing-mode behavior")
    def test_windows_span_handle_denies_writes_and_delete_until_close(self):
        payload = self.write()
        with SafeTensorReader(self.path) as reader:
            self.assertEqual(reader.read_span("w", 0, 100), payload[:100])
            self.assertTrue(reader._span_protected)
            self.assertEqual(reader.stats()["span_handle_protection"], "windows_deny_write_delete")
            with self.assertRaises(PermissionError):
                with self.path.open("r+b"):
                    pass
            with self.assertRaises(PermissionError):
                self.path.unlink()
            reader.read_span("w", 1, 100)
            self.assertEqual(reader.stats()["protected_handle_upgrades"], 1)
        with self.path.open("r+b") as writer:
            writer.seek(0)
        self.path.unlink()

    @unittest.skipUnless(os.name == "nt", "Windows sharing-mode behavior")
    def test_existing_writer_makes_upgrade_fail_and_closes_the_reader(self):
        self.write()
        reader = SafeTensorReader(self.path)
        original = reader._stream
        with self.path.open("r+b"):
            with self.assertRaisesRegex(SafeTensorError, "protected span"):
                reader.read_span("w", 0, 100)
        self.assertTrue(original.closed)
        self.assertIsNone(reader._stream)
        self.assertEqual(reader.stats()["tensor_read_calls"], 0)
        reader.close()

    @unittest.skipUnless(os.name == "nt", "Windows protected reopen")
    def test_replacement_during_upgrade_does_not_refresh_original_fingerprint(self):
        self.write()
        original_bytes = self.path.read_bytes()
        opened = []
        real_open = storage._open_protected_read
        def replace_before_open(path):
            self.path.rename(self.root / "old.safetensors")
            self.path.write_bytes(original_bytes)
            result = real_open(path); opened.append(result); return result
        reader = SafeTensorReader(self.path)
        with patch.object(storage, "_open_protected_read", side_effect=replace_before_open):
            with self.assertRaisesRegex(SafeTensorError, "changed"):
                reader.read_span("w", 0, 100)
        self.assertTrue(all(stream.closed for stream in opened))
        self.assertIsNone(reader._stream)
        self.assertEqual(reader.stats()["tensor_read_calls"], 0)

    def test_selected_catalogue_binds_one_shard_per_span_and_keeps_stats_after_close(self):
        directory, _, _ = projection_fixture(self.root, rows=257, cols=259)
        desc = build_projection_descriptor(self.root, settings(), NAME)
        reader = SelectedCatalogueReader(directory, desc, max_open_shards=1)
        with reader:
            with patch.object(reader, "_reader", wraps=reader._reader) as bind:
                result = reader.read_span(NAME, 0, 128 * 259)
            self.assertEqual(bind.call_count, 1)
            self.assertIs(type(result), bytearray)
            reader.read_span(desc.scale.name, 0, 4)
            self.assertEqual(reader.stats()["peak_open_shards"], 1)
            self.assertEqual(reader.stats()["span_read_calls"], 2)
        self.assertEqual(reader.stats()["span_read_bytes"], 128 * 259 + 4)
        self.assertEqual(reader.stats()["open_shards"], 0)


class CountingSpanReader(CountingReader):
    def __init__(self, specifications):
        super().__init__(specifications)
        self.span_calls = []
        self.last_buffer = None
        self.bad_span = None

    def read_span(self, name, offset, count):
        self.assert_tensor_unchanged(name)
        info = self.tensors[name]
        _uint(offset, info.nbytes, "offset"); _uint(count, MAX_READ_SPAN_BYTES, "count")
        if offset + count > info.nbytes:
            raise SafeTensorError("span exceeds tensor")
        self.span_calls.append((name, offset, count))
        self.last_buffer = bytearray(self.expected_bytes(name, offset, count))
        if self.bad_span == "short":
            return self.last_buffer[:-1]
        if self.bad_span == "immutable":
            return bytes(self.last_buffer)
        if self.bad_span == "changed":
            self.changed = True
        self.assert_tensor_unchanged(name)
        return self.last_buffer


class CachedSpanTests(unittest.TestCase):
    def test_cache_prefers_span_and_retains_its_owned_buffer_without_copying(self):
        source = CountingSpanReader({"w": ((256, 1024), "U8")})
        ledger = ReservationLedger(128 * 1024)
        with RowBandCacheReader(source, ledger) as reader:
            actual = reader.read_matrix_tile("w", 0, 0, 128, 128)
            self.assertEqual(actual, source.expected_tile("w", 0, 0, 128, 128))
            self.assertIs(reader._bands["w"].payload, source.last_buffer)
            reader.read_matrix_tile("w", 0, 128, 128, 128)
            self.assertEqual(source.span_calls, [("w", 0, 128 * 1024)])
            self.assertEqual(source.reads, [])
            self.assertEqual(reader.stats()["row_band_cache"]["cache_span_calls"], 1)
            self.assertEqual(ledger.snapshot()["cpu"]["used_bytes"], 128 * 1024)
        source.last_buffer = None
        self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_span_bad_type_size_or_mutation_releases_cache_lease(self):
        for failure in ("short", "immutable", "changed"):
            with self.subTest(failure=failure):
                source = CountingSpanReader({"w": ((16, 16), "U8")})
                source.bad_span = failure
                ledger = ReservationLedger(1024)
                with RowBandCacheReader(source, ledger) as reader:
                    with self.assertRaises(SafeTensorError):
                        reader.read_matrix_tile("w", 0, 0, 4, 4)
                    self.assertEqual(reader.stats()["retained_weight_payload_bytes"], 0)
                    self.assertEqual(ledger.snapshot()["active_leases"], 0)

    def test_cache_span_reservation_precedes_source_read(self):
        source = CountingSpanReader({"w": ((16, 16), "U8")})
        ledger = ReservationLedger(63)
        with RowBandCacheReader(source, ledger) as reader:
            with self.assertRaises(BudgetExceededError):
                reader.read_matrix_tile("w", 0, 0, 4, 4)
        self.assertEqual(source.span_calls, [])

    def test_public_span_passthrough_and_chunk_fallback_return_owned_buffers(self):
        for source_type in (CountingReader, CountingSpanReader):
            with self.subTest(source=source_type.__name__):
                source = source_type({"w": ((256, 1024), "U8")})
                with RowBandCacheReader(source) as reader:
                    result = reader.read_span("w", 3, 150_000)
                    self.assertIs(type(result), bytearray)
                    self.assertEqual(result, source.expected_bytes("w", 3, 150_000))
                    self.assertEqual(reader.stats()["retained_weight_payload_bytes"], 0)
                    if source_type is CountingReader:
                        self.assertEqual([item[2] for item in source.reads], [MAX_READ_BYTES, MAX_READ_BYTES, 18_928])
                    with self.assertRaises(SafeTensorError):
                        reader.read_span("w", 0, MAX_READ_SPAN_BYTES + 1)
                with self.assertRaisesRegex(SafeTensorError, "closed"):
                    reader.read_span("w", 0, 0)


if __name__ == "__main__":
    unittest.main()
