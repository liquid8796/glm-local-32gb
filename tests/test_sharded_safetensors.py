import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from glm_local.fp8_blocks import FP8BlockMatrix
from glm_local.safetensor_reader import SafeTensorError
from glm_local.sharded_safetensors import ShardedSafeTensorReader, MAX_INDEX_BYTES
from sharded_test_helpers import write_safe, write_index, read_index


class ShardedReaderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        write_safe(self.path / "a.safetensors", {"w": ("F8_E4M3", (2, 2), b"\x38\x40\xb8\xc0")})
        write_safe(self.path / "b.safetensors", {"s": ("F32", (1, 1), struct.pack("<f", .25))})
        write_safe(self.path / "c.safetensors", {"v": ("F32", (2,), struct.pack("<2f", 2., 3.))})
        write_index(self.path, {"metadata": {"total_size": 16},
                               "weight_map": {"w": "a.safetensors", "s": "b.safetensors", "v": "c.safetensors"}})

    def reader(self, **kwargs):
        return ShardedSafeTensorReader(self.path, **kwargs)

    def alter_index(self, edit):
        data = read_index(self.path)
        edit(data)
        write_index(self.path, data)

    def test_open_reads_only_headers(self):
        with self.reader() as reader:
            self.assertEqual(set(reader.tensors), {"w", "s", "v"})
            self.assertEqual(reader.stats()["tensor_read_bytes"], 0)
            self.assertEqual(reader.stats()["open_shards"], 0)
            self.assertEqual(reader.stats()["shard_open_count"], 3)

    def test_cross_shard_fp8_scale_pair_works_with_one_open_handle(self):
        with self.reader(max_open_shards=1) as reader:
            block = FP8BlockMatrix(reader, "w", "s").read_block(0, 0)
            self.assertEqual((block.rows, block.cols, block.scale), (2, 2, .25))
            self.assertEqual(block.weights, b"\x38\x40\xb8\xc0")
            self.assertEqual(reader.stats()["tensor_read_bytes"], 8)
            self.assertEqual(reader.stats()["peak_open_shards"], 1)
            self.assertEqual(reader.stats()["shard_evictions"], 1)

    def test_lru_reuses_hits_and_evicts_least_recent(self):
        with self.reader(max_open_shards=2) as reader:
            for name in ("w", "s", "w", "v"):
                reader.read_bytes(name, 0, 4)
            stats = reader.stats()
            self.assertEqual(stats["shard_open_count"], 6)
            self.assertEqual(stats["peak_open_shards"], 2)
            self.assertEqual(stats["shard_evictions"], 1)
            reader.read_bytes("w", 0, 4)
            self.assertEqual(reader.stats()["shard_open_count"], 6)
            reader.read_bytes("s", 0, 4)
            self.assertEqual(reader.stats()["shard_open_count"], 7)

    def test_statistics_survive_close_without_double_counting(self):
        reader = self.reader()
        reader.read_bytes("w", 0, 4)
        before = reader.stats()
        reader.close()
        reader.close()
        after = reader.stats()
        self.assertEqual(after["tensor_read_bytes"], 4)
        self.assertEqual(after["actual_read_bytes"], before["actual_read_bytes"])
        self.assertEqual(after["open_shards"], 0)

    def test_metadata_views_cannot_mutate_internal_contract(self):
        with self.reader() as reader:
            with self.assertRaises(TypeError):
                reader.weight_map["w"] = "c.safetensors"
            with self.assertRaises(TypeError):
                reader.tensors["w"] = None
            metadata = reader.metadata
            metadata["total_size"] = -1
            self.assertEqual(reader.metadata["total_size"], 16)

    def test_invalid_handle_budgets(self):
        for value in (0, -1, 9, True, 1.5, None):
            with self.subTest(value=value), self.assertRaises(SafeTensorError):
                self.reader(max_open_shards=value)

    def test_missing_shard(self):
        (self.path / "b.safetensors").unlink()
        with self.assertRaises(SafeTensorError):
            self.reader()

    def test_unknown_tensor_is_explicit_error(self):
        with self.reader() as reader, self.assertRaisesRegex(SafeTensorError, "Unknown"):
            reader.read_bytes("not-here", 0, 1)

    def test_closed_reader_rejects_reads_and_context_entry(self):
        reader = self.reader()
        reader.close()
        for operation in (lambda: reader.read_bytes("w", 0, 1), reader.__enter__):
            with self.assertRaisesRegex(SafeTensorError, "closed"):
                operation()

    def test_changed_open_shard_is_rejected(self):
        with self.reader() as reader:
            reader.read_bytes("w", 0, 1)
            with (self.path / "a.safetensors").open("ab") as file:
                file.write(b"x")
            with self.assertRaisesRegex(SafeTensorError, "changed"):
                reader.read_bytes("w", 0, 1)

    def test_changed_evicted_shard_is_not_accepted_as_a_new_baseline(self):
        with self.reader(max_open_shards=1) as reader:
            reader.read_bytes("w", 0, 1)
            reader.read_bytes("s", 0, 4)
            write_safe(self.path / "a.safetensors", {"w": ("F8_E4M3", (2, 2), b"\x00" * 4)})
            # Force a changed timestamp even on coarse timestamp filesystems.
            info = (self.path / "a.safetensors").stat()
            os.utime(self.path / "a.safetensors", ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
            with self.assertRaisesRegex(SafeTensorError, "changed"):
                reader.read_bytes("w", 0, 1)

    def test_changed_index_is_rejected_before_next_payload_read(self):
        with self.reader() as reader:
            with (self.path / "model.safetensors.index.json").open("a") as file:
                file.write(" ")
            with self.assertRaisesRegex(SafeTensorError, "index changed"):
                reader.read_bytes("w", 0, 1)

    def test_total_size_is_payload_not_serialized_file_size(self):
        self.alter_index(lambda data: data["metadata"].update(total_size=999))
        with self.assertRaisesRegex(SafeTensorError, "total_size"):
            self.reader()

    def test_index_missing_header_tensor_is_rejected(self):
        self.alter_index(lambda data: data["weight_map"].update(other="a.safetensors"))
        with self.assertRaisesRegex(SafeTensorError, "mapping mismatch"):
            self.reader()

    def test_unindexed_extra_tensor_in_referenced_shard_is_rejected(self):
        write_safe(self.path / "a.safetensors", {"w": ("F8_E4M3", (2, 2), bytes(4)),
                                                "extra": ("F32", (1,), bytes(4))})
        with self.assertRaisesRegex(SafeTensorError, "mapping mismatch"):
            self.reader()

    def test_wrong_shard_assignment_is_rejected(self):
        self.alter_index(lambda data: data["weight_map"].update(w="b.safetensors"))
        with self.assertRaisesRegex(SafeTensorError, "mapping mismatch"):
            self.reader()

    def test_required_shard_set_checked_before_opening_any_shard(self):
        with patch("glm_local.sharded_safetensors.SafeTensorReader") as opener:
            with self.assertRaisesRegex(SafeTensorError, "required shard"):
                self.reader(expected_shards=("different.safetensors",))
            opener.assert_not_called()

    def test_unsafe_nonportable_names_are_rejected(self):
        original = read_index(self.path)
        for value in ("../a.safetensors", "/a.safetensors", "C:\\a.safetensors",
                      "sub/a.safetensors", "sub\\a.safetensors", "a.safetensors:other",
                      "a.bin", "", None, "CON.safetensors", "COM1.safetensors"):
            with self.subTest(value=value):
                data = json.loads(json.dumps(original))
                data["weight_map"]["w"] = value
                write_index(self.path, data)
                with self.assertRaisesRegex(SafeTensorError, "filename"):
                    self.reader()

    def test_case_colliding_shard_names_are_rejected(self):
        self.alter_index(lambda data: data["weight_map"].update(s="A.safetensors"))
        with self.assertRaisesRegex(SafeTensorError, "case-colliding"):
            self.reader()

    def test_symlink_shard_is_rejected(self):
        file = self.path / "a.safetensors"
        file.rename(self.path / "real.safetensors")
        try:
            file.symlink_to(self.path / "real.safetensors")
        except OSError:
            self.skipTest("Creating symlinks not permitted on this platform")
        with self.assertRaisesRegex(SafeTensorError, "symlink"):
            self.reader()

    def test_duplicate_index_json_key_is_rejected(self):
        (self.path / "model.safetensors.index.json").write_text(
            '{"metadata":{"total_size":0},"weight_map":{"w":"a.safetensors","w":"b.safetensors"}}')
        with self.assertRaisesRegex(SafeTensorError, "Duplicate"):
            self.reader()

    def test_invalid_metadata_and_mapping_types(self):
        invalid = [{}, [], {"metadata": {}, "weight_map": {}},
                   {"metadata": {"total_size": True}, "weight_map": {"w": "a.safetensors"}},
                   {"metadata": {"total_size": -1}, "weight_map": {"w": "a.safetensors"}},
                   {"metadata": {"total_size": 0}, "weight_map": []}]
        for value in invalid:
            with self.subTest(value=value):
                write_index(self.path, value)
                with self.assertRaises(SafeTensorError):
                    self.reader()

    def test_nonfinite_json_and_invalid_encoding(self):
        for value in (b'\xff', b'{', b'{"metadata":{"total_size":NaN},"weight_map":{}}'):
            (self.path / "model.safetensors.index.json").write_bytes(value)
            with self.subTest(value=value), self.assertRaises(SafeTensorError):
                self.reader()

    def test_oversize_index_is_rejected_before_read(self):
        with (self.path / "model.safetensors.index.json").open("wb") as stream:
            stream.truncate(MAX_INDEX_BYTES + 1)
        with patch("glm_local.sharded_safetensors.SafeTensorReader") as opener:
            with self.assertRaisesRegex(SafeTensorError, "1-MiB"):
                self.reader()
            opener.assert_not_called()

    def test_index_read_is_chunked(self):
        self.alter_index(lambda data: data["metadata"].update(note="x" * 100_000))
        with self.reader() as reader:
            self.assertGreater(reader.stats()["index_read_calls"], 1)
            self.assertLessEqual(reader.stats()["index_max_read_bytes"], 65536)

    def test_reader_preserves_small_bounded_io(self):
        with self.reader() as reader:
            reader.read_matrix_tile("w", 0, 0, 2, 2)
            self.assertLessEqual(reader.stats()["max_actual_read_bytes"], 65536)
            self.assertEqual(reader.stats()["tensor_read_bytes"], 4)
            with self.assertRaises(SafeTensorError):
                reader.read_bytes("w", 0, 65537)

    def test_zero_length_tensor_with_zero_payload_is_valid(self):
        write_safe(self.path / "empty.safetensors", {"e": ("F32", (0, 2), b"")})
        write_index(self.path, {"metadata": {"total_size": 0}, "weight_map": {"e": "empty.safetensors"}})
        with self.reader() as reader:
            self.assertEqual(reader.read_bytes("e", 0, 0), b"")
            self.assertEqual(reader.stats()["tensor_read_bytes"], 0)
