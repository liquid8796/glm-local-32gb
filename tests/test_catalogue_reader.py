"""Selected local shard validation, large audit indexes and bounded LRU reads."""

from pathlib import Path
import tempfile
import unittest

from glm_local.catalogue_reader import SelectedCatalogueReader
from glm_local.execution import build_projection_descriptor
from glm_local.sharded_safetensors import MAX_INDEX_BYTES, MAX_INDEX_TENSORS
from checkpoint_test_helpers import NAME, SCALE, settings
from test_execution import projection_fixture


class CatalogueReaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory, self.report, self.run = projection_fixture(self.root)
        self.descriptor = build_projection_descriptor(self.root, settings(), NAME)

    def test_construction_reads_headers_only_and_exposes_only_selected_tensors(self):
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            self.assertEqual(set(reader.tensors), {NAME, SCALE})
            self.assertEqual(reader.stats()["tensor_read_bytes"], 0)
            self.assertEqual(reader.stats()["peak_open_shards"], 2)
            self.assertLessEqual(reader.stats()["max_actual_read_bytes"], 65536)
            self.assertEqual(len(reader.read_matrix_tile(NAME, 256, 256, 1, 3)), 3)
            with self.assertRaises(KeyError):
                reader.read_bytes("bf16.aux", 0, 2)
            with self.assertRaises(TypeError):
                reader.tensors[NAME] = None
        self.assertEqual(reader.stats()["open_shards"], 0)
        self.assertEqual(reader.stats()["tensor_read_bytes"], 3)

    def test_one_handle_lru_reopens_and_detects_replaced_evicted_shard(self):
        with SelectedCatalogueReader(self.directory, self.descriptor, max_open_shards=1) as reader:
            reader.read_bytes(NAME, 0, 1)
            reader.read_bytes(SCALE, 0, 4)
            reader.read_bytes(NAME, 0, 1)
            self.assertEqual(reader.stats()["peak_open_shards"], 1)
            self.assertGreaterEqual(reader.stats()["lru_evictions"], 3)
            path = self.directory / self.descriptor.scale.shard
            original = path.read_bytes()
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "changed"):
                reader.read_bytes(SCALE, 0, 4)

    def test_selected_file_size_and_exact_header_digest_required(self):
        path = self.directory / self.descriptor.weight.shard
        original = path.read_bytes()
        for modified, error in ((original[:-1], "file size"),
                                (original.replace(b'"format":"pt"', b'"format":"xx"', 1), "SHA-256")):
            path.write_bytes(modified)
            with self.assertRaisesRegex(ValueError, error):
                SelectedCatalogueReader(self.directory, self.descriptor)
        path.write_bytes(original)

    def test_unselected_files_are_not_required_and_bounds_unchanged(self):
        with SelectedCatalogueReader(self.directory, self.descriptor) as reader:
            for operation in (lambda: reader.read_bytes(NAME, 0, 65537),
                              lambda: reader.read_bytes(SCALE, 1, 4),
                              lambda: reader.read_matrix_tile(NAME, 0, 0, 129, 1),
                              lambda: reader.read_matrix_tile(NAME, 256, 258, 2, 1)):
                with self.assertRaises(ValueError):
                    operation()
        with self.assertRaisesRegex(ValueError, "closed"):
            reader.read_bytes(NAME, 0, 0)
        with self.assertRaises(ValueError):
            SelectedCatalogueReader(self.directory, self.descriptor, max_open_shards=3)

    def test_large_audit_index_never_loosens_legacy_reader_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, report, run = projection_fixture(root, rows=1, cols=1, extra_tensors=8193)
            self.assertGreater(report["coverage"]["total_index_tensors"], MAX_INDEX_TENSORS)
            self.assertGreater((run / "evidence/model.safetensors.index.json").stat().st_size, MAX_INDEX_BYTES)
            descriptor = build_projection_descriptor(root, settings(), NAME)
            for path in directory.glob("extra-*.safetensors"):
                path.unlink()
            with SelectedCatalogueReader(directory, descriptor) as reader:
                self.assertEqual(len(reader.read_bytes(NAME, 0, 1)), 1)
                self.assertEqual(reader.stats()["selected_shards"], 2)
                self.assertFalse(reader.stats()["legacy_reader_limits_changed"])
            self.assertEqual(MAX_INDEX_TENSORS, 8192)
            self.assertEqual(MAX_INDEX_BYTES, 1024**2)


if __name__ == "__main__":
    unittest.main()
