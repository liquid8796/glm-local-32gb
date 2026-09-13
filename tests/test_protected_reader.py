"""Protected-handle lifetime and portable polling over small synthetic files."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local import runtime_weights as runtime
from glm_local import safetensor_reader as storage
from glm_local.catalogue_reader import SelectedCatalogueReader
from glm_local.execution import build_projection_descriptor
from glm_local.runtime_io import RowBandCacheReader
from glm_local.safetensor_reader import SafeTensorError, SafeTensorReader
from checkpoint_test_helpers import NAME, SCALE, settings
from test_execution import projection_fixture
from test_runtime_weights import prepare_runtime_fixture
from test_safetensor_reader import encode, entry


class ProtectedReaderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="glm-protected-tests-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def fixture(self):
        path = self.root / "weights.safetensors"
        payload = bytes(range(256)) * 1024
        path.write_bytes(encode({"w": entry("U8", [512, 512], [0, len(payload)])}, payload))
        return path, payload

    @unittest.skipUnless(os.name == "nt", "Windows protected read handles")
    def test_live_protected_span_and_scalar_reads_do_not_poll_filesystem(self):
        path, payload = self.fixture()
        with SafeTensorReader(path) as reader:
            self.assertFalse(reader.protected_immutable)
            reader.read_span("w", 0, 1)
            self.assertTrue(reader.protected_immutable)
            before = reader.stats()["identity_checks"]
            with patch.object(storage.os, "fstat", side_effect=AssertionError("unexpected fstat")), \
                    patch.object(Path, "stat", side_effect=AssertionError("unexpected stat")):
                self.assertEqual(reader.read_span("w", 3, 150_000), payload[3:150_003])
                self.assertEqual(reader.read_bytes("w", 17, 4), payload[17:21])
                self.assertEqual(reader.read_matrix_tile("w", 0, 3, 2, 4), payload[3:7] + payload[515:519])
            self.assertEqual(reader.stats()["identity_checks"], before)
            self.assertGreater(reader.stats()["protected_immutable_checks"], 0)
        self.assertFalse(reader.protected_immutable)

    def test_unprotected_span_and_scalar_reads_keep_original_polling(self):
        path, _ = self.fixture()
        with SafeTensorReader(path) as reader:
            with patch.object(reader, "_protect_span_handle", return_value=None), \
                    patch.object(storage.os, "fstat", wraps=os.fstat) as fstat:
                reader.read_span("w", 0, 150_000)
                self.assertEqual(fstat.call_count, 2)
                reader.read_bytes("w", 0, 4)
                self.assertEqual(fstat.call_count, 5)
            self.assertFalse(reader.protected_immutable)
            self.assertEqual(reader.stats()["protected_immutable_checks"], 0)

    @unittest.skipUnless(os.name == "nt", "Windows protected read handles")
    def test_externally_closed_protected_handle_cannot_return_cached_or_new_data(self):
        path, _ = self.fixture()
        with SafeTensorReader(path) as reader:
            reader.read_span("w", 0, 1)
            reader._stream.close()
            self.assertFalse(reader.protected_immutable)
            with patch.object(storage.os, "fstat", side_effect=AssertionError("closed handle must fail first")):
                with self.assertRaisesRegex(SafeTensorError, "closed"):
                    reader.read_bytes("w", 0, 1)
                with self.assertRaisesRegex(SafeTensorError, "closed"):
                    reader.read_span("w", 0, 1)

    @unittest.skipUnless(os.name == "nt", "Windows protected read handles")
    def test_upgrade_checks_exact_header_even_if_fingerprints_appear_unchanged(self):
        path, _ = self.fixture()
        original = path.read_bytes()
        # Same length and parsed descriptor, different raw JSON order.
        changed = original.replace(b'"dtype":"U8","shape":[512,512]', b'"shape":[512,512],"dtype":"U8"', 1)
        self.assertNotEqual(changed, original)
        opened = []
        real_open = storage._open_protected_read
        def altered_open(target):
            path.write_bytes(changed)
            stream = real_open(target)
            opened.append(stream)
            return stream
        reader = SafeTensorReader(path)
        # Isolate the header check from timestamp/identity detection deliberately.
        with patch.object(reader, "_assert_unchanged", return_value=None), \
                patch.object(storage, "_open_protected_read", side_effect=altered_open):
            with self.assertRaisesRegex(SafeTensorError, "header changed"):
                reader.read_span("w", 0, 1)
        self.assertFalse(reader.protected_immutable)
        self.assertIsNone(reader._stream)
        self.assertTrue(all(stream.closed for stream in opened))
        self.assertEqual(reader.stats()["tensor_read_bytes"], 0)

    @unittest.skipUnless(os.name == "nt", "Windows protected read handles")
    def test_selected_protected_shard_reuse_skips_catalogue_polling(self):
        directory, _, _ = projection_fixture(self.root)
        descriptor = build_projection_descriptor(self.root, settings(), NAME)
        with SelectedCatalogueReader(directory, descriptor) as reader:
            reader.read_span(NAME, 0, 1)
            with patch.object(reader, "_identity", side_effect=AssertionError("unexpected shard lstat")):
                self.assertEqual(len(reader.read_span(NAME, 0, 5)), 5)
                self.assertEqual(len(reader.read_bytes(NAME, 0, 1)), 1)

    @unittest.skipUnless(os.name == "nt", "Windows protected read handles")
    def test_cached_row_band_revalidates_identity_after_its_shard_was_evicted(self):
        directory, _, _ = projection_fixture(self.root)
        descriptor = build_projection_descriptor(self.root, settings(), NAME)
        class CacheableSelected(SelectedCatalogueReader):
            def assert_tensor_unchanged(self, name):
                self._reader(self._selected[name].shard)._assert_unchanged()
        source = CacheableSelected(directory, descriptor, max_open_shards=1)
        with RowBandCacheReader(source) as reader:
            reader.read_matrix_tile(NAME, 0, 0, 128, 128)
            reader.read_span(SCALE, 0, 4)  # Evicts and unlocks the weight shard.
            path = directory / descriptor.weight.shard
            original = path.read_bytes()
            path.rename(directory / "old.safetensors")
            path.write_bytes(original)
            with self.assertRaisesRegex(SafeTensorError, "changed since validation"):
                reader.read_matrix_tile(NAME, 0, 128, 128, 128)
            self.assertEqual(reader.stats()["row_band_cache"]["retained_bytes"], 0)


class ProtectedMetadataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="glm-metadata-lock-tests-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directory, _ = prepare_runtime_fixture(self.root)

    @unittest.skipUnless(os.name == "nt", "Windows metadata read locks")
    def test_config_and_index_are_write_delete_and_rename_locked_until_close(self):
        paths = [self.directory / name for name in ("config.json", "model.safetensors.index.json")]
        before = [path.read_bytes() for path in paths]
        with runtime.FullCatalogueReader(self.root, settings()) as reader:
            self.assertEqual(reader.stats()["protected_metadata_handles"], 2)
            self.assertLessEqual(reader.stats()["open_shards"], 2)
            for path in paths:
                with self.assertRaises(PermissionError), path.open("r+b"):
                    pass
                with self.assertRaises(PermissionError):
                    path.unlink()
                with self.assertRaises(PermissionError):
                    path.rename(path.with_suffix(".renamed"))
        self.assertEqual(reader.stats()["protected_metadata_handles"], 0)
        for path, original in zip(paths, before):
            with path.open("r+b") as writer:
                writer.seek(0)
            self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "nt", "Windows metadata read locks")
    def test_full_reader_span_scalar_and_cache_hits_need_no_filesystem_polls(self):
        name = "lm_head.weight"
        with RowBandCacheReader(runtime.FullCatalogueReader(self.root, settings())) as cache:
            cache.read_matrix_tile(name, 0, 0, 128, 8)
            source = cache._source
            before = source.stats()["identity_checks"]
            with patch.object(runtime, "_identity", side_effect=AssertionError("metadata lstat")), \
                    patch.object(source, "_identity", side_effect=AssertionError("shard lstat")), \
                    patch.object(storage.os, "fstat", side_effect=AssertionError("shard fstat")):
                for _ in range(8):
                    self.assertEqual(len(cache.read_span(name, 0, 32)), 32)
                    self.assertEqual(len(cache.read_bytes(name, 0, 2)), 2)
                    self.assertEqual(len(cache.read_matrix_tile(name, 0, 8, 128, 8)), 2048)
            self.assertEqual(source.stats()["identity_checks"], before)
            self.assertEqual(source.stats()["metadata_identity_checks"], 0)
            self.assertGreater(source.stats()["metadata_protected_checks"], 0)

    def test_portable_metadata_fallback_still_rejects_post_validation_changes(self):
        with patch.object(runtime, "_seal_local_document", return_value=None), \
                runtime.FullCatalogueReader(self.root, settings()) as reader:
            self.assertEqual(reader.stats()["protected_metadata_handles"], 0)
            with (self.directory / "config.json").open("ab") as writer:
                writer.write(b" ")
            with self.assertRaisesRegex(SafeTensorError, "changed since validation"):
                reader.read_bytes("lm_head.weight", 0, 2)

    @unittest.skipUnless(os.name == "nt", "Windows metadata read locks")
    def test_closed_metadata_lock_fails_and_releases_all_other_handles(self):
        reader = runtime.FullCatalogueReader(self.root, settings())
        next(iter(reader._document_handles.values())).close()
        with self.assertRaisesRegex(SafeTensorError, "handle is closed"):
            reader.read_bytes("lm_head.weight", 0, 2)
        self.assertEqual(reader.stats()["open_shards"], 0)
        self.assertEqual(reader.stats()["protected_metadata_handles"], 0)
        reader.close()
        with (self.directory / "config.json").open("r+b"):
            pass

    @unittest.skipUnless(os.name == "nt", "Windows metadata read locks")
    def test_second_document_failure_releases_first_document_lock(self):
        index = self.directory / "model.safetensors.index.json"
        index.write_bytes(index.read_bytes() + b" ")
        opened = []
        real_open = runtime._open_protected_read
        def track(path):
            stream = real_open(path)
            opened.append(stream)
            return stream
        with patch.object(runtime, "_open_protected_read", side_effect=track):
            with self.assertRaisesRegex(SafeTensorError, "length differs"):
                runtime.FullCatalogueReader(self.root, settings())
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)
        with (self.directory / "config.json").open("r+b"):
            pass

    @unittest.skipUnless(os.name == "nt", "Windows metadata read locks")
    def test_existing_index_writer_rejects_seal_and_releases_config(self):
        with (self.directory / "model.safetensors.index.json").open("r+b"):
            with self.assertRaisesRegex(SafeTensorError, "protected config/index"):
                runtime.FullCatalogueReader(self.root, settings())
        with (self.directory / "config.json").open("r+b"):
            pass

    @unittest.skipUnless(os.name == "nt", "Windows metadata read locks")
    def test_same_bytes_replacement_between_validation_and_seal_is_rejected(self):
        config = self.directory / "config.json"
        original = config.read_bytes()
        opened = []
        real_open = runtime._open_protected_read
        def replace(path):
            if path == config:
                config.rename(self.directory / "old-config.json")
                config.write_bytes(original)
            stream = real_open(path)
            opened.append(stream)
            return stream
        with patch.object(runtime, "_open_protected_read", side_effect=replace):
            with self.assertRaisesRegex(SafeTensorError, "changed while acquiring"):
                runtime.FullCatalogueReader(self.root, settings())
        self.assertTrue(all(stream.closed for stream in opened))
        with config.open("r+b"):
            pass


if __name__ == "__main__":
    unittest.main()
