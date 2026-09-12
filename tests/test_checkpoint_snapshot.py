from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local.checkpoint_http import FetchLimits, MetadataError
from glm_local.checkpoint_snapshot import EvidenceStore, OfflineMetadataSource, digest, write_json
from checkpoint_test_helpers import MODEL, REVISION, checkpoint, encode


class CheckpointSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence"
        self.data = checkpoint()
        self.store = EvidenceStore(self.path, MODEL, REVISION)
        for kind in ("model", "config", "index"):
            self.store.put_json(kind, encode(self.data[kind]))
        for name, (raw, size) in self.data["headers"].items():
            self.store.put_header(name, raw, size)

    def save_manifest(self, value):
        write_json(self.path / "snapshot.json", value)

    def test_replay_reads_only_metadata_and_never_initializes_http(self):
        with patch("urllib.request.build_opener", side_effect=AssertionError("No network")):
            source = OfflineMetadataSource(self.path, MODEL, REVISION)
            for kind in ("model", "config", "index"):
                self.assertEqual(json.loads(source.json_bytes(kind)), self.data[kind])
            for name, (raw, size) in self.data["headers"].items():
                self.assertEqual(source.header_bytes(name, size), raw)
        self.assertEqual(source.stats()["requests"], 0)
        self.assertFalse(source.stats()["remote_provenance_authenticated"])
        self.assertLessEqual(source.stats()["max_actual_read_bytes"], 65536)
        self.assertEqual(set(source.available_shards()), set(self.data["headers"]))

    def test_config_hash_corruption_detected(self):
        path = self.path / "config.json"
        path.write_bytes(path.read_bytes().replace(b"259", b"258"))
        source = OfflineMetadataSource(self.path, MODEL, REVISION)
        with self.assertRaisesRegex(MetadataError, "SHA-256"):
            source.json_bytes("config")

    def test_header_hash_corruption_detected(self):
        name = next(iter(self.data["headers"]))
        path = self.path / "headers" / (name + ".header")
        raw = path.read_bytes(); path.write_bytes(raw[:-1] + b" ")
        source = OfflineMetadataSource(self.path, MODEL, REVISION)
        with self.assertRaisesRegex(MetadataError, "SHA-256"):
            source.header_bytes(name, self.data["headers"][name][1])

    def test_wrong_revision_rejected_before_artifacts_read(self):
        with self.assertRaisesRegex(MetadataError, "revision"):
            OfflineMetadataSource(self.path, MODEL, "b" * 40)

    def test_snapshot_payload_smuggling_is_rejected_even_with_recomputed_hash(self):
        name = next(iter(self.data["headers"]))
        path = self.path / "headers" / (name + ".header")
        raw = path.read_bytes() + b"payload"
        path.write_bytes(raw)
        manifest = deepcopy(self.store.manifest)
        manifest["headers"][name].update(digest(raw))
        self.save_manifest(manifest)
        source = OfflineMetadataSource(self.path, MODEL, REVISION)
        with self.assertRaisesRegex(MetadataError, "unexpected tensor payload"):
            source.header_bytes(name, self.data["headers"][name][1])

    def test_storing_payload_or_bad_prefix_fails(self):
        name, (raw, size) = next(iter(self.data["headers"].items()))
        for bad in (raw + b"payload", raw[:-1], b"broken"):
            with self.subTest(bad_length=len(bad)), self.assertRaises(MetadataError):
                self.store.put_header(name, bad, size)

    def test_snapshot_shard_size_must_match_pinned_manifest(self):
        name, (_, size) = next(iter(self.data["headers"].items()))
        source = OfflineMetadataSource(self.path, MODEL, REVISION)
        with self.assertRaisesRegex(MetadataError, "file size"):
            source.header_bytes(name, size + 1)

    def test_unsafe_manifest_names_and_unknown_fields_rejected(self):
        manifest = deepcopy(self.store.manifest)
        manifest["headers"]["../escape.safetensors"] = {}
        self.save_manifest(manifest)
        with self.assertRaises(MetadataError):
            OfflineMetadataSource(self.path, MODEL, REVISION)
        manifest = deepcopy(self.store.manifest); manifest["download_url"] = "https://evil.invalid"
        self.save_manifest(manifest)
        with self.assertRaises(MetadataError):
            OfflineMetadataSource(self.path, MODEL, REVISION)

    def test_invalid_descriptor_hash_and_boolean_size_rejected(self):
        for value in ({"sha256": "x" * 64}, {"bytes": True}):
            manifest = deepcopy(self.store.manifest); manifest["objects"]["config"].update(value)
            self.save_manifest(manifest)
            source = OfflineMetadataSource(self.path, MODEL, REVISION)
            with self.assertRaises(MetadataError):
                source.json_bytes("config")

    def test_partial_snapshot_reports_missing_objects_and_available_headers(self):
        manifest = deepcopy(self.store.manifest)
        manifest["headers"].pop(next(iter(manifest["headers"])))
        manifest["objects"].pop("index")
        self.save_manifest(manifest)
        source = OfflineMetadataSource(self.path, MODEL, REVISION)
        self.assertEqual(len(source.available_shards()), 1)
        with self.assertRaisesRegex(MetadataError, "lacks required"):
            source.json_bytes("index")

    def test_symlink_artifact_or_headers_directory_rejected(self):
        target = Path(self.temp.name) / "config-copy.json"
        target.write_bytes((self.path / "config.json").read_bytes())
        (self.path / "config.json").unlink()
        try:
            (self.path / "config.json").symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("Local platform does not permit symlink creation")
        source = OfflineMetadataSource(self.path, MODEL, REVISION)
        with self.assertRaisesRegex(MetadataError, "regular files"):
            source.json_bytes("config")

    def test_offline_budget_enforced(self):
        with self.assertRaisesRegex(MetadataError, "budget"):
            OfflineMetadataSource(self.path, MODEL, REVISION, limits=FetchLimits(total_body_bytes=1))

    def test_existing_evidence_directory_never_silently_overwritten(self):
        with self.assertRaises(FileExistsError):
            EvidenceStore(self.path, MODEL, REVISION)
