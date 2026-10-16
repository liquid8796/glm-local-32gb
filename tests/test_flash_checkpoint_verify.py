"""Offline-only audit of a pinned local Flash checkpoint (synthetic shards)."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from glm_local.flash_checkpoint_verify import FlashCheckpointError, verify_flash_checkpoint


def sha(data):
    return hashlib.sha256(data).hexdigest()


class FlashLocalCheckpointVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shard_name = "model-00001-of-00001.safetensors"
        self.tensor_name = "model.language_model.fake.weight"
        header = {self.tensor_name: {"dtype": "F32", "shape": [2],
                                     "data_offsets": [0, 8]}}
        raw_header = json.dumps(header, separators=(",", ":")).encode()
        shard = struct.pack("<Q", len(raw_header)) + raw_header + struct.pack("<ff", 1., -1.)
        self.write(self.shard_name, shard)
        idx = {"metadata": {"total_size": len(shard)},
               "weight_map": {self.tensor_name: self.shard_name}}
        self.write("model.safetensors.index.json", json.dumps(idx).encode())
        self.write("config.json", b'{"model_type":"glm5_next"}')
        self.write("tokenizer_config.json", b'{}')
        self.write("tokenizer.json", b'{"version":"1.0"}')
        self.snapshot = {
            "model_id": "example/flash-test", "revision": "fixed-revision",
            "weight_shards": 1, "weight_bytes": len(shard),
            "weights": [{"name": self.shard_name, "bytes": len(shard)}],
            "index_summary": {"tensor_count": 1, "weight_shards": 1,
                              "declared_total_size": len(shard)},
            "verification": {"source_sha256": {name: sha((self.root / name).read_bytes())
                             for name in ("config.json", "model.safetensors.index.json",
                                          "tokenizer_config.json")}},
            "metadata_files": [{"rfilename": "tokenizer.json",
                                "size": (self.root / "tokenizer.json").stat().st_size,
                                "lfs": {"sha256": sha((self.root / "tokenizer.json").read_bytes())}}]
        }

    def write(self, name, payload):
        (self.root / name).write_bytes(payload)

    def test_matching_headers_index_files_and_hashes_pass_without_hashing_weights(self):
        result = verify_flash_checkpoint(self.root, self.snapshot)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["shard_count"], 1)
        self.assertEqual(result["size_convention"], "complete_shard_files")
        self.assertFalse(result["weight_payload_hashes_verified"])
        self.assertEqual(result["model_id"], "example/flash-test")

    def test_required_pinned_hash_cannot_be_omitted(self):
        del self.snapshot["verification"]["source_sha256"]["config.json"]
        with self.assertRaisesRegex(FlashCheckpointError, "Missing pinned SHA-256"):
            verify_flash_checkpoint(self.root, self.snapshot)

    def test_tokenizer_digest_is_in_verified_receipt(self):
        self.assertIn("tokenizer.json",
                      verify_flash_checkpoint(self.root, self.snapshot)["metadata_hashes_verified"])

    def test_corrupted_model_config_fails_sha256(self):
        self.write("config.json", b'{"model_type":"wrong"}')
        with self.assertRaisesRegex(FlashCheckpointError, "SHA-256.*config.json"):
            verify_flash_checkpoint(self.root, self.snapshot)

    def test_shard_wrong_size_fails(self):
        with (self.root / self.shard_name).open("ab") as stream:
            stream.write(b"extra")
        with self.assertRaisesRegex(FlashCheckpointError, "shard size"):
            verify_flash_checkpoint(self.root, self.snapshot)

    def test_index_points_to_nonexistent_tensor_fails(self):
        idx_path = self.root / "model.safetensors.index.json"
        index = json.loads(idx_path.read_bytes())
        index["weight_map"] = {"missing.weight": self.shard_name}
        payload = json.dumps(index).encode()
        self.write("model.safetensors.index.json", payload)
        self.snapshot["verification"]["source_sha256"]["model.safetensors.index.json"] = sha(payload)
        with self.assertRaisesRegex(FlashCheckpointError, "tensor mapping"):
            verify_flash_checkpoint(self.root, self.snapshot)

    def test_bad_traversal_name_is_rejected(self):
        self.snapshot["weights"][0]["name"] = "../model-00001-of-00001.safetensors"
        with self.assertRaisesRegex(FlashCheckpointError, "shard name"):
            verify_flash_checkpoint(self.root, self.snapshot)

    def test_payload_bytes_not_mistaken_for_total_size(self):
        result = verify_flash_checkpoint(self.root, self.snapshot)
        self.assertEqual(result["total_file_bytes"], self.snapshot["weight_bytes"])
        self.assertLess(result["total_tensor_payload_bytes"], result["total_file_bytes"])
        self.assertEqual(result["total_tensor_payload_bytes"] + result["total_header_bytes"],
                         result["total_file_bytes"])


if __name__ == "__main__":
    unittest.main()
