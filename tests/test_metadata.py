"""Validate metadata using only small in-memory HTTP responses."""

from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch

from glm_local import metadata


MODEL = "example/local-checkpoint"
REVISION = "a" * 40


def fixture_metadata():
    return {
        "id": MODEL, "sha": REVISION,
        "siblings": [
            {"rfilename": "config.json", "size": 1234},
            {"rfilename": "weights/part-01.safetensors", "size": 11},
            {"rfilename": "weights/part-02.safetensors", "size": 17},
        ],
    }


class MetadataValidationTests(unittest.TestCase):
    def summarize(self, data=None, config=None):
        return metadata.summarize_metadata(
            MODEL, REVISION, fixture_metadata() if data is None else data,
            {"model_type": "glm_moe_dsa"} if config is None else config)

    def test_only_checkpoint_shards_contribute_to_weight_bytes(self):
        result = self.summarize()
        self.assertEqual(result["weight_bytes"], 28)
        self.assertEqual(result["weight_shards"], 2)
        self.assertEqual(result["weights"], [
            {"name": "weights/part-01.safetensors", "bytes": 11},
            {"name": "weights/part-02.safetensors", "bytes": 17},
        ])
        self.assertEqual(result["architecture"]["model_type"], "glm_moe_dsa")

    def test_model_and_revision_must_both_match(self):
        for key, value in (("id", "other/model"), ("sha", "b" * 40),
                           ("id", None), ("sha", None)):
            data = fixture_metadata()
            data[key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                self.summarize(data)

    def test_missing_or_invalid_shard_sizes_fail(self):
        for value in (None, 0, -1, 3.5, "11", True, float("nan"), float("inf")):
            data = fixture_metadata()
            data["siblings"][1]["size"] = value
            with self.subTest(size=value), self.assertRaises(ValueError):
                self.summarize(data)
        data = fixture_metadata()
        del data["siblings"][1]["size"]
        with self.assertRaises(ValueError):
            self.summarize(data)

    def test_no_shards_and_malformed_collections_fail_explicitly(self):
        for siblings in (None, {}, "shards", [], [{"rfilename": "README.md"}],
                         [None], ["part.safetensors"], [{"rfilename": None}]):
            data = fixture_metadata()
            data["siblings"] = siblings
            with self.subTest(siblings=siblings), self.assertRaises(ValueError):
                self.summarize(data)
        for data in (None, [], "metadata", 1):
            with self.subTest(metadata=data), self.assertRaises(ValueError):
                metadata.summarize_metadata(MODEL, REVISION, data, {"model_type": "glm_moe_dsa"})

    def test_missing_architecture_and_malformed_quantization_fail(self):
        for config in ({}, [], {"model_type": ""},
                       {"model_type": "glm_moe_dsa", "quantization_config": None},
                       {"model_type": "glm_moe_dsa", "quantization_config": []}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.summarize(config=config)

    def test_duplicate_shards_include_windows_case_aliases(self):
        for duplicate in ("weights/part-01.safetensors", "WEIGHTS/PART-01.safetensors"):
            data = fixture_metadata()
            data["siblings"].append({"rfilename": duplicate, "size": 11})
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                self.summarize(data)

    def test_unsafe_shard_paths_fail_before_summary_is_created(self):
        names = ("../part.safetensors", "/part.safetensors", "weights/../../part.safetensors",
                 "weights/./part.safetensors", "weights//part.safetensors",
                 "C:/part.safetensors", "weights\\part.safetensors", "part\0.safetensors",
                 "CON.safetensors", "weights. /part.safetensors")
        for name in names:
            data = fixture_metadata()
            data["siblings"][1]["rfilename"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.summarize(data)

    def test_safe_path_preserves_nested_unicode_filename(self):
        path = metadata.safe_relative_path("trọng số/part-01.safetensors")
        self.assertEqual(path.as_posix(), "trọng số/part-01.safetensors")


class MetadataNetworkBoundaryTests(unittest.TestCase):
    def test_json_response_is_read_with_a_finite_bound(self):
        payload = {"status": "metadata only"}
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(payload).encode("utf-8")
        with patch.object(metadata.urllib.request, "urlopen", return_value=response) as urlopen:
            self.assertEqual(metadata.fetch_json("https://example.invalid/metadata.json"), payload)
        response.read.assert_called_once_with(metadata.MAX_METADATA_BYTES + 1)
        self.assertGreater(urlopen.call_args.kwargs["timeout"], 0)

    def test_oversized_response_fails_without_large_allocation(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b"x" * 17
        with patch.object(metadata, "MAX_METADATA_BYTES", 16), \
                patch.object(metadata.urllib.request, "urlopen", return_value=response), \
                self.assertRaisesRegex(ValueError, "Metadata exceeds"):
            metadata.fetch_json("https://example.invalid/metadata.json")
        response.read.assert_called_once_with(17)

    def test_invalid_identifiers_and_unpinned_revisions_never_make_requests(self):
        with patch.object(metadata, "fetch_json") as fetch:
            for model in ("model", "owner/model/extra", "https://example.invalid/model",
                          "../model", "./model", "owner/.."):
                with self.subTest(model=model), self.assertRaises(ValueError):
                    metadata.refresh_metadata(model, REVISION)
            for revision in ("main", "a" * 39, "A" * 40, "../config.json", ""):
                with self.subTest(revision=revision), self.assertRaises(ValueError):
                    metadata.refresh_metadata(MODEL, revision)
        fetch.assert_not_called()

    def test_refresh_fetches_only_pinned_model_metadata_and_config(self):
        with patch.object(metadata, "fetch_json", side_effect=[
                deepcopy(fixture_metadata()), {"model_type": "glm_moe_dsa"}]) as fetch:
            result = metadata.refresh_metadata(MODEL, REVISION)
        self.assertEqual(result["revision"], REVISION)
        urls = [call.args[0] for call in fetch.call_args_list]
        self.assertEqual(urls, [
            f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true",
            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/config.json",
        ])


if __name__ == "__main__":
    unittest.main()
