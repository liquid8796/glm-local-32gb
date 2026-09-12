"""Pinned tokenizer file provenance and native JSON encoding; no model downloads."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glm_local import tokenizer
from glm_local.checkpoint_http import HttpMetadataSource
from glm_local.checkpoint_snapshot import digest, write_json
from checkpoint_test_helpers import MODEL, REVISION, Opener, Response, settings
from test_execution import projection_fixture


def mini_tokenizer():
    single = [{"SpecialToken": {"id": "<s>", "type_id": 0}},
              {"Sequence": {"id": "A", "type_id": 0}},
              {"SpecialToken": {"id": "</s>", "type_id": 0}}]
    document = {"version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": index, "content": text, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": True}
                         for index, text in ((3, "<s>"), (4, "</s>"))],
        "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {"type": "TemplateProcessing", "single": single,
            "pair": single[:-1] + [{"Sequence": {"id": "B", "type_id": 0}}, single[-1]],
            "special_tokens": {text: {"id": text, "ids": [index], "tokens": [text]}
                               for index, text in ((3, "<s>"), (4, "</s>"))}},
        "decoder": None, "model": {"type": "WordLevel", "unk_token": "[UNK]",
                                    "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "<s>": 3, "</s>": 4}}}
    return json.dumps(document, separators=(",", ":")).encode()


def manifest_for(files):
    return {"id": MODEL, "sha": REVISION, "siblings": [
        {"rfilename": "tokenizer.json", "size": len(files["tokenizer.json"]),
         "lfs": {"size": len(files["tokenizer.json"]), "sha256": hashlib.sha256(files["tokenizer.json"]).hexdigest()}},
        {"rfilename": "tokenizer_config.json", "size": len(files["tokenizer_config.json"]),
         "blobId": hashlib.sha1(b"blob " + str(len(files["tokenizer_config.json"])).encode() + b"\0" + files["tokenizer_config.json"]).hexdigest()}]}


class TokenizerFixture:
    def initialize(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory, _, self.run = projection_fixture(self.root, rows=1, cols=1)
        self.files = {"tokenizer.json": mini_tokenizer(), "tokenizer_config.json": json.dumps({
            "backend": "tokenizers", "tokenizer_class": "TokenizersBackend", "model_max_length": 128,
            "eos_token": "</s>", "pad_token": "</s>", "extra_special_tokens": ["<s>", "</s>"]}).encode()}
        self.config = {"vocab_size": 8, "max_position_embeddings": 128,
                       "bos_token_id": 3, "eos_token_id": [4], "pad_token_id": 4}
        self.install_manifest()

    def install_manifest(self, modify=None):
        self.manifest = manifest_for(self.files)
        if modify:
            modify(self.manifest)
        path = self.run / "evidence/model.json"
        model = json.loads(path.read_bytes())
        model["siblings"] = [item for item in model["siblings"] if item["rfilename"] not in tokenizer.ARTIFACTS]
        model["siblings"].extend(self.manifest["siblings"])
        raw = json.dumps(model).encode()
        path.write_bytes(raw)
        snapshot_path = self.run / "evidence/snapshot.json"
        snapshot = json.loads(snapshot_path.read_bytes())
        snapshot["objects"]["model"] = digest(raw)
        write_json(snapshot_path, snapshot)

    def write_files(self):
        for name, raw in self.files.items():
            (self.directory / name).write_bytes(raw)

    def transport(self, change=None):
        responses = []
        for name in tokenizer.ARTIFACTS:
            raw = self.files[name]
            response = Response(raw, headers={"Content-Length": len(raw), "X-Repo-Commit": REVISION})
            if change:
                change(name, response)
            responses.append(response)
        opener = Opener(responses)
        return HttpMetadataSource(MODEL, REVISION, opener=opener), opener, responses

    def load(self, config=None):
        return tokenizer.load_tokenizer(self.directory, config or self.config,
                                        model_id=MODEL, revision=REVISION, manifest=self.manifest)


class TokenizerArtifactTests(TokenizerFixture, unittest.TestCase):
    def setUp(self):
        self.initialize()

    def test_offline_verification_uses_captured_model_hashes_without_network_or_native_code(self):
        self.write_files()
        with patch.object(tokenizer, "HttpMetadataSource") as network, patch.object(tokenizer, "_native_tokenizers") as native:
            report = tokenizer.prepare_tokenizer(self.root, settings(), self.directory)
        network.assert_not_called()
        native.assert_not_called()
        self.assertTrue(report["tokenizer_verified"])
        self.assertFalse(report["tokenizer_native_loaded"])
        self.assertFalse(report["remote_code_executed"])
        self.assertEqual(report["tensor_payload_bytes_requested"], 0)
        self.assertEqual(report["manifest"], self.manifest)
        self.assertEqual([item["manifest_hash_algorithm"] for item in report["tokenizer_files"]], ["sha256", "git_blob_sha1"])
        self.assertTrue(all(report[key] is False for key in tokenizer.FULL_MODEL_FLAGS))

    def test_missing_offline_files_never_create_or_download(self):
        directory = self.root / "missing"
        with patch.object(tokenizer, "HttpMetadataSource") as network, self.assertRaisesRegex(FileNotFoundError, "tokenizer-check --online"):
            tokenizer.prepare_tokenizer(self.root, settings(), directory)
        self.assertFalse(directory.exists())
        network.assert_not_called()

    def test_online_downloads_exactly_two_json_files_and_publishes_verified_bytes(self):
        # Padding metadata ensures actual HTTP reads cross the 64-KiB boundary.
        self.files["tokenizer.json"] = self.files["tokenizer.json"] + b" " * 70000
        self.install_manifest()
        source, opener, responses = self.transport()
        with patch.object(tokenizer, "HttpMetadataSource", return_value=source):
            report = tokenizer.prepare_tokenizer(self.root, settings(), self.directory, online=True)
        self.assertTrue(all(item["downloaded"] for item in report["tokenizer_files"]))
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual([request.full_url for request, _ in opener.requests],
            [f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}" for name in tokenizer.ARTIFACTS])
        for name, raw in self.files.items():
            self.assertEqual((self.directory/name).read_bytes(), raw)
        self.assertLessEqual(max(size for response in responses for size in response.requested_reads), 65536)
        self.assertEqual(report["network"]["body_bytes_read"], sum(map(len, self.files.values())))
        self.assertEqual(list(self.directory.glob("*.part-*")), [])

    def test_existing_wrong_file_is_never_overwritten_or_downloaded(self):
        path = self.directory / "tokenizer_config.json"
        original = b"x" * len(self.files[path.name])
        path.write_bytes(original)
        with patch.object(tokenizer, "HttpMetadataSource") as network, self.assertRaisesRegex(ValueError, "git_blob_sha1"):
            tokenizer.prepare_tokenizer(self.root, settings(), self.directory, online=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.directory / "tokenizer.json").exists())
        network.assert_not_called()

    def test_same_size_hash_mismatch_and_truncation_leave_no_completed_file(self):
        original = self.files["tokenizer.json"]
        for value, pattern in ((b"x" * len(original), "sha256"), (original[:-1], "truncated")):
            def change(name, response):
                if name == "tokenizer.json":
                    response.seek(0)
                    response.truncate(0)
                    response.write(value)
                    response.seek(0)
            source, _, _ = self.transport(change)
            with self.subTest(pattern=pattern), patch.object(tokenizer, "HttpMetadataSource", return_value=source), self.assertRaisesRegex(ValueError, pattern):
                tokenizer.prepare_tokenizer(self.root, settings(), self.directory, online=True)
            self.assertFalse((self.directory / "tokenizer.json").exists())
            self.assertEqual(list(self.directory.glob("*.part-*")), [])

    def test_bad_http_status_lengths_encoding_or_revision_fail_before_body_read(self):
        def status(name, response):
            response.status = 206
        def length(name, response):
            response.headers.replace_header("Content-Length", "1")
        def missing_length(name, response):
            del response.headers["Content-Length"]
        def encoding(name, response):
            response.headers["Content-Encoding"] = "gzip"
        def revision(name, response):
            response.headers.replace_header("X-Repo-Commit", "b"*40)
        for change in (status, length, missing_length, encoding, revision):
            source, _, responses = self.transport(change)
            with self.subTest(change=change.__name__), patch.object(tokenizer, "HttpMetadataSource", return_value=source), self.assertRaises(ValueError):
                tokenizer.prepare_tokenizer(self.root, settings(), self.directory, online=True)
            self.assertEqual(responses[0].requested_reads, [])
            self.assertTrue(responses[0].closed)

    def test_git_blob_sha1_includes_git_object_header(self):
        self.write_files()
        self.install_manifest(lambda m: m["siblings"][1].update(blobId=hashlib.sha1(self.files["tokenizer_config.json"]).hexdigest()))
        with self.assertRaisesRegex(ValueError, "git_blob_sha1"):
            tokenizer.prepare_tokenizer(self.root, settings(), self.directory)

    def test_manifest_identity_hashes_sizes_duplicates_and_combined_budget_fail_closed(self):
        changes = [lambda m: m.update(sha="b"*40), lambda m: m["siblings"][0].update(size=True),
                   lambda m: m["siblings"][0]["lfs"].update(size=123),
                   lambda m: m["siblings"][0]["lfs"].update(sha256="x"*64),
                   lambda m: m["siblings"].append(m["siblings"][0]),
                   lambda m: m["siblings"].pop(),
                   lambda m: m["siblings"][0].update(size=32*1024**2+1)]
        for change in changes:
            manifest = deepcopy(self.manifest)
            change(manifest)
            with self.assertRaises(ValueError):
                tokenizer._artifacts(manifest, MODEL, REVISION)
        manifest = deepcopy(self.manifest)
        manifest["siblings"][0]["size"] = 32*1024**2
        manifest["siblings"][0]["lfs"]["size"] = 32*1024**2
        with self.assertRaisesRegex(ValueError, "Combined"):
            tokenizer._artifacts(manifest, MODEL, REVISION)

    def test_captured_manifest_corruption_cannot_be_replaced_by_docs_guess(self):
        self.write_files()
        (self.run / "evidence/model.json").write_bytes(b"{}")
        write_json(self.root / "docs/tokenizer-manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            tokenizer.prepare_tokenizer(self.root, settings(), self.directory)

    def test_remote_code_config_and_unsupported_backend_are_rejected_without_import(self):
        for change in ({"auto_map": {"AutoTokenizer": "repository.Tokenizer"}}, {"backend": "python"},
                       {"tokenizer_class": "RepositoryTokenizer"}, {"trust_remote_code": True}):
            config = {"backend": "tokenizers", **change}
            self.files["tokenizer_config.json"] = json.dumps(config).encode()
            self.install_manifest()
            self.write_files()
            with patch.object(tokenizer, "_native_tokenizers") as native, self.assertRaises(ValueError):
                tokenizer.prepare_tokenizer(self.root, settings(), self.directory)
            native.assert_not_called()

    def test_missing_or_wrong_native_package_has_actionable_reference_environment_error(self):
        self.write_files()
        import builtins
        original = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == "tokenizers":
                raise ImportError("absent")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=missing), self.assertRaisesRegex(RuntimeError, "venv-reference"):
            self.load()
        with patch.dict("sys.modules", {"tokenizers": SimpleNamespace(__version__="0.0")}), self.assertRaisesRegex(RuntimeError, "0.23.2"):
            self.load()


@unittest.skipUnless(importlib.util.find_spec("tokenizers"), "Native tokenizer package is optional; run in .venv-reference")
class NativeTokenizerTests(TokenizerFixture, unittest.TestCase):
    def setUp(self):
        self.initialize()
        self.write_files()

    def test_native_postprocessor_and_round_trip_are_used_without_transformers(self):
        with patch.dict("sys.modules", {"transformers": None}):
            native = self.load()
            self.assertEqual(native.encode("hello world"), [3, 1, 2, 4])
            self.assertEqual(native.encode("hello world", add_special_tokens=False), [1, 2])
            self.assertEqual(native.decode([3, 1, 2, 4]), "hello world")
            self.assertEqual(native.decode([3, 1, 2, 4], skip_special_tokens=False), "<s> hello world </s>")

    def test_text_token_ids_options_and_vocabulary_limits_are_validated(self):
        native = self.load()
        for ids in ([True], [-1], [8], [5], [1]*129, "1", iter([1])):
            with self.assertRaises(ValueError):
                native.decode(ids)
        for text in (b"hello", None, "x"*(1024**2+1), "\u20ac"*(1024**2//3+1)):
            with self.assertRaises(ValueError):
                native.encode(text)
        with self.assertRaises(ValueError):
            native.encode("hello", add_special_tokens=1)
        with self.assertRaises(ValueError):
            native.decode([1], skip_special_tokens="yes")
        for config in ({**self.config, "vocab_size": 4}, {**self.config, "vocab_size": True},
                       {**self.config, "vocab_size": tokenizer.MAX_VOCAB_SIZE+1},
                       {**self.config, "eos_token_id": [7]}):
            with self.assertRaises(ValueError):
                self.load(config)

    def test_native_loader_rehashes_files_and_rejects_hash_correct_unsupported_decoder(self):
        path = self.directory / "tokenizer.json"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "file size"):
            self.load()
        document = json.loads(self.files["tokenizer.json"])
        document["decoder"] = {"type": "Replace", "pattern": {"String": "a"}, "content": "x"*1000}
        self.files["tokenizer.json"] = json.dumps(document).encode()
        self.install_manifest()
        self.write_files()
        with self.assertRaisesRegex(ValueError, "bounded native profile"):
            self.load()

    def test_decode_size_is_preflighted_before_calling_native_library(self):
        native = self.load()
        native._token_lengths[1] = 16*1024**2
        with self.assertRaisesRegex(ValueError, "would exceed"):
            native.decode([1])


if __name__ == "__main__":
    unittest.main()
