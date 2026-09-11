"""Bounded installed-package provenance checks without importing Torch."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from glm_local import reference_env


class InstalledFixture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.version = reference_env.EXPECTED_VERSIONS["transformers"]
        self.metadata_name = "transformers-5.18.0.dev0.dist-info/direct_url.json"
        self.files = [PurePosixPath(self.metadata_name)]
        self.direct_url = {
            "url": reference_env.TRANSFORMERS_URL,
            "archive_info": {"hashes": {"sha256": reference_env.ARCHIVE_SHA256},
                             "hash": "sha256=" + reference_env.ARCHIVE_SHA256},
        }
        self.sources = {name: ("# invented test fixture " + name + "\n").encode("utf-8")
                        for name in reference_env.SOURCE_SHA256}
        self.hashes = {name: hashlib.sha256(contents).hexdigest()
                       for name, contents in self.sources.items()}
        self.write_metadata()
        for name, contents in self.sources.items():
            path = self.locate_file(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)

    def write_metadata(self):
        path = self.locate_file(self.metadata_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.direct_url), encoding="utf-8")

    def locate_file(self, name):
        return self.directory / str(name)

    def distribution(self, name):
        if name == "transformers":
            return self
        return type("PackageVersion", (), {"version": reference_env.EXPECTED_VERSIONS[name]})()


class ReferenceEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = InstalledFixture(self.temp.name)
        self.lookup = patch.object(reference_env.metadata, "distribution", self.fixture.distribution)
        self.lookup.start()
        self.addCleanup(self.lookup.stop)
        self.hashes = patch.object(reference_env, "SOURCE_SHA256", self.fixture.hashes)
        self.hashes.start()
        self.addCleanup(self.hashes.stop)

    def verify(self):
        return reference_env.verify_reference_environment()

    def assert_unsupported(self, reason):
        return self.assertRaisesRegex(reference_env.ReferenceEnvironmentError,
                                      "Unsupported or modified reference environment: .*" + reason)

    def test_valid_fixture_returns_json_safe_scoped_evidence(self):
        result = self.verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["packages"], reference_env.EXPECTED_VERSIONS)
        self.assertEqual(result["transformers"]["source_sha256"], self.fixture.hashes)
        self.assertIn("not the entire environment", result["verification_scope"])
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertNotIn(self.temp.name, json.dumps(result))

    def test_missing_package_and_wrong_version_fail_explicitly(self):
        with patch.object(reference_env.metadata, "distribution",
                          side_effect=metadata.PackageNotFoundError("torch")):
            with self.assert_unsupported("torch is not installed"):
                self.verify()
        self.fixture.version = "5.18.0"
        with self.assert_unsupported("exact pinned version"):
            self.verify()

    def test_modified_or_missing_source_fails(self):
        name = next(iter(self.fixture.sources))
        path = self.fixture.locate_file(name)
        path.write_bytes(b"# changed\n")
        with self.assert_unsupported("SHA-256 does not match"):
            self.verify()
        path.unlink()
        with self.assert_unsupported("missing or not a regular file"):
            self.verify()

    def test_archive_url_must_be_exact_and_cannot_use_equivalent_looking_identity(self):
        for url in (reference_env.TRANSFORMERS_URL + "?download=1",
                    reference_env.TRANSFORMERS_URL.replace("github.com", "example.com"),
                    reference_env.TRANSFORMERS_URL.replace(reference_env.TRANSFORMERS_REVISION, "main")):
            with self.subTest(url=url):
                self.fixture.direct_url["url"] = url
                self.fixture.write_metadata()
                with self.assert_unsupported("archive URL"):
                    self.verify()

    def test_archive_hash_missing_wrong_or_inconsistent_fails(self):
        for archive_info in ({}, {"hashes": {"sha256": "0" * 64}},
                             {"hashes": {"sha256": reference_env.ARCHIVE_SHA256},
                              "hash": "sha256=" + "0" * 64}, {"hashes": []}):
            with self.subTest(archive_info=archive_info):
                self.fixture.direct_url["archive_info"] = archive_info
                self.fixture.write_metadata()
                with self.assert_unsupported("SHA-256 metadata"):
                    self.verify()

    def test_legacy_and_modern_pip_archive_hash_formats_are_supported(self):
        original = deepcopy(self.fixture.direct_url["archive_info"])
        for key in ("hash", "hashes"):
            self.fixture.direct_url["archive_info"] = {key: original[key]}
            self.fixture.write_metadata()
            self.assertEqual(self.verify()["status"], "verified")

    def test_metadata_missing_ambiguous_invalid_json_and_nonobject_fail(self):
        original_files = list(self.fixture.files)
        for entries in (None, [], original_files * 2):
            self.fixture.files = entries
            with self.assert_unsupported("metadata"):
                self.verify()
        self.fixture.files = original_files
        path = self.fixture.locate_file(self.fixture.metadata_name)
        for content in (b"not json", b"\xff", b"[]"):
            path.write_bytes(content)
            with self.assert_unsupported("JSON|archive URL"):
                self.verify()

    def test_oversized_sources_and_metadata_fail_before_hash_or_json(self):
        path = self.fixture.locate_file(self.fixture.metadata_name)
        path.write_bytes(b" " * (reference_env.METADATA_MAX_BYTES + 1))
        with self.assert_unsupported("size limit"):
            self.verify()
        self.fixture.write_metadata()
        source = self.fixture.locate_file(next(iter(self.fixture.sources)))
        with source.open("wb") as stream:
            stream.truncate(reference_env.SOURCE_MAX_BYTES + 1)
        with self.assert_unsupported("size limit"):
            self.verify()

    def test_cli_success_and_failure_are_json_with_exit_zero_or_one(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(reference_env.main(["--verify"]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "verified")
        self.fixture.version = "wrong"
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(reference_env.main(["--verify"]), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "unsupported_or_modified_environment")


class ReferenceLockTests(unittest.TestCase):
    def test_install_lock_agrees_with_verifier_provenance_contract(self):
        lock_path = Path(__file__).resolve().parents[1] / "config" / "reference-lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock["packages"], reference_env.EXPECTED_VERSIONS)
        self.assertEqual(lock["transformers"]["url"], reference_env.TRANSFORMERS_URL)
        self.assertEqual(lock["transformers"]["revision"], reference_env.TRANSFORMERS_REVISION)
        self.assertEqual(lock["transformers"]["archive_sha256"], reference_env.ARCHIVE_SHA256)
        self.assertEqual(lock["transformers"]["source_sha256"], reference_env.SOURCE_SHA256)
        self.assertEqual(lock["verification"]["source_max_bytes"], reference_env.SOURCE_MAX_BYTES)
        self.assertEqual(lock["verification"]["direct_url_max_bytes"], reference_env.METADATA_MAX_BYTES)


if __name__ == "__main__":
    unittest.main()
