"""Resume partial metadata with tiny cached headers and strict fake HTTP ranges."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local import __main__ as cli, checkpoint_check as check
from glm_local.checkpoint_http import FetchLimits, MetadataError
from glm_local.checkpoint_resume import ResumingMetadataSource
from glm_local.checkpoint_snapshot import EvidenceStore, digest, write_json
from glm_local.winjob import InstalledLimits
from checkpoint_test_helpers import MODEL, REVISION, Opener, checkpoint, encode, ranged, settings


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cached/evidence"
        self.data = checkpoint()
        self.names = sorted(self.data["headers"])
        self.store = EvidenceStore(self.cache, MODEL, REVISION)
        for kind in ("model", "config", "index"):
            self.store.put_json(kind, encode(self.data[kind]))
        raw, size = self.data["headers"][self.names[0]]
        self.store.put_header(self.names[0], raw, size)
        write_json(self.root / "docs/model-metadata.json", self.data["expected"])

    def transport(self, name=None):
        raw, size = self.data["headers"][name or self.names[1]]
        return Opener([ranged(raw[:8], 0, size), ranged(raw[8:], 8, size)])

    def cached_files(self):
        return {str(path.relative_to(self.cache)): path.read_bytes()
                for path in self.cache.rglob("*") if path.is_file()}

    def test_only_missing_headers_are_fetched_with_one_shared_budget(self):
        before = self.cached_files()
        opener = self.transport()
        source = ResumingMetadataSource(self.cache, MODEL, REVISION, opener=opener)
        self.assertEqual(opener.requests, [])
        cached_bytes = source.stats()["cached_body_bytes_read"]
        self.assertEqual(cached_bytes, sum(map(len, before.values())))
        self.assertIsNone(source.available_shards())
        for kind in ("model", "config", "index"):
            self.assertEqual(json.loads(source.json_bytes(kind)), self.data[kind])
        for name in self.names:
            raw, size = self.data["headers"][name]
            self.assertEqual(source.header_bytes(name, size), raw)
        remote_bytes = len(self.data["headers"][self.names[1]][0])
        stats = source.stats()
        self.assertEqual(len(opener.requests), 2)
        for request, _ in opener.requests:
            self.assertIn(f"/{REVISION}/{self.names[1]}", request.full_url)
            self.assertIsNotNone(request.get_header("Range"))
        self.assertEqual(stats["body_bytes_read"], cached_bytes + remote_bytes)
        self.assertEqual(stats["remote_body_bytes_read"], remote_bytes)
        self.assertEqual((stats["cached_headers_verified"], stats["cached_headers_reused"],
                          stats["remote_headers_fetched"]), (1, 1, 1))
        self.assertEqual(stats["tensor_payload_bytes_requested"], 0)
        self.assertFalse(stats["remote_provenance_authenticated"])
        self.assertEqual(self.cached_files(), before)

    def test_combined_cache_and_network_limit_cannot_be_exceeded(self):
        cached_bytes = ResumingMetadataSource(self.cache, MODEL, REVISION).budget.bytes
        raw, size = self.data["headers"][self.names[1]]
        limit = cached_bytes + len(raw) - 1
        source = ResumingMetadataSource(self.cache, MODEL, REVISION,
            limits=FetchLimits(total_body_bytes=limit), opener=self.transport())
        with self.assertRaisesRegex(MetadataError, "read budget exhausted"):
            source.header_bytes(self.names[1], size)
        self.assertLessEqual(source.stats()["body_bytes_read"], limit)
        self.assertEqual(source.stats()["remote_body_bytes_read"], 8)
        self.assertEqual(source.stats()["remote_headers_fetched"], 0)

    def test_complete_cached_snapshot_never_initializes_http(self):
        raw, size = self.data["headers"][self.names[1]]
        self.store.put_header(self.names[1], raw, size)
        with patch("glm_local.checkpoint_resume.HttpMetadataSource", side_effect=AssertionError("No HTTP")):
            source = ResumingMetadataSource(self.cache, MODEL, REVISION)
            for name in self.names:
                source.header_bytes(name, self.data["headers"][name][1])
        self.assertEqual(source.stats()["requests"], 0)
        self.assertEqual(source.stats()["cached_headers_reused"], 2)

    def test_corrupt_cached_header_fails_before_any_missing_header_request(self):
        # Cache the last header only: the first indexed shard would need HTTP.
        self.store.manifest["headers"].clear()
        raw, size = self.data["headers"][self.names[1]]
        self.store.put_header(self.names[1], raw, size)
        path = self.cache / "headers" / (self.names[1] + ".header")
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        opener = self.transport(self.names[0])
        with self.assertRaisesRegex(MetadataError, "SHA-256"):
            ResumingMetadataSource(self.cache, MODEL, REVISION, opener=opener)
        self.assertEqual(opener.requests, [])

    def test_wrong_identity_missing_json_and_unindexed_cached_headers_fail_closed(self):
        baseline = deepcopy(self.store.manifest)
        for change, message in ((lambda m: m.update(revision="b" * 40), "revision"),
                                (lambda m: m["objects"].pop("index"), "lacks required"),
                                (lambda m: m["headers"].update({"extra.safetensors": {}}), "not referenced")):
            manifest = deepcopy(baseline)
            change(manifest)
            write_json(self.cache / "snapshot.json", manifest)
            opener = self.transport()
            with self.subTest(message=message), self.assertRaisesRegex(MetadataError, message):
                ResumingMetadataSource(self.cache, MODEL, REVISION, opener=opener)
            self.assertEqual(opener.requests, [])
        write_json(self.cache / "snapshot.json", baseline)
        model = {**self.data["model"], "id": "other/model"}
        raw = encode(model)
        (self.cache / "model.json").write_bytes(raw)
        baseline["objects"]["model"] = digest(raw)
        write_json(self.cache / "snapshot.json", baseline)
        with self.assertRaisesRegex(MetadataError, "identity/revision"):
            ResumingMetadataSource(self.cache, MODEL, REVISION)

    def test_requested_header_must_match_verified_index_and_file_size(self):
        opener = self.transport()
        source = ResumingMetadataSource(self.cache, MODEL, REVISION, opener=opener)
        for name, size in (("unknown.safetensors", 32), (self.names[1], True),
                           (self.names[1], self.data["headers"][self.names[1]][1] + 1)):
            with self.subTest(name=name, size=size), self.assertRaisesRegex(MetadataError, "index/manifest"):
                source.header_bytes(name, size)
        self.assertEqual(opener.requests, [])

    def test_resume_executor_emits_complete_fresh_evidence_without_modifying_cache(self):
        before = self.cached_files()
        run = self.root / "new-run"
        run.mkdir()
        opener = self.transport()
        source = ResumingMetadataSource(self.cache, MODEL, REVISION, opener=opener)
        parameters = {"max_shards": 512, "budget_mib": 64, "offline": None, "resume": str(self.cache)}
        with patch.object(check, "ResumingMetadataSource", return_value=source) as resume, redirect_stdout(io.StringIO()):
            report, code = check.execute_metadata(self.root, settings(), parameters, run)
        self.assertEqual((report["status"], code), ("PASS", 0), report)
        resume.assert_called_once()
        self.assertEqual(report["source_mode"], "online_resume")
        self.assertIn("not remotely reauthenticated", report["provenance_scope"])
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["io"]["remote_headers_fetched"], 1)
        snapshot = json.loads((run / "evidence/snapshot.json").read_text())
        self.assertEqual(set(snapshot["headers"]), set(self.names))
        self.assertEqual(self.cached_files(), before)

    def test_windows_resume_launcher_installs_job_before_worker_and_preserves_parameter(self):
        events = []
        def child(command, *, cwd, limits, on_policy, timeout):
            self.assertEqual((limits.cpu_percent, limits.committed_memory_bytes), (70, 32000000000))
            on_policy(InstalledLimits(70, 32000000000, True, True, True))
            events.append("policy")
            request = Path(command[-1])
            document = json.loads(request.read_text())
            self.assertEqual(document["parameters"]["resume"], str(self.cache.absolute()))
            source = ResumingMetadataSource(self.cache, MODEL, REVISION, opener=self.transport())
            events.append("worker")
            report, code = check.execute_metadata(cwd, document["settings"], document["parameters"],
                                                  request.parent, source=source)
            write_json(request.parent / "result.json", report)
            return code
        with patch.object(check.sys, "platform", "win32"), patch.object(check, "run_local_process", side_effect=child), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(check.launch_metadata(self.root, settings(), resume=self.cache), 0)
        self.assertEqual(events, ["policy", "worker"])
        result = json.loads((self.root / "reports/metadata-latest.json").read_text())
        self.assertTrue(result["job_policy_verified"])

    def test_cli_resume_and_offline_are_exclusive_and_legacy_validation_is_preserved(self):
        write_json(self.root / "config/local.json", settings())
        with patch.object(cli, "ROOT", self.root), patch.object(check, "launch_metadata", return_value=0) as launch:
            self.assertEqual(cli.main(["metadata-check", "--resume", str(self.cache)]), 0)
        self.assertEqual(launch.call_args.kwargs["resume"], self.cache)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main(["metadata-check", "--resume", str(self.cache), "--offline", str(self.cache)])
        self.assertEqual(error.exception.code, 2)
        check.validate_parameters(512, 64, None)
        for offline, resume in ((None, ""), (None, 1), ("evidence", "evidence")):
            with self.subTest(offline=offline, resume=resume), self.assertRaises(MetadataError):
                check.validate_parameters(512, 64, offline, resume)


if __name__ == "__main__":
    unittest.main()
