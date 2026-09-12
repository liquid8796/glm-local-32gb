"""Offline metadata producer -> architecture consumer regression tests."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local import __main__ as cli, __version__
from glm_local.architecture import report as architecture
from glm_local.checkpoint_check import execute_metadata, publish_report
from glm_local.checkpoint_snapshot import write_json
from checkpoint_test_helpers import MODEL, REVISION, MemorySource, checkpoint, make_header, settings
from test_architecture_mapper import fixture_config, fixture_catalogue


def full_checkpoint():
    """Use independent complete header fixture, never real tensor payloads."""
    config = fixture_config()
    records = fixture_catalogue()
    specs = {"model-00001-of-00002.safetensors": [], "model-00002-of-00002.safetensors": []}
    filenames = list(specs)
    for index, record in enumerate(records):
        specs[filenames[index % 2]].append((record["name"], record["dtype"], record["shape"]))
    headers, mapping, siblings, total = {}, {}, [], 0
    for filename, tensors in specs.items():
        raw, size, payload = make_header(tensors)
        headers[filename] = raw, size
        mapping.update({name: filename for name, _, _ in tensors})
        siblings.append({"rfilename": filename, "size": size})
        total += payload
    expected = {"model_id": MODEL, "revision": REVISION,
                "weights": [{"name": item["rfilename"], "bytes": item["size"]} for item in siblings],
                "architecture": {**{k: v for k, v in config.items() if k != "quantization_config"},
                                 "quantization": deepcopy(config["quantization_config"])}}
    return {"model": {"id": MODEL, "sha": REVISION, "siblings": siblings}, "config": config,
            "index": {"metadata": {"total_size": total}, "weight_map": mapping},
            "headers": headers, "expected": expected}


class ArchitectureReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "docs").mkdir()
        (self.root / "config").mkdir()
        write_json(self.root / "config/local.json", settings())
        self.count = 0

    def publish(self, data=None, *, max_shards=512):
        data = data or full_checkpoint()
        write_json(self.root / "docs/model-metadata.json", data["expected"])
        self.count += 1
        directory = self.root / "reports/metadata" / f"run-{self.count}"
        directory.mkdir(parents=True)
        with redirect_stdout(io.StringIO()):
            report, code = execute_metadata(self.root, settings(),
                {"max_shards": max_shards, "budget_mib": 64, "offline": None}, directory,
                source=MemorySource(data))
            publish_report(self.root, directory, report, code)
        self.directory = directory
        self.source = report
        return report, code

    def modify_source(self, change):
        change(self.source)
        write_json(self.root / "reports/metadata-latest.json", self.source)

    def rewrite_catalogue(self, records):
        raw = b"".join((json.dumps(record, separators=(",", ":")) + "\n").encode() for record in records)
        (self.directory / "tensor-catalogue.jsonl").write_bytes(raw)
        self.source["catalogue"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.modify_source(lambda _: None)

    def catalogue_records(self):
        return [json.loads(line) for line in (self.directory / "tensor-catalogue.jsonl").read_text().splitlines()]

    def check_error(self, message):
        result = architecture.run_architecture(self.root, settings())
        self.assertEqual(result["status"], "ERROR", result)
        self.assertFalse(result["metadata_mapping_verified"])
        self.assertFalse(result["source_eligibility_verified"])
        self.assertIn(message, result["error"])
        return result

    def test_actual_full_workflow_reads_external_catalogue_and_passes_metadata_only(self):
        source, code = self.publish()
        self.assertEqual(code, 0, source)
        self.assertNotIn("tensors", source)
        self.assertNotIn("tensor_catalogue", source)
        result = architecture.run_architecture(self.root)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["catalogue"]["records"], len(fixture_catalogue()))
        self.assertEqual(result["layers"]["ids"], [0, 1])
        self.assertTrue(result["catalogue"]["digest_verified"])
        source_bytes = (self.root / "reports/metadata-latest.json").read_bytes()
        self.assertEqual(result["source_report"]["sha256"], hashlib.sha256(source_bytes).hexdigest())
        self.assertTrue(result["metadata_mapping_verified"])
        self.assertEqual(result["tool_version"], __version__)
        for flag in ("inference_verified", "payload_values_verified", "full_model_loaded", "real_checkpoint_compatible", "full_model_limits_verified"):
            self.assertIs(result[flag], False)
        rendered = (self.root / "reports/architecture-latest.md").read_text()
        self.assertIn("Catalogue evidence", rendered)
        self.assertIn("Metadata only", rendered)

    def test_original_six_record_metadata_fixture_is_read_but_needs_config_review(self):
        source, code = self.publish(checkpoint())
        self.assertEqual(code, 0)
        result = architecture.run_architecture(self.root, settings())
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["catalogue"]["records"], 6)
        self.assertEqual(result["layers"]["ids"], [0])

    def test_complete_file_total_requires_independent_header_evidence(self):
        data = full_checkpoint()
        data["index"]["metadata"]["total_size"] = sum(size for _, size in data["headers"].values())
        source, code = self.publish(data)
        self.assertEqual(code, 0, source)
        result = architecture.run_architecture(self.root, settings())
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue(result["size_accounting"]["captured_headers_reverified"])
        header = next((self.directory / "evidence/headers").iterdir())
        header.write_bytes(header.read_bytes() + b" ")
        result = architecture.run_architecture(self.root, settings())
        self.assertEqual(result["status"], "ERROR", result)

    def test_missing_source_never_falls_back_to_baseline_or_fake_inline_tensors(self):
        write_json(self.root / "docs/model-metadata.json", checkpoint()["expected"])
        self.check_error("metadata-check first")
        (self.root / "reports").mkdir(exist_ok=True)
        write_json(self.root / "reports/metadata-latest.json", {
            "status": "ERROR", "tensors": [{"name": "model.layers.0.self_attn.q_proj.weight", "dtype": "INVALID", "shape": [1]}]})
        self.check_error("actual metadata-check")

    def test_source_error_identity_and_revision_fail_before_mapper(self):
        self.publish()
        original = deepcopy(self.source)
        for key, value, message in (("status", "ERROR", "not analyzable"), ("model_id", "different/model", "differs"),
                                    ("revision", "b" * 40, "differs")):
            self.source = deepcopy(original)
            self.modify_source(lambda r: r.update({key: value}))
            with patch.object(architecture, "analyze_catalogue", side_effect=AssertionError("must not map")):
                self.check_error(message)

    def test_partial_and_upstream_review_can_never_pass(self):
        self.publish(max_shards=1)
        result = architecture.run_architecture(self.root, settings())
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["source_eligibility_verified"])
        self.publish()
        self.modify_source(lambda r: r.update(status="REVIEW_REQUIRED"))
        result = architecture.run_architecture(self.root, settings())
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["architecture_mapping_verified"])
        self.assertFalse(result["layers"]["verified"])

    def test_source_baseline_false_blocks_even_complete_valid_inventory(self):
        self.publish()
        self.modify_source(lambda r: r["baseline_comparison"].update(matched=False))
        result = architecture.run_architecture(self.root, settings())
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["metadata_mapping_verified"])
        self.assertFalse(result["fp8"]["metadata_verified"])

    def test_catalogue_hash_size_and_descriptor_path_are_checked(self):
        self.publish()
        original = deepcopy(self.source)
        for change, message in ((lambda r: r["catalogue"].update(sha256="0"*64), "SHA-256"),
                                (lambda r: r["catalogue"].update(bytes=r["catalogue"]["bytes"]+1), "size differs"),
                                (lambda r: r["catalogue"].update(name="../outside.jsonl"), "hashed tensor-catalogue"),
                                (lambda r: r.update(run_directory=str(self.root)), "inside reports/metadata")):
            self.source = deepcopy(original)
            self.modify_source(change)
            self.check_error(message)

    def test_payload_accounting_cannot_be_downgraded_to_report_only(self):
        self.publish()
        self.modify_source(lambda r: r.update(declared_tensor_payload_bytes=r["declared_tensor_payload_bytes"]+1))
        self.check_error("byte accounting")

    def test_duplicate_invalid_dtype_wrong_nbytes_and_wrong_shard_records_fail(self):
        for change, message in ((lambda records: records.__setitem__(1, deepcopy(records[0])), "Duplicate tensor"),
                                (lambda records: records[0].update(dtype="INVALID"), "unsupported dtype"),
                                (lambda records: records[0].update(nbytes=1), "byte count"),
                                (lambda records: records[0].update(shard="other.safetensors"), "uninspected shard")):
            self.publish()
            records = self.catalogue_records()
            change(records)
            self.rewrite_catalogue(records)
            self.check_error(message)

    def test_wrong_config_shape_is_review_not_pass_even_when_report_claims_pass(self):
        self.publish()
        self.modify_source(lambda r: r["config"].update(hidden_size=17))
        result = architecture.run_architecture(self.root, settings())
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertIn("CONFIG_TENSOR_SHAPE_MISMATCH", result["findings"]["by_code"])

    def test_per_shard_overlapping_or_gapped_offsets_cannot_pass_with_recomputed_digest(self):
        for mode in ("overlap", "gap"):
            self.publish()
            records = self.catalogue_records()
            for record in records:
                if mode == "overlap":
                    record["data_offsets"] = [0, record["nbytes"]]
                else:
                    record["data_offsets"] = [x + 1 for x in record["data_offsets"]]
            self.rewrite_catalogue(records)
            self.check_error("overlapping tensor offsets or a payload gap")

    def test_concurrent_source_replacement_cannot_publish_stale_pass(self):
        self.publish()
        actual_mapper = architecture.analyze_catalogue
        def changed_source(*args, **kwargs):
            result = actual_mapper(*args, **kwargs)
            self.modify_source(lambda r: r.update(status="ERROR", error="A newer metadata run failed"))
            return result
        with patch.object(architecture, "analyze_catalogue", side_effect=changed_source):
            self.check_error("source changed")

    def test_coverage_counts_and_boolean_flags_are_not_coerced(self):
        self.publish()
        original = deepcopy(self.source)
        for change in (lambda r: r["coverage"].update(checked_tensors=r["coverage"]["checked_tensors"]-1),
                       lambda r: r["coverage"].update(complete=1),
                       lambda r: r.update(metadata_structure_verified=1),
                       lambda r: r.update(headers_checked=1)):
            self.source = deepcopy(original)
            self.modify_source(change)
            self.check_error("")

    def test_record_count_and_bounded_line_policy(self):
        self.publish()
        records = self.catalogue_records()
        self.rewrite_catalogue(records[:-1])
        self.check_error("tensor count")
        self.publish()
        with patch.object(architecture, "MAX_RECORD_BYTES", 100):
            self.check_error("byte/line/tensor policy")
        with patch.object(architecture, "MAX_CATALOGUE_BYTES", 10):
            self.check_error("catalogue byte count")

    def test_cli_returns_distinct_codes_and_passes_configured_identity(self):
        self.publish()
        with patch.object(cli, "ROOT", self.root), redirect_stdout(io.StringIO()):
            code = cli.main(["--config", str(self.root / "config/local.json"), "architecture-check"])
        self.assertEqual(code, 0)
        self.modify_source(lambda r: r.update(status="ERROR"))
        with patch.object(cli, "ROOT", self.root), redirect_stdout(io.StringIO()):
            code = cli.main(["--config", str(self.root / "config/local.json"), "architecture-check"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
