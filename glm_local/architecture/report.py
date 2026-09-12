"""Validate the local JSONL evidence emitted by metadata-check; no payload I/O."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat

from .. import __version__
from ..checkpoint_http import MetadataError, MAX_SHARDS, MAX_TENSORS, shard_filename, strict_json, validate_target
from ..checkpoint_snapshot import write_json
from ..safetensor_reader import DTYPE_ITEMSIZE, MAX_RANK, MAX_DIMENSION, MAX_INTEGER
from .mapper import analyze_catalogue

MAX_REPORT_BYTES = 4 * 1024**2
MAX_CATALOGUE_BYTES = 128 * 1024**2
MAX_RECORD_BYTES = 16384
EXIT_CODES = {"PASS": 0, "REVIEW_REQUIRED": 2, "ERROR": 1}


def _fingerprint(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise MetadataError("Architecture evidence must be a regular file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _bounded_json(path, cap, *, capture=False):
    identity = _fingerprint(path)
    if not 1 <= identity[2] <= cap:
        raise MetadataError("Architecture input JSON exceeds its size policy")
    with path.open("rb") as stream:
        data = stream.read(cap + 1)
    if len(data) != identity[2] or _fingerprint(path) != identity:
        raise MetadataError("Architecture input changed during read")
    value = strict_json(data)
    if not isinstance(value, dict):
        raise MetadataError("Architecture input JSON must be an object")
    if capture:
        return value, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, identity
    return value


def _integer(value, maximum, label, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise MetadataError(f"Invalid bounded {label}")
    return value


def _validate_record(record, shards):
    required = {"name", "shard", "dtype", "shape", "nbytes", "data_offsets"}
    if not isinstance(record, dict) or not required <= set(record):
        raise MetadataError("Catalogue record lacks required tensor metadata")
    name = record["name"]
    if (not isinstance(name, str) or not 1 <= len(name.encode("utf-8")) <= 512
            or any(ord(c) < 32 for c in name)):
        raise MetadataError("Invalid catalogue tensor name")
    if shard_filename(record["shard"]) not in shards:
        raise MetadataError("Catalogue tensor references an uninspected shard")
    dtype, shape = record["dtype"], record["shape"]
    if not isinstance(dtype, str) or dtype not in DTYPE_ITEMSIZE:
        raise MetadataError("Catalogue tensor has an unsupported dtype")
    if not isinstance(shape, list) or len(shape) > MAX_RANK:
        raise MetadataError("Catalogue tensor has an invalid rank")
    for dimension in shape:
        _integer(dimension, MAX_DIMENSION, "shape dimension")
    elements = 0 if 0 in shape else 1
    if elements:
        for dimension in shape:
            if elements > MAX_INTEGER // dimension:
                raise MetadataError("Catalogue shape exceeds integer policy")
            elements *= dimension
    if elements > MAX_INTEGER // DTYPE_ITEMSIZE[dtype]:
        raise MetadataError("Catalogue byte count exceeds integer policy")
    nbytes = _integer(record["nbytes"], MAX_INTEGER, "tensor byte count")
    if nbytes != elements * DTYPE_ITEMSIZE[dtype]:
        raise MetadataError("Catalogue tensor byte count differs from dtype/shape")
    offsets = record["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise MetadataError("Catalogue tensor needs two data offsets")
    begin, end = [_integer(v, MAX_INTEGER, "tensor offset") for v in offsets]
    if end < begin or end - begin != nbytes:
        raise MetadataError("Catalogue tensor offsets disagree with byte count")


def _load_catalogue(path, descriptor, coverage):
    if (not isinstance(descriptor, dict) or descriptor.get("name") != "tensor-catalogue.jsonl"
            or not isinstance(descriptor.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])):
        raise MetadataError("Metadata report must reference a hashed tensor-catalogue.jsonl")
    declared = _integer(descriptor.get("bytes"), MAX_CATALOGUE_BYTES, "catalogue byte count")
    identity = _fingerprint(path)
    if identity[2] != declared:
        raise MetadataError("Catalogue size differs from metadata report")
    shards = set(coverage["checked_shard_names"])
    records, names = [], set()
    digest, observed = hashlib.sha256(), 0
    # Bounded buffered reads and line lengths; no unbounded file read/readlines.
    with path.open("rb", buffering=65536) as stream:
        while True:
            line = stream.readline(MAX_RECORD_BYTES + 1)
            if not line:
                break
            observed += len(line)
            if len(line) > MAX_RECORD_BYTES or observed > declared or len(records) >= MAX_TENSORS:
                raise MetadataError("Catalogue exceeds byte/line/tensor policy")
            digest.update(line)
            record = strict_json(line)
            _validate_record(record, shards)
            if record["name"] in names:
                raise MetadataError("Duplicate tensor in architecture catalogue")
            names.add(record["name"])
            records.append(record)
    if (_fingerprint(path) != identity or observed != declared
            or digest.hexdigest() != descriptor["sha256"]):
        raise MetadataError("Catalogue changed or SHA-256 differs from metadata report")
    if len(records) != coverage["checked_tensors"]:
        raise MetadataError("Catalogue tensor count differs from inspected coverage")
    if {record["shard"] for record in records} != shards:
        raise MetadataError("Catalogue shard coverage differs from metadata report")
    intervals = {name: [] for name in shards}
    for record in records:
        intervals[record["shard"]].append(tuple(record["data_offsets"]))
    for shard, ranges in intervals.items():
        cursor = 0
        for begin, end in sorted(ranges):
            if begin != cursor:
                raise MetadataError(f"Catalogue has overlapping tensor offsets or a payload gap in {shard}")
            cursor = end
    return records, {"name": path.name, "bytes": observed, "sha256": digest.hexdigest(),
                     "records": len(records), "max_record_bytes": MAX_RECORD_BYTES,
                     "read_buffer_bytes": 65536, "digest_verified": True}


def _coverage(source):
    coverage = source.get("coverage")
    if not isinstance(coverage, dict) or type(coverage.get("complete")) is not bool:
        raise MetadataError("Metadata report lacks explicit coverage")
    total_shards = _integer(coverage.get("total_shards"), MAX_SHARDS, "total shards", 1)
    checked = _integer(coverage.get("checked_shards"), total_shards, "checked shards")
    total_tensors = _integer(coverage.get("total_index_tensors"), MAX_TENSORS, "index tensors", 1)
    count = _integer(coverage.get("checked_tensors"), total_tensors, "checked tensors")
    names = coverage.get("checked_shard_names")
    if not isinstance(names, list) or len(names) != checked:
        raise MetadataError("Inspected shard names differ from coverage count")
    for name in names:
        shard_filename(name)
    if len({name.casefold() for name in names}) != checked:
        raise MetadataError("Duplicate inspected shard names")
    if source.get("headers_checked") != checked or type(source.get("headers_checked")) is not int:
        raise MetadataError("Report header count disagrees with coverage")
    complete = checked == total_shards and count == total_tensors
    if coverage["complete"] != complete:
        raise MetadataError("Report claims contradictory completeness")
    if type(source.get("metadata_structure_verified")) is not bool or source["metadata_structure_verified"] != complete:
        raise MetadataError("Report structure-verification flag disagrees with coverage")
    if source["status"] == "PASS" and not complete:
        raise MetadataError("A partial metadata report cannot claim PASS")
    return coverage


def _analyze_source(root, settings, source_path):
    source, source_evidence, source_identity = _bounded_json(source_path, MAX_REPORT_BYTES, capture=True)
    if source.get("scope") != "checkpoint_metadata_only":
        raise MetadataError("Source must be an actual metadata-check report")
    if source.get("model_id") != settings["model_id"] or source.get("revision") != settings["revision"]:
        raise MetadataError("Metadata source model/revision differs from current settings")
    status = source.get("status")
    if status not in ("PASS", "REVIEW_REQUIRED", "PARTIAL"):
        raise MetadataError(f"Metadata source is not analyzable: status={status!r}")
    coverage = _coverage(source)
    config = source.get("config")
    if not isinstance(config, dict):
        raise MetadataError("Metadata source lacks model config")
    baseline = source.get("baseline_comparison")
    if not isinstance(baseline, dict) or type(baseline.get("matched")) is not bool:
        raise MetadataError("Metadata source lacks its baseline comparison")
    run_directory = source.get("run_directory")
    if not isinstance(run_directory, str) or not run_directory:
        raise MetadataError("Metadata report lacks its run directory")
    run_path = Path(run_directory)
    if not run_path.is_absolute():
        run_path = root / run_path
    run_path = run_path.resolve()
    reports_root = (root / "reports" / "metadata").resolve()
    if run_path == reports_root or not run_path.is_relative_to(reports_root):
        raise MetadataError("Referenced metadata run must stay inside reports/metadata")
    path = (run_path / "tensor-catalogue.jsonl").resolve()
    if not path.is_relative_to(run_path):
        raise MetadataError("Catalogue resolves outside the metadata run directory")
    records, evidence = _load_catalogue(path, source.get("catalogue"), coverage)
    observed = sum(record["nbytes"] for record in records)
    reported = _integer(source.get("observed_tensor_payload_bytes"), MAX_INTEGER, "observed payload bytes")
    declared = _integer(source.get("declared_tensor_payload_bytes"), MAX_INTEGER, "declared payload bytes")
    if observed != reported or observed > declared or (coverage["complete"] and observed != declared):
        raise MetadataError("Catalogue/index tensor byte accounting differs")
    analysis = analyze_catalogue(records, config, complete=coverage["complete"])
    source_eligible = status == "PASS" and baseline["matched"] and coverage["complete"]
    if not source_eligible:
        analysis["status"] = "REVIEW_REQUIRED"
        for flag in ("metadata_mapping_verified", "architecture_mapping_verified"):
            analysis[flag] = False
        for group in ("layers", "attention", "moe"):
            if group in analysis:
                analysis[group]["verified"] = False
        if "fp8" in analysis:
            analysis["fp8"]["metadata_verified"] = False
        analysis["source_review_reason"] = "Source status, completeness or baseline comparison still needs review"
    if _fingerprint(source_path) != source_identity:
        raise MetadataError("Metadata source changed during architecture analysis; rerun against the latest report")
    analysis.update(source=str(source_path), source_status=status,
                    source_mode=source.get("source_mode", "unspecified"),
                    model_id=settings["model_id"], revision=settings["revision"],
                    source_coverage=coverage, catalogue=evidence, catalogue_path=str(path),
                    source_report=source_evidence,
                    source_eligibility_verified=source_eligible,
                    provenance_scope="Local model/revision, coverage, catalogue size/hash and accounting consistency only; no remote reauthentication")
    return analysis


def render_architecture(result):
    lines = ["# Architecture metadata review", "", f"Status: **{result['status']}**", "",
             "Metadata only; real checkpoint inference, payload values and resource limits remain unverified.", "",
             f"Tool version: {result['tool_version']}"]
    if "error" in result:
        lines.extend(["", "## Error", "", result["error"]])
    for name in ("source", "source_status", "source_mode", "source_review_reason", "provenance_scope"):
        if name in result:
            lines.append(f"{name}: {result[name]}")
    for title, key in (("Captured source report", "source_report"), ("Catalogue evidence", "catalogue"), ("Layer inventory", "layers"),
                       ("Tensor roles", "groups"), ("Shape and completeness checks", "shape_checks"),
                       ("Findings", "findings")):
        if key in result:
            lines.extend(["", f"## {title}", "", "```json", json.dumps(result[key], indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines) + "\n"


def run_architecture(root, settings=None):
    root = Path(root).resolve()
    source_path = root / "reports" / "metadata-latest.json"
    try:
        if settings is None:
            settings = _bounded_json(root / "config" / "local.json", 65536)
        validate_target(settings["model_id"], settings["revision"])
        if not source_path.is_file():
            raise MetadataError("Run metadata-check first; the baseline snapshot has no tensor catalogue")
        result = _analyze_source(root, settings, source_path)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        result = {"status": "ERROR", "error": str(error)[:2000], "source": str(source_path),
                  "metadata_mapping_verified": False, "architecture_mapping_verified": False,
                  "source_eligibility_verified": False}
    result.update(tool_version=__version__, metadata_only=True, payload_values_verified=False,
                  real_checkpoint_compatible=False, inference_verified=False, full_model_loaded=False,
                  full_model_limits_verified=False, checked_at=datetime.now(timezone.utc).isoformat())
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "architecture-latest.json", result)
    (report_dir / "architecture-latest.md").write_text(render_architecture(result), encoding="utf-8")
    return result
