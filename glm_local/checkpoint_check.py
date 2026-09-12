"""Metadata-only checkpoint audit, evidence capture and per-run diagnostics."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import uuid

from . import __version__
from .audit import validate_settings
from .checkpoint_http import (FetchLimits, HttpMetadataSource, MAX_SHARDS, MetadataError,
                              strict_json, validate_target)
from .checkpoint_schema import (compare_snapshot, review_tensors, runtime_index_policy,
                                validate_index, validate_manifest)
from .checkpoint_snapshot import EvidenceStore, OfflineMetadataSource, write_json
from .safetensor_reader import MAX_READ_BYTES, parse_header_bytes
from .winjob import JobLimits, run_local_process

EXIT_CODES = {"PASS": 0, "ERROR": 1, "PARTIAL": 2, "REVIEW_REQUIRED": 3, "INTERRUPTED": 130}


def validate_parameters(max_shards, budget_mib, offline):
    if type(max_shards) is not int or not 1 <= max_shards <= MAX_SHARDS:
        raise MetadataError("max_shards must be an integer from 1 to 512")
    if type(budget_mib) is not int or not 1 <= budget_mib <= 128:
        raise MetadataError("budget_mib must be an integer from 1 to 128")
    if offline is not None and (not isinstance(offline, str) or not offline):
        raise MetadataError("offline must be an evidence directory path or null")


def initial_report(settings, parameters):
    return {"tool_version": __version__, "scope": "checkpoint_metadata_only", "status": "ERROR",
            "model_id": settings["model_id"], "revision": settings["revision"],
            "parameters": parameters, "stage": "initialization", "headers_checked": 0,
            "metadata_structure_verified": False, "architecture_mapping_verified": False,
            "inference_verified": False, "real_checkpoint_compatible": False,
            "full_model_loaded": False, "full_model_limits_verified": False,
            "payload_values_verified": False, "gpu_used": False, "gpu_cap_verified": False,
            "job_policy_verified": False, "physical_ram_hard_cap_verified": False,
            "notes": [
                "PASS is metadata-only; it never unblocks doctor or proves full-model inference.",
                "Scale values, NaNs, tensor payload checksums and numeric projection were not read/tested.",
                "Known-name shape checks are incomplete; unreviewed tensors are retained in the catalogue.",
                "Snapshot hashes detect local changes, not a forged snapshot with recomputed hashes.",
                "HTTP/body and header limits do not cap OS/TLS buffers or whole-system physical RAM."]}


def _file_digest(path):
    result = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while raw := stream.read(MAX_READ_BYTES):
            count += len(raw)
            result.update(raw)
    return {"name": path.name, "bytes": count, "sha256": result.hexdigest()}


def execute_metadata(root, settings, parameters, directory, *, source=None):
    """Run bounded inspection; source injection is used by offline unit tests."""
    report = initial_report(settings, parameters)
    root, directory = Path(root), Path(directory)
    try:
        validate_settings(settings)
        validate_target(settings["model_id"], settings["revision"])
        validate_parameters(**parameters)
        expected_path = root / "docs" / "model-metadata.json"
        if not 1 <= expected_path.stat().st_size <= 1024**2:
            raise MetadataError("Project baseline metadata snapshot exceeds 1 MiB")
        expected = strict_json(expected_path.read_bytes())
        if (not isinstance(expected, dict) or expected.get("model_id") != settings["model_id"]
                or expected.get("revision") != settings["revision"]):
            raise MetadataError("Project snapshot and configured checkpoint identity/revision differ")
        limits = FetchLimits(total_body_bytes=parameters["budget_mib"] * 1024**2)
        report["stage"] = "source_initialization"
        if source is None:
            source = (OfflineMetadataSource(parameters["offline"], settings["model_id"], settings["revision"], limits=limits)
                      if parameters["offline"] is not None else
                      HttpMetadataSource(settings["model_id"], settings["revision"], limits=limits))
        store = EvidenceStore(directory / "evidence", settings["model_id"], settings["revision"])
        report["evidence"] = "evidence/snapshot.json"
        report["source_mode"] = source.mode
        report["provenance_scope"] = ("Pinned HTTPS URLs plus model API id/sha and any supplied X-Repo-Commit"
                                      if source.mode == "online" else
                                      "Local snapshot identity/hash consistency only; no remote reauthentication")
        documents, document_sizes = {}, {}
        for kind in ("model", "config", "index"):
            report["stage"] = f"fetch_{kind}"
            raw = source.json_bytes(kind)
            store.put_json(kind, raw)
            documents[kind], document_sizes[kind] = strict_json(raw), len(raw)
            if kind == "model":
                sizes = validate_manifest(settings["model_id"], settings["revision"], documents[kind])
                report["manifest_identity_matches"] = True
        report["stage"] = "validate_config_and_index"
        by_shard = validate_index(documents["index"], sizes)
        report["baseline_comparison"] = compare_snapshot(
            settings["model_id"], settings["revision"], sizes, documents["config"], expected)
        report["config"] = documents["config"]
        mapping = documents["index"]["weight_map"]
        available = source.available_shards()
        if available is not None and not available <= set(by_shard):
            raise MetadataError("Snapshot contains a header not referenced by the checkpoint index")
        selected = sorted(by_shard if available is None else available)[:parameters["max_shards"]]
        tensors, checked = {}, []
        report["coverage"] = {"total_shards": len(by_shard), "selected_shards": len(selected),
                              "checked_shards": 0, "total_index_tensors": len(mapping),
                              "checked_tensors": 0, "complete": False}
        payload_bytes = 0
        for number, filename in enumerate(selected, 1):
            report["stage"], report["active_shard"] = "fetch_header", filename
            raw = source.header_bytes(filename, sizes[filename])
            store.put_header(filename, raw, sizes[filename])
            report["stage"] = "validate_header"
            header_tensors, _ = parse_header_bytes(raw[8:], sizes[filename])
            if set(header_tensors) != by_shard[filename]:
                raise MetadataError(f"Index/header tensor set mismatch in {filename}")
            tensors.update(header_tensors)
            payload_bytes += sum(t.nbytes for t in header_tensors.values())
            if payload_bytes > documents["index"]["metadata"]["total_size"]:
                raise MetadataError("Observed header payload bytes already exceed index total_size")
            checked.append(filename)
            report["headers_checked"] = number
            report["coverage"].update(checked_shards=number, checked_tensors=len(tensors))
            if number == 1 or number % 16 == 0 or number == len(selected):
                print(f"Metadata headers: {number}/{len(selected)} selected, {len(by_shard)} total shards", flush=True)
        report.pop("active_shard", None)
        complete = len(checked) == len(by_shard)
        if complete and payload_bytes != documents["index"]["metadata"]["total_size"]:
            raise MetadataError("Index total_size differs from complete header payload byte counts")
        report["coverage"].update(complete=complete, checked_shard_names=checked,
                                  uninspected_shards=sorted(set(by_shard) - set(checked)))
        report["metadata_structure_verified"] = complete
        report["observed_tensor_payload_bytes"] = payload_bytes  # Described bytes, NOT fetched bytes.
        report["declared_tensor_payload_bytes"] = documents["index"]["metadata"]["total_size"]
        report["stage"] = "review_tensor_metadata"
        catalogue = directory / "tensor-catalogue.jsonl"
        report["tensor_review"] = review_tensors(tensors, mapping, documents["config"],
                                                  complete=complete, catalogue_path=catalogue)
        report["catalogue"] = _file_digest(catalogue)
        report["runtime_index_policy"] = runtime_index_policy(document_sizes["index"], len(mapping))
        needs_review = (not report["baseline_comparison"]["matched"]
                        or report["tensor_review"]["findings"]["count"] != 0)
        report["status"] = "PARTIAL" if not complete else "REVIEW_REQUIRED" if needs_review else "PASS"
        report["stage"] = "complete"
    except KeyboardInterrupt:
        report.update(status="INTERRUPTED", error="Metadata inspection interrupted")
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}"[:2000],
                      traceback=traceback.format_exc(limit=12, chain=False)[-16000:])
    if report["status"] in ("ERROR", "INTERRUPTED"):
        report["metadata_structure_verified"] = False
    if source is not None:
        report["io"] = source.stats()
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    report["worker_environment"] = {"python_version": sys.version.split()[0],
                                    "python_executable": sys.executable}
    return report, EXIT_CODES[report["status"]]


def render_metadata(report):
    lines = ["# Checkpoint metadata audit", "", f"Status: **{report['status']}**",
             "", "Metadata only. No full-model inference or tensor payload validation.", "",
             f"Tool: `{report['tool_version']}`", f"Model: `{report['model_id']}`",
             f"Revision: `{report['revision']}`", f"Stage: `{report.get('stage')}`", "",
             f"Metadata structure verified: **{report['metadata_structure_verified']}**",
             "Real checkpoint runtime compatible: **False**", "Full model loaded: **False**", ""]
    if report.get("error"):
        lines += ["## Error", "", report["error"], ""]
    for title, key in (("Coverage", "coverage"), ("Metadata I/O", "io"),
                       ("Baseline comparison", "baseline_comparison"),
                       ("Current runtime index policy", "runtime_index_policy")):
        if key in report:
            value = dict(report[key])
            value.pop("checked_shard_names", None)
            value.pop("uninspected_shards", None)
            lines += [f"## {title}", "", "```json", json.dumps(value, indent=2, ensure_ascii=False), "```", ""]
    review = report.get("tensor_review")
    if review:
        lines += ["## Tensor review", "", "```json", json.dumps(review, indent=2, ensure_ascii=False), "```", ""]
    lines += ["## Scope limits", "", *["- " + note for note in report["notes"]], "",
              "Per-run evidence: `evidence/snapshot.json`; full observed catalogue: `tensor-catalogue.jsonl`.",
              "These paths are relative to the per-run directory, not reports/.", ""]
    if report.get("traceback"):
        lines += ["## Traceback", "", "```text", report["traceback"], "```", ""]
    return "\n".join(lines)


def publish_report(root, directory, report, code, policy=None):
    report.update(installed_job_policy=policy, job_policy_verified=policy is not None,
                  child_exit_code=code, run_directory=str(directory))
    write_json(directory / "result.json", report)
    write_json(root / "reports" / "metadata-latest.json", report)
    rendered = render_metadata(report)
    (directory / "result.md").write_text(rendered, encoding="utf-8")
    (root / "reports" / "metadata-latest.md").write_text(rendered, encoding="utf-8")
    print(f"Metadata status: {report['status']}; full-model runtime remains unverified.", flush=True)
    print(f"Report: {directory / 'result.md'}", flush=True)
    return code


def launch_metadata(root, settings, max_shards=MAX_SHARDS, budget_mib=64, offline=None):
    validate_settings(settings)
    parameters = {"max_shards": max_shards, "budget_mib": budget_mib,
                  "offline": str(Path(offline).absolute()) if offline is not None else None}
    validate_parameters(**parameters)
    root = Path(root).resolve()
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = root / "reports" / "metadata" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    write_json(request, {"settings": settings, "parameters": parameters})
    captured = []
    print("Inspecting pinned checkpoint metadata only; no tokenizer, tensor payload, Torch or CUDA.", flush=True)
    if sys.platform != "win32":
        report, code = execute_metadata(root, settings, parameters, directory)
        report["notes"].append("Non-Windows metadata audit: OS CPU/committed-memory quota was not installed.")
    else:
        try:
            code = run_local_process([sys.executable, "-m", "glm_local.checkpoint_worker", str(request)],
                                     cwd=root, limits=JobLimits(settings["cpu_job_percent"], settings["ram_budget_bytes"]),
                                     on_policy=lambda policy: captured.append(asdict(policy)),
                                     timeout=FetchLimits().total_seconds + 60)
            result = directory / "result.json"
            if not result.is_file() or result.stat().st_size > 4 * 1024**2:
                raise MetadataError("Metadata worker did not produce a bounded result")
            report = strict_json(result.read_bytes())
            if EXIT_CODES.get(report.get("status")) != code:
                raise MetadataError("Metadata worker exit code disagrees with report status")
            if not captured:
                raise MetadataError("Metadata worker has no verified Windows Job policy")
        except (Exception, KeyboardInterrupt) as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else 1
            report = initial_report(settings, parameters)
            report.update(status="INTERRUPTED" if code == 130 else "ERROR", stage="worker_launch_or_readback",
                          error=f"{type(error).__name__}: {error}"[:2000],
                          traceback=traceback.format_exc(limit=12, chain=False)[-16000:])
    return publish_report(root, directory, report, code, captured[0] if captured else None)
