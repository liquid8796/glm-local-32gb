"""Small official-format fixtures -> bounded reader -> native FP8 operations.

This validation command never accepts a real model, checkpoint directory, or
arbitrary file path. Official libraries are used only on bounded synthetic files.
"""

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import uuid

from .audit import validate_settings
from .backend_probe import compare_outputs
from .cpu_probe import NativeCpuBackend
from .cuda_probe import CudaTileBackend
from .gpu_gate import GpuBoundaryGate
from .process_metrics import sample_process, average_cpu_percent
from .reference_env import verify_reference_environment
from .winjob import JobLimits, run_local_process

FP8_SOURCE = "transformers/integrations/finegrained_fp8.py"
FP8_SOURCE_SHA256 = "e86cb993a148ba7e1baffb2f8c892fa5afb31504e81992a27da75a505c914624"
CASES = ((256, 384), (257, 259), (1, 1))


def validate_check(backend, seed):
    if backend not in ("cpu", "hybrid"):
        raise ValueError("Storage validation backend must be cpu or hybrid")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Storage validation seed must be uint32")


def verify_storage_reference():
    result = verify_reference_environment()
    if metadata.version("safetensors") != "0.8.0":
        raise RuntimeError("Storage validation requires safetensors 0.8.0 in .venv-reference")
    path = Path(metadata.distribution("transformers").locate_file(FP8_SOURCE))
    if path.stat().st_size > 2 * 1024**2:
        raise RuntimeError("FP8 reference source exceeds verification bound")
    with path.open("rb") as stream:
        source = stream.read(2 * 1024**2 + 1)
    if len(source) > 2 * 1024**2 or hashlib.sha256(source).hexdigest() != FP8_SOURCE_SHA256:
        raise RuntimeError("FP8 reference source does not match pinned revision")
    result["storage_reference"] = {"safetensors_version": "0.8.0",
                                   "fp8_source_sha256": FP8_SOURCE_SHA256,
                                   "scope": "library version and pinned dequantizer source; not all dependency contents"}
    return result


def compare_raw_tensors(reader, official):
    """Compare bytes in bounded requests. Only the small oracle retains tensors."""
    tensors = official["tensors"]
    if set(reader.tensors) != set(tensors):
        raise RuntimeError("Reader and official library found different tensor names")
    compared = 0
    hashes = {}
    for name, reference in tensors.items():
        info = reader.tensors[name]
        if tuple(reference["shape"]) != info.shape or reference["dtype"] != info.dtype:
            raise RuntimeError(f"Tensor shape/dtype mismatch for {name}")
        raw = reference["raw_bytes"]
        if info.nbytes != len(raw):
            raise RuntimeError(f"Tensor byte count mismatch for {name}")
        digest = hashlib.sha256()
        for offset in range(0, len(raw), 65536):
            count = min(65536, len(raw) - offset)
            data = reader.read_bytes(name, offset, count)
            if data != raw[offset:offset+count]:
                raise RuntimeError(f"Tensor bytes mismatch for {name}")
            digest.update(data)
            compared += len(data)
        if not raw and reader.read_bytes(name, 0, 0) != b"":
            raise RuntimeError("Empty tensor read failed")
        hashes[name] = digest.hexdigest()
    return {"passed": True, "tensor_count": len(tensors), "bytes_compared": compared,
            "tensor_sha256": hashes,
            "checksum_scope": "computed against this synthetic oracle; safetensors has no intrinsic payload checksum"}


def run_blocks(matrix, vector, cpu, gpu=None, gate=None):
    if len(vector) != matrix.cols or any(not math.isfinite(x) for x in vector):
        raise ValueError("Input vector does not match FP8 matrix")
    if gpu is not None and gate is None:
        raise ValueError("GPU block work requires a telemetry admission gate")
    result = [0.0] * matrix.rows
    counts = {"cpu_blocks": 0, "gpu_blocks": 0, "fp8_bytes": 0,
              "max_block_bytes": 0, "edge_blocks": 0, "scales": []}
    for block in matrix.iter_blocks():
        use_gpu = gpu is not None and (block.row_start // 128) % 2 == 1
        if use_gpu:
            gate.before_submit()
        output = (gpu if use_gpu else cpu).matvec_tile(
            block.weights, block.rows, block.cols,
            vector[block.col_start:block.col_start+block.cols], block.scale)
        if len(output) != block.rows or any(not math.isfinite(x) for x in output):
            raise RuntimeError("Native FP8 block returned invalid output")
        for i, value in enumerate(output):
            result[block.row_start+i] += value
        counts["gpu_blocks" if use_gpu else "cpu_blocks"] += 1
        counts["fp8_bytes"] += len(block.weights)
        counts["max_block_bytes"] = max(counts["max_block_bytes"], len(block.weights))
        counts["edge_blocks"] += int(block.rows != 128 or block.cols != 128)
        counts["scales"].append(block.scale)
    return result, counts


def official_dequantization(path, rows, cols):
    """Pinned converter parity only where it really uses 128x128 blocks."""
    import torch
    from safetensors import safe_open
    from transformers.integrations import finegrained_fp8
    if rows % 128 or cols % 128:
        return {"status": "not_applicable", "reason":
                "Pinned converter derives block sizes from grid divisibility; ragged 128-block edges use explicit independent expansion."}
    source_path = Path(finegrained_fp8.__file__)
    if source_path.stat().st_size > 2 * 1024**2:
        raise RuntimeError("Imported FP8 reference source exceeds verification bound")
    with source_path.open("rb") as source_file:
        source_bytes = source_file.read(2 * 1024**2 + 1)
    if hashlib.sha256(source_bytes).hexdigest() != FP8_SOURCE_SHA256:
        raise RuntimeError("Imported FP8 reference source differs from the pinned installation")
    with safe_open(str(path), framework="pt", device="cpu") as file:
        weight = file.get_tensor("weight")
        scales = file.get_tensor("weight_scale_inv")
        vector = file.get_tensor("vector")
    with torch.inference_mode():
        decoded = finegrained_fp8.Fp8Dequantize(None)._dequantize_one(weight, scales, output_dtype=torch.float32)
        result = torch.mv(decoded.double(), vector.double()).tolist()
    return {"status": "available", "expected_output": result,
            "scope": "unmodified pinned HF dequantizer FP32 then independent FP64 matrix-vector"}


def execute_check(settings, parameters, directory):
    validate_settings(settings)
    validate_check(**parameters)
    provenance = verify_storage_reference()
    import torch
    from .safetensor_fixture import create_fixture, reference_fixture
    from .safetensor_reader import SafeTensorReader
    from .fp8_blocks import FP8BlockMatrix
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = sample_process()
    cases = []
    with ExitStack() as stack:
        cpu = stack.enter_context(NativeCpuBackend())
        hybrid = parameters["backend"] == "hybrid"
        gpu = stack.enter_context(CudaTileBackend(settings["gpu_index"])) if hybrid else None
        gate = GpuBoundaryGate(device_index=settings["gpu_index"],
                               target=settings["gpu_average_target"],
                               window_seconds=settings["gpu_window_seconds"], max_wait_seconds=10) if hybrid else None
        for rows, cols in CASES:
            path = Path(directory) / f"synthetic-{rows}x{cols}.safetensors"
            fixture = create_fixture(path, rows, cols, parameters["seed"])
            reference = reference_fixture(path)
            with SafeTensorReader(path) as reader:
                raw_check = compare_raw_tensors(reader, reference)
                matrix = FP8BlockMatrix(reader, "weight", "weight_scale_inv")
                output, counts = run_blocks(matrix, reference["vector"], cpu, gpu, gate)
                numeric = compare_outputs(output, reference["expected_output"])
                hf = official_dequantization(path, rows, cols)
                if hf["status"] == "available":
                    hf["comparison"] = compare_outputs(output, hf.pop("expected_output"))
                stats = reader.stats()
            after_case = sample_process()
            if after_case.peak_working_set_bytes > settings["ram_budget_bytes"]:
                raise RuntimeError("Observed storage worker peak RSS exceeded budget")
            passed = numeric["passed"] and (hf["status"] != "available" or hf["comparison"]["passed"])
            cases.append({"passed": passed, "shape": [rows, cols], "fixture": fixture,
                          "raw_format_comparison": raw_check, "native_comparison": numeric,
                          "hf_dequantization": hf, "blocks": counts, "reader_stats": stats,
                          "peak_worker_rss_bytes": after_case.peak_working_set_bytes})
            print(f"Safetensors {rows}x{cols}: {'PASS' if passed else 'MISMATCH'}", flush=True)
            del reference
        if gate:
            gate.finish()
        execution = {"cpu": cpu.metadata, "gpu": gpu.device_info if gpu else None,
                     "gpu_launches": gpu.operations if gpu else 0,
                     "peak_explicit_device_buffer_bytes": gpu.peak_explicit_device_bytes if gpu else 0}
        pacing = gate.summary() if gate else {"status": "not_used", "gpu_cap_verified": False}
    after = sample_process()
    if after.peak_working_set_bytes > settings["ram_budget_bytes"]:
        raise RuntimeError("Observed storage worker peak RSS exceeded budget")
    passed = all(case["passed"] for case in cases)
    return {"status": "PASS" if passed else "NUMERICAL_MISMATCH",
            "scope": "bounded safetensors subset and FP8 block reading on synthetic fixtures",
            "synthetic_storage_verified": passed, "inference_verified": False,
            "full_model_loaded": False, "real_checkpoint_compatible": False,
            "parameters": parameters, "provenance": provenance, "cases": cases,
            "execution": execution, "pacing": pacing,
            "resources": {"peak_worker_rss_bytes": after.peak_working_set_bytes,
                          "peak_worker_private_commit_bytes": after.peak_private_commit_bytes,
                          "worker_cpu_percent": average_cpu_percent(before, after),
                          "physical_ram_hard_cap_verified": False, "full_model_limits_verified": False},
            "limitations": [
                "Only small official-library-created synthetic files were used; no real model or shard set was opened.",
                "Native reader retains bounded header metadata and one requested block, while the independent oracle loads each tiny tensor.",
                "weight_scale_inv is a dequantization multiplier, not a divisor. Only 2D E4M3 plus F32 128x128 grids are handled by the block adapter.",
                "Aligned blocks compare against the pinned HF dequantizer; ragged edges use explicit independent PyTorch scale expansion.",
                "File fingerprints detect common changes but are not cryptographic verification; full checkpoint hashes need separate verification.",
                "Reader limits are a supported subset, not full safetensors format coverage; payload NaN bytes are rejected only when computing FP8 blocks.",
                "Worker RSS includes Torch and the oracle; GPU samples may miss tiny kernels. Production RAM/GPU caps remain unverified.",
            ]}


def render_check(report):
    lines = ["# Bounded safetensors / FP8 validation", "", f"Status: **{report['status']}**", "",
             "**Real model checkpoint inference and resource limits remain unverified.**", ""]
    if "error" in report:
        lines.append(f"Error: {report['error']}")
    if "cases" in report:
        lines.extend(["| Matrix | CPU / GPU blocks | Edge blocks | Native max error | HF converter |",
                      "|---|---:|---:|---:|---|"])
        for case in report["cases"]:
            block = case["blocks"]
            lines.append(f"| {case['shape'][0]} x {case['shape'][1]} | {block['cpu_blocks']} / {block['gpu_blocks']} | "
                         f"{block['edge_blocks']} | {case['native_comparison']['max_absolute_error']:.9g} | "
                         f"{case['hf_dequantization']['status']} |")
        resource = report["resources"]
        lines.extend(["", f"Peak worker RSS: {resource['peak_worker_rss_bytes']/1024**2:.2f} MiB",
                      f"Peak worker private commit: {resource['peak_worker_private_commit_bytes']/1024**2:.2f} MiB",
                      "", "## Scope and limits", ""])
        lines.extend(f"- {item}" for item in report["limitations"])
    if "installed_job_policy" in report:
        lines.extend(["", "## Installed job policy", "", json.dumps(report["installed_job_policy"], indent=2)])
    return "\n".join(lines) + "\n"


def launch_check(root, settings, backend="hybrid", seed=7):
    from .__main__ import save_json
    validate_settings(settings)
    validate_check(backend, seed)
    root = Path(root).resolve()
    executable = root / ".venv-reference/Scripts/python.exe"
    if not executable.is_file():
        raise RuntimeError("Run setup-reference.bat first")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = root / "reports/storage" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    save_json(request, {"settings": settings, "parameters": {"backend": backend, "seed": seed}})
    captured = []
    print("Validating small synthetic safetensors files; no checkpoint download.", flush=True)
    try:
        code = run_local_process([str(executable), "-m", "glm_local.safetensor_worker", str(request)], cwd=root,
                                 limits=JobLimits(settings["cpu_job_percent"], settings["ram_budget_bytes"]),
                                 on_policy=lambda policy: captured.append(asdict(policy)), timeout=120)
        path = directory / "result.json"
        if not path.is_file() or path.stat().st_size > 2 * 1024**2:
            raise RuntimeError(f"Storage worker exited {code} without a bounded result")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (code == 0) != (report.get("status") == "PASS"):
            raise RuntimeError("Storage worker exit disagrees with reported status")
    except Exception as error:
        code = 1
        report = {"status": "ERROR", "error": str(error), "inference_verified": False,
                  "synthetic_storage_verified": False}
    report.update(installed_job_policy=captured[0] if captured else None, job_policy_verified=bool(captured),
                  child_exit_code=code, run_directory=str(directory), checked_at=datetime.now(timezone.utc).isoformat())
    save_json(directory / "result.json", report)
    save_json(root / "reports/storage-latest.json", report)
    rendered = render_check(report)
    (directory / "result.md").write_text(rendered, encoding="utf-8")
    (root / "reports/storage-latest.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Report: {root / 'reports/storage-latest.md'}")
    return code
