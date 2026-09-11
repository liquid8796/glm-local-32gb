"""Bounded synthetic FP8 experiment, deliberately separate from model inference.

The file format and numerical operation here belong to a tiny fixture. No HF
checkpoint, tokenizer, GLM graph, attention or model-generation path is loaded.
"""

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import uuid

from .audit import validate_settings
from .cpu_probe import NativeCpuBackend
from .cuda_probe import CudaTileBackend, MAX_OPERATIONS
from .gpu_gate import GpuBoundaryGate
from .process_metrics import sample_process, average_cpu_percent
from .synthetic_fp8 import (
    BLOCK, MAX_DIM, MAX_READ_BYTES, write_fixture, iter_tiles,
    read_fixture_info, fixture_vector, reference_matvec,
)
from .winjob import JobLimits, run_local_process


def validate_probe(backend, rows, cols, iterations, seed):
    if backend not in ("cpu", "gpu", "hybrid"):
        raise ValueError("backend must be cpu, gpu or hybrid")
    for name, value in (("rows", rows), ("cols", cols)):
        if type(value) is not int or not 1 <= value <= MAX_DIM:
            raise ValueError(f"{name} must be an integer in [1, {MAX_DIM}]")
    if type(iterations) is not int or not 1 <= iterations <= 8:
        raise ValueError("iterations must be an integer in [1, 8]")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be an unsigned 32-bit integer")
    row_tiles = (rows + BLOCK - 1) // BLOCK
    col_tiles = (cols + BLOCK - 1) // BLOCK
    if backend == "hybrid" and row_tiles < 2:
        raise ValueError("hybrid needs more than 128 rows so both CPU and GPU do work")
    gpu_tiles = row_tiles if backend == "gpu" else row_tiles // 2 if backend == "hybrid" else 0
    if gpu_tiles * col_tiles * iterations > MAX_OPERATIONS:
        raise ValueError(f"Requested experiment exceeds {MAX_OPERATIONS} GPU launches")


def streamed_matvec(path, vector, *, backend, cpu=None, gpu=None, gate=None):
    """Partition alternating output row-blocks across CPU/GPU, synchronously.

    There is no overlap: each bounded operation completes before the next tile
    is read. This validates both compute paths without claiming throughput gains.
    """
    info = read_fixture_info(path)
    validate_probe(backend, info["rows"], info["cols"], 1, info["seed"])
    if len(vector) != info["cols"]:
        raise ValueError("Vector length does not match fixture")
    if backend in ("cpu", "hybrid") and cpu is None:
        raise ValueError("Explicit CPU backend is required")
    if backend in ("gpu", "hybrid") and (gpu is None or gate is None):
        raise ValueError("Explicit GPU backend and telemetry gate are required")
    result = [0.0] * info["rows"]
    stats = {"cpu_tiles": 0, "gpu_tiles": 0, "weight_bytes_read": 0,
             "max_weight_tile_bytes": 0, "cpu_call_seconds": 0.0, "gpu_call_seconds": 0.0}
    for tile in iter_tiles(path):
        use_gpu = backend == "gpu" or (backend == "hybrid" and tile.row_start // BLOCK % 2 == 1)
        selected = gpu if use_gpu else cpu
        if use_gpu:
            gate.before_submit()
        started = time.perf_counter()
        values = selected.matvec_tile(tile.weights, tile.rows, tile.cols,
                                      vector[tile.col_start:tile.col_start + tile.cols], tile.scale)
        elapsed = time.perf_counter() - started
        if len(values) != tile.rows or any(not math.isfinite(value) for value in values):
            raise RuntimeError("Backend returned invalid/non-finite output")
        key = "gpu" if use_gpu else "cpu"
        stats[f"{key}_tiles"] += 1
        stats[f"{key}_call_seconds"] += elapsed
        stats["weight_bytes_read"] += len(tile.weights)
        stats["max_weight_tile_bytes"] = max(stats["max_weight_tile_bytes"], len(tile.weights))
        for offset, value in enumerate(values):
            result[tile.row_start + offset] += value
    return result, stats


def compare_outputs(actual, expected):
    if len(actual) != len(expected) or not actual:
        raise ValueError("Output vectors must have equal nonzero lengths")
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    passed = all(math.isfinite(a) and math.isfinite(b)
                 and abs(a - b) <= 1e-5 + 2e-5 * abs(b) for a, b in zip(actual, expected))
    return {"passed": passed, "rows_compared": len(actual),
            "max_absolute_error": max(errors), "absolute_tolerance": 1e-5,
            "relative_tolerance": 2e-5,
            "reference": "independent coordinate-generated E4M3FN decode and float64 math.fsum"}


def execute_probe(settings, parameters, directory):
    """Worker-side experiment. Parent attaches job policy to the final report."""
    validate_settings(settings)
    validate_probe(**parameters)
    directory = Path(directory)
    before = sample_process()
    started = time.perf_counter()
    fixture = directory / "synthetic.f8probe"
    info = write_fixture(fixture, parameters["rows"], parameters["cols"], parameters["seed"])
    vector = fixture_vector(parameters["cols"])
    backend = parameters["backend"]
    metrics = []
    comparisons = []
    latest = None
    with ExitStack() as stack:
        cpu = stack.enter_context(NativeCpuBackend()) if backend in ("cpu", "hybrid") else None
        gpu = stack.enter_context(CudaTileBackend(settings["gpu_index"])) if backend in ("gpu", "hybrid") else None
        gate = GpuBoundaryGate(device_index=settings["gpu_index"],
                               target=settings["gpu_average_target"],
                               window_seconds=settings["gpu_window_seconds"],
                               max_wait_seconds=10) if gpu else None
        # Independent dense oracle uses only fixture parameters, never the file reader.
        reference = reference_matvec(parameters["rows"], parameters["cols"], parameters["seed"])
        compute_before = sample_process()
        compute_started = time.perf_counter()
        for _ in range(parameters["iterations"]):
            latest, stats = streamed_matvec(fixture, vector, backend=backend, cpu=cpu, gpu=gpu, gate=gate)
            comparisons.append(compare_outputs(latest, reference))
            metrics.append(stats)
            current = sample_process()
            # This is a measured abort guard, not an instantaneous physical RAM cap.
            if current.peak_working_set_bytes > settings["ram_budget_bytes"]:
                raise RuntimeError("Observed worker RSS exceeded requested budget; aborting probe")
        if gate:
            gate.finish()
        compute_seconds = time.perf_counter() - compute_started
        compute_after = sample_process()
        details = {
            "cpu": cpu.metadata if cpu else None,
            "gpu": gpu.device_info if gpu else None,
            "gpu_launches": gpu.operations if gpu else 0,
            "peak_explicit_device_buffer_bytes": gpu.peak_explicit_device_bytes if gpu else 0,
        }
        pacing = gate.summary() if gate else {"status": "not_used", "gpu_cap_verified": False}
    after = sample_process()
    total_seconds = time.perf_counter() - started
    numerical_pass = all(item["passed"] for item in comparisons)
    assert latest is not None
    result = {
        "status": "PASS" if numerical_pass else "NUMERICAL_MISMATCH",
        "scope": "synthetic FP8 matrix-vector probe only",
        "inference_verified": False, "full_model_loaded": False,
        "job_policy_verified": False,
        "parameters": parameters, "fixture": info, "backend_details": details,
        "partition": "alternating output row blocks; synchronous CPU/GPU operations",
        "comparisons": comparisons, "iterations": metrics,
        "output_sha256": hashlib.sha256(struct.pack(f"<{len(latest)}d", *latest)).hexdigest(),
        "resources": {
            "before": asdict(before), "after": asdict(after),
            "compute_before": asdict(compute_before), "compute_after": asdict(compute_after),
            "total_wall_seconds": total_seconds, "compute_wall_seconds_including_pacing": compute_seconds,
            "worker_average_cpu_percent": average_cpu_percent(before, after),
            "compute_worker_average_cpu_percent": average_cpu_percent(compute_before, compute_after),
            "peak_worker_rss_bytes": after.peak_working_set_bytes,
            "peak_worker_private_commit_bytes": after.peak_private_commit_bytes,
            "max_allowed_single_file_read_bytes": MAX_READ_BYTES,
            "physical_ram_hard_cap_verified": False, "full_model_resource_limits_verified": False,
        },
        "pacing": pacing,
        "limitations": [
            "This is one linear algebra primitive, not GLM inference or token generation.",
            "RSS/private-commit peaks are OS process-lifetime counters; other processes and file cache are excluded.",
            "Explicit CUDA buffer bytes exclude CUDA context, driver and JIT overhead.",
            "GPU device telemetry includes other applications; tiny kernels can be missed between samples.",
            "Pacing delays submission; it cannot enforce a strict GPU utilization ceiling.",
            "Repeated tiny-file reads may hit OS cache; this does not measure full-checkpoint SSD streaming speed.",
        ],
    }
    return result


def render_probe_report(report):
    lines = ["# Synthetic FP8 CPU/GPU probe", "", f"Status: **{report['status']}**", "",
             "**No model inference or token generation has been verified.**", ""]
    if "error" in report:
        lines.append(f"Error: {report['error']}")
    if "comparisons" in report:
        params, resource = report["parameters"], report["resources"]
        lines.extend([
            f"Backend: {params['backend']}; shape: {params['rows']} x {params['cols']}; iterations: {params['iterations']}",
            f"Maximum absolute error: {max(c['max_absolute_error'] for c in report['comparisons']):.9g}",
            f"CPU tiles: {sum(i['cpu_tiles'] for i in report['iterations'])}; GPU tiles: {sum(i['gpu_tiles'] for i in report['iterations'])}",
            f"Peak worker RSS: {resource['peak_worker_rss_bytes']/1024**2:.2f} MiB",
            f"Peak worker private commit: {resource['peak_worker_private_commit_bytes']/1024**2:.2f} MiB",
            f"Average worker CPU: {resource['worker_average_cpu_percent']:.3f}%",
            f"Total wall time: {resource['total_wall_seconds']:.3f} seconds",
            f"Per-read byte bound: {resource['max_allowed_single_file_read_bytes']} bytes",
            "", "## GPU pacing observations", "",
            f"{json.dumps(report['pacing'], indent=2)}", "",
            "## Interpretation", "",
        ])
        lines.extend(f"- {item}" for item in report["limitations"])
    if "installed_job_policy" in report:
        lines.extend(["", "## Installed Windows job policy", "",
                      json.dumps(report["installed_job_policy"], indent=2)])
    return "\n".join(lines) + "\n"


def launch_probe(root, settings, backend="hybrid", rows=384, cols=384, iterations=3, seed=7):
    from .__main__ import save_json
    validate_settings(settings)
    parameters = dict(backend=backend, rows=rows, cols=cols, iterations=iterations, seed=seed)
    validate_probe(**parameters)
    root = Path(root).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = root / "reports" / "probes" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    save_json(request, {"settings": settings, "parameters": parameters})
    captured = []
    print(f"Running bounded {backend} FP8 probe. No model weights are used.", flush=True)
    try:
        code = run_local_process(
            [sys.executable, "-m", "glm_local.probe_worker", str(request)], cwd=root,
            limits=JobLimits(settings["cpu_job_percent"], settings["ram_budget_bytes"]),
            on_policy=lambda policy: captured.append(asdict(policy)), timeout=120,
        )
        result_path = directory / "result.json"
        if not result_path.is_file() or result_path.stat().st_size > 2 * 1024**2:
            raise RuntimeError(f"Worker exited {code} without a bounded result report")
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if code != 0 and report.get("status") == "PASS":
            raise RuntimeError("Worker exit disagrees with reported success")
    except Exception as error:
        code = 1
        report = {"status": "ERROR", "error": str(error), "inference_verified": False,
                  "full_model_loaded": False, "parameters": parameters}
    report["installed_job_policy"] = captured[0] if captured else None
    report["job_policy_verified"] = bool(captured)
    report["child_exit_code"] = code
    report["run_directory"] = str(directory)
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    save_json(directory / "result.json", report)
    save_json(root / "reports" / "backend-probe-latest.json", report)
    rendered = render_probe_report(report)
    (directory / "result.md").write_text(rendered, encoding="utf-8")
    (root / "reports" / "backend-probe-latest.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Report: {root / 'reports' / 'backend-probe-latest.md'}")
    return code
