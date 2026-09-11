"""Explicit readiness gates; no unverified backend is labeled runnable."""

import math
import shutil
from pathlib import Path

from .metadata import safe_relative_path


def validate_settings(settings):
    bounds = {
        "ram_budget_bytes": (1, 32_000_000_000),
        "cpu_job_percent": (1, 70),
        "gpu_average_target": (0.01, 0.6),
        "gpu_window_seconds": (1, 3600),
        "disk_reserve_bytes": (0, 10**13),
    }
    for key, (lower, upper) in bounds.items():
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid {key}")
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}]")
    for key in ("ram_budget_bytes", "cpu_job_percent", "disk_reserve_bytes"):
        if type(settings[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if type(settings.get("gpu_index")) is not int or settings["gpu_index"] < 0:
        raise ValueError("gpu_index must be a nonnegative integer")
    if not isinstance(settings.get("model_directory"), str) or not settings["model_directory"]:
        raise ValueError("model_directory must be a path")


def inspect_local_weights(directory, weights):
    directory = Path(directory).resolve()
    missing = []
    wrong = []
    present = 0
    required = 0
    for entry in weights:
        relative = safe_relative_path(entry["name"])
        path = (directory / str(relative)).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Checkpoint shard resolves outside model_directory")
        expected = entry["bytes"]
        if type(expected) is not int or expected <= 0:
            raise ValueError("Invalid weight size in snapshot")
        if not path.is_file():
            missing.append(entry["name"])
            required += expected
        elif path.stat().st_size != expected:
            wrong.append(entry["name"])
            required += expected
        else:
            present += expected
    return {"matching_size_bytes": present, "missing_shards": missing,
            "wrong_size_shards": wrong, "additional_bytes_required": required,
            "verification": "file sizes only; content and revision NOT verified"}


def disk_free_for(directory):
    probe = Path(directory).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def evaluate(settings, snapshot, hardware, project_root):
    validate_settings(settings)
    if snapshot.get("model_id") != settings["model_id"] or snapshot.get("revision") != settings["revision"]:
        raise ValueError("Snapshot does not match configured model revision; refresh metadata")
    weights = snapshot.get("weights", [])
    if not weights or sum(w["bytes"] for w in weights) != snapshot.get("weight_bytes"):
        raise ValueError("Missing or inconsistent checkpoint manifest")
    names = [w["name"].casefold() for w in weights]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate checkpoint shard in snapshot")
    model_dir = Path(settings["model_directory"])
    if not model_dir.is_absolute():
        model_dir = Path(project_root) / model_dir
    model_dir = model_dir.resolve()
    local = inspect_local_weights(model_dir, weights)
    free = disk_free_for(model_dir)
    needed = local["additional_bytes_required"] + settings["disk_reserve_bytes"]
    blockers = [
        {"code": "GLM_BACKEND_NOT_IMPLEMENTED", "message":
         "The pinned Kimi engine is CPU-only and cannot execute glm_moe_dsa FP8. "
         "A fixed synthetic decoder and FP8 CPU/CUDA kernels are available. "
         "The miniature also has an official FP32 comparison command. Production GLM checkpoint mapping, "
         "full dtype fidelity, tokenizer and full-model integration remain unverified."},
        {"code": "RAM_RESIDENT_CAP_UNVERIFIED", "message":
         "Windows Job Object caps committed memory, not total resident memory or OS file cache. "
         "A 32 GB physical RAM ceiling has not been established."},
        {"code": "GPU_PACING_NOT_INTEGRATED", "message":
         "Cooperative pacing gates synthetic FP8 and miniature decoder work. Real GLM inference is unavailable, "
         "so the 60% average target under full-model load remains unverified."},
    ]
    if local["missing_shards"] or local["wrong_size_shards"]:
        blockers.append({"code": "CHECKPOINT_INCOMPLETE", "message":
                         f"{len(local['missing_shards'])} missing and "
                         f"{len(local['wrong_size_shards'])} wrong-size shards."})
    else:
        blockers.append({"code": "CHECKPOINT_CONTENT_UNVERIFIED", "message":
                         "All shard sizes match; full content/revision verification is still required."})
    if free < needed:
        blockers.append({"code": "INSUFFICIENT_DISK", "message":
                         f"Model volume needs {needed} free bytes; currently {free}."})
    if hardware.get("errors"):
        blockers.append({"code": "HARDWARE_PROBE_INCOMPLETE", "message": "; ".join(hardware["errors"])})
    gpu = next((g for g in hardware.get("gpus", []) if g["index"] == settings["gpu_index"]), None)
    if gpu is None:
        blockers.append({"code": "GPU_UNAVAILABLE", "message": "Configured NVIDIA GPU was not detected."})
    notes = [
        "GB uses 1,000,000,000 bytes; GiB uses 1,073,741,824 bytes.",
        "CPU quota applies to job processes per scheduler interval, not all applications combined.",
        "GPU target is time-average compute utilization, distinct from VRAM or power limits.",
        "Disk streaming may reduce resident weights; speed requires actual storage and inference measurements.",
    ]
    if gpu and gpu.get("compute_capability") not in (None, "N/A", "[N/A]"):
        try:
            if float(gpu["compute_capability"]) < 8.9:
                notes.append("GPU compute capability is below 8.9: original FP8 storage requires "
                             "a fallback compute path; native FP8 tensor-core inference is unavailable.")
        except ValueError:
            notes.append("GPU compute capability could not be parsed.")
    return {
        "status": "BLOCKED", "inference_verified": False,
        "model_id": snapshot["model_id"], "revision": snapshot["revision"],
        "model_directory": str(model_dir), "weight_bytes": snapshot["weight_bytes"],
        "architecture": snapshot["architecture"], "limits": settings,
        "hardware": hardware, "checkpoint": local,
        "storage": {"free_bytes": free, "required_free_bytes": needed,
                    "shortfall_bytes": max(0, needed - free)},
        "blockers": blockers, "notes": notes,
    }


def render_report(report):
    lines = ["# Local GLM feasibility report", "", f"Status: **{report['status']}**",
             "", "Full-model inference has not run or been verified.", "",
             f"Model: `{report['model_id']}`", f"Revision: `{report['revision']}`", "",
             f"Weights: {report['weight_bytes']/1e9:.2f} GB "
             f"({report['weight_bytes']/1024**3:.2f} GiB)",
             f"Model directory: `{report['model_directory']}`", "",
             "## Blocking conditions", ""]
    for item in report["blockers"]:
        lines.append(f"- **{item['code']}**: {item['message']}")
    lines.extend(["", "## Detected hardware", "",
                  f"CPU: {report['hardware'].get('cpu_name', 'unknown')}",
                  f"Logical processors: {report['hardware'].get('logical_processors', 'unknown')}"])
    ram = report["hardware"].get("physical_memory_bytes")
    if ram is not None:
        lines.append(f"Visible physical RAM: {ram/1024**3:.2f} GiB")
    for gpu in report["hardware"].get("gpus", []):
        vram = gpu.get("memory_total_bytes")
        lines.append(f"GPU {gpu['index']}: {gpu['name']}; VRAM: "
                     + (f"{vram/1024**3:.2f} GiB" if vram is not None else "unknown"))
    lines.extend(["", "| Volume | Free GB | Free GiB |", "|---|---:|---:|"])
    for disk in report["hardware"].get("disks", []):
        lines.append(f"| {disk['root']} | {disk['free_bytes']/1e9:.2f} | "
                     f"{disk['free_bytes']/1024**3:.2f} |")
    lines.extend(["", "## Limits and interpretation", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"
