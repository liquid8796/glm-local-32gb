"""Run with python -m glm_local from the project directory."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from .audit import evaluate, render_report, validate_settings
from .hardware import detect_hardware, read_gpus
from .metadata import refresh_metadata
from .pacing import CooperativeGpuPacer, PacingConfig
from .winjob import JobLimits, run_local_process

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def doctor(settings, refresh):
    cache = ROOT / "reports" / "model-metadata.json"
    if refresh:
        print("Fetching public metadata for pinned revision; no model weights are downloaded.")
        snapshot = refresh_metadata(settings["model_id"], settings["revision"])
        save_json(cache, snapshot)
    else:
        source = cache if cache.is_file() else ROOT / "docs" / "model-metadata.json"
        if not source.is_file():
            raise ValueError("No metadata snapshot exists. Run: glm.bat doctor --refresh")
        snapshot = read_json(source)
    report = evaluate(settings, snapshot, detect_hardware(), ROOT)
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    save_json(ROOT / "reports" / "latest.json", report)
    rendered = render_report(report)
    (ROOT / "reports" / "latest.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Saved: {ROOT / 'reports' / 'latest.md'}")
    return 2  # Readiness failed, even when the diagnostic itself succeeded.


def policy_check(settings):
    captured = []
    limits = JobLimits(cpu_percent=settings["cpu_job_percent"],
                       committed_memory_bytes=settings["ram_budget_bytes"])
    code = run_local_process(
        [sys.executable, "-I", "-c", "print('Local child process completed.')"],
        limits=limits, on_policy=lambda policy: captured.append(asdict(policy)), timeout=10,
    )
    if not captured:
        raise RuntimeError("No installed policy was returned")
    report = {"installed": captured[0], "child_exit_code": code,
              "validation": "OS policy readback and small child process only; no model inference",
              "physical_ram_cap_verified": False, "gpu_cap_verified": False}
    save_json(ROOT / "reports" / "policy-check.json", report)
    print(json.dumps(report, indent=2))
    return code


def monitor(settings, seconds):
    if not 1 <= seconds <= 60:
        raise ValueError("Monitor duration must be 1-60 seconds")
    pacer = CooperativeGpuPacer(PacingConfig(
        target_utilization=settings["gpu_average_target"],
        window_seconds=settings["gpu_window_seconds"],
        warmup_seconds=min(2, settings["gpu_window_seconds"]),
    ))
    samples = []
    start = time.monotonic()
    print("Read-only GPU observation; no inference or GPU control is active.")
    while time.monotonic() - start < seconds:
        gpus = read_gpus()
        gpu = next((item for item in gpus if item["index"] == settings["gpu_index"]), None)
        value = None if gpu is None else gpu["utilization_percent"]
        now = time.monotonic()
        pacer.add_sample(now, None if value is None else value / 100)
        decision = asdict(pacer.recommend(now))
        sample = {"elapsed_seconds": round(now - start, 3), "gpu_percent": value,
                  "pacing_recommendation": decision}
        samples.append(sample)
        print(f"{sample['elapsed_seconds']:5.1f}s GPU={value}% recommendation={decision['status']}")
        remaining = seconds - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(min(0.5, remaining))
    save_json(ROOT / "reports" / "gpu-observation.json", {
        "inference_running": False, "control_active": False,
        "utilization_target_verified": False, "samples": samples,
    })
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="GLM feasibility and resource-control foundation; no inference backend")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "local.json")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("doctor", help="Check metadata, hardware and blocking conditions")
    probe.add_argument("--refresh", action="store_true", help="Refresh pinned metadata only")
    sub.add_parser("policy-check", help="Read back Windows CPU/commit quota and run a small child")
    observe = sub.add_parser("monitor", help="Observe GPU and pacing recommendations without controlling GPU")
    observe.add_argument("--seconds", type=float, default=10)
    experiment = sub.add_parser("probe", help="Run a bounded synthetic FP8 CPU/GPU experiment under a Windows job")
    experiment.add_argument("--backend", choices=("cpu", "gpu", "hybrid"), default="hybrid")
    experiment.add_argument("--rows", type=int, default=384)
    experiment.add_argument("--cols", type=int, default=384)
    experiment.add_argument("--iterations", type=int, default=3)
    experiment.add_argument("--seed", type=int, default=7)
    miniature = sub.add_parser("mini", help="Validate a fixed synthetic two-layer decoder, not a real checkpoint")
    miniature.add_argument("--backend", choices=("cpu", "hybrid"), default="hybrid")
    miniature.add_argument("--lengths", default="8,32,64", help="1-4 increasing prompt lengths, comma separated")
    miniature.add_argument("--generate", type=int, default=4)
    miniature.add_argument("--seed", type=int, default=7)
    parity = sub.add_parser("parity", help="Compare the fixed miniature with pinned offline Transformers")
    parity.add_argument("--backend", choices=("cpu", "hybrid"), default="hybrid")
    parity.add_argument("--lengths", default="8,32,64")
    parity.add_argument("--generate", type=int, default=4)
    parity.add_argument("--seed", type=int, default=7)
    storage = sub.add_parser("storage-check", help="Validate bounded safetensors and FP8 scales using only small synthetic files")
    storage.add_argument("--backend", choices=("cpu", "hybrid"), default="hybrid")
    storage.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        settings = read_json(args.config)
        validate_settings(settings)
        if args.command == "doctor":
            return doctor(settings, args.refresh)
        if args.command == "policy-check":
            return policy_check(settings)
        if args.command == "probe":
            from .backend_probe import launch_probe
            return launch_probe(ROOT, settings, args.backend, args.rows, args.cols, args.iterations, args.seed)
        if args.command == "mini":
            from .mini_run import launch_mini
            lengths = [int(value) for value in args.lengths.split(",")]
            return launch_mini(ROOT, settings, args.backend, lengths, args.generate, args.seed)
        if args.command == "parity":
            from .parity_run import launch_parity
            return launch_parity(ROOT, settings, args.backend, [int(v) for v in args.lengths.split(",")], args.generate, args.seed)
        if args.command == "storage-check":
            from .safetensor_check import launch_check
            return launch_check(ROOT, settings, args.backend, args.seed)
        return monitor(settings, args.seconds)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
