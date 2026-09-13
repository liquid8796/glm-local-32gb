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
from .model_profiles import (PROFILE_NAMES, load_profile, metadata_snapshot_path,
                             reports_directory)
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
    report_dir = reports_directory(ROOT, settings)
    cache = report_dir / "model-metadata.json"
    if refresh:
        print("Fetching public metadata for pinned revision; no model weights are downloaded.")
        snapshot = refresh_metadata(settings["model_id"], settings["revision"])
        save_json(cache, snapshot)
    else:
        source = cache if cache.is_file() else metadata_snapshot_path(ROOT, settings)
        if not source.is_file():
            raise ValueError("No metadata snapshot exists. Run: glm.bat doctor --refresh")
        snapshot = read_json(source)
    report = evaluate(settings, snapshot, detect_hardware(), ROOT)
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    save_json(report_dir / "latest.json", report)
    rendered = render_report(report)
    (report_dir / "latest.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Saved: {report_dir / 'latest.md'}")
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
    save_json(reports_directory(ROOT, settings) / "policy-check.json", report)
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
    save_json(reports_directory(ROOT, settings) / "gpu-observation.json", {
        "inference_running": False, "control_active": False,
        "utilization_target_verified": False, "samples": samples,
    })
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="GLM feasibility and resource-control foundation; no inference backend")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config", type=Path, help="Settings JSON; defaults to config/local.json")
    selection.add_argument("--profile", choices=PROFILE_NAMES, help="Select a bundled pinned checkpoint")
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
    miniature.add_argument("--storage", choices=("private", "safetensors"), default="private",
                           help="Private fixture or four-shard safetensors; synthetic only")
    parity = sub.add_parser("parity", help="Compare the fixed miniature with pinned offline Transformers")
    parity.add_argument("--backend", choices=("cpu", "hybrid"), default="hybrid")
    parity.add_argument("--lengths", default="8,32,64")
    parity.add_argument("--generate", type=int, default=4)
    parity.add_argument("--seed", type=int, default=7)
    parity.add_argument("--storage", choices=("private", "safetensors"), default="private",
                        help="Native-side storage; official oracle keeps the independent private fixture")
    storage = sub.add_parser("storage-check", help="Validate bounded safetensors and FP8 scales using only small synthetic files")
    storage.add_argument("--backend", choices=("cpu", "hybrid"), default="hybrid")
    storage.add_argument("--seed", type=int, default=7)
    metadata = sub.add_parser("metadata-check", help="Inspect pinned config/index/shard headers only; never tensor payload")
    metadata.add_argument("--max-shards", type=int, default=512,
                          help="Maximum headers to inspect; partial coverage cannot PASS")
    metadata.add_argument("--budget-mib", type=int, default=64,
                          help="Aggregate application metadata body read budget, 1..128 MiB")
    evidence = metadata.add_mutually_exclusive_group()
    evidence.add_argument("--offline", type=Path,
                          help="Replay an evidence directory without network access")
    evidence.add_argument("--resume", type=Path,
                          help="Revalidate saved evidence and fetch only missing pinned shard headers")
    sub.add_parser("architecture-check", help="Map GLM architecture from metadata catalogue only")
    runtime_plan = sub.add_parser("runtime-plan", help="Estimate streamed checkpoint cache/CPU/GPU budgets from verified metadata")
    runtime_plan.add_argument("--backend", choices=("cpu", "hybrid"), default="cpu")
    runtime_plan.add_argument("--context", type=int, default=4096)
    runtime_plan.add_argument("--generate", type=int, default=32)
    runtime_plan.add_argument("--vram-budget-mib", type=int)
    projection = sub.add_parser("projection-check", help="Verify one named real FP8 projection with bounded CPU/GPU execution")
    projection.add_argument("--tensor", default=None,
                            help="Named projection; defaults to a suitable tensor for the checkpoint profile")
    projection.add_argument("--backend", choices=("cpu", "hybrid"), default="hybrid")
    projection.add_argument("--online", action="store_true", help="Download only the selected weight/scale slices, within --budget-mib")
    projection.add_argument("--budget-mib", type=int, default=64)
    projection.add_argument("--model-directory", type=Path)
    projection.add_argument("--seed", type=int, default=7)
    generate = sub.add_parser("generate", help="Experimental token generation from complete local pinned shards; full parity is unverified")
    prompt = generate.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--tokens", help="Comma-separated integer token IDs")
    prompt.add_argument("--prompt", help="Text prompt; requires a verified local tokenizer")
    prompt.add_argument("--messages-file", type=Path, help="Bounded JSON array of text chat messages; implies chat format")
    generate.add_argument("--prompt-format", choices=("raw", "chat"), default=None,
                          help="Raw completion (default) or verified pinned text chat formatting")
    generate.add_argument("--reasoning-effort", choices=("low", "high", "max"), default=None,
                          help="Chat reasoning effort; template default is max (thinking remains enabled)")
    generate.add_argument("--keep-thinking", action="store_true", help="Retain prior assistant reasoning in chat history")
    generate.add_argument("--direct-answer", action="store_true",
                          help="Chat only: explicitly close the assistant thinking prefix; model may still reopen thinking")
    generate.add_argument("--stream-events", action="store_true", help="Emit bounded MODELDESK_EVENT JSON response updates")
    generate.add_argument("--model-directory", type=Path)
    generate.add_argument("--backend", choices=("cpu", "hybrid"), default="cpu")
    generate.add_argument("--context", type=int, default=4096)
    generate.add_argument("--generate", type=int, default=32)
    generate.add_argument("--timeout", type=int, default=1800)
    generate.add_argument("--prefill-batch-size", type=int, choices=range(1, 17), default=16,
                          help="Bounded layer-wise prompt batch; 1 keeps scalar reference prefill")
    tokenizer = sub.add_parser("tokenizer-check", help="Verify the pinned native tokenizer; optional bounded download, no model weights")
    tokenizer.add_argument("--online", action="store_true")
    tokenizer.add_argument("--model-directory", type=Path)
    args = parser.parse_args(argv)
    try:
        settings = (load_profile(ROOT, args.profile) if args.profile is not None else
                    read_json(args.config or ROOT / "config" / "local.json"))
        validate_settings(settings)
        if args.command in ("runtime-plan", "projection-check", "generate", "tokenizer-check"):
            from .runtime_commands import launch_runtime
            parameters = {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items() if key not in ("command", "config", "profile")}
            if args.command == "runtime-plan":
                value = parameters.pop("vram_budget_mib")
                if value is not None:
                    if not 1 <= value <= 1024**2:
                        raise ValueError("VRAM budget must be 1..1048576 MiB")
                    parameters["vram_budget_bytes"] = value * 1024**2
            if args.command == "projection-check" and not 1 <= args.budget_mib <= 128:
                raise ValueError("Projection budget must be 1..128 MiB")
            if args.command == "generate":
                if not 1 <= args.timeout <= 86400:
                    raise ValueError("Generation timeout must be 1..86400 seconds")
                if args.tokens is not None:
                    if args.prompt_format == "chat" or args.reasoning_effort is not None or args.keep_thinking or args.stream_events or args.direct_answer:
                        raise ValueError("Token-ID generation does not accept chat or decoded-text streaming options")
                    if len(args.tokens) > 1024 * 1024:
                        raise ValueError("Token argument exceeds 1 MiB")
                    parameters["tokens"] = [int(value) for value in args.tokens.split(",")]
                if args.prompt is not None and len(args.prompt.encode("utf-8")) > 1024 * 1024:
                    raise ValueError("Prompt exceeds 1 MiB")
                if args.messages_file is not None and args.prompt_format == "raw":
                    raise ValueError("--messages-file requires chat format")
                parameters["prompt_format"] = args.prompt_format or ("chat" if args.messages_file is not None else "raw")
                if parameters["prompt_format"] != "chat" and (args.reasoning_effort is not None or args.keep_thinking or args.direct_answer):
                    raise ValueError("Reasoning effort, thinking history and direct-answer options require chat format")
                parameters["reasoning_effort"] = args.reasoning_effort or ("low" if args.direct_answer else "max")
            action = {"runtime-plan": "plan", "projection-check": "projection", "generate": "generate", "tokenizer-check": "tokenizer"}[args.command]
            return launch_runtime(ROOT, settings, action, parameters)
        if args.command == "metadata-check":
            from .checkpoint_check import launch_metadata
            if args.resume is not None:
                return launch_metadata(ROOT, settings, args.max_shards, args.budget_mib,
                                       args.offline, resume=args.resume)
            return launch_metadata(ROOT, settings, args.max_shards, args.budget_mib, args.offline)
        if args.command == "architecture-check":
            from .architecture.report import EXIT_CODES, run_architecture
            result = run_architecture(ROOT, settings)
            print(json.dumps(result, indent=2))
            return EXIT_CODES[result["status"]]
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
            return launch_mini(ROOT, settings, args.backend, lengths, args.generate, args.seed, args.storage)
        if args.command == "parity":
            from .parity_run import launch_parity
            return launch_parity(ROOT, settings, args.backend, [int(v) for v in args.lengths.split(",")], args.generate, args.seed, args.storage)
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
