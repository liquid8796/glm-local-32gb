"""Synthetic miniature decoder validation, not a production checkpoint path."""

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import uuid

from .audit import validate_settings
from .cpu_probe import NativeCpuBackend
from .cuda_probe import CudaTileBackend, MAX_OPERATIONS
from .gpu_gate import GpuBoundaryGate
from .mini_engine import MiniDecoder
from .mini_spec import SPEC, SOURCE_URL
from .mini_weights import MiniWeights, write_mini_bundle
from .process_metrics import sample_process, average_cpu_percent
from .winjob import JobLimits, run_local_process


def validate_mini(backend, lengths, generate, seed):
    if backend not in ("cpu", "hybrid"):
        raise ValueError("mini backend must be cpu or hybrid")
    if not isinstance(lengths, (list, tuple)) or not 1 <= len(lengths) <= 4:
        raise ValueError("Provide 1-4 synthetic sequence lengths")
    if type(generate) is not int or not 1 <= generate <= 8:
        raise ValueError("Generate count must be an integer in [1, 8]")
    if any(type(length) is not int or not 1 <= length <= 128 - generate for length in lengths):
        raise ValueError("Each prompt plus generated tokens must fit the 128-token synthetic bound")
    if list(lengths) != sorted(set(lengths)):
        raise ValueError("Synthetic sequence lengths must be unique and increasing")
    if backend == "hybrid" and sum(length + generate for length in lengths) > MAX_OPERATIONS:
        raise ValueError("Synthetic run exceeds the 256 GPU projection bound")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Synthetic seed must be uint32")


class NativeMiniLinear:
    """Demand-read each small matrix; hybrid offloads only the output head.

    CPU executes attention/FFN projections. The GPU computes the output head
    once per input token; this keeps the entire trial within 256 CUDA calls.
    Scalar nonlinear operations in MiniDecoder remain float64 on the host.
    """

    def __init__(self, weights, cpu, gpu=None, gate=None):
        if cpu is None or (gpu is not None and gate is None):
            raise ValueError("CPU backend and a gate for any GPU are required")
        self.weights, self.cpu, self.gpu, self.gate = weights, cpu, gpu, gate
        self.cpu_calls = self.gpu_calls = 0
        self.max_matrix_bytes = 0

    def __call__(self, name, vector):
        matrix = self.weights.matrix(name)
        self.max_matrix_bytes = max(self.max_matrix_bytes, len(matrix.weights))
        use_gpu = self.gpu is not None and name == "lm_head"
        if use_gpu:
            self.gate.before_submit()
        result = (self.gpu if use_gpu else self.cpu).matvec_tile(
            matrix.weights, matrix.rows, matrix.cols, vector, matrix.scale)
        if use_gpu:
            self.gpu_calls += 1
        else:
            self.cpu_calls += 1
        return result


def _argmax(values):
    if len(values) != SPEC["vocab"] or any(not math.isfinite(value) for value in values):
        raise ValueError("Synthetic logits must have 32 finite values")
    return max(range(len(values)), key=lambda index: (values[index], -index))


def compare_sequences(actual, expected):
    if len(actual) != len(expected) or not actual:
        raise ValueError("Expected equal non-empty logit sequences")
    max_error = 0.0
    failures = []
    for position, (left, right) in enumerate(zip(actual, expected)):
        if len(left) != 32 or len(right) != 32:
            raise ValueError("Each synthetic logit vector must have 32 values")
        for token, (a, b) in enumerate(zip(left, right)):
            finite = math.isfinite(a) and math.isfinite(b)
            error = abs(a - b) if finite else float("inf")
            max_error = max(max_error, error)
            if (not finite or error > 2e-5 + 3e-4 * abs(b)) and len(failures) < 8:
                failures.append({"position": position, "token_id": token,
                                 "actual": a if math.isfinite(a) else None,
                                 "expected": b if math.isfinite(b) else None})
    return {"passed": not failures, "positions": len(actual),
            "logits_compared": len(actual) * 32,
            "max_absolute_error": max_error if math.isfinite(max_error) else None,
            "atol": 2e-5, "rtol": 3e-4, "first_failures": failures}


def compare_traces(actual, expected):
    if len(actual) != len(expected):
        raise ValueError("Synthetic traces have unequal lengths")
    differences = []
    max_weight_error = 0.0
    for position, (a, b) in enumerate(zip(actual, expected)):
        for key in ("selected_indices", "routed_experts"):
            if a[key] != b[key] and len(differences) < 8:
                differences.append({"position": position, "field": key,
                                    "actual": a[key], "expected": b[key]})
        weights_a, weights_b = a["router_weights"], b["router_weights"]
        if len(weights_a) != 2 or len(weights_b) != 2:
            raise ValueError("Synthetic router must have two combination weights")
        for left, right in zip(weights_a, weights_b):
            finite = math.isfinite(left) and math.isfinite(right)
            error = abs(left - right) if finite else float("inf")
            max_weight_error = max(max_weight_error, error)
            if (not finite or error > 2e-5 + 3e-4 * abs(right)) and len(differences) < 8:
                differences.append({"position": position, "field": "router_weights"})
    return {"passed": not differences, "positions": len(actual),
            "max_router_weight_error": max_weight_error if math.isfinite(max_weight_error) else None,
            "first_differences": differences}


def evaluate_case(decoder, reference, prompt, generate, linear, ram_budget_bytes):
    decoder.reset()
    before = sample_process()
    native_before = (linear.cpu_calls, linear.gpu_calls)
    started = time.perf_counter()
    actual = []
    sampled_peaks = []

    def step(token):
        output = decoder.step(token)
        snapshot = sample_process()
        sampled_peaks.append(snapshot.peak_working_set_bytes)
        if snapshot.peak_working_set_bytes > ram_budget_bytes:
            raise RuntimeError("Observed worker peak RSS exceeded requested budget; stopping miniature")
        return output

    for token in prompt:
        actual.append(step(token))
    generated = []
    for _ in range(generate):
        token = _argmax(actual[-1])
        generated.append(token)
        actual.append(step(token))
    engine_seconds = time.perf_counter() - started
    engine_after = sample_process()
    full_tokens = list(prompt) + generated
    # Full-sequence teacher forcing and an independent greedy trajectory.
    expected = reference.forward(full_tokens)
    reference_tokens = list(prompt)
    expected_generated = []
    for _ in range(generate):
        token = _argmax(reference.forward(reference_tokens)["logits"][-1])
        expected_generated.append(token)
        reference_tokens.append(token)
    numeric = compare_sequences(actual, expected["logits"])
    traces = compare_traces(decoder.trace, expected["traces"])
    after = sample_process()
    if after.peak_working_set_bytes > ram_budget_bytes:
        raise RuntimeError("Observed reference/worker peak RSS exceeded requested budget; stopping miniature")
    generated_match = generated == expected_generated
    return {
        "passed": numeric["passed"] and traces["passed"] and generated_match,
        "prompt_length": len(prompt), "processed_tokens": len(full_tokens),
        "generated_token_ids": generated, "reference_generated_token_ids": expected_generated,
        "greedy_tokens_match": generated_match, "logits": numeric, "selection_traces": traces,
        "cpu_projection_calls": linear.cpu_calls - native_before[0],
        "gpu_projection_calls": linear.gpu_calls - native_before[1],
        "engine_seconds_including_pacing": engine_seconds,
        "engine_worker_cpu_percent": average_cpu_percent(before, engine_after),
        "peak_worker_rss_bytes": after.peak_working_set_bytes,
        "peak_worker_private_commit_bytes": after.peak_private_commit_bytes,
        "max_token_boundary_peak_rss_bytes": max(sampled_peaks),
        "cache_allocated_payload_bytes": decoder.cache_bytes,
        "cache_occupied_payload_bytes": decoder.cache_used_bytes,
    }


def execute_mini(settings, parameters, directory):
    validate_settings(settings)
    validate_mini(**parameters)
    # Optional dependency belongs only to this synthetic validation command.
    try:
        from .mini_reference import ReferenceDecoder
        import numpy
    except ImportError as error:
        raise RuntimeError("Miniature reference validation requires NumPy; see README optional validation setup") from error
    before = sample_process()
    directory = Path(directory)
    bundle = directory / "synthetic-model"
    manifest = write_mini_bundle(bundle, parameters["seed"])
    reference = ReferenceDecoder(bundle)
    cases = []
    with ExitStack() as stack:
        weights = stack.enter_context(MiniWeights(bundle))
        cpu = stack.enter_context(NativeCpuBackend())
        hybrid = parameters["backend"] == "hybrid"
        gpu = stack.enter_context(CudaTileBackend(settings["gpu_index"])) if hybrid else None
        gate = GpuBoundaryGate(device_index=settings["gpu_index"],
                               target=settings["gpu_average_target"],
                               window_seconds=settings["gpu_window_seconds"],
                               max_wait_seconds=10) if hybrid else None
        linear = NativeMiniLinear(weights, cpu, gpu, gate)
        decoder = MiniDecoder(weights, linear)
        for length in parameters["lengths"]:
            prompt = [(parameters["seed"] + 7 * index) % 32 for index in range(length)]
            cases.append(evaluate_case(decoder, reference, prompt, parameters["generate"],
                                       linear, settings["ram_budget_bytes"]))
            print(f"Miniature {length}+{parameters['generate']} tokens: "
                  f"{'PASS' if cases[-1]['passed'] else 'MISMATCH'}", flush=True)
        if gate:
            gate.finish()
        execution = {"cpu": cpu.metadata, "gpu": gpu.device_info if gpu else None,
                     "cpu_projection_calls": linear.cpu_calls,
                     "gpu_projection_calls": linear.gpu_calls,
                     "max_matrix_bytes": linear.max_matrix_bytes,
                     "peak_explicit_gpu_buffer_bytes": gpu.peak_explicit_device_bytes if gpu else 0}
        storage_stats = weights.stats()
        pacing = gate.summary() if gate else {"status": "not_used", "gpu_cap_verified": False}
    after = sample_process()
    passed = all(case["passed"] for case in cases)
    return {
        "status": "PASS" if passed else "NUMERICAL_MISMATCH",
        "scope": "fixed synthetic two-layer decoder only",
        "synthetic_decoder_verified": passed, "inference_verified": False,
        "full_model_loaded": False, "checkpoint_compatible": False,
        "official_transformers_parity_verified": False, "job_policy_verified": False,
        "parameters": parameters, "synthetic_spec": SPEC,
        "fixture_weight_bytes": manifest["weight_bytes"], "fixture_sha256": manifest["sha256"],
        "reference": {"kind": "independent full-sequence NumPy float64",
                      "numpy_version": numpy.__version__, "equation_source": SOURCE_URL},
        "cases": cases, "execution": execution,
        "storage": {"scope": "incremental MiniWeights reader only", "reader_stats": storage_stats,
                    "reference_included": False,
                    "reference_read_bound_bytes": 65537,
                    "reference_resident_weights": "all tiny matrices decoded to NumPy float64"},
        "pacing": pacing,
        "resources": {"before": asdict(before), "after": asdict(after),
                      "worker_average_cpu_percent": average_cpu_percent(before, after),
                      "peak_worker_rss_bytes": after.peak_working_set_bytes,
                      "peak_worker_private_commit_bytes": after.peak_private_commit_bytes,
                      "physical_ram_hard_cap_verified": False, "full_model_limits_verified": False},
        "limitations": [
            "Weights and token vocabulary are invented; generated IDs have no language meaning.",
            "This fixed two-layer graph is not the 78-layer checkpoint and cannot open real model weights.",
            "Independent NumPy agreement does not prove official Transformers parity or checkpoint compatibility.",
            "Projections compute FP32; scalar nonlinear math and cache arrays are float64, unlike a full dtype-faithful runtime.",
            "Cache payload excludes Python overhead, weights, activations, trace lists and driver/context memory.",
            "Worker lifetime RSS/commit counters exclude other processes and OS file cache; abort checks occur after token steps.",
            "Storage counters describe only the incremental reader; the reference loads all bounded synthetic matrices separately.",
            "Hybrid sends only the output head to CUDA; this is not optimized parallel CPU/GPU inference.",
            "GPU telemetry includes other apps and may miss tiny kernels; no sustained-utilization or production speed claim.",
            "Equal top-k scores use lower indices first; this deterministic rule is not guaranteed to match torch.topk ties.",
        ],
    }


def render_mini_report(report):
    lines = ["# Synthetic miniature decoder validation", "", f"Status: **{report['status']}**", "",
             "**Real GLM checkpoint inference remains unverified.**", ""]
    if "error" in report:
        lines.append(f"Error: {report['error']}")
    if "cases" in report:
        lines.extend(["| Prompt + generated | Logit max error | Greedy IDs match | Selection match | Occupied cache bytes |",
                      "|---|---:|---|---|---:|"])
        for case in report["cases"]:
            lines.append(f"| {case['prompt_length']} + {len(case['generated_token_ids'])} | "
                         f"{case['logits']['max_absolute_error']} | {case['greedy_tokens_match']} | "
                         f"{case['selection_traces']['passed']} | {case['cache_occupied_payload_bytes']} |")
        resources = report["resources"]
        lines.extend(["", f"FP8 fixture weights: {report['fixture_weight_bytes']} bytes",
                      f"CPU projections: {report['execution']['cpu_projection_calls']}",
                      f"GPU projections: {report['execution']['gpu_projection_calls']}",
                      f"Peak worker RSS: {resources['peak_worker_rss_bytes']/1024**2:.2f} MiB",
                      f"Peak worker private commit: {resources['peak_worker_private_commit_bytes']/1024**2:.2f} MiB",
                      "", "## Interpretation", ""])
        lines.extend(f"- {item}" for item in report["limitations"])
    if "installed_job_policy" in report:
        lines.extend(["", "## Installed job policy", "", json.dumps(report["installed_job_policy"], indent=2)])
    return "\n".join(lines) + "\n"


def launch_mini(root, settings, backend="hybrid", lengths=(8, 32, 64), generate=4, seed=7):
    from .__main__ import save_json
    validate_settings(settings)
    parameters = dict(backend=backend, lengths=list(lengths), generate=generate, seed=seed)
    validate_mini(**parameters)
    root = Path(root).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = root / "reports" / "mini" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    save_json(request, {"settings": settings, "parameters": parameters})
    captured = []
    print("Running fixed synthetic decoder validation; no checkpoint or text tokenizer is loaded.", flush=True)
    try:
        code = run_local_process(
            [sys.executable, "-m", "glm_local.mini_worker", str(request)], cwd=root,
            limits=JobLimits(settings["cpu_job_percent"], settings["ram_budget_bytes"]),
            on_policy=lambda value: captured.append(asdict(value)), timeout=180)
        path = directory / "result.json"
        if not path.is_file() or path.stat().st_size > 2 * 1024**2:
            raise RuntimeError(f"Miniature worker exited {code} without a bounded report")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (code == 0) != (report.get("status") == "PASS"):
            raise RuntimeError("Miniature exit code disagrees with reported status")
    except Exception as error:
        code = 1
        report = {"status": "ERROR", "error": str(error), "inference_verified": False,
                  "full_model_loaded": False, "parameters": parameters}
    report.update(installed_job_policy=captured[0] if captured else None,
                  job_policy_verified=bool(captured), child_exit_code=code,
                  run_directory=str(directory), checked_at=datetime.now(timezone.utc).isoformat())
    save_json(directory / "result.json", report)
    save_json(root / "reports" / "mini-latest.json", report)
    rendered = render_mini_report(report)
    (directory / "result.md").write_text(rendered, encoding="utf-8")
    (root / "reports" / "mini-latest.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Report: {root / 'reports' / 'mini-latest.md'}")
    return code
