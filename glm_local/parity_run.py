"""Run the unmodified official graph alongside the fixed invented native mini."""

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from . import __version__
from .audit import validate_settings
from .cpu_probe import NativeCpuBackend
from .cuda_probe import CudaTileBackend
from .gpu_gate import GpuBoundaryGate
from .mini_engine import MiniDecoder
from .mini_run import NativeMiniLinear, validate_mini, _argmax
from .mini_weights import write_mini_bundle
from .mini_storage import open_mini_storage
from .native_topk import NativeTopK
from .parity_compare import compare_step, HIDDEN_NAMES
from .process_metrics import sample_process, average_cpu_percent
from .reference_env import verify_reference_environment
from .winjob import JobLimits, run_local_process


def run_case(native, official, linear, prompt, generate, ram_budget_bytes):
    native.reset()
    official.reset()
    before = sample_process()
    calls_before = (linear.cpu_calls, linear.gpu_calls)
    compared = []
    native_logits = official_output = None

    def check(token):
        output = native.step(token)
        reference = official.step(token)
        compared.append(compare_step(output, native.trace[-1], reference))
        if sample_process().peak_working_set_bytes > ram_budget_bytes:
            raise RuntimeError("Observed parity worker RSS exceeded budget; aborting")
        return output, reference

    for token in prompt:
        native_logits, official_output = check(token)
    generated, teacher_next = [], []
    for _ in range(generate):
        token = _argmax(native_logits)
        generated.append(token)
        teacher_next.append(_argmax(official_output["logits"]))
        native_logits, official_output = check(token)
    # Truly independent official trajectory: regenerate from the original prompt.
    official.reset()
    for token in prompt:
        reference = official.step(token)
    reference_generated = []
    for _ in range(generate):
        token = _argmax(reference["logits"])
        reference_generated.append(token)
        reference = official.step(token)
    after = sample_process()
    if after.peak_working_set_bytes > ram_budget_bytes:
        raise RuntimeError("Observed official replay RSS exceeded budget; aborting")
    nodes = {name: max(c["hidden_states"]["nodes"][name]["max_absolute_error"] for c in compared)
             for name in HIDDEN_NAMES}
    failures = [c for c in compared if not c["passed"]]
    ids_match = generated == reference_generated
    return {"passed": not failures and ids_match, "prompt_length": len(prompt),
            "processed_tokens": len(compared), "generated_ids": generated,
            "official_generated_ids": reference_generated, "greedy_ids_match": ids_match,
            "teacher_forced_next_ids": teacher_next,
            "max_logit_error": max(c["logits"]["max_absolute_error"] for c in compared),
            "max_hidden_error": max(nodes.values()), "hidden_max_errors": nodes,
            "attention_selection_matches": all(c["selection"]["passed"] for c in compared),
            "expert_routing_matches": all(c["routing"]["passed"] for c in compared),
            "compared_values": len(compared) * (32 + 12 * 16),
            "failed_steps": len(failures), "first_failures": failures[:8],
            "cpu_projection_calls": linear.cpu_calls-calls_before[0],
            "gpu_projection_calls": linear.gpu_calls-calls_before[1],
            "cache_payload_bytes": native.cache_bytes,
            "cache_occupied_bytes": native.cache_used_bytes,
            "peak_worker_rss_bytes": after.peak_working_set_bytes,
            "peak_worker_private_commit_bytes": after.peak_private_commit_bytes,
            "worker_cpu_percent": average_cpu_percent(before, after)}


def execute_parity(settings, parameters, directory):
    validate_settings(settings)
    validate_mini(**parameters)
    provenance = verify_reference_environment()  # Verify before optional graph imports.
    from .official_reference import OfficialMiniReference
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = sample_process()
    directory = Path(directory)
    bundle = directory / "synthetic-model"
    manifest = write_mini_bundle(bundle, parameters["seed"])
    cases = []
    with ExitStack() as stack:
        weights, exported = open_mini_storage(
            stack, bundle, directory, parameters.get("storage", "private"))
        cpu = stack.enter_context(NativeCpuBackend())
        hybrid = parameters["backend"] == "hybrid"
        gpu = stack.enter_context(CudaTileBackend(settings["gpu_index"])) if hybrid else None
        gate = GpuBoundaryGate(device_index=settings["gpu_index"], target=settings["gpu_average_target"],
                               window_seconds=settings["gpu_window_seconds"], max_wait_seconds=10) if hybrid else None
        official = stack.enter_context(OfficialMiniReference(bundle))
        loaded_metadata = official.metadata
        expected_sources = provenance["transformers"]["source_sha256"]
        for label, filename in (("modeling", "modeling_glm_moe_dsa.py"),
                                ("configuration", "configuration_glm_moe_dsa.py")):
            expected = expected_sources["transformers/models/glm_moe_dsa/" + filename]
            if loaded_metadata["source_files"][label]["sha256"] != expected:
                raise RuntimeError("Imported official model source differs from the verified installation")
        linear = NativeMiniLinear(weights, cpu, gpu, gate)
        selector = NativeTopK()
        decoder = MiniDecoder(weights, linear, capture_states=True, attention_topk=selector)
        for length in parameters["lengths"]:
            prompt = [(parameters["seed"] + 7*i) % 32 for i in range(length)]
            cases.append(run_case(decoder, official, linear, prompt, parameters["generate"], settings["ram_budget_bytes"]))
            print(f"Official parity {length}+{parameters['generate']}: "
                  f"{'PASS' if cases[-1]['passed'] else 'MISMATCH'}", flush=True)
        if gate:
            gate.finish()
        metadata = official.metadata
        storage_stats = weights.stats()
        pacing = gate.summary() if gate else {"status": "not_used", "gpu_cap_verified": False}
        execution = {"cpu_projection_calls": linear.cpu_calls, "gpu_projection_calls": linear.gpu_calls,
                     "attention_selector": selector.metadata,
                     "cpu": cpu.metadata, "gpu": gpu.device_info if gpu else None,
                     "max_matrix_bytes": linear.max_matrix_bytes,
                     "peak_explicit_gpu_buffer_bytes": gpu.peak_explicit_device_bytes if gpu else 0}
    after = sample_process()
    if after.peak_working_set_bytes > settings["ram_budget_bytes"]:
        raise RuntimeError("Observed parity worker peak RSS exceeded budget")
    passed = all(case["passed"] for case in cases)
    return {"status": "PASS" if passed else "NUMERICAL_MISMATCH", "parameters": parameters,
            "scope": "fixed synthetic miniature, tolerance-based official FP32 comparison only",
            "synthetic_official_parity_verified": passed, "inference_verified": False,
            "real_checkpoint_compatible": False, "full_model_loaded": False,
            "bit_exact": False, "fixture_sha256": manifest["sha256"],
            "provenance": provenance, "official": metadata, "execution": execution,
            "storage": {"format": parameters.get("storage", "private"),
                        "reader_stats": storage_stats, "exported_fixture": exported,
                        "reference_storage": "independent original private fixture",
                        "reference_included_in_reader_stats": False},
            "cases": cases, "pacing": pacing,
            "resources": {"peak_worker_rss_bytes": after.peak_working_set_bytes,
                          "peak_worker_private_commit_bytes": after.peak_private_commit_bytes,
                          "worker_cpu_percent": average_cpu_percent(before, after),
                          "physical_ram_hard_cap_verified": False, "full_model_limits_verified": False},
            "precision": {"official": "CPU FP32 parameters, activations and eager attention",
                          "native": "FP32 C/CUDA projections; host nonlinear operations and cache float64",
                          "comparison": "per-element atol=2e-5, rtol=3e-4; not bitwise equality",
                          "selection": "set membership exact; expert weights aligned by ID; no tie mismatch waiver"},
            "limitations": [
                "Only invented 2-layer weights and IDs are tested; no real checkpoint or tokenizer was loaded.",
                "Official source is unmodified; hooks capture outputs without replacing graph operations.",
                "Precision differs at intermediate host operations; agreement is measured within declared tolerances.",
                "Top-k ordering is ignored, but different members fail even if caused by ties.",
                "The native parity path uses a bounded MSVC top-k compatibility selector; the legacy NumPy mini command keeps lower-index ties.",
                "Hybrid sends only the output head to CUDA; no production-scale placement or sustained GPU cap is established.",
                "Worker RSS includes the CPU Torch runtime and oracle; excludes other processes and OS file cache.",
                "Source verification covers versions, archive metadata and two model files, not every dependency byte.",
            ]}


def render_parity(report):
    lines = ["# Official Transformers miniature comparison", "", f"Status: **{report['status']}**", "",
             "**Real GLM checkpoint compatibility and resource limits remain unverified.**", ""]
    if "tool_version" in report:
        lines.extend([f"Tool version: {report['tool_version']}", ""])
    if "parameters" in report:
        lines.extend(["Run parameters: " + json.dumps(report["parameters"], sort_keys=True), ""])
    if "worker_environment" in report:
        lines.extend(["Worker environment: " + json.dumps(report["worker_environment"], sort_keys=True), ""])
    if "storage" in report:
        lines.extend([f"Native weight storage: {report['storage']['format']}", ""])
    if "error" in report:
        lines.append(f"Error: {report['error']}")
    if "traceback" in report:
        lines.extend(["", "## Worker traceback", "", "```text", report["traceback"].rstrip(), "```", ""])
    if "cases" in report:
        lines.extend(["| Prompt + generated | Max logit error | Max hidden error | Attention / experts | Greedy IDs |",
                      "|---|---:|---:|---|---|"])
        for case in report["cases"]:
            lines.append(f"| {case['prompt_length']} + {len(case['generated_ids'])} | {case['max_logit_error']:.9g} | "
                         f"{case['max_hidden_error']:.9g} | {case['attention_selection_matches']} / "
                         f"{case['expert_routing_matches']} | {case['greedy_ids_match']} |")
        r = report["resources"]
        lines.extend(["", f"Peak worker RSS: {r['peak_worker_rss_bytes']/1024**2:.2f} MiB",
                      f"Peak worker private commit: {r['peak_worker_private_commit_bytes']/1024**2:.2f} MiB",
                      "", "## Limits of the result", ""])
        lines.extend(f"- {x}" for x in report["limitations"])
    if "installed_job_policy" in report:
        lines.extend(["", "## Installed job policy", "", json.dumps(report["installed_job_policy"], indent=2)])
    return "\n".join(lines) + "\n"


def launch_parity(root, settings, backend="hybrid", lengths=(8, 32, 64), generate=4, seed=7, storage="private"):
    from .__main__ import save_json
    validate_settings(settings)
    parameters = dict(backend=backend, lengths=list(lengths), generate=generate, seed=seed, storage=storage)
    validate_mini(**parameters)
    root = Path(root).resolve()
    python = root / ".venv-reference" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError("Run setup-reference.bat first to create the pinned optional environment")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = root / "reports" / "parity" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    save_json(request, {"settings": settings, "parameters": parameters})
    captured = []
    print("Comparing fixed invented weights against pinned official Transformers, offline.", flush=True)
    try:
        code = run_local_process([str(python), "-m", "glm_local.parity_worker", str(request)], cwd=root,
                                 limits=JobLimits(settings["cpu_job_percent"], settings["ram_budget_bytes"]),
                                 on_policy=lambda value: captured.append(asdict(value)), timeout=180)
        result_path = directory / "result.json"
        if not result_path.is_file() or result_path.stat().st_size > 2 * 1024**2:
            raise RuntimeError(f"Parity worker exited {code} without a bounded report")
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if (code == 0) != (report.get("status") == "PASS"):
            raise RuntimeError("Parity worker exit disagrees with reported status")
    except Exception as error:
        code = 1
        report = {"status": "ERROR", "error": str(error), "inference_verified": False,
                  "synthetic_official_parity_verified": False, "parameters": parameters}
    report.setdefault("tool_version", __version__)
    report.update(parameters=parameters, installed_job_policy=captured[0] if captured else None,
                  job_policy_verified=bool(captured), child_exit_code=code,
                  run_directory=str(directory), checked_at=datetime.now(timezone.utc).isoformat())
    save_json(directory / "result.json", report)
    save_json(root / "reports" / "parity-latest.json", report)
    rendered = render_parity(report)
    (directory / "result.md").write_text(rendered, encoding="utf-8")
    (root / "reports" / "parity-latest.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Report: {root / 'reports' / 'parity-latest.md'}")
    return code
