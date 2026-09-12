"""Windows-job launchers for checkpoint planning and bounded runtime checks."""

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import uuid

from . import __version__
from .architecture.report import _analyze_source, _bounded_json, MAX_REPORT_BYTES
from .audit import validate_settings
from .checkpoint_snapshot import write_json
from .execution import (build_projection_descriptor, execute_projection,
                        reference_projection, compare_projection, FULL_MODEL_FLAGS)
from .hardware import read_gpus
from .residency import PlannerSettings, build_plan
from .winjob import JobLimits, run_local_process
from .model_profiles import reports_directory
from .runtime_progress import RuntimeProgress, load_last_progress

EXIT_CODES = {"PASS": 0, "ESTIMATE_FITS": 0, "GENERATED_UNVERIFIED": 0,
              "ERROR": 1, "NUMERICAL_MISMATCH": 2, "INTERRUPTED": 130}


def verified_source(root, settings):
    path = reports_directory(root, settings) / "metadata-latest.json"
    analysis = _analyze_source(Path(root), settings, path)
    if analysis["status"] != "PASS":
        raise ValueError("Complete architecture metadata must PASS before runtime planning or generation")
    source, digest, _ = _bounded_json(path, MAX_REPORT_BYTES, capture=True)
    if digest != analysis["source_report"]:
        raise ValueError("Metadata source changed before runtime planning")
    return source, analysis


def make_plan(config, settings, parameters):
    backend = parameters.get("backend", "cpu")
    vram = parameters.get("vram_budget_bytes")
    if backend == "hybrid" and vram is None:
        gpu = next((gpu for gpu in read_gpus() if gpu["index"] == settings["gpu_index"]), None)
        if (gpu is None or type(gpu.get("memory_free_bytes")) not in (int, float)
                or not math.isfinite(gpu["memory_free_bytes"]) or gpu["memory_free_bytes"] <= 0):
            raise ValueError("A current NVIDIA free-VRAM measurement is required for hybrid planning")
        vram = int(gpu["memory_free_bytes"])
    return build_plan(config, PlannerSettings(
        ram_budget_bytes=settings["ram_budget_bytes"], vram_budget_bytes=vram or 0,
        device=backend, context_tokens=parameters.get("context", 4096),
        max_new_tokens=parameters.get("generate", 32)),
        model_id=settings["model_id"], revision=settings["revision"])


def _kernels(stack, settings, backend, *, operation_budget=65536, quant_format="fp8"):
    from .cpu_probe import NativeCpuBackend
    from .cuda_probe import CudaTileBackend
    from .gpu_gate import GpuBoundaryGate
    cpu_type, gpu_type = NativeCpuBackend, CudaTileBackend
    if quant_format == "nvfp4":
        from .nvfp4_kernels import NativeNVFP4CpuBackend, CudaNVFP4TileBackend
        cpu_type, gpu_type = NativeNVFP4CpuBackend, CudaNVFP4TileBackend
    cpu = stack.enter_context(cpu_type())
    gpu = gate = None
    if backend == "hybrid":
        gpu = stack.enter_context(gpu_type(settings["gpu_index"], max_operations=operation_budget))
        gate = GpuBoundaryGate(device_index=settings["gpu_index"], target=settings["gpu_average_target"],
                              window_seconds=settings["gpu_window_seconds"], max_wait_seconds=10)
    return cpu, gpu, gate


def execute_runtime(root, settings, action, parameters, directory):
    root, directory = Path(root), Path(directory)
    validate_settings(settings)
    result = {"status": "ERROR", "action": action, "tool_version": __version__,
              "model_id": settings["model_id"], "revision": settings["revision"],
              "parameters": parameters, "scope": "experimental_checkpoint_runtime",
              **FULL_MODEL_FLAGS}
    progress = RuntimeProgress(directory, settings, action)
    try:
        if action == "plan":
            progress("metadata_validation")
            source, analysis = verified_source(root, settings)
            progress("memory_planning")
            plan = make_plan(source["config"], settings, parameters)
            result.update(status="ESTIMATE_FITS", plan=plan.to_dict(),
                          source_report=analysis["source_report"], catalogue=analysis["catalogue"])
        elif action == "projection":
            progress("projection_metadata_validation")
            if type(parameters.get("seed", 7)) is not int or not 0 <= parameters.get("seed", 7) <= 0xFFFFFFFF:
                raise ValueError("Projection seed must be uint32")
            tensor = parameters.get("tensor")
            if tensor is None:
                from .checkpoint_schema import quantization_format
                source = _bounded_json(reports_directory(root, settings) / "metadata-latest.json", MAX_REPORT_BYTES)
                tensor = ("model.layers.3.mlp.experts.0.gate_proj.weight"
                          if quantization_format(source.get("config")) == "nvfp4"
                          else "model.layers.0.self_attn.q_a_proj.weight")
            descriptor = build_projection_descriptor(root, settings, tensor)
            result["descriptor"] = descriptor.to_dict()
            quant_format = descriptor.quant_format
            rows, cols = descriptor.logical_shape
            from .execution import MAX_REFERENCE_ELEMENTS
            if rows * cols > MAX_REFERENCE_ELEMENTS:
                raise ValueError("Selected projection is too large for the bounded scalar reference")
            from .residency import ReservationLedger
            from .catalogue_reader import SelectedCatalogueReader
            from .projection_remote import RemoteProjectionReader
            from .process_metrics import sample_process
            before = sample_process() if sys.platform == "win32" else None
            backend = parameters.get("backend", "cpu")
            # A projection has no decoder KV cache. Runtime headroom remains reserved.
            vram = 0
            if backend == "hybrid":
                gpu_info = next((g for g in read_gpus() if g["index"] == settings["gpu_index"]), None)
                if gpu_info is None:
                    raise ValueError("Configured NVIDIA GPU is unavailable")
                value = gpu_info.get("memory_free_bytes")
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Current free VRAM could not be measured")
                vram = int(value)
            ledger = ReservationLedger(settings["ram_budget_bytes"], vram,
                       base_cpu_bytes=2 * 1024**3, base_cuda_bytes=256 * 1024**2 if vram else 0)
            generator = random.Random(parameters.get("seed", 7))
            vector = [generator.uniform(-0.25, 0.25) for _ in range(cols)]
            with ExitStack() as stack:
                progress("projection_weights_opening", tensor_name=tensor, rows=rows, cols=cols)
                if parameters.get("online", False):
                    reader = stack.enter_context(RemoteProjectionReader(descriptor, directory / "payload",
                        budget_bytes=parameters.get("budget_mib", 64) * 1024**2))
                else:
                    model_dir = Path(parameters.get("model_directory") or settings["model_directory"])
                    if not model_dir.is_absolute():
                        model_dir = root / model_dir
                    reader = stack.enter_context(SelectedCatalogueReader(model_dir, descriptor))
                progress("projection_kernel_initializing", backend=backend)
                cpu, gpu, gate = _kernels(stack, settings, backend, quant_format=quant_format)
                execute, reference = execute_projection, reference_projection
                if quant_format == "nvfp4":
                    from .nvfp4_execution import execute_nvfp4_projection, reference_nvfp4_projection
                    execute, reference = execute_nvfp4_projection, reference_nvfp4_projection
                progress("projection_execution")
                actual, counters = execute(reader, descriptor, vector, backend=backend,
                                                      cpu=cpu, gpu=gpu, gate=gate, ledger=ledger)
                progress("projection_reference")
                expected = reference(reader, descriptor, vector)
                progress("projection_comparison")
                comparison = compare_projection(actual, expected)
                if quant_format == "nvfp4":
                    comparison.update(reference="independent row-wise E2M1/E4M3/F32 weight dequantization and FP32 matvec",
                                      activation_quantization="none", native_w4a4_parity_verified=False)
                result.update(status="PASS" if comparison["passed"] else "NUMERICAL_MISMATCH",
                              selected_projection_verified=comparison["passed"], comparison=comparison,
                              execution=counters, reader=reader.stats(), allocations=ledger.snapshot())
                if gate:
                    progress("gpu_finalization")
                    gate.finish()
                    result["pacing"] = gate.summary()
            if before:
                progress("resource_validation")
                after = sample_process()
                result["process"] = asdict(after)
                if max(after.peak_working_set_bytes, after.peak_private_commit_bytes) > settings["ram_budget_bytes"]:
                    raise RuntimeError("Observed projection worker memory exceeded the configured budget")
        elif action == "generate":
            result.update(_generate(root, settings, parameters, directory, progress=progress))
        elif action == "tokenizer":
            progress("tokenizer_preparing")
            from .tokenizer import prepare_tokenizer
            result.update(prepare_tokenizer(root, settings, parameters.get("model_directory"),
                                            online=parameters.get("online", False)))
        else:
            raise ValueError("Unknown runtime action")
    except KeyboardInterrupt:
        result.update(status="INTERRUPTED", error="Runtime command interrupted")
    except Exception as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {error}"[:2000],
                      traceback=traceback.format_exc(limit=12, chain=False)[-16000:])
    finally:
        result["last_progress"] = progress.finish(result["status"])
        result["elapsed_seconds"] = result["last_progress"]["elapsed_seconds"]
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    return result, EXIT_CODES[result["status"]]


def _generate(root, settings, parameters, directory, *, progress=None):
    from .runtime_weights import RuntimeWeights
    from .streaming_decoder import StreamingDecoder
    from .tokenizer import load_tokenizer, prepare_tokenizer
    report = progress if progress is not None else lambda *args, **kwargs: None
    report("metadata_validation")
    source, analysis = verified_source(root, settings)
    report("memory_planning")
    plan = make_plan(source["config"], settings, parameters)
    model_dir = Path(parameters.get("model_directory") or settings["model_directory"])
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    tokens = parameters.get("tokens")
    tokenizer = None
    if tokens is None:
        report("tokenizer_preparing")
        prepared = prepare_tokenizer(root, settings, model_dir, online=False)
        report("tokenizer_loading")
        tokenizer = load_tokenizer(model_dir, source["config"], model_id=settings["model_id"],
                                   revision=settings["revision"], manifest=prepared["manifest"])
        report("prompt_encoding")
        tokens = tokenizer.encode(parameters["prompt"], add_special_tokens=True)
    plan.validate_prompt(len(tokens))
    ledger = plan.allocator(include_cache=False)
    before = None
    if sys.platform == "win32":
        from .process_metrics import sample_process, average_cpu_percent
        before = sample_process()
    with ExitStack() as stack:
        report("kernel_initializing", backend=parameters.get("backend", "cpu"))
        cpu, _, _ = _kernels(stack, settings, "cpu")
        gate = None
        if parameters.get("backend", "cpu") == "hybrid":
            from .gpu_gate import GpuBoundaryGate
            gate = GpuBoundaryGate(device_index=settings["gpu_index"], target=settings["gpu_average_target"],
                                  window_seconds=settings["gpu_window_seconds"], max_wait_seconds=10)
        report("weights_initializing")
        weights = stack.enter_context(RuntimeWeights(root, settings, model_directory=model_dir,
            backend=parameters.get("backend", "cpu"), cpu=cpu, gpu=None, gate=gate, ledger=ledger, plan=plan))
        report("weights_ready")
        report("decoder_initializing", prompt_tokens=len(tokens), requested_new_tokens=parameters.get("generate", 32))
        decoder = stack.enter_context(StreamingDecoder(source["config"], weights, plan, ledger=ledger,
            model_id=settings["model_id"], revision=settings["revision"], progress=progress))
        generated = decoder.generate(tokens, max_new_tokens=parameters.get("generate", 32),
                                     eos_token_ids=source["config"].get("eos_token_id"))
        result = {"status": "GENERATED_UNVERIFIED", "generated_token_ids": generated,
                  "prompt_tokens": len(tokens), "plan": plan.to_dict(),
                  "reader": weights.stats(), "allocations": ledger.snapshot(),
                  "source_report": analysis["source_report"], **FULL_MODEL_FLAGS,
                  "note": "Experimental native FP32 backbone output; full-checkpoint numerical parity is unverified"}
        if tokenizer is not None:
            report("output_decoding")
            result["text"] = tokenizer.decode(generated, skip_special_tokens=True)
        if gate:
            report("gpu_finalization")
            gate.finish()
            result["pacing"] = gate.summary()
        if before:
            report("resource_validation")
            after = sample_process()
            result["process"] = asdict(after)
            result["observed_cpu_percent"] = average_cpu_percent(before, after)
            if max(after.peak_working_set_bytes, after.peak_private_commit_bytes) > settings["ram_budget_bytes"]:
                raise RuntimeError("Observed generation worker memory exceeded the configured budget")
        return result


def render_runtime(result):
    return (f"# Checkpoint {result['action']}\n\nStatus: **{result['status']}**\n\n"
            "Full-checkpoint numerical compatibility and resource limits remain unverified.\n\n"
            "```json\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n```\n")


def launch_runtime(root, settings, action, parameters):
    root = Path(root).resolve()
    validate_settings(settings)
    if action not in ("plan", "projection", "generate", "tokenizer"):
        raise ValueError("Unknown runtime action")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    report_root = reports_directory(root, settings)
    directory = report_root / action / identifier
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    write_json(request, {"root": str(root), "settings": settings, "action": action, "parameters": parameters})
    captured = []
    started = time.monotonic()
    if sys.platform == "win32":
        try:
            code = run_local_process([sys.executable, "-m", "glm_local.runtime_worker", str(request)], cwd=root,
                limits=JobLimits(settings["cpu_job_percent"], settings["ram_budget_bytes"]),
                on_policy=lambda policy: captured.append(asdict(policy)), timeout=parameters.get("timeout", 1800))
            result = _bounded_json(directory / "result.json", 4 * 1024**2)
            if EXIT_CODES.get(result.get("status")) != code or not captured:
                raise ValueError("Runtime worker result or Windows policy readback is inconsistent")
            if (result.get("action") != action or result.get("model_id") != settings["model_id"]
                    or result.get("revision") != settings["revision"]
                    or any(result.get(flag) is not False for flag in FULL_MODEL_FLAGS)):
                raise ValueError("Runtime worker identity, action or capability flags differ from the request")
        except subprocess.TimeoutExpired as error:
            code = 1
            elapsed = time.monotonic() - started
            result = {"status": "ERROR", "action": action, "model_id": settings["model_id"],
                      "revision": settings["revision"], "error_type": "TIMEOUT", "timed_out": True,
                      "timeout_seconds": error.timeout, "elapsed_seconds": elapsed,
                      "error": f"Runtime exceeded its {error.timeout:g}-second timeout after {elapsed:.1f}s; the worker job tree was stopped.",
                      **FULL_MODEL_FLAGS}
        except (Exception, KeyboardInterrupt) as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else 1
            result = {"status": "INTERRUPTED" if code == 130 else "ERROR", "action": action,
                      "model_id": settings["model_id"], "revision": settings["revision"],
                      "error": f"{type(error).__name__}: {error}", **FULL_MODEL_FLAGS}
    else:
        result, code = execute_runtime(root, settings, action, parameters, directory)
    result.setdefault("elapsed_seconds", time.monotonic() - started)
    if result["status"] in ("ERROR", "INTERRUPTED"):
        last, recovery_error = load_last_progress(directory, settings, action)
        if last is not None:
            result["last_progress"] = last
            if result.get("timed_out"):
                result["error"] += f" Last stage: {last['stage']}"
                if last.get("tensor_name"):
                    result["error"] += f" ({last['tensor_name']})"
                result["error"] += f"; stage elapsed {last['stage_elapsed_seconds']:.1f}s."
        else:
            result.pop("last_progress", None)
            result["last_progress_error"] = recovery_error
    result.update(run_directory=str(directory), installed_job_policy=captured[0] if captured else None,
                  job_policy_verified=bool(captured), child_exit_code=code)
    write_json(directory / "result.json", result)
    write_json(report_root / f"{action}-latest.json", result)
    rendered = render_runtime(result)
    (directory / "result.md").write_text(rendered, encoding="utf-8")
    (report_root / f"{action}-latest.md").write_text(rendered, encoding="utf-8")
    print(f"{action}: {result['status']}", flush=True)
    if result.get("error"):
        print(result["error"], flush=True)
    print(f"Report: {directory / 'result.md'}", flush=True)
    return code
