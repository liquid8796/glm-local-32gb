"""Internal worker: public parity command attaches its Windows Job first."""

import os
import json
from pathlib import Path
import sys
import traceback

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

from . import __version__
from .__main__ import ROOT, save_json
from .parity_run import execute_parity


def main():
    if len(sys.argv) != 2:
        raise ValueError("Expected generated parity request")
    request = Path(sys.argv[1]).resolve()
    if request.name != "request.json" or not request.is_relative_to((ROOT / "reports" / "parity").resolve()):
        raise ValueError("Parity requests must be generated inside reports/parity")
    if request.stat().st_size > 65536:
        raise ValueError("Parity request exceeds 64 KiB")
    data = {}
    try:
        data = json.loads(request.read_text(encoding="utf-8"))
        report = execute_parity(data["settings"], data["parameters"], request.parent)
        code = 0 if report["status"] == "PASS" else 3
    except Exception as error:
        report = {"status": "ERROR", "error": f"{type(error).__name__}: {error}",
                  "synthetic_official_parity_verified": False, "inference_verified": False,
                  "full_model_loaded": False, "job_policy_verified": False,
                  "traceback": traceback.format_exc(limit=12)[-16000:]}
        code = 1
    report["parameters"] = data.get("parameters", {}) if isinstance(data, dict) else {}
    report["tool_version"] = __version__
    report["worker_environment"] = {
        "python_version": sys.version.split()[0], "python_executable": sys.executable,
        "safetensors_version": getattr(sys.modules.get("safetensors"), "__version__", None),
    }
    save_json(request.parent / "result.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
