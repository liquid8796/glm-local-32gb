"""Internal worker; public `glm.bat mini` installs the Job Object first."""

import json
import os
from pathlib import Path
import sys

# Bound tiny reference BLAS thread creation within this worker only.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from .__main__ import ROOT, save_json
from .mini_run import execute_mini


def main():
    if len(sys.argv) != 2:
        raise ValueError("Expected one generated miniature request")
    request = Path(sys.argv[1]).resolve()
    if request.name != "request.json" or not request.is_relative_to((ROOT / "reports" / "mini").resolve()):
        raise ValueError("Miniature request must be generated inside reports/mini")
    if request.stat().st_size > 65536:
        raise ValueError("Miniature request exceeds 64 KiB")
    try:
        data = json.loads(request.read_text(encoding="utf-8"))
        report = execute_mini(data["settings"], data["parameters"], request.parent)
        code = 0 if report["status"] == "PASS" else 3
    except Exception as error:
        report = {"status": "ERROR", "error": f"{type(error).__name__}: {error}",
                  "synthetic_decoder_verified": False, "inference_verified": False,
                  "full_model_loaded": False, "job_policy_verified": False}
        code = 1
    save_json(request.parent / "result.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
