"""Generated storage-validation worker; the launcher installs its Job first."""

import json
import os
from pathlib import Path
import sys

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from .__main__ import ROOT, save_json
from .safetensor_check import execute_check


def main():
    if len(sys.argv) != 2:
        raise ValueError("Expected a generated storage validation request")
    request = Path(sys.argv[1]).resolve()
    if request.name != "request.json" or not request.is_relative_to((ROOT / "reports/storage").resolve()):
        raise ValueError("Storage requests must be generated inside reports/storage")
    if request.stat().st_size > 65536:
        raise ValueError("Storage request exceeds 64 KiB")
    try:
        data = json.loads(request.read_text(encoding="utf-8"))
        report = execute_check(data["settings"], data["parameters"], request.parent)
        code = 0 if report["status"] == "PASS" else 3
    except Exception as error:
        report = {"status": "ERROR", "error": f"{type(error).__name__}: {error}",
                  "inference_verified": False, "synthetic_storage_verified": False, "full_model_loaded": False}
        code = 1
    save_json(request.parent / "result.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
