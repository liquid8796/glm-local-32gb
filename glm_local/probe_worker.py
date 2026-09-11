"""Internal worker. Use `glm.bat probe` to apply the Windows job before startup."""

import json
from pathlib import Path
import sys

from .backend_probe import execute_probe
from .__main__ import ROOT, save_json


def main():
    if len(sys.argv) != 2:
        raise ValueError("Expected one generated probe request path")
    request = Path(sys.argv[1]).resolve()
    if request.name != "request.json" or not request.is_relative_to((ROOT / "reports" / "probes").resolve()):
        raise ValueError("Worker request must be generated inside reports/probes")
    if request.stat().st_size > 64 * 1024:
        raise ValueError("Probe request is too large")
    try:
        data = json.loads(request.read_text(encoding="utf-8"))
        report = execute_probe(data["settings"], data["parameters"], request.parent)
        code = 0 if report["status"] == "PASS" else 3
    except Exception as error:
        report = {"status": "ERROR", "error": f"{type(error).__name__}: {error}",
                  "inference_verified": False, "full_model_loaded": False,
                  "job_policy_verified": False}
        code = 1
    save_json(request.parent / "result.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
