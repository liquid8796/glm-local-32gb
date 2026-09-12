"""Internal Windows metadata worker; public launcher attaches its Job first."""
import sys
from pathlib import Path

from .__main__ import ROOT
from .checkpoint_check import execute_metadata
from .checkpoint_http import strict_json
from .checkpoint_snapshot import write_json


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise ValueError("Expected a generated metadata request")
    request = Path(argv[0]).resolve()
    if (request.name != "request.json"
            or not request.is_relative_to((ROOT / "reports" / "metadata").resolve())
            or not 1 <= request.stat().st_size <= 65536):
        raise ValueError("Metadata request must be bounded and generated inside reports/metadata")
    data = strict_json(request.read_bytes())
    report, code = execute_metadata(ROOT, data["settings"], data["parameters"], request.parent)
    write_json(request.parent / "result.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
