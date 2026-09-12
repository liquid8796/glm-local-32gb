"""Internal Windows metadata worker; public launcher attaches its Job first."""
import sys
import stat
from pathlib import Path

from .__main__ import ROOT
from .architecture.report import _bounded_json
from .checkpoint_check import execute_metadata
from .checkpoint_snapshot import write_json
from .model_profiles import reports_directory


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise ValueError("Expected a generated metadata request")
    original = Path(argv[0])
    info = original.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Metadata request must be a regular file")
    request = original.resolve()
    report_root = reports_directory(ROOT, {})
    if request.name != "request.json" or not request.is_relative_to(report_root):
        raise ValueError("Metadata request must be bounded and generated inside reports/metadata")
    relative = request.relative_to(report_root)
    if not ((len(relative.parts) == 3 and relative.parts[0] == "metadata")
            or (len(relative.parts) == 4 and relative.parts[1] == "metadata")):
        raise ValueError("Metadata request must be in a metadata report run directory")
    data = _bounded_json(request, 65536)
    if request.parent.parent != reports_directory(ROOT, data["settings"]) / "metadata":
        raise ValueError("Metadata request namespace differs from its settings")
    report, code = execute_metadata(ROOT, data["settings"], data["parameters"], request.parent)
    write_json(request.parent / "result.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
