"""Child entrypoint; launch_runtime attaches the Windows job before resume.

execute_runtime emits flushed stage diagnostics and atomically checkpoints its
latest stage in this run directory; the final result remains a separate file.
"""
import sys
import stat
from pathlib import Path

from .architecture.report import _bounded_json
from .checkpoint_snapshot import write_json
from .model_profiles import reports_directory
from .runtime_commands import execute_runtime


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise ValueError("Expected one runtime request path")
    request_path = Path(args[0])
    info = request_path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Runtime request must be a regular file")
    request = _bounded_json(request_path, 1024 * 1024)
    root, action = Path(request["root"]).resolve(), request["action"]
    if (action not in ("plan", "projection", "generate", "tokenizer") or request_path.name != "request.json"
            or request_path.resolve().parent.parent != (reports_directory(root, request["settings"]) / action).resolve()):
        raise ValueError("Runtime request must stay in its action's report run directory")
    result, code = execute_runtime(request["root"], request["settings"], request["action"],
                                   request["parameters"], request_path.parent)
    write_json(request_path.parent / "result.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
