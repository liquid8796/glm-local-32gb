"""Named checkpoint settings and isolated metadata/report paths.

Profiles select pinned settings without moving the project root or mixing one
checkpoint's metadata with another checkpoint's reports. Settings without the
optional path fields keep the original on-disk layout.
"""
from __future__ import annotations

from pathlib import Path
import re
import stat

from .audit import validate_settings
from .checkpoint_http import strict_json, validate_target
from .metadata import safe_relative_path


PROFILE_NAMES = ("fp8", "nvfp4")
_PROFILES = {
    "fp8": ("cybersecurity-fp8.json", "dealignai/GLM-5.3-CYBERSECURITY-FP8"),
    "nvfp4": ("abliterated-nvfp4.json", "dealignai/GLM-5.3-ABLITERATED-NVFP4"),
}
_MAX_SETTINGS_BYTES = 64 * 1024


def _confined_path(root, relative, *, directory, label):
    """Reject links/reparse points and non-regular existing path components."""
    root = Path(root).resolve()
    path = root
    for index, part in enumerate(relative.parts):
        path /= part
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"{label} cannot use links or reparse points")
        is_directory = directory or index < len(relative.parts) - 1
        if is_directory and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} parent must be a regular directory")
        if not is_directory and not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} resolves outside the project directory")
    return resolved


def profile_config_path(root, profile):
    """Return one of the bundled profile files; arbitrary names are rejected."""
    if not isinstance(profile, str) or profile not in _PROFILES:
        raise ValueError(f"Unknown model profile; choose one of: {', '.join(PROFILE_NAMES)}")
    relative = Path("config") / "models" / _PROFILES[profile][0]
    return _confined_path(root, relative, directory=False, label="Model profile")


def load_profile(root, profile):
    """Read bounded, pinned profile settings and validate their identity/paths."""
    path = profile_config_path(root, profile)
    with path.open("rb") as handle:
        raw = handle.read(_MAX_SETTINGS_BYTES + 1)
    if len(raw) > _MAX_SETTINGS_BYTES:
        raise ValueError("Model profile exceeds 64 KiB")
    settings = strict_json(raw.removeprefix(b"\xef\xbb\xbf"))
    if not isinstance(settings, dict):
        raise ValueError("Model profile settings must be a JSON object")
    validate_settings(settings)
    validate_target(settings.get("model_id"), settings.get("revision"))
    if settings["model_id"] != _PROFILES[profile][1]:
        raise ValueError("Model profile identity does not match its named checkpoint")
    metadata_snapshot_path(root, settings)
    reports_directory(root, settings)
    return settings


def metadata_snapshot_path(root, settings):
    """Resolve a project-relative JSON snapshot beneath docs (legacy by default)."""
    name = settings.get("metadata_snapshot", "docs/model-metadata.json")
    if not isinstance(name, str) or len(name) > 1024 or any(c in name for c in '<>"|?*'):
        raise ValueError("metadata_snapshot must be a portable relative JSON path beneath docs")
    relative = safe_relative_path(name)
    if len(relative.parts) < 2 or relative.parts[0] != "docs" or relative.suffix != ".json":
        raise ValueError("metadata_snapshot must be a relative JSON path beneath docs")
    return _confined_path(root, relative, directory=False, label="Metadata snapshot")


def reports_directory(root, settings):
    """Use reports/<namespace> when selected, retaining reports for old settings."""
    if "report_namespace" not in settings:
        relative = Path("reports")
    else:
        namespace = settings["report_namespace"]
        if (not isinstance(namespace, str) or len(namespace) > 64
                or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", namespace) is None):
            raise ValueError("report_namespace must be a lowercase portable slug of 1..64 characters")
        relative = Path("reports") / str(safe_relative_path(namespace))
    return _confined_path(root, relative, directory=True, label="Reports directory")
