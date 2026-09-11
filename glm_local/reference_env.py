"""Standard-library provenance checks for the optional synthetic oracle.

This verifies three package versions, the Transformers archive identity, and
two model source files. It is not a verification of the entire environment,
all package contents, or any downloaded real checkpoint. Importing this module
does not import Torch, NumPy or Transformers.
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path


EXPECTED_VERSIONS = {"torch": "2.14.0+cpu", "numpy": "2.4.6", "transformers": "5.18.0.dev0"}
TRANSFORMERS_REVISION = "3f601734a3580f55484720770850966bba060e4f"
TRANSFORMERS_URL = ("https://github.com/huggingface/transformers/archive/"
                    + TRANSFORMERS_REVISION + ".zip")
ARCHIVE_SHA256 = "356276eb5f8fe21bf7f913bd66e6231eb21efdda8102de49000c05aed5a9261d"
SOURCE_SHA256 = {
    "transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py":
        "da0030e70764e9dae083cf9f9de54bae661082f04b702386d30dab91c52632f8",
    "transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py":
        "8c8b95bb0ecfcb502f768905964926ea55a83234e4b2f494b6f3b68333891a37",
}
SOURCE_MAX_BYTES = 2 * 1024 * 1024
METADATA_MAX_BYTES = 64 * 1024
VERIFICATION_SCOPE = (
    "Three installed package versions, Transformers direct URL/archive SHA-256 metadata, "
    "and SHA-256 of two GLM-MoE-DSA source files only; not the entire environment."
)


class ReferenceEnvironmentError(RuntimeError):
    """The optional reference environment does not meet its pinned contract."""


def _unsupported(reason):
    return ReferenceEnvironmentError("Unsupported or modified reference environment: " + reason)


def _read_bounded(path, limit, label):
    try:
        path = Path(path)
        if not path.is_file():
            raise _unsupported(label + " is missing or not a regular file")
        if path.stat().st_size > limit:
            raise _unsupported(label + " exceeds the verification size limit")
        with path.open("rb") as stream:
            contents = stream.read(limit + 1)
        if len(contents) > limit:
            raise _unsupported(label + " exceeds the verification size limit")
        return contents
    except (OSError, TypeError, ValueError) as exc:
        raise _unsupported(label + " could not be read") from exc


def _direct_url_path(distribution):
    files = distribution.files
    if files is None:
        raise _unsupported("Transformers installed-file metadata is unavailable")
    matches = [entry for entry in files
               if len(entry.parts) == 2 and entry.parts[-1] == "direct_url.json"
               and entry.parts[-2].startswith("transformers-")
               and entry.parts[-2].endswith(".dist-info")]
    if len(matches) != 1:
        raise _unsupported("Transformers direct_url.json metadata is missing or ambiguous")
    return distribution.locate_file(matches[0])


def verify_reference_environment():
    """Return JSON-safe verified provenance, or fail before importing an oracle.

    Package locations come from importlib.metadata distributions. Source reads
    are capped at 2 MiB each and direct-url metadata at 64 KiB.
    """
    distributions = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError as exc:
            raise _unsupported(package + " is not installed") from exc
        if distribution.version != expected:
            raise _unsupported(package + " must have the exact pinned version " + expected)
        distributions[package] = distribution

    transformer_distribution = distributions["transformers"]
    encoded = _read_bounded(_direct_url_path(transformer_distribution), METADATA_MAX_BYTES,
                            "Transformers direct_url.json")
    try:
        direct_url = json.loads(encoded)
    except (ValueError, UnicodeError) as exc:
        raise _unsupported("Transformers direct_url.json is not valid JSON") from exc
    if not isinstance(direct_url, dict) or direct_url.get("url") != TRANSFORMERS_URL:
        raise _unsupported("Transformers archive URL does not match the pinned revision")
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        raise _unsupported("Transformers archive SHA-256 metadata is missing")
    hashes = archive_info.get("hashes", {})
    if not isinstance(hashes, dict):
        raise _unsupported("Transformers archive SHA-256 metadata is malformed")
    modern_hash = hashes.get("sha256")
    legacy_hash = archive_info.get("hash")
    if modern_hash is None and legacy_hash is None:
        raise _unsupported("Transformers archive SHA-256 metadata is missing")
    if ((modern_hash is not None and modern_hash != ARCHIVE_SHA256)
            or (legacy_hash is not None and legacy_hash != "sha256=" + ARCHIVE_SHA256)):
        raise _unsupported("Transformers archive SHA-256 metadata does not match")

    observed_sources = {}
    for name, expected in SOURCE_SHA256.items():
        contents = _read_bounded(transformer_distribution.locate_file(name), SOURCE_MAX_BYTES, name)
        observed = hashlib.sha256(contents).hexdigest()
        if observed != expected:
            raise _unsupported(name + " SHA-256 does not match the pinned source")
        observed_sources[name] = observed

    return {
        "status": "verified",
        "verification_scope": VERIFICATION_SCOPE,
        "packages": dict(EXPECTED_VERSIONS),
        "transformers": {"revision": TRANSFORMERS_REVISION, "url": TRANSFORMERS_URL,
                         "archive_sha256": ARCHIVE_SHA256, "source_sha256": observed_sources},
        "bounds": {"source_max_bytes": SOURCE_MAX_BYTES,
                   "direct_url_max_bytes": METADATA_MAX_BYTES},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", required=True, action="store_true",
                        help="Verify the pinned optional reference dependencies")
    parser.parse_args(argv)
    try:
        result = verify_reference_environment()
    except ReferenceEnvironmentError as exc:
        print(json.dumps({"status": "unsupported_or_modified_environment", "error": str(exc)},
                         sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
