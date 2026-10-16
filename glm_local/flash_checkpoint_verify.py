"""Read-only offline structural audit for the pinned GLM5Next Flash safetensors.

No network access, weight downloads, payload hashing, tensor materialization,
or inference. Returns only scope-labeled evidence for metadata/header checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat

from .safetensor_reader import SafeTensorReader


class FlashCheckpointError(ValueError):
    """Pinned local Flash checkpoint does not match its declared metadata."""


_SHARD_NAME = re.compile(r"model-\d{5}-of-\d{5}\.safetensors\Z")
_HASH_CHUNK = 65536


def _file(path: Path) -> int:
    try:
        info = path.lstat()
    except OSError as exc:
        raise FlashCheckpointError(f"Required file inaccessible: {path.name}") from exc
    if not stat.S_ISREG(info.st_mode) or (getattr(info, "st_file_attributes", 0) & 0x400):
        raise FlashCheckpointError(f"Expected regular file, not link: {path.name}")
    return info.st_size


def _verify_sha(path: Path, expected: str, *, size: int | None = None) -> None:
    actual_size = _file(path)
    if size is not None and actual_size != size:
        raise FlashCheckpointError(f"Metadata file size mismatch: {path.name}")
    if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise FlashCheckpointError(f"Missing pinned SHA-256: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while block := source.read(_HASH_CHUNK):
            digest.update(block)
    if digest.hexdigest() != expected or _file(path) != actual_size:
        raise FlashCheckpointError(f"SHA-256 mismatch: {path.name}")


def verify_flash_checkpoint(model_dir: str | Path, snapshot: dict) -> dict:
    """Verify all shard headers and pinned small-file hashes without payload scans.

    The caller supplies a trusted pinned metadata snapshot, typically from
    docs/models/flash-cybersecurity-w4a16/model-metadata.json.
    """
    root = Path(model_dir).absolute()
    if not root.is_dir() or root.is_symlink():
        raise FlashCheckpointError("Model directory missing or a symbolic link")
    if not isinstance(snapshot, dict):
        raise FlashCheckpointError("Expected pinned metadata object")
    expected_weights = snapshot.get("weights")
    if not isinstance(expected_weights, list) or not expected_weights:
        raise FlashCheckpointError("Pinned weight shard list is empty")
    sizes = {}
    for row in expected_weights:
        if not isinstance(row, dict):
            raise FlashCheckpointError("Invalid pinned shard entry")
        name, size = row.get("name"), row.get("bytes")
        if (not isinstance(name, str) or _SHARD_NAME.fullmatch(name) is None
                or type(size) is not int or size < 8 or name in sizes):
            raise FlashCheckpointError("Invalid or duplicate shard name/size")
        sizes[name] = size
    if snapshot.get("weight_shards") != len(sizes):
        raise FlashCheckpointError("Pinned shard count mismatch")
    total = sum(sizes.values())
    if snapshot.get("weight_bytes") != total:
        raise FlashCheckpointError("Pinned shard size total mismatch")
    actual_shards = {p.name for p in root.glob("*.safetensors")}
    if actual_shards != set(sizes):
        raise FlashCheckpointError(
            f"Shard inventory mismatch: {len(set(sizes)-actual_shards)} missing, "
            f"{len(actual_shards-set(sizes))} extra")
    for name, expected in sizes.items():
        if _file(root / name) != expected:
            raise FlashCheckpointError(f"Wrong shard size: {name}")

    hashes = snapshot.get("verification", {}).get("source_sha256", {})
    if not isinstance(hashes, dict):
        raise FlashCheckpointError("Missing pinned metadata digests")
    verified_names = []
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        if name not in hashes:
            raise FlashCheckpointError(f"Missing pinned SHA-256: {name}")
        _verify_sha(root / name, hashes[name])
        verified_names.append(name)
    for entry in snapshot.get("metadata_files", []):
        if isinstance(entry, dict) and entry.get("rfilename") == "tokenizer.json":
            sha = entry.get("lfs", {}).get("sha256")
            if sha:
                _verify_sha(root / "tokenizer.json", sha, size=entry.get("size"))
                verified_names.append("tokenizer.json")

    index_path = root / "model.safetensors.index.json"
    if _file(index_path) > 64 * 1024**2:
        raise FlashCheckpointError("Safetensors index exceeds local size limit")
    try:
        index = json.loads(index_path.read_bytes())
        mapping = index["weight_map"]
        declared = index["metadata"]["total_size"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FlashCheckpointError("Invalid local safetensors index") from exc
    count = snapshot.get("index_summary", {}).get("tensor_count")
    if (not isinstance(mapping, dict) or type(count) is not int or count < 1
            or len(mapping) != count or set(mapping.values()) != set(sizes)
            or type(declared) is not int or declared != total
            or declared != snapshot.get("index_summary", {}).get("declared_total_size")):
        raise FlashCheckpointError("Index count, shard mapping or total_size mismatch")

    seen = set()
    payload_bytes = 0
    header_bytes = 0
    for shard in sizes:
        with SafeTensorReader(root / shard) as reader:
            info = reader.stats()
            header_bytes += info["header_bytes"] + 8
            for tensor_name, tensor in reader.tensors.items():
                if tensor_name in seen or mapping.get(tensor_name) != shard:
                    raise FlashCheckpointError(f"Invalid tensor mapping in shard {shard}")
                seen.add(tensor_name)
                payload_bytes += tensor.nbytes
    if seen != set(mapping):
        raise FlashCheckpointError("Missing indexed tensor mapping in shard headers")
    if payload_bytes + header_bytes != total:
        raise FlashCheckpointError("Shard header and payload byte totals do not match")
    return {
        "status": "PASS",
        "scope": "local_pinned_metadata_and_all_shard_headers",
        "model_id": snapshot.get("model_id"),
        "revision": snapshot.get("revision"),
        "shard_count": len(sizes),
        "tensor_count": len(seen),
        "total_file_bytes": total,
        "total_tensor_payload_bytes": payload_bytes,
        "total_header_bytes": header_bytes,
        "size_convention": "complete_shard_files",
        "metadata_hashes_verified": sorted(verified_names),
        "weight_payload_hashes_verified": False,
        "full_model_inference_verified": False,
        "source": "offline_local_disk",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only pinned Flash checkpoint audit")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--snapshot", default=str(
        Path(__file__).resolve().parent.parent /
        "docs/models/flash-cybersecurity-w4a16/model-metadata.json"))
    args = parser.parse_args(argv)
    try:
        snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
        result = verify_flash_checkpoint(args.model_dir, snapshot)
    except (FlashCheckpointError, OSError, ValueError) as error:
        print(json.dumps({"status": "ERROR", "message": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
