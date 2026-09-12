"""Portable metadata evidence: JSON and prefix+header only, never tensor payload.

SHA-256 detects ordinary local changes. A user-editable snapshot is not a signed
remote attestation; offline replay never claims freshly authenticated provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .checkpoint_http import (FetchLimits, JSON_LIMITS, MAX_SHARDS, MetadataError,
                              ReadBudget, header_length, shard_filename, strict_json,
                              validate_target)
from .safetensor_reader import MAX_HEADER_BYTES, MAX_READ_BYTES

OBJECT_FILES = {"model": "model.json", "config": "config.json",
                "index": "model.safetensors.index.json"}
MAX_MANIFEST_BYTES = 1024**2


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise MetadataError("Snapshot artifacts must be regular files, not links/reparse points")
    return info


def _directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise MetadataError("Snapshot directories must not be links/reparse points")


class EvidenceStore:
    def __init__(self, directory, model_id, revision):
        validate_target(model_id, revision)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "headers").mkdir()
        self.manifest = {"format_version": 1, "model_id": model_id, "revision": revision,
                         "objects": {}, "headers": {}}
        self._flush()

    def _flush(self):
        write_json(self.directory / "snapshot.json", self.manifest)

    def put_json(self, kind, raw):
        if kind not in OBJECT_FILES or not 1 <= len(raw) <= JSON_LIMITS[kind]:
            raise MetadataError("Invalid snapshot JSON object or size")
        strict_json(raw)
        (self.directory / OBJECT_FILES[kind]).write_bytes(raw)
        self.manifest["objects"][kind] = digest(raw)
        self._flush()

    def put_header(self, name, raw, file_size):
        name = shard_filename(name)
        length = header_length(raw[:8], file_size)
        if len(raw) != 8 + length:
            raise MetadataError("Snapshot must contain only prefix+header, not payload")
        (self.directory / "headers" / (name + ".header")).write_bytes(raw)
        self.manifest["headers"][name] = {**digest(raw), "file_size": file_size}
        self._flush()


class OfflineMetadataSource:
    mode = "offline_replay"

    def __init__(self, directory, model_id, revision, *, limits=None):
        validate_target(model_id, revision)
        self.model_id, self.revision = model_id, revision
        self.directory = Path(directory).absolute()
        _directory(self.directory)
        self.budget = ReadBudget((limits or FetchLimits()).total_body_bytes)
        self.manifest = strict_json(self._read("snapshot.json", MAX_MANIFEST_BYTES))
        root = self.manifest
        if (not isinstance(root, dict) or set(root) != {
                "format_version", "model_id", "revision", "objects", "headers"}
                or type(root["format_version"]) is not int or root["format_version"] != 1
                or root["model_id"] != model_id or root["revision"] != revision):
            raise MetadataError("Snapshot model/revision/schema does not match settings")
        if (not isinstance(root["objects"], dict) or not set(root["objects"]) <= set(OBJECT_FILES)
                or not isinstance(root["headers"], dict) or len(root["headers"]) > MAX_SHARDS):
            raise MetadataError("Invalid snapshot artifact manifest")
        names = root["headers"]
        for name in names:
            shard_filename(name)
        if len({n.casefold() for n in names}) != len(names):
            raise MetadataError("Case-colliding snapshot shard names")

    def _read(self, relative, cap):
        path = self.directory / relative  # All callers supply generated fixed paths.
        _directory(path.parent)
        before = _regular(path)
        if not 1 <= before.st_size <= cap:
            raise MetadataError("Snapshot artifact exceeds its bounded size policy")
        self.budget.ensure(before.st_size)
        parts = []
        with path.open("rb", buffering=0) as stream:
            descriptor = os.fstat(stream.fileno())
            # Windows path stat and descriptor stat can differ in ctime.
            if _identity(descriptor)[:4] != _identity(before)[:4]:
                raise MetadataError("Snapshot artifact changed while opening")
            remaining = before.st_size
            while remaining:
                part = self.budget.read(stream, min(remaining, MAX_READ_BYTES))
                if not part:
                    raise MetadataError("Snapshot artifact was truncated")
                remaining -= len(part)
                parts.append(part)
            if (_identity(os.fstat(stream.fileno())) != _identity(descriptor)
                    or _identity(_regular(path)) != _identity(before)):
                raise MetadataError("Snapshot artifact changed during read")
        return b"".join(parts)

    def _verify(self, raw, descriptor, *, header=False):
        fields = {"bytes", "sha256", "file_size"} if header else {"bytes", "sha256"}
        if (not isinstance(descriptor, dict) or set(descriptor) != fields
                or type(descriptor["bytes"]) is not int
                or not isinstance(descriptor["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
                or digest(raw) != {key: descriptor[key] for key in ("bytes", "sha256")}):
            raise MetadataError("Snapshot artifact SHA-256/byte count mismatch")

    def json_bytes(self, kind):
        if kind not in OBJECT_FILES or kind not in self.manifest["objects"]:
            raise MetadataError(f"Snapshot lacks required JSON object: {kind}")
        raw = self._read(OBJECT_FILES[kind], JSON_LIMITS[kind])
        self._verify(raw, self.manifest["objects"][kind])
        strict_json(raw)
        return raw

    def header_bytes(self, name, file_size):
        name = shard_filename(name)
        if name not in self.manifest["headers"]:
            raise MetadataError("Snapshot lacks requested shard header")
        descriptor = self.manifest["headers"][name]
        raw = self._read("headers/" + name + ".header", MAX_HEADER_BYTES + 8)
        self._verify(raw, descriptor, header=True)
        if type(descriptor["file_size"]) is not int or descriptor["file_size"] != file_size:
            raise MetadataError("Snapshot shard file size differs from model manifest")
        length = header_length(raw[:8], file_size)
        if len(raw) != 8 + length:
            raise MetadataError("Snapshot contains truncated header or unexpected tensor payload")
        return raw

    def available_shards(self):
        return set(self.manifest["headers"])

    def stats(self):
        return {**self.budget.stats(), "mode": self.mode, "requests": 0,
                "tensor_payload_bytes_requested": 0,
                "remote_provenance_authenticated": False,
                "counter_scope": "Local metadata artifact reads, including snapshot manifest"}
