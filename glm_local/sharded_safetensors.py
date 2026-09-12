"""Read-only, bounded local safetensors index with an LRU of open shards.

Opening scans headers only, one file at a time. Payload is read on demand via
SafeTensorReader. Index paths are flat, regular, non-symlink files. Fingerprints
catch ordinary file changes, not adversarial changes or checkpoint provenance.
Local limits here are not limits of the safetensors/Hugging Face file formats.
Instances are single-thread-owned; no tensors or decoded weights are cached.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

from .safetensor_reader import SafeTensorError, SafeTensorReader

INDEX_NAME = "model.safetensors.index.json"
MAX_INDEX_BYTES = 1024 * 1024
MAX_READ_BYTES = 65536
MAX_SHARDS = 512
MAX_INDEX_TENSORS = 8192
MAX_OPEN_SHARDS = 8
_SUM_COUNTERS = ("actual_read_bytes", "actual_read_calls", "tensor_read_bytes", "tensor_read_calls")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _regular(path):
    try:
        info = path.lstat()
    except OSError as error:
        raise SafeTensorError(f"Cannot inspect local shard/index: {path.name}") from error
    # Reject Windows junction/reparse points too. lstat follows neither symlinks
    # nor those file entries; parent path is resolved once by the caller.
    if (not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise SafeTensorError("Shard/index must be a regular file, not a symlink/reparse point")
    return info


def _unique(pairs):
    value = {}
    for name, entry in pairs:
        if name in value:
            raise SafeTensorError(f"Duplicate index JSON key: {name!r}")
        value[name] = entry
    return value


def _reject_constant(value):
    raise SafeTensorError(f"Nonfinite index JSON constant: {value}")


def _read_index(path):
    initial = _regular(path)
    if not 1 <= initial.st_size <= MAX_INDEX_BYTES:
        raise SafeTensorError("Index exceeds its 1-byte to 1-MiB policy")
    pieces = []
    try:
        with path.open("rb", buffering=0) as stream:
            descriptor = os.fstat(stream.fileno())
            # Compare stable cross-API fields; ctime differs on Windows.
            if _identity(descriptor)[:4] != _identity(initial)[:4]:
                raise SafeTensorError("Index changed while opening")
            remaining = initial.st_size
            while remaining:
                part = stream.read(min(remaining, MAX_READ_BYTES))
                if not part:
                    raise SafeTensorError("Index truncated during bounded read")
                pieces.append(part)
                remaining -= len(part)
            if (_identity(os.fstat(stream.fileno())) != _identity(descriptor)
                    or _identity(_regular(path)) != _identity(initial)):
                raise SafeTensorError("Index changed during bounded read")
        root = json.loads(b"".join(pieces).decode("utf-8"), object_pairs_hook=_unique,
                          parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise SafeTensorError(f"Invalid bounded shard index: {error}") from error
    if not isinstance(root, dict) or set(root) != {"metadata", "weight_map"}:
        raise SafeTensorError("Index requires exactly metadata and weight_map")
    metadata, mapping = root["metadata"], root["weight_map"]
    if not isinstance(metadata, dict) or not isinstance(mapping, dict):
        raise SafeTensorError("Index metadata/weight_map must be JSON objects")
    size = metadata.get("total_size")
    if type(size) is not int or not 0 <= size <= 2**63 - 1:
        raise SafeTensorError("Index metadata.total_size must be a nonnegative integer")
    if not 1 <= len(mapping) <= MAX_INDEX_TENSORS:
        raise SafeTensorError("Index tensor count exceeds local policy")
    for name, filename in mapping.items():
        if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 512:
            raise SafeTensorError("Index tensor names must be nonempty and at most 512 UTF-8 bytes")
        if (not isinstance(filename, str) or len(filename) > 128
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors", filename)
                or filename.split(".", 1)[0].upper() in {
                    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}):
            raise SafeTensorError("Shard name must be a portable flat .safetensors filename")
    names = set(mapping.values())
    if len(names) > MAX_SHARDS or len({name.casefold() for name in names}) != len(names):
        raise SafeTensorError("Too many shards or case-colliding shard filenames")
    return root, _identity(initial), {
        "index_read_bytes": initial.st_size,
        "index_read_calls": len(pieces),
        "index_max_read_bytes": max(map(len, pieces)),
    }


class ShardedSafeTensorReader:
    """Same tensor read interface as SafeTensorReader, including cross-shard pairs.

    ``max_open_shards`` bounds persistent file handles/header caches (1..8).
    All index entries/header descriptors are bounded metadata, not payload.
    Every referenced shard must exist and match the exact index tensor set.
    """

    def __init__(self, directory: str | Path, *, max_open_shards: int = 2, expected_shards=None):
        if type(max_open_shards) is not int or not 1 <= max_open_shards <= MAX_OPEN_SHARDS:
            raise SafeTensorError("max_open_shards must be an integer in [1, 8]")
        self.directory = Path(directory).resolve()
        self._closed = False
        self._readers = OrderedDict()
        self._max_open_shards = max_open_shards
        self._peak_open = self._open_count = self._evictions = 0
        self._totals = {key: 0 for key in _SUM_COUNTERS}
        self._max_actual_read = 0
        self._index_path = self.directory / INDEX_NAME
        root, self._index_identity, self._index_stats = _read_index(self._index_path)
        if expected_shards is not None and set(root["weight_map"].values()) != set(expected_shards):
            raise SafeTensorError("Index does not reference the required shard filenames")
        self._metadata = root["metadata"]
        self._weight_map = MappingProxyType(root["weight_map"])
        self._identities, self._shard_metadata, self._names = {}, {}, {}
        tensors = {}
        try:
            for filename in sorted(set(self._weight_map.values())):
                names = {name for name, shard in self._weight_map.items() if shard == filename}
                path = self.directory / filename
                before = _identity(_regular(path))
                with SafeTensorReader(path) as reader:
                    self._open_count += 1
                    self._peak_open = max(self._peak_open, 1)
                    if set(reader.tensors) != names:
                        raise SafeTensorError(f"Index/header tensor mapping mismatch in {filename}")
                    tensors.update(reader.tensors)
                    self._shard_metadata[filename] = dict(reader.metadata)
                    self._retire_stats(reader)
                if _identity(_regular(path)) != before:
                    raise SafeTensorError("Shard changed during header inspection")
                self._identities[filename] = before
                self._names[filename] = names
            if sum(info.nbytes for info in tensors.values()) != self._metadata["total_size"]:
                raise SafeTensorError("Index total_size differs from tensor payload bytes")
            self._tensors = MappingProxyType(tensors)
            self._check_open()
        except BaseException:
            self.close()
            raise

    @property
    def tensors(self):
        return self._tensors

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def weight_map(self):
        return self._weight_map

    @property
    def shard_metadata(self):
        return deepcopy(self._shard_metadata)

    def _check_open(self):
        if self._closed:
            raise SafeTensorError("Sharded reader is closed")
        if _identity(_regular(self._index_path)) != self._index_identity:
            raise SafeTensorError("Shard index changed since validation")

    def _retire_stats(self, reader):
        counters = reader.stats()
        for key in _SUM_COUNTERS:
            self._totals[key] += counters[key]
        self._max_actual_read = max(self._max_actual_read, counters["max_actual_read_bytes"])

    def _reader(self, name):
        self._check_open()
        if not isinstance(name, str) or name not in self._weight_map:
            raise SafeTensorError(f"Unknown indexed tensor: {name!r}")
        filename = self._weight_map[name]
        path = self.directory / filename
        if _identity(_regular(path)) != self._identities[filename]:
            raise SafeTensorError("Shard changed since initial header validation")
        if filename in self._readers:
            reader = self._readers.pop(filename)
            self._readers[filename] = reader
            return reader
        if len(self._readers) >= self._max_open_shards:
            _, evicted = self._readers.popitem(last=False)
            self._retire_stats(evicted)
            evicted.close()
            self._evictions += 1
        reader = SafeTensorReader(path)
        self._open_count += 1
        try:
            if (set(reader.tensors) != self._names[filename]
                    or dict(reader.metadata) != self._shard_metadata[filename]
                    or any(info != self._tensors[key] for key, info in reader.tensors.items())
                    or _identity(_regular(path)) != self._identities[filename]):
                raise SafeTensorError("Reopened shard differs from validated metadata")
        except BaseException:
            reader.close()
            raise
        self._readers[filename] = reader
        self._peak_open = max(self._peak_open, len(self._readers))
        return reader

    def read_bytes(self, name, offset, count):
        result = self._reader(name).read_bytes(name, offset, count)
        self._check_open()
        return result

    def read_matrix_tile(self, name, row, col, rows, cols):
        result = self._reader(name).read_matrix_tile(name, row, col, rows, cols)
        self._check_open()
        return result

    def stats(self):
        totals = dict(self._totals)
        largest = self._max_actual_read
        for reader in self._readers.values():
            counters = reader.stats()
            for key in _SUM_COUNTERS:
                totals[key] += counters[key]
            largest = max(largest, counters["max_actual_read_bytes"])
        return {**totals, **self._index_stats, "max_actual_read_bytes": largest,
                "shard_count": len(self._names), "tensor_count": len(self._weight_map),
                "open_shards": len(self._readers), "peak_open_shards": self._peak_open,
                "max_open_shards": self._max_open_shards, "shard_open_count": self._open_count,
                "shard_evictions": self._evictions, "resident_fp8_cache_bytes": 0,
                "payload_reads_are_lazy": True,
                "content_verification": "stat_fingerprints_only_no_checkpoint_checksum"}

    def close(self):
        while self._readers:
            _, reader = self._readers.popitem()
            self._retire_stats(reader)
            reader.close()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
