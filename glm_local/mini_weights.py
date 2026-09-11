"""Fixed tiny synthetic FP8 fixtures and bounded, uncached tensor reads.

This private fixture format cannot open real checkpoints or arbitrary shapes.
All file reads are unbuffered and at most 4096 bytes. Opening verifies the full
payload hash and saves only validation digests, not FP8 tensor data. Later reads
check both file identity/stat and the digest of the requested immutable bytes.
An external writer can still change other bytes after validation; those changes
are detected by a stat check or when those bytes are requested. This is not a
file lock. Returned matrices belong to the caller and are not a reader cache.
Small FP32 normalization/bias vectors remain resident in the bounded manifest.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import struct

from .mini_spec import SPEC, matrix_shapes, vector_lengths


MAX_FILE_BYTES = 64 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_READ_BYTES = 4096
FORMAT = "glm-synthetic-mini-v1"
_MANIFEST_KEYS = {"format", "spec", "seed", "weight_file", "weight_bytes",
                  "sha256", "matrices", "vectors"}


@dataclass(frozen=True)
class Matrix:
    weights: bytes
    rows: int
    cols: int
    scale: float


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _decode(code):
    """Local E4M3FN decode; the mathematical oracle has its own decoder."""
    magnitude = code & 127
    if magnitude == 127:
        raise ValueError("Synthetic FP8 payload contains NaN")
    exponent, fraction = divmod(magnitude, 8)
    value = fraction / 512 if exponent == 0 else (1 + fraction / 8) * 2 ** (exponent - 7)
    return -value if code & 128 else value


_FINITE_CODES = tuple((code, _decode(code)) for code in range(256) if code & 127 != 127)


def _encode(value):
    return min(_FINITE_CODES, key=lambda entry: (abs(entry[1] - value), entry[0] & 1,
                                                entry[0]))[0]


def _rng(seed, name):
    digest = hashlib.sha256(struct.pack("<I", seed) + name.encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "little"))


def _seed_valid(seed):
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Synthetic seed must be a uint32 integer")


def write_mini_bundle(directory: str | Path, seed: int = 7) -> dict:
    """Create deterministic invented weights, never overwrite either bundle file.

    The directory may already exist; missing parents must be created by the
    caller. A write failure may leave partial new files, which the reader rejects.
    Only one generated FP8 matrix is resident at a time.
    """
    _seed_valid(seed)
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir()
    if not directory.is_dir():
        raise ValueError("Synthetic bundle directory must be a directory")
    weight_path, manifest_path = directory / "weights.bin", directory / "manifest.json"
    if os.path.lexists(weight_path) or os.path.lexists(manifest_path):
        raise FileExistsError("Synthetic bundle files already exist; refusing to overwrite")
    manifest = {"format": FORMAT, "spec": deepcopy(SPEC), "seed": seed,
                "weight_file": "weights.bin", "weight_bytes": 0, "sha256": "",
                "matrices": {}, "vectors": {}}
    payload_hash = hashlib.sha256()
    with open(weight_path, "xb", buffering=0) as stream:
        offset = 0
        for name, (rows, cols) in matrix_shapes().items():
            rng = _rng(seed, name)
            scale = 0.0625 if name == "embed" else 0.03125
            payload = bytes(_encode(rng.uniform(-3.0, 3.0)) for _ in range(rows * cols))
            manifest["matrices"][name] = {"offset": offset, "rows": rows, "cols": cols,
                                           "scale": scale}
            if stream.write(payload) != len(payload):
                raise OSError("Incomplete synthetic weight write")
            payload_hash.update(payload)
            offset += len(payload)
        manifest["weight_bytes"] = offset
    if offset >= MAX_FILE_BYTES:
        raise ValueError("Fixed synthetic payload exceeded its size bound")
    manifest["sha256"] = payload_hash.hexdigest()
    for name, length in vector_lengths().items():
        rng = _rng(seed, name)
        if name == "layer.1.router_bias":
            values = [0.035, -0.025, 0.015, -0.005]
        elif name == "layer.0.index_norm_bias":
            values = [(-1 if index & 1 else 1) * rng.uniform(0.015, 0.045)
                      for index in range(length)]
        else:
            values = [rng.uniform(0.9, 1.1) for _ in range(length)]
        manifest["vectors"][name] = [_f32(value) for value in values]
    encoded = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("Fixed synthetic metadata exceeded its size bound")
    with open(manifest_path, "xb", buffering=0) as stream:
        if stream.write(encoded) != len(encoded):
            raise OSError("Incomplete synthetic manifest write")
    return deepcopy(manifest)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate synthetic manifest key")
        result[key] = value
    return result


def _finite_f32(value, *, positive=False):
    if type(value) not in (int, float):
        return False
    try:
        if not math.isfinite(value):
            return False
        rounded = _f32(value)
    except (OverflowError, struct.error):
        return False
    return rounded == value and (not positive or value > 0)


def _exact_json(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _exact_json(value[key], item) for key, item in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact_json(item, target) for item, target in zip(value, expected))
    return value == expected


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Invalid fixed synthetic manifest schema")
    if manifest["format"] != FORMAT or not _exact_json(manifest["spec"], SPEC):
        raise ValueError("Only the exact fixed synthetic spec is supported")
    _seed_valid(manifest["seed"])
    if manifest["weight_file"] != "weights.bin":
        raise ValueError("Synthetic weight_file must be the literal weights.bin")
    if not isinstance(manifest["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", manifest["sha256"]):
        raise ValueError("Invalid synthetic SHA256")
    matrices = manifest["matrices"]
    if not isinstance(matrices, dict) or set(matrices) != set(matrix_shapes()):
        raise ValueError("Synthetic matrix names must match the fixed contract")
    expected_offset = 0
    for name, (rows, cols) in matrix_shapes().items():
        entry = matrices[name]
        if not isinstance(entry, dict) or set(entry) != {"offset", "rows", "cols", "scale"}:
            raise ValueError("Invalid synthetic matrix schema")
        if any(type(entry[key]) is not int or entry[key] != expected
               for key, expected in (("offset", expected_offset), ("rows", rows), ("cols", cols))):
            raise ValueError("Synthetic matrix shape/offset must match the fixed contract")
        if not _finite_f32(entry["scale"], positive=True):
            raise ValueError("Synthetic scales must be positive finite exact float32")
        expected_offset += rows * cols
    if (type(manifest["weight_bytes"]) is not int or manifest["weight_bytes"] != expected_offset
            or not 0 < expected_offset < MAX_FILE_BYTES):
        raise ValueError("Invalid fixed synthetic payload size")
    vectors = manifest["vectors"]
    if not isinstance(vectors, dict) or set(vectors) != set(vector_lengths()):
        raise ValueError("Synthetic vector names must match the fixed contract")
    for name, length in vector_lengths().items():
        vector = vectors[name]
        if (not isinstance(vector, list) or len(vector) != length
                or any(not _finite_f32(value) for value in vector)):
            raise ValueError("Synthetic vectors must have fixed lengths and finite float32 values")
        if not name.endswith("bias") and any(value <= 0 for value in vector):
            raise ValueError("Synthetic normalization weights must be positive")


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _regular_stat(path, maximum):
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        raise ValueError("Synthetic bundle member must be a bounded regular file, not a link")
    return info


class MiniWeights:
    """Read only the fixed fixture; one FP8 matrix or embedding row per request."""

    def __init__(self, directory: str | Path):
        self._stream = None
        self._reads = self._bytes = self._max_read = 0
        self._payload_reads = self._payload_bytes = 0
        self._matrix_hashes = {}
        self._row_hashes = {}
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest_stat = _regular_stat(manifest_path, MAX_METADATA_BYTES)
        chunks = []
        with open(manifest_path, "rb", buffering=0) as stream:
            descriptor_stat = os.fstat(stream.fileno())
            # Windows Python can report creation time in fstat.st_ctime and
            # change time in stat.st_ctime. Compare identity/size/mtime across
            # the APIs, then compare each API's full baseline to itself.
            if _identity(descriptor_stat)[:4] != _identity(manifest_stat)[:4]:
                raise ValueError("Synthetic manifest changed while opening")
            remaining = manifest_stat.st_size
            while remaining:
                chunk = self._read(stream, min(remaining, MAX_READ_BYTES), payload=False)
                chunks.append(chunk)
                remaining -= len(chunk)
            if (_identity(os.fstat(stream.fileno())) != _identity(descriptor_stat)
                    or _identity(_regular_stat(manifest_path, MAX_METADATA_BYTES))
                    != _identity(manifest_stat)):
                raise ValueError("Synthetic manifest changed while reading")
        self._metadata_bytes = manifest_stat.st_size
        try:
            self._manifest = json.loads(b"".join(chunks), object_pairs_hook=_no_duplicates)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Invalid synthetic manifest JSON") from exc
        _validate_manifest(self._manifest)
        self._path = directory / "weights.bin"
        initial = _regular_stat(self._path, MAX_FILE_BYTES)
        if initial.st_size != self._manifest["weight_bytes"]:
            raise ValueError("Synthetic payload size mismatch")
        try:
            self._stream = open(self._path, "rb", buffering=0)
            self._baseline = _identity(initial)
            self._descriptor_baseline = _identity(os.fstat(self._stream.fileno()))
            if self._descriptor_baseline[:4] != self._baseline[:4]:
                raise ValueError("Synthetic weights changed while opening")
            self._check_unchanged()
            digest = hashlib.sha256()
            for name, (rows, cols) in matrix_shapes().items():
                data = self._read(self._stream, rows * cols, payload=True)
                self._validate_codes(data)
                digest.update(data)
                self._matrix_hashes[name] = hashlib.sha256(data).digest()
                if name == "embed":
                    for token in range(rows):
                        self._row_hashes[token] = hashlib.sha256(data[token * cols:(token + 1) * cols]).digest()
            self._check_unchanged()
            if digest.hexdigest() != self._manifest["sha256"]:
                raise ValueError("Synthetic payload SHA256 mismatch")
        except BaseException:
            self.close()
            raise

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def _read(self, stream, size, *, payload):
        if type(size) is not int or not 0 < size <= MAX_READ_BYTES:
            raise ValueError("Synthetic read exceeds its fixed bound")
        data = stream.read(size)
        self._reads += 1
        self._bytes += len(data)
        self._max_read = max(self._max_read, len(data))
        if payload:
            self._payload_reads += 1
            self._payload_bytes += len(data)
        if len(data) != size:
            raise ValueError("Truncated synthetic bundle read")
        return data

    def _check_unchanged(self):
        if self._stream is None or self._stream.closed:
            raise ValueError("Synthetic weight reader is closed")
        if (_identity(os.fstat(self._stream.fileno())) != self._descriptor_baseline
                or _identity(_regular_stat(self._path, MAX_FILE_BYTES)) != self._baseline):
            raise ValueError("Synthetic weights changed since validation")

    @staticmethod
    def _validate_codes(data):
        if any(code & 127 == 127 for code in data):
            raise ValueError("Synthetic FP8 payload contains NaN")

    def _verified_read(self, offset, size, digest):
        self._check_unchanged()
        self._stream.seek(offset)
        data = self._read(self._stream, size, payload=True)
        self._check_unchanged()
        if hashlib.sha256(data).digest() != digest:
            raise ValueError("Synthetic requested weight bytes changed since validation")
        self._validate_codes(data)
        return data

    def matrix(self, name: str) -> Matrix:
        if name not in self._manifest["matrices"]:
            raise KeyError(f"Unknown fixed synthetic matrix: {name}")
        entry = self._manifest["matrices"][name]
        data = self._verified_read(entry["offset"], entry["rows"] * entry["cols"],
                                   self._matrix_hashes[name])
        return Matrix(data, entry["rows"], entry["cols"], entry["scale"])

    def vector(self, name: str) -> list[float]:
        self._check_unchanged()
        if name not in self._manifest["vectors"]:
            raise KeyError(f"Unknown fixed synthetic vector: {name}")
        return list(self._manifest["vectors"][name])

    def embedding(self, token: int) -> list[float]:
        if type(token) is not int or not 0 <= token < SPEC["vocab"]:
            raise ValueError("Synthetic token ID must be an integer in [0, 31]")
        entry = self._manifest["matrices"]["embed"]
        data = self._verified_read(entry["offset"] + token * entry["cols"], entry["cols"],
                                   self._row_hashes[token])
        return [_decode(code) * entry["scale"] for code in data]

    def stats(self) -> dict:
        return {"reads": self._reads, "bytes": self._bytes, "max_read_bytes": self._max_read,
                "payload_reads": self._payload_reads, "payload_bytes": self._payload_bytes,
                "resident_weight_bytes": sum(vector_lengths().values()) * 4,
                "resident_fp8_cache_bytes": 0,
                "resident_vector_value_bytes": sum(vector_lengths().values()) * 4,
                "metadata_file_bytes": self._metadata_bytes,
                "validation_digest_bytes": 32 * (len(self._matrix_hashes) + len(self._row_hashes)),
                "resident_weight_scope": "Logical resident vector values; excludes caller matrices, JSON text, Python overhead",
                "synthetic_only": True}

    def close(self):
        if self._stream is not None:
            self._stream.close()

    def __enter__(self):
        self._check_unchanged()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
