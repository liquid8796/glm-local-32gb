"""Four-shard storage for the exact invented miniature, never a GLM checkpoint.

The official safetensors serializer writes FP8 bytes without converting their
values. Synthetic canonical names are deliberate: they do not assert the real
checkpoint's parameter names, dtypes, expert layout or scale mapping.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

from .fp8_blocks import FP8BlockMatrix
from .mini_spec import SPEC, matrix_shapes, vector_lengths
from .mini_weights import Matrix, MiniWeights, _decode
from .safetensor_reader import SafeTensorError
from .safetensor_serializer import serialize_raw_tensors
from .sharded_safetensors import INDEX_NAME, ShardedSafeTensorReader

MARKER = "glm-synthetic-mini-sharded-v1"
SHARD_NAMES = tuple(f"model-{number:05d}-of-00004.safetensors" for number in range(1, 5))
MAX_FIXTURE_FILE_BYTES = 64 * 1024


def matrix_names(name):
    if name not in matrix_shapes():
        raise KeyError(f"Unknown fixed miniature matrix: {name}")
    return name + ".weight", name + ".weight_scale_inv"


def vector_name(name):
    if name not in vector_lengths():
        raise KeyError(f"Unknown fixed miniature vector: {name}")
    return name + ".value"


def tensor_contract():
    """Explicit fixed mapping; separate FP32 vectors and per-matrix FP32 scales."""
    result = {}
    for name, shape in matrix_shapes().items():
        weight, scale = matrix_names(name)
        result[weight] = ("F8_E4M3", shape)
        result[scale] = ("F32", (1, 1))
    for name, length in vector_lengths().items():
        result[vector_name(name)] = ("F32", (length,))
    return result


def write_mini_shards(source_directory: str | Path, directory: str | Path) -> dict:
    """Export the validated 9,440-byte private fixture, without overwriting files.

    Only this fixed tiny exporter holds all synthetic serialized shard inputs.
    Runtime readers never use the serializer or retain these payloads. A failed
    export may leave partial *new* files; the index is published last.
    """
    try:
        import safetensors
    except ImportError as error:
        raise RuntimeError("Sharded fixture creation needs safetensors; use the storage-validation extra") from error
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir()
    if not directory.is_dir():
        raise ValueError("Synthetic shard destination must be a directory")
    if any(os.path.lexists(directory / name) for name in (*SHARD_NAMES, INDEX_NAME)):
        raise FileExistsError("Synthetic shard/index exists; refusing to overwrite")
    shards = [dict() for _ in SHARD_NAMES]
    mapping = {}
    total_size = 0

    def add(number, name, dtype, shape, payload):
        nonlocal total_size
        if name in mapping:
            raise ValueError("Synthetic tensor mapped more than once")
        shards[number][name] = {"dtype": dtype, "shape": list(shape), "data": payload}
        mapping[name] = SHARD_NAMES[number]
        total_size += len(payload)

    with MiniWeights(source_directory) as source:
        manifest = source.manifest
        for number, name in enumerate(matrix_shapes()):
            matrix = source.matrix(name)
            weight, scale = matrix_names(name)
            # Deliberately put every weight and its scale in different shards.
            add(2 * number % 4, weight, "float8_e4m3fn", (matrix.rows, matrix.cols), matrix.weights)
            add((2 * number + 1) % 4, scale, "float32", (1, 1), struct.pack("<f", matrix.scale))
        for number, name in enumerate(vector_lengths()):
            values = source.vector(name)
            add(number % 4, vector_name(name), "float32", (len(values),),
                struct.pack(f"<{len(values)}f", *values))
    metadata = {"synthetic_fixture": MARKER, "seed": manifest["seed"],
                "source_sha256": manifest["sha256"], "total_size": total_size}
    file_records = []
    for name, tensors in zip(SHARD_NAMES, shards):
        encoded = serialize_raw_tensors(tensors, metadata={
            "synthetic_fixture": MARKER, "seed": str(manifest["seed"]),
            "source_sha256": manifest["sha256"]})
        if len(encoded) > MAX_FIXTURE_FILE_BYTES:
            raise ValueError("Serialized miniature shard exceeds its 64-KiB bound")
        with (directory / name).open("xb", buffering=0) as stream:
            if stream.write(encoded) != len(encoded):
                raise OSError("Incomplete synthetic safetensors write")
        file_records.append({"name": name, "bytes": len(encoded),
                             "sha256": hashlib.sha256(encoded).hexdigest()})
    index = (json.dumps({"metadata": metadata, "weight_map": mapping},
                       indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(index) > MAX_FIXTURE_FILE_BYTES:
        raise ValueError("Synthetic shard index exceeds its 64-KiB bound")
    with (directory / INDEX_NAME).open("xb", buffering=0) as stream:
        if stream.write(index) != len(index):
            raise OSError("Incomplete synthetic index write")
    return {"format": MARKER, "synthetic_only": True, "shard_count": 4,
            "tensor_count": len(mapping), "tensor_payload_bytes": total_size,
            "index_bytes": len(index), "index_sha256": hashlib.sha256(index).hexdigest(),
            "source_fixture_sha256": manifest["sha256"], "files": file_records,
            "writer": "safetensors.serialize", "safetensors_version": safetensors.__version__,
            "serializer_api": "TensorSpec" if hasattr(safetensors, "TensorSpec") else "raw-dict",
            "weight_scale_pairs_cross_shards": True,
            "hash_scope": "identities of generated fixture files, not external checkpoint authentication"}


class MiniSafetensorWeights:
    """MiniWeights-compatible reader restricted to the exact tiny four-shard spec.

    Vectors are read on demand too. Every matrix is smaller than one 128x128
    block; larger shapes are rejected, not silently treated as single-scale.
    The original NumPy/official oracles continue reading the private fixture so
    native-vs-reference validation does not share this new loader.
    """

    def __init__(self, directory: str | Path, *, max_open_shards: int = 2):
        directory = Path(directory)
        # Check fixed filenames/sizes before any generic index/header parsing.
        for filename in (*SHARD_NAMES, INDEX_NAME):
            path = directory / filename
            if not path.is_file() or not 1 <= path.stat().st_size <= MAX_FIXTURE_FILE_BYTES:
                raise SafeTensorError("Miniature shard/index missing or outside its 64-KiB file bound")
        self._reader = ShardedSafeTensorReader(directory, max_open_shards=max_open_shards,
                                                expected_shards=SHARD_NAMES)
        try:
            metadata = self._reader.metadata
            if (set(metadata) != {"synthetic_fixture", "seed", "source_sha256", "total_size"}
                    or metadata["synthetic_fixture"] != MARKER
                    or type(metadata["seed"]) is not int or not 0 <= metadata["seed"] <= 0xFFFFFFFF
                    or not isinstance(metadata["source_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", metadata["source_sha256"])):
                raise SafeTensorError("Not the fixed synthetic sharded miniature metadata")
            if set(self._reader.weight_map.values()) != set(SHARD_NAMES):
                raise SafeTensorError("Miniature requires the four exact generated shard names")
            expected_marker = {"synthetic_fixture": MARKER, "seed": str(metadata["seed"]),
                               "source_sha256": metadata["source_sha256"]}
            if any(value != expected_marker for value in self._reader.shard_metadata.values()):
                raise SafeTensorError("Miniature shard provenance markers disagree")
            contract = tensor_contract()
            if set(self._reader.tensors) != set(contract):
                raise SafeTensorError("Synthetic matrix/scale/vector names differ from the fixed mapping")
            for name, (dtype, shape) in contract.items():
                info = self._reader.tensors[name]
                if (info.dtype, info.shape) != (dtype, shape):
                    raise SafeTensorError(f"Synthetic dtype/shape mismatch: {name}")
            self._matrices = {
                name: FP8BlockMatrix(self._reader, *matrix_names(name)) for name in matrix_shapes()}
        except BaseException:
            self.close()
            raise

    def matrix(self, name: str) -> Matrix:
        if name not in self._matrices:
            raise KeyError(f"Unknown fixed miniature matrix: {name}")
        block = self._matrices[name].read_block(0, 0)
        return Matrix(block.weights, block.rows, block.cols, block.scale)

    def vector(self, name: str) -> list[float]:
        tensor = vector_name(name)
        length = vector_lengths()[name]
        raw = self._reader.read_bytes(tensor, 0, length * 4)
        values = list(struct.unpack(f"<{length}f", raw))
        if any(not math.isfinite(value) for value in values):
            raise SafeTensorError("Synthetic vector values must be finite")
        return values

    def embedding(self, token: int) -> list[float]:
        if type(token) is not int or not 0 <= token < SPEC["vocab"]:
            raise ValueError("Synthetic token ID must be an integer in [0, 31]")
        weight_name, scale_name = matrix_names("embed")
        raw_scale = self._reader.read_bytes(scale_name, 0, 4)
        scale = struct.unpack("<f", raw_scale)[0]
        if not math.isfinite(scale) or scale <= 0:
            raise SafeTensorError("Embedding scale must be positive and finite")
        raw = self._reader.read_bytes(weight_name, token * SPEC["hidden"], SPEC["hidden"])
        return [_decode(code) * scale for code in raw]

    def stats(self):
        return {**self._reader.stats(), "format": MARKER, "synthetic_only": True,
                "resident_weight_bytes": 0, "resident_vector_value_bytes": 0,
                "resident_weight_scope": "No retained payload; excludes metadata, caller buffers and Python overhead",
                "fixed_mapping_verified": True,
                "cross_shard_pairs": sum(self._reader.weight_map[weight] != self._reader.weight_map[scale]
                                         for weight, scale in (matrix_names(n) for n in matrix_shapes()))}

    def close(self):
        self._reader.close()

    def __enter__(self):
        self._reader.__enter__()
        return self

    def __exit__(self, *_):
        self.close()
