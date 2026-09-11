"""Small deterministic E4M3FN fixtures for arithmetic and bounded-I/O probes.

This is a custom synthetic test format, not a checkpoint reader or model graph.
The 24-byte little-endian header is <8sIIII: magic, version, rows, cols, seed.
Tiles follow in block-row, block-column order. Each tile stores a positive f32
scale followed by row-major E4M3FN bytes. Edge tiles have their actual shape.
All reads are unbuffered and explicitly limited to MAX_READ_BYTES; no mmap is
used. The independent dense mathematical oracle never reads fixture files.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct
from typing import BinaryIO, Iterator


BLOCK = 128
MAX_DIM = 1024
MAX_READ_BYTES = BLOCK * BLOCK
MAGIC = b"FP8PROBE"
VERSION = 1
_HEADER = struct.Struct("<8sIIII")
_SCALE = struct.Struct("<f")


@dataclass(frozen=True)
class Tile:
    row_start: int
    col_start: int
    rows: int
    cols: int
    scale: float
    weights: bytes


def _validate_dimensions(rows: int, cols: int, seed: int) -> None:
    for name, value in (("rows", rows), ("cols", cols)):
        if type(value) is not int or not 1 <= value <= MAX_DIM:
            raise ValueError(f"{name} must be an integer from 1 to {MAX_DIM}")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be a uint32 integer")


def _info(rows: int, cols: int, seed: int) -> dict:
    tile_count = ((rows + BLOCK - 1) // BLOCK) * ((cols + BLOCK - 1) // BLOCK)
    return {
        "version": VERSION,
        "rows": rows,
        "cols": cols,
        "seed": seed,
        "file_bytes": _HEADER.size + rows * cols + _SCALE.size * tile_count,
        "max_tile_bytes": min(rows, BLOCK) * min(cols, BLOCK),
        "tile_count": tile_count,
        "weight_format": "E4M3FN",
        "synthetic_only": True,
    }


def _weight_code(seed: int, row: int, col: int) -> int:
    """Coordinate-based integer mixing; independent of traversal or RNG state."""
    mixed = ((seed ^ ((row + 1) * 0x9E3779B1) ^ ((col + 1) * 0x85EBCA77))
             * 0xC2B2AE3D) & 0xFFFFFFFF
    mixed ^= mixed >> 16
    code = mixed & 0xFF
    return code ^ 1 if (code & 0x7F) == 0x7F else code


def _block_scale(seed: int, block_row: int, block_col: int) -> float:
    # Powers of two are represented exactly in the on-disk f32 and the oracle.
    return math.ldexp(1.0, ((seed + block_row * 3 + block_col * 5) % 7) - 10)


def _tiles_geometry(rows: int, cols: int) -> Iterator[tuple[int, int, int, int]]:
    for row_start in range(0, rows, BLOCK):
        for col_start in range(0, cols, BLOCK):
            yield row_start, col_start, min(BLOCK, rows - row_start), min(BLOCK, cols - col_start)


def write_fixture(path: str | Path, rows: int = 384, cols: int = 384, seed: int = 7) -> dict:
    """Exclusively create a fixture; the caller must create the parent directory.

    Existing paths are never overwritten. A write failure can leave a partial
    generated file, which the reader will reject by its size or payload.
    Generation holds only one tile of weights at a time.
    """
    _validate_dimensions(rows, cols, seed)
    info = _info(rows, cols, seed)
    with open(path, "xb") as stream:
        stream.write(_HEADER.pack(MAGIC, VERSION, rows, cols, seed))
        for row_start, col_start, tile_rows, tile_cols in _tiles_geometry(rows, cols):
            scale = _block_scale(seed, row_start // BLOCK, col_start // BLOCK)
            weights = bytes(_weight_code(seed, row, col)
                            for row in range(row_start, row_start + tile_rows)
                            for col in range(col_start, col_start + tile_cols))
            stream.write(_SCALE.pack(scale))
            stream.write(weights)
        stream.flush()
        if os.fstat(stream.fileno()).st_size != info["file_bytes"]:
            raise OSError("Synthetic fixture write did not produce the expected file size")
    return info


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    if type(count) is not int or not 1 <= count <= MAX_READ_BYTES:
        raise ValueError(f"Read length must be from 1 to {MAX_READ_BYTES} bytes")
    data = stream.read(count)
    if len(data) != count:
        raise ValueError(f"Truncated synthetic fixture: expected {count} bytes, read {len(data)}")
    return data


def _read_header(stream: BinaryIO) -> dict:
    magic, version, rows, cols, seed = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    if magic != MAGIC:
        raise ValueError("Not an FP8PROBE synthetic fixture")
    if version != VERSION:
        raise ValueError(f"Unsupported synthetic fixture version: {version}")
    # Validate before computing any payload sizes or allocating a weight buffer.
    _validate_dimensions(rows, cols, seed)
    info = _info(rows, cols, seed)
    actual_size = os.fstat(stream.fileno()).st_size
    if actual_size != info["file_bytes"]:
        raise ValueError(f"Synthetic fixture size mismatch: expected {info['file_bytes']} bytes, "
                         f"found {actual_size}")
    return info


def read_fixture_info(path: str | Path) -> dict:
    """Validate header and exact file size; iter_tiles additionally validates payloads."""
    with open(path, "rb", buffering=0) as stream:
        return _read_header(stream)


def iter_tiles(path: str | Path) -> Iterator[Tile]:
    """Validate and stream small tiles; fully exhaust the iterator for a complete check.

    The file remains open while iterating. Invalid scales and either E4M3FN NaN
    code fail before their tile is yielded. The final size/EOF check also detects
    a trailing append while iterating. This is not an atomic file snapshot.
    """
    with open(path, "rb", buffering=0) as stream:
        info = _read_header(stream)
        for row_start, col_start, rows, cols in _tiles_geometry(info["rows"], info["cols"]):
            scale = _SCALE.unpack(_read_exact(stream, _SCALE.size))[0]
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Synthetic tile scale must be positive and finite")
            weights = _read_exact(stream, rows * cols)
            if 0x7F in weights or 0xFF in weights:
                raise ValueError("Synthetic tile contains an E4M3FN NaN encoding")
            yield Tile(row_start, col_start, rows, cols, scale, weights)
        if stream.read(1):
            raise ValueError("Synthetic fixture has trailing bytes")
        if os.fstat(stream.fileno()).st_size != info["file_bytes"]:
            raise ValueError("Synthetic fixture size changed while reading")


def fixture_vector(cols: int) -> list[float]:
    """A deterministic vector of exactly represented binary fractions."""
    _validate_dimensions(1, cols, 0)
    return [((col * 13 + 3) % 31 - 15) / 16.0 for col in range(cols)]


def _oracle_decode(code: int) -> float:
    """Mathematical E4M3FN decode in Python double, independent of either backend."""
    magnitude = code & 0x7F
    if magnitude == 0x7F:
        raise ValueError("E4M3FN NaN has no finite mathematical reference value")
    exponent, fraction = divmod(magnitude, 8)
    value = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(8 + fraction, exponent - 10)
    return -value if code & 0x80 else value


def reference_matvec(rows: int, cols: int, seed: int) -> list[float]:
    """Compute the dense synthetic mathematical result directly from coordinates.

    No file, tile iterator, CPU DLL, or CUDA backend is used. Products and fsum
    accumulation use Python double, not the sequential FP32 backend reduction.
    The dense matrix is never materialized; only the vector and output are held.
    """
    _validate_dimensions(rows, cols, seed)
    vector = fixture_vector(cols)
    return [math.fsum(
        _oracle_decode(_weight_code(seed, row, col))
        * _block_scale(seed, row // BLOCK, col // BLOCK) * vector[col]
        for col in range(cols)
    ) for row in range(rows)]
