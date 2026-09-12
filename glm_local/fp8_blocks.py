"""Bounded 128 by 128 FP8 blocks from an explicitly named weight/scale pair.

Weights are safetensors ``F8_E4M3`` (E4M3FN) matrices; scales are an F32
matrix with ceil(rows / 128) by ceil(cols / 128) entries. ``scale`` is the
stored multiplicative dequantization factor: decoded_fp8 * scale, including
when a checkpoint calls that factor ``weight_scale_inv``.

The caller owns the already validated SafeTensorReader and its lifetime.
This adapter never loads a whole tensor, discovers tensor names, allocates a
model, or invokes a compute backend. Each block reads exactly four scale
bytes and at most 16384 packed weight bytes. Iteration retains no blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .safetensor_reader import SafeTensorReader


BLOCK = 128
_SCALE = struct.Struct("<f")


@dataclass(frozen=True)
class Block:
    row_start: int
    col_start: int
    rows: int
    cols: int
    scale: float
    weights: bytes


def _matrix_shape(info, label: str) -> tuple[int, int]:
    shape = info.shape
    if (not isinstance(shape, tuple) or len(shape) != 2
            or any(type(dim) is not int or dim <= 0 for dim in shape)):
        raise ValueError(f"{label} must have exactly two positive integer dimensions")
    return shape


class FP8BlockMatrix:
    """Validate a named matrix pair and read its blocks lazily.

    ``block_rows`` and ``block_cols`` are scale-grid counts, not tile sizes.
    The reader validates file metadata and enforces its own resource limits.
    This adapter additionally rejects nonfinite weights and invalid scales
    before passing any block to its caller. Other blocks remain unchecked
    until read; fully consume ``iter_blocks`` to validate every payload.
    """

    def __init__(self, reader: SafeTensorReader, weight_name: str, scale_name: str):
        if (not isinstance(weight_name, str) or not weight_name
                or not isinstance(scale_name, str) or not scale_name):
            raise ValueError("Weight and scale tensor names must be explicit nonempty strings")
        if weight_name == scale_name:
            raise ValueError("Weight and scale tensor names must be different")
        try:
            weight = reader.tensors[weight_name]
            scale = reader.tensors[scale_name]
        except KeyError as exc:
            raise ValueError("Named weight or scale tensor is absent from the reader") from exc

        if weight.dtype != "F8_E4M3":
            raise ValueError("Weight tensor dtype must be F8_E4M3 (E4M3FN)")
        if scale.dtype != "F32":
            raise ValueError("Scale tensor dtype must be F32")
        rows, cols = _matrix_shape(weight, "Weight tensor")
        scale_rows, scale_cols = _matrix_shape(scale, "Scale tensor")
        block_rows = (rows + BLOCK - 1) // BLOCK
        block_cols = (cols + BLOCK - 1) // BLOCK
        if (scale_rows, scale_cols) != (block_rows, block_cols):
            raise ValueError("Scale shape must be (ceil(weight rows / 128), ceil(weight cols / 128))")
        if weight.itemsize != 1 or weight.nbytes != rows * cols:
            raise ValueError("Weight tensor byte size is inconsistent with its shape and dtype")
        if scale.itemsize != _SCALE.size or scale.nbytes != scale_rows * scale_cols * _SCALE.size:
            raise ValueError("Scale tensor byte size is inconsistent with its shape and dtype")

        self._reader = reader
        self._weight_name = weight_name
        self._scale_name = scale_name
        self.rows = rows
        self.cols = cols
        self.block_rows = block_rows
        self.block_cols = block_cols

    def read_block(self, br: int, bc: int) -> Block:
        """Read one row-major packed block using block-grid coordinates."""
        if type(br) is not int or not 0 <= br < self.block_rows:
            raise ValueError("Block row must be an integer within the scale grid")
        if type(bc) is not int or not 0 <= bc < self.block_cols:
            raise ValueError("Block column must be an integer within the scale grid")

        scale_offset = (br * self.block_cols + bc) * _SCALE.size
        encoded_scale = self._reader.read_bytes(self._scale_name, scale_offset, _SCALE.size)
        if not isinstance(encoded_scale, bytes) or len(encoded_scale) != _SCALE.size:
            raise ValueError("Scale read must return exactly four bytes")
        scale = _SCALE.unpack(encoded_scale)[0]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("FP8 block scale must be positive and finite")

        row_start = br * BLOCK
        col_start = bc * BLOCK
        rows = min(BLOCK, self.rows - row_start)
        cols = min(BLOCK, self.cols - col_start)
        weights = self._reader.read_matrix_tile(
            self._weight_name, row_start, col_start, rows, cols)
        if not isinstance(weights, bytes) or len(weights) != rows * cols:
            raise ValueError("Weight block read returned an incorrect byte count or byte type")
        if 0x7F in weights or 0xFF in weights:
            raise ValueError("FP8 block contains an E4M3FN NaN encoding")
        return Block(row_start, col_start, rows, cols, scale, weights)

    def iter_blocks(self) -> Iterator[Block]:
        """Yield blocks in block-row, then block-column order without caching."""
        for br in range(self.block_rows):
            for bc in range(self.block_cols):
                yield self.read_block(br, bc)
