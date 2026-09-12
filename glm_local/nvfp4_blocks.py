"""Bounded row-major ModelOpt NVFP4 weight tiles; no activation quantization.

ModelOpt source, revision 51de53e48ccae8804f8fe1198b7cf89475c5c4f4:
https://github.com/NVIDIA/Model-Optimizer/blob/51de53e48ccae8804f8fe1198b7cf89475c5c4f4/modelopt/torch/quantization/qtensor/nvfp4_tensor.py

Each byte holds the even column in its low nibble and the odd column in its
high nibble. A row shares one E4M3FN scale per 16 logical columns, and the
matrix has one multiplicative FP32 global weight scale. Stored input_scale
is separate activation calibration data: validated here, never multiplied
into weights. This decoded-weight reference uses unquantized FP32 inputs.
HF scales are unswizzled; hardware-specific post-load layouts are unsupported.
"""

from dataclasses import dataclass
import math
import struct
from typing import Iterator


BLOCK = 16
TILE = 128
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_e4m3_scale(code: int) -> float:
    """Decode a nonnegative finite E4M3FN scale, allowing an all-zero block."""
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError("NVFP4 scale code must be an integer byte")
    if (code & 127) == 127:
        raise ValueError("NVFP4 block scale contains an E4M3FN NaN encoding")
    if code & 128:
        raise ValueError("NVFP4 block scale must be nonnegative with an unsigned sign bit")
    exponent, fraction = code >> 3, code & 7
    return math.ldexp(float(fraction), -9) if exponent == 0 else math.ldexp(float(8 + fraction), exponent - 10)


def _matrix_shape(info, label):
    shape = info.shape
    if (not isinstance(shape, tuple) or len(shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)):
        raise ValueError(f"{label} must have exactly two positive integer dimensions")
    return shape


@dataclass(frozen=True)
class NVFP4Block:
    row_start: int
    col_start: int
    rows: int
    cols: int
    weights: bytes
    scales: bytes
    global_scale: float
    input_scale: float


class NVFP4BlockMatrix:
    """Validate explicit tensor names and lazily read at most 128 by 128 tiles.

    Shape/byte metadata is checked on construction. Payload values are checked
    only as each tile is read. Nothing allocates or retains a full matrix.
    A maximum tile reads 8192 weight bytes, 1024 scale bytes, and two scalars.
    """

    activation_quantization = "none"

    def __init__(self, reader, weight_name, scale_name, global_scale_name, input_scale_name):
        names = (weight_name, scale_name, global_scale_name, input_scale_name)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 4:
            raise ValueError("NVFP4 weight and three scale names must be explicit and distinct")
        try:
            weight, scale, global_scale, input_scale = (reader.tensors[name] for name in names)
        except KeyError as error:
            raise ValueError("Named NVFP4 weight or scale tensor is absent") from error
        if weight.dtype != "U8" or weight.itemsize != 1:
            raise ValueError("NVFP4 packed weight tensor must have U8 dtype")
        rows, packed_cols = _matrix_shape(weight, "NVFP4 weight tensor")
        cols = 2 * packed_cols
        if cols % BLOCK:
            raise ValueError("NVFP4 logical input columns must be a multiple of 16")
        if weight.nbytes != rows * packed_cols:
            raise ValueError("NVFP4 weight byte size is inconsistent with its shape")
        if scale.dtype != "F8_E4M3" or scale.itemsize != 1:
            raise ValueError("NVFP4 block scale tensor must have F8_E4M3 dtype")
        if _matrix_shape(scale, "NVFP4 block scale tensor") != (rows, cols // BLOCK):
            raise ValueError("NVFP4 block scale shape must be (rows, logical columns / 16)")
        if scale.nbytes != rows * (cols // BLOCK):
            raise ValueError("NVFP4 block scale byte size is inconsistent with its shape")
        for info, label in ((global_scale, "global weight scale"), (input_scale, "input scale")):
            if (info.dtype != "F32" or info.itemsize != 4 or info.nbytes != 4
                    or not isinstance(info.shape, tuple) or info.shape not in ((), (1,))
                    or any(type(dimension) is not int for dimension in info.shape)):
                raise ValueError(f"NVFP4 {label} must be one F32 scalar with shape () or (1,)")
        self._reader = reader
        self._weight_name, self._scale_name, self._global_scale_name, self._input_scale_name = names
        self.rows, self.cols = rows, cols
        self.block_rows, self.block_cols = (rows + TILE - 1) // TILE, (cols + TILE - 1) // TILE

    def _scalar(self, name, label):
        raw = self._reader.read_bytes(name, 0, 4)
        if not isinstance(raw, bytes) or len(raw) != 4:
            raise ValueError(f"NVFP4 {label} read must return exactly four bytes")
        value = struct.unpack("<f", raw)[0]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"NVFP4 {label} must be positive and finite")
        return value

    def read_block(self, br: int, bc: int) -> NVFP4Block:
        if type(br) is not int or not 0 <= br < self.block_rows:
            raise ValueError("NVFP4 block row must be an integer within the tile grid")
        if type(bc) is not int or not 0 <= bc < self.block_cols:
            raise ValueError("NVFP4 block column must be an integer within the tile grid")
        global_scale = self._scalar(self._global_scale_name, "global weight scale")
        input_scale = self._scalar(self._input_scale_name, "input scale")
        row, col = br * TILE, bc * TILE
        rows, cols = min(TILE, self.rows - row), min(TILE, self.cols - col)
        scales = self._reader.read_matrix_tile(self._scale_name, row, col // BLOCK, rows, cols // BLOCK)
        if not isinstance(scales, bytes) or len(scales) != rows * (cols // BLOCK):
            raise ValueError("NVFP4 scale tile read returned an incorrect byte count or type")
        for code in scales:
            decode_e4m3_scale(code)
        weights = self._reader.read_matrix_tile(self._weight_name, row, col // 2, rows, cols // 2)
        if not isinstance(weights, bytes) or len(weights) != rows * (cols // 2):
            raise ValueError("NVFP4 weight tile read returned an incorrect byte count or type")
        return NVFP4Block(row, col, rows, cols, weights, scales, global_scale, input_scale)

    def iter_blocks(self) -> Iterator[NVFP4Block]:
        for br in range(self.block_rows):
            for bc in range(self.block_cols):
                yield self.read_block(br, bc)
