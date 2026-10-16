"""Bounded CPU reference for Flash W4A16 compressed-tensors expert projections.

This is NOT a GLM5Next decoder. The reader owns the local safetensors shard and
supplies read_bytes(name, offset, length); no packed matrix or expert bank is
loaded. Intended for offline weight layout/numerical validation on small cases.
"""
from __future__ import annotations

import math
import struct

from .safetensor_reader import MAX_READ_BYTES

GROUP_SIZE = 32
PACK_FACTOR = 8              # 8 signed 4-bit values in one I32 word
MAX_GROUPS_PER_READ = 128     # <= 2048 packed bytes + <= 512 scale bytes
MAX_VECTOR_ELEMENTS = 1_048_576


class W4A16Projection:
    """Validate one symmetric INT4 expert and multiply it by an FP vector.

    Layout: I32 weight_packed[out, in/8], (BF16 or F32) weight_scale[out, in/32],
    I64 weight_shape[2], packed along the input dimension. Nibble 0 is
    stored in the low four bits; nibble values >=8 represent negatives.
    """

    def __init__(self, reader, prefix: str):
        if not isinstance(prefix, str) or not prefix or len(prefix) > 1024:
            raise ValueError("Invalid W4A16 projection name")
        self._reader = reader
        self._packed = prefix + ".weight_packed"
        self._scale = prefix + ".weight_scale"
        shape_name = prefix + ".weight_shape"
        try:
            packed = reader.tensors[self._packed]
            scale = reader.tensors[self._scale]
            shape = reader.tensors[shape_name]
        except (KeyError, AttributeError, TypeError) as error:
            raise ValueError("Missing W4A16 weight, scale or shape metadata") from error
        if (shape.dtype != "I64" or tuple(shape.shape) != (2,)
                or shape.itemsize != 8 or shape.nbytes != 16):
            raise ValueError("Expected I64 W4A16 weight_shape[2]")
        raw = reader.read_bytes(shape_name, 0, 16)
        if not isinstance(raw, bytes) or len(raw) != 16:
            raise ValueError("Invalid W4A16 weight_shape payload")
        rows, cols = struct.unpack("<qq", raw)
        if (rows <= 0 or cols <= 0 or cols > MAX_VECTOR_ELEMENTS
                or cols % GROUP_SIZE != 0 or rows > 1_048_576):
            raise ValueError("Unsupported W4A16 shape or group alignment")
        if (packed.dtype != "I32" or packed.itemsize != 4
                or tuple(packed.shape) != (rows, cols // PACK_FACTOR)
                or packed.nbytes != rows * (cols // PACK_FACTOR) * 4):
            raise ValueError("Invalid INT4 packed weight geometry")
        scale_size = {"BF16": 2, "F32": 4}.get(scale.dtype)
        if (scale_size is None or scale.itemsize != scale_size
                or tuple(scale.shape) != (rows, cols // GROUP_SIZE)
                or scale.nbytes != rows * (cols // GROUP_SIZE) * scale_size):
            raise ValueError("Invalid group-wise W4A16 scale geometry")
        self._scale_size = scale_size
        self.rows, self.cols = rows, cols

    def matvec(self, vector, *, first_row=0, row_count=None):
        """Return FP64-reference dot products, streaming <=128 groups at once.

        FP64 accumulation is deliberate for reference comparison, not a claim
        of parity with production BF16/CUDA execution.
        """
        if (type(first_row) is not int or not 0 <= first_row < self.rows
                or (row_count is not None and
                    (type(row_count) is not int or row_count <= 0 or
                     first_row + row_count > self.rows))):
            raise ValueError("Invalid requested W4A16 rows")
        if row_count is None:
            row_count = self.rows - first_row
        if len(vector) != self.cols:
            raise ValueError("Activation width differs from W4A16 weight_shape")
        values = [float(v) for v in vector]
        if any(not math.isfinite(v) for v in values):
            raise ValueError("W4A16 activations must be finite")
        out = []
        groups = self.cols // GROUP_SIZE
        packed_row_bytes = self.cols // 2
        scale_row_bytes = groups * self._scale_size
        for row in range(first_row, first_row + row_count):
            total = 0.0
            for start in range(0, groups, MAX_GROUPS_PER_READ):
                take = min(MAX_GROUPS_PER_READ, groups - start)
                weight_bytes = take * GROUP_SIZE // 2
                scale_bytes = take * self._scale_size
                assert weight_bytes <= MAX_READ_BYTES and scale_bytes <= MAX_READ_BYTES
                encoded = self._reader.read_bytes(
                    self._packed, row * packed_row_bytes + start * GROUP_SIZE // 2,
                    weight_bytes)
                scales = self._reader.read_bytes(
                    self._scale, row * scale_row_bytes + start * self._scale_size, scale_bytes)
                if (not isinstance(encoded, bytes) or len(encoded) != weight_bytes
                        or not isinstance(scales, bytes) or len(scales) != scale_bytes):
                    raise ValueError("Incomplete W4A16 tensor payload")
                for g in range(take):
                    scale_raw = scales[g * self._scale_size:(g + 1) * self._scale_size]
                    if self._scale_size == 2:
                        scale_raw = b"\x00\x00" + scale_raw
                    scale = struct.unpack("<f", scale_raw)[0]
                    if not math.isfinite(scale) or scale < 0:
                        raise ValueError("Non-finite or negative W4A16 scale")
                    begin = (start + g) * GROUP_SIZE
                    for k in range(GROUP_SIZE):
                        code = encoded[g * 16 + k // 2]
                        nibble = ((code >> 4) if k % 2 else code) & 15
                        signed = nibble - 16 if nibble >= 8 else nibble
                        total += signed * values[begin + k] * scale
            out.append(total)
        return out
