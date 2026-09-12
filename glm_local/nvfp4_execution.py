"""Bounded ModelOpt NVFP4 projections using decoded weights and FP32 inputs.

This storage-format fallback does not reproduce W4A4 activation rounding or
native FP4 tensor-core execution. The input calibration scalar is verified but
never folded into the decoded weights.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
import struct

from .cpu_probe import _finite_float32
from .execution import (ProjectionDescriptor, TensorDescriptor, FULL_MODEL_FLAGS,
                        MAX_VECTOR_ELEMENTS, MAX_PROJECTION_TILES, MAX_REFERENCE_ELEMENTS,
                        _vector, _reference_decode)
from .nvfp4_blocks import NVFP4BlockMatrix, TILE


@dataclass(frozen=True)
class NVFP4ProjectionDescriptor(ProjectionDescriptor):
    global_scale: TensorDescriptor
    input_scale: TensorDescriptor

    @property
    def tensors(self):
        return self.weight, self.scale, self.global_scale, self.input_scale

    @property
    def logical_shape(self):
        return self.weight.shape[0], self.weight.shape[1] * 2

    @property
    def quant_format(self):
        return "nvfp4"

    def to_dict(self):
        return {**asdict(self), "format_version": 1, "selected_projection_eligible": True,
                "quant_format": "nvfp4", "logical_shape": list(self.logical_shape),
                "packing": "low nibble even column, high nibble odd column; E2M1",
                "weight_scale_semantics": "E2M1 * (E4M3FN per-row group16 scale * F32 global weight scale)",
                "scale_group_size": 16, "activation_quantization": "none",
                "input_scale_usage": "validated calibration metadata; not applied in FP32-input fallback",
                "native_w4a4_parity_verified": False, "payload_values_verified": False,
                "provenance_scope": "Local metadata digests and selected headers; no full payload checksum attestation",
                **FULL_MODEL_FLAGS}


def _matrix(reader, descriptor):
    if not isinstance(descriptor, NVFP4ProjectionDescriptor):
        raise ValueError("Expected an explicit NVFP4 projection descriptor")
    for item in descriptor.tensors:
        if reader.tensors.get(item.name) != item.info():
            raise ValueError("NVFP4 reader tensor metadata differs from descriptor")
    matrix = NVFP4BlockMatrix(reader, *(item.name for item in descriptor.tensors))
    if max(matrix.rows, matrix.cols) > MAX_VECTOR_ELEMENTS:
        raise ValueError("NVFP4 projection exceeds bounded vector policy")
    if matrix.block_rows * matrix.block_cols > MAX_PROJECTION_TILES:
        raise ValueError("NVFP4 projection exceeds bounded tile policy")
    return matrix


def execute_nvfp4_projection(reader, descriptor, vector, *, backend="cpu", cpu=None,
                             gpu=None, gate=None, ledger=None):
    matrix = _matrix(reader, descriptor)
    if backend not in ("cpu", "hybrid") or not callable(getattr(cpu, "matvec_nvfp4_tile", None)):
        raise ValueError("NVFP4 execution requires cpu/hybrid backend and an explicit NVFP4 CPU kernel")
    gpu_tiles = matrix.block_rows // 2 * matrix.block_cols if backend == "hybrid" else 0
    if backend == "hybrid":
        if not gpu_tiles or not callable(getattr(gpu, "matvec_nvfp4_tile", None)) or gate is None:
            raise ValueError("Hybrid NVFP4 projection requires both row paths, GPU kernel and admission gate")
        if gpu_tiles > getattr(gpu, "max_operations", 256) - getattr(gpu, "operations", 0):
            raise ValueError("NVFP4 projection exceeds remaining CUDA operation budget")
    host_bytes = 3 * (TILE * TILE // 2 + TILE * TILE // 16) + 4 * (matrix.rows + matrix.cols + 4 * TILE) + 8
    with ledger.reserve(host_bytes, label="nvfp4_projection") if ledger else nullcontext():
        values = _vector(vector, matrix.cols)
        result = [0.0] * matrix.rows
        counts = {"cpu_tiles": 0, "gpu_tiles": 0, "packed_weight_bytes": 0, "block_scale_bytes": 0,
                  "max_packed_tile_bytes": 0, "logical_host_buffer_bytes": host_bytes,
                  "retained_decoded_weight_bytes": 0, "activation_quantization": "none",
                  "input_scale_validated": True, "native_w4a4_parity_verified": False, **FULL_MODEL_FLAGS}
        for block in matrix.iter_blocks():
            use_gpu = backend == "hybrid" and block.row_start // TILE % 2 == 1
            device_bytes = len(block.weights) + len(block.scales) + 4 * (block.rows + block.cols)
            with ledger.reserve(device_bytes, device="cuda", label="nvfp4_tile") if ledger and use_gpu else nullcontext():
                if use_gpu:
                    gate.before_submit()
                output = (gpu if use_gpu else cpu).matvec_nvfp4_tile(block.weights, block.rows, block.cols,
                    block.scales, values[block.col_start:block.col_start + block.cols], block.global_scale)
                if not hasattr(output, "__len__") or len(output) != block.rows:
                    raise ValueError("NVFP4 kernel returned an invalid output length")
                for row in range(block.rows):
                    index = block.row_start + row
                    result[index] = _finite_float32(result[index] + _finite_float32(output[row], "NVFP4 output"), "NVFP4 accumulation")
            counts["gpu_tiles" if use_gpu else "cpu_tiles"] += 1
            counts["packed_weight_bytes"] += len(block.weights)
            counts["block_scale_bytes"] += len(block.scales)
            counts["max_packed_tile_bytes"] = max(counts["max_packed_tile_bytes"], len(block.weights))
    return result, counts


def reference_nvfp4_projection(reader, descriptor, vector):
    """Independent row reads and format arithmetic, without the tile adapter/kernel."""
    rows, cols = descriptor.logical_shape
    _matrix(reader, descriptor)
    if rows * cols > MAX_REFERENCE_ELEMENTS:
        raise ValueError("NVFP4 scalar reference exceeds bounded operation count")
    values = _vector(vector, cols)
    def scalar(item):
        raw = reader.read_bytes(item.name, 0, 4)
        if not isinstance(raw, bytes) or len(raw) != 4:
            raise ValueError("Reference requires four scalar bytes")
        return _finite_float32(struct.unpack("<f", raw)[0], "NVFP4 reference scale", positive=True)
    global_scale = scalar(descriptor.global_scale)
    scalar(descriptor.input_scale)
    output = [0.0] * rows
    for row in range(rows):
        total = 0.0
        for col in range(0, cols, TILE):
            count = min(TILE, cols - col)
            packed = reader.read_bytes(descriptor.weight.name, (row * cols + col) // 2, count // 2)
            scales = reader.read_bytes(descriptor.scale.name, (row * cols + col) // 16, count // 16)
            if not isinstance(packed, bytes) or len(packed) != count // 2 or not isinstance(scales, bytes) or len(scales) != count // 16:
                raise ValueError("Reference packed weight/scale read returned the wrong byte count")
            partial = 0.0
            for column in range(count):
                code = (packed[column // 2] >> (4 * (column % 2))) & 15
                exponent, fraction = (code & 7) >> 1, code & 1
                magnitude = fraction * 0.5 if exponent == 0 else math.ldexp(1 + fraction * 0.5, exponent - 1)
                value = -magnitude if code & 8 else magnitude
                scale_code = scales[column // 16]
                if scale_code >= 127:
                    raise ValueError("Reference block scale must be unsigned finite E4M3FN")
                combined = _finite_float32(_reference_decode(scale_code) * global_scale, "reference combined scale")
                weight = _finite_float32(value * combined, "reference decoded NVFP4 weight")
                product = _finite_float32(weight * values[col + column], "reference NVFP4 product")
                partial = _finite_float32(partial + product, "reference NVFP4 reduction")
            total = _finite_float32(total + partial, "reference NVFP4 sum")
        output[row] = total
    return output
