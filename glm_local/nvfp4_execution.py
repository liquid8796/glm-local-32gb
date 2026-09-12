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
from .nvfp4_blocks import BLOCK, NVFP4BlockMatrix, TILE, decode_e4m3_scale
from .safetensor_reader import MAX_READ_BYTES

NVFP4_ROW_BAND_SCRATCH_BYTES = 5 * 1024**2
MAX_ROW_BAND_COLUMNS = 16384
_MAX_ENCODED_BAND_BYTES = TILE * (MAX_ROW_BAND_COLUMNS // 2 + MAX_ROW_BAND_COLUMNS // BLOCK)


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
        if (backend == "cpu" and matrix.cols <= MAX_ROW_BAND_COLUMNS
                and callable(getattr(cpu, "matvec_nvfp4_row_band", None))
                and getattr(cpu, "supports_nvfp4_row_band", True)):
            with ledger.reserve(NVFP4_ROW_BAND_SCRATCH_BYTES, label="nvfp4_row_band_scratch") if ledger else nullcontext():
                return _execute_cpu_row_bands(reader, descriptor, matrix, vector, cpu, host_bytes)
        values = _vector(vector, matrix.cols)
        result = [0.0] * matrix.rows
        counts = {"cpu_tiles": 0, "gpu_tiles": 0, "packed_weight_bytes": 0, "block_scale_bytes": 0,
                  "max_packed_tile_bytes": 0, "logical_host_buffer_bytes": host_bytes,
                  "cpu_row_bands": 0, "native_row_band_calls": 0, "native_row_band_batching": False,
                  "row_band_read_calls": 0, "row_band_read_bytes": 0, "row_band_scalar_reads": 0,
                  "max_row_band_read_bytes": 0, "max_encoded_band_bytes": 0,
                  "max_packed_band_bytes": 0, "row_band_scratch_bytes": 0,
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


def _read_encoded_band(reader, name, offset, count, counts):
    """Gather one leased band using only bounded tensor-relative reads."""
    if type(offset) is not int or offset < 0 or type(count) is not int or not 1 <= count <= _MAX_ENCODED_BAND_BYTES:
        raise ValueError("NVFP4 encoded row band exceeds its bounded byte policy")
    data = bytearray(count)
    for at in range(0, count, MAX_READ_BYTES):
        size = min(MAX_READ_BYTES, count - at)
        raw = reader.read_bytes(name, offset + at, size)
        if type(raw) is not bytes or len(raw) != size:
            raise ValueError("NVFP4 encoded row-band read returned an incorrect byte count")
        data[at:at + size] = raw
        counts["row_band_read_calls"] += 1
        counts["row_band_read_bytes"] += size
        counts["max_row_band_read_bytes"] = max(counts["max_row_band_read_bytes"], size)
        del raw
    return bytes(data)


def _execute_cpu_row_bands(reader, descriptor, matrix, vector, cpu, host_bytes):
    if isinstance(vector, (str, bytes, bytearray)) or not hasattr(vector, "__len__") or len(vector) != matrix.cols:
        raise ValueError("Input vector must have exactly the selected matrix column count")
    prepare = getattr(cpu, "prepare_nvfp4_vector", None)
    values = prepare(vector) if callable(prepare) else _vector(vector, matrix.cols)
    if isinstance(values, (str, bytes, bytearray)) or not hasattr(values, "__len__") or len(values) != matrix.cols:
        raise ValueError("Prepared NVFP4 vector must preserve the selected matrix column count")
    result = [0.0] * matrix.rows
    counts = {"cpu_tiles": matrix.block_rows * matrix.block_cols, "gpu_tiles": 0,
              "cpu_row_bands": 0, "native_row_band_calls": 0, "native_row_band_batching": True,
              "packed_weight_bytes": 0, "block_scale_bytes": 0, "max_packed_band_bytes": 0,
              "row_band_read_calls": 0, "row_band_read_bytes": 0, "row_band_scalar_reads": 0,
              "max_row_band_read_bytes": 0, "max_encoded_band_bytes": 0,
              "max_packed_tile_bytes": min(TILE, matrix.rows) * min(TILE, matrix.cols) // 2,
              "logical_host_buffer_bytes": host_bytes + NVFP4_ROW_BAND_SCRATCH_BYTES,
              "row_band_scratch_bytes": NVFP4_ROW_BAND_SCRATCH_BYTES,
              "retained_encoded_weight_bytes": 0,
              "retained_decoded_weight_bytes": 0, "activation_quantization": "none",
              "input_scale_validated": True, "native_w4a4_parity_verified": False, **FULL_MODEL_FLAGS}
    for row in range(0, matrix.rows, TILE):
        rows = min(TILE, matrix.rows - row)
        global_scale = matrix._scalar(descriptor.global_scale.name, "global weight scale")
        matrix._scalar(descriptor.input_scale.name, "input scale")
        counts["row_band_scalar_reads"] += 2
        packed = scales = output = None
        try:
            packed = _read_encoded_band(reader, descriptor.weight.name, row * (matrix.cols // 2), rows * (matrix.cols // 2), counts)
            scales = _read_encoded_band(reader, descriptor.scale.name, row * (matrix.cols // BLOCK), rows * (matrix.cols // BLOCK), counts)
            # The byte fast path avoids re-decoding valid scales in Python, while an
            # invalid byte still follows the same validation as NVFP4BlockMatrix.
            if not scales.isascii() or b"\x7f" in scales:
                for code in scales:
                    decode_e4m3_scale(code)
            counts["native_row_band_calls"] += 1
            output = cpu.matvec_nvfp4_row_band(packed, rows, matrix.cols, scales, values, global_scale)
            if not hasattr(output, "__len__") or len(output) != rows:
                raise ValueError("NVFP4 row-band kernel returned an invalid output length")
            for index in range(rows):
                result[row + index] = _finite_float32(output[index], "NVFP4 row-band output")
            counts["cpu_row_bands"] += 1
            counts["packed_weight_bytes"] += len(packed)
            counts["block_scale_bytes"] += len(scales)
            counts["max_packed_band_bytes"] = max(counts["max_packed_band_bytes"], len(packed))
            counts["max_encoded_band_bytes"] = max(counts["max_encoded_band_bytes"], len(packed) + len(scales))
        finally:
            # Do not carry prior row-band buffers into the next read/allocation.
            packed = scales = output = None
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
