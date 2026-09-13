"""CPU projections that reuse one encoded band across a bounded input batch."""
from array import array
from contextlib import nullcontext
import math

from .cpu_probe import _finite_float32
from .nvfp4_execution import NVFP4_ROW_BAND_SCRATCH_BYTES, _matrix
from .safetensor_reader import MAX_READ_BYTES, SafeTensorError
from .runtime_read_ahead import clear_error_frames

DENSE_ROW_BAND_SCRATCH_BYTES = 33 * 1024**2
MAX_LINEAR_BATCH = 16
MAX_BAND_COLUMNS = 16384
BAND_ROWS = 128


def read_span(reader, name, offset, count):
    if not 1 <= count <= 8 * 1024**2:
        raise SafeTensorError("Runtime band exceeds the 8-MiB encoded read bound")
    operation = getattr(reader, "read_span", None)
    if callable(operation):
        data = operation(name, offset, count)
        if not isinstance(data, (bytes, bytearray)) or len(data) != count:
            raise SafeTensorError("Runtime span returned an incorrect byte count")
        return data
    data = bytearray(count)
    for at in range(0, count, MAX_READ_BYTES):
        size = min(MAX_READ_BYTES, count - at)
        raw = reader.read_bytes(name, offset + at, size)
        if not isinstance(raw, bytes) or len(raw) != size:
            raise SafeTensorError("Runtime band returned an incorrect byte count")
        data[at:at + size] = raw
    return data


def _output_band(values, height):
    """Keep finite native FP32 arrays as FP32; validate generic adapters fully."""
    if not hasattr(values, "__len__") or len(values) != height:
        raise ValueError("Native batched projection returned an invalid output shape")
    if type(values) is array and values.typecode == "f" and values.itemsize == 4:
        if not all(map(math.isfinite, values)):
            raise ValueError("Native projection returned a non-finite band output")
        return values
    return array("f", (_finite_float32(value, "band output") for value in values))


class _LoadedBand:
    __slots__ = ("item", "payload")

    def __init__(self, item, payload):
        self.item, self.payload = item, payload

    def release(self):
        self.payload = None


def _sequential_bands(items, load):
    for item in items:
        yield _LoadedBand(item, load(item))


def project_many(reader, name, info, vectors, cpu, *, nvfp4=False, ledger=None, read_ahead=None):
    """Return FP32 arrays; retain only a row band, prepared inputs and outputs.

    Input batches share disk reads without changing each vector's arithmetic.
    Native methods own row scheduling. Optional read-ahead grants one producer
    exclusive reader access until it has joined at the projection boundary.
    """
    if not isinstance(vectors, (list, tuple)) or not 1 <= len(vectors) <= MAX_LINEAR_BATCH:
        raise ValueError("A projection batch must contain 1..16 vectors")
    rows, encoded_cols = info.shape
    cols = encoded_cols * 2 if nvfp4 else encoded_cols
    if not 1 <= cols <= MAX_BAND_COLUMNS:
        raise ValueError("Batched projection columns exceed the native band bound")
    for vector in vectors:
        if isinstance(vector, (str, bytes, bytearray)) or not hasattr(vector, "__len__") or len(vector) != cols:
            raise ValueError("Batched projection vector shape differs from the tensor")
    scratch = NVFP4_ROW_BAND_SCRATCH_BYTES if nvfp4 else DENSE_ROW_BAND_SCRATCH_BYTES
    declared = scratch + 4 * (rows + cols) * len(vectors) + 2 * MAX_READ_BYTES
    with ledger.reserve(declared, label="cpu_projection_batch") if ledger else nullcontext():
        many_method = getattr(cpu, "matvec_nvfp4_row_band_many" if nvfp4 else "matvec_dense_row_band_many", None)
        many_prepare = getattr(cpu, "prepare_nvfp4_batch" if nvfp4 else "prepare_dense_batch", None)
        use_many = len(vectors) > 1 and callable(many_method) and callable(many_prepare) and getattr(
            cpu, "supports_nvfp4_row_band_many" if nvfp4 else "supports_dense_row_band_many", False)
        prepare = cpu.prepare_nvfp4_vector if nvfp4 else cpu.prepare_dense_vector
        prepared = many_prepare(vectors) if use_many else [prepare(vector) for vector in vectors]
        single_method = None
        if not use_many:
            single_method = getattr(cpu, "matvec_nvfp4_row_band_array" if nvfp4 else "matvec_dense_row_band_array", None)
            if not callable(single_method):
                single_method = cpu.matvec_nvfp4_row_band if nvfp4 else cpu.matvec_dense_row_band
        output = [array("f", [0.0]) * rows for _ in vectors]
        matrix = descriptor = None
        if nvfp4:
            descriptor = reader.projection(name)
            matrix = _matrix(reader, descriptor)
        def load_band(row):
            height = min(BAND_ROWS, rows - row)
            weights = read_span(reader, name, row * encoded_cols * info.itemsize,
                                height * encoded_cols * info.itemsize)
            scales = global_scale = None
            if nvfp4:
                scales = read_span(reader, descriptor.scale.name, row * (cols // 16), height * (cols // 16))
                global_scale = matrix._scalar(descriptor.global_scale.name, "global weight scale")
                matrix._scalar(descriptor.input_scale.name, "input scale")
            return weights, scales, global_scale

        def band_bytes(row):
            height = min(BAND_ROWS, rows - row)
            return height * encoded_cols * info.itemsize + (height * (cols // 16) + 8 if nvfp4 else 0)

        use_read_ahead = read_ahead is not None and ledger is not None and rows > BAND_ROWS
        positions = range(0, rows, BAND_ROWS)
        context = read_ahead.bands(positions, load_band, band_bytes, ledger=ledger) if use_read_ahead else nullcontext()
        reads = calls = 0
        io_stats = {"enabled": False}
        try:
            with context as pipeline:
                for packet in pipeline if use_read_ahead else _sequential_bands(positions, load_band):
                    weights = scales = partial = values = None
                    try:
                        row = packet.item
                        height = min(BAND_ROWS, rows - row)
                        weights, scales, global_scale = packet.payload
                        reads += 2 if nvfp4 else 1
                        if use_many:
                            partial = (many_method(weights, height, cols, scales, prepared, global_scale) if nvfp4
                                       else many_method(weights, height, cols, prepared, info.dtype))
                            if not isinstance(partial, (list, tuple)) or len(partial) != len(vectors):
                                raise ValueError("Native projection returned an invalid vector batch")
                            for index, values in enumerate(partial):
                                output[index][row:row + height] = _output_band(values, height)
                            calls += 1
                        else:
                            for index, vector in enumerate(prepared):
                                partial = (single_method(weights, height, cols, scales, vector, global_scale)
                                           if nvfp4 else single_method(weights, height, cols, vector, info.dtype))
                                output[index][row:row + height] = _output_band(partial, height)
                                calls += 1
                    except BaseException as error:
                        weights = scales = partial = values = None
                        clear_error_frames(error)
                        raise
                    finally:
                        weights = scales = partial = values = None
                        packet.release()
                        packet = None
            if use_read_ahead:
                io_stats = pipeline.stats()
        except BaseException as error:
            output = prepared = vector = matrix = descriptor = None
            clear_error_frames(error)
            raise
        return output, {"native_band_calls": calls, "band_reads": reads, "simd_input_batch": use_many,
                        "logical_tiles": math.ceil(rows / BAND_ROWS) * math.ceil(cols / BAND_ROWS) * len(vectors),
                        "vectors": len(vectors), "declared_buffer_bytes": declared, "read_ahead": io_stats}
