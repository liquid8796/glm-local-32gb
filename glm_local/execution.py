"""Explicit, bounded FP8 projections from freshly checked checkpoint evidence.

This module executes one named matrix, not a decoder or a model graph. Evidence
digests bind local metadata; they are not signed remote payload checksums.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import struct

from .architecture import report as architecture
from .checkpoint_http import MetadataError, strict_json, validate_target
from .checkpoint_schema import validate_index, validate_manifest
from .checkpoint_snapshot import OfflineMetadataSource
from .cpu_probe import _finite_float32
from .cuda_probe import MAX_OPERATIONS
from .fp8_blocks import BLOCK, FP8BlockMatrix
from .safetensor_reader import DTYPE_ITEMSIZE, TensorInfo, parse_header_bytes
from .model_profiles import reports_directory

MAX_VECTOR_ELEMENTS = 65536
MAX_PROJECTION_TILES = 65536
MAX_REFERENCE_ELEMENTS = 16 * 1024**2
FULL_MODEL_FLAGS = {"real_checkpoint_compatible": False, "inference_verified": False,
                    "full_model_loaded": False, "full_model_limits_verified": False}


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    dtype: str
    shape: tuple[int, int]
    nbytes: int
    data_offsets: tuple[int, int]
    shard: str

    def info(self):
        return TensorInfo(self.dtype, self.shape, self.data_offsets, self.nbytes,
                          DTYPE_ITEMSIZE[self.dtype])


@dataclass(frozen=True)
class ShardDescriptor:
    name: str
    file_size: int
    header_bytes: int  # Prefix plus header; also the absolute payload start.
    header_sha256: str


@dataclass(frozen=True)
class ProjectionDescriptor:
    weight: TensorDescriptor
    scale: TensorDescriptor
    shards: tuple[ShardDescriptor, ...]
    model_id: str
    revision: str
    source_path: str
    source_sha256: str
    catalogue_path: str
    catalogue_sha256: str
    snapshot_path: str
    snapshot_sha256: str
    architecture_metadata_verified: bool
    source_status: str

    @property
    def tensors(self):
        return self.weight, self.scale

    @property
    def logical_shape(self):
        return self.weight.shape

    @property
    def quant_format(self):
        return "fp8"

    def to_dict(self):
        return {**asdict(self), "format_version": 1, "selected_projection_eligible": True,
                "scale_semantics": "decoded_e4m3fn_times_stored_scale",
                "block_shape": [BLOCK, BLOCK], "payload_values_verified": False,
                "provenance_scope": "Local source/catalogue/snapshot and selected headers; no payload checksum attestation",
                **FULL_MODEL_FLAGS}


def _tensor(record):
    return TensorDescriptor(record["name"], record["dtype"], tuple(record["shape"]),
                            record["nbytes"], tuple(record["data_offsets"]), record["shard"])


def _validate_pair(weight, scale):
    class Pair:
        tensors = {weight.name: weight.info(), scale.name: scale.info()}
    matrix = FP8BlockMatrix(Pair(), weight.name, scale.name)
    if max(matrix.rows, matrix.cols) > MAX_VECTOR_ELEMENTS:
        raise ValueError(f"Selected projection exceeds the {MAX_VECTOR_ELEMENTS}-element vector bound")
    if matrix.block_rows * matrix.block_cols > MAX_PROJECTION_TILES:
        raise ValueError("Selected projection exceeds the bounded tile count")
    return matrix


def build_projection_descriptor(root, settings, weight_name, scale_name=None):
    """Recheck producer artifacts and select one supported pair independently of graph PASS.

    Partial catalogues may describe a selected pair if both complete shard headers
    are present. Baseline mismatch, failed reports and missing evidence fail closed.
    No tensor payload is read and no network operation is performed.
    """
    root = Path(root).resolve()
    validate_target(settings["model_id"], settings["revision"])
    if not isinstance(weight_name, str) or not weight_name:
        raise ValueError("An explicit weight tensor name is required")
    source_path = reports_directory(root, settings) / "metadata-latest.json"
    analysis = architecture._analyze_source(root, settings, source_path)
    source, source_evidence, source_identity = architecture._bounded_json(
        source_path, architecture.MAX_REPORT_BYTES, capture=True)
    if source_evidence != analysis["source_report"]:
        raise MetadataError("Metadata source changed while selecting projection")
    if source["baseline_comparison"]["matched"] is not True:
        raise MetadataError("Selected projection requires matched baseline metadata")
    quant = source["config"].get("quantization_config")
    from .checkpoint_schema import quantization_format, validate_nvfp4_config, nvfp4_ancillary_names, Findings
    nvfp4 = quantization_format(source["config"]) == "nvfp4"
    if nvfp4:
        findings = Findings()
        validate_nvfp4_config(source["config"], findings)
        if findings.report()["count"]:
            raise MetadataError("Selected NVFP4 projection requires the supported ModelOpt configuration")
        ancillary = nvfp4_ancillary_names(weight_name)
        if scale_name is not None and scale_name != ancillary[0]:
            raise ValueError("NVFP4 scale name must match the explicit module sibling")
        scale_name = ancillary[0]
    elif (not isinstance(quant, dict) or quant.get("quant_method") != "fp8"
            or quant.get("fmt") != "e4m3" or quant.get("scale_fmt", "float") != "float"
            or quant.get("weight_block_size") != [128, 128]
            or any(type(value) is not int for value in quant["weight_block_size"])):
        raise MetadataError("Selected projection requires fp8/e4m3 float scales and 128x128 layout")
    else:
        scale_name = weight_name + "_scale_inv" if scale_name is None else scale_name
    if not isinstance(scale_name, str) or not scale_name or scale_name == weight_name:
        raise ValueError("An explicit distinct scale tensor name is required")
    catalogue_path = Path(analysis["catalogue_path"])
    records, catalogue = architecture._load_catalogue(
        catalogue_path, source["catalogue"], analysis["source_coverage"])
    by_name = {record["name"]: record for record in records}
    selected_names = (weight_name, *ancillary) if nvfp4 else (weight_name, scale_name)
    if any(name not in by_name for name in selected_names):
        raise MetadataError("Selected weight/scale tensor is absent from the verified catalogue")
    weight, scale = _tensor(by_name[weight_name]), _tensor(by_name[scale_name])
    selected = tuple(_tensor(by_name[name]) for name in selected_names)
    if nvfp4:
        from .nvfp4_blocks import NVFP4BlockMatrix
        from types import SimpleNamespace
        matrix = NVFP4BlockMatrix(SimpleNamespace(tensors={item.name: item.info() for item in selected}), *selected_names)
        if max(matrix.rows, matrix.cols) > MAX_VECTOR_ELEMENTS or matrix.block_rows * matrix.block_cols > MAX_PROJECTION_TILES:
            raise ValueError("Selected NVFP4 projection exceeds bounded vector/tile policy")
    else:
        _validate_pair(weight, scale)
    if source.get("evidence") != "evidence/snapshot.json":
        raise MetadataError("Selected projection requires the captured metadata snapshot")
    snapshot_path = catalogue_path.parent / "evidence/snapshot.json"
    _, snapshot_evidence, snapshot_identity = architecture._bounded_json(
        snapshot_path, 1024**2, capture=True)
    evidence = OfflineMetadataSource(snapshot_path.parent, settings["model_id"], settings["revision"])
    model = strict_json(evidence.json_bytes("model"))
    config = strict_json(evidence.json_bytes("config"))
    index = strict_json(evidence.json_bytes("index"))
    sizes = validate_manifest(settings["model_id"], settings["revision"], model)
    by_shard = validate_index(index, sizes)  # Audit policy, not the legacy small-index reader.
    if (config != source["config"]
            or index["metadata"]["total_size"] != source["declared_tensor_payload_bytes"]
            or len(index["weight_map"]) != source["coverage"]["total_index_tensors"]
            or len(sizes) != source["coverage"]["total_shards"]):
        raise MetadataError("Captured index/config differs from the metadata report")
    checked = set(source["coverage"]["checked_shard_names"])
    if (set(by_name) != {name for shard in checked for name in by_shard.get(shard, ())}
            or any(index["weight_map"].get(record["name"]) != record["shard"] for record in records)):
        raise MetadataError("Catalogue differs from the captured index mapping")
    shards = []
    for name in sorted({item.shard for item in selected}):
        raw = evidence.header_bytes(name, sizes[name])
        header_tensors, _ = parse_header_bytes(raw[8:], sizes[name])
        expected = {record["name"]: _tensor(record).info() for record in records if record["shard"] == name}
        if set(header_tensors) != by_shard[name] or dict(header_tensors) != expected:
            raise MetadataError("Selected captured header differs from catalogue/index")
        shards.append(ShardDescriptor(name, sizes[name], len(raw), hashlib.sha256(raw).hexdigest()))
    if (architecture._fingerprint(source_path) != source_identity
            or architecture._fingerprint(snapshot_path) != snapshot_identity):
        raise MetadataError("Metadata evidence changed during projection selection")
    # Rehashing the catalogue also catches payload-free evidence replacement during snapshot checks.
    _, final_catalogue = architecture._load_catalogue(catalogue_path, source["catalogue"], source["coverage"])
    if final_catalogue != catalogue:
        raise MetadataError("Catalogue changed during projection selection")
    descriptor_type = ProjectionDescriptor
    extras = {}
    if nvfp4:
        from .nvfp4_execution import NVFP4ProjectionDescriptor
        descriptor_type = NVFP4ProjectionDescriptor
        extras = {"global_scale": selected[2], "input_scale": selected[3]}
    return descriptor_type(weight, scale, tuple(shards), settings["model_id"], settings["revision"],
        str(source_path), source_evidence["sha256"], str(catalogue_path), catalogue["sha256"],
        str(snapshot_path), snapshot_evidence["sha256"], analysis["architecture_mapping_verified"], source["status"], **extras)


def _vector(vector, cols):
    if isinstance(vector, (str, bytes, bytearray)) or not hasattr(vector, "__len__") or len(vector) != cols:
        raise ValueError("Input vector must have exactly the selected matrix column count")
    return [_finite_float32(vector[index], f"vector[{index}]") for index in range(cols)]


def _matrix(reader, descriptor):
    _validate_pair(descriptor.weight, descriptor.scale)
    for item in (descriptor.weight, descriptor.scale):
        if reader.tensors.get(item.name) != item.info():
            raise ValueError("Reader tensor metadata differs from selected descriptor")
    return FP8BlockMatrix(reader, descriptor.weight.name, descriptor.scale.name)


def execute_projection(reader, descriptor, vector, *, backend="cpu", cpu=None, gpu=None, gate=None, ledger=None):
    """Execute alternating output row blocks synchronously; never decode a whole matrix.

    Inject NativeCpuBackend/CudaTileBackend-compatible kernels. The caller owns
    backends, reader and telemetry gate lifetime and calls gate.finish(). Returned
    output becomes caller-owned; a graph must account its retained activations.
    """
    matrix = _matrix(reader, descriptor)
    if backend not in ("cpu", "hybrid"):
        raise ValueError("Selected projection backend must be cpu or hybrid")
    if cpu is None or not callable(getattr(cpu, "matvec_tile", None)):
        raise ValueError("An explicit CPU kernel is required")
    gpu_tiles = (matrix.block_rows // 2) * matrix.block_cols if backend == "hybrid" else 0
    if backend == "hybrid":
        if not gpu_tiles or gpu is None or gate is None:
            raise ValueError("Hybrid projection needs both row paths, a GPU kernel and an admission gate")
        if gpu_tiles > getattr(gpu, "max_operations", MAX_OPERATIONS) - getattr(gpu, "operations", 0):
            raise ValueError("Selected projection exceeds the remaining bounded CUDA launch count")
    # Reserve before allocating vectors. Python object and backend/runtime overhead
    # belongs to the planner's separate conservative headroom, not this logical counter.
    logical_bytes = 3 * BLOCK * BLOCK + 4 * (matrix.rows + matrix.cols + 4 * BLOCK)
    reservation = ledger.reserve(logical_bytes, label="selected_projection") if ledger else nullcontext()
    with reservation:
        values = _vector(vector, matrix.cols)
        result = [0.0] * matrix.rows
        counts = {"cpu_tiles": 0, "gpu_tiles": 0, "fp8_bytes": 0, "scale_bytes": 0,
                  "max_packed_tile_bytes": 0, "logical_host_buffer_bytes": logical_bytes,
                  "input_fp32_bytes": matrix.cols * 4, "output_fp32_bytes": matrix.rows * 4,
                  "retained_decoded_weight_bytes": 0,
                  "accounting_scope": "Logical buffers; Python/metadata/runtime overhead requires separate headroom",
                  **FULL_MODEL_FLAGS}
        for block in matrix.iter_blocks():
            use_gpu = backend == "hybrid" and block.row_start // BLOCK % 2 == 1
            device_bytes = len(block.weights) + 4 * (block.cols + block.rows)
            device_reservation = (ledger.reserve(device_bytes, device="cuda", label="projection_tile")
                                  if ledger and use_gpu else nullcontext())
            with device_reservation:
                if use_gpu:
                    gate.before_submit()
                output = (gpu if use_gpu else cpu).matvec_tile(
                    block.weights, block.rows, block.cols,
                    values[block.col_start:block.col_start + block.cols], block.scale)
                if not hasattr(output, "__len__") or len(output) != block.rows:
                    raise ValueError("Projection kernel returned an invalid output length")
                for index in range(block.rows):
                    number = _finite_float32(output[index], "kernel output")
                    target = block.row_start + index
                    result[target] = _finite_float32(result[target] + number, "projection accumulation")
            counts["gpu_tiles" if use_gpu else "cpu_tiles"] += 1
            counts["fp8_bytes"] += len(block.weights)
            counts["scale_bytes"] += 4
            counts["max_packed_tile_bytes"] = max(counts["max_packed_tile_bytes"], len(block.weights))
        return result, counts


def _reference_decode(code):
    """Independent E4M3FN scalar definition, separate from kernels and block adapter."""
    magnitude = code & 127
    if magnitude == 127:
        raise ValueError("Reference encountered E4M3FN NaN")
    exponent, mantissa = magnitude >> 3, magnitude & 7
    decoded = math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
    return -decoded if code & 128 else decoded


def reference_projection(reader, descriptor, vector):
    """Independent row-streamed FP32 dequantization/reduction; <=128 packed bytes per read.

    The reference uses the same documented 128-column partial reduction order as
    tile kernels, with a separate row-wise read/decode loop and no FP8BlockMatrix.
    """
    _matrix(reader, descriptor)
    rows, cols = descriptor.weight.shape
    if rows * cols > MAX_REFERENCE_ELEMENTS:
        raise ValueError("Selected reference exceeds the bounded scalar operation count")
    values = _vector(vector, cols)
    result = [0.0] * rows
    block_cols = (cols + BLOCK - 1) // BLOCK
    for row in range(rows):
        total = 0.0
        for start in range(0, cols, BLOCK):
            size = min(BLOCK, cols - start)
            scale_raw = reader.read_bytes(descriptor.scale.name, (row // BLOCK * block_cols + start // BLOCK) * 4, 4)
            if not isinstance(scale_raw, bytes) or len(scale_raw) != 4:
                raise ValueError("Reference scale read must return exactly four bytes")
            scale = _finite_float32(struct.unpack("<f", scale_raw)[0], "reference scale", positive=True)
            packed = reader.read_bytes(descriptor.weight.name, row * cols + start, size)
            if not isinstance(packed, bytes) or len(packed) != size:
                raise ValueError("Reference weight read returned an invalid byte count")
            partial = 0.0
            for column, code in enumerate(packed):
                decoded = _finite_float32(_reference_decode(code) * scale, "reference dequantization")
                product = _finite_float32(decoded * values[start + column], "reference product")
                partial = _finite_float32(partial + product, "reference partial sum")
            total = _finite_float32(total + partial, "reference sum")
        result[row] = total
    return result


def compare_projection(actual, expected):
    if not hasattr(actual, "__len__") or not hasattr(expected, "__len__") or not 1 <= len(actual) == len(expected) <= MAX_VECTOR_ELEMENTS:
        raise ValueError("Projection outputs must have equal bounded nonempty lengths")
    maximum, passed = 0.0, True
    for index in range(len(actual)):
        left, right = _finite_float32(actual[index], "actual"), _finite_float32(expected[index], "reference")
        error = abs(left - right)
        maximum = max(maximum, error)
        passed = passed and error <= 1e-5 + 2e-5 * abs(right)
    return {"passed": passed, "rows_compared": len(actual), "max_absolute_error": maximum,
            "absolute_tolerance": 1e-5, "relative_tolerance": 2e-5,
            "reference": "independent row-streamed FP32 E4M3FN, multiplicative scales, 128-column partial reductions"}
