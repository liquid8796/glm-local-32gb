"""Experimental full-catalogue, local-only streamed checkpoint weight access.

Opening checks complete metadata and every local shard header. Weight payloads
are read only on demand in <=64-KiB reads and <=128-square tiles. No weight
matrix or expert bank is retained. Local header evidence does not authenticate
the contents of all payloads or establish full-model numerical/resource limits.
"""
from array import array
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import stat
import struct
from types import MappingProxyType

from .architecture import report as architecture
from .catalogue_reader import SelectedCatalogueReader, _BoundHeaderReader
from .checkpoint_http import MetadataError, strict_json
from .checkpoint_schema import validate_index, validate_manifest, quantization_format, nvfp4_ancillary_names, nvfp4_quantized_weight
from .checkpoint_snapshot import OfflineMetadataSource
from .cpu_probe import NativeCpuBackend, _finite_float32
from .execution import ProjectionDescriptor, ShardDescriptor, TensorDescriptor, execute_projection
from .safetensor_reader import MAX_READ_BYTES, SafeTensorError, parse_header_bytes
from .model_profiles import reports_directory

MAX_VECTOR_ELEMENTS = 1_048_576
MAX_LINEAR_TILES = 65_536
BLOCK = 128


def _identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise SafeTensorError("Local checkpoint metadata must be a regular file, not a link")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _check_local_document(path, expected):
    """Compare exact captured JSON bytes with bounded local reads."""
    before = _identity(path)
    if before[2] != len(expected):
        raise SafeTensorError("Local config/index length differs from captured evidence")
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        remaining = len(expected)
        while remaining:
            raw = stream.read(min(MAX_READ_BYTES, remaining))
            if not raw:
                raise SafeTensorError("Local config/index truncated during validation")
            digest.update(raw)
            remaining -= len(raw)
    if _identity(path) != before or digest.digest() != hashlib.sha256(expected).digest():
        raise SafeTensorError("Local config/index differs from captured evidence")
    return before


class FullCatalogueReader(SelectedCatalogueReader):
    """Audit-policy metadata plus the existing bounded payload reader and LRU.

    The legacy small-index reader's constants stay unchanged. This constructor
    requires an eligible complete architecture; selected-projection approval
    alone is insufficient for a decoder.
    """

    def __init__(self, root, settings, model_directory=None, *, max_open_shards=2):
        if type(max_open_shards) is not int or not 1 <= max_open_shards <= 2:
            raise ValueError("Runtime reader permits one or two open shards")
        root = Path(root).resolve()
        source_path = reports_directory(root, settings) / "metadata-latest.json"
        analysis = architecture._analyze_source(root, settings, source_path)
        if (analysis["status"] != "PASS" or analysis["architecture_mapping_verified"] is not True
                or analysis["source_eligibility_verified"] is not True):
            raise MetadataError("Runtime weights require a complete eligible architecture metadata PASS")
        source, proof, source_identity = architecture._bounded_json(
            source_path, architecture.MAX_REPORT_BYTES, capture=True)
        if proof != analysis["source_report"] or source.get("evidence") != "evidence/snapshot.json":
            raise MetadataError("Runtime source changed or lacks captured snapshot evidence")
        catalogue_path = Path(analysis["catalogue_path"])
        records, catalogue = architecture._load_catalogue(catalogue_path, source["catalogue"], source["coverage"])
        snapshot_path = catalogue_path.parent / "evidence/snapshot.json"
        _, snapshot_proof, snapshot_identity = architecture._bounded_json(snapshot_path, 1024**2, capture=True)
        evidence = OfflineMetadataSource(snapshot_path.parent, settings["model_id"], settings["revision"])
        model = strict_json(evidence.json_bytes("model"))
        config_raw, index_raw = evidence.json_bytes("config"), evidence.json_bytes("index")
        config, index = strict_json(config_raw), strict_json(index_raw)
        sizes = validate_manifest(settings["model_id"], settings["revision"], model)
        by_shard = validate_index(index, sizes)
        if (config != source["config"] or index["metadata"]["total_size"] != source["declared_tensor_payload_bytes"]
                or len(index["weight_map"]) != len(records) or set(sizes) != set(source["coverage"]["checked_shard_names"])):
            raise MetadataError("Runtime snapshot config/index differs from complete metadata source")
        self._selected = {r["name"]: TensorDescriptor(r["name"], r["dtype"], tuple(r["shape"]),
            r["nbytes"], tuple(r["data_offsets"]), r["shard"]) for r in records}
        if (set(self._selected) != set(index["weight_map"])
                or any(index["weight_map"][n] != item.shard for n, item in self._selected.items())):
            raise MetadataError("Runtime catalogue/index tensor mapping differs")
        self._by_shard = {name: {} for name in sizes}
        for name, item in self._selected.items():
            self._by_shard[item.shard][name] = item.info()
        self._proofs = {}
        for name, size in sizes.items():
            raw = evidence.header_bytes(name, size)
            tensors, _ = parse_header_bytes(raw[8:], size)
            if set(tensors) != by_shard[name] or dict(tensors) != self._by_shard[name]:
                raise MetadataError("Captured runtime header differs from complete catalogue/index")
            self._proofs[name] = ShardDescriptor(name, size, len(raw), hashlib.sha256(raw).hexdigest())
        if (architecture._fingerprint(source_path) != source_identity
                or architecture._fingerprint(snapshot_path) != snapshot_identity):
            raise MetadataError("Runtime metadata changed during validation")
        _, final_catalogue = architecture._load_catalogue(catalogue_path, source["catalogue"], source["coverage"])
        if final_catalogue != catalogue:
            raise MetadataError("Runtime catalogue changed during validation")
        self._provenance = dict(model_id=settings["model_id"], revision=settings["revision"],
            source_path=str(source_path), source_sha256=proof["sha256"], catalogue_path=str(catalogue_path),
            catalogue_sha256=catalogue["sha256"], snapshot_path=str(snapshot_path),
            snapshot_sha256=snapshot_proof["sha256"], architecture_metadata_verified=True, source_status="PASS")
        self.config = deepcopy(config)
        directory = Path(model_directory if model_directory is not None else settings["model_directory"])
        self.directory = (directory if directory.is_absolute() else root / directory).absolute()
        directory_info = self.directory.lstat()
        if not stat.S_ISDIR(directory_info.st_mode) or getattr(directory_info, "st_file_attributes", 0) & 0x400:
            raise SafeTensorError("Runtime checkpoint directory must be a regular directory")
        self._documents = {self.directory / "config.json": _check_local_document(self.directory / "config.json", config_raw),
                           self.directory / "model.safetensors.index.json": _check_local_document(
                               self.directory / "model.safetensors.index.json", index_raw)}
        self._tensors = MappingProxyType({name: item.info() for name, item in self._selected.items()})
        self._open = OrderedDict()
        self._max_open, self._closed, self._identities = max_open_shards, False, {}
        self._totals = dict(actual_read_bytes=0, actual_read_calls=0, tensor_read_bytes=0,
                            tensor_read_calls=0, max_actual_read_bytes=0)
        self._opens = self._evictions = self._peak_open = 0
        try:
            for name in self._proofs:
                self._reader(name)
        except BaseException:
            self.close()
            raise

    def _reader(self, name):
        if self._closed:
            raise SafeTensorError("Runtime catalogue reader is closed")
        if any(_identity(path) != identity for path, identity in self._documents.items()):
            raise SafeTensorError("Local checkpoint config/index changed since validation")
        identity = self._identity(name)
        if name in self._identities and self._identities[name] != identity:
            raise SafeTensorError("Runtime shard changed since validation")
        if name in self._open:
            reader = self._open.pop(name)
            reader._assert_unchanged()
            self._open[name] = reader
            return reader
        if len(self._open) >= self._max_open:
            _, old = self._open.popitem(last=False)
            self._retire(old)
            self._evictions += 1
        reader = _BoundHeaderReader(self.directory / name, self._proofs[name])
        try:
            if self._identity(name) != identity or dict(reader.tensors) != self._by_shard[name]:
                raise SafeTensorError("Runtime local shard header differs from captured catalogue")
        except BaseException:
            reader.close()
            raise
        self._identities[name] = identity
        self._open[name] = reader
        self._opens += 1
        self._peak_open = max(self._peak_open, len(self._open))
        return reader

    def projection(self, name):
        if self._selected[name].dtype == "U8":
            if not nvfp4_quantized_weight(name, self.config):
                raise SafeTensorError("Packed tensor is not a supported NVFP4 routed projection")
            from .nvfp4_execution import NVFP4ProjectionDescriptor
            names = (name, *nvfp4_ancillary_names(name))
            selected = [self._selected[key] for key in names]
            return NVFP4ProjectionDescriptor(selected[0], selected[1],
                tuple(self._proofs[s] for s in sorted({item.shard for item in selected})),
                **self._provenance, global_scale=selected[2], input_scale=selected[3])
        weight, scale = self._selected[name], self._selected[name + "_scale_inv"]
        return ProjectionDescriptor(weight, scale, tuple(self._proofs[s] for s in sorted({weight.shard, scale.shard})),
                                    **self._provenance)

    def stats(self):
        return {**super().stats(), "complete_catalogue_bound": True, "local_config_index_verified": True,
                "local_all_shard_headers_verified": True, "full_model_loaded": False,
                "real_checkpoint_compatible": False, "full_model_limits_verified": False,
                "inference_verified": False, "payload_values_verified": False,
                "retained_weight_payload_bytes": 0}


def _decode(raw, dtype):
    if dtype == "BF16":
        values = array("f", (struct.unpack("<f", struct.pack("<I", bits[0] << 16))[0]
                              for bits in struct.iter_unpack("<H", raw)))
    elif dtype in ("F16", "F32"):
        values = array("f", (value[0] for value in struct.iter_unpack("<e" if dtype == "F16" else "<f", raw)))
    else:
        raise SafeTensorError("Dense runtime payload requires BF16/F16/F32")
    if any(not math.isfinite(value) for value in values):
        raise SafeTensorError("Runtime payload contains nonfinite dense values")
    return values


class RuntimeWeights:
    """Official tensor names with uncached vectors, embedding rows and matvecs."""

    def __init__(self, root, settings, *, model_directory=None, backend="cpu", cpu=None, gpu=None,
                 gate=None, ledger=None, plan=None, max_open_shards=2):
        if backend not in ("cpu", "hybrid"):
            raise ValueError("Runtime backend must be cpu or hybrid")
        if backend == "hybrid" and gate is None:
            raise ValueError("Hybrid runtime requires an explicit GPU telemetry gate")
        self._reader = FullCatalogueReader(root, settings, model_directory, max_open_shards=max_open_shards)
        self.config, self._settings = deepcopy(self._reader.config), dict(settings)
        self.backend, self._gpu, self._gate, self._ledger, self._plan = backend, gpu, gate, ledger, plan
        self._cpu, self._owns_cpu, self._closed = cpu, cpu is None, False
        self._nv_cpu, self._owns_nv_cpu = (cpu if callable(getattr(cpu, "matvec_nvfp4_tile", None)) else None), False
        self._quant_format = quantization_format(self.config)
        self._counts = dict(linear_calls=0, vector_calls=0, embedding_calls=0, cpu_tiles=0, gpu_tiles=0,
                            max_decoded_dense_tile_bytes=0, scoped_cuda_contexts=0,
                            maximum_context_launch_budget=0)
        try:
            if self._cpu is None:
                self._cpu = NativeCpuBackend()
        except BaseException:
            self._reader.close()
            raise

    def _info(self, name, rank):
        if self._closed:
            raise SafeTensorError("Runtime weight adapter is closed")
        info = self._reader.tensors[name]
        if len(info.shape) != rank or any(n > MAX_VECTOR_ELEMENTS for n in info.shape):
            raise SafeTensorError("Runtime tensor rank/vector dimension exceeds the bounded operation policy")
        return info

    def _reserve(self, count, label):
        return self._ledger.reserve(count, label=label) if self._ledger else nullcontext()

    def _dense_span(self, name, info, start, length):
        result = array("f")
        chunk_elements = MAX_READ_BYTES // info.itemsize
        for offset in range(0, length, chunk_elements):
            count = min(chunk_elements, length - offset)
            raw = self._reader.read_bytes(name, (start + offset) * info.itemsize, count * info.itemsize)
            result.extend(_decode(raw, info.dtype))
        return result

    def vector(self, name):
        info = self._info(name, 1)
        with self._reserve(4 * info.shape[0] + 3 * MAX_READ_BYTES, "runtime_vector"):
            result = self._dense_span(name, info, 0, info.shape[0])
        self._counts["vector_calls"] += 1
        return result

    def embedding(self, token):
        name = "model.embed_tokens.weight"
        info = self._info(name, 2)
        rows, cols = info.shape
        if type(token) is not int or not 0 <= token < rows:
            raise ValueError("Embedding token ID lies outside checkpoint vocabulary")
        with self._reserve(4 * cols + 3 * MAX_READ_BYTES, "runtime_embedding"):
            result = self._dense_span(name, info, token * cols, cols)
        self._counts["embedding_calls"] += 1
        return result

    def linear(self, name, values):
        info = self._info(name, 2)
        rows, cols = info.shape
        nvfp4 = info.dtype == "U8"
        if nvfp4:
            if self._quant_format != "nvfp4" or not nvfp4_quantized_weight(name, self.config):
                raise SafeTensorError("U8 tensor has no supported NVFP4 logical projection")
            cols *= 2
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__") or len(values) != cols:
            raise ValueError("Runtime linear input does not match tensor columns")
        tiles = ((rows + BLOCK - 1) // BLOCK) * ((cols + BLOCK - 1) // BLOCK)
        if tiles > MAX_LINEAR_TILES:
            raise ValueError("Runtime projection exceeds the bounded tile operation budget")
        if self._plan is not None and (rows > self._plan.max_projection_rows or cols > self._plan.max_projection_columns):
            raise ValueError("Runtime projection exceeds residency plan dimensions")
        if info.dtype == "F8_E4M3" or nvfp4:
            descriptor = self._reader.projection(name)
            execute, cpu_kernel = execute_projection, self._cpu
            if nvfp4:
                from .nvfp4_execution import execute_nvfp4_projection
                from .nvfp4_kernels import NativeNVFP4CpuBackend
                if self._nv_cpu is None:
                    self._nv_cpu, self._owns_nv_cpu = NativeNVFP4CpuBackend(), True
                execute, cpu_kernel = execute_nvfp4_projection, self._nv_cpu
            gpu_tiles = ((rows + BLOCK - 1) // BLOCK // 2) * ((cols + BLOCK - 1) // BLOCK)
            selected_backend = "hybrid" if self.backend == "hybrid" and gpu_tiles else "cpu"
            context = nullcontext(self._gpu)
            if selected_backend == "hybrid" and self._gpu is None:
                from .cuda_probe import CudaTileBackend
                gpu_type = CudaTileBackend
                if nvfp4:
                    from .nvfp4_kernels import CudaNVFP4TileBackend
                    gpu_type = CudaNVFP4TileBackend
                context = gpu_type(self._settings["gpu_index"], max_operations=gpu_tiles)
                self._counts["scoped_cuda_contexts"] += 1
                self._counts["maximum_context_launch_budget"] = max(
                    self._counts["maximum_context_launch_budget"], gpu_tiles)
            with context as gpu:
                output, stats = execute(self._reader, descriptor, values, backend=selected_backend,
                    cpu=cpu_kernel, gpu=gpu, gate=self._gate, ledger=self._ledger)
            self._counts["cpu_tiles"] += stats["cpu_tiles"]
            self._counts["gpu_tiles"] += stats["gpu_tiles"]
            result = array("f", output)
        else:
            with self._reserve(4 * (rows + cols) + 3 * MAX_READ_BYTES, "runtime_dense_linear"):
                vector = array("f", (_finite_float32(value, "runtime input") for value in values))
                result = array("f", [0.0]) * rows
                for row in range(0, rows, BLOCK):
                    nr = min(BLOCK, rows - row)
                    for col in range(0, cols, BLOCK):
                        nc = min(BLOCK, cols - col)
                        raw = self._reader.read_matrix_tile(name, row, col, nr, nc)
                        weights = _decode(raw, info.dtype)
                        self._counts["max_decoded_dense_tile_bytes"] = max(
                            self._counts["max_decoded_dense_tile_bytes"], len(weights) * 4)
                        for r in range(nr):
                            subtotal = 0.0
                            for c in range(nc):
                                product = _finite_float32(weights[r * nc + c] * vector[col + c], "dense product")
                                subtotal = _finite_float32(subtotal + product, "dense reduction")
                            result[row + r] = _finite_float32(result[row + r] + subtotal, "dense accumulation")
                        self._counts["cpu_tiles"] += 1
        self._counts["linear_calls"] += 1
        return result

    def stats(self):
        return {**self._reader.stats(), **self._counts, "experimental_runtime": True,
                "weight_quantization_format": self._quant_format, "activation_quantization": "none",
                "native_w4a4_parity_verified": False,
                "configured_backend": self.backend, "dense_projections_device": "cpu",
                "maximum_projection_tiles": MAX_LINEAR_TILES,
                "output_buffer_ownership": "Caller owns returned FP32 arrays; decoder plan accounts retained activations"}

    def close(self):
        if not self._closed:
            self._reader.close()
            if self._owns_cpu and self._cpu is not None:
                self._cpu.close()
            if self._owns_nv_cpu and self._nv_cpu is not None:
                self._nv_cpu.close()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise SafeTensorError("Runtime weight adapter is closed")
        return self

    def __exit__(self, *_):
        self.close()
