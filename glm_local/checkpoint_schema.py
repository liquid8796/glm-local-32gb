"""Checkpoint metadata inventory and explicitly limited GLM/FP8 schema review.

Names/shape formulae are candidate checks, not a complete checkpoint-to-graph
mapper. Unrecognized tensors remain unreviewed; no synthetic name substitution,
implicit reshape, dtype conversion, missing scale invention, or payload reads.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
import math
import re

from .checkpoint_http import MAX_SHARDS, MAX_TENSORS, MetadataError, shard_filename
from .safetensor_reader import MAX_FILE_BYTES
from .sharded_safetensors import MAX_INDEX_BYTES, MAX_INDEX_TENSORS


class Findings:
    def __init__(self):
        self.counts = Counter()
        self.examples = defaultdict(list)

    def add(self, code, detail):
        self.counts[code] += 1
        if len(self.examples[code]) < 8:
            self.examples[code].append(detail)

    def report(self):
        return {"count": sum(self.counts.values()), "by_code": dict(self.counts),
                "examples": dict(self.examples), "max_examples_per_code": 8}


_NV_EXPERT = re.compile(r"model\.layers\.(0|[1-9][0-9]*)\.mlp\.experts\.(0|[1-9][0-9]*)\.(gate|up|down)_proj\.weight")
NVFP4_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


def quantization_format(config):
    """Identify an advertised format; callers must separately validate its policy."""
    quant = config.get("quantization_config") if isinstance(config, dict) else None
    if not isinstance(quant, dict):
        return None
    if quant.get("quant_method") == "fp8" and quant.get("fmt") == "e4m3":
        return "fp8"
    if quant.get("quant_method") == "modelopt" and quant.get("quant_algo") == "NVFP4":
        return "nvfp4"
    return None


def nvfp4_weight_name(name):
    """Resolve only anchored expert weights and their three distinct ancillaries."""
    if not isinstance(name, str):
        return None
    candidate = name
    for suffix in NVFP4_SUFFIXES:
        if name.endswith(suffix):
            candidate = name[:-len(suffix)] + ".weight"
            break
    return candidate if _NV_EXPERT.fullmatch(candidate) else None


def nvfp4_ancillary_names(weight_name):
    if not isinstance(weight_name, str) or not _NV_EXPERT.fullmatch(weight_name):
        raise ValueError("NVFP4 ancillaries require a canonical unpacked expert weight")
    return tuple(weight_name[:-len(".weight")] + suffix for suffix in NVFP4_SUFFIXES)


def nvfp4_quantized_weight(name, config):
    """Candidate routed-backbone policy; config ignore rules must pass validation.

    It intentionally does not quantize MTP, shared experts or dense layers.
    The accepted ignore policy is exact, so it need not be rescanned per tensor.
    """
    match = _NV_EXPERT.fullmatch(name) if isinstance(name, str) else None
    if match is None or quantization_format(config) != "nvfp4":
        return False
    layer, expert = int(match[1]), int(match[2])
    layers, experts = config.get("num_hidden_layers"), config.get("n_routed_experts")
    if type(layers) is not int or type(experts) is not int or layer >= layers or expert >= experts:
        return False
    schedule = config.get("mlp_layer_types")
    if isinstance(schedule, list) and len(schedule) == layers:
        return schedule[layer] in ("sparse", "moe")
    dense = config.get("first_k_dense_replace")
    return type(dense) is int and 0 <= dense <= layer


def validate_nvfp4_config(config, findings):
    """One ModelOpt 0.45.0 Linear-group profile with explicit ignore coverage.

    Unknown producer/group/activation/cache policies and altered ignore globs
    require review; format recognition alone never approves this profile.
    """
    quant = config.get("quantization_config") if isinstance(config, dict) else None
    if quantization_format(config) != "nvfp4":
        findings.add("UNSUPPORTED_NVFP4_PROFILE", "Requires ModelOpt NVFP4")
        return False
    before = sum(findings.counts.values())
    expected_group = {"group_0": {"input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                                 "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                                 "targets": ["Linear"]}}
    for field, expected in (("config_groups", expected_group),
                            ("producer", {"name": "modelopt", "version": "0.45.0"}),
                            ("kv_cache_scheme", {"dynamic": False, "num_bits": 8, "type": "float"})):
        if json.dumps(quant.get(field), sort_keys=True) != json.dumps(expected, sort_keys=True):
            findings.add("UNSUPPORTED_NVFP4_CONFIG", {"field": field, "required": expected})
    if set(quant) != {"config_groups", "ignore", "quant_algo", "kv_cache_scheme", "producer", "quant_method"}:
        findings.add("UNSUPPORTED_NVFP4_CONFIG_FIELDS", "NVFP4 quantization fields differ from the reviewed producer profile")
    layers, schedule, mtp = config.get("num_hidden_layers"), config.get("mlp_layer_types"), config.get("num_nextn_predict_layers", 0)
    if (type(layers) is not int or not 1 <= layers <= 256 or not isinstance(schedule, list)
            or len(schedule) != layers or any(value not in ("dense", "sparse", "moe") for value in schedule)
            or type(mtp) is not int or mtp not in (0, 1)):
        findings.add("NVFP4_SCHEDULE_REQUIRED", "NVFP4 requires explicit bounded dense/sparse and MTP schedules")
    else:
        expected_ignore = {"lm_head", "model.embed_tokens"}
        for layer, kind in enumerate(schedule):
            if kind == "dense":
                expected_ignore.add(f"model.layers.{layer}*" if layer == 0 else f"model.layers.{layer}.*")
            else:
                expected_ignore.update((f"model.layers.{layer}.self_attn*", f"model.layers.{layer}.mlp.shared_experts*"))
        if mtp:
            expected_ignore.add(f"model.layers.{layers}*")
        ignore = quant.get("ignore")
        if (not isinstance(ignore, list) or len(ignore) > 1024 or any(not isinstance(value, str) for value in ignore)
                or len(ignore) != len(expected_ignore) or set(ignore) != expected_ignore):
            findings.add("UNSUPPORTED_NVFP4_IGNORE_POLICY", "Ignore globs must exactly protect embeddings/head, dense, attention/shared and MTP modules")
    for key in ("hidden_size", "moe_intermediate_size"):
        value = config.get(key)
        if type(value) is not int or value <= 0 or value % 16:
            findings.add("NVFP4_LOGICAL_COLUMNS_ALIGNMENT", {"field": key, "required_multiple": 16})
    return sum(findings.counts.values()) == before


def stored_shape(name, config):
    """Expected on-disk dimensions, distinct from decoder logical known_shape."""
    base = nvfp4_weight_name(name)
    if base is not None and nvfp4_quantized_weight(base, config):
        logical = known_shape(base, config)
        if logical is None or len(logical) != 2 or logical[1] % 16:
            return None
        rows, cols = logical
        if name == base:
            return rows, cols // 2
        if name.endswith(".weight_scale"):
            return rows, cols // 16
        return ()  # Scalar weight_scale_2 or independent input_scale.
    return known_shape(name, config)


def nvfp4_storage_dtype(name, config):
    base = nvfp4_weight_name(name)
    if base is not None and nvfp4_quantized_weight(base, config):
        return "U8" if name == base else "F8_E4M3" if name.endswith(".weight_scale") else "F32"
    if known_shape(name, config) is not None:
        return "F32" if name.endswith(".mlp.gate.e_score_correction_bias") else "BF16"
    return None


def validate_manifest(model_id, revision, model):
    if not isinstance(model, dict) or model.get("id") != model_id or model.get("sha") != revision:
        raise MetadataError("Model manifest identity/revision differs from settings")
    entries = model.get("siblings")
    if not isinstance(entries, list):
        raise MetadataError("Model manifest siblings must be an array")
    sizes = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            raise MetadataError("Invalid model manifest sibling")
        name = item["rfilename"]
        if not name.endswith(".safetensors"):
            continue
        shard_filename(name)
        size = item.get("size")
        if type(size) is not int or not 8 <= size <= MAX_FILE_BYTES:
            raise MetadataError("Shard manifest requires a valid bounded file size")
        if name in sizes:
            raise MetadataError("Duplicate shard in model manifest")
        sizes[name] = size
    if (not 1 <= len(sizes) <= MAX_SHARDS
            or len({name.casefold() for name in sizes}) != len(sizes)):
        raise MetadataError("Manifest shard count/case aliases violate local policy")
    return sizes


def validate_index(index, sizes):
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise MetadataError("Index requires metadata and weight_map")
    mapping, metadata = index["weight_map"], index["metadata"]
    if not isinstance(mapping, dict) or not 1 <= len(mapping) <= MAX_TENSORS:
        raise MetadataError("Index tensor count exceeds metadata-audit policy")
    if (not isinstance(metadata, dict) or type(metadata.get("total_size")) is not int
            or not 0 <= metadata["total_size"] <= 2**63 - 1):
        raise MetadataError("Index metadata.total_size must be a nonnegative integer")
    by_shard = defaultdict(set)
    for name, filename in mapping.items():
        try:
            valid = (isinstance(name, str) and 1 <= len(name.encode("utf-8")) <= 512
                     and not any(ord(c) < 32 for c in name))
        except UnicodeError:
            valid = False
        if not valid:
            raise MetadataError("Invalid/oversized index tensor name")
        shard_filename(filename)
        if filename not in sizes:
            raise MetadataError("Index references a shard absent from the pinned manifest")
        by_shard[filename].add(name)
    if set(by_shard) != set(sizes):
        raise MetadataError("Manifest/index shard sets differ; refusing to silently omit shards")
    return dict(by_shard)


def compare_snapshot(model_id, revision, sizes, config, expected):
    """Compare only fields recorded in the pre-existing project snapshot."""
    if not isinstance(config, dict) or not isinstance(config.get("model_type"), str):
        raise MetadataError("Checkpoint config must declare model_type")
    if not isinstance(config.get("quantization_config"), dict):
        raise MetadataError("Checkpoint config must declare quantization_config")
    if (not isinstance(expected, dict) or expected.get("model_id") != model_id
            or expected.get("revision") != revision):
        raise MetadataError("Project metadata snapshot identity/revision differs from settings")
    expected_sizes = {item["name"]: item["bytes"] for item in expected["weights"]}
    mismatches = []
    if sizes != expected_sizes:
        mismatches.append({"field": "weights", "detail": "Shard names/sizes differ from project snapshot"})
    fields = 1
    architecture = expected["architecture"]
    for key, value in architecture.items():
        items = value.items() if key == "quantization" else [(key, value)]
        for field, wanted in items:
            actual = config["quantization_config"].get(field) if key == "quantization" else config.get(field)
            fields += 1
            # Avoid treating True and 1, or integer and float shapes, as identical.
            if json.dumps(actual, sort_keys=True) != json.dumps(wanted, sort_keys=True):
                mismatches.append({"field": f"quantization_config.{field}" if key == "quantization" else field,
                                   "expected": wanted, "actual": actual})
    return {"matched": not mismatches, "checked_fields": fields, "mismatches": mismatches,
            "scope": "Only the fields and shard sizes in docs/model-metadata.json"}


def known_shape(name, config):
    """Return a formula only for recognized names with explicitly supplied dimensions.

    This does not establish that every required name/layer/expert exists, nor that
    MTP/packed experts, attention sharing or full runtime mapping are supported.
    One explicitly declared MTP layer has candidate shape formulas; acceptance of
    that storage profile and its complete inventory belongs to architecture.mapper.
    """
    def dim(key):
        value = config.get(key)
        if type(value) is not int or not 1 <= value <= 2**31 - 1:
            raise KeyError(key)
        return value

    try:
        hidden = dim("hidden_size")
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return (dim("vocab_size"), hidden)
        if name == "model.norm.weight":
            return (hidden,)
        match = re.fullmatch(r"model\.layers\.([0-9]+)\.(.+)", name)
        if not match:
            return None
        layer, backbone = int(match[1]), dim("num_hidden_layers")
        is_mtp = layer == backbone and type(config.get("num_nextn_predict_layers")) is int and config["num_nextn_predict_layers"] == 1
        if layer >= backbone and not is_mtp:
            return None  # Never silently remap an undeclared prediction layer.
        suffix = match[2]
        if is_mtp and suffix == "eh_proj.weight":
            return (hidden, 2 * hidden)
        if is_mtp and suffix in ("enorm.weight", "hnorm.weight", "shared_head.norm.weight"):
            return (hidden,)
        if suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            return (hidden,)
        if suffix in ("self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight"):
            return ((dim("q_lora_rank"), hidden) if suffix.endswith("proj.weight")
                    else (dim("q_lora_rank"),))
        if suffix == "self_attn.kv_a_proj_with_mqa.weight":
            return (dim("kv_lora_rank") + dim("qk_rope_head_dim"), hidden)
        if suffix == "self_attn.kv_a_layernorm.weight":
            return (dim("kv_lora_rank"),)
        if suffix == "self_attn.q_b_proj.weight":
            return (dim("num_attention_heads") * (dim("qk_nope_head_dim") + dim("qk_rope_head_dim")),
                    dim("q_lora_rank"))
        if suffix == "self_attn.kv_b_proj.weight":
            return (dim("num_attention_heads") * (dim("qk_nope_head_dim") + dim("v_head_dim")),
                    dim("kv_lora_rank"))
        if suffix == "self_attn.o_proj.weight":
            return (hidden, dim("num_attention_heads") * dim("v_head_dim"))
        if suffix == "mlp.gate.weight":
            return (dim("n_routed_experts"), hidden)
        if suffix == "mlp.gate.e_score_correction_bias":
            return (dim("n_routed_experts"),)
        projection = re.fullmatch(r"mlp\.(?:(shared_experts|experts\.([0-9]+))\.)?(gate|up|down)_proj\.weight", suffix)
        if projection:
            group, expert, direction = projection.groups()
            if expert is not None and int(expert) >= dim("n_routed_experts"):
                return None
            middle = dim("intermediate_size") if group is None else dim("moe_intermediate_size")
            if group == "shared_experts":
                middle *= dim("n_shared_experts")
            return (hidden, middle) if direction == "down" else (middle, hidden)
        if suffix == "mlp.experts.gate_up_proj":
            return (dim("n_routed_experts"), 2 * dim("moe_intermediate_size"), hidden)
        if suffix == "mlp.experts.down_proj":
            return (dim("n_routed_experts"), hidden, dim("moe_intermediate_size"))
        if suffix == "self_attn.indexer.wq_b.weight":
            return (dim("index_n_heads") * dim("index_head_dim"), dim("q_lora_rank"))
        if suffix == "self_attn.indexer.wk.weight":
            return (dim("index_head_dim"), hidden)
        if suffix == "self_attn.indexer.weights_proj.weight":
            return (dim("index_n_heads"), hidden)
        if suffix in ("self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias"):
            return (dim("index_head_dim"),)
    except KeyError:
        pass  # Missing config dimension is unreviewed, never filled from miniature defaults.
    return None


def _review_nvfp4_tensors(tensors, mapping, config, *, complete, catalogue_path):
    findings = Findings()
    supported = validate_nvfp4_config(config, findings)
    dtypes = defaultdict(lambda: {"tensors": 0, "elements": 0, "payload_bytes": 0})
    shapes, groups = Counter(), Counter()
    invalid, unreviewed = set(), []
    with catalogue_path.open("w", encoding="utf-8", newline="\n") as catalogue:
        for name, tensor in sorted(tensors.items()):
            record = {"name": name, "shard": mapping[name], "dtype": tensor.dtype, "shape": list(tensor.shape),
                      "data_offsets": list(tensor.data_offsets), "nbytes": tensor.nbytes}
            dtype = dtypes[tensor.dtype]
            dtype["tensors"] += 1
            dtype["elements"] += math.prod(tensor.shape)
            dtype["payload_bytes"] += tensor.nbytes
            base = nvfp4_weight_name(name)
            quantized = base is not None and nvfp4_quantized_weight(base, config)
            if quantized:
                record["role"] = ("nvfp4_packed_weight" if name == base else "nvfp4_block_scale" if name.endswith(".weight_scale")
                                  else "nvfp4_global_weight_scale" if name.endswith(".weight_scale_2") else "nvfp4_activation_scale")
                groups[record["role"]] += 1
                record["logical_shape"] = list(known_shape(base, config) or ())
                if name != base and base not in mapping:
                    findings.add("ORPHAN_NVFP4_ANCILLARY", name)
                    invalid.add(name)
                if name == base:
                    record["weight_scale"], record["weight_scale_2"], record["input_scale"] = nvfp4_ancillary_names(name)
                    for ancillary in nvfp4_ancillary_names(name):
                        if ancillary not in mapping:
                            findings.add("MISSING_NVFP4_ANCILLARY", {"weight": name, "missing": ancillary})
                            invalid.add(name)
                        elif ancillary not in tensors:
                            groups["deferred_ancillary_headers"] += 1
            expected = stored_shape(name, config)
            expected_dtype = nvfp4_storage_dtype(name, config)
            if expected is None:
                record["name_shape_check"] = "not_reviewed"
                shapes["not_reviewed"] += 1
                if len(unreviewed) < 32:
                    unreviewed.append(name)
                findings.add("UNREVIEWED_NVFP4_TENSOR", name)
                invalid.add(name)
            else:
                matches = tuple(tensor.shape) == expected
                record["name_shape_check"] = "match" if matches else "mismatch"
                record["expected_shape_from_config"] = list(expected)
                shapes["matched" if matches else "mismatched"] += 1
                if not matches:
                    findings.add("CONFIG_TENSOR_SHAPE_MISMATCH", {"name": name, "expected": list(expected), "actual": list(tensor.shape)})
                    invalid.add(name)
            if expected_dtype is not None and tensor.dtype != expected_dtype:
                findings.add("NVFP4_STORAGE_DTYPE_MISMATCH", {"name": name, "expected": expected_dtype, "actual": tensor.dtype})
                invalid.add(name)
            catalogue.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    for name in tensors:
        if nvfp4_quantized_weight(name, config):
            names = (name, *nvfp4_ancillary_names(name))
            if all(item in tensors and item not in invalid for item in names):
                groups["valid_metadata_quadruples"] += 1
    count = groups["nvfp4_packed_weight"]
    if complete and not count:
        findings.add("NO_NVFP4_WEIGHTS_OBSERVED", "NVFP4 profile requires routed packed weights")
    verified = (complete and supported and count > 0 and groups["valid_metadata_quadruples"] == count
                and sum(findings.counts.values()) == 0)
    return {"quant_format": "nvfp4", "dtype_inventory": dict(dtypes), "nvfp4_groups": dict(groups),
            "nvfp4_adapter_metadata_verified": verified, "fp8_pairs": {}, "fp8_adapter_metadata_verified": False,
            "findings": findings.report(), "known_name_shape_checks": dict(shapes), "unreviewed_tensor_examples": unreviewed,
            "architecture_mapping_verified": False, "payload_values_verified": False,
            "real_checkpoint_compatible": False, "inference_verified": False,
            "largest_observed_matrices": [{"name": n, "shape": list(t.shape), "dtype": t.dtype, "nbytes": t.nbytes, "shard": mapping[n]}
                for n, t in heapq.nlargest(12, ((n, t) for n, t in tensors.items() if len(t.shape) == 2), key=lambda item: item[1].nbytes)],
            "scope": "Packed E2M1 and FP8/F32 weight-scale metadata only; independent input_scale is activation calibration, not a weight multiplier"}


def review_tensors(tensors, mapping, config, *, complete, catalogue_path):
    if quantization_format(config) == "nvfp4":
        return _review_nvfp4_tensors(tensors, mapping, config, complete=complete, catalogue_path=catalogue_path)
    findings = Findings()
    quant = config["quantization_config"]
    block = quant.get("weight_block_size")
    block_supported = (isinstance(block, list) and len(block) == 2
                       and all(type(n) is int and n == 128 for n in block))
    profile_supported = (quant.get("quant_method") == "fp8" and quant.get("fmt") == "e4m3"
                         and block_supported and quant.get("scale_fmt", "float") == "float")
    if not profile_supported:
        findings.add("UNSUPPORTED_QUANTIZATION_PROFILE", quant)
    dtypes = defaultdict(lambda: {"tensors": 0, "elements": 0, "payload_bytes": 0})
    pair_counts, shape_counts = Counter(), Counter()
    unreviewed_names = []
    with catalogue_path.open("w", encoding="utf-8", newline="\n") as catalogue:
        for name, tensor in sorted(tensors.items()):
            record = {"name": name, "shard": mapping[name], "dtype": tensor.dtype,
                      "shape": list(tensor.shape), "data_offsets": list(tensor.data_offsets),
                      "nbytes": tensor.nbytes}
            stats = dtypes[tensor.dtype]
            stats["tensors"] += 1
            stats["elements"] += math.prod(tensor.shape)
            stats["payload_bytes"] += tensor.nbytes
            if name.endswith(".weight_scale_inv"):
                record["role"] = "candidate_block_scale"
                base = name[:-len("_scale_inv")]
                if base not in mapping:
                    findings.add("ORPHAN_SCALE", name)
                elif base in tensors and tensors[base].dtype != "F8_E4M3":
                    findings.add("SCALE_FOR_NON_FP8_WEIGHT", name)
            else:
                expected = known_shape(name, config)
                if expected is None:
                    record["name_shape_check"] = "not_reviewed"
                    shape_counts["not_reviewed"] += 1
                    if len(unreviewed_names) < 32:
                        unreviewed_names.append(name)
                else:
                    matches = tuple(tensor.shape) == expected
                    record["name_shape_check"] = "match" if matches else "mismatch"
                    record["expected_shape_from_config"] = list(expected)
                    shape_counts["matched" if matches else "mismatched"] += 1
                    if not matches:
                        findings.add("CONFIG_TENSOR_SHAPE_MISMATCH", {
                            "name": name, "expected": list(expected), "actual": list(tensor.shape)})
            if tensor.dtype == "F8_E4M3":
                pair_counts["fp8_weights"] += 1
                # This is an explicit candidate storage profile, not inferred compatibility.
                scale_name = name + "_scale_inv" if name.endswith(".weight") else None
                record["candidate_scale"] = scale_name
                supported = profile_supported
                if scale_name is None:
                    findings.add("UNREVIEWED_FP8_NAME", name)
                    supported = False
                if len(tensor.shape) != 2 or any(n <= 0 for n in tensor.shape):
                    findings.add("FP8_REQUIRES_NONEMPTY_2D", name)
                    supported = False
                if scale_name is not None:
                    if scale_name not in mapping:
                        findings.add("MISSING_SCALE", name)
                        supported = False
                    elif scale_name not in tensors:
                        pair_counts["deferred_scale_headers"] += 1
                        record["fp8_profile_check"] = "deferred_uninspected_scale"
                        supported = False
                    else:
                        scale = tensors[scale_name]
                        record["scale_shard"] = mapping[scale_name]
                        record["scale_dtype"] = scale.dtype
                        record["scale_shape"] = list(scale.shape)
                        pair_counts["inspected_pairs"] += 1
                        pair_counts["cross_shard_pairs"] += mapping[name] != mapping[scale_name]
                        if scale.dtype != "F32":
                            findings.add("CURRENT_ADAPTER_REQUIRES_F32_SCALE", {
                                "weight": name, "scale": scale_name, "actual_dtype": scale.dtype})
                            supported = False
                        if len(tensor.shape) == 2 and block_supported:
                            grid = tuple((n + 127) // 128 for n in tensor.shape)
                            if scale.shape != grid:
                                findings.add("SCALE_GRID_MISMATCH", {
                                    "weight": name, "expected": list(grid), "actual": list(scale.shape)})
                                supported = False
                if supported:
                    pair_counts["current_adapter_metadata_matches"] += 1
                record.setdefault("fp8_profile_check", "match_metadata_only" if supported else "needs_review")
            catalogue.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    fp8_count = pair_counts["fp8_weights"]
    fp8_verified = (complete and profile_supported and fp8_count > 0
                    and pair_counts["current_adapter_metadata_matches"] == fp8_count
                    and not any(findings.counts[k] for k in ("ORPHAN_SCALE", "SCALE_FOR_NON_FP8_WEIGHT")))
    if complete and not fp8_count:
        findings.add("NO_FP8_WEIGHTS_OBSERVED", "Checkpoint declares FP8, but no F8_E4M3 tensor was found")
    return {"dtype_inventory": dict(dtypes), "fp8_pairs": dict(pair_counts),
            "fp8_adapter_metadata_verified": fp8_verified, "findings": findings.report(),
            "known_name_shape_checks": dict(shape_counts),
            "unreviewed_tensor_examples": unreviewed_names,
            "architecture_mapping_verified": False,
            "largest_observed_matrices": [
                {"name": n, "shape": list(t.shape), "dtype": t.dtype,
                 "nbytes": t.nbytes, "shard": mapping[n]}
                for n, t in heapq.nlargest(12, ((n, t) for n, t in tensors.items() if len(t.shape) == 2),
                                          key=lambda pair: pair[1].nbytes)],
            "scope": "Header dtype/shape/grid only; no scale values, graph completeness or dequantization verified"}


def runtime_index_policy(index_bytes, tensor_count):
    return {"metadata_audit_max_index_tensors": MAX_TENSORS,
            "current_reader_max_index_tensors": MAX_INDEX_TENSORS,
            "current_reader_max_index_bytes": MAX_INDEX_BYTES,
            "observed_index_bytes": index_bytes, "observed_index_tensors": tensor_count,
            "fits_current_reader_index_policy": index_bytes <= MAX_INDEX_BYTES and tensor_count <= MAX_INDEX_TENSORS,
            "reader_limits_changed": False}
