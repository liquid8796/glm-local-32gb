"""Bounded metadata-only GLM inventory validation.

A passing report establishes only that supplied headers describe one explicit
unpacked-expert storage profile. No values are read and no graph is built.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from ..checkpoint_schema import Findings, known_shape

MAX_INVENTORY = 262144
MAX_LAYERS = 256
MAX_EXPERTS = 1024
MAX_DIMENSION = 2**31 - 1
MAX_PAYLOAD_BYTES = 2**63 - 1
DTYPE_BYTES = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}
ROLES = (
    "embedding", "output_head", "norm", "attention_projection", "attention_norm",
    "attention_indexer", "dense_ffn", "expert_projection", "shared_expert",
    "moe_router", "router_bias", "fp8_scale", "unknown",
)
_LAYER = re.compile(r"model\.layers\.(0|[1-9][0-9]*)\.(.+)")
_ROOT_ROLES = {"model.embed_tokens.weight": "embedding", "lm_head.weight": "output_head",
               "model.norm.weight": "norm"}
_ATTENTION = (
    "q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight",
    "kv_b_proj.weight", "o_proj.weight", "q_a_layernorm.weight", "kv_a_layernorm.weight",
)
_INDEXER = ("wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias")


def _base_role(name):
    if name in _ROOT_ROLES:
        return _ROOT_ROLES[name]
    match = _LAYER.fullmatch(name)
    if not match:
        return "unknown"
    suffix = match[2]
    if suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return "norm"
    if suffix.startswith("self_attn.") and suffix[len("self_attn."):] in _ATTENTION:
        return "attention_norm" if "layernorm" in suffix else "attention_projection"
    if (suffix.startswith("self_attn.indexer.")
            and suffix[len("self_attn.indexer."):] in _INDEXER):
        return "attention_indexer"
    if suffix == "mlp.gate.weight":
        return "moe_router"
    if suffix == "mlp.gate.e_score_correction_bias":
        return "router_bias"
    if re.fullmatch(r"mlp\.(gate|up|down)_proj\.weight", suffix):
        return "dense_ffn"
    if re.fullmatch(r"mlp\.shared_experts\.(gate|up|down)_proj\.weight", suffix):
        return "shared_expert"
    if (re.fullmatch(r"mlp\.experts\.(0|[1-9][0-9]*)\.(gate|up|down)_proj\.weight", suffix)
            or suffix in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
        return "expert_projection"
    return "unknown"


def _classify(name):
    """Return one anchored role; a projection's scale is never a router."""
    if not isinstance(name, str):
        return ["unknown"]
    if name.endswith(".weight_scale_inv"):
        base = name[:-len("_scale_inv")]
        if _base_role(base) != "unknown":
            return ["fp8_scale"]
    return [_base_role(name)]


def _dimension(config, key, findings, maximum=MAX_DIMENSION):
    value = config.get(key)
    if type(value) is not int or not 1 <= value <= maximum:
        findings.add("CONFIG_DIMENSION_REQUIRED", {"field": key, "maximum": maximum})
        return None
    return value


def _profile(config, findings):
    """Validate shape-affecting configuration without inventing defaults."""
    if not isinstance(config, dict):
        findings.add("CONFIG_REQUIRED", "An explicit checkpoint config is required")
        return None
    before = sum(findings.counts.values())
    if config.get("model_type") != "glm_moe_dsa":
        findings.add("UNSUPPORTED_MODEL_TYPE", "Only glm_moe_dsa has a metadata profile")
    layers = _dimension(config, "num_hidden_layers", findings, MAX_LAYERS)
    experts = _dimension(config, "n_routed_experts", findings, MAX_EXPERTS)
    for key in ("hidden_size", "vocab_size", "intermediate_size", "moe_intermediate_size",
                "num_attention_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
                "qk_rope_head_dim", "v_head_dim", "index_n_heads", "index_head_dim"):
        _dimension(config, key, findings)
    _dimension(config, "n_shared_experts", findings, MAX_EXPERTS)
    for key in ("attention_bias", "mlp_bias", "tie_word_embeddings"):
        if config.get(key) is not False:
            findings.add("UNSUPPORTED_OR_MISSING_CONFIG_FLAG", {"field": key, "required": False})
    quant = config.get("quantization_config")
    if (not isinstance(quant, dict) or quant.get("quant_method") != "fp8"
            or quant.get("fmt") != "e4m3" or quant.get("scale_fmt", "float") != "float"
            or not isinstance(quant.get("weight_block_size"), list)
            or len(quant["weight_block_size"]) != 2
            or any(type(n) is not int or n != 128 for n in quant["weight_block_size"])):
        findings.add("UNSUPPORTED_QUANTIZATION_PROFILE", "Requires fp8/e4m3, 128x128 blocks, float scales")
    mlps, indexers = None, None
    if layers is not None:
        attention = config.get("layer_types")
        if (not isinstance(attention, list) or len(attention) != layers
                or any(kind != "deepseek_sparse_attention" for kind in attention)):
            findings.add("ATTENTION_SCHEDULE_REQUIRED", "layer_types must explicitly select deepseek_sparse_attention")
        indexers = config.get("indexer_types")
        if (not isinstance(indexers, list) or len(indexers) != layers
                or any(kind not in ("full", "shared") for kind in indexers)
                or indexers[0] != "full"):
            findings.add("INDEXER_SCHEDULE_REQUIRED", "indexer_types must use full/shared and start with full")
            indexers = None
        if "mlp_layer_types" in config:
            declared = config["mlp_layer_types"]
            if (not isinstance(declared, list) or len(declared) != layers
                    or any(kind not in ("dense", "sparse", "moe") for kind in declared)):
                findings.add("MLP_SCHEDULE_REQUIRED", "mlp_layer_types must explicitly select dense/sparse")
            else:
                mlps = ["dense" if kind == "dense" else "moe" for kind in declared]
        if "first_k_dense_replace" in config:
            first = config["first_k_dense_replace"]
            if type(first) is not int or not 0 <= first <= layers:
                findings.add("INVALID_DENSE_LAYER_COUNT", "first_k_dense_replace must be in [0, num_hidden_layers]")
            else:
                schedule = ["dense"] * first + ["moe"] * (layers - first)
                if mlps is not None and mlps != schedule:
                    findings.add("CONFLICTING_MLP_SCHEDULES", "Explicit MLP schedules disagree")
                elif mlps is None and "mlp_layer_types" not in config:
                    mlps = schedule
        if mlps is None:
            findings.add("MLP_SCHEDULE_REQUIRED", "Supply mlp_layer_types or first_k_dense_replace")
    if sum(findings.counts.values()) != before:
        return None
    required_count = (3 + 9 * layers + 5 * indexers.count("full")
                      + sum(3 if kind == "dense" else 5 + 3 * experts for kind in mlps))
    if required_count > MAX_INVENTORY:
        findings.add("REQUIRED_INVENTORY_LIMIT", {"required": required_count, "maximum": MAX_INVENTORY})
        return None
    required = set(_ROOT_ROLES)
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        required.update(prefix + suffix for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"))
        required.update(prefix + "self_attn." + suffix for suffix in _ATTENTION)
        if indexers[layer] == "full":
            required.update(prefix + "self_attn.indexer." + suffix for suffix in _INDEXER)
        groups = [""] if mlps[layer] == "dense" else ["shared_experts."] + [f"experts.{e}." for e in range(experts)]
        for group in groups:
            required.update(prefix + "mlp." + group + direction + "_proj.weight" for direction in ("gate", "up", "down"))
        if mlps[layer] == "moe":
            required.update((prefix + "mlp.gate.weight", prefix + "mlp.gate.e_score_correction_bias"))
    return {"layers": layers, "experts": experts, "mlps": mlps, "indexers": indexers, "required": required}


def _valid_name(name):
    try:
        return (isinstance(name, str) and 1 <= len(name.encode("utf-8")) <= 512
                and not any(ord(char) < 32 for char in name))
    except UnicodeError:
        return False


def analyze_catalogue(tensors, config=None, *, complete=False):
    """Review bounded header records; only complete metadata can pass.

    complete must come from verified catalogue provenance and successful full
    shard audit. Each record requires name, dtype, shape and nbytes. Shard/offset
    provenance belongs to the caller. This function reads no files or payloads.
    """
    findings = Findings()
    profile = _profile(config, findings)
    if complete is not True:
        findings.add("INCOMPLETE_CATALOGUE", "Full successful metadata audit has not been established")
    inventory, groups, layers = {}, defaultdict(lambda: {"count": 0, "samples": []}), set()
    invalid, scanned = set(), 0
    try:
        iterator = iter(tensors) if not isinstance(tensors, (str, bytes, dict)) else iter(())
        if isinstance(tensors, (str, bytes, dict)):
            findings.add("INVALID_CATALOGUE", "Expected an iterable of tensor records")
    except TypeError:
        iterator = iter(())
        findings.add("INVALID_CATALOGUE", "Expected an iterable of tensor records")
    for record in iterator:
        scanned += 1
        if scanned > MAX_INVENTORY:
            findings.add("INVENTORY_LIMIT", {"maximum": MAX_INVENTORY})
            break
        if not isinstance(record, dict) or not _valid_name(record.get("name")):
            findings.add("INVALID_TENSOR_RECORD", {"record_number": scanned})
            continue
        name = record["name"]
        if name in inventory:
            findings.add("DUPLICATE_TENSOR", name)
            invalid.add(name)
            continue
        role = _classify(name)[0]
        groups[role]["count"] += 1
        if len(groups[role]["samples"]) < 8:
            groups[role]["samples"].append(name)
        match = _LAYER.fullmatch(name)
        if match:
            layers.add(int(match[1]))
        if role == "unknown":
            findings.add("UNKNOWN_TENSOR_NAME", name)
        dtype, shape, nbytes = record.get("dtype"), record.get("shape"), record.get("nbytes")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            findings.add("INVALID_DTYPE", name)
            invalid.add(name)
            dtype = None
        if (not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 8
                or any(type(n) is not int or not 1 <= n <= MAX_DIMENSION for n in shape)):
            findings.add("INVALID_SHAPE", name)
            invalid.add(name)
            shape = None
        else:
            shape = tuple(shape)
        if (type(nbytes) is not int or not 0 <= nbytes <= MAX_PAYLOAD_BYTES
                or (shape is not None and dtype is not None
                    and nbytes != math.prod(shape) * DTYPE_BYTES[dtype])):
            findings.add("INVALID_NBYTES", name)
            invalid.add(name)
        inventory[name] = {"dtype": dtype, "shape": shape, "role": role}
    if not inventory:
        findings.add("EMPTY_CATALOGUE", "No valid tensor names were supplied")
    required = profile["required"] if profile else set()
    missing = required.difference(inventory)
    for name in sorted(missing):
        findings.add("MISSING_REQUIRED_TENSOR", name)
    shapes, fp8 = Counter(), Counter()
    for name, tensor in inventory.items():
        shape, dtype, role = tensor["shape"], tensor["dtype"], tensor["role"]
        if role == "fp8_scale":
            continue
        if profile and name not in required:
            findings.add("UNEXPECTED_TENSOR", name)
            shapes["unexpected"] += 1
        if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
            findings.add("UNSUPPORTED_PACKED_EXPERT_LAYOUT", name)
        expected = known_shape(name, config) if isinstance(config, dict) else None
        if expected is None:
            shapes["unreviewed"] += 1
        elif shape != expected:
            findings.add("CONFIG_TENSOR_SHAPE_MISMATCH", {"name": name, "expected": list(expected),
                                                         "actual": list(shape) if shape is not None else None})
            invalid.add(name)
        else:
            shapes["matched"] += 1
        if dtype == "F8_E4M3":
            fp8["weight_count"] += 1
            scale_name = name + "_scale_inv"
            if not name.endswith(".weight") or shape is None or len(shape) != 2 or role == "unknown":
                findings.add("UNSUPPORTED_FP8_WEIGHT", name)
                invalid.add(name)
            elif scale_name not in inventory:
                findings.add("MISSING_FP8_SCALE", name)
                invalid.add(name)
    for name, tensor in inventory.items():
        if tensor["role"] != "fp8_scale":
            continue
        base = name[:-len("_scale_inv")]
        weight = inventory.get(base)
        fp8["scale_count"] += 1
        if weight is None:
            findings.add("ORPHAN_FP8_SCALE", name)
            invalid.add(name)
            continue
        if weight["dtype"] != "F8_E4M3":
            findings.add("SCALE_FOR_NON_FP8_WEIGHT", name)
            invalid.add(name)
        if tensor["dtype"] != "F32":
            findings.add("INVALID_FP8_SCALE_DTYPE", name)
            invalid.add(name)
        shape = weight["shape"]
        expected = tuple((n + 127) // 128 for n in shape) if shape is not None and len(shape) == 2 else None
        if expected is None or tensor["shape"] != expected:
            findings.add("FP8_SCALE_GRID_MISMATCH", name)
            invalid.add(name)
        if name not in invalid and base not in invalid:
            fp8["valid_metadata_pairs"] += 1
    if not fp8["weight_count"]:
        findings.add("NO_FP8_WEIGHTS_OBSERVED", "This profile requires at least one FP8 weight")
    verified = profile is not None and complete is True and sum(findings.counts.values()) == 0
    expected_layers = profile["layers"] if profile else None
    return {
        "status": "PASS" if verified else "REVIEW_REQUIRED",
        "metadata_mapping_verified": verified, "architecture_mapping_verified": verified,
        "metadata_only": True, "payload_values_verified": False,
        "real_checkpoint_compatible": False, "inference_verified": False,
        "complete_catalogue": complete is True and scanned <= MAX_INVENTORY,
        "inventory": {"record_count": scanned, "unique_tensor_count": len(inventory),
                      "maximum_records": MAX_INVENTORY},
        "profile": {"name": "glm_moe_dsa_unpacked_fp8_metadata_v1", "supported": profile is not None,
                    "limitations": ["Header inventory and shape/dtype/scale-grid checks only",
                                    "Explicit dense/MoE and full/shared indexer schedules required",
                                    "Packed experts, MTP/extra layers and unknown tensors require review",
                                    "No payload values, graph execution or real-checkpoint compatibility verified"]},
        "layers": {"ids": sorted(layers), "count_detected": len(layers),
                   "count_expected": expected_layers, "verified": verified},
        "groups": {role: groups.get(role, {"count": 0, "samples": []}) for role in ROLES},
        "shape_checks": {"required_tensor_count": len(required), "missing_required_count": len(missing),
                         "matched_count": shapes["matched"], "invalid_count": len(invalid),
                         "unreviewed_count": shapes["unreviewed"], "unexpected_count": shapes["unexpected"]},
        "attention": {"verified": verified, "projection_tensor_count": groups["attention_projection"]["count"]},
        "moe": {"verified": verified and "moe" in profile["mlps"],
                "expert_tensor_count": groups["expert_projection"]["count"]},
        "fp8": {"scale_tensor_count": fp8["scale_count"], "weight_tensor_count": fp8["weight_count"],
                "valid_metadata_pairs": fp8["valid_metadata_pairs"], "metadata_verified": verified},
        "findings": findings.report(),
    }
