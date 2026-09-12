from collections import Counter, defaultdict


def _classify(name):
    groups = []
    n = name.lower()
    if "embed" in n:
        groups.append("embedding")
    if "lm_head" in n:
        groups.append("output_head")
    if "layer" in n or "layers" in n:
        groups.append("transformer")
    if "q_proj" in n or "k_proj" in n or "v_proj" in n or "o_proj" in n or "attention" in n or "attn" in n:
        groups.append("attention")
    if "expert" in n:
        groups.append("moe_expert")
    if "router" in n or "gate" in n:
        groups.append("moe_router")
    if "scale" in n:
        groups.append("fp8_scale")
    return groups or ["unknown"]


def analyze_catalogue(tensors):
    groups = defaultdict(list)
    layers = set()
    for tensor in tensors:
        name = tensor.get("name", "")
        for group in _classify(name):
            groups[group].append(name)
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layers.add(int(parts[i + 1]))
    experts = [x for x in groups["moe_expert"]]
    return {
        "layers": {
            "count_detected": len(layers),
            "ids": sorted(layers),
            "verified": bool(layers),
        },
        "groups": {k: {"count": len(v), "samples": v[:5]} for k, v in groups.items()},
        "attention": {"verified": bool(groups["attention"])},
        "moe": {"verified": bool(experts), "expert_tensor_count": len(experts)},
        "fp8": {"scale_tensor_count": len(groups["fp8_scale"])},
        "metadata_only": True,
        "payload_values_verified": False,
    }
