"""Fixed synthetic-only decoder contract; never accepts a real model config.

Equations follow selected GLM-MoE-DSA mechanisms. These bounds and invented
weights deliberately do not constitute a real-checkpoint loader/runtime.
"""

SPEC = {
    "format": "glm-synthetic-mini-v1", "hidden": 16, "vocab": 32,
    "layers": 2, "heads": 2, "q_rank": 8, "kv_rank": 4,
    "nope_dim": 4, "rope_dim": 4, "value_dim": 4,
    "index_heads": 2, "index_dim": 8, "index_topk": 4,
    "experts": 4, "experts_per_token": 2, "intermediate": 24,
    "layer_types": ["dense", "moe"], "indexer_types": ["full", "shared"],
    "norm_eps": 1e-5, "latent_eps": 1e-6, "rope_theta": 10000.0,
    "routed_scale": 2.5, "max_context": 128,
}
SOURCE_REVISION = "3f601734a3580f55484720770850966bba060e4f"
SOURCE_URL = ("https://github.com/huggingface/transformers/blob/" + SOURCE_REVISION
              + "/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py")


def matrix_shapes():
    result = {"embed": (32, 16), "lm_head": (32, 16)}
    for layer in range(2):
        p = f"layer.{layer}."
        result.update({p + "q_a": (8, 16), p + "q_b": (16, 8),
                       p + "kv_a": (8, 16), p + "kv_b": (16, 4),
                       p + "o": (16, 8)})
    result.update({"layer.0.index_q": (16, 8), "layer.0.index_k": (8, 16),
                   "layer.0.index_weight": (2, 16), "layer.1.router": (4, 16)})
    for p in ["layer.0.mlp.", "layer.1.shared."] + [f"layer.1.expert.{e}." for e in range(4)]:
        result.update({p + "gate": (24, 16), p + "up": (24, 16), p + "down": (16, 24)})
    return result


def vector_lengths():
    result = {"final_norm": 16, "layer.0.index_norm_weight": 8,
              "layer.0.index_norm_bias": 8, "layer.1.router_bias": 4}
    for layer in range(2):
        p = f"layer.{layer}."
        result.update({p + "in_norm": 16, p + "post_norm": 16,
                       p + "q_norm": 8, p + "kv_norm": 4})
    return result


def validate_tokens(tokens, *, allow_empty=False):
    if not isinstance(tokens, (list, tuple)):
        raise ValueError("Synthetic tokens must be a list/tuple of integers")
    if len(tokens) > SPEC["max_context"] or (not tokens and not allow_empty):
        raise ValueError("Synthetic context must contain 1-128 tokens")
    if any(type(token) is not int or not 0 <= token < SPEC["vocab"] for token in tokens):
        raise ValueError("Synthetic token IDs must be integers in [0, 31]")
