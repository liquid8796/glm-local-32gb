"""Bounded public JSON metadata access; never fetch weights or execute remote code."""

import json
import re
import urllib.parse
import urllib.request
from pathlib import PurePosixPath

MAX_METADATA_BYTES = 8 * 1024 * 1024


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "glm-local-feasibility/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Metadata exceeds 8 MiB limit")
    return json.loads(raw)


def safe_relative_path(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(char) < 32 for char in name)):
        raise ValueError("Invalid checkpoint relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("Unsafe checkpoint relative path")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)}
    if any(p.rstrip(". ") != p or p.split(".")[0].upper() in reserved for p in path.parts):
        raise ValueError("Unsafe Windows checkpoint path")
    return path


def summarize_metadata(model_id, revision, metadata, config):
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be an object")
    if metadata.get("id") != model_id or metadata.get("sha") != revision:
        raise ValueError("Model identity/revision mismatch; refusing mixed snapshots")
    if not isinstance(config, dict) or not config.get("model_type"):
        raise ValueError("Model config has no architecture")
    weights = []
    seen = set()
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("Model siblings must be a list")
    for entry in siblings:
        if not isinstance(entry, dict) or not isinstance(entry.get("rfilename"), str):
            raise ValueError("Invalid model sibling entry")
        name = entry.get("rfilename", "")
        if not name.endswith(".safetensors"):
            continue
        safe_relative_path(name)
        if name.casefold() in seen:
            raise ValueError("Duplicate checkpoint shard")
        seen.add(name.casefold())
        size = entry.get("size")
        if type(size) is not int or size <= 0:
            raise ValueError("Missing or invalid checkpoint shard size")
        weights.append({"name": name, "bytes": size})
    if not weights:
        raise ValueError("Metadata contains no sized safetensors shards")
    keys = ("model_type", "architectures", "num_hidden_layers", "hidden_size",
            "n_routed_experts", "num_experts_per_tok", "first_k_dense_replace",
            "moe_intermediate_size", "vocab_size", "index_topk", "kv_lora_rank")
    architecture = {key: config.get(key) for key in keys}
    quant = config.get("quantization_config", {})
    if not isinstance(quant, dict):
        raise ValueError("quantization_config must be an object")
    architecture["quantization"] = {
        key: quant.get(key) for key in ("quant_method", "fmt", "weight_block_size")
    }
    return {
        "model_id": model_id,
        "revision": revision,
        "weight_bytes": sum(w["bytes"] for w in weights),
        "weight_shards": len(weights),
        "parameters": metadata.get("safetensors", {}),
        "architecture": architecture,
        "weights": weights,
        "sources": {
            "model": f"https://huggingface.co/{model_id}/tree/{revision}",
            "config": f"https://huggingface.co/{model_id}/blob/{revision}/config.json",
        },
    }


def refresh_metadata(model_id, revision):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        raise ValueError("Expected Hugging Face owner/model identifier")
    if any(part in (".", "..") for part in model_id.split("/")):
        raise ValueError("Invalid Hugging Face owner/model identifier")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A pinned 40-character revision is required")
    repo = urllib.parse.quote(model_id, safe="/")
    base = "https://huggingface.co"
    metadata = fetch_json(f"{base}/api/models/{repo}/revision/{revision}?blobs=true")
    config = fetch_json(f"{base}/{repo}/resolve/{revision}/config.json")
    return summarize_metadata(model_id, revision, metadata, config)
