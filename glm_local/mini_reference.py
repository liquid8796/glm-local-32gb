"""Independent, dense float64 oracle for the fixed synthetic mini decoder.

The oracle deliberately recomputes the complete causal sequence on every call.
It does not use the incremental engine, native kernels, weight reader, or their
nonlinear helpers.  Only the immutable synthetic dimensions are shared.  No
real checkpoint names, configuration, or tokenizer are accepted here.

Selection is deterministic: descending score, then ascending position/expert
number.  This tie rule is a test contract, not a claim about PyTorch topk ties.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .mini_spec import SPEC, matrix_shapes, vector_lengths


_FILE_LIMIT = 64 * 1024


def _bounded_read(path):
    if path.stat().st_size > _FILE_LIMIT:
        raise ValueError("Synthetic reference files must each fit in 64 KiB")
    with path.open("rb") as stream:
        payload = stream.read(_FILE_LIMIT + 1)
    if len(payload) > _FILE_LIMIT:
        raise ValueError("Synthetic reference files must each fit in 64 KiB")
    return payload


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate synthetic manifest key")
        result[key] = value
    return result


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _rms(values, weight, epsilon):
    return values / np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True)
                            + epsilon) * weight


def _rotate_adjacent(values):
    """GLM interleaved input pairs -> rotated even half followed by odd half."""
    length, width = values.shape[0], values.shape[-1]
    frequencies = np.power(SPEC["rope_theta"], -np.arange(0, width, 2) / width)
    angles = np.arange(length)[:, None] * frequencies[None, :]
    shape = (length,) + (1,) * (values.ndim - 2) + (width // 2,)
    cosine, sine = np.cos(angles).reshape(shape), np.sin(angles).reshape(shape)
    even, odd = values[..., 0::2], values[..., 1::2]
    return np.concatenate((even * cosine - odd * sine,
                           odd * cosine + even * sine), axis=-1)


def _sigmoid(values):
    return np.exp(-np.logaddexp(0.0, -values))


class ReferenceDecoder:
    """Load at most two 64 KiB synthetic files and evaluate bounded sequences."""

    def __init__(self, directory):
        directory = Path(directory)
        try:
            manifest = json.loads(_bounded_read(directory / "manifest.json"),
                                  object_pairs_hook=_unique_object)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid synthetic reference manifest") from error
        required = {"format", "spec", "seed", "weight_file", "weight_bytes",
                    "sha256", "matrices", "vectors"}
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ValueError("Unsupported synthetic reference manifest fields")
        if (manifest["format"] != "glm-synthetic-mini-v1"
                or json.dumps(manifest["spec"], sort_keys=True)
                != json.dumps(SPEC, sort_keys=True)):
            raise ValueError("Only the fixed synthetic mini specification is accepted")
        if type(manifest["seed"]) is not int or not 0 <= manifest["seed"] < 2**32:
            raise ValueError("Invalid synthetic seed")
        if manifest["weight_file"] != "weights.bin":
            raise ValueError("Only the local synthetic weights.bin file is accepted")

        shapes, lengths = matrix_shapes(), vector_lengths()
        records, vectors = manifest["matrices"], manifest["vectors"]
        if not isinstance(records, dict) or set(records) != set(shapes):
            raise ValueError("Synthetic matrix names do not match the fixed specification")
        if not isinstance(vectors, dict) or set(vectors) != set(lengths):
            raise ValueError("Synthetic vector names do not match the fixed specification")

        expected_bytes = sum(rows * cols for rows, cols in shapes.values())
        if (type(manifest["weight_bytes"]) is not int
                or manifest["weight_bytes"] != expected_bytes):
            raise ValueError("Invalid synthetic weight byte count")
        payload = _bounded_read(directory / "weights.bin")
        if len(payload) != expected_bytes:
            raise ValueError("Synthetic weight file is truncated or has trailing data")
        checksum = manifest["sha256"]
        if (not isinstance(checksum, str) or len(checksum) != 64
                or hashlib.sha256(payload).hexdigest() != checksum):
            raise ValueError("Synthetic weight SHA-256 mismatch")

        # Build the E4M3FN table independently in NumPy. Exponent 15 is finite
        # except mantissa 7; subnormals have magnitude mantissa / 512.
        codes = np.arange(256, dtype=np.uint16)
        exponents, fractions = (codes & 0x7f) >> 3, codes & 7
        magnitudes = np.where(exponents == 0, fractions / 512.0,
                              (1.0 + fractions / 8.0)
                              * np.exp2(exponents.astype(np.float64) - 7.0))
        table = np.where(codes < 128, magnitudes, -magnitudes)
        table[(codes & 0x7f) == 0x7f] = np.nan

        matrices, intervals = {}, []
        for name, (rows, cols) in shapes.items():
            record = records[name]
            if not isinstance(record, dict) or set(record) != {"offset", "rows", "cols", "scale"}:
                raise ValueError("Invalid synthetic matrix record")
            if any(type(record[key]) is not int for key in ("offset", "rows", "cols")):
                raise ValueError("Synthetic matrix offsets and dimensions must be integers")
            start = record["offset"]
            if (record["rows"] != rows or record["cols"] != cols or start < 0
                    or start + rows * cols > len(payload)):
                raise ValueError("Synthetic matrix shape or offset mismatch")
            scale = record["scale"]
            if not _number(scale) or scale <= 0:
                raise ValueError("Synthetic matrix scale must be finite and positive")
            with np.errstate(over="ignore", invalid="ignore"):
                matrix = table[np.frombuffer(payload, dtype=np.uint8,
                                             count=rows * cols, offset=start)] * scale
            if not np.isfinite(matrix).all():
                raise ValueError("Synthetic matrix contains nonfinite FP8 weights")
            matrices[name] = matrix.reshape(rows, cols)
            intervals.append((start, start + rows * cols))
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("Synthetic matrix payloads must be contiguous and disjoint")
            cursor = end
        if cursor != len(payload):
            raise ValueError("Synthetic matrix payloads must cover the entire weight file")

        vector_arrays = {}
        for name, length in lengths.items():
            vector = vectors[name]
            if (not isinstance(vector, list) or len(vector) != length
                    or any(not _number(item) for item in vector)):
                raise ValueError("Invalid synthetic vector shape or nonfinite value")
            vector_arrays[name] = np.asarray(vector, dtype=np.float64)
        self._matrices = matrices
        self._vectors = vector_arrays

    def forward(self, tokens):
        """Return T x 32 logits and selection traces, without retaining a KV cache."""
        if (not isinstance(tokens, (list, tuple)) or not 1 <= len(tokens) <= 128
                or any(type(token) is not int or not 0 <= token < 32 for token in tokens)):
            raise ValueError("Synthetic reference requires 1-128 integer token IDs in [0, 31]")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                result = self._forward(tokens)
            except FloatingPointError as error:
                raise ValueError("Synthetic reference arithmetic produced a nonfinite value") from error
        if not np.isfinite(result["logits"]).all():
            raise ValueError("Synthetic reference logits must be finite")
        return result

    def _forward(self, tokens):
        matrices, vectors = self._matrices, self._vectors
        length, heads = len(tokens), SPEC["heads"]
        hidden = matrices["embed"][np.asarray(tokens, dtype=np.int64)].copy()
        selections = None
        routed_experts, router_weights = None, None
        for layer in range(SPEC["layers"]):
            prefix = f"layer.{layer}."
            normalized = _rms(hidden, vectors[prefix + "in_norm"], SPEC["norm_eps"])
            query_latent = _rms(normalized @ matrices[prefix + "q_a"].T,
                                vectors[prefix + "q_norm"], SPEC["latent_eps"])
            query = (query_latent @ matrices[prefix + "q_b"].T).reshape(length, heads, 8)
            query = np.concatenate((query[..., :4], _rotate_adjacent(query[..., 4:])), axis=-1)
            compressed = normalized @ matrices[prefix + "kv_a"].T
            key_latent = _rms(compressed[:, :4], vectors[prefix + "kv_norm"],
                              SPEC["latent_eps"])
            key_value = (key_latent @ matrices[prefix + "kv_b"].T).reshape(length, heads, 8)
            rotated_key = np.broadcast_to(_rotate_adjacent(compressed[:, 4:])[:, None, :],
                                          (length, heads, 4))
            key = np.concatenate((key_value[..., :4], rotated_key), axis=-1)
            value = key_value[..., 4:]

            if layer == 0:
                index_query = (query_latent @ matrices[prefix + "index_q"].T).reshape(
                    length, SPEC["index_heads"], SPEC["index_dim"])
                index_key = normalized @ matrices[prefix + "index_k"].T
                centered = index_key - index_key.mean(axis=-1, keepdims=True)
                index_key = centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True)
                                               + SPEC["latent_eps"])
                index_key = (index_key * vectors[prefix + "index_norm_weight"]
                             + vectors[prefix + "index_norm_bias"])
                index_query = np.concatenate((_rotate_adjacent(index_query[..., :4]),
                                               index_query[..., 4:]), axis=-1)
                index_key = np.concatenate((_rotate_adjacent(index_key[..., :4]),
                                             index_key[..., 4:]), axis=-1)
                head_weight = (normalized @ matrices[prefix + "index_weight"].T
                               / math.sqrt(SPEC["index_heads"]))
                score_per_head = np.einsum("thd,sd->ths", index_query, index_key) / math.sqrt(8)
                index_scores = np.sum(np.maximum(score_per_head, 0.0)
                                      * head_weight[:, :, None], axis=1)
                selections = [sorted(range(position + 1),
                                     key=lambda earlier: (-index_scores[position, earlier], earlier))[:4]
                              for position in range(length)]

            # A dense mask over the whole causal sequence makes this evaluation
            # structurally independent from a per-token incremental KV cache.
            attention_scores = np.einsum("thd,shd->ths", query, key) / math.sqrt(8)
            mask = np.zeros((length, length), dtype=bool)
            for position, selected in enumerate(selections):
                mask[position, selected] = True
            attention_scores = np.where(mask[:, None, :], attention_scores, -np.inf)
            maximum = np.max(attention_scores, axis=-1, keepdims=True)
            probabilities = np.exp(attention_scores - maximum)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attended = np.einsum("ths,shv->thv", probabilities, value).reshape(length, 8)
            hidden = hidden + attended @ matrices[prefix + "o"].T

            normalized = _rms(hidden, vectors[prefix + "post_norm"], SPEC["norm_eps"])

            def feed_forward(name):
                gate = normalized @ matrices[name + "gate"].T
                activated = gate * _sigmoid(gate)
                return (activated * (normalized @ matrices[name + "up"].T)) @ matrices[name + "down"].T

            if layer == 0:
                hidden = hidden + feed_forward(prefix + "mlp.")
            else:
                router_scores = _sigmoid(normalized @ matrices[prefix + "router"].T)
                ranking_scores = router_scores + vectors[prefix + "router_bias"]
                routed_experts = [sorted(range(4), key=lambda expert: (
                    -ranking_scores[position, expert], expert))[:2] for position in range(length)]
                selected_scores = np.take_along_axis(router_scores, np.asarray(routed_experts), axis=1)
                router_weights = (selected_scores / (selected_scores.sum(axis=1, keepdims=True) + 1e-20)
                                  * SPEC["routed_scale"])
                all_experts = np.stack([feed_forward(prefix + f"expert.{expert}.")
                                        for expert in range(4)], axis=1)
                chosen = all_experts[np.arange(length)[:, None], np.asarray(routed_experts)]
                mixture = np.sum(chosen * router_weights[:, :, None], axis=1)
                hidden = hidden + mixture + feed_forward(prefix + "shared.")

        hidden = _rms(hidden, vectors["final_norm"], SPEC["norm_eps"])
        logits = hidden @ matrices["lm_head"].T
        traces = [{"selected_indices": selections[position],
                   "routed_experts": routed_experts[position],
                   "router_weights": router_weights[position].tolist()}
                  for position in range(length)]
        return {"logits": logits.tolist(), "traces": traces}
