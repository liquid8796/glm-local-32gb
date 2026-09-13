"""Layer-wise prompt execution with bounded reuse of deterministic projections.

The scalar attention/router functions remain the source of arithmetic order.
Only independent linear projections are prepared in batches; a per-layer cache
returns the same FP32 values when those functions request them.
"""
from array import array
from collections import OrderedDict
import math

from .checkpoint_schema import known_shape
from .runtime_linear import MAX_LINEAR_BATCH

PREFILL_SCRATCH_BYTES = 128 * 1024**2
_CACHE_BYTES = 96 * 1024**2
_CACHE_ENTRIES = 4096


def workspace_required(config, count):
    """Conservative retained arrays plus bounded memo, before reading weights."""
    largest_ffn = max(config.get("intermediate_size", 0), config["moe_intermediate_size"],
                      config["moe_intermediate_size"] * config["n_shared_experts"])
    elements = (4 * config["hidden_size"] + 3 * config["q_lora_rank"]
                + 3 * (config["kv_lora_rank"] + config["qk_rope_head_dim"])
                + 4 * config["num_attention_heads"] * (config["qk_nope_head_dim"] + config["qk_rope_head_dim"] + config["v_head_dim"])
                + 4 * config["index_n_heads"] * config["index_head_dim"] + 6 * largest_ffn)
    return _CACHE_BYTES + 4 * count * elements + 8 * config["vocab_size"]


class _Projections:
    def __init__(self, decoder, original):
        self.decoder, self.original = decoder, original
        many = getattr(decoder.weights, "linear_many", None)
        default = getattr(decoder.weights, "linear", None)
        self.many = many if (callable(many) and original == default) else None
        self.entries = OrderedDict()
        self.bytes = self.peak = self.hits = self.misses = 0

    def key(self, name, values):
        return name, array("f", values).tobytes()

    def store(self, key, result):
        from .streaming_decoder import _compact
        shape = known_shape(key[0], self.decoder.config)
        if shape is None or len(shape) != 2:
            raise ValueError("Prefill projection has no supported matrix shape")
        result = _compact(result, shape[0], key[0])
        size = len(key[1]) + len(result) * 4 + len(key[0].encode("utf-8"))
        if size > _CACHE_BYTES:
            return result
        if key in self.entries:
            _, old = self.entries.pop(key)
            self.bytes -= old
        while self.entries and (self.bytes + size > _CACHE_BYTES or len(self.entries) >= _CACHE_ENTRIES):
            _, (_, old) = self.entries.popitem(last=False)
            self.bytes -= old
        self.entries[key] = result, size
        self.bytes += size
        self.peak = max(self.peak, self.bytes)
        return result

    def __call__(self, name, values):
        key = self.key(name, values)
        if key in self.entries:
            validate = getattr(self.decoder.weights, "assert_weight_unchanged", None)
            if callable(validate):
                validate(name)
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses += 1
        return self.store(key, self.original(name, values))

    def prepare(self, name, vectors):
        shape = known_shape(name, self.decoder.config)
        if shape is None or len(shape) != 2:
            raise ValueError("Prefill projection has no supported matrix shape")
        keys, missing, ready = [], OrderedDict(), {}
        for vector in vectors:
            if len(vector) != shape[1] or any(not math.isfinite(value) for value in vector):
                raise ValueError("Prefill input shape or values are invalid")
            key = self.key(name, vector)
            keys.append(key)
            if key in self.entries:
                if key not in ready:
                    ready[key] = self(name, vector)
            else:
                missing.setdefault(key, vector)
        pending = list(missing.items())
        for start in range(0, len(pending), MAX_LINEAR_BATCH):
            chunk = pending[start:start + MAX_LINEAR_BATCH]
            inputs = [vector for _, vector in chunk]
            self.decoder._report_progress("projection_batch", tensor_name=name, rows=shape[0], cols=shape[1])
            outputs = self.many(name, inputs) if self.many is not None else [self.original(name, vector) for vector in inputs]
            if not isinstance(outputs, (list, tuple)) or len(outputs) != len(inputs):
                raise ValueError("Prefill batched projection returned an invalid batch")
            for (key, _), result in zip(chunk, outputs):
                # Fresh results are already bound/validated by the projection.
                # Keep this bounded batch even if the LRU evicts its early keys;
                # fetching it through the memo again could recompute those rows.
                ready[key] = self.store(key, result)
        return [ready[key] for key in keys]

    def clear(self):
        self.entries.clear()
        self.bytes = 0


def _ffn(cache, prefix, vectors):
    from .streaming_decoder import _sigmoid
    gates = cache.prepare(prefix + "gate_proj.weight", vectors)
    ups = cache.prepare(prefix + "up_proj.weight", vectors)
    activated = [array("f", (g * _sigmoid(g) * u for g, u in zip(gate, up))) for gate, up in zip(gates, ups)]
    return cache.prepare(prefix + "down_proj.weight", activated)


def _prepare_moe(decoder, cache, layer, vectors):
    from .streaming_decoder import _sigmoid, _topk
    config = decoder.config
    prefix = f"model.layers.{layer}.mlp."
    logits = cache.prepare(prefix + "gate.weight", vectors)
    bias = decoder._vector(prefix + "gate.e_score_correction_bias", config["n_routed_experts"])
    by_expert = {}
    for index, (vector, scores) in enumerate(zip(vectors, logits)):
        scores = array("f", (_sigmoid(value) for value in scores))
        choice = array("f", (value + correction for value, correction in zip(scores, bias)))
        width = len(choice) // config["n_group"]
        import heapq
        group_scores = [math.fsum(heapq.nlargest(2, choice[start:start + width]))
                        for start in range(0, len(choice), width)]
        groups = set(_topk(group_scores, config["topk_group"]))
        for expert in range(len(choice)):
            if expert // width not in groups:
                choice[expert] = float("-inf")
        selected = _topk(choice, config["num_experts_per_tok"])
        denominator = math.fsum(scores[expert] for expert in selected) + 1e-20 if config["norm_topk_prob"] else 1.0
        for expert in selected:
            weight = scores[expert] / denominator * config["routed_scaling_factor"]
            by_expert.setdefault(expert, []).append((index, vector, weight))
    output = [array("f", [0.0]) * config["hidden_size"] for _ in vectors]
    # Each token receives expert contributions in precisely the scalar _moe
    # ascending-ID order; batching only interleaves independent tokens.
    for expert, selected in sorted(by_expert.items()):
        contributions = _ffn(cache, prefix + f"experts.{expert}.", [vector for _, vector, _ in selected])
        for (index, _, weight), contribution in zip(selected, contributions):
            for column, value in enumerate(contribution):
                output[index][column] += weight * value
        contributions = None
    shared = _ffn(cache, prefix + "shared_experts.", vectors)
    return [array("f", (base + delta for base, delta in zip(row, common))) for row, common in zip(output, shared)]


def prefill_chunk(decoder, token_ids, *, output_logits=True):
    """Process 1..16 causal prompt tokens, committing position after all layers."""
    from .streaming_decoder import _compact
    if not isinstance(token_ids, (list, tuple, array)) or not 1 <= len(token_ids) <= MAX_LINEAR_BATCH:
        raise ValueError("Prefill chunk must contain 1..16 token IDs")
    decoder._require_open()
    start = decoder.position
    if start + len(token_ids) > decoder.plan.settings.context_tokens:
        raise ValueError("Prefill chunk exceeds the planned context")
    for token in token_ids:
        decoder._token(token)
    if workspace_required(decoder.config, len(token_ids)) > PREFILL_SCRATCH_BYTES:
        raise ValueError("Prefill batch activations exceed the 128-MiB workspace; choose a smaller prefill batch")
    original = decoder._linear
    old_trace = decoder.last_trace
    cache = _Projections(decoder, original)
    config = decoder.config
    hidden_size = config["hidden_size"]
    rank = config["kv_lora_rank"]
    previous = [None] * len(token_ids)
    decoder._report_progress("prefill_batch", phase="prefill", token_index=start,
                             token_count=start + len(token_ids), prompt_tokens=start + len(token_ids))
    with decoder.ledger.reserve(PREFILL_SCRATCH_BYTES + decoder._scratch_bytes, label="prefill_batch_scratch"):
        try:
            hidden = [_compact(decoder.weights.embedding(token), hidden_size, "embedding") for token in token_ids]
            decoder._linear = cache
            for layer in range(decoder.plan.layer_count):
                prefix = f"model.layers.{layer}."
                attention = prefix + "self_attn."
                decoder._report_progress("prefill_layer", layer_index=layer, layer_count=decoder.plan.layer_count)
                normalized = [decoder._norm(prefix + "input_layernorm.weight", row, config["rms_norm_eps"]) for row in hidden]
                residual = cache.prepare(attention + "q_a_proj.weight", normalized)
                residual = [decoder._norm(attention + "q_a_layernorm.weight", row, 1e-6) for row in residual]
                cache.prepare(attention + "q_b_proj.weight", residual)
                compressed = cache.prepare(attention + "kv_a_proj_with_mqa.weight", normalized)
                latent = [decoder._norm(attention + "kv_a_layernorm.weight", row[:rank], 1e-6) for row in compressed]
                cache.prepare(attention + "kv_b_proj.weight", latent)
                if decoder._indexers[layer] == "full":
                    index = attention + "indexer."
                    cache.prepare(index + "wq_b.weight", residual)
                    cache.prepare(index + "wk.weight", normalized)
                    cache.prepare(index + "weights_proj.weight", normalized)
                del residual, compressed, latent
                # Scalar attention retains the established causal/indexer/softmax
                # order while all expensive independent projections hit the cache.
                attn_outputs = []
                for index, row in enumerate(normalized):
                    decoder._position = start + index
                    result, previous[index] = decoder._attention(layer, row, previous[index], project_output=False)
                    attn_outputs.append(result)
                attn_outputs = cache.prepare(attention + "o_proj.weight", attn_outputs)
                hidden = [array("f", (base + delta for base, delta in zip(row, result))) for row, result in zip(hidden, attn_outputs)]
                del attn_outputs, normalized
                cache.clear()
                normalized = [decoder._norm(prefix + "post_attention_layernorm.weight", row, config["rms_norm_eps"]) for row in hidden]
                if decoder._mlps[layer] == "dense":
                    mlp_output = _ffn(cache, prefix + "mlp.", normalized)
                else:
                    mlp_output = _prepare_moe(decoder, cache, layer, normalized)
                hidden = [array("f", (base + delta for base, delta in zip(row, result))) for row, result in zip(hidden, mlp_output)]
                del normalized, mlp_output
                cache.clear()
                decoder._report_progress("prefill_layer_complete", layer_index=layer, layer_count=decoder.plan.layer_count)
            decoder._linear = original
            decoder._position = start + len(token_ids) - 1
            logits = None
            if output_logits:
                decoder._report_progress("output_head", position=decoder._position)
                logits = decoder._project("lm_head.weight", decoder._norm("model.norm.weight", hidden[-1], config["rms_norm_eps"]))
            decoder.last_trace = {"position": decoder._position, "token": token_ids[-1],
                                  "backbone_layers": decoder.plan.layer_count, "last_selected_indices": list(previous[-1]),
                                  "prefill_batch_tokens": len(token_ids), "projection_cache_peak_bytes": cache.peak,
                                  "projection_cache_hits": cache.hits, "projection_cache_misses": cache.misses}
            decoder._position = start + len(token_ids)
            return logits
        except BaseException:
            decoder._position = start
            decoder.last_trace = old_trace
            decoder._expanded_cache.truncate(start)
            raise
        finally:
            decoder._linear = original
            cache.clear()
