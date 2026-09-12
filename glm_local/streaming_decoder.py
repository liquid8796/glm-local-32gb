"""Config-derived GLM backbone decoder with bounded, compressed CPU caches.

The equations follow the pinned Transformers GLM-MoE-DSA implementation;
attention reconstructs one selected token's K/V at a time and uses online
softmax instead of retaining expanded keys/values. Stored values are FP32;
Python scalar reductions use float64. This is not a BF16/FP8-kernel numerical
emulation or evidence of real-checkpoint parity. Weights and GPU scheduling
belong to the injected adapter. MTP/speculative decoding is not executed.
"""
from __future__ import annotations

from array import array
from copy import deepcopy
import heapq
import math

from .architecture.mapper import _profile
from .checkpoint_schema import Findings, known_shape
from .residency import ReservationLedger, ResidencyError, ResidencyPlan, build_plan


class DecoderError(ValueError):
    """Unsupported execution semantics or invalid decoder input/output."""


def _number(value, name, *, positive=False):
    if type(value) not in (int, float):
        raise DecoderError(f"{name} must be finite")
    try:
        valid = math.isfinite(value) and (not positive or value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise DecoderError(f"{name} must be finite" + (" and positive" if positive else ""))
    return float(value)


def _compact(values, length, label):
    # Never drain an arbitrary/unbounded adapter generator. Providers must
    # return a sized vector; validate its length before copying its payload.
    if not isinstance(values, (array, list, tuple)) or len(values) != length:
        raise DecoderError(f"{label} must return exactly {length} values")
    try:
        if any(not math.isfinite(value) for value in values):
            raise DecoderError(f"{label} contains nonfinite values")
        result = array("f", values)
    except (TypeError, OverflowError) as exc:
        raise DecoderError(f"{label} cannot be represented as FP32") from exc
    if any(not math.isfinite(value) for value in result):
        raise DecoderError(f"{label} overflows FP32")
    return result


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _topk(scores, count):
    # An explicit portable tie rule. Torch/STL equal-score order is not part
    # of this contract; callers may inject a platform-verified top-k adapter.
    return heapq.nlargest(count, range(len(scores)), key=lambda i: (scores[i], -i))


def _validate_numeric_config(config):
    if config.get("hidden_act") != "silu":
        raise DecoderError("Only hidden_act=silu is implemented")
    if config.get("num_key_value_heads") != config["num_attention_heads"]:
        raise DecoderError("num_key_value_heads must equal num_attention_heads")
    if type(config.get("norm_topk_prob")) is not bool:
        raise DecoderError("norm_topk_prob must be explicitly boolean")
    for key in ("rope_interleave", "indexer_rope_interleave"):
        if key in config and config[key] is not True:
            raise DecoderError(f"Only {key}=true is implemented")
    if config.get("scoring_func", "sigmoid") != "sigmoid":
        raise DecoderError("Only sigmoid MoE routing is implemented")
    if config.get("topk_method", "noaux_tc") != "noaux_tc":
        raise DecoderError("Only noaux_tc grouped routing is implemented")
    if config.get("use_cache") is not True:
        raise DecoderError("Sequential execution requires use_cache=true")
    if _number(config.get("attention_dropout"), "attention_dropout") != 0:
        raise DecoderError("Inference requires attention_dropout=0")
    _number(config.get("rms_norm_eps"), "rms_norm_eps", positive=True)
    _number(config.get("routed_scaling_factor"), "routed_scaling_factor", positive=True)
    rope = config.get("rope_parameters")
    if not isinstance(rope, dict) or rope.get("rope_type") != "default":
        raise DecoderError("Only explicit default rope_parameters are implemented")
    _number(rope.get("rope_theta"), "rope_theta", positive=True)
    groups, selected_groups = config.get("n_group"), config.get("topk_group")
    experts = config["n_routed_experts"]
    if (type(groups) is not int or not 1 <= groups <= experts or experts % groups
            or experts // groups < 2 or type(selected_groups) is not int
            or not 1 <= selected_groups <= groups):
        raise DecoderError("Grouped routing requires integral groups with at least two experts each")
    if config["num_experts_per_tok"] > selected_groups * (experts // groups):
        raise DecoderError("num_experts_per_tok exceeds the selected groups' expert count")


class StreamingDecoder:
    """One-token autoregressive decoder over an explicit weight protocol.

    ``weights.vector(name)``, ``weights.embedding(token)`` and
    ``weights.linear(name, values)`` return finite sized vectors; names are
    canonical checkpoint names. ``linear`` optionally replaces that method.
    Adapters retain no decoded weights and own native/GPU gate enforcement.
    Share ``plan.allocator(include_cache=False)`` with an adapter to account
    all cache and temporary payload reservations in one ledger.

    Caller-retained logits are outside the decoder lifetime: consume each
    result before the next step. A failed step does not advance position;
    retrying overwrites every uncommitted slot before reading it.
    """

    def __init__(self, config, weights, plan, *, linear=None, ledger=None,
                 attention_topk=None, model_id=None, revision=None, progress=None):
        if not isinstance(plan, ResidencyPlan):
            raise DecoderError("A validated ResidencyPlan is required before decoder allocation")
        try:
            rebuilt = build_plan(config, plan.settings, model_id=model_id, revision=revision)
        except ResidencyError as exc:
            raise DecoderError(str(exc)) from exc
        if rebuilt != plan:
            raise DecoderError("Decoder configuration does not match its residency plan")
        _validate_numeric_config(config)
        project = linear if linear is not None else getattr(weights, "linear", None)
        if (not callable(project) or not callable(getattr(weights, "vector", None))
                or not callable(getattr(weights, "embedding", None))):
            raise DecoderError("Weights must implement vector, embedding and linear")
        if hasattr(weights, "config") and weights.config != config:
            raise DecoderError("Weight source configuration differs from the decoder configuration")
        if attention_topk is not None and not callable(attention_topk):
            raise DecoderError("attention_topk must be callable")
        if progress is not None and not callable(progress):
            raise DecoderError("progress must be a callable stage observer")
        if ledger is not None and not isinstance(ledger, ReservationLedger):
            raise DecoderError("ledger must be ReservationLedger")
        self.ledger = ledger if ledger is not None else plan.allocator(include_cache=False)
        state = self.ledger.snapshot()
        if (state["cpu"]["budget_bytes"] > plan.settings.ram_budget_bytes
                or state["cuda"]["budget_bytes"] > (plan.settings.vram_budget_bytes
                                                     if plan.settings.device == "hybrid" else 0)
                or state["cpu"]["baseline_bytes"] < plan.settings.runtime_headroom_bytes
                or (plan.settings.device == "hybrid"
                    and state["cuda"]["baseline_bytes"] < plan.settings.gpu_headroom_bytes)):
            raise DecoderError("Ledger budgets/headroom do not enforce the supplied plan")
        self.config = deepcopy(config)
        self.plan = plan
        self.weights = weights
        self._linear = project
        self._attention_topk = attention_topk or _topk
        self._progress = progress
        self._position = 0
        self._closed = False
        self.last_trace = None
        self._cache_lease = None
        self._cache, self._index_cache = [], {}
        findings = Findings()
        profile = _profile(config, findings, model_id=model_id, revision=revision)
        self._mlps = tuple(profile["mlps"])
        self._indexers = tuple(profile["indexers"])
        self._width = config["kv_lora_rank"] + config["qk_rope_head_dim"]
        self._scratch_bytes = sum(value for key, value in plan.cpu_components
                                  if key in ("token_activations", "attention_and_selection_scratch"))
        self._report_progress("decoder_cache_allocation", layer_count=plan.layer_count)
        self._cache_lease = self.ledger.reserve(plan.cache_bytes, label="decoder compressed cache")
        try:
            # array multiplication allocates a single compact payload, unlike
            # constructing a context-size Python list of scalar zeros.
            self._cache = [array("f", [0.0]) * (plan.settings.context_tokens * self._width)
                           for _ in range(plan.layer_count)]
            self._index_cache = {
                layer: array("f", [0.0]) * (plan.settings.context_tokens * config["index_head_dim"])
                for layer, kind in enumerate(self._indexers) if kind == "full"
            }
            if self.cache_bytes != plan.cache_bytes:
                raise DecoderError("Allocated cache payload differs from the residency estimate")
            self._report_progress("decoder_ready", layer_count=plan.layer_count)
        except BaseException:
            self.close()
            raise

    @property
    def position(self):
        return self._position

    @property
    def cache_bytes(self):
        return sum(len(values) * values.itemsize for values in self._cache) + sum(
            len(values) * values.itemsize for values in self._index_cache.values())

    @property
    def cache_used_bytes(self):
        return self.cache_bytes // self.plan.settings.context_tokens * self._position

    def _require_open(self):
        if self._closed:
            raise DecoderError("Decoder is closed")

    def _token(self, token):
        if type(token) is not int or not 0 <= token < self.config["vocab_size"]:
            raise DecoderError("Token ID must be an integer in the configured vocabulary")
        return token

    def _report_progress(self, stage, **fields):
        """Optional diagnostics only; never includes token IDs, text or weights."""
        if self._progress is not None:
            self._progress(stage, **fields)

    def _vector(self, name, length):
        return _compact(self.weights.vector(name), length, name)

    def _project(self, name, values):
        shape = known_shape(name, self.config)
        if shape is None or len(shape) != 2 or len(values) != shape[1]:
            raise DecoderError(f"Projection shape mismatch for {name}")
        if any(not math.isfinite(value) for value in values):
            raise DecoderError(f"Projection input contains nonfinite values for {name}")
        self._report_progress("projection_start", position=self._position, tensor_name=name, rows=shape[0], cols=shape[1])
        result = _compact(self._linear(name, values), shape[0], name)
        self._report_progress("projection_complete", position=self._position, tensor_name=name, rows=shape[0], cols=shape[1])
        return result

    def _norm(self, name, values, epsilon):
        gain = self._vector(name, len(values))
        inverse = 1.0 / math.sqrt(math.fsum(value * value for value in values) / len(values) + epsilon)
        return array("f", (value * inverse * weight for value, weight in zip(values, gain)))

    def _rope(self, values):
        width = self.config["qk_rope_head_dim"]
        theta = self.config["rope_parameters"]["rope_theta"]
        output = array("f", [0.0]) * width
        for pair in range(width // 2):
            angle = self._position * theta ** (-2.0 * pair / width)
            cosine, sine = math.cos(angle), math.sin(angle)
            left, right = values[2 * pair], values[2 * pair + 1]
            output[pair] = left * cosine - right * sine
            output[pair + width // 2] = right * cosine + left * sine
        return output

    def _index(self, layer, hidden, query_residual):
        config = self.config
        prefix = f"model.layers.{layer}.self_attn.indexer."
        heads, width, rope = config["index_n_heads"], config["index_head_dim"], config["qk_rope_head_dim"]
        query = self._project(prefix + "wq_b.weight", query_residual)
        key = self._project(prefix + "wk.weight", hidden)
        gain = self._vector(prefix + "k_norm.weight", width)
        bias = self._vector(prefix + "k_norm.bias", width)
        mean = math.fsum(key) / width
        inverse = 1.0 / math.sqrt(math.fsum((value - mean) ** 2 for value in key) / width + 1e-6)
        key = array("f", ((value - mean) * inverse * g + b for value, g, b in zip(key, gain, bias)))
        key[:rope] = self._rope(key[:rope])
        cache = self._index_cache[layer]
        start = self._position * width
        cache[start:start + width] = key
        head_weights = self._project(prefix + "weights_proj.weight", hidden)
        scores = array("f", [0.0]) * (self._position + 1)
        for head in range(heads):
            offset = head * width
            query[offset:offset + rope] = self._rope(query[offset:offset + rope])
            multiplier = head_weights[head] / math.sqrt(heads)
            for token in range(self._position + 1):
                dot = math.fsum(query[offset + d] * cache[token * width + d] for d in range(width))
                scores[token] += multiplier * max(0.0, dot / math.sqrt(width))
        count = min(config["index_topk"], len(scores))
        selected = self._attention_topk(scores, count)
        if (not isinstance(selected, (list, tuple, array)) or len(selected) != count
                or any(type(i) is not int or not 0 <= i <= self._position for i in selected)
                or len(set(selected)) != count):
            raise DecoderError("Attention top-k returned invalid token indices")
        return array("q", selected)

    def _attention(self, layer, hidden, previous_indices):
        config = self.config
        prefix = f"model.layers.{layer}.self_attn."
        heads = config["num_attention_heads"]
        nope, rope, value = config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"]
        rank, query_width = config["kv_lora_rank"], nope + rope
        residual = self._norm(prefix + "q_a_layernorm.weight",
                              self._project(prefix + "q_a_proj.weight", hidden), 1e-6)
        query = self._project(prefix + "q_b_proj.weight", residual)
        compressed = self._project(prefix + "kv_a_proj_with_mqa.weight", hidden)
        latent = self._norm(prefix + "kv_a_layernorm.weight", compressed[:rank], 1e-6)
        rotated_key = self._rope(compressed[rank:])
        slot = self._position * self._width
        cache = self._cache[layer]
        cache[slot:slot + rank] = latent
        cache[slot + rank:slot + self._width] = rotated_key
        for head in range(heads):
            start = head * query_width + nope
            query[start:start + rope] = self._rope(query[start:start + rope])
        selected = (self._index(layer, hidden, residual) if self._indexers[layer] == "full"
                    else previous_indices)
        if selected is None:
            raise DecoderError("Shared indexer has no preceding full-indexer selection")
        # Online softmax: one expanded token at a time. Reconstructing K/V
        # adds projection work but keeps cache storage proportional to rank.
        maximum = array("d", [float("-inf")]) * heads
        denominator = array("d", [0.0]) * heads
        output = array("d", [0.0]) * (heads * value)
        scale = 1.0 / math.sqrt(query_width)
        for token in selected:
            slot = token * self._width
            expanded = self._project(prefix + "kv_b_proj.weight", cache[slot:slot + rank])
            for head in range(heads):
                q, kv = head * query_width, head * (nope + value)
                score = (math.fsum(query[q + d] * expanded[kv + d] for d in range(nope))
                         + math.fsum(query[q + nope + d] * cache[slot + rank + d] for d in range(rope))) * scale
                new_max = max(maximum[head], score)
                previous_scale = math.exp(maximum[head] - new_max)
                probability = math.exp(score - new_max)
                denominator[head] = denominator[head] * previous_scale + probability
                for d in range(value):
                    at = head * value + d
                    output[at] = output[at] * previous_scale + probability * expanded[kv + nope + d]
                maximum[head] = new_max
        normalized = array("f", (output[head * value + d] / denominator[head]
                                  for head in range(heads) for d in range(value)))
        return self._project(prefix + "o_proj.weight", normalized), selected

    def _ffn(self, prefix, hidden):
        gate = self._project(prefix + "gate_proj.weight", hidden)
        up = self._project(prefix + "up_proj.weight", hidden)
        activated = array("f", (g * _sigmoid(g) * u for g, u in zip(gate, up)))
        return self._project(prefix + "down_proj.weight", activated)

    def _moe(self, layer, hidden):
        config = self.config
        prefix = f"model.layers.{layer}.mlp."
        logits = self._project(prefix + "gate.weight", hidden)
        scores = array("f", (_sigmoid(value) for value in logits))
        bias = self._vector(prefix + "gate.e_score_correction_bias", len(scores))
        choice = array("f", (value + correction for value, correction in zip(scores, bias)))
        width = len(scores) // config["n_group"]
        group_scores = [math.fsum(heapq.nlargest(2, choice[start:start + width]))
                        for start in range(0, len(scores), width)]
        groups = set(_topk(group_scores, config["topk_group"]))
        for expert in range(len(choice)):
            if expert // width not in groups:
                choice[expert] = float("-inf")
        selected = _topk(choice, config["num_experts_per_tok"])
        denominator = math.fsum(scores[expert] for expert in selected) + 1e-20 if config["norm_topk_prob"] else 1.0
        # Official eager experts accumulate in expert ID order. Each selected
        # expert is loaded and discarded sequentially; there is no expert bank.
        output = array("f", [0.0]) * config["hidden_size"]
        for expert in sorted(selected):
            contribution = self._ffn(prefix + f"experts.{expert}.", hidden)
            weight = scores[expert] / denominator * config["routed_scaling_factor"]
            for index, value in enumerate(contribution):
                output[index] += weight * value
        shared = self._ffn(prefix + "shared_experts.", hidden)
        return array("f", (base + delta for base, delta in zip(output, shared)))

    def step(self, token):
        """Process one token and return FP32 next-token logits."""
        self._require_open()
        self._token(token)
        if self._position >= self.plan.settings.context_tokens:
            raise DecoderError("Decoder context is full; reset before a new sequence")
        self._report_progress("token_start", position=self._position)
        with self.ledger.reserve(self._scratch_bytes, label="decoder token scratch"):
            self._report_progress("embedding", position=self._position)
            hidden = _compact(self.weights.embedding(token), self.config["hidden_size"], "embedding")
            previous = None
            for layer in range(self.plan.layer_count):
                self._report_progress("layer_start", position=self._position, layer_index=layer, layer_count=self.plan.layer_count)
                prefix = f"model.layers.{layer}."
                normalized = self._norm(prefix + "input_layernorm.weight", hidden,
                                        self.config["rms_norm_eps"])
                attention, previous = self._attention(layer, normalized, previous)
                hidden = array("f", (base + delta for base, delta in zip(hidden, attention)))
                normalized = self._norm(prefix + "post_attention_layernorm.weight", hidden,
                                        self.config["rms_norm_eps"])
                mlp = (self._ffn(prefix + "mlp.", normalized) if self._mlps[layer] == "dense"
                       else self._moe(layer, normalized))
                hidden = array("f", (base + delta for base, delta in zip(hidden, mlp)))
                self._report_progress("layer_complete", position=self._position, layer_index=layer, layer_count=self.plan.layer_count)
            self._report_progress("output_head", position=self._position)
            normalized = self._norm("model.norm.weight", hidden, self.config["rms_norm_eps"])
            logits = self._project("lm_head.weight", normalized)
            self.last_trace = {"position": self._position, "token": token,
                               "backbone_layers": self.plan.layer_count,
                               "last_selected_indices": list(previous)}
            self._report_progress("token_complete", position=self._position + 1)
            self._position += 1
            return logits

    def generate(self, prompt_ids, max_new_tokens=None, eos_token_ids=None):
        """Generate greedy token IDs from a nonempty prompt and an empty cache."""
        self._require_open()
        if self._position:
            raise DecoderError("generate requires an empty cache; call reset first")
        if not isinstance(prompt_ids, (list, tuple, array)) or not prompt_ids:
            raise DecoderError("prompt_ids must be a nonempty sized token sequence")
        generated_count = self.plan.settings.max_new_tokens if max_new_tokens is None else max_new_tokens
        try:
            self.plan.validate_prompt(len(prompt_ids), generated_count)
        except ResidencyError as exc:
            raise DecoderError(str(exc)) from exc
        for token in prompt_ids:
            self._token(token)
        eos = self.config.get("eos_token_id") if eos_token_ids is None else eos_token_ids
        if eos is None:
            eos = []
        elif type(eos) is int:
            eos = [eos]
        if not isinstance(eos, (list, tuple, set)) or len(eos) > 256:
            raise DecoderError("EOS IDs must be at most 256 configured token IDs")
        stop = {self._token(token) for token in eos}
        for index, token in enumerate(prompt_ids):
            self._report_progress("prefill", phase="prefill", token_index=index, token_count=len(prompt_ids),
                                  prompt_tokens=len(prompt_ids), requested_new_tokens=generated_count, generated_tokens=0)
            logits = self.step(token)
        generated = []
        for index in range(generated_count):
            token = max(range(len(logits)), key=lambda item: (logits[item], -item))
            generated.append(token)
            self._report_progress("generated_token", generated_tokens=len(generated))
            if token in stop or index + 1 == generated_count:
                break
            self._report_progress("decode", phase="decode", token_index=index, token_count=max(0, generated_count - 1),
                                  generated_tokens=len(generated))
            logits = self.step(token)
        self._report_progress("generation_complete", generated_tokens=len(generated), prompt_tokens=len(prompt_ids))
        return generated

    def reset(self):
        self._require_open()
        self._position = 0
        self.last_trace = None

    def close(self):
        self._cache.clear()
        self._index_cache.clear()
        self.last_trace = None
        self._position = 0
        self._closed = True
        if self._cache_lease is not None and self._cache_lease.active:
            self._cache_lease.release()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
