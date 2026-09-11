"""Incremental inference for the fixed, invented miniature in ``mini_spec``.

This is a synthetic numerical fixture, not a real-checkpoint runtime. Scalar
operations use Python float64; the injected linear implementation determines
projection precision (the native adapters use FP32). This is not an emulation
of every dtype conversion in the official Transformers implementation.
"""

from array import array
import math

from .mini_spec import SPEC, matrix_shapes


def _rms(values, gain, epsilon):
    inverse = 1.0 / math.sqrt(math.fsum(x * x for x in values) / len(values) + epsilon)
    return [x * inverse * g for x, g in zip(values, gain)]


def _layer_norm(values, gain, bias, epsilon):
    mean = math.fsum(values) / len(values)
    variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
    inverse = 1.0 / math.sqrt(variance + epsilon)
    return [(x - mean) * inverse * g + b for x, g, b in zip(values, gain, bias)]


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _rope(values, position):
    """Rotate adjacent input pairs and concatenate rotated evens, then odds."""
    even, odd = [], []
    for pair in range(len(values) // 2):
        angle = position * SPEC["rope_theta"] ** (-2.0 * pair / len(values))
        cosine, sine = math.cos(angle), math.sin(angle)
        left, right = values[2 * pair], values[2 * pair + 1]
        even.append(left * cosine - right * sine)
        odd.append(right * cosine + left * sine)
    return even + odd


def _topk(scores, count):
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:count]


class MiniDecoder:
    """Bounded single-token decoder with expanded K/V and a full-indexer cache.

    ``weights`` supplies ``vector(name)`` and ``embedding(token)``. ``linear``
    takes a canonical matrix name and input vector. No configuration, arbitrary
    dimensions, checkpoint path, tokenizer, or training API is accepted.

    ``cache_bytes`` counts allocated array('d') payload only, excluding Python
    object headers, transient arithmetic, weights and the bounded trace list.
    ``cache_used_bytes`` is the logical occupied prefix of those same arrays.
    ``capture_states=True`` adds independent copies of selected 16-value hidden
    states to each trace record for numerical validation of this fixed fixture.
    """

    def __init__(self, weights, linear, *, capture_states=False, attention_topk=None):
        if type(capture_states) is not bool:
            raise ValueError("Miniature capture_states must be a bool")
        if not callable(linear):
            raise ValueError("Miniature linear adapter must be callable")
        if attention_topk is not None and not callable(attention_topk):
            raise ValueError("Optional attention top-k must be callable")
        self._attention_topk = attention_topk or _topk
        self.weights = weights
        self.linear = linear
        self._capture_states = capture_states
        self._shapes = matrix_shapes()
        self._position = 0
        self.trace = []
        self._keys = [array("d", [0.0]) * (128 * 2 * 8) for _ in range(2)]
        self._values = [array("d", [0.0]) * (128 * 2 * 4) for _ in range(2)]
        self._index_keys = array("d", [0.0]) * (128 * 8)

    @property
    def position(self):
        return self._position

    @property
    def cache_bytes(self):
        arrays = self._keys + self._values + [self._index_keys]
        return sum(len(values) * values.itemsize for values in arrays)

    @property
    def cache_used_bytes(self):
        return self.cache_bytes // SPEC["max_context"] * self._position

    def reset(self):
        # Slots outside the live prefix are never read. A reset need not zero
        # backing storage: each new token overwrites all its own cache entries.
        self._position = 0
        self.trace.clear()

    def _project(self, name, values):
        rows, columns = self._shapes[name]
        if len(values) != columns:
            raise ValueError(f"Wrong miniature input size for {name}")
        result = list(self.linear(name, values))
        if len(result) != rows or any(not math.isfinite(value) for value in result):
            raise ValueError(f"Invalid miniature linear output for {name}")
        return result

    def _norm(self, name, values, epsilon):
        gain = self.weights.vector(name)
        if len(gain) != len(values):
            raise ValueError(f"Wrong miniature normalization size for {name}")
        return _rms(values, gain, epsilon)

    def _index(self, hidden, query_residual):
        query = self._project("layer.0.index_q", query_residual)
        key = self._project("layer.0.index_k", hidden)
        key = _layer_norm(key, self.weights.vector("layer.0.index_norm_weight"),
                          self.weights.vector("layer.0.index_norm_bias"), SPEC["latent_eps"])
        key = _rope(key[:4], self._position) + key[4:]
        start = self._position * 8
        self._index_keys[start:start + 8] = array("d", key)
        head_weights = self._project("layer.0.index_weight", hidden)
        scores = [0.0] * (self._position + 1)
        for head in range(2):
            q = query[head * 8:(head + 1) * 8]
            q = _rope(q[:4], self._position) + q[4:]
            multiplier = head_weights[head] / math.sqrt(2.0)
            for token in range(self._position + 1):
                dot = math.fsum(q[d] * self._index_keys[token * 8 + d] for d in range(8))
                scores[token] += multiplier * max(0.0, dot / math.sqrt(8.0))
        count = min(SPEC["index_topk"], self._position + 1)
        indices = self._attention_topk(scores, count)
        if (not isinstance(indices, (list, tuple)) or len(indices) != count
                or any(type(i) is not int or not 0 <= i <= self._position for i in indices)
                or len(set(indices)) != count):
            raise ValueError("Attention top-k returned invalid indices")
        return list(indices)

    def _attention(self, layer, hidden, previous_indices):
        prefix = f"layer.{layer}."
        query_residual = self._norm(prefix + "q_norm", self._project(prefix + "q_a", hidden),
                                    SPEC["latent_eps"])
        queries = self._project(prefix + "q_b", query_residual)
        compressed = self._project(prefix + "kv_a", hidden)
        latent = self._norm(prefix + "kv_norm", compressed[:4], SPEC["latent_eps"])
        key_rotated = _rope(compressed[4:], self._position)
        expanded = self._project(prefix + "kv_b", latent)
        rotated_queries = []
        for head in range(2):
            q = queries[head * 8:(head + 1) * 8]
            rotated_queries.append(q[:4] + _rope(q[4:], self._position))
            key = expanded[head * 8:head * 8 + 4] + key_rotated
            value = expanded[head * 8 + 4:(head + 1) * 8]
            key_start = (self._position * 2 + head) * 8
            value_start = (self._position * 2 + head) * 4
            self._keys[layer][key_start:key_start + 8] = array("d", key)
            self._values[layer][value_start:value_start + 4] = array("d", value)

        selected = self._index(hidden, query_residual) if layer == 0 else list(previous_indices)
        output, head_probabilities = [], []
        for head, q in enumerate(rotated_queries):
            scores = [math.fsum(q[d] * self._keys[layer][(token * 2 + head) * 8 + d]
                                for d in range(8)) / math.sqrt(8.0) for token in selected]
            maximum = max(scores)
            exponentials = [math.exp(score - maximum) for score in scores]
            denominator = math.fsum(exponentials)
            probabilities = [value / denominator for value in exponentials]
            head_probabilities.append(probabilities)
            output.extend(math.fsum(probability * self._values[layer][(token * 2 + head) * 4 + d]
                                    for token, probability in zip(selected, probabilities))
                          for d in range(4))
        return self._project(prefix + "o", output), selected, head_probabilities

    def _mlp(self, prefix, hidden):
        gate = self._project(prefix + "gate", hidden)
        up = self._project(prefix + "up", hidden)
        activated = [g * _sigmoid(g) * u for g, u in zip(gate, up)]
        return self._project(prefix + "down", activated)

    def _moe(self, hidden):
        logits = self._project("layer.1.router", hidden)
        scores = [_sigmoid(value) for value in logits]
        correction = self.weights.vector("layer.1.router_bias")
        selected = _topk([score + bias for score, bias in zip(scores, correction)], 2)
        denominator = math.fsum(scores[index] for index in selected) + 1e-20
        weights = [scores[index] / denominator * SPEC["routed_scale"] for index in selected]
        output = self._mlp("layer.1.shared.", hidden)
        for expert, weight in zip(selected, weights):
            contribution = self._mlp(f"layer.1.expert.{expert}.", hidden)
            output = [base + weight * delta for base, delta in zip(output, contribution)]
        return output, selected, weights

    def step(self, token):
        """Return 32 logits for one invented token, committing one cache slot."""
        if type(token) is not int or not 0 <= token < SPEC["vocab"]:
            raise ValueError("Synthetic token ID must be an integer in [0, 31]")
        if self._position >= SPEC["max_context"]:
            raise ValueError("Synthetic context limit is 128 tokens; call reset()")
        hidden = list(self.weights.embedding(token))
        if len(hidden) != SPEC["hidden"] or any(not math.isfinite(value) for value in hidden):
            raise ValueError("Invalid synthetic embedding")
        record = {"position": self._position, "token": token, "layers": []}
        states = {"embedding": list(hidden)} if self._capture_states else None
        previous_indices = None
        for layer in range(2):
            prefix = f"layer.{layer}."
            normalized = self._norm(prefix + "in_norm", hidden, SPEC["norm_eps"])
            if states is not None:
                states[prefix + "input_norm"] = list(normalized)
            attention, selected, probabilities = self._attention(layer, normalized, previous_indices)
            if states is not None:
                states[prefix + "attention_output"] = list(attention)
            hidden = [base + delta for base, delta in zip(hidden, attention)]
            if states is not None:
                states[prefix + "post_attention"] = list(hidden)
            normalized = self._norm(prefix + "post_norm", hidden, SPEC["norm_eps"])
            if states is not None:
                states[prefix + "post_attention_norm"] = list(normalized)
            layer_record = {"layer": layer, "selected_indices": selected,
                            "attention_probabilities": probabilities}
            if layer == 0:
                feed_forward = self._mlp("layer.0.mlp.", normalized)
                record["selected_indices"] = selected
            else:
                feed_forward, experts, weights = self._moe(normalized)
                layer_record.update(routed_experts=experts, router_weights=weights)
                record.update(routed_experts=experts, router_weights=weights)
            hidden = [base + delta for base, delta in zip(hidden, feed_forward)]
            if states is not None:
                states[prefix + "output"] = list(hidden)
            record["layers"].append(layer_record)
            previous_indices = selected
        normalized = self._norm("final_norm", hidden, SPEC["norm_eps"])
        if states is not None:
            states["final_norm"] = list(normalized)
        logits = self._project("lm_head", normalized)
        if states is not None:
            record["hidden_states"] = states
        self.trace.append(record)
        self._position += 1
        return logits
