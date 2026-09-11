"""Bounded comparisons for synthetic native-vs-official miniature steps.

Order inside top-k sets is irrelevant; membership and ID-associated expert
weights are not. Equal-score differences are never silently waived.
"""

import math

HIDDEN_NAMES = ("embedding", *(f"layer.{layer}.{name}" for layer in range(2)
                 for name in ("input_norm", "attention_output", "post_attention",
                              "post_attention_norm", "output")), "final_norm")


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"Expected a bounded vector of {size} values")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("Comparison values must be finite numbers")
    return value


def _indices(value, count, upper):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError("Invalid bounded selection length")
    if any(type(x) is not int or not 0 <= x < upper for x in value) or len(set(value)) != count:
        raise ValueError("Selections must contain distinct valid integers")
    return value


def compare_step(native_logits, native_trace, official_step, *, atol=2e-5, rtol=3e-4):
    for tolerance in (atol, rtol):
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Tolerances must be finite and nonnegative")
    position = native_trace.get("position")
    if type(position) is not int or not 0 <= position < 128:
        raise ValueError("Synthetic trace position must be 0-127")
    failures = []

    def compare(name, a, b, width):
        _vector(a, width)
        _vector(b, width)
        errors = [abs(x-y) for x, y in zip(a, b)]
        passed = all(error <= atol + rtol * abs(y) for error, y in zip(errors, b))
        if not passed and len(failures) < 8:
            failures.append(name)
        return {"passed": passed, "max_absolute_error": max(errors), "values_compared": width}

    logits = compare("logits", native_logits, official_step["logits"], 32)
    left, right = native_trace["hidden_states"], official_step["hidden_states"]
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(HIDDEN_NAMES) or set(right) != set(HIDDEN_NAMES):
        raise ValueError("Exactly twelve named hidden states are required")
    nodes = {name: compare(name, left[name], right[name], 16) for name in HIDDEN_NAMES}
    count = min(4, position + 1)
    native_selected = _indices(native_trace["selected_indices"], count, position+1)
    official_selected = _indices(official_step["selected_indices"], count, position+1)
    selection_passed = set(native_selected) == set(official_selected)
    if not selection_passed and len(failures) < 8:
        failures.append("attention_selection")
    native_experts = _indices(native_trace["routed_experts"], 2, 4)
    official_experts = _indices(official_step["routed_experts"], 2, 4)
    native_weights = _vector(native_trace["router_weights"], 2)
    official_weights = _vector(official_step["router_weights"], 2)
    ids_match = set(native_experts) == set(official_experts)
    if ids_match:
        weights_a, weights_b = dict(zip(native_experts, native_weights)), dict(zip(official_experts, official_weights))
        weight_comparison = compare("router_weights", [weights_a[i] for i in sorted(weights_a)],
                                    [weights_b[i] for i in sorted(weights_b)], 2)
    else:
        weight_comparison = {"passed": False, "max_absolute_error": None}
        if len(failures) < 8:
            failures.append("expert_selection")
    hidden_passed = all(value["passed"] for value in nodes.values())
    return {"passed": logits["passed"] and hidden_passed and selection_passed and weight_comparison["passed"],
            "position": position, "logits": logits,
            "hidden_states": {"passed": hidden_passed, "nodes": nodes,
                              "max_absolute_error": max(v["max_absolute_error"] for v in nodes.values())},
            "selection": {"passed": selection_passed, "native": list(native_selected), "official": list(official_selected)},
            "routing": {"passed": ids_match and weight_comparison["passed"], "expert_ids_match": ids_match,
                        "max_weight_error": weight_comparison["max_absolute_error"]},
            "first_failures": failures, "atol": atol, "rtol": rtol}
