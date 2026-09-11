from copy import deepcopy
import math
import unittest

from glm_local.parity_compare import compare_step, HIDDEN_NAMES


def records():
    trace = {"position": 4, "hidden_states": {n: [0.0]*16 for n in HIDDEN_NAMES},
             "selected_indices": [0, 1, 2, 3], "routed_experts": [1, 3], "router_weights": [1.1, 1.4]}
    official = deepcopy(trace)
    official["logits"] = [0.0]*32
    return [0.0]*32, trace, official


class ParityCompareTests(unittest.TestCase):
    def test_selection_order_and_id_aligned_weights(self):
        logits, trace, official = records()
        official["selected_indices"] = [3, 1, 0, 2]
        official["routed_experts"] = [3, 1]
        official["router_weights"] = [1.4, 1.1]
        self.assertTrue(compare_step(logits, trace, official)["passed"])
        official["router_weights"] = [1.1, 1.4]
        self.assertFalse(compare_step(logits, trace, official)["routing"]["passed"])

    def test_different_tied_selection_never_waived(self):
        logits, trace, official = records()
        official["selected_indices"] = [1, 2, 3, 4]
        result = compare_step(logits, trace, official)
        self.assertFalse(result["passed"])
        self.assertIn("attention_selection", result["first_failures"])

    def test_per_element_errors_and_hidden_nodes(self):
        logits, trace, official = records()
        logits[0] = official["logits"][0] = 1e8
        logits[1] = 1e-3
        trace["hidden_states"]["layer.1.output"][2] = 0.1
        result = compare_step(logits, trace, official)
        self.assertFalse(result["logits"]["passed"])
        self.assertFalse(result["hidden_states"]["passed"])
        self.assertEqual(result["hidden_states"]["nodes"]["layer.1.output"]["max_absolute_error"], 0.1)

    def test_missing_states_bad_shapes_and_nonfinite_rejected(self):
        for change in (lambda o: o["hidden_states"].pop("embedding"),
                       lambda o: o.update(logits=[0.0]*31),
                       lambda o: o.update(router_weights=[math.nan, 1.0]),
                       lambda o: o.update(selected_indices=[0, 0, 1, 2]),
                       lambda o: o.update(selected_indices=[0, 1, 2, 5])):
            logits, trace, official = records()
            change(official)
            with self.assertRaises(ValueError):
                compare_step(logits, trace, official)

    def test_tolerance_and_position_validation(self):
        for tolerance in (-1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                compare_step(*records(), atol=tolerance)
        logits, trace, official = records()
        trace["position"] = 128
        with self.assertRaises(ValueError):
            compare_step(logits, trace, official)

    def test_bounded_diagnostic_failures(self):
        logits, trace, official = records()
        for name in HIDDEN_NAMES:
            trace["hidden_states"][name] = [1.0]*16
        self.assertEqual(len(compare_step(logits, trace, official)["first_failures"]), 8)
