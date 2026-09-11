"""Miniature runner contracts with no DLL, GPU, telemetry, or subprocess access.

The reference stand-in recomputes each prefix and independently chooses its own
greedy trajectory. Native numerical parity belongs to the separate live probe.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from glm_local import mini_run
from glm_local.process_metrics import ProcessSnapshot
from glm_local.winjob import InstalledLimits


def settings():
    return {"model_directory": "unused", "ram_budget_bytes": 32_000_000_000,
            "cpu_job_percent": 70, "gpu_average_target": 0.6,
            "gpu_window_seconds": 10, "gpu_index": 0, "disk_reserve_bytes": 0}


def sampler(*, peak=2000):
    count = 0

    def sample():
        nonlocal count
        count += 1
        return ProcessSnapshot(
            process_id=123, monotonic_seconds=float(count),
            working_set_bytes=min(1000, peak), peak_working_set_bytes=peak,
            private_commit_bytes=1500, peak_private_commit_bytes=2500,
            process_cpu_seconds=count / 10, logical_cpu_count=8,
        )

    return sample


def trace_for(position):
    return {"selected_indices": list(range(min(position + 1, 4))),
            "routed_experts": [0, 1], "router_weights": [1.25, 1.25]}


def logits_for(tokens, *, advance=1, amplitude=1.0):
    values = [0.0] * 32
    values[(tokens[-1] + advance) % 32] = amplitude
    return values


class FakeDecoder:
    def __init__(self, linear, *, amplitude=1.0):
        self.linear, self.amplitude = linear, amplitude
        self.reset_calls = 0
        self.tokens, self.trace = [], []
        self.cache_bytes = 128 * 448

    def reset(self):
        self.reset_calls += 1
        self.tokens, self.trace = [], []

    def step(self, token):
        self.tokens.append(token)
        self.trace.append(trace_for(len(self.tokens) - 1))
        self.linear.cpu_calls += 2
        self.linear.gpu_calls += 1
        return logits_for(self.tokens, amplitude=self.amplitude)

    @property
    def cache_used_bytes(self):
        return len(self.tokens) * 448


class FakeReference:
    def __init__(self, *, advance=1, amplitude=1.0, wrong_selection=False):
        self.advance, self.amplitude = advance, amplitude
        self.wrong_selection = wrong_selection
        self.calls = []

    def forward(self, tokens):
        self.calls.append(list(tokens))
        traces = [trace_for(position) for position in range(len(tokens))]
        if self.wrong_selection:
            traces[-1]["routed_experts"] = [1, 2]
        return {"logits": [logits_for(tokens[:position + 1], advance=self.advance,
                                      amplitude=self.amplitude)
                           for position in range(len(tokens))], "traces": traces}


class MiniRunValidationTests(unittest.TestCase):
    def test_invalid_types_and_ranges_fail_before_work(self):
        defaults = dict(backend="cpu", lengths=[8, 32, 64], generate=4, seed=7)
        invalid = {
            "backend": ("", "auto", "gpu", None),
            "lengths": ([], [1, 2, 3, 4, 5], [0], [-1], [125], [True], [8.0],
                        "8", None, [8, 8], [32, 8]),
            "generate": (0, 9, -1, True, 4.0, "4"),
            "seed": (-1, 2**32, True, 7.0, "7"),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    mini_run.validate_mini(**(defaults | {name: value}))

    def test_context_bound_counts_prompt_and_every_generated_step(self):
        mini_run.validate_mini("cpu", [120], 8, 0)
        mini_run.validate_mini("cpu", (127,), 1, 2**32 - 1)
        with self.assertRaisesRegex(ValueError, "128-token"):
            mini_run.validate_mini("cpu", [121], 8, 0)

    def test_hybrid_budget_counts_all_cases_including_generation(self):
        self.assertEqual(mini_run.MAX_OPERATIONS, 256)
        # 56 + 80 + 96 input tokens plus 3 * 8 generated steps = 256 heads.
        mini_run.validate_mini("hybrid", [56, 80, 96], 8, 0)
        with self.assertRaisesRegex(ValueError, "256 GPU"):
            mini_run.validate_mini("hybrid", [56, 80, 97], 8, 0)
        mini_run.validate_mini("cpu", [56, 80, 97], 8, 0)


class NativeMiniLinearTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.weights, self.cpu, self.gpu, self.gate = Mock(), Mock(), Mock(), Mock()

        def read(name):
            self.events.append(("read", name))
            size = 8 if name == "lm_head" else 4
            return SimpleNamespace(weights=b"\x38" * size, rows=size // 2, cols=2, scale=0.25)

        self.weights.matrix.side_effect = read
        self.cpu.matvec_tile.side_effect = lambda *_: self.events.append("cpu") or [2.0, 3.0]
        self.gpu.matvec_tile.side_effect = lambda *_: self.events.append("gpu") or [4.0] * 4
        self.gate.before_submit.side_effect = lambda: self.events.append("gate")

    def test_hybrid_routes_non_head_to_cpu_and_gates_gpu_head_first(self):
        linear = mini_run.NativeMiniLinear(self.weights, self.cpu, self.gpu, self.gate)
        self.assertEqual(linear("layer.0.q_a", [1.0, 2.0]), [2.0, 3.0])
        self.assertEqual(linear("lm_head", [3.0, 4.0]), [4.0] * 4)
        self.assertEqual(self.events, [("read", "layer.0.q_a"), "cpu",
                                       ("read", "lm_head"), "gate", "gpu"])
        self.cpu.matvec_tile.assert_called_once_with(b"\x38" * 4, 2, 2, [1.0, 2.0], 0.25)
        self.gpu.matvec_tile.assert_called_once_with(b"\x38" * 8, 4, 2, [3.0, 4.0], 0.25)
        self.assertEqual((linear.cpu_calls, linear.gpu_calls, linear.max_matrix_bytes), (1, 1, 8))

    def test_cpu_mode_also_executes_head_without_gpu_or_gate(self):
        linear = mini_run.NativeMiniLinear(self.weights, self.cpu)
        self.assertEqual(linear("lm_head", [1.0, 2.0]), [2.0, 3.0])
        self.assertEqual((linear.cpu_calls, linear.gpu_calls), (1, 0))
        self.gate.before_submit.assert_not_called()
        self.gpu.matvec_tile.assert_not_called()

    def test_gate_failure_prevents_gpu_submission_and_cpu_fallback(self):
        self.gate.before_submit.side_effect = RuntimeError("telemetry unavailable")
        linear = mini_run.NativeMiniLinear(self.weights, self.cpu, self.gpu, self.gate)
        with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
            linear("lm_head", [1.0, 2.0])
        self.cpu.matvec_tile.assert_not_called()
        self.gpu.matvec_tile.assert_not_called()
        self.assertEqual((linear.cpu_calls, linear.gpu_calls), (0, 0))

    def test_backend_failure_is_propagated_without_fallback_or_success_count(self):
        self.gpu.matvec_tile.side_effect = OSError("GPU unavailable")
        linear = mini_run.NativeMiniLinear(self.weights, self.cpu, self.gpu, self.gate)
        with self.assertRaisesRegex(OSError, "GPU unavailable"):
            linear("lm_head", [1.0, 2.0])
        self.cpu.matvec_tile.assert_not_called()
        self.assertEqual((linear.cpu_calls, linear.gpu_calls), (0, 0))

    def test_required_backend_and_gate_are_checked_at_construction(self):
        for cpu, gpu, gate in ((None, None, None), (None, self.gpu, self.gate),
                               (self.cpu, self.gpu, None)):
            with self.subTest(cpu=cpu, gpu=gpu, gate=gate), self.assertRaises(ValueError):
                mini_run.NativeMiniLinear(self.weights, cpu, gpu, gate)
        self.weights.matrix.assert_not_called()


class MiniComparisonTests(unittest.TestCase):
    def test_router_combination_weights_must_match_and_be_finite(self):
        for wrong in (1.5, math.nan, math.inf):
            for side in (0, 1):
                traces = [[trace_for(0)], [trace_for(0)]]
                traces[side][0]["router_weights"][0] = wrong
                result = mini_run.compare_traces(*traces)
                self.assertFalse(result["passed"])
                self.assertEqual(result["first_differences"][0]["field"], "router_weights")
                json.dumps(result, allow_nan=False)

    def test_tolerance_is_per_logit_including_near_zero(self):
        actual, expected = [[0.0] * 32], [[0.0] * 32]
        actual[0][0] = expected[0][0] = 1e9
        actual[0][1] = 1e-4
        result = mini_run.compare_sequences(actual, expected)
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_failures"][0]["token_id"], 1)
        self.assertEqual(result["max_absolute_error"], 1e-4)
        actual[0][1] = 1e-6
        self.assertTrue(mini_run.compare_sequences(actual, expected)["passed"])
        actual[0][2], expected[0][2] = 100.01, 100.0
        self.assertTrue(mini_run.compare_sequences(actual, expected)["passed"])
        actual[0][2] = 100.1
        self.assertFalse(mini_run.compare_sequences(actual, expected)["passed"])

    def test_nonfinite_values_fail_and_diagnostics_are_json_safe(self):
        for value in (math.nan, math.inf, -math.inf):
            for side in ("actual", "expected"):
                with self.subTest(value=value, side=side):
                    actual, expected = [[0.0] * 32], [[0.0] * 32]
                    (actual if side == "actual" else expected)[0][3] = value
                    result = mini_run.compare_sequences(actual, expected)
                    self.assertFalse(result["passed"])
                    self.assertIsNone(result["max_absolute_error"])
                    self.assertIsNone(result["first_failures"][0][side])
                    json.dumps(result, allow_nan=False)

    def test_sequences_need_equal_nonempty_32_wide_shapes(self):
        for actual, expected in (([], []), ([[0.0] * 32], []),
                                 ([[0.0] * 31], [[0.0] * 32]),
                                 ([[0.0] * 32], [[0.0] * 33])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                mini_run.compare_sequences(actual, expected)

    def test_diagnostic_failure_count_is_bounded(self):
        result = mini_run.compare_sequences([[1.0] * 32] * 128, [[0.0] * 32] * 128)
        self.assertFalse(result["passed"])
        self.assertEqual(result["logits_compared"], 4096)
        self.assertEqual(len(result["first_failures"]), 8)

    def test_trace_comparison_checks_both_sparse_indices_and_expert_ids(self):
        for key, value in (("selected_indices", [1]), ("routed_experts", [2, 3])):
            actual, expected = [trace_for(1)], [trace_for(1)]
            actual[0][key] = value
            with self.subTest(key=key):
                result = mini_run.compare_traces(actual, expected)
                self.assertFalse(result["passed"])
                self.assertEqual(result["first_differences"][0]["field"], key)
        with self.assertRaises(ValueError):
            mini_run.compare_traces([], [trace_for(0)])


class MiniCaseTests(unittest.TestCase):
    def setUp(self):
        self.linear = SimpleNamespace(cpu_calls=11, gpu_calls=13)

    def evaluate(self, decoder, reference, prompt=(3, 5), generate=3):
        with patch.object(mini_run, "sample_process", side_effect=sampler()):
            return mini_run.evaluate_case(decoder, reference, prompt, generate, self.linear, 10000)

    def test_prompt_generation_projection_and_cache_counts_match_processed_tokens(self):
        decoder, reference = FakeDecoder(self.linear), FakeReference()
        result = self.evaluate(decoder, reference)
        self.assertTrue(result["passed"])
        self.assertEqual(decoder.reset_calls, 1)
        self.assertEqual(decoder.tokens, [3, 5, 6, 7, 8])
        self.assertEqual(result["prompt_length"], 2)
        self.assertEqual(result["processed_tokens"], 5)
        self.assertEqual(result["generated_token_ids"], [6, 7, 8])
        self.assertEqual(result["reference_generated_token_ids"], [6, 7, 8])
        self.assertEqual((result["cpu_projection_calls"], result["gpu_projection_calls"]), (10, 5))
        self.assertEqual(result["cache_allocated_payload_bytes"], 57344)
        self.assertEqual(result["cache_occupied_payload_bytes"], 5 * 448)
        self.assertEqual(result["logits"]["logits_compared"], 5 * 32)
        self.assertEqual(result["selection_traces"]["positions"], 5)
        self.assertEqual(result["max_token_boundary_peak_rss_bytes"], 2000)
        self.assertEqual(result["peak_worker_private_commit_bytes"], 2500)
        self.assertEqual(reference.calls, [[3, 5, 6, 7, 8], [3, 5], [3, 5, 6], [3, 5, 6, 7]])

    def test_independent_greedy_path_can_fail_even_when_numeric_tolerance_passes(self):
        decoder = FakeDecoder(self.linear, amplitude=1e-6)
        reference = FakeReference(advance=2, amplitude=1e-6)
        result = self.evaluate(decoder, reference)
        self.assertTrue(result["logits"]["passed"])
        self.assertTrue(result["selection_traces"]["passed"])
        self.assertFalse(result["greedy_tokens_match"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["generated_token_ids"], [6, 7, 8])
        self.assertEqual(result["reference_generated_token_ids"], [7, 9, 11])
        self.assertEqual(reference.calls, [[3, 5, 6, 7, 8], [3, 5], [3, 5, 7], [3, 5, 7, 9]])

    def test_selection_mismatch_fails_even_with_identical_logits_and_ids(self):
        result = self.evaluate(FakeDecoder(self.linear), FakeReference(wrong_selection=True))
        self.assertTrue(result["logits"]["passed"])
        self.assertTrue(result["greedy_tokens_match"])
        self.assertFalse(result["selection_traces"]["passed"])
        self.assertFalse(result["passed"])

    def test_observed_peak_over_budget_aborts_before_next_token_or_reference(self):
        decoder, reference = FakeDecoder(self.linear), FakeReference()
        with patch.object(mini_run, "sample_process", side_effect=sampler(peak=10001)), \
                self.assertRaisesRegex(RuntimeError, "RSS exceeded"):
            mini_run.evaluate_case(decoder, reference, [3, 5], 3, self.linear, 10000)
        self.assertEqual(decoder.tokens, [3])
        self.assertEqual(reference.calls, [])

    def test_peak_after_reference_cannot_publish_passing_case(self):
        decoder, reference = FakeDecoder(self.linear), FakeReference()
        low, high = sampler(), sampler(peak=10001)
        def snapshot():
            return high() if reference.calls else low()
        with patch.object(mini_run, "sample_process", side_effect=snapshot), \
                self.assertRaisesRegex(RuntimeError, "reference/worker peak RSS"):
            mini_run.evaluate_case(decoder, reference, [3, 5], 3, self.linear, 10000)
        self.assertTrue(reference.calls)


class MiniLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def launch(self, runner):
        with patch.object(mini_run, "run_local_process", side_effect=runner), \
                patch.object(mini_run, "execute_mini") as inline_work, \
                patch("builtins.print"):
            code = mini_run.launch_mini(self.root, settings(), backend="cpu", lengths=(2,), generate=1)
        inline_work.assert_not_called()
        report = json.loads((self.root / "reports" / "mini-latest.json").read_text(encoding="utf-8"))
        return code, report

    def test_launcher_passes_requested_limits_to_worker_and_records_queried_policy(self):
        policy = InstalledLimits(70.0, 32_000_000_000, True, True, True)

        def runner(command, *, cwd, limits, on_policy, timeout):
            self.assertEqual(command[1:3], ["-m", "glm_local.mini_worker"])
            self.assertEqual(cwd, self.root.resolve())
            self.assertEqual(limits.cpu_percent, 70)
            self.assertEqual(limits.committed_memory_bytes, 32_000_000_000)
            self.assertEqual(timeout, 180)
            request = Path(command[-1])
            data = json.loads(request.read_text(encoding="utf-8"))
            self.assertEqual(data["parameters"], {"backend": "cpu", "lengths": [2],
                                                   "generate": 1, "seed": 7})
            on_policy(policy)
            (request.parent / "result.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            return 0

        code, report = self.launch(runner)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["job_policy_verified"])
        self.assertEqual(report["installed_job_policy"]["cpu_percent"], 70)
        self.assertEqual(report["child_exit_code"], 0)

    def test_launch_exception_overwrites_even_a_stale_pass_report(self):
        def runner(command, **kwargs):
            (Path(command[-1]).parent / "result.json").write_text('{"status":"PASS"}', encoding="utf-8")
            raise OSError("job installation failed")

        code, report = self.launch(runner)
        self.assertNotEqual(code, 0)
        self.assertEqual(report["status"], "ERROR")
        self.assertIn("job installation failed", report["error"])
        self.assertFalse(report["job_policy_verified"])
        self.assertFalse(report["inference_verified"])

    def test_failed_exit_cannot_publish_pass_and_success_cannot_publish_mismatch(self):
        for exit_code, status in ((1, "PASS"), (0, "NUMERICAL_MISMATCH")):
            with self.subTest(exit_code=exit_code, status=status):
                def runner(command, **kwargs):
                    path = Path(command[-1]).parent / "result.json"
                    path.write_text(json.dumps({"status": status}), encoding="utf-8")
                    return exit_code

                code, report = self.launch(runner)
                self.assertNotEqual(code, 0)
                self.assertEqual(report["status"], "ERROR")
                self.assertIn("disagrees", report["error"])

    def test_missing_oversized_and_invalid_json_reports_cannot_publish_pass(self):
        for payload in (None, " " * (2 * 1024**2 + 1), "not json"):
            with self.subTest(payload="missing" if payload is None else len(payload)):
                def runner(command, **kwargs):
                    if payload is not None:
                        path = Path(command[-1]).parent / "result.json"
                        path.write_text(payload, encoding="utf-8")
                    return 0

                code, report = self.launch(runner)
                self.assertNotEqual(code, 0)
                self.assertEqual(report["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
