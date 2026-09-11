"""Synthetic streaming/control integration with independent CPU/GPU stand-ins.

These tests never load native DLLs, CUDA, telemetry, models, or subprocesses.
"""

import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from glm_local import backend_probe, synthetic_fp8
from glm_local.process_metrics import ProcessSnapshot


def mathematical_decode(code):
    sign = -1 if code >= 128 else 1
    exponent, fraction = divmod(code % 128, 8)
    if exponent == 15 and fraction == 7:
        raise ValueError("NaN")
    if exponent == 0:
        return sign * fraction / 512
    return sign * (1 + fraction / 8) * 2 ** (exponent - 7)


class MathematicalBackend:
    def __init__(self, name, events=None):
        self.name = name
        self.events = events if events is not None else []
        self.calls = []
        self.operations = 0
        self.peak_explicit_device_bytes = 0
        self.metadata = {"backend": "test mathematical CPU"}
        self.device_info = {"name": "test mathematical GPU; no device access"}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def matvec_tile(self, weights, rows, cols, vector, scale):
        self.events.append(self.name)
        self.calls.append((weights, rows, cols, tuple(vector), scale))
        self.operations += 1
        return [math.fsum(mathematical_decode(weights[row * cols + col]) * scale * vector[col]
                          for col in range(cols)) for row in range(rows)]


class RecordingGate:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.count = 0
        self.finished = False

    def before_submit(self):
        self.events.append("gate")
        self.count += 1

    def finish(self):
        self.finished = True

    def summary(self):
        return {"status": "mocked", "gpu_cap_verified": False, "submissions": self.count}


def settings():
    return {"model_directory": "unused", "ram_budget_bytes": 32_000_000_000,
            "cpu_job_percent": 70, "gpu_average_target": 0.6,
            "gpu_window_seconds": 10, "gpu_index": 0, "disk_reserve_bytes": 0}


def snapshot_sequence():
    count = 0

    def sample():
        nonlocal count
        count += 1
        return ProcessSnapshot(
            process_id=123, monotonic_seconds=float(count),
            working_set_bytes=1000, peak_working_set_bytes=2000,
            private_commit_bytes=1500, peak_private_commit_bytes=2500,
            process_cpu_seconds=count / 10, logical_cpu_count=8,
        )

    return sample


class ProbeValidationTests(unittest.TestCase):
    def test_dimensions_iterations_and_seed_reject_invalid_types_and_ranges(self):
        defaults = dict(backend="cpu", rows=257, cols=131, iterations=1, seed=7)
        invalid = {
            "backend": ("", "auto", "CUDA", None),
            "rows": (0, -1, 1025, True, 2.0, "128"),
            "cols": (0, -1, 1025, False, 2.0, "128"),
            "iterations": (0, -1, 9, True, 1.0, "1"),
            "seed": (-1, 2**32, True, 0.0, "7"),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    backend_probe.validate_probe(**(defaults | {name: value}))
        backend_probe.validate_probe("cpu", 1, 1, 1, 0)
        backend_probe.validate_probe("cpu", 1024, 1024, 8, 2**32 - 1)

    def test_hybrid_requires_two_output_row_blocks(self):
        for rows in (1, 128):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "both CPU and GPU"):
                backend_probe.validate_probe("hybrid", rows, 256, 1, 7)
        backend_probe.validate_probe("hybrid", 129, 1, 1, 7)

    def test_gpu_launch_bound_counts_column_tiles_and_all_iterations(self):
        self.assertEqual(backend_probe.MAX_OPERATIONS, 256)
        # 8 row blocks * 8 column blocks * 4 iterations = 256 GPU operations.
        backend_probe.validate_probe("gpu", 1024, 1024, 4, 0)
        with self.assertRaisesRegex(ValueError, "256 GPU launches"):
            backend_probe.validate_probe("gpu", 1024, 1024, 5, 0)
        # Hybrid sends 4 of 8 row blocks to GPU, leaving room for 8 iterations.
        backend_probe.validate_probe("hybrid", 1024, 1024, 8, 0)
        # Edge rows/columns count as complete launches, including a 1-cell edge.
        with self.assertRaisesRegex(ValueError, "256 GPU launches"):
            backend_probe.validate_probe("gpu", 897, 897, 5, 0)

    def test_comparison_rejects_errors_near_zero_without_scaling_by_other_rows(self):
        result = backend_probe.compare_outputs([1e9, 0.0001], [1e9, 0.0])
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_absolute_error"], 0.0001)
        self.assertTrue(backend_probe.compare_outputs([0.000001], [0.0])["passed"])
        self.assertTrue(backend_probe.compare_outputs([100.001], [100.0])["passed"])
        self.assertFalse(backend_probe.compare_outputs([100.01], [100.0])["passed"])

    def test_comparison_requires_finite_equal_nonempty_vectors(self):
        for actual, expected in (([], []), ([1], []), ([1, 2], [1])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                backend_probe.compare_outputs(actual, expected)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertFalse(backend_probe.compare_outputs([value], [0])["passed"])
                self.assertFalse(backend_probe.compare_outputs([0], [value])["passed"])


class StreamedProbeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="backend-probe-test-")
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.path = self.directory / "manual.f8probe"

    def write_partition_fixture(self):
        # Independent writer: 257x131 => 3 output row blocks and 2 column blocks.
        # Row blocks hold +1, +2 and -1; column blocks have scales 1/2 and 1/4.
        with self.path.open("wb") as stream:
            stream.write(struct.pack("<8sIIII", b"FP8PROBE", 1, 257, 131, 7))
            for code, rows in ((0x38, 128), (0x40, 128), (0xB8, 1)):
                for cols, scale in ((128, 0.5), (3, 0.25)):
                    stream.write(struct.pack("<f", scale))
                    stream.write(bytes([code]) * rows * cols)

    def test_hybrid_alternates_output_blocks_and_gates_each_gpu_operation(self):
        self.write_partition_fixture()
        events = []
        cpu = MathematicalBackend("cpu", events)
        gpu = MathematicalBackend("gpu", events)
        gate = RecordingGate(events)
        vector = [1.0] * 128 + [2.0, 3.0, 4.0]
        result, stats = backend_probe.streamed_matvec(
            self.path, vector, backend="hybrid", cpu=cpu, gpu=gpu, gate=gate
        )
        self.assertEqual(events, ["cpu", "cpu", "gate", "gpu", "gate", "gpu", "cpu", "cpu"])
        self.assertEqual([call[0][0] for call in cpu.calls], [0x38, 0x38, 0xB8, 0xB8])
        self.assertEqual([call[0][0] for call in gpu.calls], [0x40, 0x40])
        self.assertEqual([call[2] for call in gpu.calls], [128, 3])
        self.assertEqual(gpu.calls[1][3], (2.0, 3.0, 4.0))
        self.assertEqual(result, [66.25] * 128 + [132.5] * 128 + [-66.25])
        self.assertEqual(stats["cpu_tiles"], 4)
        self.assertEqual(stats["gpu_tiles"], 2)
        self.assertEqual(gate.count, 2)
        self.assertEqual(stats["weight_bytes_read"], 257 * 131)
        self.assertEqual(stats["max_weight_tile_bytes"], 16384)
        self.assertGreaterEqual(stats["cpu_call_seconds"], 0)
        self.assertGreaterEqual(stats["gpu_call_seconds"], 0)

    def test_explicit_cpu_and_gpu_modes_never_invoke_the_other_backend(self):
        self.write_partition_fixture()
        for mode in ("cpu", "gpu"):
            with self.subTest(mode=mode):
                events = []
                active = MathematicalBackend(mode, events)
                unused = Mock()
                gate = RecordingGate(events)
                cpu, gpu = (active, unused) if mode == "cpu" else (unused, active)
                result, stats = backend_probe.streamed_matvec(
                    self.path, [1.0] * 131, backend=mode, cpu=cpu, gpu=gpu, gate=gate
                )
                unused.matvec_tile.assert_not_called()
                self.assertEqual(len(active.calls), 6)
                self.assertEqual(stats[f"{mode}_tiles"], 6)
                self.assertEqual(gate.count, 6 if mode == "gpu" else 0)
                self.assertEqual(events, ["gate", "gpu"] * 6 if mode == "gpu" else ["cpu"] * 6)
                self.assertEqual(result, [64.75] * 128 + [129.5] * 128 + [-64.75])

    def test_missing_explicit_backend_or_gate_fails_without_fallback(self):
        self.write_partition_fixture()
        cpu, gpu, gate = Mock(), Mock(), Mock()
        missing = [
            {"backend": "cpu", "gpu": gpu, "gate": gate},
            {"backend": "gpu", "cpu": cpu, "gate": gate},
            {"backend": "gpu", "cpu": cpu, "gpu": gpu},
            {"backend": "hybrid", "gpu": gpu, "gate": gate},
            {"backend": "hybrid", "cpu": cpu, "gate": gate},
            {"backend": "hybrid", "cpu": cpu, "gpu": gpu},
        ]
        for arguments in missing:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                backend_probe.streamed_matvec(self.path, [1.0] * 131, **arguments)
        cpu.matvec_tile.assert_not_called()
        gpu.matvec_tile.assert_not_called()
        gate.before_submit.assert_not_called()

    def test_vector_length_and_backend_failures_cannot_silently_return_results(self):
        synthetic_fp8.write_fixture(self.path, 2, 3)
        cpu = Mock()
        for vector in ([], [1, 2], [1, 2, 3, 4]):
            with self.subTest(vector=vector), self.assertRaisesRegex(ValueError, "length"):
                backend_probe.streamed_matvec(self.path, vector, backend="cpu", cpu=cpu)
        cpu.matvec_tile.assert_not_called()
        for output in ([], [1], [1, 2, 3], [math.nan, 1], [math.inf, 1]):
            cpu.matvec_tile.return_value = output
            with self.subTest(output=output), self.assertRaisesRegex(RuntimeError, "invalid"):
                backend_probe.streamed_matvec(self.path, [1, 2, 3], backend="cpu", cpu=cpu)

    def test_gate_failure_prevents_gpu_submission_and_cpu_fallback(self):
        synthetic_fp8.write_fixture(self.path, 1, 1)
        gpu, cpu, gate = Mock(), Mock(), Mock()
        gate.before_submit.side_effect = RuntimeError("telemetry unavailable")
        with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
            backend_probe.streamed_matvec(self.path, [1], backend="gpu", cpu=cpu, gpu=gpu, gate=gate)
        gpu.matvec_tile.assert_not_called()
        cpu.matvec_tile.assert_not_called()

    def test_truncated_or_corrupted_fixture_is_rejected_before_first_computation(self):
        synthetic_fp8.write_fixture(self.path, 2, 3)
        original = self.path.read_bytes()
        corruptions = [original[:10], original[:-1], original + b"extra",
                       b"BADMAGIC" + original[8:]]
        for scale in (0, -1, math.nan, math.inf):
            corruptions.append(original[:24] + struct.pack("<f", scale) + original[28:])
        for code in (0x7F, 0xFF):
            corruptions.append(original[:28] + bytes([code]) + original[29:])
        cpu = Mock()
        for index, corrupted in enumerate(corruptions):
            self.path.write_bytes(corrupted)
            with self.subTest(corruption=index), self.assertRaises(ValueError):
                backend_probe.streamed_matvec(self.path, [1, 2, 3], backend="cpu", cpu=cpu)
        cpu.matvec_tile.assert_not_called()

    def test_streaming_stats_match_actual_unbuffered_bounded_reads(self):
        self.write_partition_fixture()
        actual_read_sizes = []
        requests = []
        real_open = open
        testcase = self

        class Reader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.stream.close()

            def fileno(self):
                return self.stream.fileno()

            def read(self, size=-1):
                testcase.assertGreater(size, 0)
                testcase.assertLessEqual(size, 16384)
                requests.append(size)
                data = self.stream.read(size)
                actual_read_sizes.append(len(data))
                return data

        def tracked_open(*args, **kwargs):
            self.assertEqual(kwargs.get("buffering"), 0)
            return Reader(real_open(*args, **kwargs))

        with patch.object(synthetic_fp8, "open", create=True, side_effect=tracked_open):
            _, stats = backend_probe.streamed_matvec(
                self.path, [1] * 131, backend="cpu", cpu=MathematicalBackend("cpu")
            )
        self.assertEqual(stats["weight_bytes_read"], 257 * 131)
        self.assertEqual(sum(actual_read_sizes), 2 * 24 + 6 * 4 + 257 * 131)
        self.assertEqual(max(requests), stats["max_weight_tile_bytes"])
        self.assertEqual(stats["cpu_tiles"] + stats["gpu_tiles"], 6)

    def test_worker_reports_both_paths_iterations_and_unverified_limits(self):
        cpu, gpu, gate = MathematicalBackend("cpu"), MathematicalBackend("gpu"), RecordingGate()
        parameters = dict(backend="hybrid", rows=129, cols=3, iterations=2, seed=7)
        with patch.object(backend_probe, "NativeCpuBackend", return_value=cpu), \
                patch.object(backend_probe, "CudaTileBackend", return_value=gpu), \
                patch.object(backend_probe, "GpuBoundaryGate", return_value=gate), \
                patch.object(backend_probe, "sample_process", side_effect=snapshot_sequence()):
            result = backend_probe.execute_probe(settings(), parameters, self.directory)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["comparisons"]), 2)
        self.assertEqual([item["gpu_tiles"] for item in result["iterations"]], [1, 1])
        self.assertEqual([item["cpu_tiles"] for item in result["iterations"]], [1, 1])
        self.assertEqual(result["backend_details"]["gpu_launches"], 2)
        self.assertEqual(gate.count, 2)
        self.assertTrue(gate.finished)
        self.assertTrue(cpu.closed)
        self.assertTrue(gpu.closed)
        self.assertFalse(result["inference_verified"])
        self.assertFalse(result["full_model_loaded"])
        self.assertFalse(result["job_policy_verified"])
        self.assertFalse(result["resources"]["physical_ram_hard_cap_verified"])
        self.assertFalse(result["resources"]["full_model_resource_limits_verified"])
        self.assertEqual(result["resources"]["max_allowed_single_file_read_bytes"], 16384)

    def test_worker_initialization_failure_is_explicit_and_closes_started_cpu(self):
        cpu = MathematicalBackend("cpu")
        parameters = dict(backend="hybrid", rows=129, cols=1, iterations=1, seed=7)
        with patch.object(backend_probe, "NativeCpuBackend", return_value=cpu), \
                patch.object(backend_probe, "CudaTileBackend", side_effect=OSError("CUDA missing")), \
                patch.object(backend_probe, "sample_process", side_effect=snapshot_sequence()), \
                self.assertRaisesRegex(OSError, "CUDA missing"):
            backend_probe.execute_probe(settings(), parameters, self.directory)
        self.assertTrue(cpu.closed)
        self.assertEqual(cpu.calls, [])

    def test_observed_rss_over_budget_stops_before_next_iteration_and_closes_backends(self):
        cpu, gpu, gate = MathematicalBackend("cpu"), MathematicalBackend("gpu"), RecordingGate()
        parameters = dict(backend="hybrid", rows=129, cols=1, iterations=2, seed=7)
        sample = snapshot_sequence()
        oversized = ProcessSnapshot(
            process_id=123, monotonic_seconds=3.0,
            working_set_bytes=4096, peak_working_set_bytes=4096,
            private_commit_bytes=4096, peak_private_commit_bytes=4096,
            process_cpu_seconds=0.3, logical_cpu_count=8,
        )
        with patch.object(backend_probe, "NativeCpuBackend", return_value=cpu), \
                patch.object(backend_probe, "CudaTileBackend", return_value=gpu), \
                patch.object(backend_probe, "GpuBoundaryGate", return_value=gate), \
                patch.object(backend_probe, "sample_process", side_effect=[sample(), sample(), oversized]), \
                self.assertRaisesRegex(RuntimeError, "RSS exceeded"):
            backend_probe.execute_probe(settings() | {"ram_budget_bytes": 2000},
                                        parameters, self.directory)
        self.assertEqual(len(cpu.calls), 1)
        self.assertEqual(len(gpu.calls), 1)
        self.assertEqual(gate.count, 1)
        self.assertTrue(cpu.closed)
        self.assertTrue(gpu.closed)

    def test_finite_weight_corruption_is_detected_by_independent_worker_reference(self):
        actual_writer = synthetic_fp8.write_fixture

        def corrupting_writer(path, rows, cols, seed):
            info = actual_writer(path, rows, cols, seed)
            with open(path, "r+b") as stream:
                stream.seek(28)
                stream.write(b"\x7e")  # Finite +448; valid format, wrong generated weight.
            return info

        parameters = dict(backend="cpu", rows=2, cols=3, iterations=1, seed=7)
        with patch.object(backend_probe, "write_fixture", side_effect=corrupting_writer), \
                patch.object(backend_probe, "NativeCpuBackend", return_value=MathematicalBackend("cpu")), \
                patch.object(backend_probe, "sample_process", side_effect=snapshot_sequence()):
            report = backend_probe.execute_probe(settings(), parameters, self.directory)
        self.assertEqual(report["status"], "NUMERICAL_MISMATCH")
        self.assertFalse(report["comparisons"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
