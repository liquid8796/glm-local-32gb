"""Readiness gates use tiny local fixtures and mocked free-space/hardware data."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glm_local import audit


def settings():
    return {
        "model_id": "example/local-checkpoint", "revision": "a" * 40,
        "model_directory": "weights", "ram_budget_bytes": 32_000_000_000,
        "cpu_job_percent": 70, "gpu_average_target": 0.6, "gpu_window_seconds": 10,
        "gpu_index": 0, "disk_reserve_bytes": 100,
    }


def snapshot():
    return {
        "model_id": "example/local-checkpoint", "revision": "a" * 40,
        "weights": [{"name": "part.safetensors", "bytes": 4}],
        "weight_bytes": 4, "architecture": {"model_type": "glm_moe_dsa"},
    }


def hardware():
    return {"errors": [], "gpus": [{"index": 0, "name": "Mock GPU", "compute_capability": "8.6"}],
            "disks": [], "logical_processors": 8}


class SettingsValidationTests(unittest.TestCase):
    def test_documented_budget_boundaries_are_accepted(self):
        upper = settings()
        upper.update(gpu_window_seconds=3600, disk_reserve_bytes=10**13)
        lower = settings()
        lower.update(ram_budget_bytes=1, cpu_job_percent=1, gpu_average_target=0.01,
                     gpu_window_seconds=1, disk_reserve_bytes=0)
        audit.validate_settings(upper)
        audit.validate_settings(lower)

    def test_invalid_budget_limits_and_types_are_rejected(self):
        invalid = {
            "ram_budget_bytes": (0, 32_000_000_001, 32 * 1024**3, 1.5, True, "32GB"),
            "cpu_job_percent": (0, 71, 70.5, True),
            "gpu_average_target": (0, 0.601, float("nan"), float("inf"), True),
            "gpu_window_seconds": (0, 3601, float("-inf"), True),
            "disk_reserve_bytes": (-1, 10**13 + 1, 1.5, True),
            "gpu_index": (-1, 0.5, True, "0"),
            "model_directory": ("", None),
        }
        for key, values in invalid.items():
            for value in values:
                configuration = settings()
                configuration[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    audit.validate_settings(configuration)

    def test_missing_required_budget_is_rejected(self):
        for key in ("ram_budget_bytes", "cpu_job_percent", "gpu_average_target",
                    "gpu_window_seconds", "disk_reserve_bytes", "gpu_index", "model_directory"):
            configuration = settings()
            del configuration[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.validate_settings(configuration)


class LocalCheckpointTests(unittest.TestCase):
    def test_missing_and_wrong_size_shards_require_full_expected_bytes(self):
        with tempfile.TemporaryDirectory(prefix="glm-audit-") as temp:
            root = Path(temp)
            (root / "good.safetensors").write_bytes(b"good")
            (root / "wrong.safetensors").write_bytes(b"x")
            result = audit.inspect_local_weights(root, [
                {"name": "good.safetensors", "bytes": 4},
                {"name": "wrong.safetensors", "bytes": 7},
                {"name": "missing.safetensors", "bytes": 11},
            ])
        self.assertEqual(result["matching_size_bytes"], 4)
        self.assertEqual(result["missing_shards"], ["missing.safetensors"])
        self.assertEqual(result["wrong_size_shards"], ["wrong.safetensors"])
        self.assertEqual(result["additional_bytes_required"], 18)
        self.assertIn("content and revision NOT verified", result["verification"])

    def test_local_path_traversal_and_invalid_sizes_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="glm-audit-") as temp:
            for name in ("../outside.safetensors", "C:/outside.safetensors",
                         "nested/../../outside.safetensors", "nested\\outside.safetensors"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    audit.inspect_local_weights(temp, [{"name": name, "bytes": 4}])
            for size in (None, -1, 0, 4.0, True):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    audit.inspect_local_weights(temp, [{"name": "part.safetensors", "bytes": size}])

    def test_free_space_probe_uses_existing_parent_of_future_model_directory(self):
        with tempfile.TemporaryDirectory(prefix="glm-audit-") as temp:
            root = Path(temp).resolve()
            with patch.object(audit.shutil, "disk_usage", return_value=SimpleNamespace(free=12345)) as usage:
                result = audit.disk_free_for(root / "future" / "checkpoint")
            self.assertEqual(result, 12345)
            self.assertEqual(usage.call_args.args[0], root)


class ReadinessGateTests(unittest.TestCase):
    def evaluate(self, root, *, configuration=None, manifest=None, machine=None, free=1000):
        with patch.object(audit, "disk_free_for", return_value=free):
            return audit.evaluate(configuration or settings(), manifest or snapshot(),
                                  machine if machine is not None else hardware(), root)

    def test_ready_claims_and_matching_sizes_cannot_enable_unimplemented_backend(self):
        with tempfile.TemporaryDirectory(prefix="glm-audit-") as temp:
            root = Path(temp)
            (root / "weights").mkdir()
            (root / "weights" / "part.safetensors").write_bytes(b"xxxx")
            configuration = settings()
            configuration.update(backend_status="ready", inference_verified=True)
            manifest = snapshot()
            manifest.update(backend_status="ready", inference_verified=True)
            result = self.evaluate(root, configuration=configuration, manifest=manifest)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIs(result["inference_verified"], False)
        codes = {item["code"] for item in result["blockers"]}
        self.assertTrue({"GLM_BACKEND_NOT_IMPLEMENTED", "RAM_RESIDENT_CAP_UNVERIFIED",
                         "GPU_PACING_NOT_INTEGRATED", "CHECKPOINT_CONTENT_UNVERIFIED"} <= codes)
        self.assertNotIn("CHECKPOINT_INCOMPLETE", codes)

    def test_missing_shards_and_exact_disk_boundary_are_reported(self):
        with tempfile.TemporaryDirectory(prefix="glm-audit-") as temp:
            exact = self.evaluate(temp, free=104)
            short = self.evaluate(temp, free=103)
        self.assertIn("CHECKPOINT_INCOMPLETE", {b["code"] for b in exact["blockers"]})
        self.assertNotIn("INSUFFICIENT_DISK", {b["code"] for b in exact["blockers"]})
        self.assertIn("INSUFFICIENT_DISK", {b["code"] for b in short["blockers"]})
        self.assertEqual(exact["storage"]["required_free_bytes"], 104)
        self.assertEqual(exact["storage"]["shortfall_bytes"], 0)
        self.assertEqual(short["storage"]["shortfall_bytes"], 1)

    def test_identity_total_size_and_duplicate_snapshot_checks_precede_disk_probe(self):
        invalid = []
        for key, value in (("model_id", "another/model"), ("revision", "b" * 40),
                           ("weight_bytes", 5), ("weights", [])):
            manifest = snapshot()
            manifest[key] = value
            invalid.append(manifest)
        for name in ("part.safetensors", "PART.safetensors"):
            manifest = snapshot()
            manifest["weights"].append({"name": name, "bytes": 4})
            manifest["weight_bytes"] = 8
            invalid.append(manifest)
        with patch.object(audit, "disk_free_for") as disk:
            for manifest in invalid:
                with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                    audit.evaluate(settings(), manifest, hardware(), ".")
        disk.assert_not_called()

    def test_missing_gpu_and_probe_errors_remain_visible_blockers(self):
        machine = {"errors": ["mock probe unavailable"], "gpus": []}
        with tempfile.TemporaryDirectory(prefix="glm-audit-") as temp:
            result = self.evaluate(temp, machine=machine)
        codes = {item["code"] for item in result["blockers"]}
        self.assertTrue({"HARDWARE_PROBE_INCOMPLETE", "GPU_UNAVAILABLE"} <= codes)
        self.assertEqual(result["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
