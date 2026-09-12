"""Workflow tests. Scalar FP32 is a test double, not native/CUDA validation."""
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from glm_local import __main__ as cli
from glm_local.mini_run import NativeMiniLinear, execute_mini, launch_mini, validate_mini
from glm_local.parity_run import launch_parity
from glm_local.winjob import InstalledLimits
from test_mini_run import sampler, settings

HAS_STORAGE = importlib.util.find_spec("safetensors") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class ScalarFP32:
    """Independent scalar test double for sequential FP32 tile arithmetic."""
    metadata = {"backend": "test-double-scalar-fp32-not-native"}

    @staticmethod
    def _f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    def matvec_tile(self, weights, rows, cols, vector, scale):
        values = [self._f32(value) for value in vector]
        scale = self._f32(scale)
        output = []
        for row in range(rows):
            total = 0.0
            for col in range(cols):
                code = weights[row * cols + col]
                magnitude = code & 127
                if magnitude == 127:
                    raise ValueError("FP8 NaN in scalar test double")
                exponent, fraction = divmod(magnitude, 8)
                decoded = fraction * 2**-9 if exponent == 0 else (8 + fraction) * 2**(exponent - 10)
                if code & 128:
                    decoded = -decoded
                product = self._f32(self._f32(decoded * scale) * values[col])
                total = self._f32(total + product)
            output.append(total)
        return output

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class StorageOptionTests(unittest.TestCase):
    def test_invalid_storage_fails_before_launch_or_creating_reports(self):
        for value in ("auto", "real-checkpoint", "", None, True):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "storage"):
                    launch_mini(tmp, settings(), storage=value)
                self.assertFalse((Path(tmp) / "reports").exists())

    def test_legacy_validate_defaults_and_safetensors_are_both_valid(self):
        validate_mini("cpu", [8], 4, 7)
        validate_mini("cpu", [8], 4, 7, "safetensors")

    def test_cli_passes_storage_choice_to_each_launcher(self):
        for command, module, function in (("mini", "mini_run", "launch_mini"),
                                           ("parity", "parity_run", "launch_parity")):
            with self.subTest(command=command), patch(f"glm_local.{module}.{function}", return_value=0) as launch:
                self.assertEqual(cli.main([command, "--backend", "cpu", "--storage", "safetensors"]), 0)
                self.assertEqual(launch.call_args.args[-1], "safetensors")

    def test_both_launchers_propagate_storage_without_changing_job_limits(self):
        for launcher, module, worker, report_name in (
                (launch_mini, "mini_run", "glm_local.mini_worker", "mini-latest.json"),
                (launch_parity, "parity_run", "glm_local.parity_worker", "parity-latest.json")):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reference_python = root / ".venv-reference/Scripts/python.exe"
                reference_python.parent.mkdir(parents=True)
                reference_python.touch()

                def run(command, *, cwd, limits, on_policy, timeout):
                    self.assertEqual(command[1:3], ["-m", worker])
                    self.assertEqual(limits.cpu_percent, 70)
                    self.assertEqual(limits.committed_memory_bytes, 32_000_000_000)
                    self.assertEqual(timeout, 180)
                    request = Path(command[-1])
                    data = json.loads(request.read_text())
                    self.assertEqual(data["parameters"]["storage"], "safetensors")
                    on_policy(InstalledLimits(70, 32_000_000_000, True, True, True))
                    (request.parent / "result.json").write_text('{"status":"PASS"}')
                    return 0

                with patch(f"glm_local.{module}.run_local_process", side_effect=run), patch("builtins.print"):
                    self.assertEqual(launcher(root, settings(), backend="cpu", storage="safetensors"), 0)
                report = json.loads((root / "reports" / report_name).read_text())
                self.assertTrue(report["job_policy_verified"])


@unittest.skipUnless(HAS_STORAGE and HAS_NUMPY, "Decoder storage checks need safetensors and NumPy")
class ShardedDecoderTests(unittest.TestCase):
    def test_full_128_tokens_hidden_states_and_logits_equal_private_reader(self):
        from glm_local.mini_engine import MiniDecoder
        from glm_local.mini_weights import MiniWeights, write_mini_bundle
        from glm_local.mini_safetensors import MiniSafetensorWeights, write_mini_shards
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "private"
            sharded = root / "sharded"
            write_mini_bundle(original)
            write_mini_shards(original, sharded)
            with MiniWeights(original) as first, MiniSafetensorWeights(sharded) as second:
                old = MiniDecoder(first, NativeMiniLinear(first, ScalarFP32()), capture_states=True)
                new = MiniDecoder(second, NativeMiniLinear(second, ScalarFP32()), capture_states=True)
                tokens = [(7 + 7 * i) % 32 for i in range(120)]
                logits = None
                for position in range(128):
                    token = tokens[position] if position < 120 else max(range(32), key=lambda i: (logits[i], -i))
                    logits = old.step(token)
                    self.assertEqual(new.step(token), logits)
                    self.assertEqual(new.trace[-1], old.trace[-1])
                self.assertEqual(new.position, 128)
                self.assertEqual(second.stats()["resident_fp8_cache_bytes"], 0)
                self.assertLessEqual(second.stats()["peak_open_shards"], 2)

    def test_runner_safetensors_path_matches_independent_numpy_oracle(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch("glm_local.mini_run.NativeCpuBackend", ScalarFP32), \
                patch("glm_local.mini_run.sample_process", side_effect=sampler()), patch("builtins.print"):
            report = execute_mini(settings(), {"backend": "cpu", "lengths": [8, 32, 64],
                                                "generate": 4, "seed": 19, "storage": "safetensors"}, Path(tmp))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["storage"]["format"], "safetensors")
        self.assertEqual(report["storage"]["exported_fixture"]["tensor_count"], 80)
        self.assertEqual(report["storage"]["reader_stats"]["cross_shard_pairs"], 34)
        self.assertTrue(all(case["greedy_tokens_match"] for case in report["cases"]))
        self.assertFalse(report["inference_verified"])
        self.assertFalse(report["official_transformers_parity_verified"])
        self.assertFalse(report["job_policy_verified"])


@unittest.skipUnless(os.environ.get("GLM_TEST_OFFICIAL") == "1",
                     "Pinned official graph and native Windows CPU are opt-in")
class ShardedOfficialTests(unittest.TestCase):
    def test_sharded_native_hidden_states_and_selections_against_official_graph(self):
        from glm_local.cpu_probe import NativeCpuBackend
        from glm_local.mini_engine import MiniDecoder
        from glm_local.mini_weights import write_mini_bundle
        from glm_local.mini_safetensors import MiniSafetensorWeights, write_mini_shards
        from glm_local.native_topk import NativeTopK
        from glm_local.official_reference import OfficialMiniReference
        from glm_local.parity_compare import compare_step
        from glm_local.reference_env import verify_reference_environment
        verify_reference_environment()  # Never waive the project's revision lock.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mini_bundle(root / "private")
            write_mini_shards(root / "private", root / "sharded")
            with MiniSafetensorWeights(root / "sharded") as weights, NativeCpuBackend() as cpu, \
                    OfficialMiniReference(root / "private") as official:
                native = MiniDecoder(weights, NativeMiniLinear(weights, cpu),
                                     capture_states=True, attention_topk=NativeTopK())
                for token in [(7 + 7 * i) % 32 for i in range(12)]:
                    logits = native.step(token)
                    expected = official.step(token)
                    # Same comparison as the public parity runner.
                    result = compare_step(logits, native.trace[-1], expected)
                    self.assertTrue(result["passed"], result)
