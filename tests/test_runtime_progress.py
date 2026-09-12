"""Stage diagnostics remain bounded, recoverable and separate from inference."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from checkpoint_test_helpers import settings
from glm_local.checkpoint_snapshot import write_json
from glm_local.execution import FULL_MODEL_FLAGS
from glm_local.runtime_progress import MAX_PROGRESS_BYTES, RuntimeProgress, load_last_progress


class RuntimeProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_many_stage_events_keep_one_bounded_atomic_record_and_throttled_stdout(self):
        now = [100.0]
        output = io.StringIO()
        with redirect_stdout(output), patch("glm_local.runtime_progress.write_json", wraps=write_json) as writes:
            progress = RuntimeProgress(self.directory, settings(), "generate", heartbeat=False, clock=lambda: now[0])
            for index in range(1000):
                progress("projection_start", tensor_name="model.layers.0.self_attn.q_a_proj.weight", position=index,
                         rows=2048, cols=6144)
            self.assertEqual(writes.call_count, 1)
            now[0] += 7
            snapshot = progress.finish("ERROR")
            self.assertEqual(writes.call_count, 2)
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertEqual(snapshot["sequence"], 1000)
        self.assertEqual(snapshot["stage_elapsed_seconds"], 7)
        self.assertEqual(snapshot["elapsed_seconds"], 7)
        self.assertEqual(snapshot["position"], 999)
        self.assertLessEqual((self.directory / "progress.json").stat().st_size, MAX_PROGRESS_BYTES)
        self.assertEqual([path.name for path in self.directory.iterdir()], ["progress.json"])
        restored, error = load_last_progress(self.directory, settings(), "generate")
        self.assertIsNone(error)
        self.assertEqual(restored["stage"], "projection_start")
        self.assertTrue(restored["diagnostic_only"])
        for flag in FULL_MODEL_FLAGS:
            self.assertIs(restored[flag], False)

    def test_heartbeat_publishes_a_quiet_long_stage_without_decoder_callbacks(self):
        output = io.StringIO()
        with redirect_stdout(output):
            progress = RuntimeProgress(self.directory, settings(), "generate", interval=0.02, stdout_interval=0.02)
            try:
                progress("projection_start", tensor_name="model.layers.0.self_attn.q_a_proj.weight", rows=2048, cols=6144)
                deadline = time.monotonic() + 2
                while True:
                    record, _ = load_last_progress(self.directory, settings(), "generate")
                    if record and record["stage"] == "projection_start" and record["stage_elapsed_seconds"] >= 0.05:
                        break
                    if time.monotonic() > deadline:
                        self.fail("Heartbeat did not publish a quiet stage")
                    time.sleep(0.005)
            finally:
                progress.finish("INTERRUPTED")
        self.assertFalse(progress._thread.is_alive())
        self.assertIn("tensor=model.layers.0.self_attn.q_a_proj.weight", output.getvalue())
        self.assertIn("stage_elapsed=", output.getvalue())

    def test_missing_broken_or_mismatched_progress_is_rejected(self):
        self.assertIsNone(load_last_progress(self.directory, settings(), "generate")[0])
        with redirect_stdout(io.StringIO()):
            progress = RuntimeProgress(self.directory, settings(), "generate", heartbeat=False)
            original = progress.finish("ERROR")
        for change in ({"model_id": "wrong/model"}, {"revision": "b" * 40}, {"action": "projection"},
                       {"run_directory": str(self.directory.parent)}, {"inference_verified": True},
                       {"diagnostic_only": False}, {"elapsed_seconds": float("nan")}, {"sequence": True},
                       {"unexpected": "untrusted metadata"}, {"stage": "bad\nline"}, {"format_version": True}):
            with self.subTest(change=change):
                write_json(self.directory / "progress.json", {**original, **change})
                record, error = load_last_progress(self.directory, settings(), "generate")
                self.assertIsNone(record)
                self.assertIsNotNone(error)
        (self.directory / "progress.json").write_text("broken", encoding="utf-8")
        self.assertIsNone(load_last_progress(self.directory, settings(), "generate")[0])
        (self.directory / "progress.json").write_bytes(b" " * (MAX_PROGRESS_BYTES + 1))
        self.assertIsNone(load_last_progress(self.directory, settings(), "generate")[0])

    def test_observer_never_accepts_text_tokens_or_payload_arrays(self):
        with redirect_stdout(io.StringIO()):
            progress = RuntimeProgress(self.directory, settings(), "generate", heartbeat=False)
            for fields in ({"prompt": "private prompt"}, {"token_ids": [1, 2]}, {"position": True},
                           {"tensor_name": "x" * 513}, {"phase": "\ud800"}):
                with self.subTest(fields=repr(fields)), self.assertRaises(ValueError):
                    progress("projection_start", **fields)
            record = progress.finish("ERROR")
        self.assertEqual(record["sequence"], 0)
        self.assertNotIn("private prompt", (self.directory / "progress.json").read_text())

    def test_diagnostic_io_failure_does_not_raise_or_change_terminal_state(self):
        with redirect_stdout(io.StringIO()), patch("glm_local.runtime_progress.write_json", side_effect=OSError("test failure")):
            progress = RuntimeProgress(self.directory, settings(), "generate", heartbeat=False)
            progress("weights_initializing")
            record = progress.finish("GENERATED_UNVERIFIED")
        self.assertEqual(record["status"], "GENERATED_UNVERIFIED")
        self.assertIn("diagnostic_io_error", record)
        self.assertFalse(record["inference_verified"])

    def test_new_stage_clears_stale_projection_and_phase_position_fields(self):
        with redirect_stdout(io.StringIO()):
            progress = RuntimeProgress(self.directory, settings(), "generate", heartbeat=False)
            progress("prefill", phase="prefill", token_index=2, token_count=3)
            progress("layer_start", layer_index=77, layer_count=78)
            progress("projection_start", tensor_name="model.layers.77.self_attn.o_proj.weight", rows=16, cols=16)
            progress("output_head")
            current = progress.snapshot()
            self.assertNotIn("layer_index", current)
            self.assertNotIn("tensor_name", current)
            progress("decode", phase="decode", token_index=0, token_count=1)
            current = progress.finish("GENERATED_UNVERIFIED")
        self.assertEqual(current["phase"], "decode")
        self.assertEqual(current["token_index"], 0)
        self.assertEqual(current["token_count"], 1)


if __name__ == "__main__":
    unittest.main()
