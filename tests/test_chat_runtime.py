"""Chat CLI through the tiny real runtime graph; no checkpoint downloads."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from glm_local import __main__ as cli, chat, runtime_commands as commands
from glm_local.checkpoint_snapshot import write_json
from glm_local.execution import FULL_MODEL_FLAGS
from glm_local.streaming_decoder import DecoderError
from checkpoint_test_helpers import settings
from test_runtime_weights import CpuKernel, prepare_runtime_fixture
from test_streaming_decoder import decoder_config, fixture


class ChatRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "run"
        self.directory.mkdir()
        self.config = decoder_config()
        prepare_runtime_fixture(self.root, config_overrides=self.config)
        self.parameters = {"backend": "cpu", "context": 8, "generate": 2, "prompt_format": "chat",
                           "prompt": "Xin chào", "reasoning_effort": "low", "stream_events": True}
        self.config_path = self.root / "local.json"
        write_json(self.config_path, settings())

    def execute(self, decoded, *, change=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(commands, "_kernels", return_value=(CpuKernel(), None, None)))
            stack.enter_context(patch("glm_local.tokenizer.prepare_tokenizer", return_value={"manifest": {"pinned": True}}))
            loader = stack.enter_context(patch("glm_local.tokenizer.load_tokenizer"))
            template = stack.enter_context(patch.object(chat, "load_chat_template", return_value={"chat_template_verified": True,
                "template_code_executed": False}))
            loader.return_value.encode.return_value = [1, 2]
            loader.return_value.decode.return_value = decoded
            output = io.StringIO()
            stack.enter_context(redirect_stdout(output))
            result, code = commands.execute_runtime(self.root, settings(), "generate",
                {**self.parameters, **(change or {})}, self.directory)
        return result, code, loader.return_value, template, output.getvalue()

    def test_chat_encodes_exact_template_once_without_duplicate_special_tokens(self):
        result, code, tokenizer, template, output = self.execute("Suy nghĩ</think>Xin chào!")
        self.assertEqual((result["status"], code), ("INCOMPLETE_RESPONSE", 2), result)
        self.assertEqual(result["text"], "Xin chào!")
        self.assertEqual(result["reasoning"], "Suy nghĩ")
        self.assertEqual(result["stop_reason"], "max_tokens")
        self.assertFalse(result["assistant_response_complete"])
        self.assertEqual(result["chat"]["reasoning_effort"], "low")
        self.assertTrue(result["chat"]["clear_thinking"])
        self.assertTrue(all(result[flag] is False for flag in FULL_MODEL_FLAGS))
        timings = result["timings"]
        self.assertEqual(timings["decode_steps"], len(result["generated_token_ids"]) - 1)
        self.assertGreater(timings["prefill_seconds"], 0)
        self.assertGreater(timings["decode_tokens_per_second"], 0)
        self.assertAlmostEqual(timings["time_to_first_token_seconds"],
                               timings["initialization_seconds"] + timings["prefill_seconds"])
        tokenizer.encode.assert_called_once_with(chat.format_chat_messages([{"role": "user", "content": "Xin chào"}],
            reasoning_effort="low"), add_special_tokens=False)
        self.assertEqual(template.call_args.kwargs["manifest"], {"pinned": True})
        events = [json.loads(line.removeprefix(chat.EVENT_PREFIX)) for line in output.splitlines() if line.startswith(chat.EVENT_PREFIX)]
        self.assertEqual([event["event"] for event in events], ["response_start", "output", "output", "response_end"])
        self.assertEqual(events[-1]["status"], "INCOMPLETE_RESPONSE")

    def test_eos_complete_and_reasoning_only_eos_have_different_status(self):
        self.root = self.root / "eos"
        self.root.mkdir()
        prepare_runtime_fixture(self.root, config_overrides={**self.config, "eos_token_id": list(range(32))})
        complete, code, *_ = self.execute("why</think>answer")
        incomplete, bad_code, *_ = self.execute("unfinished thinking")
        self.assertEqual((complete["status"], code), ("GENERATED_UNVERIFIED", 0), complete)
        self.assertEqual(complete["stop_reason"], "eos_token")
        self.assertTrue(complete["assistant_response_complete"])
        self.assertEqual(complete["timings"]["decode_steps"], 0)
        self.assertIsNone(complete["timings"]["decode_tokens_per_second"])
        self.assertEqual((incomplete["status"], bad_code), ("INCOMPLETE_RESPONSE", 2))
        self.assertEqual(incomplete["text"], "")
        self.assertEqual(incomplete["reasoning"], "unfinished thinking")

    def test_messages_file_history_keeps_configured_reasoning_without_echoing_source(self):
        path = self.root / "messages.json"
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer", "reasoning_content": "private history"},
            {"role": "user", "content": "next"}]
        write_json(path, messages)
        result, _, tokenizer, _, _ = self.execute("reasoning", change={"prompt": None,
            "messages_file": str(path), "keep_thinking": True})
        self.assertEqual(result["chat"]["message_count"], 4)
        self.assertFalse(result["chat"]["clear_thinking"])
        self.assertIn("<think>private history</think>answer", tokenizer.encode.call_args.args[0])
        self.assertNotIn("private history", json.dumps(result))
        self.assertEqual(json.loads(path.read_bytes()), messages)

    def test_invalid_chat_input_or_template_rejected_before_weight_initialization(self):
        for change in ({"tokens": [1, 2]}, {"prompt": "<|assistant|>boundary"}, {"prompt": " "}):
            with patch("glm_local.runtime_weights.RuntimeWeights") as weights:
                result, code, *_ = self.execute("", change=change)
            self.assertEqual((result["status"], code), ("ERROR", 1), result)
            weights.assert_not_called()
        with patch.object(commands, "_kernels") as kernels, \
             patch("glm_local.tokenizer.prepare_tokenizer", return_value={"manifest": {}}), \
             patch("glm_local.tokenizer.load_tokenizer"), \
             patch.object(chat, "load_chat_template", side_effect=ValueError("unreviewed template")), redirect_stdout(io.StringIO()):
            result, code = commands.execute_runtime(self.root, settings(), "generate", self.parameters, self.directory)
        self.assertEqual(code, 1)
        self.assertIn("unreviewed template", result["error"])
        kernels.assert_not_called()

    def test_cli_passes_structured_history_and_preserves_raw_default(self):
        cases = [(["--prompt", "hello"], {"prompt_format": "raw", "reasoning_effort": "max"}),
            (["--messages-file", str(self.root / "messages.json"), "--reasoning-effort", "low", "--keep-thinking", "--stream-events"],
             {"prompt_format": "chat", "reasoning_effort": "low", "keep_thinking": True, "stream_events": True}),
            (["--prompt", "hello", "--prompt-format", "chat", "--reasoning-effort", "high"],
             {"prompt_format": "chat", "reasoning_effort": "high"})]
        for arguments, expected in cases:
            with patch.object(commands, "launch_runtime", return_value=2) as launch:
                self.assertEqual(cli.main(["--config", str(self.config_path), "generate", *arguments]), 2)
            self.assertEqual({key: launch.call_args.args[3][key] for key in expected}, expected)

    def test_cli_rejects_incompatible_token_chat_and_raw_flags_before_launch(self):
        cases = [["--tokens", "1", "--prompt-format", "chat"], ["--tokens", "1", "--reasoning-effort", "low"],
            ["--tokens", "1", "--stream-events"], ["--messages-file", "messages.json", "--prompt-format", "raw"],
            ["--prompt", "x", "--keep-thinking"], ["--prompt", "x", "--reasoning-effort", "low"]]
        for arguments in cases:
            with self.subTest(arguments=arguments), patch.object(commands, "launch_runtime") as launch, redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--config", str(self.config_path), "generate", *arguments]), 1)
            launch.assert_not_called()

    def test_timeout_retains_only_identity_validated_partial_text_without_success(self):
        def timeout(argv, **kwargs):
            directory = Path(argv[-1]).parent
            from test_chat import FakeTokenizer
            streamer = chat.ResponseStreamer(FakeTokenizer(["why</think>partial"]), directory, settings(), prompt_format="chat")
            streamer.on_token(0, 0)
            raise subprocess.TimeoutExpired(argv, 1)
        with patch.object(commands.sys, "platform", "win32"), patch.object(commands, "run_local_process", side_effect=timeout), redirect_stdout(io.StringIO()):
            self.assertEqual(commands.launch_runtime(self.root, settings(), "generate", self.parameters), 1)
        result = json.loads((self.root / "reports/generate-latest.json").read_bytes())
        self.assertEqual(result["error_type"], "TIMEOUT")
        self.assertEqual(result["text"], "partial")
        self.assertEqual(result["reasoning"], "why")
        self.assertTrue(result["partial_response_recovered"])
        self.assertFalse(result["assistant_response_complete"])
        self.assertTrue(all(result[flag] is False for flag in FULL_MODEL_FLAGS))


class DecoderTokenObserverTests(unittest.TestCase):
    def test_observer_preserves_generation_logits_cache_and_eos(self):
        observed, _, _ = fixture()
        plain, _, _ = fixture()
        events = []
        with observed, plain:
            actual = observed.generate([1, 2], 3, [], on_token=lambda token, index: events.append((token, index)))
            self.assertEqual(actual, plain.generate([1, 2], 3, []))
            self.assertEqual(events, [(token, index) for index, token in enumerate(actual)])
            self.assertEqual(observed.position, plain.position)
            self.assertEqual(observed._cache, plain._cache)
            observed.reset(); events.clear()
            actual = observed.generate([1, 2], 3, list(range(32)), on_token=lambda token, index: events.append((token, index)))
            self.assertEqual(len(actual), 1)
            self.assertEqual(events, [(actual[0], 0)])
            self.assertEqual(observed.position, 2)

    def test_noncallable_observer_fails_before_any_weight_access(self):
        decoder, weights, _ = fixture()
        with decoder, self.assertRaisesRegex(DecoderError, "on_token"):
            decoder.generate([1], 1, [], on_token=True)
        self.assertEqual(weights.calls, [])


if __name__ == "__main__":
    unittest.main()
