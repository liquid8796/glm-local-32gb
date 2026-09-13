"""Explicit assistant-prefix extension; never an official thinking-off claim."""
from contextlib import redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glm_local import __main__ as cli, chat, runtime_commands as commands
from glm_local.execution import FULL_MODEL_FLAGS
from checkpoint_test_helpers import settings
from test_chat import FakeTokenizer, replay
import test_chat_runtime as runtime_fixtures


class DirectAnswerFormattingTests(unittest.TestCase):
    def test_only_the_explicit_generation_prefix_extension_is_added(self):
        messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hi"}]
        for effort in ("low", "high", "max"):
            normal = chat.format_chat_messages(messages, reasoning_effort=effort)
            self.assertEqual(chat.format_chat_messages(messages, reasoning_effort=effort, direct_answer=False), normal)
            direct = chat.format_chat_messages(messages, reasoning_effort=effort, direct_answer=True)
            self.assertEqual(direct, normal + "</think>")
            self.assertTrue(direct.endswith("<|assistant|><think></think>"))
            self.assertEqual(direct.count("<|user|>Hi"), 1)

    def test_extension_requires_boolean_and_a_generation_prefix(self):
        messages = [{"role": "user", "content": "Hi"}]
        for value in (1, "true", None):
            with self.assertRaises(ValueError):
                chat.format_chat_messages(messages, direct_answer=value)
        with self.assertRaisesRegex(ValueError, "generation prefix"):
            chat.format_chat_messages(messages, direct_answer=True, add_generation_prompt=False)

    @unittest.skipUnless(os.environ.get("GLM_TEST_CHAT_TEMPLATE"), "Explicit hash-approved local template oracle")
    def test_direct_prefix_equals_actual_pinned_template_plus_one_literal_closing_marker(self):
        raw = Path(os.environ["GLM_TEST_CHAT_TEMPLATE"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), chat.TEMPLATE_SHA256)
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"]).from_string(raw.decode("utf-8"))
        conversations = [[{"role": "user", "content": "Xin chào 🌍"}],
            [{"role": "user", "content": "first"}, {"role": "assistant", "content": "old reply",
              "reasoning_content": "old reasoning"}, {"role": "user", "content": "next"}]]
        for messages in conversations:
            for effort in ("low", "high", "max"):
                for clear in (True, False):
                    approved = template.render(messages=messages, tools=None, reasoning_effort=effort,
                        clear_thinking=clear, add_generation_prompt=True)
                    self.assertEqual(chat.format_chat_messages(messages, reasoning_effort=effort,
                        clear_thinking=clear, direct_answer=True), approved + "</think>")


class DirectAnswerStreamingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def stream(self, chunks, *, eos=True):
        events = []
        streamer = chat.ResponseStreamer(FakeTokenizer(chunks), self.directory, settings(),
            prompt_format="chat", thinking_open=False, event_sink=events.append)
        for index in range(len(chunks)):
            streamer.on_token(index, index)
        result = streamer.finish(list(range(len(chunks))), requested_tokens=len(chunks),
            eos_token_ids=[len(chunks) - 1] if eos else [])
        self.assertEqual(events[0], {"event": "response_start", "prompt_format": "chat", "thinking_open": False})
        self.assertEqual(replay(events), {"assistant": result["text"], "reasoning": result["reasoning"]})
        return result, events

    def test_direct_eos_answer_is_visible_without_requiring_another_thinking_close(self):
        result, events = self.stream(["Xin ", "chào 🌍", "!"])
        self.assertEqual(result["text"], "Xin chào 🌍!")
        self.assertEqual(result["reasoning"], "")
        self.assertTrue(result["thinking_complete"])
        self.assertTrue(result["assistant_response_complete"])
        self.assertEqual(result["status"], "GENERATED_UNVERIFIED")
        self.assertFalse(any(event.get("channel") == "reasoning" for event in events))

    def test_reopening_after_whitespace_and_visible_text_is_split_incrementally_and_at_finish(self):
        result, _ = self.stream(["\n", "<th", "ink>", "First 🧪", "</thi", "nk>", "Hello", "<think>", "Again", "</think>", "!"])
        self.assertEqual(result["text"], "\nHello!")
        self.assertEqual(result["reasoning"], "First 🧪Again")
        self.assertTrue(result["assistant_response_complete"])

    def test_unclosed_reopened_thinking_empty_eos_and_token_cap_are_incomplete(self):
        for chunks, eos in ((["<think>reasoning"], True), (["Hi <think>still thinking"], True),
                            ([""], True), (["Hi"], False)):
            with self.subTest(chunks=chunks, eos=eos):
                result, _ = self.stream(chunks, eos=eos)
                self.assertEqual(result["status"], "INCOMPLETE_RESPONSE")
                self.assertFalse(result["assistant_response_complete"])

    def test_partial_recovery_preserves_channels_and_never_claims_completion(self):
        streamer = chat.ResponseStreamer(FakeTokenizer(["Hi ", "<think>why"]), self.directory, settings(),
            prompt_format="chat", thinking_open=False)
        with patch.object(chat.time, "monotonic", side_effect=[0.0, 1.0]):
            streamer.on_token(0, 0); streamer.on_token(1, 1)
        partial = chat.load_partial_response(self.directory, settings())
        self.assertEqual((partial["text"], partial["reasoning"]), ("Hi ", "why"))
        self.assertFalse(partial["assistant_response_complete"])
        self.assertTrue(partial["partial_response_recovered"])
        record = json.loads((self.directory / "response-progress.json").read_bytes())
        self.assertIs(record["initial_thinking_open"], False)
        self.assertTrue(all(record[flag] is False for flag in FULL_MODEL_FLAGS))

    def test_invalid_initial_thinking_override_fails_before_decoding(self):
        for value in (0, "false", []):
            with self.assertRaises(ValueError):
                chat.ResponseStreamer(FakeTokenizer([]), self.directory, settings(), thinking_open=value)


class DirectAnswerRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runtime_fixtures.ChatRuntimeTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_runtime_receipt_and_stream_use_explicit_direct_prefix_and_default_low_effort(self):
        result, code, tokenizer, template, output = self.fixture.execute("Hello!", change={
            "direct_answer": True, "reasoning_effort": None})
        self.assertEqual((result["status"], code), ("INCOMPLETE_RESPONSE", 2))  # Tiny model ends at requested cap.
        self.assertEqual((result["text"], result["reasoning"]), ("Hello!", ""))
        self.assertIs(result["chat"]["direct_answer"], True)
        self.assertIs(result["chat"]["thinking_enabled"], False)
        self.assertEqual(result["chat"]["assistant_prefix_extension"], "</think>")
        self.assertIn("model may reopen", result["chat"]["thinking_policy"])
        self.assertEqual(result["chat"]["reasoning_effort"], "low")
        tokenizer.encode.assert_called_once_with(chat.format_chat_messages([{"role": "user", "content": "Xin chào"}],
            reasoning_effort="low", direct_answer=True), add_special_tokens=False)
        template.assert_called_once()
        first = next(json.loads(line[len(chat.EVENT_PREFIX):]) for line in output.splitlines() if line.startswith(chat.EVENT_PREFIX))
        self.assertIs(first["thinking_open"], False)
        self.assertTrue(all(result[flag] is False for flag in FULL_MODEL_FLAGS))

    def test_cli_direct_defaults_low_and_keeps_explicit_effort(self):
        for extra, effort in (([], "low"), (["--reasoning-effort", "high"], "high"),
                              (["--reasoning-effort", "max"], "max")):
            with patch.object(commands, "launch_runtime", return_value=0) as launch:
                self.assertEqual(cli.main(["--config", str(self.fixture.config_path), "generate", "--prompt", "Hi",
                    "--prompt-format", "chat", "--direct-answer", *extra]), 0)
            parameters = launch.call_args.args[3]
            self.assertIs(parameters["direct_answer"], True)
            self.assertEqual(parameters["reasoning_effort"], effort)

    def test_cli_and_launcher_reject_raw_token_or_invalid_direct_modes_before_worker(self):
        for arguments in (["--prompt", "Hi"], ["--prompt", "Hi", "--prompt-format", "raw"], ["--tokens", "1,2"]):
            with patch.object(commands, "launch_runtime") as launch, redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--config", str(self.fixture.config_path), "generate", *arguments, "--direct-answer"]), 1)
            launch.assert_not_called()
        cases = [{"prompt": "Hi", "direct_answer": True}, {"prompt_format": "chat", "tokens": [1], "direct_answer": True},
                 {"prompt_format": "chat", "prompt": "Hi", "direct_answer": "true"}]
        for parameters in cases:
            with patch.object(commands, "run_local_process") as worker, self.assertRaises(ValueError):
                commands.launch_runtime(self.fixture.root, settings(), "generate", parameters)
            worker.assert_not_called()
        self.assertFalse((self.fixture.root / "reports/generate").exists())


if __name__ == "__main__":
    unittest.main()
