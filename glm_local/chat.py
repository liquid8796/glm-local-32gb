"""Reviewed GLM text chat formatting and bounded, structured response streaming.

Production never evaluates Jinja. The pinned local template is verified as
data and must match the reviewed SHA-256 before the equivalent text-only
formatter is used. Tools, media and repository Python are not supported here.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import time

from .checkpoint_http import MetadataError, strict_json, validate_target
from .checkpoint_snapshot import write_json
from .execution import FULL_MODEL_FLAGS
from .runtime_progress import write_runtime_line
from .tokenizer import _Artifact, _directory, _regular, _verify_file

TEMPLATE_NAME = "chat_template.jinja"
TEMPLATE_SHA256 = "4a4b64df09bd4f54fb18a2cfc86b99c329d37362a8c289d466b443a99bac0645"
MAX_TEMPLATE_BYTES = 65536
MAX_MESSAGES = 256
MAX_CHAT_BYTES = 1024**2
MAX_RESPONSE_BYTES = 1024**2
EVENT_PREFIX = "MODELDESK_EVENT "
_RESERVED = ("[gMASK]", "[MASK]", "[sMASK]", "<sop>", "<eop>", "<|system|>", "<|user|>",
             "<|assistant|>", "<|observation|>", "<|endoftext|>")


def validate_messages(messages, *, require_user_tail=False):
    if not isinstance(messages, (list, tuple)) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError("Chat requires 1..256 structured messages")
    result, size = [], 0
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content", "reasoning_content"}:
            raise ValueError("Chat messages support only role, text content and optional assistant reasoning_content")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise ValueError("Only text system/user/assistant messages are supported")
        row = {"role": role, "content": content}
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            if role != "assistant" or not isinstance(reasoning, str):
                raise ValueError("reasoning_content must be assistant text")
            row["reasoning_content"] = reasoning
        for text in (content, reasoning or ""):
            if any(marker in text for marker in _RESERVED):
                raise ValueError("Message text contains reserved chat boundary markers")
            size += len(text.encode("utf-8"))
        if size > MAX_CHAT_BYTES:
            raise ValueError("Combined conversation text exceeds 1 MiB")
        result.append(row)
    if require_user_tail and (result[-1]["role"] != "user" or not result[-1]["content"].strip()):
        raise ValueError("Chat generation requires a nonempty final user message")
    return result


def read_messages_file(path):
    path = Path(path)
    before = _regular(path)
    if not 1 <= before.st_size <= MAX_CHAT_BYTES:
        raise ValueError("Conversation JSON must be a bounded file of at most 1 MiB")
    with path.open("rb") as stream:
        raw = stream.read(MAX_CHAT_BYTES + 1)
    after = _regular(path)
    if (len(raw) != before.st_size or (before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("Conversation file changed during bounded read")
    return validate_messages(strict_json(raw), require_user_tail=True)


def load_chat_template(model_directory, *, model_id, revision, manifest):
    validate_target(model_id, revision)
    if not isinstance(manifest, dict) or manifest.get("id") != model_id or manifest.get("sha") != revision:
        raise MetadataError("Chat template manifest differs from the pinned model identity")
    siblings = manifest.get("siblings")
    if not isinstance(siblings, list) or len(siblings) > 8192:
        raise MetadataError("Chat template requires a bounded pinned model manifest")
    matches = [item for item in siblings if isinstance(item, dict) and item.get("rfilename") == TEMPLATE_NAME]
    if len(matches) != 1:
        raise MetadataError("Pinned model manifest must identify exactly one chat_template.jinja")
    item = matches[0]
    size = item.get("size")
    if type(size) is not int or not 1 <= size <= MAX_TEMPLATE_BYTES:
        raise MetadataError("Chat template size exceeds its 64-KiB policy")
    if "lfs" in item:
        lfs = item["lfs"]
        if not isinstance(lfs, dict) or type(lfs.get("size")) is not int or lfs["size"] != size or not re.fullmatch(r"[0-9a-f]{64}", str(lfs.get("sha256", ""))):
            raise MetadataError("Chat template LFS digest is invalid")
        artifact = _Artifact(TEMPLATE_NAME, size, "sha256", lfs["sha256"])
    else:
        digest = item.get("blobId")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest):
            raise MetadataError("Chat template requires a pinned Git blob hash")
        artifact = _Artifact(TEMPLATE_NAME, size, "git_blob_sha1", digest)
    directory = Path(model_directory).absolute()
    _directory(directory)
    try:
        receipt, raw = _verify_file(directory / TEMPLATE_NAME, artifact, capture=True)
    except FileNotFoundError as error:
        raise FileNotFoundError("Chat requires the pinned chat_template.jinja in the model folder; download that file using ModelDesk's model file list") from error
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise MetadataError("Chat template is not the reviewed GLM text-chat profile; refusing to execute arbitrary Jinja")
    return {**receipt, "chat_template_verified": True, "format": "glm_text_chat_v1",
            "template_code_executed": False, "scope": "Supported text roles only; no tools or media"}


def format_chat_messages(messages, *, reasoning_effort="max", clear_thinking=True, add_generation_prompt=True,
                         direct_answer=False):
    """Reviewed text template, with an explicit optional assistant-prefix extension."""
    messages = validate_messages(messages)
    if reasoning_effort not in ("low", "high", "max"):
        raise ValueError("Reasoning effort must be low, high or max")
    if type(clear_thinking) is not bool or type(add_generation_prompt) is not bool or type(direct_answer) is not bool:
        raise ValueError("Chat formatting switches must be boolean")
    if direct_answer and not add_generation_prompt:
        raise ValueError("Direct answer requires an assistant generation prefix")
    last_user = max((index for index, message in enumerate(messages) if message["role"] == "user"), default=-1)
    pieces = ["[gMASK]<sop><|system|>Reasoning Effort: " + reasoning_effort.capitalize()]
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"]
        pieces.append("<|" + role + "|>")
        if role != "assistant":
            pieces.append(content)
            continue
        reasoning = message.get("reasoning_content")
        if reasoning is None and "</think>" in content:
            reasoning = content.split("</think>")[0].split("<think>")[-1]
            content = content.split("</think>")[-1]
        pieces.append("<think>" + (reasoning if reasoning is not None and (not clear_thinking or index > last_user) else "") + "</think>")
        if content.strip():
            pieces.append(content.strip())
    if add_generation_prompt:
        pieces.append("<|assistant|><think>")
        if direct_answer:
            # The official reviewed template has no thinking-off switch. This
            # explicit extension closes its generation prefix without adding a reply.
            pieces.append("</think>")
    rendered = "".join(pieces)
    if len(rendered.encode("utf-8")) > MAX_CHAT_BYTES:
        raise ValueError("Formatted conversation exceeds 1 MiB")
    return rendered


def split_assistant_text(text, *, thinking_open=False, detect_reopening=False):
    """Separate observed model reasoning from its visible assistant response."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Decoded response exceeds its 1-MiB text bound")
    if detect_reopening:
        channels = _IncrementalChannels(thinking_open, detect_reopening=True)
        parts = {"reasoning": [], "assistant": []}
        for channel, piece in channels.push(text):
            parts[channel].append(piece)
        parts["reasoning" if channels.state == "reasoning" else "assistant"].append(channels.pending)
        return {"reasoning": "".join(parts["reasoning"]), "text": "".join(parts["assistant"]),
                "thinking_complete": channels.state != "reasoning"}
    if text.startswith("<think>"):
        text = text[len("<think>"):]
        thinking_open = True
    if thinking_open:
        if "</think>" not in text:
            return {"reasoning": text, "text": "", "thinking_complete": False}
        reasoning, visible = text.split("</think>", 1)
        return {"reasoning": reasoning, "text": visible, "thinking_complete": True}
    return {"reasoning": "", "text": text, "thinking_complete": True}


class _IncrementalChannels:
    def __init__(self, thinking_open, *, detect_reopening=False):
        self.state = "opening_reasoning" if thinking_open else "detect"
        self.pending = ""
        self.detect_reopening = detect_reopening

    def _push_reopening(self, chunk):
        """Direct-prefix output may reopen thinking even after whitespace/text."""
        emitted = []
        self.pending += chunk
        if self.state in ("detect", "opening_reasoning"):
            self.state = "reasoning" if self.state == "opening_reasoning" else "assistant"
        while self.pending:
            marker = "</think>" if self.state == "reasoning" else "<think>"
            offset = self.pending.find(marker)
            if offset >= 0:
                if offset:
                    emitted.append((self.state, self.pending[:offset]))
                self.pending = self.pending[offset + len(marker):]
                self.state = "assistant" if self.state == "reasoning" else "reasoning"
                continue
            keep = max((count for count in range(1, min(len(self.pending), len(marker) - 1) + 1)
                        if self.pending.endswith(marker[:count])), default=0)
            piece = self.pending[:-keep] if keep else self.pending
            if piece:
                emitted.append((self.state, piece))
            self.pending = self.pending[-keep:] if keep else ""
            break
        return emitted

    def push(self, chunk):
        if self.detect_reopening:
            return self._push_reopening(chunk)
        emitted = []
        def append(channel, text):
            if text:
                emitted.append((channel, text))
        self.pending += chunk
        if self.state in ("opening_reasoning", "detect"):
            if self.pending.startswith("<think>"):
                self.pending = self.pending[len("<think>"):]
                self.state = "reasoning"
            elif "<think>".startswith(self.pending):
                return emitted
            else:
                self.state = "reasoning" if self.state == "opening_reasoning" else "assistant"
        if self.state == "reasoning":
            if "</think>" in self.pending:
                reasoning, self.pending = self.pending.split("</think>", 1)
                append("reasoning", reasoning)
                self.state = "assistant"
            else:
                keep = max((count for count in range(1, min(len(self.pending), len("</think>") - 1) + 1)
                            if self.pending.endswith("</think>"[:count])), default=0)
                append("reasoning", self.pending[:-keep] if keep else self.pending)
                self.pending = self.pending[-keep:] if keep else ""
        if self.state == "assistant":
            append("assistant", self.pending)
            self.pending = ""
        return emitted


class ResponseStreamer:
    """Decode incrementally, emit bounded JSON line edits, then reconcile once.

    `replace_from_utf16` is a .NET string index, avoiding ambiguity for emoji.
    The native DecodeStream handles incomplete UTF-8 token boundaries. Only the
    final batch decode reconciles any residual native-decoder differences.
    """
    def __init__(self, tokenizer, directory, settings, *, prompt_format="raw", enabled=False, event_sink=None,
                 thinking_open=None):
        if thinking_open is not None and type(thinking_open) is not bool:
            raise ValueError("Initial thinking state must be boolean or unspecified")
        self.tokenizer, self.directory, self.settings = tokenizer, Path(directory), settings
        self.prompt_format, self.enabled = prompt_format, enabled
        self.thinking_open = prompt_format == "chat" if thinking_open is None else thinking_open
        self.detect_reopening = prompt_format == "chat" and not self.thinking_open
        self._sink = event_sink
        self._native = tokenizer.decode_stream() if callable(getattr(type(tokenizer), "decode_stream", None)) else None
        self._channels = _IncrementalChannels(self.thinking_open, detect_reopening=self.detect_reopening)
        self._sent = {"reasoning": io.StringIO(), "assistant": io.StringIO()}
        self._utf16 = {"reasoning": 0, "assistant": 0}
        self._ids = []
        self._decoded_bytes = 0
        self._last_snapshot = float("-inf")
        self._emit({"event": "response_start", "prompt_format": prompt_format, "thinking_open": self.thinking_open})

    def _emit(self, event):
        if self._sink is not None:
            self._sink(event)
        if self.enabled:
            try:
                write_runtime_line(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            except (OSError, UnicodeError, ValueError):
                self.enabled = False

    @staticmethod
    def _chunks(text):
        chunks, current, size = [], [], 0
        for character in text:
            count = len(character.encode("utf-8"))
            if size + count > 2048:
                chunks.append("".join(current)); current, size = [], 0
            current.append(character); size += count
        if current or not chunks:
            chunks.append("".join(current))
        return chunks

    def _append(self, channel, text):
        for chunk in self._chunks(text):
            self._emit({"event": "output", "channel": channel, "replace_from_utf16": self._utf16[channel],
                        "text": chunk, "generated_tokens": len(self._ids)})
            self._sent[channel].write(chunk)
            self._utf16[channel] += len(chunk.encode("utf-16-le")) // 2

    def _send_edit(self, channel, target):
        previous = self._sent[channel].getvalue()
        common = 0
        for left, right in zip(previous, target):
            if left != right:
                break
            common += 1
        if common == len(previous) == len(target):
            return
        offset = len(target[:common].encode("utf-16-le")) // 2
        for chunk in self._chunks(target[common:]):
            self._emit({"event": "output", "channel": channel, "replace_from_utf16": offset,
                        "text": chunk, "generated_tokens": len(self._ids)})
            offset += len(chunk.encode("utf-16-le")) // 2
        self._sent[channel] = io.StringIO(target)
        self._sent[channel].seek(0, 2)
        self._utf16[channel] = len(target.encode("utf-16-le")) // 2

    def on_token(self, token, index):
        if type(index) is not int or index != len(self._ids) or index >= 1024**2:
            raise ValueError("Generated token callback indices must be sequential")
        if type(token) is not int or not 0 <= token < 262144:
            raise ValueError("Generated token must be a bounded vocabulary ID")
        self._ids.append(token)
        if self._native is not None:
            chunk = self._native.push(token)
            if chunk:
                self._decoded_bytes += len(chunk.encode("utf-8"))
                if self._decoded_bytes > MAX_RESPONSE_BYTES:
                    raise ValueError("Generated response exceeds 1 MiB")
                for channel, piece in self._channels.push(chunk):
                    self._append(channel, piece)
        # Adapters without incremental decoding still receive a correct final
        # reconciliation. Production LocalTokenizer always supplies DecodeStream.
        self._snapshot()

    def _snapshot(self, *, force=False, status="RUNNING", final=None):
        now = time.monotonic()
        if not force and now - self._last_snapshot < 0.25:
            return
        self._last_snapshot = now
        record = {"scope": "partial_generated_response", "model_id": self.settings["model_id"],
            "revision": self.settings["revision"], "run_directory": str(self.directory.resolve()),
            "status": status, "prompt_format": self.prompt_format, "generated_tokens": len(self._ids),
            "initial_thinking_open": self.thinking_open,
            "text": self._sent["assistant"].getvalue(), "reasoning": self._sent["reasoning"].getvalue(),
            "assistant_response_complete": False, **FULL_MODEL_FLAGS}
        if final is not None:
            record.update(final)
        try:
            write_json(self.directory / "response-progress.json", record)
        except OSError:
            pass

    def finish(self, token_ids, *, eos_token_ids, requested_tokens):
        if list(token_ids) != self._ids:
            raise ValueError("Generated callback sequence differs from returned token IDs")
        raw = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        final = split_assistant_text(raw, thinking_open=self.thinking_open, detect_reopening=self.detect_reopening)
        self._send_edit("reasoning", final["reasoning"])
        self._send_edit("assistant", final["text"])
        eos = [eos_token_ids] if type(eos_token_ids) is int else eos_token_ids or []
        stop = "prefill_only" if requested_tokens == 0 else "eos_token" if token_ids and token_ids[-1] in eos else "max_tokens"
        visible = bool(final["text"].strip())
        complete = visible and final["thinking_complete"] and (stop == "eos_token" or self.prompt_format == "raw")
        incomplete = requested_tokens > 0 and not complete
        status = "INCOMPLETE_RESPONSE" if incomplete else "GENERATED_UNVERIFIED"
        result = {**final, "stop_reason": stop, "assistant_response_complete": complete, "status": status}
        if incomplete:
            result["error_type"] = "INCOMPLETE_RESPONSE"
            result["error"] = ("Token budget ended before a complete assistant reply; reasoning and partial text are retained"
                               if stop == "max_tokens" else "The model ended without a visible assistant reply")
        self._emit({"event": "response_end", "status": status, "stop_reason": stop,
                    "assistant_response_complete": complete, "generated_tokens": len(token_ids)})
        self._snapshot(force=True, status=status, final=result)
        return result


def load_partial_response(directory, settings):
    """Recover a bounded, identity-bound response without promoting success."""
    directory = Path(directory).resolve()
    path = directory / "response-progress.json"
    try:
        before = _regular(path)
        if not 1 <= before.st_size <= 6 * MAX_RESPONSE_BYTES + 8192:
            raise ValueError("Partial response record exceeds its bound")
        with path.open("rb") as stream:
            raw = stream.read(6 * MAX_RESPONSE_BYTES + 8193)
        after = _regular(path)
        if (len(raw) != before.st_size or (before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError("Partial response changed during recovery")
        value = strict_json(raw)
        if (not isinstance(value, dict) or value.get("scope") != "partial_generated_response"
                or value.get("model_id") != settings["model_id"] or value.get("revision") != settings["revision"]
                or Path(value["run_directory"]).resolve() != directory
                or value.get("prompt_format") not in ("raw", "chat")
                or value.get("status") not in ("RUNNING", "GENERATED_UNVERIFIED", "INCOMPLETE_RESPONSE")
                or type(value.get("generated_tokens")) is not int or not 0 <= value["generated_tokens"] <= 1024**2
                or any(value.get(flag) is not False for flag in FULL_MODEL_FLAGS)):
            raise ValueError("Partial response identity or capability flags differ")
        text, reasoning = value["text"], value["reasoning"]
        if (not isinstance(text, str) or not isinstance(reasoning, str)
                or len(text.encode("utf-8")) + len(reasoning.encode("utf-8")) > MAX_RESPONSE_BYTES):
            raise ValueError("Partial response text exceeds its bound")
        return {"text": text, "reasoning": reasoning, "generated_tokens": value["generated_tokens"],
                "prompt_format": value["prompt_format"], "assistant_response_complete": False,
                "partial_response_recovered": True}
    except (OSError, ValueError, TypeError, KeyError):
        return None
