"""Pinned local tokenizer artifacts and optional, explicitly requested small downloads.

Only tokenizer.json and tokenizer_config.json are accepted. Native ``tokenizers``
loads verified JSON bytes; no Transformers auto class, repository Python, chat
template, remote code, tensor payload or model download is executed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from urllib.parse import quote
import uuid

from .architecture import report as architecture
from .checkpoint_http import FetchLimits, HttpMetadataSource, MetadataError, strict_json, validate_target
from .checkpoint_snapshot import OfflineMetadataSource
from .execution import FULL_MODEL_FLAGS
from .model_profiles import reports_directory
from .safetensor_reader import MAX_READ_BYTES

ARTIFACTS = ("tokenizer.json", "tokenizer_config.json")
MAX_TOKENIZER_BYTES = 32 * 1024**2
MAX_CONFIG_BYTES = 256 * 1024
MAX_VOCAB_SIZE = 262144
MAX_TEXT_BYTES = 1024**2
MAX_TOKEN_IDS = 1024**2
EXPECTED_TOKENIZERS_VERSION = "0.23.2"


@dataclass(frozen=True)
class _Artifact:
    name: str
    size: int
    algorithm: str
    expected_digest: str


def _artifacts(manifest, model_id, revision):
    validate_target(model_id, revision)
    if not isinstance(manifest, dict) or manifest.get("id") != model_id or manifest.get("sha") != revision:
        raise MetadataError("Tokenizer manifest model/revision differs from the pinned settings")
    siblings = manifest.get("siblings")
    if not isinstance(siblings, list) or len(siblings) > 8192:
        raise MetadataError("Tokenizer manifest requires bounded model API siblings")
    selected = {}
    for item in siblings:
        if not isinstance(item, dict):
            raise MetadataError("Invalid tokenizer model API sibling")
        name = item.get("rfilename")
        if name not in ARTIFACTS:
            continue
        if name in selected:
            raise MetadataError("Duplicate tokenizer artifact in pinned manifest")
        size = item.get("size")
        maximum = MAX_CONFIG_BYTES if name == "tokenizer_config.json" else MAX_TOKENIZER_BYTES
        if type(size) is not int or not 1 <= size <= maximum:
            raise MetadataError("Tokenizer artifact size exceeds its bounded policy")
        if "lfs" in item:
            lfs = item["lfs"]
            if (not isinstance(lfs, dict) or type(lfs.get("size")) is not int or lfs["size"] != size
                    or not isinstance(lfs.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", lfs["sha256"])):
                raise MetadataError("Tokenizer LFS size/SHA-256 is missing or inconsistent")
            selected[name] = _Artifact(name, size, "sha256", lfs["sha256"])
        else:
            digest = item.get("blobId")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest):
                raise MetadataError("Tokenizer non-LFS artifact requires its Git blob SHA-1")
            selected[name] = _Artifact(name, size, "git_blob_sha1", digest)
    if set(selected) != set(ARTIFACTS):
        raise MetadataError("Pinned model manifest lacks both tokenizer JSON artifacts")
    result = tuple(selected[name] for name in ARTIFACTS)
    if sum(item.size for item in result) > MAX_TOKENIZER_BYTES:
        raise MetadataError("Combined tokenizer artifacts exceed the 32-MiB budget")
    return result


def _hashes(artifact):
    sha256 = hashlib.sha256()
    git = hashlib.sha1(b"blob " + str(artifact.size).encode("ascii") + b"\0")
    return sha256, git


def _receipt(artifact, sha256, git):
    observed = sha256.hexdigest() if artifact.algorithm == "sha256" else git.hexdigest()
    if observed != artifact.expected_digest:
        raise MetadataError(f"Tokenizer {artifact.name} {artifact.algorithm} differs from the pinned manifest")
    return {"name": artifact.name, "bytes": artifact.size, "sha256": sha256.hexdigest(),
            "manifest_hash_algorithm": artifact.algorithm, "manifest_hash": observed,
            "manifest_hash_verified": True}


def _file_identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise MetadataError("Tokenizer artifacts must be regular files, not links/reparse points")
    return info


def _directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise MetadataError("Tokenizer directory must be a regular directory, not a link/reparse point")


def _verify_file(path, artifact, *, capture=False):
    try:
        before = _regular(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {artifact.name}; run tokenizer-check --online to download only the two tokenizer JSON files") from error
    if before.st_size != artifact.size:
        raise MetadataError(f"Tokenizer {artifact.name} file size differs from the pinned manifest")
    sha256, git = _hashes(artifact)
    parts = [] if capture else None
    with path.open("rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        if _file_identity(opened)[:4] != _file_identity(before)[:4]:
            raise MetadataError("Tokenizer artifact changed while opening")
        remaining = artifact.size
        while remaining:
            raw = stream.read(min(remaining, MAX_READ_BYTES))
            if not raw:
                raise MetadataError("Tokenizer artifact was truncated during verification")
            remaining -= len(raw)
            sha256.update(raw)
            git.update(raw)
            if parts is not None:
                parts.append(raw)
        if (stream.read(1) or _file_identity(os.fstat(stream.fileno())) != _file_identity(opened)
                or _file_identity(_regular(path)) != _file_identity(before)):
            raise MetadataError("Tokenizer artifact changed during verification")
    return _receipt(artifact, sha256, git), b"".join(parts) if parts is not None else None


def _tokenizer_config(raw):
    config = strict_json(raw)
    if not isinstance(config, dict):
        raise MetadataError("Tokenizer configuration must be a JSON object")
    if config.get("auto_map") or config.get("trust_remote_code"):
        raise MetadataError("Repository tokenizer code is unsupported; native tokenizer JSON is required")
    if config.get("backend", "tokenizers") != "tokenizers":
        raise MetadataError("Tokenizer configuration requires an unsupported backend")
    # The class label is data. No import or attribute lookup uses this value.
    if config.get("tokenizer_class", "TokenizersBackend") not in (
            "TokenizersBackend", "PreTrainedTokenizerFast"):
        raise MetadataError("Only the reviewed native tokenizers configuration is supported")
    return config


def _captured_manifest(root, settings):
    source_path = reports_directory(root, settings) / "metadata-latest.json"
    analysis = architecture._analyze_source(root, settings, source_path)
    source, digest, identity = architecture._bounded_json(source_path, architecture.MAX_REPORT_BYTES, capture=True)
    if digest != analysis["source_report"] or source["baseline_comparison"]["matched"] is not True:
        raise MetadataError("Tokenizer requires unchanged metadata evidence with a matched baseline")
    if source.get("evidence") != "evidence/snapshot.json":
        raise MetadataError("Tokenizer requires the captured pinned model API manifest")
    snapshot_path = Path(analysis["catalogue_path"]).parent / "evidence/snapshot.json"
    _, snapshot_digest, snapshot_identity = architecture._bounded_json(snapshot_path, 1024**2, capture=True)
    evidence = OfflineMetadataSource(snapshot_path.parent, settings["model_id"], settings["revision"])
    model = strict_json(evidence.json_bytes("model"))
    config = strict_json(evidence.json_bytes("config"))
    _artifacts(model, settings["model_id"], settings["revision"])
    if config != source["config"]:
        raise MetadataError("Captured model config differs from latest verified metadata")
    if (architecture._fingerprint(source_path) != identity
            or architecture._fingerprint(snapshot_path) != snapshot_identity):
        raise MetadataError("Tokenizer metadata evidence changed while selecting artifacts")
    manifest = {"id": model["id"], "sha": model["sha"],
                "siblings": [item for item in model["siblings"] if item.get("rfilename") in ARTIFACTS]}
    return manifest, {"source_report": digest, "snapshot": snapshot_digest,
                      "catalogue": analysis["catalogue"], "source": str(source_path)}


def _download(source, model_id, revision, directory, artifact):
    destination = directory / artifact.name
    temporary = directory / (artifact.name + ".part-" + uuid.uuid4().hex)
    url = f"https://huggingface.co/{quote(model_id, safe='/')}/resolve/{revision}/{artifact.name}"
    sha256, git = _hashes(artifact)
    try:
        with source._open(url, {}) as response:
            if response.status != 200:
                raise MetadataError("Tokenizer downloads require HTTP 200 for the exact bounded JSON file")
            if source._length(response) != artifact.size:
                raise MetadataError("Tokenizer Content-Length differs from pinned manifest size")
            source.budget.ensure(artifact.size)
            with temporary.open("xb", buffering=0) as stream:
                remaining = artifact.size
                while remaining:
                    source._deadline()
                    raw = source.budget.read(response, min(remaining, MAX_READ_BYTES))
                    if not raw:
                        raise MetadataError("Tokenizer HTTP body was truncated")
                    remaining -= len(raw)
                    sha256.update(raw)
                    git.update(raw)
                    if stream.write(raw) != len(raw):
                        raise OSError("Tokenizer download could not write all bytes")
                stream.flush()
                os.fsync(stream.fileno())
        receipt = _receipt(artifact, sha256, git)
        # Publish by a same-directory hard link: atomic create, never replace an
        # existing file. This also closes the existence-check/rename race.
        try:
            os.link(temporary, destination)
            receipt["downloaded"] = True
        except FileExistsError:
            receipt, _ = _verify_file(destination, artifact)
            receipt["downloaded"] = False
        return receipt
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_tokenizer(root, settings, model_directory=None, *, online=False):
    """Verify the two pinned files, optionally obtaining missing files only.

    ``tokenizer_verified`` means artifact hashes/config were verified, independent
    of whether the optional native library is installed. Existing mismatched
    artifacts are never overwritten. Exceptions let the runtime command publish
    its normal ERROR report.
    """
    if type(online) is not bool:
        raise ValueError("Tokenizer online selection must be an explicit boolean")
    root = Path(root).resolve()
    validate_target(settings["model_id"], settings["revision"])
    manifest, provenance = _captured_manifest(root, settings)
    artifacts = _artifacts(manifest, settings["model_id"], settings["revision"])
    directory = Path(model_directory or settings["model_directory"])
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.absolute()
    if directory.exists():
        _directory(directory)
    receipts, missing = {}, []
    # Preflight all existing files before downloading anything.
    for artifact in artifacts:
        path = directory / artifact.name
        if path.exists() or path.is_symlink():
            receipt, _ = _verify_file(path, artifact)
            receipts[artifact.name] = {**receipt, "downloaded": False}
        else:
            missing.append(artifact)
    if missing and not online:
        raise FileNotFoundError("Missing tokenizer files: " + ", ".join(item.name for item in missing)
                                + "; run tokenizer-check --online for the two small JSON artifacts")
    source = None
    if missing:
        directory.mkdir(parents=True, exist_ok=True)
        _directory(directory)
        source = HttpMetadataSource(settings["model_id"], settings["revision"],
                                     limits=FetchLimits(total_body_bytes=MAX_TOKENIZER_BYTES))
        for artifact in missing:
            receipts[artifact.name] = _download(source, settings["model_id"], settings["revision"], directory, artifact)
    config_artifact = next(item for item in artifacts if item.name == "tokenizer_config.json")
    _, raw = _verify_file(directory / config_artifact.name, config_artifact, capture=True)
    config = _tokenizer_config(raw)
    return {"status": "PASS", "tokenizer_verified": True, "tokenizer_native_loaded": False,
            "model_id": settings["model_id"], "revision": settings["revision"],
            "model_directory": str(directory), "manifest": manifest, "provenance": provenance,
            "tokenizer_files": [receipts[name] for name in ARTIFACTS],
            "tokenizer_backend": config.get("backend", "tokenizers"),
            "verification_scope": "Pinned tokenizer artifact hashes and native configuration; no text/model parity claim",
            "online_requested": online, "network": source.stats() if source else None,
            "remote_code_executed": False, "tensor_payload_bytes_requested": 0,
            "full_checkpoint_downloaded": False, "max_artifact_read_bytes": MAX_READ_BYTES,
            **FULL_MODEL_FLAGS}


def _native_tokenizers():
    try:
        import tokenizers
    except ImportError as error:
        raise RuntimeError("Native tokenizer support needs tokenizers==0.23.2; run setup-reference.bat, "
                           "then use .venv-reference\\Scripts\\python.exe -m glm_local") from error
    if tokenizers.__version__ != EXPECTED_TOKENIZERS_VERSION:
        raise RuntimeError("Tokenizer library version differs from pinned tokenizers==0.23.2; "
                           "use setup-reference.bat and .venv-reference\\Scripts\\python.exe")
    return tokenizers.Tokenizer


class LocalTokenizer:
    """Plain text encode/decode through the verified JSON's native postprocessor."""

    def __init__(self, native, vocab_size, max_tokens, token_lengths):
        self._native, self.vocab_size, self.max_tokens = native, vocab_size, max_tokens
        self._token_lengths = token_lengths

    def _ids(self, ids):
        if isinstance(ids, (str, bytes, bytearray)) or not hasattr(ids, "__len__") or len(ids) > self.max_tokens:
            raise ValueError("Tokenizer IDs must be a bounded sequence")
        result = []
        for index in range(len(ids)):
            value = ids[index]
            if type(value) is not int or not 0 <= value < self.vocab_size or value not in self._token_lengths:
                raise ValueError("Tokenizer ID is outside the checkpoint vocabulary")
            result.append(value)
        return result

    def encode(self, text, add_special_tokens=True):
        if not isinstance(text, str) or len(text) > MAX_TEXT_BYTES:
            raise ValueError("Tokenizer text must be a bounded Unicode string")
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES or type(add_special_tokens) is not bool:
            raise ValueError("Tokenizer text exceeds 1 MiB or special-token option is not boolean")
        return self._ids(self._native.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids, skip_special_tokens=True):
        if type(skip_special_tokens) is not bool:
            raise ValueError("skip_special_tokens must be boolean")
        values = self._ids(ids)
        # Supported native decoders only remove/merge token text, add separators,
        # or repair invalid UTF-8. Bound their output before calling native code.
        if sum((self._token_lengths[value] + 1) * 4 for value in values) > 16 * MAX_TEXT_BYTES:
            raise ValueError("Decoded tokenizer text would exceed the 16-MiB output bound")
        result = self._native.decode(values, skip_special_tokens=skip_special_tokens)
        # The underlying vocabulary and token count are bounded. Check the output
        # size as well so callers cannot accidentally retain a very large string.
        if len(result) > 16 * MAX_TEXT_BYTES or len(result.encode("utf-8")) > 16 * MAX_TEXT_BYTES:
            raise ValueError("Decoded tokenizer text exceeds the 16-MiB output bound")
        return result


def load_tokenizer(model_directory, config, *, model_id, revision, manifest):
    """Load only verified local JSON using tokenizers; never fetch or execute repository code."""
    artifacts = _artifacts(manifest, model_id, revision)
    if not isinstance(config, dict) or type(config.get("vocab_size")) is not int or not 1 <= config["vocab_size"] <= MAX_VOCAB_SIZE:
        raise ValueError("Checkpoint vocabulary size exceeds the supported tokenizer bound")
    maximum = config.get("max_position_embeddings", MAX_TOKEN_IDS)
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Checkpoint token context must be a positive integer")
    maximum = min(maximum, MAX_TOKEN_IDS)
    directory = Path(model_directory).absolute()
    _directory(directory)
    raw = {}
    for artifact in artifacts:
        _, raw[artifact.name] = _verify_file(directory / artifact.name, artifact, capture=True)
    tokenizer_config = _tokenizer_config(raw["tokenizer_config.json"])
    tokenizer_json = strict_json(raw["tokenizer.json"])
    if not isinstance(tokenizer_json, dict):
        raise ValueError("Native tokenizer JSON must be an object")
    decoder = tokenizer_json.get("decoder")
    if decoder is not None and (not isinstance(decoder, dict) or decoder.get("type") not in (
            "ByteLevel", "WordPiece", "BPEDecoder", "Metaspace")):
        raise ValueError("Tokenizer decoder is outside the supported bounded native profile")
    del tokenizer_json
    Native = _native_tokenizers()
    native = Native.from_buffer(raw["tokenizer.json"])
    if native.padding is not None or native.truncation is not None:
        raise ValueError("Tokenizer JSON must not silently pad or truncate native prompt encoding")
    vocab = native.get_vocab(with_added_tokens=True)
    if not 1 <= len(vocab) <= config["vocab_size"] or len(vocab) != native.get_vocab_size(with_added_tokens=True):
        raise ValueError("Tokenizer vocabulary exceeds the checkpoint vocabulary")
    ids = list(vocab.values())
    if (any(type(value) is not int or not 0 <= value < config["vocab_size"] for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Tokenizer vocabulary contains out-of-range or duplicate IDs")
    token_lengths = {value: len(token.encode("utf-8")) for token, value in vocab.items()}
    if max(token_lengths.values()) > MAX_READ_BYTES:
        raise ValueError("Tokenizer vocabulary token text exceeds the 64-KiB policy")
    special_tokens = tokenizer_config.get("extra_special_tokens", [])
    if not isinstance(special_tokens, list) or len(special_tokens) > 1024:
        raise ValueError("Tokenizer special token list exceeds its bound")
    special_tokens = special_tokens + [tokenizer_config[key] for key in ("bos_token", "eos_token", "pad_token", "unk_token")
                                        if tokenizer_config.get(key) is not None]
    for token in special_tokens:
        if not isinstance(token, str) or len(token.encode("utf-8")) > 4096 or native.token_to_id(token) is None:
            raise ValueError("Configured special token is absent from the native tokenizer vocabulary")
    wrapper = LocalTokenizer(native, config["vocab_size"], maximum, token_lengths)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if config.get(key) is not None:
            wrapper._ids(config[key] if isinstance(config[key], list) else [config[key]])
    return wrapper
