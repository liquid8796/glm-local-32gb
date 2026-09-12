"""Bounded public Hugging Face metadata transport; never download tensor payload.

No SDK, token discovery, remote Python, retry, or whole-file Range fallback.
Counters bound application body reads, not TLS buffers/network framing/total RAM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request

from . import __version__
from .safetensor_reader import MAX_FILE_BYTES, MAX_HEADER_BYTES, MAX_READ_BYTES

JSON_LIMITS = {"model": 8 * 1024**2, "config": 1024**2, "index": 32 * 1024**2}
MAX_SHARDS = 512
MAX_TENSORS = 262144


class MetadataError(ValueError):
    """Metadata is unavailable, unsafe, inconsistent, or outside local policy."""


def strict_json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MetadataError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject(value):
        raise MetadataError(f"Nonfinite JSON constant: {value}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise MetadataError(f"Invalid metadata JSON: {error}") from None


def validate_target(model_id, revision):
    if (not isinstance(model_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id)
            or any(part in (".", "..") for part in model_id.split("/"))):
        raise MetadataError("Expected a Hugging Face owner/model identifier")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise MetadataError("A pinned lowercase 40-character revision is required")


def shard_filename(name):
    if (not isinstance(name, str) or len(name) > 128
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors", name)
            or name.split(".", 1)[0].upper() in {
                "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}):
        raise MetadataError("Checkpoint requires portable flat .safetensors shard names")
    return name


def header_length(prefix, file_size):
    if not isinstance(prefix, bytes) or len(prefix) != 8:
        raise MetadataError("Safetensors prefix must contain exactly 8 bytes")
    length = struct.unpack("<Q", prefix)[0]
    if (type(file_size) is not int or not 8 <= file_size <= MAX_FILE_BYTES
            or not 1 <= length <= MAX_HEADER_BYTES or length > file_size - 8):
        raise MetadataError("Safetensors header exceeds the 1-MiB/file-size policy")
    return length


@dataclass(frozen=True)
class FetchLimits:
    total_body_bytes: int = 64 * 1024**2
    max_requests: int = 4096  # Includes redirects; no retries.
    timeout_seconds: int = 30
    total_seconds: int = 1800
    max_redirects: int = 5

    def __post_init__(self):
        values = asdict(self)
        ceilings = dict(total_body_bytes=128 * 1024**2, max_requests=8192,
                        timeout_seconds=60, total_seconds=3600, max_redirects=5)
        for key, value in values.items():
            minimum = 0 if key == "max_redirects" else 1
            if type(value) is not int or not minimum <= value <= ceilings[key]:
                raise MetadataError(f"Invalid bounded fetch limit: {key}")


class ReadBudget:
    def __init__(self, limit):
        self.limit = limit
        self.bytes = self.calls = self.largest_read = 0

    def ensure(self, count):
        if count < 0 or count > self.limit - self.bytes:
            raise MetadataError("Metadata application-body read budget exhausted")

    def read(self, stream, count):
        if not 1 <= count <= MAX_READ_BYTES:
            raise MetadataError("Metadata read must be within 1..64 KiB")
        self.ensure(count)
        raw = stream.read(count)
        if not isinstance(raw, bytes) or len(raw) > count:
            raise MetadataError("Transport returned an invalid bounded read")
        self.bytes += len(raw)
        self.calls += 1
        self.largest_read = max(self.largest_read, len(raw))
        return raw

    def stats(self):
        return {"body_bytes_read": self.bytes, "read_calls": self.calls,
                "max_actual_read_bytes": self.largest_read, "budget_bytes": self.limit}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The standard redirect handler consumes the response body. Returning
        # None gives the caller an HTTPError which it closes without reading.
        return None


def _trusted_url(url):
    try:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        raise MetadataError("Invalid metadata redirect URL") from None
    if (parsed.scheme != "https" or parsed.username is not None or parsed.password is not None
            or port not in (None, 443) or parsed.fragment
            or not any(host == suffix or host.endswith("." + suffix)
                       for suffix in ("huggingface.co", "hf.co"))):
        raise MetadataError("Metadata redirect must remain HTTPS on Hugging Face/CDN hosts")
    return host


class HttpMetadataSource:
    """Sequential strict Range transport for one pinned public repository."""
    mode = "online"

    def __init__(self, model_id, revision, *, limits=None, opener=None):
        validate_target(model_id, revision)
        self.model_id, self.revision = model_id, revision
        self.limits = limits or FetchLimits()
        self.budget = ReadBudget(self.limits.total_body_bytes)
        self._opener = opener or urllib.request.build_opener(_NoRedirect())
        self._started = time.monotonic()
        self._requests = self._redirects = self._range_requests = 0
        self._hosts = set()
        self._commit_headers = 0

    def _deadline(self):
        remaining = self.limits.total_seconds - (time.monotonic() - self._started)
        if remaining <= 0:
            raise MetadataError("Metadata run exceeded its elapsed-time budget")
        return min(self.limits.timeout_seconds, remaining)

    def _check_commit(self, headers):
        commit = headers.get("X-Repo-Commit")
        if commit is not None:
            if commit != self.revision:
                raise MetadataError("Response X-Repo-Commit differs from the pinned revision")
            self._commit_headers += 1

    def _open(self, url, extra_headers):
        for redirect in range(self.limits.max_redirects + 1):
            self._hosts.add(_trusted_url(url))
            if self._requests >= self.limits.max_requests:
                raise MetadataError("Metadata request budget exhausted")
            self._requests += 1
            headers = {"User-Agent": f"glm-local-metadata/{__version__}",
                       "Accept-Encoding": "identity", **extra_headers}
            request = urllib.request.Request(url, headers=headers)
            try:
                response = self._opener.open(request, timeout=self._deadline())
            except urllib.error.HTTPError as error:
                try:
                    self._check_commit(error.headers)
                    if error.code not in (301, 302, 303, 307, 308):
                        raise MetadataError(f"Metadata HTTP {error.code}; no retry or revision fallback")
                    location = error.headers.get("Location")
                    if not location or redirect == self.limits.max_redirects:
                        raise MetadataError("Missing redirect location or redirect limit exceeded")
                    url = urllib.parse.urljoin(url, location)
                    _trusted_url(url)
                    self._redirects += 1
                finally:
                    error.close()  # Never consume a redirect/error body.
                continue
            except (OSError, urllib.error.URLError) as error:
                # Do not retain signed redirect URLs or credentials in reports.
                raise MetadataError(f"Metadata connection failed ({type(error).__name__}); "
                                    "check DNS, HTTPS access and the configured revision") from None
            try:
                self._check_commit(response.headers)
                self._hosts.add(_trusted_url(response.geturl()))
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise MetadataError("Compressed metadata response is not supported")
            except BaseException:
                response.close()
                raise
            return response
        raise MetadataError("Metadata redirect limit exceeded")

    def _exact(self, response, count):
        self.budget.ensure(count)
        parts = []
        while count:
            self._deadline()
            part = self.budget.read(response, min(count, MAX_READ_BYTES))
            if not part:
                raise MetadataError("Truncated metadata response")
            parts.append(part)
            count -= len(part)
        return b"".join(parts)

    def _length(self, response):
        raw = response.headers.get("Content-Length")
        if raw is None:
            return None
        if not re.fullmatch(r"[0-9]{1,20}", raw):
            raise MetadataError("Invalid response Content-Length")
        return int(raw)

    def json_bytes(self, kind):
        if kind not in JSON_LIMITS:
            raise MetadataError("Only model/config/index JSON may be fetched")
        repo = urllib.parse.quote(self.model_id, safe="/")
        if kind == "model":
            url = f"https://huggingface.co/api/models/{repo}/revision/{self.revision}?blobs=true"
        else:
            filename = "config.json" if kind == "config" else "model.safetensors.index.json"
            url = f"https://huggingface.co/{repo}/resolve/{self.revision}/{filename}"
        cap = JSON_LIMITS[kind]
        with self._open(url, {}) as response:
            if response.status != 200:
                raise MetadataError(f"JSON metadata requires HTTP 200 ({kind})")
            size = self._length(response)
            if size is not None:
                if not 1 <= size <= cap:
                    raise MetadataError(f"JSON {kind} exceeds its bounded size policy")
                raw = self._exact(response, size)
            else:
                pieces, received = [], 0
                while True:
                    self._deadline()
                    part = self.budget.read(response, min(MAX_READ_BYTES, cap + 1 - received))
                    if not part:
                        break
                    pieces.append(part)
                    received += len(part)
                    if received > cap:
                        raise MetadataError(f"JSON {kind} exceeds its bounded size policy")
                raw = b"".join(pieces)
        strict_json(raw)
        return raw

    def _range(self, url, start, count, file_size, etag=None):
        self._range_requests += 1
        extra = {"Range": f"bytes={start}-{start + count - 1}"}
        if etag and not etag.startswith("W/"):
            extra["If-Match"] = etag
        with self._open(url, extra) as response:
            if response.status != 206:
                raise MetadataError(f"Shard metadata requires HTTP 206; got HTTP {response.status}; refused whole-file fallback")
            value = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes ([0-9]{1,20})-([0-9]{1,20})/([0-9]{1,20})", value)
            if not match or tuple(map(int, match.groups())) != (start, start + count - 1, file_size):
                raise MetadataError(f"Shard Content-Range does not match requested bytes/manifest size: "
                                    f"got {value[:160]!r}, expected bytes {start}-{start + count - 1}/{file_size}")
            length = self._length(response)
            if length is not None and length != count:
                raise MetadataError("Shard Content-Length does not match requested Range")
            current_etag = response.headers.get("ETag")
            if current_etag is not None and (len(current_etag) > 512
                                            or any(ord(c) < 32 for c in current_etag)):
                raise MetadataError("Invalid shard ETag")
            if etag is not None and current_etag != etag:
                raise MetadataError("Shard ETag changed or disappeared between prefix/header reads")
            return self._exact(response, count), current_etag

    def header_bytes(self, name, file_size):
        name = shard_filename(name)
        if type(file_size) is not int or not 8 <= file_size <= MAX_FILE_BYTES:
            raise MetadataError("Shard size exceeds the local file-size policy")
        repo = urllib.parse.quote(self.model_id, safe="/")
        url = f"https://huggingface.co/{repo}/resolve/{self.revision}/{name}"
        prefix, etag = self._range(url, 0, 8, file_size)
        length = header_length(prefix, file_size)
        header, _ = self._range(url, 8, length, file_size, etag)
        return prefix + header

    def available_shards(self):
        return None  # Online mode may inspect every indexed shard.

    def stats(self):
        return {**self.budget.stats(), "mode": self.mode, "requests": self._requests,
                "redirects": self._redirects, "range_requests": self._range_requests,
                "response_hosts": sorted(self._hosts),
                "matching_repo_commit_headers": self._commit_headers,
                "tensor_payload_bytes_requested": 0,
                "limits": asdict(self.limits),
                "counter_scope": "Application metadata body reads; excludes HTTP/TLS buffers"}
