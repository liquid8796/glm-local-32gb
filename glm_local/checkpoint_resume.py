"""Resume a bounded metadata audit using verified evidence and missing headers.

All cached JSON and headers are revalidated before the first network request.
The existing evidence directory is read only; the metadata executor copies the
validated inputs into a new run, including newly fetched prefix/header bytes.
"""
from __future__ import annotations

from .checkpoint_http import FetchLimits, HttpMetadataSource, MetadataError, strict_json
from .checkpoint_schema import validate_index, validate_manifest
from .checkpoint_snapshot import OfflineMetadataSource
from .safetensor_reader import parse_header_bytes


class ResumingMetadataSource:
    mode = "online_resume"

    def __init__(self, directory, model_id, revision, *, limits=None, opener=None):
        self.model_id, self.revision = model_id, revision
        self.limits = limits or FetchLimits()
        cached = OfflineMetadataSource(directory, model_id, revision, limits=self.limits)
        self.directory, self.budget = cached.directory, cached.budget
        self._remote = None
        self._opener = opener
        self._remote_bytes = 0
        self._remote_headers = 0
        self._cached_reused = set()
        self._json = {kind: cached.json_bytes(kind) for kind in ("model", "config", "index")}
        model, config, index = [strict_json(self._json[kind]) for kind in ("model", "config", "index")]
        self._sizes = validate_manifest(model_id, revision, model)
        if not isinstance(config, dict):
            raise MetadataError("Resume evidence config must be a JSON object")
        self._by_shard = validate_index(index, self._sizes)
        available = cached.available_shards()
        if not available <= set(self._by_shard):
            raise MetadataError("Resume evidence contains a header not referenced by the checkpoint index")
        self._headers = {}
        for name in sorted(available):
            raw = cached.header_bytes(name, self._sizes[name])
            tensors, _ = parse_header_bytes(raw[8:], self._sizes[name])
            if set(tensors) != self._by_shard[name]:
                raise MetadataError(f"Resume evidence index/header tensor set mismatch in {name}")
            self._headers[name] = raw
        self._cached_bytes = self.budget.bytes

    def json_bytes(self, kind):
        if kind not in self._json:
            raise MetadataError("Resume supports only cached model/config/index JSON")
        return self._json[kind]

    def header_bytes(self, name, file_size):
        if (not isinstance(name, str) or name not in self._by_shard
                or type(file_size) is not int or self._sizes[name] != file_size):
            raise MetadataError("Resume requested a header outside its verified checkpoint index/manifest")
        if name in self._headers:
            self._cached_reused.add(name)
            return self._headers[name]
        if self._remote is None:
            self._remote = HttpMetadataSource(self.model_id, self.revision,
                                               limits=self.limits, opener=self._opener)
            self._remote.budget = self.budget
        before = self.budget.bytes
        try:
            raw = self._remote.header_bytes(name, file_size)
        finally:
            self._remote_bytes += self.budget.bytes - before
        self._remote_headers += 1
        return raw

    def available_shards(self):
        return None  # Missing indexed headers can be fetched, unlike offline replay.

    def stats(self):
        remote = self._remote.stats() if self._remote is not None else {
            "requests": 0, "redirects": 0, "range_requests": 0,
            "response_hosts": [], "matching_repo_commit_headers": 0,
        }
        return {**remote, **self.budget.stats(), "mode": self.mode,
                "cached_body_bytes_read": self._cached_bytes,
                "remote_body_bytes_read": self._remote_bytes,
                "cached_headers_verified": len(self._headers),
                "cached_headers_reused": len(self._cached_reused),
                "remote_headers_fetched": self._remote_headers,
                "resumed_from": str(self.directory),
                "tensor_payload_bytes_requested": 0,
                "remote_provenance_authenticated": False,
                "counter_scope": ("Shared application read budget for cached snapshot/JSON/headers "
                                  "and remote prefix/header bodies; cached evidence is not remotely reauthenticated")}
