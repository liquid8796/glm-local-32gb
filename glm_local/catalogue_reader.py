"""Read only selected local shards bound to captured metadata, with <=2 open files.

The index has already been checked by the metadata audit policy. No legacy reader
limit changes, downloads, eager payload reads, mmap or whole matrix allocation.
"""

from collections import OrderedDict
import hashlib
from pathlib import Path
import stat
import struct
from types import MappingProxyType

from .safetensor_reader import (MAX_HEADER_BYTES, MAX_READ_BYTES, SafeTensorError,
                                SafeTensorReader, parse_header_bytes)


class _BoundHeaderReader(SafeTensorReader):
    def __init__(self, path, proof):
        self._proof = proof
        super().__init__(path)

    def _parse_header(self):
        if self._file_size != self._proof.file_size:
            raise SafeTensorError("Selected shard file size differs from captured metadata")
        prefix = self._read_at(0, 8)
        length = struct.unpack("<Q", prefix)[0]
        if (not 1 <= length <= MAX_HEADER_BYTES or length + 8 != self._proof.header_bytes
                or length + 8 > self._file_size):
            raise SafeTensorError("Selected shard header length differs from captured metadata")
        header, digest = bytearray(length), hashlib.sha256(prefix)
        for offset in range(0, length, MAX_READ_BYTES):
            raw = self._read_at(8 + offset, min(MAX_READ_BYTES, length - offset))
            header[offset:offset + len(raw)] = raw
            digest.update(raw)
        if digest.hexdigest() != self._proof.header_sha256:
            raise SafeTensorError("Selected shard header SHA-256 differs from captured metadata")
        self._header_bytes, self._payload_start = length, length + 8
        self._tensors, self._metadata = parse_header_bytes(header, self._file_size)


class SelectedCatalogueReader:
    """Read an immutable selected-projection descriptor against local full-size shards."""

    def __init__(self, model_directory, descriptor, *, max_open_shards=2):
        if type(max_open_shards) is not int or not 1 <= max_open_shards <= 2:
            raise ValueError("Selected reader permits one or two open shards")
        self.directory = Path(model_directory).absolute()
        directory_info = self.directory.lstat()
        if not stat.S_ISDIR(directory_info.st_mode) or getattr(directory_info, "st_file_attributes", 0) & 0x400:
            raise SafeTensorError("Selected shard directory must be a regular directory")
        self._descriptor = descriptor
        self._proofs = {shard.name: shard for shard in descriptor.shards}
        self._selected = {item.name: item for item in getattr(descriptor, "tensors", (descriptor.weight, descriptor.scale))}
        self._tensors = MappingProxyType({name: item.info() for name, item in self._selected.items()})
        self._open = OrderedDict()
        self._max_open = max_open_shards
        self._closed = False
        self._identities = {}
        self._totals = {"actual_read_bytes": 0, "actual_read_calls": 0,
                        "tensor_read_bytes": 0, "tensor_read_calls": 0,
                        "max_actual_read_bytes": 0}
        self._opens = self._evictions = self._peak_open = 0
        try:
            # Validate every selected file/header before allowing the first payload read.
            for name in self._proofs:
                self._reader(name)
        except BaseException:
            self.close()
            raise

    @property
    def tensors(self):
        return self._tensors

    def _identity(self, name):
        path = self.directory / name
        if path.parent != self.directory:
            raise SafeTensorError("Selected shard path must be flat")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise SafeTensorError("Selected shard must be a regular file, not a link")
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _retire(self, reader):
        counts = reader.stats()
        for key in self._totals:
            self._totals[key] = (max(self._totals[key], counts[key]) if key == "max_actual_read_bytes"
                                 else self._totals[key] + counts[key])
        reader.close()

    def _reader(self, name):
        if self._closed:
            raise SafeTensorError("Selected catalogue reader is closed")
        identity = self._identity(name)
        if name in self._identities and self._identities[name] != identity:
            raise SafeTensorError("Selected shard changed since validation")
        if name in self._open:
            reader = self._open.pop(name)
            reader._assert_unchanged()
            self._open[name] = reader
            return reader
        if len(self._open) >= self._max_open:
            _, old = self._open.popitem(last=False)
            self._retire(old)
            self._evictions += 1
        reader = _BoundHeaderReader(self.directory / name, self._proofs[name])
        try:
            if self._identity(name) != identity:
                raise SafeTensorError("Selected shard changed while validating")
            for item in self._selected.values():
                if item.shard == name and reader.tensors.get(item.name) != item.info():
                    raise SafeTensorError("Selected tensor differs from descriptor metadata")
        except BaseException:
            reader.close()
            raise
        self._identities[name] = identity
        self._open[name] = reader
        self._opens += 1
        self._peak_open = max(self._peak_open, len(self._open))
        return reader

    def read_bytes(self, name, offset, count):
        item = self._selected[name]
        return self._reader(item.shard).read_bytes(name, offset, count)

    def read_matrix_tile(self, name, row, col, rows, cols):
        item = self._selected[name]
        return self._reader(item.shard).read_matrix_tile(name, row, col, rows, cols)

    def stats(self):
        counts = dict(self._totals)
        for reader in self._open.values():
            current = reader.stats()
            for key in counts:
                counts[key] = max(counts[key], current[key]) if key == "max_actual_read_bytes" else counts[key] + current[key]
        return {**counts, "open_shards": len(self._open), "peak_open_shards": self._peak_open,
                "shard_opens": self._opens, "lru_evictions": self._evictions,
                "selected_shards": len(self._proofs), "selected_tensors": len(self._selected),
                "policy_max_open_shards": self._max_open, "policy_max_read_bytes": MAX_READ_BYTES,
                "selected_header_digest_verified": True, "whole_payload_checksum_verified": False,
                "legacy_reader_limits_changed": False}

    def close(self):
        for reader in self._open.values():
            self._retire(reader)
        self._open.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise SafeTensorError("Selected catalogue reader is closed")
        return self

    def __exit__(self, *_):
        self.close()
