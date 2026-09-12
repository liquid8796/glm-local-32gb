"""Small, explicit selected-tensor downloads for projection verification.

This command's payload transport is separate from the metadata-only workflow.
Only the two descriptor tensors are downloaded; 206 responses and live header
identity are mandatory. Raw slices are local run artifacts, never full shards.
"""

from contextlib import ExitStack
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
from types import MappingProxyType
from urllib.parse import quote

from .checkpoint_http import FetchLimits, HttpMetadataSource, MetadataError
from .checkpoint_snapshot import write_json
from .safetensor_reader import MAX_READ_BYTES

MAX_PROJECTION_BYTES = 128 * 1024**2
RANGE_BYTES = 4 * 1024**2


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class RemoteProjectionReader:
    """Download a bounded pair, then serve tensor-relative reads from disk.

    Descriptor construction validates captured metadata before this is called.
    A newly fetched header must hash exactly to that captured header. Payload
    hashes record the bytes observed here; they are not full-shard LFS hashes.
    """

    def __init__(self, descriptor, directory, *, budget_bytes=64 * 1024**2, _transport=None):
        if type(budget_bytes) is not int or not 1 <= budget_bytes <= MAX_PROJECTION_BYTES:
            raise ValueError("Projection download budget must be 1..128 MiB")
        tensors = getattr(descriptor, "tensors", (descriptor.weight, descriptor.scale))
        if not 2 <= len(tensors) <= 4 or len({t.name for t in tensors}) != len(tensors):
            raise ValueError("Selected projection requires two to four distinct tensors")
        total = sum(t.nbytes for t in tensors)
        needed = total + sum(shard.header_bytes for shard in descriptor.shards)
        if needed > budget_bytes:
            raise ValueError(f"Selected tensors and headers require {needed} bytes; budget is {budget_bytes}")
        self._stack = ExitStack()
        self._files = {}
        self._open = OrderedDict()
        self._closed = False
        self._tensors = MappingProxyType({t.name: t.info() for t in tensors})
        self._stats = {"tensor_read_bytes": 0, "tensor_read_calls": 0,
                       "max_actual_read_bytes": 0, "max_open_shards_observed": 0}
        self.directory = Path(directory).absolute()
        self.directory.mkdir(parents=True, exist_ok=False)
        source = _transport or HttpMetadataSource(descriptor.model_id, descriptor.revision,
                      limits=FetchLimits(total_body_bytes=budget_bytes))
        self._source = source
        self._receipt = {"scope": "selected_real_projection_payload", "complete": False,
                         "model_id": descriptor.model_id, "revision": descriptor.revision,
                         "selected_tensor_payload_bytes": total, "full_checkpoint_downloaded": False,
                         "payload_hash_scope": "Observed selected tensor bytes only; not full-shard hash verification",
                         "tensors": [], "live_headers": []}
        try:
            shards = {shard.name: shard for shard in descriptor.shards}
            urls, etags = {}, {}
            for name, shard in shards.items():
                url = f"https://huggingface.co/{quote(descriptor.model_id, safe='/')}/resolve/{descriptor.revision}/{name}"
                raw, etag = source._range(url, 0, shard.header_bytes, shard.file_size)
                if hashlib.sha256(raw).hexdigest() != shard.header_sha256:
                    raise MetadataError("Live shard header differs from captured projection evidence")
                urls[name], etags[name] = url, etag
                self._receipt["live_headers"].append({"name": name, "sha256": shard.header_sha256,
                                                      "bytes": len(raw), "matched": True})
            for number, tensor in enumerate(tensors):
                shard = shards[tensor.shard]
                path = self.directory / f"tensor-{number}.bin"
                digest = hashlib.sha256()
                with path.open("xb", buffering=0) as stream:
                    for offset in range(0, tensor.nbytes, RANGE_BYTES):
                        count = min(RANGE_BYTES, tensor.nbytes - offset)
                        raw, current_etag = source._range(urls[tensor.shard],
                            shard.header_bytes + tensor.data_offsets[0] + offset,
                            count, shard.file_size, etags[tensor.shard])
                        if etags[tensor.shard] is None:
                            etags[tensor.shard] = current_etag
                        if len(raw) != count:
                            raise MetadataError("Selected tensor download returned an incorrect byte count")
                        digest.update(raw)
                        for start in range(0, len(raw), MAX_READ_BYTES):
                            part = raw[start:start + MAX_READ_BYTES]
                            if stream.write(part) != len(part):
                                raise OSError("Short write of selected tensor slice")
                    stream.flush()
                before = _identity(path.stat())
                self._files[tensor.name] = (path, before)
                self._receipt["tensors"].append({"name": tensor.name, "file": path.name,
                    "bytes": tensor.nbytes, "sha256": digest.hexdigest(),
                    "original_shard": tensor.shard,
                    "original_data_offsets": list(tensor.data_offsets)})
            self._receipt["complete"] = True
            write_json(self.directory / "receipt.json", self._receipt)
        except BaseException:
            self.close()
            raise

    @property
    def tensors(self):
        return self._tensors

    def read_bytes(self, name, offset, count):
        if self._closed:
            raise ValueError("Selected projection reader is closed")
        tensor = self.tensors[name]
        if (type(offset) is not int or type(count) is not int or offset < 0
                or not 0 <= count <= MAX_READ_BYTES or offset + count > tensor.nbytes
                or offset % tensor.itemsize or count % tensor.itemsize):
            raise ValueError("Selected tensor read exceeds bounds")
        path, identity = self._files[name]
        stream = self._open.pop(name, None)
        if stream is None:
            if len(self._open) >= 2:
                _, retired = self._open.popitem(last=False)
                retired.close()
            stream = path.open("rb", buffering=0)
        self._open[name] = stream
        self._stats["max_open_shards_observed"] = max(self._stats["max_open_shards_observed"], len(self._open))
        if _identity(path.stat()) != identity or _identity(os.fstat(stream.fileno())) != identity:
            raise MetadataError("Selected tensor slice changed during verification")
        stream.seek(offset)
        raw = stream.read(count)
        if (len(raw) != count or _identity(os.fstat(stream.fileno())) != identity
                or _identity(path.stat()) != identity):
            raise MetadataError("Selected tensor slice truncated or changed")
        self._stats["tensor_read_bytes"] += len(raw)
        self._stats["tensor_read_calls"] += 1
        self._stats["max_actual_read_bytes"] = max(self._stats["max_actual_read_bytes"], len(raw))
        return raw

    def read_matrix_tile(self, name, row_start, col_start, rows, cols):
        tensor = self.tensors[name]
        if (len(tensor.shape) != 2 or any(type(x) is not int for x in (row_start, col_start, rows, cols))
                or min(row_start, col_start) < 0 or not 1 <= rows <= 128 or not 1 <= cols <= 128
                or row_start + rows > tensor.shape[0] or col_start + cols > tensor.shape[1]
                or rows * cols * tensor.itemsize > MAX_READ_BYTES):
            raise ValueError("Selected tensor tile exceeds bounds")
        return b"".join(self.read_bytes(name,
            ((row_start + row) * tensor.shape[1] + col_start) * tensor.itemsize,
            cols * tensor.itemsize) for row in range(rows))

    def stats(self):
        network = self._source.stats()
        network.update(tensor_payload_bytes_requested=self._receipt["selected_tensor_payload_bytes"],
                       counter_scope="Selected tensor payload and live header body reads")
        return {**self._stats, "network": network, "payload_receipt": self._receipt,
                "persistent_decoded_weight_bytes": 0, "max_transport_chunk_bytes": RANGE_BYTES}

    def close(self):
        self._closed = True
        for stream in self._open.values():
            stream.close()
        self._open.clear()
        self._stack.close()

    def __enter__(self):
        if self._closed:
            raise ValueError("Selected projection reader is closed")
        return self

    def __exit__(self, *_):
        self.close()
