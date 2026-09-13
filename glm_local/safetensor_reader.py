"""Bounded, read-only safetensors subset; no mmap or eager tensor loading.

The 1 MiB header, 2 TiB file, 4096 tensor, rank-eight and metadata limits below
are local policy, not safetensors format restrictions. Only byte-aligned dtypes
are supported. Every actual file read and returned tile is at most 64 KiB.
The explicit read_span API returns an owned buffer of at most 8 MiB, only after
validating the complete logical read. On Windows its first use upgrades the
reader to a share-read-only handle which denies writes/deletes until close.
These allocation bounds are not a claim about total process or system RAM.

Unprotected files have identity, size and timestamps checked around each read.
Protected Windows handles instead enforce immutability through sharing rules;
their original identity and header are revalidated when acquiring the handle.
Neither approach is cryptographic payload verification: safetensors embeds no
payload checksum. Reader instances are intended for a single owning thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from types import MappingProxyType
from typing import BinaryIO, Mapping


MAX_READ_BYTES = 65536
MAX_READ_SPAN_BYTES = 8 * 1024**2
MAX_HEADER_BYTES = 1024 * 1024
MAX_FILE_BYTES = 2 * 1024**4
MAX_TENSORS = 4096
MAX_RANK = 8
MAX_DIMENSION = 2**31 - 1
MAX_INTEGER = 2**63 - 1
MAX_TILE_EDGE = 128
MAX_METADATA_ENTRIES = 1024
MAX_METADATA_TEXT_BYTES = 256 * 1024
DTYPE_ITEMSIZE = MappingProxyType({
    "F8_E4M3": 1, "F32": 4, "BF16": 2, "F16": 2, "F64": 8,
    "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
    "I64": 8, "U64": 8, "BOOL": 1,
})
_WINDOWS_READ_API = None


def _open_protected_read(path):
    """Open one Windows read handle that denies competing writes and deletes."""
    import ctypes
    from ctypes import wintypes
    import msvcrt

    global _WINDOWS_READ_API
    if _WINDOWS_READ_API is None:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes, close.restype = [wintypes.HANDLE], wintypes.BOOL
        _WINDOWS_READ_API = create, close
    create, close = _WINDOWS_READ_API
    handle = create(str(path), 0x80000000, 0x00000001, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise
    try:
        return os.fdopen(descriptor, "rb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


class SafeTensorError(ValueError):
    """Invalid/unsupported safetensors data, invalid read, or changed file."""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    nbytes: int
    itemsize: int


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise SafeTensorError(f"Duplicate JSON key: {name!r}")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise SafeTensorError(f"Non-finite JSON constant: {value}")


def _uint(value: object, maximum: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise SafeTensorError(f"{label} must be an integer from 0 to {maximum}")
    return value


def _fingerprint(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


class SafeTensorReader:
    """Open and validate a local regular file; close with a context manager.

    ``read_bytes`` offsets are relative to one tensor. Tensor metadata and
    general metadata are immutable mappings. ``stats()`` returns a fresh dict
    and remains available after close. Opening reads only the bounded header.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).absolute()
        self._stream: BinaryIO | None = None
        self._span_protected = False
        self._tensors: Mapping[str, TensorInfo] = MappingProxyType({})
        self._metadata: Mapping[str, str] = MappingProxyType({})
        self._counters = {
            "actual_read_bytes": 0,
            "actual_read_calls": 0,
            "max_actual_read_bytes": 0,
            "tensor_read_bytes": 0,
            "tensor_read_calls": 0,
            "identity_checks": 0,
            "protected_immutable_checks": 0,
            "span_read_calls": 0,
            "span_read_bytes": 0,
            "span_inner_read_calls": 0,
            "max_span_bytes": 0,
            "protected_handle_upgrades": 0,
        }
        try:
            # Check before open to reject FIFO/device paths without blocking.
            initial = self.path.stat()
            if not stat.S_ISREG(initial.st_mode):
                raise SafeTensorError("Safetensors input must be a regular file")
            if not 8 <= initial.st_size <= MAX_FILE_BYTES:
                raise SafeTensorError("File size outside local 8-byte to 2-TiB policy")
            self._stream = open(self.path, "rb", buffering=0)
            # Windows stat and fstat may expose different ctime meanings; retain
            # separate baselines while cross-checking stable identity fields.
            descriptor = os.fstat(self._stream.fileno())
            if (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns) != (
                    descriptor.st_dev, descriptor.st_ino, descriptor.st_size, descriptor.st_mtime_ns):
                raise SafeTensorError("Safetensors file changed while opening")
            self._path_identity = _fingerprint(initial)
            self._descriptor_identity = _fingerprint(descriptor)
            self._file_size = initial.st_size
            self._assert_unchanged()
            self._parse_header()
            self._assert_unchanged()
        except BaseException as error:
            self.close()
            if isinstance(error, OSError):
                raise SafeTensorError(f"Cannot read safetensors file: {error}") from error
            raise

    @property
    def tensors(self) -> Mapping[str, TensorInfo]:
        return self._tensors

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    @property
    def protected_immutable(self) -> bool:
        """Whether this live handle denies Windows file writes and deletion."""
        return self._span_protected and self._stream is not None and not self._stream.closed

    def __enter__(self) -> SafeTensorReader:
        self._assert_unchanged()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._span_protected = False

    def _assert_unchanged(self) -> None:
        if self._stream is None or self._stream.closed:
            raise SafeTensorError("Safetensors reader is closed")
        if self.protected_immutable:
            self._counters["protected_immutable_checks"] += 1
            return
        self._counters["identity_checks"] += 1
        try:
            descriptor = os.fstat(self._stream.fileno())
            current_path = self.path.stat()
        except OSError as error:
            raise SafeTensorError("Safetensors file changed or became unavailable") from error
        if (_fingerprint(descriptor) != self._descriptor_identity
                or _fingerprint(current_path) != self._path_identity):
            raise SafeTensorError("Safetensors file changed since opening")

    def _protect_span_handle(self):
        if os.name != "nt" or self._span_protected:
            return
        try:
            self._assert_unchanged()
            assert self._stream is not None
            # Close before reopening to preserve the catalogue's open-handle bound.
            # Never refresh the original identities: replacement in this window fails.
            self._stream.close()
            self._stream = None
            self._stream = _open_protected_read(self.path)
            self._assert_unchanged()
            # The old handle was closed. Revalidate the bounded header under
            # the new lock, retaining both original identity baselines.
            tensors, metadata, header_sha256 = self._tensors, self._metadata, self._header_sha256
            self._parse_header()
            self._assert_unchanged()
            if (self._tensors != tensors or self._metadata != metadata
                    or self._header_sha256 != header_sha256):
                raise SafeTensorError("Safetensors header changed while acquiring protected access")
        except BaseException as error:
            self.close()
            if isinstance(error, OSError):
                raise SafeTensorError("Cannot acquire protected span read access; file changed or is open for writing") from error
            raise
        self._span_protected = True
        self._counters["protected_handle_upgrades"] += 1

    def _read_at(self, offset: int, count: int, *, tensor: bool = False) -> bytes:
        if not 1 <= count <= MAX_READ_BYTES:
            raise SafeTensorError("Internal read exceeds the bounded I/O policy")
        self._assert_unchanged()
        assert self._stream is not None
        try:
            self._stream.seek(offset)
            data = self._stream.read(count)
        except OSError as error:
            raise SafeTensorError("Safetensors bounded read failed") from error
        self._counters["actual_read_calls"] += 1
        self._counters["actual_read_bytes"] += len(data)
        self._counters["max_actual_read_bytes"] = max(
            self._counters["max_actual_read_bytes"], len(data))
        if tensor:
            self._counters["tensor_read_bytes"] += len(data)
            self._counters["tensor_read_calls"] += 1
        self._assert_unchanged()
        if len(data) != count:
            raise SafeTensorError("Truncated safetensors file during bounded read")
        return data

    def _parse_header(self) -> None:
        prefix = self._read_at(0, 8)
        length = struct.unpack("<Q", prefix)[0]
        if not 1 <= length <= MAX_HEADER_BYTES:
            raise SafeTensorError("Header length outside local 1-byte to 1-MiB policy")
        if length > self._file_size - 8:
            raise SafeTensorError("Header length exceeds file size")
        self._header_bytes = length
        self._payload_start = 8 + length
        header = bytearray(length)
        for offset in range(0, length, MAX_READ_BYTES):
            chunk = min(MAX_READ_BYTES, length - offset)
            header[offset:offset + chunk] = self._read_at(8 + offset, chunk)
        self._tensors, self._metadata = parse_header_bytes(header, self._file_size)
        digest = hashlib.sha256(prefix)
        digest.update(header)
        self._header_sha256 = digest.hexdigest()

    @staticmethod
    def _validate_metadata(metadata: object) -> None:
        if not isinstance(metadata, dict):
            raise SafeTensorError("__metadata__ must be a string-to-string object")
        if len(metadata) > MAX_METADATA_ENTRIES:
            raise SafeTensorError("Too many metadata entries for local policy")
        text_bytes = 0
        for name, value in metadata.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise SafeTensorError("__metadata__ must contain only strings")
            try:
                text_bytes += len(name.encode("utf-8")) + len(value.encode("utf-8"))
            except UnicodeError as error:
                raise SafeTensorError("Metadata contains invalid Unicode") from error
        if text_bytes > MAX_METADATA_TEXT_BYTES:
            raise SafeTensorError("Metadata text exceeds local 256-KiB policy")

    @staticmethod
    def _validate_tensor(name: str, entry: object, payload_bytes: int) -> TensorInfo:
        try:
            name.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise SafeTensorError("Tensor name contains invalid Unicode") from error
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise SafeTensorError(f"Tensor {name!r} requires exactly dtype/shape/data_offsets")
        dtype = entry["dtype"]
        if not isinstance(dtype, str) or dtype not in DTYPE_ITEMSIZE:
            raise SafeTensorError(f"Unsupported tensor dtype for {name!r}: {dtype!r}")
        raw_shape = entry["shape"]
        if not isinstance(raw_shape, list) or len(raw_shape) > MAX_RANK:
            raise SafeTensorError(f"Tensor {name!r} shape must be a list of rank at most eight")
        shape = tuple(_uint(dim, MAX_DIMENSION, "Shape dimension") for dim in raw_shape)
        elements = 0 if 0 in shape else 1
        if elements:
            for dim in shape:
                if elements > MAX_INTEGER // dim:
                    raise SafeTensorError("Tensor element count overflows signed 64-bit policy")
                elements *= dim
        itemsize = DTYPE_ITEMSIZE[dtype]
        if elements > MAX_INTEGER // itemsize:
            raise SafeTensorError("Tensor byte count overflows signed 64-bit policy")
        offsets = entry["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise SafeTensorError("data_offsets must be a list of two integers")
        start, end = (_uint(value, MAX_INTEGER, "Data offset") for value in offsets)
        if end < start or end > payload_bytes:
            raise SafeTensorError("Tensor offsets are reversed or exceed file payload")
        nbytes = elements * itemsize
        if end - start != nbytes:
            raise SafeTensorError("Tensor byte length does not match dtype and shape")
        return TensorInfo(dtype, shape, (start, end), nbytes, itemsize)

    def _tensor(self, name: str) -> TensorInfo:
        if not isinstance(name, str):
            raise SafeTensorError("Tensor name must be a string")
        return self._tensors[name]

    def read_bytes(self, name: str, offset: int, count: int) -> bytes:
        """Read <=64 KiB; offsets/counts must align with the tensor itemsize."""
        tensor = self._tensor(name)
        offset = _uint(offset, tensor.nbytes, "Tensor-relative byte offset")
        count = _uint(count, MAX_READ_BYTES, "Read byte count")
        if offset % tensor.itemsize or count % tensor.itemsize:
            raise SafeTensorError("Byte offset and count must align with tensor itemsize")
        if count > tensor.nbytes - offset:
            raise SafeTensorError("Read extends beyond tensor bounds")
        self._assert_unchanged()
        if count == 0:
            return b""
        return self._read_at(self._payload_start + tensor.data_offsets[0] + offset,
                             count, tensor=True)

    def read_matrix_tile(self, name: str, row: int, col: int, rows: int, cols: int) -> bytes:
        """Gather a rank-two row-major tile with edges 1..128 and <=64 KiB."""
        tensor = self._tensor(name)
        if len(tensor.shape) != 2:
            raise SafeTensorError("Matrix tiles require a rank-two tensor")
        row = _uint(row, tensor.shape[0], "Tile row")
        col = _uint(col, tensor.shape[1], "Tile column")
        rows = _uint(rows, MAX_TILE_EDGE, "Tile row count")
        cols = _uint(cols, MAX_TILE_EDGE, "Tile column count")
        if not rows or not cols:
            raise SafeTensorError("Tile edges must be from 1 to 128")
        if rows > tensor.shape[0] - row or cols > tensor.shape[1] - col:
            raise SafeTensorError("Tile extends beyond matrix bounds")
        total = rows * cols * tensor.itemsize
        if total > MAX_READ_BYTES:
            raise SafeTensorError("Matrix tile exceeds the 64-KiB result policy")
        self._assert_unchanged()
        result = bytearray(total)
        row_bytes = cols * tensor.itemsize
        for local_row in range(rows):
            tensor_offset = ((row + local_row) * tensor.shape[1] + col) * tensor.itemsize
            begin = local_row * row_bytes
            result[begin:begin + row_bytes] = self.read_bytes(name, tensor_offset, row_bytes)
        self._assert_unchanged()
        return bytes(result)

    def read_span(self, name: str, offset: int, count: int) -> bytearray:
        """Read an owned <=8-MiB buffer with <=64-KiB actual reads.

        Alignment/bounds are checked before allocation. Before/after validation
        polls the descriptor/path for unprotected files or checks the lifetime
        of a Windows deny-write/delete handle. No buffer is returned before the
        final check. The caller accounts and owns the returned buffer.
        """
        tensor = self._tensor(name)
        offset = _uint(offset, tensor.nbytes, "Tensor-relative byte offset")
        count = _uint(count, MAX_READ_SPAN_BYTES, "Read span byte count")
        if offset % tensor.itemsize or count % tensor.itemsize:
            raise SafeTensorError("Byte offset and count must align with tensor itemsize")
        if count > tensor.nbytes - offset:
            raise SafeTensorError("Read extends beyond tensor bounds")
        if count == 0:
            self._assert_unchanged()
            return bytearray()
        self._protect_span_handle()
        self._assert_unchanged()
        data = bytearray(count)
        assert self._stream is not None
        self._counters["span_read_calls"] += 1
        self._counters["max_span_bytes"] = max(self._counters["max_span_bytes"], count)
        try:
            self._stream.seek(self._payload_start + tensor.data_offsets[0] + offset)
            with memoryview(data) as target:
                for at in range(0, count, MAX_READ_BYTES):
                    size = min(MAX_READ_BYTES, count - at)
                    received = self._stream.readinto(target[at:at + size])
                    if type(received) is not int or not 0 <= received <= size:
                        raise SafeTensorError("Safetensors span returned an invalid bounded read")
                    self._counters["actual_read_calls"] += 1
                    self._counters["actual_read_bytes"] += received
                    self._counters["tensor_read_calls"] += 1
                    self._counters["tensor_read_bytes"] += received
                    self._counters["span_inner_read_calls"] += 1
                    self._counters["span_read_bytes"] += received
                    self._counters["max_actual_read_bytes"] = max(self._counters["max_actual_read_bytes"], received)
                    if received != size:
                        raise SafeTensorError("Truncated safetensors file during bounded span read")
        except OSError as error:
            raise SafeTensorError("Safetensors bounded span read failed") from error
        finally:
            self._assert_unchanged()
        return data

    def stats(self) -> dict:
        """Actual I/O counters, not total RAM or whole-checkpoint verification."""
        return {
            **self._counters,
            "file_size": self._file_size,
            "header_bytes": self._header_bytes,
            "tensor_count": len(self._tensors),
            "policy_max_read_bytes": MAX_READ_BYTES,
            "policy_max_span_bytes": MAX_READ_SPAN_BYTES,
            "span_handle_protection": "windows_deny_write_delete" if self._span_protected else "identity_before_after",
            "protected_immutable": self.protected_immutable,
            "span_validation_scope": "live deny-write/delete handle; otherwise descriptor/path fingerprints around each complete logical span",
            "content_verification": "header_sha256_and_file_identity_no_payload_checksum",
        }


def parse_header_bytes(header: bytes | bytearray, file_size: int):
    """Validate only header metadata against a declared whole-file length.

    Shares every dtype/offset/shape rule with the local reader. Does not open a
    file, allocate payload, or establish authenticity of the declared length.
    """
    if not isinstance(header, (bytes, bytearray)) or not 1 <= len(header) <= MAX_HEADER_BYTES:
        raise SafeTensorError("Header length outside local 1-byte to 1-MiB policy")
    if type(file_size) is not int or not 8 + len(header) <= file_size <= MAX_FILE_BYTES:
        raise SafeTensorError("Header length exceeds file size or local file-size policy")
    if header[0] != ord("{"):
        raise SafeTensorError("Safetensors header must begin with '{'")
    try:
        root = json.loads(header.decode("utf-8", errors="strict"),
                          object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SafeTensorError(f"Invalid safetensors JSON header: {error}") from error
    if not isinstance(root, dict):
        raise SafeTensorError("Safetensors header must be a JSON object")
    metadata = root.pop("__metadata__", {})
    SafeTensorReader._validate_metadata(metadata)
    if len(root) > MAX_TENSORS:
        raise SafeTensorError("Too many tensors for local metadata policy")
    payload_bytes = file_size - 8 - len(header)
    tensors = {}
    for name, entry in root.items():
        tensors[name] = SafeTensorReader._validate_tensor(name, entry, payload_bytes)
    cursor = 0
    for name, tensor in sorted(tensors.items(), key=lambda pair: pair[1].data_offsets):
        start, end = tensor.data_offsets
        if start != cursor:
            raise SafeTensorError(f"Tensor {name!r} overlaps data or leaves a payload gap")
        cursor = end
    if cursor != payload_bytes:
        raise SafeTensorError("Trailing unclaimed safetensors payload bytes")
    return MappingProxyType(tensors), MappingProxyType(dict(metadata))
