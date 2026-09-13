"""Bounded local row-band reuse over an identity-checking tensor reader.

Only encoded payload is cached, never decoded matrices or a complete tensor.
Every source read and returned tile remains <=64 KiB. One band per tensor and
a global LRU bound both retained payload and entry count. As with the source
reader, ownership is single-threaded; ledger accounting is not an RSS limit.
"""
from collections import OrderedDict

from .safetensor_reader import MAX_READ_BYTES, MAX_READ_SPAN_BYTES, MAX_TILE_EDGE, SafeTensorError, _uint


DEFAULT_ROW_BAND_CACHE_BYTES = 8 * 1024**2
_MAX_CACHE_ENTRIES = 128


class _Band:
    __slots__ = ("row", "rows", "info", "payload", "lease")

    def __init__(self, row, rows, info, payload, lease):
        self.row, self.rows, self.info = row, rows, info
        self.payload, self.lease = payload, lease


class RowBandCacheReader:
    """Reuse one <=128-row encoded band per tensor across column tiles.

    A miss reserves retained bytes before allocation and performs contiguous
    tensor-relative reads. A hit still checks source identity before and after
    extracting the tile. Oversized bands, complete tensors and a zero cache
    capacity use the original tile reader. ``read_bytes`` remains passthrough.
    """

    def __init__(self, source, ledger=None, max_cache_bytes=DEFAULT_ROW_BAND_CACHE_BYTES,
                 *, owns_source=True):
        if type(max_cache_bytes) is not int or not 0 <= max_cache_bytes <= DEFAULT_ROW_BAND_CACHE_BYTES:
            raise ValueError("Row-band cache capacity must be an integer from 0 to 8 MiB")
        if type(owns_source) is not bool:
            raise ValueError("owns_source must be boolean")
        if not callable(getattr(source, "assert_tensor_unchanged", None)):
            raise ValueError("Row-band source must provide assert_tensor_unchanged(name)")
        if ledger is not None and not callable(getattr(ledger, "reserve", None)):
            raise ValueError("Row-band ledger must provide reserve()")
        self._source, self._ledger = source, ledger
        self._capacity, self._owns_source = max_cache_bytes, owns_source
        self._bands = OrderedDict()
        self._closed = False
        self._retained = 0
        self._counts = dict(hits=0, misses=0, bands_loaded=0, evictions=0,
                            identity_checks=0, invalidations=0, fallback_tiles=0,
                            coalesced_read_calls=0, coalesced_read_bytes=0,
                            max_coalesced_read_bytes=0, peak_retained_bytes=0,
                            tile_calls=0, tile_bytes_returned=0, passthrough_read_calls=0,
                            cache_span_calls=0, cache_span_bytes=0, passthrough_span_calls=0)

    @property
    def tensors(self):
        return self._source.tensors

    @property
    def config(self):
        return self._source.config

    def projection(self, name):
        self._require_open()
        return self._source.projection(name)

    def _require_open(self):
        if self._closed:
            raise SafeTensorError("Row-band cache reader is closed")

    def _discard(self, name, *, eviction=False):
        band = self._bands.pop(name)
        self._retained -= len(band.payload)
        # Drop the payload before releasing its reservation.
        band.payload = None
        if band.lease is not None:
            band.lease.release()
            band.lease = None
        if eviction:
            self._counts["evictions"] += 1

    def _clear(self):
        for name in list(self._bands):
            self._discard(name)

    def assert_tensor_unchanged(self, name):
        self._require_open()
        self._counts["identity_checks"] += 1
        try:
            self._source.assert_tensor_unchanged(name)
        except BaseException:
            self._counts["invalidations"] += 1
            self._clear()
            raise

    def read_bytes(self, name, offset, count):
        self._require_open()
        self._counts["passthrough_read_calls"] += 1
        return self._source.read_bytes(name, offset, count)

    def read_span(self, name, offset, count):
        """Return an owned bounded span; its lifetime is accounted by the caller."""
        self._require_open()
        if not isinstance(name, str):
            raise SafeTensorError("Tensor name must be a string")
        info = self.tensors[name]
        offset = _uint(offset, info.nbytes, "Tensor-relative byte offset")
        count = _uint(count, MAX_READ_SPAN_BYTES, "Read span byte count")
        if offset % info.itemsize or count % info.itemsize:
            raise SafeTensorError("Byte offset and count must align with tensor itemsize")
        if count > info.nbytes - offset:
            raise SafeTensorError("Read extends beyond tensor bounds")
        self._counts["passthrough_span_calls"] += 1
        try:
            span = getattr(self._source, "read_span", None)
            if callable(span):
                data = span(name, offset, count)
                if type(data) is not bytearray or len(data) != count:
                    raise SafeTensorError("Span source must return an owned bytearray of the requested size")
                return data
            self.assert_tensor_unchanged(name)
            data = bytearray(count)
            for at in range(0, count, MAX_READ_BYTES):
                size = min(MAX_READ_BYTES, count - at)
                raw = self._source.read_bytes(name, offset + at, size)
                if type(raw) is not bytes or len(raw) != size:
                    raise SafeTensorError("Span fallback returned an invalid bounded read")
                data[at:at + size] = raw
            self.assert_tensor_unchanged(name)
            return data
        except BaseException:
            self._clear()
            raise

    def _load_band(self, name, row, rows, info, size):
        # Keeping old bands of this tensor could eventually retain its full matrix.
        if name in self._bands:
            self._discard(name, eviction=True)
        while self._bands and (self._retained + size > self._capacity
                               or len(self._bands) >= _MAX_CACHE_ENTRIES):
            self._discard(next(iter(self._bands)), eviction=True)
        lease = self._ledger.reserve(size, label="row_band_cache") if self._ledger else None
        payload = None
        try:
            start = row * info.shape[1] * info.itemsize
            span = getattr(self._source, "read_span", None)
            if callable(span):
                payload = span(name, start, size)
                if type(payload) is not bytearray or len(payload) != size:
                    raise SafeTensorError("Row-band span must return an owned bytearray of the requested size")
                self._counts["cache_span_calls"] += 1
                self._counts["cache_span_bytes"] += size
            else:
                payload = bytearray(size)
                for offset in range(0, size, MAX_READ_BYTES):
                    count = min(MAX_READ_BYTES, size - offset)
                    raw = self._source.read_bytes(name, start + offset, count)
                    if not isinstance(raw, bytes) or len(raw) != count:
                        raise SafeTensorError("Row-band source returned an invalid bounded read")
                    self._counts["coalesced_read_calls"] += 1
                    self._counts["coalesced_read_bytes"] += count
                    self._counts["max_coalesced_read_bytes"] = max(self._counts["max_coalesced_read_bytes"], count)
                    payload[offset:offset + count] = raw
                    del raw
            self.assert_tensor_unchanged(name)
            band = _Band(row, rows, info, payload, lease)
            self._bands[name] = band
            self._retained += size
            self._counts["bands_loaded"] += 1
            self._counts["peak_retained_bytes"] = max(self._counts["peak_retained_bytes"], self._retained)
            return band
        except BaseException:
            payload = None
            if lease is not None:
                lease.release()
            self._clear()
            raise

    def read_matrix_tile(self, name, row, col, rows, cols):
        self._require_open()
        if not isinstance(name, str):
            raise SafeTensorError("Tensor name must be a string")
        info = self.tensors[name]
        if len(info.shape) != 2:
            raise SafeTensorError("Matrix tiles require a rank-two tensor")
        row = _uint(row, info.shape[0], "Tile row")
        col = _uint(col, info.shape[1], "Tile column")
        rows = _uint(rows, MAX_TILE_EDGE, "Tile row count")
        cols = _uint(cols, MAX_TILE_EDGE, "Tile column count")
        if not rows or not cols:
            raise SafeTensorError("Tile edges must be from 1 to 128")
        if rows > info.shape[0] - row or cols > info.shape[1] - col:
            raise SafeTensorError("Tile extends beyond matrix bounds")
        total = rows * cols * info.itemsize
        if total > MAX_READ_BYTES:
            raise SafeTensorError("Matrix tile exceeds the 64-KiB result policy")
        self.assert_tensor_unchanged(name)
        self._counts["tile_calls"] += 1
        band_bytes = rows * info.shape[1] * info.itemsize
        if band_bytes > self._capacity or rows == info.shape[0]:
            self._counts["misses"] += 1
            self._counts["fallback_tiles"] += 1
            result = self._source.read_matrix_tile(name, row, col, rows, cols)
        else:
            band = self._bands.get(name)
            if band is not None and band.row == row and band.rows == rows and band.info == info:
                self._counts["hits"] += 1
                self._bands.move_to_end(name)
            else:
                self._counts["misses"] += 1
                band = self._load_band(name, row, rows, info, band_bytes)
            result = bytearray(total)
            row_bytes, stride = cols * info.itemsize, info.shape[1] * info.itemsize
            with memoryview(band.payload) as encoded:
                for local_row in range(rows):
                    begin = local_row * stride + col * info.itemsize
                    result[local_row * row_bytes:(local_row + 1) * row_bytes] = encoded[begin:begin + row_bytes]
            result = bytes(result)
        self.assert_tensor_unchanged(name)
        if not isinstance(result, bytes) or len(result) != total:
            raise SafeTensorError("Row-band source returned an invalid matrix tile")
        self._counts["tile_bytes_returned"] += len(result)
        return result

    def stats(self):
        source = self._source.stats()
        return {**source,
                "retained_weight_payload_bytes": source.get("retained_weight_payload_bytes", 0) + self._retained,
                "row_band_cache": {**self._counts, "retained_bytes": self._retained,
                                   "capacity_bytes": self._capacity, "cached_bands": len(self._bands),
                                   "max_cached_bands": _MAX_CACHE_ENTRIES,
                                   "max_rows_per_band": MAX_TILE_EDGE,
                                   "max_source_read_bytes": MAX_READ_BYTES,
                                   "ledger_accounted": self._ledger is not None,
                                   "full_matrix_retained": False, "closed": self._closed}}

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._clear()
        finally:
            if self._owns_source:
                self._source.close()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()
