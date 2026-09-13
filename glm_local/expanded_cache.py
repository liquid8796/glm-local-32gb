"""Bounded per-layer reuse of expanded MLA projections, with exact input keys."""
from collections import OrderedDict
from array import array
import hashlib
from .cpu_probe import _finite_float32

DEFAULT_EXPANDED_CACHE_TOKENS = 256


class ExpandedKvCache:
    def __init__(self, layers, tokens, width, ledger):
        if (type(layers) is not int or not 1 <= layers <= 256 or type(tokens) is not int or not 0 <= tokens <= 256
                or type(width) is not int or not 1 <= width <= 1_048_576 or not callable(getattr(ledger, "reserve", None))):
            raise ValueError("Expanded cache requires bounded layer/token/width counts and a ledger")
        self.layers = [OrderedDict() for _ in range(layers)]
        self.leases = [None] * layers
        self.tokens, self.width, self.ledger = tokens, width, ledger
        self.hits = self.misses = self.bytes = self.peak_bytes = 0

    def _remove(self, cache, position):
        entry = cache.pop(position)
        self.bytes -= len(entry[1]) * 4 + 32
        entry[1] = None

    def _release_empty(self, layer):
        if not self.layers[layer] and self.leases[layer] is not None:
            self.leases[layer].release()
            self.leases[layer] = None

    @property
    def reserved_bytes(self):
        return sum(lease is not None for lease in self.leases) * self.tokens * (self.width * 4 + 32)

    def get(self, layer, position, latent, compute, validate=None):
        if type(layer) is not int or not 0 <= layer < len(self.layers) or type(position) is not int or not 0 <= position < 2**31:
            raise ValueError("Expanded cache layer/position must be bounded nonnegative integers")
        if not isinstance(latent, (list, tuple, array)) or not 1 <= len(latent) <= 1_048_576:
            raise ValueError("Expanded cache latent must be a bounded sized numeric sequence")
        if not callable(compute) or (validate is not None and not callable(validate)):
            raise ValueError("Expanded cache callbacks must be callable")
        if not self.tokens:
            return compute()
        cache = self.layers[layer]
        # Decoder-owned FP32 arrays have already passed projection validation;
        # avoid re-rounding each cached history vector on every attention step.
        raw = (latent.tobytes() if isinstance(latent, array) and latent.typecode == "f" and latent.itemsize == 4
               else array("f", (_finite_float32(value, "expanded latent") for value in latent)).tobytes())
        digest = hashlib.sha256(raw).digest()
        entry = cache.get(position)
        if entry is not None and entry[0] == digest:
            if validate is not None:
                validate()
            self.hits += 1
            cache.move_to_end(position)
            return entry[1]
        self.misses += 1
        if entry is not None:
            self._remove(cache, position)
        while len(cache) >= self.tokens:
            self._remove(cache, next(iter(cache)))
        size = self.width * 4 + 32
        if self.leases[layer] is None:
            self.leases[layer] = self.ledger.reserve(size * self.tokens, label="expanded_mla_layer_cache")
        try:
            values = compute()
            if not isinstance(values, array) or values.typecode != "f" or len(values) != self.width:
                raise ValueError("Expanded MLA cache needs the planned FP32 projection shape")
            cache[position] = [digest, values]
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            return values
        except BaseException:
            self._release_empty(layer)
            raise

    def truncate(self, position):
        if type(position) is not int or not 0 <= position < 2**31:
            raise ValueError("Expanded cache truncation position is invalid")
        for layer, cache in enumerate(self.layers):
            for key in list(cache):
                if key >= position:
                    self._remove(cache, key)
            self._release_empty(layer)

    def clear(self):
        for layer, cache in enumerate(self.layers):
            for key in list(cache):
                self._remove(cache, key)
            self._release_empty(layer)

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "bytes": self.bytes, "peak_bytes": self.peak_bytes,
                "reserved_bytes": self.reserved_bytes,
                "tokens_per_layer": self.tokens, "width": self.width}
