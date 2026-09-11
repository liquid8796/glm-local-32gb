"""Standalone bounded MSVC top-k candidate, verified against the pinned Torch.

No Torch import: equal-value behavior depends on the build's C++ STL. A
successful comparison on one pinned runtime is not cross-platform parity.
"""

import ctypes
import hashlib
import math
from pathlib import Path


class NativeTopK:
    def __init__(self):
        path = Path(__file__).resolve().parent.parent / "build" / "topk_cpu.dll"
        if not path.is_file():
            raise RuntimeError("Run build-native.bat to build the bounded top-k helper")
        self._dll = ctypes.CDLL(str(path))
        self._call = self._dll.mini_topk
        self._call.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
        self._call.restype = ctypes.c_int
        self.metadata = {"strategy": "MSVC STL partial_sort/nth_element FP32 candidate",
                         "dll_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "torch_dependency": False, "cross_platform_equivalence": False}

    def __call__(self, scores, count):
        if not isinstance(scores, (list, tuple)) or not 1 <= len(scores) <= 128:
            raise ValueError("Top-k scores must contain 1-128 finite values")
        if type(count) is not int or not 1 <= count <= min(4, len(scores)):
            raise ValueError("Top-k count must be in [1, min(4, length)]")
        values = []
        for score in scores:
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("Top-k scores must be finite")
            value = ctypes.c_float(score).value
            if not math.isfinite(value):
                raise ValueError("Top-k scores must fit FP32")
            values.append(value)
        data = (ctypes.c_float * len(values))(*values)
        output = (ctypes.c_int32 * count)()
        code = self._call(data, len(values), count, output)
        if code:
            raise RuntimeError(f"Native top-k rejected inputs: {code}")
        return list(output)
