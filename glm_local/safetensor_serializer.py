"""Official serialization of tiny raw-byte fixtures across safetensors APIs.

This is not a checkpoint writer: each call accepts at most 64 KiB of payload.
Safetensors 0.7 accepts raw dictionaries; 0.8 exposes pointer-based TensorSpec.
The optional dependency is loaded only while creating a synthetic fixture.
"""
from __future__ import annotations

import ctypes

MAX_PAYLOAD_BYTES = 64 * 1024


def serialize_raw_tensors(tensors: dict[str, dict], metadata: dict[str, str] | None = None) -> bytes:
    """Serialize existing bytes, without decoding or re-quantizing any FP8 value.

    TensorSpec does not own its pointers. Own a private copy of every buffer and
    keep all copies alive through serialization, including conversion to bytes.
    Detect the exposed API rather than parsing versions or retrying TypeErrors.
    """
    import safetensors

    total = 0
    for tensor in tensors.values():
        payload = tensor["data"]
        if not isinstance(payload, bytes):
            raise TypeError("Synthetic serializer payloads must be immutable bytes")
        total += len(payload)
        if total > MAX_PAYLOAD_BYTES:
            raise ValueError("Synthetic serializer payload exceeds its 64-KiB bound")

    tensor_spec = getattr(safetensors, "TensorSpec", None)
    if tensor_spec is None:
        return bytes(safetensors.serialize(tensors, metadata=metadata))

    buffers = []
    specs = {}
    for name, tensor in tensors.items():
        payload = tensor["data"]
        # Even an empty tensor receives a live, non-null address; data_len stays 0.
        buffer = ctypes.create_string_buffer(payload, max(1, len(payload)))
        buffers.append(buffer)
        specs[name] = tensor_spec(dtype=tensor["dtype"], shape=tensor["shape"],
                                  data_ptr=ctypes.addressof(buffer), data_len=len(payload))
    return bytes(safetensors.serialize(specs, metadata=metadata))
