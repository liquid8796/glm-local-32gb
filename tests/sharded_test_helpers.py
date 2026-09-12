"""Small manually serialized files for reader tests, independent of its parser."""
import json
from pathlib import Path
import struct


def write_safe(path, tensors, metadata=None):
    header = {} if metadata is None else {"__metadata__": metadata}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [len(data), len(data) + len(payload)]}
        data.extend(payload)
    encoded = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def read_safe(path):
    content = Path(path).read_bytes()
    size = struct.unpack("<Q", content[:8])[0]
    header = json.loads(content[8:8+size])
    body = content[8+size:]
    return {name: (info["dtype"], tuple(info["shape"]),
                   body[info["data_offsets"][0]:info["data_offsets"][1]])
            for name, info in header.items() if name != "__metadata__"}, header.get("__metadata__")


def read_index(directory):
    return json.loads((Path(directory) / "model.safetensors.index.json").read_text())


def write_index(directory, data):
    (Path(directory) / "model.safetensors.index.json").write_text(json.dumps(data))
