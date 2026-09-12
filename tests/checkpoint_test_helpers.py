"""Invented metadata-only fixtures. Never evidence about the remote checkpoint."""
from copy import deepcopy
import io
import json
from email.message import Message
import struct

from glm_local.safetensor_reader import DTYPE_ITEMSIZE

MODEL = "example/invented-checkpoint"
REVISION = "a" * 40
NAME = "model.layers.0.mlp.gate_proj.weight"
SCALE = NAME + "_scale_inv"


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode()


def make_header(specs):
    root, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, dtype, shape in specs:
        elements = 1
        for dim in shape:
            elements *= dim
        end = offset + elements * DTYPE_ITEMSIZE[dtype]
        root[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, end]}
        offset = end
    header = encode(root)
    return struct.pack("<Q", len(header)) + header, 8 + len(header) + offset, offset


def checkpoint():
    config = {"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"],
              "hidden_size": 259, "vocab_size": 3, "num_hidden_layers": 1,
              "intermediate_size": 257, "moe_intermediate_size": 257,
              "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}}
    specs = {
        "model-00001-of-00002.safetensors": [
            (NAME, "F8_E4M3", [257, 259]), ("model.embed_tokens.weight", "BF16", [3, 259]),
            ("lm_head.weight", "BF16", [3, 259]), ("model.norm.weight", "BF16", [259])],
        "model-00002-of-00002.safetensors": [(SCALE, "F32", [3, 3]), ("aux.stat", "F32", [])],
    }
    headers, mapping, siblings, total = {}, {}, [], 0
    for filename, values in specs.items():
        raw, size, payload = make_header(values)
        headers[filename] = (raw, size)
        siblings.append({"rfilename": filename, "size": size})
        total += payload
        mapping.update({name: filename for name, _, _ in values})
    model = {"id": MODEL, "sha": REVISION, "siblings": siblings}
    index = {"metadata": {"total_size": total}, "weight_map": mapping}
    expected = {"model_id": MODEL, "revision": REVISION,
                "weights": [{"name": s["rfilename"], "bytes": s["size"]} for s in siblings],
                "architecture": {**{k: v for k, v in config.items() if k != "quantization_config"},
                                 "quantization": deepcopy(config["quantization_config"])}}
    return {"model": model, "config": config, "index": index, "headers": headers, "expected": expected}


def settings():
    return {"model_id": MODEL, "revision": REVISION, "model_directory": "models/invented",
            "ram_budget_bytes": 32000000000, "cpu_job_percent": 70, "gpu_index": 0,
            "gpu_average_target": 0.6, "gpu_window_seconds": 10,
            "disk_reserve_bytes": 20000000000, "backend_status": "not_implemented"}


class MemorySource:
    mode = "synthetic_test"

    def __init__(self, data=None):
        self.data = deepcopy(data or checkpoint())
        self.calls = []
        self.byte_count = 0

    def json_bytes(self, kind):
        self.calls.append(kind)
        raw = encode(self.data[kind])
        self.byte_count += len(raw)
        return raw

    def header_bytes(self, name, size):
        self.calls.append(name)
        raw, stored_size = self.data["headers"][name]
        if stored_size != size:
            raise AssertionError("Fixture size mismatch")
        self.byte_count += len(raw)
        return raw

    def available_shards(self):
        return None

    def stats(self):
        return {"mode": self.mode, "body_bytes_read": self.byte_count, "requests": 0,
                "tensor_payload_bytes_requested": 0, "synthetic_fixture": True}


class Response(io.BytesIO):
    def __init__(self, raw=b"", *, status=200, headers=None, url=None):
        super().__init__(raw)
        self.status, self.headers, self.url = status, Message(), url
        for key, value in (headers or {}).items():
            self.headers[key] = str(value)
        self.requested_reads = []

    def read(self, count=-1):
        self.requested_reads.append(count)
        if count < 0:
            raise AssertionError("Unbounded read forbidden")
        return super().read(count)

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        if value.url is None:
            value.url = request.full_url
        return value


def ranged(raw, start, size, **headers):
    return Response(raw, status=206, headers={"Content-Range": f"bytes {start}-{start + len(raw) - 1}/{size}",
                                             "Content-Length": len(raw), **headers})
