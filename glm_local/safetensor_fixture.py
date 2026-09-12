"""Official-library creation and independent decoding of small invented files.

The reference expands per-block scales using ordinary PyTorch CPU float64;
it does not use the bounded reader, block adapter, native kernels or a model.
"""

import hashlib
from pathlib import Path

MAX_FILE_BYTES = 2 * 1024**2
NAMES = {"weight", "weight_scale_inv", "vector", "bf16_probe", "empty_probe", "scalar_probe"}
MARKER = "glm-local-fp8-storage-v1"


def _shape(rows, cols, seed):
    if any(type(v) is not int or not 1 <= v <= 1024 for v in (rows, cols)):
        raise ValueError("Fixture dimensions must be integers in [1, 1024]")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Fixture seed must be uint32")


def create_fixture(path, rows=257, cols=259, seed=7):
    _shape(rows, cols, seed)
    import torch
    import safetensors
    from safetensors.torch import save
    # Include signs, signed zero, subnormals and extrema, excluding the two NaNs.
    r = torch.arange(rows, dtype=torch.int64).reshape(-1, 1)
    c = torch.arange(cols, dtype=torch.int64).reshape(1, -1)
    raw = (r * 73 + c * 29 + seed) % 254
    raw = (raw + (raw >= 127).to(torch.int64)).to(torch.uint8)
    weights = raw.contiguous().view(torch.float8_e4m3fn)
    br = torch.arange((rows+127)//128, dtype=torch.float32).reshape(-1, 1)
    bc = torch.arange((cols+127)//128, dtype=torch.float32).reshape(1, -1)
    scales = (3 + 5*br + 7*bc) / 128
    vector = ((torch.arange(cols, dtype=torch.int32) % 9) - 4).float() / 256
    tensors = {"weight": weights, "weight_scale_inv": scales, "vector": vector,
               "bf16_probe": torch.tensor([1.0, -2.5, 0.25], dtype=torch.bfloat16),
               "empty_probe": torch.empty((0, 3), dtype=torch.float32),
               "scalar_probe": torch.tensor(1.25, dtype=torch.float32)}
    encoded = save(tensors, metadata={"synthetic_fixture": MARKER, "seed": str(seed)})
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError("Official fixture exceeds 2 MiB bound")
    with Path(path).open("xb") as stream:
        stream.write(encoded)
    return {"shape": [rows, cols], "seed": seed, "file_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(), "synthetic_only": True,
            "created_by": "safetensors.torch.save", "safetensors_version": safetensors.__version__,
            "determinism": "tensor values are deterministic; serializer metadata order may vary",
            "scale_grid": list(scales.shape)}


def reference_fixture(path):
    import torch
    import safetensors
    from safetensors import safe_open
    path = Path(path)
    if not 8 <= path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Reference accepts only tiny fixture files up to 2 MiB")
    with safe_open(str(path), framework="pt", device="cpu") as file:
        marker = file.metadata()
        if set(file.keys()) != NAMES or not marker or marker.get("synthetic_fixture") != MARKER:
            raise ValueError("Not the expected small synthetic storage fixture")
        tensors = {name: file.get_tensor(name) for name in NAMES}
    weight, scales, vector = tensors["weight"], tensors["weight_scale_inv"], tensors["vector"]
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Synthetic weight must be 2D E4M3FN")
    rows, cols = weight.shape
    _shape(rows, cols, int(marker["seed"]))
    if scales.shape != ((rows+127)//128, (cols+127)//128) or scales.dtype != torch.float32:
        raise ValueError("Synthetic scale grid shape/dtype mismatch")
    if vector.shape != (cols,) or vector.dtype != torch.float32:
        raise ValueError("Synthetic vector shape/dtype mismatch")
    if (not torch.isfinite(weight.float()).all().item() or not torch.isfinite(scales).all().item()
            or not (scales > 0).all().item() or not torch.isfinite(vector).all().item()):
        raise ValueError("Synthetic computation requires finite weights/vector and positive finite scales")
    probes = (("bf16_probe", (3,), torch.bfloat16), ("empty_probe", (0, 3), torch.float32),
              ("scalar_probe", (), torch.float32))
    for name, shape, dtype in probes:
        if tuple(tensors[name].shape) != shape or tensors[name].dtype != dtype:
            raise ValueError("Synthetic format probe shape/dtype mismatch")
    if tensors["bf16_probe"].float().tolist() != [1.0, -2.5, 0.25] or tensors["scalar_probe"].item() != 1.25:
        raise ValueError("Synthetic scalar/BF16 sentinel values changed")
    with torch.inference_mode():
        expanded = scales.double().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:rows, :cols]
        decoded = weight.double() * expanded
        expected = torch.mv(decoded, vector.double()).tolist()
    dtype_tags = {torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32", torch.bfloat16: "BF16"}
    tensor_records = {}
    for name, tensor in tensors.items():
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        tensor_records[name] = {"dtype": dtype_tags[tensor.dtype], "shape": list(tensor.shape),
                                "raw_bytes": raw, "sha256": hashlib.sha256(raw).hexdigest()}
    return {"rows": rows, "cols": cols, "weight_bytes": tensor_records["weight"]["raw_bytes"],
            "scale_values": scales.tolist(), "vector": vector.tolist(), "expected_output": expected,
            "tensors": tensor_records, "metadata": {"synthetic_only": True,
                "safetensors_version": safetensors.__version__, "torch_version": str(torch.__version__),
                "oracle_scope": "official safetensors decode + explicit PyTorch CPU float64 scale expansion"}}
