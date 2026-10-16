"""Offline compressed-tensors W4A16 reference projection tests (no model download)."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from glm_local.flash_w4a16 import W4A16Projection
from glm_local.safetensor_reader import SafeTensorReader


PREFIX = "model.language_model.layers.3.mlp.experts.0.gate_proj"


def pack_row(values):
    assert len(values) % 8 == 0
    return bytes(((v & 15) | ((values[i + 1] & 15) << 4))
                 for i, v in enumerate(values) if i % 2 == 0)


def bf16(value):
    return struct.pack("<f", value)[2:]


def safetensors_file(path, tensors):
    """Small real safetensors fixture: dtype, shape, payload tuples."""
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [start, start + len(raw)]}
        payload.extend(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def tensors(rows, scales, *, packed_dtype="I32", scale_dtype="BF16",
            packed_shape=None, scale_shape=None):
    count, width = len(rows), len(rows[0])
    packed = b"".join(pack_row(row) for row in rows)
    scale_raw = b"".join(bf16(value) for row in scales for value in row)
    return {
        PREFIX + ".weight_packed": (packed_dtype, packed_shape or [count, width // 8], packed),
        PREFIX + ".weight_scale": (scale_dtype, scale_shape or [count, width // 32], scale_raw),
        PREFIX + ".weight_shape": ("I64", [2], struct.pack("<qq", count, width)),
    }


class FlashW4A16Tests(unittest.TestCase):
    def with_reader(self, entries, check):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.safetensors"
            safetensors_file(path, entries)
            with SafeTensorReader(path) as reader:
                return check(reader)

    def test_signed_nibbles_and_bf16_group_scales_match_independent_dot(self):
        a = ([-8, -7, -1, 0, 1, 7, 2, -2] * 8)
        b = ([7, -8, 0, -1, 1, 2, 3, -4] * 8)
        scales = [[0.5, 2.0], [1.0, 0.25]]
        x = [(-1.0 if i % 2 else 0.5) for i in range(64)]

        def check(reader):
            projection = W4A16Projection(reader, PREFIX)
            self.assertEqual((projection.rows, projection.cols), (2, 64))
            self.assertEqual(projection.matvec(x), [
                sum(v * x[i] * scales[r][i // 32] for i, v in enumerate(row))
                for r, row in enumerate([a, b])])
            self.assertLessEqual(reader.stats()["max_actual_read_bytes"], 65536)

        self.with_reader(tensors([a, b], scales), check)

    def test_streams_rows_without_materializing_full_matrix(self):
        rows = [[1] * 4096 for _ in range(3)]
        scales = [[0.5] * 128 for _ in range(3)]
        x = [1.0] * 4096

        def check(reader):
            projection = W4A16Projection(reader, PREFIX)
            self.assertEqual(projection.matvec(x), [2048.0] * 3)
            self.assertLessEqual(reader.stats()["max_span_bytes"], 65536)
            self.assertLessEqual(reader.stats()["max_actual_read_bytes"], 65536)

        self.with_reader(tensors(rows, scales), check)

    def test_bad_packed_dtype_fails_closed(self):
        rows = [[1] * 32]
        entries = tensors(rows, [[1.0]], packed_dtype="U8", packed_shape=[1, 16])
        # Supply bytes exactly matching U8 metadata, then require constructor rejection.
        entries[PREFIX + ".weight_packed"] = ("U8", [1, 16], b"\x11" * 16)
        self.with_reader(entries, lambda reader: self.assertRaises(
            ValueError, W4A16Projection, reader, PREFIX))

    def test_nonfinite_scale_rejected_during_multiply(self):
        entries = tensors([[1] * 32], [[1.0]])
        entries[PREFIX + ".weight_scale"] = ("BF16", [1, 1], b"\x80\x7f")

        def check(reader):
            projection = W4A16Projection(reader, PREFIX)
            with self.assertRaisesRegex(ValueError, "scale"):
                projection.matvec([1] * 32)

        self.with_reader(entries, check)

    def test_f32_scales_allow_reference_when_metadata_dtype_is_f32(self):
        entries = tensors([[2] * 32], [[0.5]])
        entries[PREFIX + ".weight_scale"] = ("F32", [1, 1], struct.pack("<f", 0.5))
        self.with_reader(entries, lambda reader: self.assertEqual(
            W4A16Projection(reader, PREFIX).matvec([1.0] * 32), [32.0]))

    def test_rejects_nan_activation_before_reading_weights(self):
        entries = tensors([[1] * 32], [[1.0]])

        def check(reader):
            projection = W4A16Projection(reader, PREFIX)
            with self.assertRaisesRegex(ValueError, "finite"):
                projection.matvec([float("nan")] * 32)
            self.assertEqual(reader.stats()["tensor_read_bytes"], 16)

        self.with_reader(entries, check)
    def test_dimension_mismatch_fails_without_weight_read(self):
        entries = tensors([[1] * 32], [[1.0]])
        entries[PREFIX + ".weight_shape"] = ("I64", [2], struct.pack("<qq", 1, 64))

        def check(reader):
            with self.assertRaises(ValueError):
                W4A16Projection(reader, PREFIX)
            self.assertEqual(reader.stats()["tensor_read_bytes"], 16)

        self.with_reader(entries, check)


if __name__ == "__main__":
    unittest.main()
