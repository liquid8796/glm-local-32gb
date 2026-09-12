"""Small independent format fixtures; no optional tensor library required."""

import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from glm_local.safetensor_reader import SafeTensorReader, SafeTensorError, MAX_HEADER_BYTES, MAX_READ_BYTES


def entry(dtype="U8", shape=None, offsets=None):
    return {"dtype": dtype, "shape": [4] if shape is None else shape,
            "data_offsets": [0, 4] if offsets is None else offsets}


def encode(header, payload=b"abcd"):
    raw = header if isinstance(header, bytes) else json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode()
    return struct.pack("<Q", len(raw)) + raw + payload


class ReaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "fixture.safetensors"

    def write(self, header=None, payload=b"abcd", raw=None):
        self.path.write_bytes(raw if raw is not None else encode({"w": entry()} if header is None else header, payload))
        return self.path

    def test_open_reads_only_header_and_data_is_immutable(self):
        path = self.write({"w": entry(), "__metadata__": {"kind": "unit"}})
        with SafeTensorReader(path) as reader:
            stats = reader.stats()
            self.assertEqual(stats["actual_read_bytes"], 8+stats["header_bytes"])
            self.assertEqual(stats["tensor_read_bytes"], 0)
            self.assertEqual(reader.read_bytes("w", 1, 2), b"bc")
            with self.assertRaises(TypeError):
                reader.tensors["w"] = None
            with self.assertRaises(TypeError):
                reader.metadata["kind"] = "changed"
            with self.assertRaises(AttributeError):
                reader.tensors["w"].shape = (8,)
            changed = reader.stats()
            changed["tensor_count"] = 0
            self.assertEqual(reader.stats()["tensor_count"], 1)
        with self.assertRaises(SafeTensorError):
            reader.read_bytes("w", 0, 1)
        reader.close()

    def test_scalars_empty_tensors_and_unordered_header(self):
        header = {"last": entry("F32", [], [4, 8]),
                  "empty": entry("F32", [0, 3], [4, 4]),
                  "first": entry("U8", [4], [0, 4]),
                  "empty2": entry("F16", [2, 0], [4, 4])}
        payload = b"abcd" + struct.pack("<f", 1.25)
        with SafeTensorReader(self.write(header, payload)) as reader:
            self.assertEqual(reader.read_bytes("empty", 0, 0), b"")
            self.assertEqual(reader.read_bytes("last", 0, 4), struct.pack("<f", 1.25))
            self.assertEqual(reader.tensors["last"].shape, ())

    def test_non_aligned_header_and_empty_model_are_valid(self):
        with SafeTensorReader(self.write({}, b"")) as reader:
            self.assertEqual(reader.stats()["header_bytes"], 2)
            self.assertEqual(len(reader.tensors), 0)
        with SafeTensorReader(self.write(b'{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}  ')) as reader:
            self.assertEqual(reader.read_bytes("w", 0, 4), b"abcd")

    def test_byte_aligned_dtype_sizes_and_little_endian_sentinels(self):
        for dtype, size, payload in (("F8_E4M3", 1, b"\x38"), ("F32", 4, struct.pack("<f", -2.5)),
                                     ("F16", 2, struct.pack("<e", 1.5)), ("BF16", 2, b"\x80\x3f"),
                                     ("I64", 8, struct.pack("<q", -12)), ("BOOL", 1, b"\x01")):
            with self.subTest(dtype=dtype), SafeTensorReader(self.write({"w": entry(dtype, [], [0,size])}, payload)) as reader:
                self.assertEqual(reader.read_bytes("w", 0, size), payload)
                self.assertEqual(reader.tensors["w"].itemsize, size)

    def test_matrix_tile_gathers_strided_rows(self):
        payload = struct.pack("<20f", *range(20))
        with SafeTensorReader(self.write({"w": entry("F32", [4,5], [0,80])}, payload)) as reader:
            tile = reader.read_matrix_tile("w", 1, 2, 2, 3)
            self.assertEqual(struct.unpack("<6f", tile), (7,8,9,12,13,14))
            self.assertEqual(reader.stats()["tensor_read_bytes"], 24)

    def test_read_alignment_bounds_and_no_io_on_invalid_request(self):
        with SafeTensorReader(self.write({"w": entry("F32", [2,2], [0,16])}, b"\0"*16)) as reader:
            for offset, count in ((1,4),(0,3),(-1,4),(0,True),(True,4),(17,0),(12,8),(0,65537)):
                with self.subTest(offset=offset, count=count), self.assertRaises(SafeTensorError):
                    reader.read_bytes("w", offset, count)
            for args in ((0,0,129,1), (0,0,1,0), (1,0,2,1), (-1,0,1,1), (True,0,1,1)):
                with self.subTest(args=args), self.assertRaises(SafeTensorError):
                    reader.read_matrix_tile("w", *args)
            self.assertEqual(reader.stats()["tensor_read_calls"], 0)

    def test_short_prefix_header_length_and_truncated_payload(self):
        cases = [b"", b"1234567", struct.pack("<Q", 0), struct.pack("<Q", MAX_HEADER_BYTES+1),
                 struct.pack("<Q", 100)+b"{}", encode({"w":entry()}, b"abc")]
        for raw in cases:
            with self.subTest(size=len(raw)), self.assertRaises(SafeTensorError):
                SafeTensorReader(self.write(raw=raw))

    def test_json_syntax_duplicate_keys_unicode_and_depth(self):
        headers = [b' {"w":{}}', b'{bad}', b'{"a":1,"a":2}',
                   b'{"w":{"dtype":"U8","shape":[4],"shape":[4],"data_offsets":[0,4]}}',
                   b'{"__metadata__":{"x":NaN}}', b'{"__metadata__":{"x":"\xff"}}',
                   b'{"\\ud800":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}',
                   b'{"__metadata__":{"x":"\\ud800"}}',
                   b'{"x":'+b'['*1100+b'0'+b']'*1100+b'}']
        for raw in headers:
            with self.subTest(header=raw[:50]), self.assertRaises(SafeTensorError):
                SafeTensorReader(self.write(raw, b""))

    def test_tensor_schema_numeric_types_dtype_and_shape_fail(self):
        mutations = [dict(dtype="F8_E4M3FN"), dict(dtype="F8_E4M3FNUZ"), dict(dtype=[]),
                     dict(shape=[True]), dict(shape=[-1]), dict(shape=[4.0]), dict(shape=[2**31]),
                     dict(shape=[1]*9), dict(shape=[2**31-1]*3), dict(shape="4"),
                     dict(data_offsets=[False,4]), dict(data_offsets=[0.0,4]),
                     dict(data_offsets=[4,0]), dict(data_offsets=[0,5]), dict(data_offsets=[0,2**63])]
        for update in mutations:
            value = entry(); value.update(update)
            with self.subTest(update=update), self.assertRaises(SafeTensorError):
                SafeTensorReader(self.write({"w":value}))
        for value in ([], 1, {**entry(), "unknown":1}):
            with self.assertRaises(SafeTensorError):
                SafeTensorReader(self.write({"w":value}))

    def test_gap_overlap_and_trailing_bytes_rejected(self):
        for header, payload in (({"w":entry(offsets=[1,5])}, b"12345"),
                                ({"a":entry(), "b":entry(offsets=[2,6])}, b"123456"),
                                ({"w":entry()}, b"12345")):
            with self.subTest(header=header), self.assertRaises(SafeTensorError):
                SafeTensorReader(self.write(header, payload))

    def test_metadata_profile_rejected(self):
        for value in ([], 3, {"x":1}, {"x":False}, {"x":"a"*(256*1024+1)}):
            with self.subTest(kind=type(value)), self.assertRaises(SafeTensorError):
                SafeTensorReader(self.write({"w":entry(), "__metadata__":value}))

    def test_nan_tensor_bytes_permitted_by_format_layer(self):
        with SafeTensorReader(self.write({"w":entry("F8_E4M3",[2],[0,2])},b"\x7f\xff")) as reader:
            self.assertEqual(reader.read_bytes("w",0,2),b"\x7f\xff")

    def test_common_file_mutation_detected(self):
        self.write()
        with SafeTensorReader(self.path) as reader:
            with self.path.open("ab") as file:
                file.write(b"x")
            with self.assertRaisesRegex(SafeTensorError,"changed"):
                reader.read_bytes("w",0,1)

    def test_header_chunks_and_actual_reads_are_bounded_unbuffered(self):
        header = {"w":entry(), "__metadata__":{"note":"x"*150000}}
        self.write(header)
        real_open = open
        observations = []
        class Wrapped:
            def __init__(self, file): self.file=file
            def __getattr__(self, name): return getattr(self.file,name)
            def read(self,count):
                observations.append(count)
                return self.file.read(count)
        def bounded_open(*args,**kwargs):
            self.assertEqual(kwargs.get("buffering"),0)
            return Wrapped(real_open(*args,**kwargs))
        with patch("glm_local.safetensor_reader.open",side_effect=bounded_open):
            with SafeTensorReader(self.path) as reader:
                self.assertEqual(reader.read_bytes("w",0,4),b"abcd")
                self.assertEqual(reader.stats()["max_actual_read_bytes"],MAX_READ_BYTES)
        self.assertGreater(len(observations),3)
        self.assertTrue(all(0<count<=MAX_READ_BYTES for count in observations))

    def test_partial_read_raises_and_initialization_closes_handle(self):
        self.write()
        stream = self.path.open("rb",buffering=0)
        class ShortRead:
            def __getattr__(self,name): return getattr(stream,name)
            def read(self,count): return stream.read(max(0,count-1))
        with patch("glm_local.safetensor_reader.open",return_value=ShortRead()):
            with self.assertRaisesRegex(SafeTensorError,"Truncated"):
                SafeTensorReader(self.path)
        self.assertTrue(stream.closed)

    def test_metadata_count_caps_fail_without_payload_reads(self):
        header={str(i):entry("F32",[0],[0,0]) for i in range(4)}
        with patch("glm_local.safetensor_reader.MAX_TENSORS",3), self.assertRaises(SafeTensorError):
            SafeTensorReader(self.write(header,b""))
        with patch("glm_local.safetensor_reader.MAX_METADATA_ENTRIES",1), self.assertRaises(SafeTensorError):
            SafeTensorReader(self.write({"__metadata__":{"a":"1","b":"2"}},b""))

    def test_tile_byte_cap_accounts_for_element_width(self):
        with SafeTensorReader(self.write({"w":entry("F64",[128,128],[0,131072])},b"\0"*131072)) as reader:
            with self.assertRaisesRegex(SafeTensorError,"64-KiB"):
                reader.read_matrix_tile("w",0,0,128,128)
            self.assertEqual(reader.stats()["tensor_read_calls"],0)


if __name__ == "__main__":
    unittest.main()
