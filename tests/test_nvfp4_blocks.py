"""Independent bounded synthetic byte fixtures for NVFP4 storage geometry."""
from dataclasses import FrozenInstanceError
import math
import struct
from types import SimpleNamespace
import unittest

from glm_local.nvfp4_blocks import NVFP4BlockMatrix, decode_e4m3_scale


NAMES = ("arbitrary packed matrix", "local scales", "global scale", "activation scale")


def info(dtype, shape, size):
    return SimpleNamespace(dtype=dtype, shape=shape, itemsize=size, nbytes=math.prod(shape) * size)


def packed_byte(row, bytecol):
    return ((3 * row + bytecol * 7) % 16) | (((row * 11 + bytecol * 5 + 1) % 16) << 4)


def scale_byte(row, blockcol):
    return (row * 9 + blockcol * 13) % 127


class Reader:
    def __init__(self, rows=129, cols=144):
        self.rows, self.cols = rows, cols
        self.tensors = dict(zip(NAMES, (info("U8", (rows, cols // 2), 1),
            info("F8_E4M3", (rows, cols // 16), 1), info("F32", (), 4), info("F32", (), 4))))
        self.payloads = {NAMES[2]: struct.pack("<f", 0.125), NAMES[3]: struct.pack("<f", 3.5)}
        self.overrides, self.reads = {}, []

    def read_bytes(self, name, offset, count):
        self.reads.append(("bytes", name, offset, count))
        if name not in self.payloads or offset != 0 or count != 4:
            raise AssertionError("Only one scalar may be read as raw bytes")
        return self.overrides.get(name, self.payloads[name])

    def read_matrix_tile(self, name, row, col, rows, cols):
        self.reads.append(("tile", name, row, col, rows, cols))
        shape = self.tensors[name].shape
        if (name not in NAMES[:2] or not 1 <= rows <= 128 or not 1 <= cols <= 64
                or row < 0 or col < 0 or row + rows > shape[0] or col + cols > shape[1]):
            raise AssertionError("Unbounded or incorrect tile request")
        if name in self.overrides:
            return self.overrides[name]
        value = packed_byte if name == NAMES[0] else scale_byte
        return bytes(value(r, c) for r in range(row, row + rows) for c in range(col, col + cols))


def matrix(reader):
    return NVFP4BlockMatrix(reader, *NAMES)


class NVFP4BlockTests(unittest.TestCase):
    def test_constructor_is_metadata_only_and_shapes_are_logical(self):
        reader = Reader()
        adapter = matrix(reader)
        self.assertEqual((adapter.rows, adapter.cols, adapter.block_rows, adapter.block_cols), (129, 144, 2, 2))
        self.assertEqual(adapter.activation_quantization, "none")
        self.assertEqual(reader.reads, [])

    def test_ragged_rows_and_columns_use_packed_and_scale_coordinates(self):
        reader = Reader()
        blocks = list(matrix(reader).iter_blocks())
        self.assertEqual([(b.row_start, b.col_start, b.rows, b.cols) for b in blocks],
                         [(0, 0, 128, 128), (0, 128, 128, 16), (128, 0, 1, 128), (128, 128, 1, 16)])
        seen = set()
        for block in blocks:
            self.assertLessEqual(len(block.weights), 8192)
            self.assertLessEqual(len(block.scales), 1024)
            self.assertEqual((block.global_scale, block.input_scale), (0.125, 3.5))
            for r in range(block.rows):
                for c in range(block.cols // 2):
                    coordinate = (block.row_start + r, block.col_start // 2 + c)
                    self.assertNotIn(coordinate, seen)
                    seen.add(coordinate)
                    self.assertEqual(block.weights[r * (block.cols // 2) + c], packed_byte(*coordinate))
                for c in range(block.cols // 16):
                    self.assertEqual(block.scales[r * (block.cols // 16) + c],
                        scale_byte(block.row_start + r, block.col_start // 16 + c))
        self.assertEqual(len(seen), 129 * 72)
        self.assertEqual(reader.reads[-2:], [("tile", NAMES[1], 128, 8, 1, 1),
                                            ("tile", NAMES[0], 128, 64, 1, 8)])

    def test_input_calibration_is_kept_separate_and_not_inverted(self):
        reader = Reader(1, 16)
        first = matrix(reader).read_block(0, 0)
        reader.payloads[NAMES[3]] = struct.pack("<f", 1000)
        second = matrix(reader).read_block(0, 0)
        self.assertEqual((first.weights, first.scales, first.global_scale),
                         (second.weights, second.scales, second.global_scale))
        self.assertEqual(second.input_scale, 1000)

    def test_zero_and_all_finite_unsigned_e4m3_scales_have_exact_values(self):
        for code in range(127):
            exponent, fraction = divmod(code, 8)
            expected = fraction / 512 if exponent == 0 else (1 + fraction / 8) * 2 ** (exponent - 7)
            self.assertEqual(decode_e4m3_scale(code), expected)
        self.assertEqual(decode_e4m3_scale(126), 448.0)
        reader = Reader(1, 16)
        reader.overrides[NAMES[1]] = b"\0"
        self.assertEqual(matrix(reader).read_block(0, 0).scales, b"\0")

    def test_invalid_scale_encodings_rejected_before_weight_read(self):
        for code in range(127, 256):
            reader = Reader(1, 16)
            reader.overrides[NAMES[1]] = bytes([code])
            with self.subTest(code=code), self.assertRaises(ValueError):
                matrix(reader).read_block(0, 0)
            self.assertFalse(any(call[1] == NAMES[0] for call in reader.reads))
        for code in (-1, 256, True, 1.0, None):
            with self.subTest(code=code), self.assertRaises(ValueError):
                decode_e4m3_scale(code)

    def test_scalar_payloads_are_positive_finite_f32_and_read_exactly(self):
        for name in NAMES[2:]:
            for value in (0.0, -0.0, -1.0, math.nan, math.inf, -math.inf):
                reader = Reader(1, 16)
                reader.payloads[name] = struct.pack("<f", value)
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "positive and finite"):
                    matrix(reader).read_block(0, 0)
            for payload in (b"", b"\0" * 3, b"\0" * 5, bytearray(b"\0" * 4), None):
                reader = Reader(1, 16)
                reader.overrides[name] = payload
                with self.subTest(name=name, payload=payload), self.assertRaisesRegex(ValueError, "exactly four"):
                    matrix(reader).read_block(0, 0)

    def test_positive_subnormal_global_scale_and_scalar_one_shape_are_supported(self):
        reader = Reader(1, 16)
        for name in NAMES[2:]:
            reader.tensors[name].shape = (1,)
            reader.payloads[name] = b"\x01\0\0\0"
        block = matrix(reader).read_block(0, 0)
        self.assertEqual((block.global_scale, block.input_scale), (2 ** -149, 2 ** -149))

    def test_invalid_names_fail_before_read(self):
        for index in range(4):
            for value in (None, "", True, "missing", NAMES[(index + 1) % 4]):
                names = list(NAMES)
                names[index] = value
                reader = Reader()
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    NVFP4BlockMatrix(reader, *names)
                self.assertEqual(reader.reads, [])

    def test_bad_dtypes_shapes_sizes_and_transposed_or_swizzled_layouts_fail(self):
        mutations = [(NAMES[0], "dtype", "F8_E4M3"), (NAMES[0], "shape", (129, 71)),
                     (NAMES[0], "shape", (1, 129, 72)), (NAMES[0], "shape", (True, 72)),
                     (NAMES[1], "dtype", "U8"), (NAMES[1], "shape", (9, 129)),
                     (NAMES[1], "shape", (1161,)), (NAMES[1], "shape", (129, 8))]
        for name in NAMES[2:]:
            mutations.extend([(name, "dtype", "F16"), (name, "shape", (1, 1)),
                              (name, "shape", (2,)), (name, "shape", (True,)), (name, "shape", (1.0,))])
        for name in NAMES:
            mutations.extend([(name, "nbytes", 0), (name, "itemsize", 2)])
        for name, field, value in mutations:
            reader = Reader()
            setattr(reader.tensors[name], field, value)
            with self.subTest(name=name, field=field, value=value), self.assertRaises(ValueError):
                matrix(reader)
            self.assertEqual(reader.reads, [])

    def test_invalid_tile_coordinates_do_not_read(self):
        reader = Reader()
        adapter = matrix(reader)
        for coordinate in ((-1, 0), (0, -1), (2, 0), (0, 2), (True, 0), (0, 1.0)):
            with self.subTest(coordinate=coordinate), self.assertRaises(ValueError):
                adapter.read_block(*coordinate)
        self.assertEqual(reader.reads, [])

    def test_truncated_or_mutable_tile_payloads_fail(self):
        for name, expected_length in ((NAMES[0], 8), (NAMES[1], 1)):
            for payload in (b"", b"\0" * (expected_length + 1), bytearray(expected_length), None):
                reader = Reader(1, 16)
                reader.overrides[name] = payload
                with self.subTest(name=name, payload=payload), self.assertRaisesRegex(ValueError, "byte count or type"):
                    matrix(reader).read_block(0, 0)

    def test_blocks_are_immutable(self):
        block = matrix(Reader(1, 16)).read_block(0, 0)
        with self.assertRaises(FrozenInstanceError):
            block.global_scale = 1
        with self.assertRaises(TypeError):
            block.weights[0] = 0


if __name__ == "__main__":
    unittest.main()
