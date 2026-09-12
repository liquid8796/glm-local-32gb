"""Independent byte fixtures for the explicit FP8 weight/scale adapter."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import math
import struct
from types import SimpleNamespace
import unittest

from glm_local.fp8_blocks import BLOCK, Block, FP8BlockMatrix


WEIGHT_NAME = "arbitrary matrix"
SCALE_NAME = "explicit multiplier"


def tensor_info(dtype, shape, itemsize):
    count = math.prod(shape)
    return SimpleNamespace(dtype=dtype, shape=shape, itemsize=itemsize,
                           nbytes=count * itemsize, data_offsets=(0, count * itemsize))


def weight_byte(row, col):
    # Deliberately asymmetric coordinates; finite signed E4M3FN values only.
    return ((row * 11 + col * 5 + 3) % 127) | (128 if (row + col) % 3 == 0 else 0)


class FakeReader:
    """No full weight storage; trace every requested scale span and rectangle."""

    def __init__(self, rows=256, cols=384, scales=None):
        self.rows, self.cols = rows, cols
        scale_shape = ((rows + 127) // 128, (cols + 127) // 128)
        self.tensors = {
            WEIGHT_NAME: tensor_info("F8_E4M3", (rows, cols), 1),
            SCALE_NAME: tensor_info("F32", scale_shape, 4),
        }
        scale_count = math.prod(scale_shape)
        self.scales = struct.pack("<" + "f" * scale_count,
                                  *(scales if scales is not None else range(1, scale_count + 1)))
        self.reads = []
        self.scale_override = None
        self.weight_override = None

    def read_bytes(self, name, offset, count):
        if name != SCALE_NAME or count != 4 or offset % 4:
            raise AssertionError("Only one aligned scale may be read as raw bytes")
        self.reads.append(("scale", name, offset, count))
        return self.scales[offset:offset + count] if self.scale_override is None else self.scale_override

    def read_matrix_tile(self, name, row, col, rows, cols):
        if (name != WEIGHT_NAME or not 1 <= rows <= 128 or not 1 <= cols <= 128
                or row < 0 or col < 0 or row + rows > self.rows or col + cols > self.cols):
            raise AssertionError("Only a valid bounded weight tile may be read")
        self.reads.append(("weight", name, row, col, rows, cols))
        if self.weight_override is not None:
            return self.weight_override
        return bytes(weight_byte(r, c) for r in range(row, row + rows) for c in range(col, col + cols))


def matrix(reader):
    return FP8BlockMatrix(reader, WEIGHT_NAME, SCALE_NAME)


class FP8BlockMatrixTests(unittest.TestCase):
    def test_constructor_reads_only_metadata_with_explicit_arbitrary_names(self):
        reader = FakeReader()
        adapter = matrix(reader)
        self.assertEqual((adapter.rows, adapter.cols, adapter.block_rows, adapter.block_cols),
                         (256, 384, 2, 3))
        self.assertEqual(BLOCK, 128)
        self.assertEqual(reader.reads, [])

    def test_non_square_grid_uses_row_major_scales_and_packed_weight_rows(self):
        reader = FakeReader(scales=[0.25, 0.5, 2, 3, 5, 7])
        adapter = matrix(reader)
        block = adapter.read_block(1, 2)
        self.assertIsInstance(block, Block)
        self.assertEqual((block.row_start, block.col_start, block.rows, block.cols, block.scale),
                         (128, 256, 128, 128, 7.0))
        self.assertEqual(reader.reads, [("scale", SCALE_NAME, 20, 4),
                                       ("weight", WEIGHT_NAME, 128, 256, 128, 128)])
        expected = bytes(weight_byte(r, c) for r in range(128, 256) for c in range(256, 384))
        self.assertEqual(block.weights, expected)
        self.assertEqual([item.scale for item in adapter.iter_blocks()], [0.25, 0.5, 2, 3, 5, 7])

    def test_ragged_matrix_visits_every_coordinate_once_and_preserves_edge_shape(self):
        reader = FakeReader(257, 259)
        blocks = list(matrix(reader).iter_blocks())
        self.assertEqual(len(blocks), 9)
        expected_geometry = [(r, c, min(128, 257 - r), min(128, 259 - c))
                             for r in (0, 128, 256) for c in (0, 128, 256)]
        self.assertEqual([(b.row_start, b.col_start, b.rows, b.cols) for b in blocks], expected_geometry)
        seen = set()
        for block in blocks:
            self.assertEqual(len(block.weights), block.rows * block.cols)
            self.assertLessEqual(len(block.weights), 16384)
            for r in range(block.rows):
                for c in range(block.cols):
                    coordinate = (block.row_start + r, block.col_start + c)
                    self.assertNotIn(coordinate, seen)
                    seen.add(coordinate)
                    self.assertEqual(block.weights[r * block.cols + c], weight_byte(*coordinate))
        self.assertEqual(len(seen), 257 * 259)
        self.assertEqual([(b.rows, b.cols) for b in blocks[-3:]], [(1, 128), (1, 128), (1, 3)])
        self.assertEqual([call[2] for call in reader.reads if call[0] == "scale"], list(range(0, 36, 4)))

    def test_little_endian_scale_is_returned_unchanged_as_multiplicative_factor(self):
        reader = FakeReader(1, 1)
        reader.scale_override = b"\x00\x00\x00\x3e"  # F32 0.125
        reader.weight_override = b"\x40"  # E4M3FN 2.0
        block = matrix(reader).read_block(0, 0)
        # Consumer dequantization uses decoded weight * returned scale, never reciprocal.
        self.assertEqual(Fraction(2) * Fraction(block.scale), Fraction(1, 4))
        self.assertEqual(block.scale, 0.125)
        self.assertEqual(block.weights, b"\x40")

    def test_smallest_positive_f32_subnormal_scale_is_valid(self):
        reader = FakeReader(1, 1)
        reader.scale_override = b"\x01\x00\x00\x00"
        block = matrix(reader).read_block(0, 0)
        self.assertEqual(block.scale, math.ldexp(1.0, -149))

    def test_largest_finite_f32_scale_is_preserved(self):
        reader = FakeReader(1, 1)
        reader.scale_override = b"\xff\xff\x7f\x7f"
        block = matrix(reader).read_block(0, 0)
        self.assertEqual(block.scale, (2 - 2 ** -23) * 2 ** 127)

    def test_all_finite_fp8_codes_are_preserved_including_signed_zero(self):
        reader = FakeReader(2, 127)
        reader.weight_override = bytes(value for value in range(256) if value not in (127, 255))
        block = matrix(reader).read_block(0, 0)
        self.assertEqual(block.weights, reader.weight_override)
        self.assertIn(0, block.weights)
        self.assertIn(128, block.weights)
        self.assertIn(126, block.weights)
        self.assertIn(254, block.weights)

    def test_block_is_frozen_and_weights_are_immutable(self):
        block = matrix(FakeReader(1, 1)).read_block(0, 0)
        with self.assertRaises(FrozenInstanceError):
            block.scale = 5
        with self.assertRaises(TypeError):
            block.weights[0] = 1

    def test_explicit_names_must_exist_and_be_distinct_nonempty_strings(self):
        for weight_name, scale_name in (("", SCALE_NAME), (WEIGHT_NAME, ""),
                                       (None, SCALE_NAME), (WEIGHT_NAME, True),
                                       (WEIGHT_NAME, WEIGHT_NAME), ("missing", SCALE_NAME),
                                       (WEIGHT_NAME, "missing")):
            reader = FakeReader()
            with self.subTest(names=(weight_name, scale_name)), self.assertRaises(ValueError):
                FP8BlockMatrix(reader, weight_name, scale_name)
            self.assertEqual(reader.reads, [])

    def test_reject_wrong_weight_dtypes_including_nonstandard_and_fnuz_tags(self):
        for dtype in ("F8_E4M3FN", "F8_E4M3FNUZ", "F8_E5M2", "F32", "U8", None):
            reader = FakeReader()
            reader.tensors[WEIGHT_NAME].dtype = dtype
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "F8_E4M3"):
                matrix(reader)
            self.assertEqual(reader.reads, [])

    def test_reject_wrong_scale_dtypes(self):
        for dtype in ("BF16", "F16", "F64", "F8_E8M0", "U8", None):
            reader = FakeReader()
            reader.tensors[SCALE_NAME].dtype = dtype
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "F32"):
                matrix(reader)
            self.assertEqual(reader.reads, [])

    def test_reject_empty_scalar_wrong_rank_and_noninteger_matrix_dimensions(self):
        for name in (WEIGHT_NAME, SCALE_NAME):
            for shape in ((), (1,), (1, 1, 1), (0, 2), (2, 0), (-1, 2),
                          (True, 2), (2, False), (1.5, 2), (2, "3"), [2, 3], None):
                reader = FakeReader()
                reader.tensors[name].shape = shape
                with self.subTest(name=name, shape=shape), self.assertRaisesRegex(ValueError, "dimensions"):
                    matrix(reader)
                self.assertEqual(reader.reads, [])

    def test_reject_wrong_scale_grid_including_transpose_and_floor_edge_grid(self):
        for rows, cols, shape in ((256, 384, (3, 2)), (256, 384, (2, 2)),
                                 (129, 131, (1, 1)), (128, 128, (2, 2))):
            reader = FakeReader(rows, cols)
            reader.tensors[SCALE_NAME].shape = shape
            with self.subTest(weight=(rows, cols), scales=shape), self.assertRaisesRegex(ValueError, "Scale shape"):
                matrix(reader)
            self.assertEqual(reader.reads, [])

    def test_reject_inconsistent_metadata_byte_counts_and_item_sizes(self):
        for name in (WEIGHT_NAME, SCALE_NAME):
            for field, value in (("nbytes", 0), ("nbytes", -1), ("itemsize", 8)):
                reader = FakeReader()
                setattr(reader.tensors[name], field, value)
                with self.subTest(name=name, field=field), self.assertRaisesRegex(ValueError, "byte size"):
                    matrix(reader)
                self.assertEqual(reader.reads, [])

    def test_reject_bad_block_coordinates_before_any_payload_read(self):
        reader = FakeReader()
        adapter = matrix(reader)
        invalid = [(value, 0) for value in (-1, 2, 100, True, False, 0.0, "0", None)]
        invalid += [(0, value) for value in (-1, 3, 100, True, False, 0.0, "0", None)]
        for br, bc in invalid:
            with self.subTest(index=(br, bc)), self.assertRaisesRegex(ValueError, "Block"):
                adapter.read_block(br, bc)
        self.assertEqual(reader.reads, [])

    def test_invalid_scales_fail_before_weight_read(self):
        for value in (0.0, -0.0, -1.0, -math.ldexp(1.0, -149), math.inf, -math.inf, math.nan):
            reader = FakeReader(1, 1)
            reader.scale_override = struct.pack("<f", value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive and finite"):
                matrix(reader).read_block(0, 0)
            self.assertEqual(reader.reads, [("scale", SCALE_NAME, 0, 4)])

    def test_nan_weight_encodings_are_rejected_before_a_block_is_returned(self):
        for value in (127, 255):
            reader = FakeReader(1, 1)
            reader.weight_override = bytes([value])
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "NaN"):
                matrix(reader).read_block(0, 0)

    def test_truncated_or_wrong_type_scale_read_is_rejected(self):
        for payload in (b"", b"\x00" * 3, b"\x00" * 5, "abcd", bytearray(b"abcd")):
            reader = FakeReader(1, 1)
            reader.scale_override = payload
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "four bytes"):
                matrix(reader).read_block(0, 0)
            self.assertEqual(reader.reads, [("scale", SCALE_NAME, 0, 4)])

    def test_truncated_or_wrong_type_weight_read_is_rejected(self):
        for payload in (b"", b"\x00" * 2, "a", bytearray(b"a")):
            reader = FakeReader(1, 1)
            reader.weight_override = payload
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "byte count"):
                matrix(reader).read_block(0, 0)

    def test_iteration_is_lazy_and_has_no_prefetch_after_early_close(self):
        reader = FakeReader()
        iterator = matrix(reader).iter_blocks()
        self.assertEqual(reader.reads, [])
        self.assertEqual(next(iterator).scale, 1.0)
        self.assertEqual(reader.reads, [("scale", SCALE_NAME, 0, 4),
                                       ("weight", WEIGHT_NAME, 0, 0, 128, 128)])
        iterator.close()
        self.assertEqual(len(reader.reads), 2)

    def test_large_metadata_dimensions_still_read_just_one_edge_block(self):
        reader = FakeReader(1, 1)
        # Avoid even allocating a scale grid for a huge logical matrix.
        dimension = 2 ** 31 - 1
        grid = (dimension + 127) // 128
        reader.rows, reader.cols = dimension, dimension
        reader.tensors[WEIGHT_NAME] = tensor_info("F8_E4M3", (dimension, dimension), 1)
        reader.tensors[SCALE_NAME] = tensor_info("F32", (grid, grid), 4)
        reader.scale_override = b"\x00\x00\x80\x3f"
        adapter = matrix(reader)
        self.assertEqual(reader.reads, [])
        block = adapter.read_block(grid - 1, grid - 1)
        self.assertEqual((block.rows, block.cols), (127, 127))
        self.assertEqual(len(block.weights), 127 * 127)
        self.assertEqual(reader.reads[0], ("scale", SCALE_NAME, (grid * grid - 1) * 4, 4))
        self.assertEqual(len(reader.reads), 2)

    def test_reader_io_failure_is_not_hidden_or_retried(self):
        reader = FakeReader(1, 1)

        def unavailable(*args):
            raise OSError("reader is closed")

        reader.read_bytes = unavailable
        with self.assertRaisesRegex(OSError, "reader is closed"):
            matrix(reader).read_block(0, 0)
        self.assertEqual(reader.reads, [])


if __name__ == "__main__":
    unittest.main()
