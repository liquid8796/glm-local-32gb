"""Synthetic format validation and comparison with an independent rational oracle."""

from fractions import Fraction
import io
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from glm_local import synthetic_fp8
from glm_local.synthetic_fp8 import (
    BLOCK, MAX_DIM, MAX_READ_BYTES, fixture_vector, iter_tiles,
    read_fixture_info, reference_matvec, write_fixture,
)


def rational_decode(code):
    sign = -1 if code >= 128 else 1
    magnitude = code % 128
    exponent, fraction = divmod(magnitude, 8)
    if magnitude == 127:
        raise ValueError("NaN")
    if exponent == 0:
        return Fraction(sign * fraction, 512)
    return sign * (1 + Fraction(fraction, 8)) * Fraction(2) ** (exponent - 7)


class SyntheticFixtureTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="synthetic-fp8-")
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "fixture.fp8probe"

    def test_default_fixture_size_tiles_and_determinism(self):
        info = write_fixture(self.path)
        self.assertEqual(info, read_fixture_info(self.path))
        self.assertEqual(info["file_bytes"], 24 + 384 * 384 + 9 * 4)
        self.assertEqual(info["file_bytes"], self.path.stat().st_size)
        self.assertEqual(info["max_tile_bytes"], 16384)
        self.assertEqual(info["tile_count"], 9)
        second = self.path.with_name("second.fp8probe")
        write_fixture(second)
        self.assertEqual(self.path.read_bytes(), second.read_bytes())
        self.assertTrue(info["synthetic_only"])
        self.assertEqual(info["weight_format"], "E4M3FN")

    def test_exclusive_creation_preserves_existing_file(self):
        self.path.write_bytes(b"user data")
        with self.assertRaises(FileExistsError):
            write_fixture(self.path)
        self.assertEqual(self.path.read_bytes(), b"user data")
        missing_parent = self.path.parent / "missing" / "fixture.fp8probe"
        with self.assertRaises(FileNotFoundError):
            write_fixture(missing_parent)
        self.assertFalse(missing_parent.parent.exists())

    def test_rectangular_edge_tiles_match_independent_rational_reference(self):
        for rows, cols, seed in ((1, 1, 0), (3, 5, 7), (129, 131, 0xFFFFFFFF)):
            with self.subTest(shape=(rows, cols), seed=seed):
                path = self.path.with_name(f"{rows}-{cols}.fp8probe")
                info = write_fixture(path, rows, cols, seed)
                vector = [Fraction(value) for value in fixture_vector(cols)]
                totals = [Fraction(0) for _ in range(rows)]
                seen = set()
                geometry = []
                for tile in iter_tiles(path):
                    geometry.append((tile.row_start, tile.col_start, tile.rows, tile.cols))
                    self.assertLessEqual(len(tile.weights), MAX_READ_BYTES)
                    self.assertEqual(len(tile.weights), tile.rows * tile.cols)
                    for local_row in range(tile.rows):
                        for local_col in range(tile.cols):
                            row, col = tile.row_start + local_row, tile.col_start + local_col
                            self.assertNotIn((row, col), seen)
                            seen.add((row, col))
                            value = rational_decode(tile.weights[local_row * tile.cols + local_col])
                            totals[row] += value * Fraction(tile.scale) * vector[col]
                self.assertEqual(len(seen), rows * cols)
                expected_geometry = [(r, c, min(BLOCK, rows - r), min(BLOCK, cols - c))
                                     for r in range(0, rows, BLOCK)
                                     for c in range(0, cols, BLOCK)]
                self.assertEqual(geometry, expected_geometry)
                self.assertEqual(len(geometry), info["tile_count"])
                self.assertEqual([float(value) for value in totals],
                                 reference_matvec(rows, cols, seed))

    def test_oracle_never_reads_fixture_or_iterates_tiles(self):
        with patch.object(synthetic_fp8, "open", create=True,
                          side_effect=AssertionError("oracle cannot open files")), \
                patch.object(synthetic_fp8, "iter_tiles",
                             side_effect=AssertionError("oracle cannot use tiles")):
            self.assertEqual(len(reference_matvec(2, 3, 7)), 2)

    def test_mathematical_decoder_matches_rational_values_for_every_finite_code(self):
        for code in range(256):
            with self.subTest(code=code):
                if code in (127, 255):
                    with self.assertRaises(ValueError):
                        synthetic_fp8._oracle_decode(code)
                else:
                    self.assertEqual(synthetic_fp8._oracle_decode(code), float(rational_decode(code)))
        self.assertEqual(math.copysign(1, synthetic_fp8._oracle_decode(128)), -1)

    def test_invalid_dimensions_and_seeds_fail_before_creating_file(self):
        for changes in ({"rows": 0}, {"rows": MAX_DIM + 1}, {"rows": True},
                        {"cols": -1}, {"cols": 1.5}, {"cols": MAX_DIM + 1},
                        {"seed": -1}, {"seed": 2**32}, {"seed": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                write_fixture(self.path, **changes)
            self.assertFalse(self.path.exists())
        for count in (0, MAX_DIM + 1, True, 2.5):
            with self.subTest(vector=count), self.assertRaises(ValueError):
                fixture_vector(count)
        with self.assertRaises(ValueError):
            reference_matvec(MAX_DIM + 1, 1, 7)

    def test_every_read_is_explicit_unbuffered_and_within_tile_bound(self):
        write_fixture(self.path, 129, 129)
        real_open = open
        sizes = []

        class TrackedReader:
            def __init__(self, stream):
                self.stream = stream

            def read(self, size=-1):
                self_test.assertGreater(size, 0)
                self_test.assertLessEqual(size, MAX_READ_BYTES)
                sizes.append(size)
                return self.stream.read(size)

            def fileno(self):
                return self.stream.fileno()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

        self_test = self

        def tracked_open(*args, **kwargs):
            self.assertEqual(kwargs.get("buffering"), 0)
            return TrackedReader(real_open(*args, **kwargs))

        with patch.object(synthetic_fp8, "open", create=True, side_effect=tracked_open):
            read_fixture_info(self.path)
            self.assertEqual(len(list(iter_tiles(self.path))), 4)
        self.assertIn(MAX_READ_BYTES, sizes)
        self.assertEqual(max(sizes), MAX_READ_BYTES)

    def test_short_reads_fail_even_when_some_bytes_are_returned(self):
        with self.assertRaisesRegex(ValueError, "Truncated"):
            synthetic_fp8._read_exact(io.BytesIO(b"ab"), 3)
        for size in (0, -1, MAX_READ_BYTES + 1, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                synthetic_fp8._read_exact(io.BytesIO(b"ab"), size)

    def test_malformed_or_oversized_header_is_rejected(self):
        malformed = [b"", b"FP8PROBE", struct.pack("<8sIIII", b"BADMAGIC", 1, 1, 1, 7),
                     struct.pack("<8sIIII", b"FP8PROBE", 2, 1, 1, 7),
                     struct.pack("<8sIIII", b"FP8PROBE", 1, 0, 1, 7),
                     struct.pack("<8sIIII", b"FP8PROBE", 1, 0xFFFFFFFF, 0xFFFFFFFF, 7)]
        for data in malformed:
            with self.subTest(data=data):
                self.path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_fixture_info(self.path)
                with self.assertRaises(ValueError):
                    list(iter_tiles(self.path))

    def test_truncation_and_trailing_bytes_are_rejected(self):
        write_fixture(self.path, 3, 4)
        valid = self.path.read_bytes()
        for corrupted in (valid[:23], valid[:-1], valid + b"trailing"):
            self.path.write_bytes(corrupted)
            with self.assertRaises(ValueError):
                read_fixture_info(self.path)
            with self.assertRaises(ValueError):
                list(iter_tiles(self.path))

    def test_mutation_after_initial_size_check_is_detected_during_iteration(self):
        write_fixture(self.path, 129, 1)
        tiles = iter_tiles(self.path)
        self.addCleanup(tiles.close)
        self.assertEqual(next(tiles).rows, 128)
        with open(self.path, "r+b") as stream:
            stream.truncate(24 + 4 + 128)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            next(tiles)

        appended_path = self.path.with_name("appended.fp8probe")
        write_fixture(appended_path, 1, 1)
        tiles = iter_tiles(appended_path)
        self.addCleanup(tiles.close)
        next(tiles)
        with open(appended_path, "ab") as stream:
            stream.write(b"extra")
        with self.assertRaisesRegex(ValueError, "trailing"):
            next(tiles)

    def test_invalid_scales_and_nan_encodings_are_rejected_before_yield(self):
        write_fixture(self.path, 1, 2)
        valid = self.path.read_bytes()
        for scale in (0.0, -1.0, math.nan, math.inf, -math.inf):
            self.path.write_bytes(valid[:24] + struct.pack("<f", scale) + valid[28:])
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "scale"):
                next(iter_tiles(self.path))
        for code in (127, 255):
            self.path.write_bytes(valid[:28] + bytes([code]) + valid[29:])
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "NaN"):
                next(iter_tiles(self.path))


if __name__ == "__main__":
    unittest.main()
