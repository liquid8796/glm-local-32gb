"""Malformed-input, deterministic-fixture and bounded-reader checks."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from glm_local.mini_spec import SPEC, matrix_shapes, vector_lengths
from glm_local.mini_weights import (
    MAX_FILE_BYTES, MAX_METADATA_BYTES, MAX_READ_BYTES, MiniWeights, write_mini_bundle,
)


def rational_decode(code):
    sign = -1 if code >= 128 else 1
    exponent, mantissa = divmod(code % 128, 8)
    if code % 128 == 127:
        raise ValueError("NaN")
    if exponent == 0:
        return sign * Fraction(mantissa, 512)
    return sign * (1 + Fraction(mantissa, 8)) * Fraction(2) ** (exponent - 7)


class MiniWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_temp = tempfile.TemporaryDirectory(prefix="mini-weights-source-")
        cls.source = Path(cls.source_temp.name) / "fixture"
        cls.original = write_mini_bundle(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.source_temp.cleanup()

    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="mini-weights-test-")
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.bundle = self.folder / "fixture"
        shutil.copytree(self.source, self.bundle)

    def change_manifest(self, mutate):
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        mutate(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def assert_invalid(self):
        with self.assertRaises(ValueError):
            MiniWeights(self.bundle)

    def test_canonical_shapes_offsets_size_and_file_hash(self):
        manifest = self.original
        self.assertEqual(manifest["spec"], SPEC)
        self.assertEqual(set(manifest["vectors"]), set(vector_lengths()))
        offset = 0
        for name, (rows, cols) in matrix_shapes().items():
            entry = manifest["matrices"][name]
            self.assertEqual((entry["offset"], entry["rows"], entry["cols"]),
                             (offset, rows, cols))
            self.assertLessEqual(rows * cols, MAX_READ_BYTES)
            offset += rows * cols
        self.assertEqual(manifest["weight_bytes"], offset)
        self.assertLess(offset, MAX_FILE_BYTES)
        self.assertLess((self.bundle / "manifest.json").stat().st_size, MAX_METADATA_BYTES)
        self.assertEqual(hashlib.sha256((self.bundle / "weights.bin").read_bytes()).hexdigest(),
                         manifest["sha256"])

    def test_same_seed_exact_bytes_and_different_seed_changes_weights(self):
        same = self.folder / "same"
        other = self.folder / "other"
        self.assertEqual(write_mini_bundle(same), self.original)
        write_mini_bundle(other, seed=8)
        for name in ("weights.bin", "manifest.json"):
            self.assertEqual((same / name).read_bytes(), (self.source / name).read_bytes())
        self.assertNotEqual((same / "weights.bin").read_bytes(), (other / "weights.bin").read_bytes())

    def test_uint32_seed_endpoints_and_invalid_values(self):
        for seed in (0, 0xFFFFFFFF):
            directory = self.folder / str(seed)
            write_mini_bundle(directory, seed=seed)
            with MiniWeights(directory) as reader:
                self.assertEqual(reader.manifest["seed"], seed)
        for seed in (-1, 0x100000000, True, 1.5, "7", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                write_mini_bundle(self.folder / "invalid", seed=seed)
        self.assertFalse((self.folder / "invalid").exists())

    def test_exclusive_creation_preserves_existing_files(self):
        before = {path.name: path.read_bytes() for path in self.bundle.iterdir()}
        with self.assertRaises(FileExistsError):
            write_mini_bundle(self.bundle)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.bundle.iterdir()})
        for existing in ("manifest.json", "weights.bin"):
            directory = self.folder / existing
            directory.mkdir()
            (directory / existing).write_bytes(b"user data")
            with self.assertRaises(FileExistsError):
                write_mini_bundle(directory)
            self.assertEqual([path.name for path in directory.iterdir()], [existing])
            self.assertEqual((directory / existing).read_bytes(), b"user data")

    def test_missing_parent_and_directory_is_file(self):
        with self.assertRaises(FileNotFoundError):
            write_mini_bundle(self.folder / "missing" / "child")
        path = self.folder / "file"
        path.write_bytes(b"user data")
        with self.assertRaises(ValueError):
            write_mini_bundle(path)
        self.assertEqual(path.read_bytes(), b"user data")

    def test_embedding_row_matches_independent_rational_decode(self):
        with MiniWeights(self.bundle) as reader:
            embedding = reader.matrix("embed")
            for token in (0, 7, 31):
                before = reader.stats()
                actual = reader.embedding(token)
                expected = [float(rational_decode(code) * Fraction(embedding.scale))
                            for code in embedding.weights[token * 16:(token + 1) * 16]]
                self.assertEqual(actual, expected)
                self.assertEqual(reader.stats()["payload_bytes"] - before["payload_bytes"], 16)
                self.assertEqual(reader.stats()["payload_reads"] - before["payload_reads"], 1)

    def test_all_file_opens_unbuffered_and_all_reads_bounded(self):
        real_open = open
        with patch("builtins.open", wraps=real_open) as opener:
            with MiniWeights(self.bundle) as reader:
                initial = reader.stats()
                self.assertEqual(initial["payload_bytes"], self.original["weight_bytes"])
                for name in matrix_shapes():
                    reader.matrix(name)
                final = reader.stats()
                self.assertLessEqual(final["max_read_bytes"], MAX_READ_BYTES)
                self.assertEqual(final["payload_bytes"], 2 * self.original["weight_bytes"])
                self.assertEqual(final["resident_fp8_cache_bytes"], 0)
                self.assertEqual(final["resident_weight_bytes"], sum(vector_lengths().values()) * 4)
        self.assertEqual(opener.call_count, 2)
        self.assertTrue(all(call.kwargs.get("buffering") == 0 for call in opener.call_args_list))

    def test_repeated_reads_are_uncached_and_matrices_immutable(self):
        with MiniWeights(self.bundle) as reader:
            before = reader.stats()
            first = reader.matrix("layer.1.router")
            second = reader.matrix("layer.1.router")
            self.assertEqual(first, second)
            self.assertEqual(reader.stats()["payload_bytes"] - before["payload_bytes"], 128)
            self.assertEqual(reader.stats()["payload_reads"] - before["payload_reads"], 2)
            with self.assertRaises(FrozenInstanceError):
                first.rows = 99
            with self.assertRaises(TypeError):
                first.weights[0] = 0

    def test_vectors_and_manifest_are_defensive_copies(self):
        with MiniWeights(self.bundle) as reader:
            original = reader.vector("final_norm")
            changed = reader.vector("final_norm")
            changed[0] = 999.0
            manifest = reader.manifest
            manifest["vectors"]["final_norm"][0] = 999.0
            manifest["matrices"]["embed"]["offset"] = 999
            self.assertEqual(reader.vector("final_norm"), original)
            self.assertEqual(len(reader.embedding(0)), 16)

    def test_synthetic_weights_have_modest_magnitudes_and_nonzero_biases(self):
        with MiniWeights(self.bundle) as reader:
            for name in matrix_shapes():
                matrix = reader.matrix(name)
                values = [float(rational_decode(code)) * matrix.scale for code in matrix.weights]
                self.assertTrue(any(value < 0 for value in values))
                self.assertTrue(any(value > 0 for value in values))
                self.assertLessEqual(max(abs(value) for value in values), 0.2)
            self.assertTrue(all(reader.vector("layer.0.index_norm_bias")))
            corrections = reader.vector("layer.1.router_bias")
            self.assertEqual(len(set(corrections)), 4)
            self.assertTrue(all(corrections))

    def test_unknown_names_bad_token_ids_and_closed_reader(self):
        reader = MiniWeights(self.bundle)
        self.addCleanup(reader.close)
        for operation in (reader.matrix, reader.vector):
            with self.assertRaises(KeyError):
                operation("real_model.layers.0.weight")
        for token in (-1, 32, True, 1.0, "0", None):
            with self.subTest(token=token), self.assertRaises(ValueError):
                reader.embedding(token)
        reader.close()
        reader.close()
        for operation in (lambda: reader.matrix("embed"), lambda: reader.embedding(0),
                          lambda: reader.vector("final_norm"), reader.__enter__):
            with self.assertRaisesRegex(ValueError, "closed"):
                operation()

    def test_real_config_and_extra_manifest_fields_rejected(self):
        mutations = [lambda m: m.update(format="safetensors"),
                     lambda m: m.update(model_type="glm_moe_dsa"),
                     lambda m: m["spec"].update(hidden=6144),
                     lambda m: m["spec"].update(layers=True),
                     lambda m: m["spec"].update(max_context=4096),
                     lambda m: m.pop("seed")]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                shutil.copyfile(self.source / "manifest.json", self.bundle / "manifest.json")
                self.change_manifest(mutate)
                self.assert_invalid()

    def test_fixed_path_rejects_traversal_absolute_and_alternate_file(self):
        for path in ("../weights.bin", "C:\\weights.bin", "/tmp/weights.bin", "other.bin", None):
            with self.subTest(path=path):
                self.change_manifest(lambda m: m.update(weight_file=path))
                self.assert_invalid()

    def test_matrix_name_shape_offset_and_schema_tampering(self):
        mutations = [lambda m: m["matrices"].pop("embed"),
                     lambda m: m["matrices"].update(unknown={}),
                     lambda m: m["matrices"]["embed"].update(rows=4096),
                     lambda m: m["matrices"]["embed"].update(cols=True),
                     lambda m: m["matrices"]["lm_head"].update(offset=0),
                     lambda m: m["matrices"]["embed"].update(offset=-1),
                     lambda m: m["matrices"]["embed"].update(dtype="fp16")]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                shutil.copyfile(self.source / "manifest.json", self.bundle / "manifest.json")
                self.change_manifest(mutate)
                self.assert_invalid()

    def test_bad_scales_rejected(self):
        for value in (0, -1.0, float("nan"), float("inf"), 1e100, 0.1, True, "1", None):
            with self.subTest(value=value):
                self.change_manifest(lambda m: m["matrices"]["embed"].update(scale=value))
                self.assert_invalid()

    def test_bad_vector_length_names_values_and_norm_sign(self):
        mutations = [lambda m: m["vectors"].pop("final_norm"),
                     lambda m: m["vectors"].update(unknown=[]),
                     lambda m: m["vectors"].update(final_norm=[1.0]),
                     lambda m: m["vectors"]["final_norm"].__setitem__(0, float("nan")),
                     lambda m: m["vectors"]["final_norm"].__setitem__(0, 0),
                     lambda m: m["vectors"]["final_norm"].__setitem__(0, -1.0),
                     lambda m: m["vectors"]["final_norm"].__setitem__(0, True),
                     lambda m: m["vectors"]["final_norm"].__setitem__(0, 1e100)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                shutil.copyfile(self.source / "manifest.json", self.bundle / "manifest.json")
                self.change_manifest(mutate)
                self.assert_invalid()

    def test_unsupported_manifest_root_duplicate_keys_and_malformed_json(self):
        for text in ("[]", "null", '{"format":"a","format":"b"}', "{", "[" * 2000):
            with self.subTest(text=text[:40]):
                (self.bundle / "manifest.json").write_text(text, encoding="utf-8")
                self.assert_invalid()

    def test_bad_manifest_seed_and_hash_and_size(self):
        changes = [{"seed": True}, {"seed": -1}, {"sha256": "0" * 64}, {"sha256": "bad"},
                   {"sha256": None}, {"weight_bytes": 1}, {"weight_bytes": True}]
        for change in changes:
            with self.subTest(change=change):
                shutil.copyfile(self.source / "manifest.json", self.bundle / "manifest.json")
                self.change_manifest(lambda m: m.update(change))
                self.assert_invalid()

    def test_truncated_extra_and_oversized_payload_rejected(self):
        original = (self.source / "weights.bin").read_bytes()
        for data in (b"", original[:-1], original + b"\x00", b"\x00" * (MAX_FILE_BYTES + 1)):
            with self.subTest(length=len(data)):
                (self.bundle / "weights.bin").write_bytes(data)
                self.assert_invalid()

    def test_empty_and_oversized_manifest_rejected(self):
        for data in (b"", b" " * (MAX_METADATA_BYTES + 1)):
            with self.subTest(length=len(data)):
                (self.bundle / "manifest.json").write_bytes(data)
                self.assert_invalid()

    def test_hash_catches_finite_payload_corruption_before_first_matrix_call(self):
        path = self.bundle / "weights.bin"
        data = bytearray(path.read_bytes())
        data[10] = 0
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            MiniWeights(self.bundle)

    def test_nan_payload_rejected_even_with_matching_hash(self):
        path = self.bundle / "weights.bin"
        for code in (127, 255):
            data = bytearray((self.source / "weights.bin").read_bytes())
            data[10] = code
            path.write_bytes(data)
            self.change_manifest(lambda m: m.update(sha256=hashlib.sha256(data).hexdigest()))
            with self.assertRaisesRegex(ValueError, "NaN"):
                MiniWeights(self.bundle)

    def test_changed_or_truncated_open_payload_rejected(self):
        with MiniWeights(self.bundle) as reader:
            with open(self.bundle / "weights.bin", "ab", buffering=0) as stream:
                stream.write(b"\x00")
            with self.assertRaisesRegex(ValueError, "changed"):
                reader.matrix("embed")
        shutil.copyfile(self.source / "weights.bin", self.bundle / "weights.bin")
        with MiniWeights(self.bundle) as reader:
            with open(self.bundle / "weights.bin", "r+b", buffering=0) as stream:
                stream.truncate(1)
            with self.assertRaises(ValueError):
                reader.embedding(0)

    def test_matrix_and_embedding_digests_detect_change_even_if_stat_gate_bypassed(self):
        for operation in (lambda reader: reader.matrix("embed"), lambda reader: reader.embedding(0)):
            shutil.copyfile(self.source / "weights.bin", self.bundle / "weights.bin")
            with MiniWeights(self.bundle) as reader:
                with open(self.bundle / "weights.bin", "r+b", buffering=0) as stream:
                    stream.write(b"\x01")
                with patch.object(reader, "_check_unchanged"):
                    with self.assertRaisesRegex(ValueError, "bytes changed"):
                        operation(reader)

    def test_short_read_after_initial_validation_rejected(self):
        with MiniWeights(self.bundle) as reader:
            with open(self.bundle / "weights.bin", "r+b", buffering=0) as stream:
                stream.truncate(1)
            with patch.object(reader, "_check_unchanged"):
                with self.assertRaisesRegex(ValueError, "Truncated"):
                    reader.matrix("embed")

    def test_symlink_bundle_member_rejected_when_supported(self):
        path = self.bundle / "weights.bin"
        path.unlink()
        try:
            path.symlink_to(self.source / "weights.bin")
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        self.assert_invalid()


if __name__ == "__main__":
    unittest.main()
