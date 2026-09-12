import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("safetensors") is None:
    if os.environ.get("GLM_TEST_STORAGE") == "1":
        raise RuntimeError("Requested storage tests require safetensors")
    raise unittest.SkipTest("Optional sharded fixture tests require safetensors")

from glm_local.mini_safetensors import (
    MiniSafetensorWeights, SHARD_NAMES, MARKER, matrix_names, vector_name,
    tensor_contract, write_mini_shards,
)
from glm_local.mini_spec import matrix_shapes, vector_lengths
from glm_local.mini_weights import MiniWeights, write_mini_bundle
from glm_local.safetensor_reader import SafeTensorError
from glm_local.sharded_safetensors import INDEX_NAME
from sharded_test_helpers import write_safe, read_safe, read_index, write_index


class MiniShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_temp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.source_temp.name)
        write_mini_bundle(cls.source, seed=19)

    @classmethod
    def tearDownClass(cls):
        cls.source_temp.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.exported = write_mini_shards(self.source, self.path)

    def replace_tensor(self, name, *, dtype=None, shape=None, payload=None):
        index = read_index(self.path)
        path = self.path / index["weight_map"][name]
        tensors, metadata = read_safe(path)
        old_dtype, old_shape, old_payload = tensors[name]
        tensors[name] = (old_dtype if dtype is None else dtype,
                         old_shape if shape is None else shape,
                         old_payload if payload is None else payload)
        write_safe(path, tensors, metadata)

    def test_all_matrix_bytes_shapes_and_scales_equal_original_private_fixture(self):
        with MiniWeights(self.source) as old, MiniSafetensorWeights(self.path) as new:
            for name in matrix_shapes():
                with self.subTest(name=name):
                    self.assertEqual(old.matrix(name), new.matrix(name))

    def test_all_vectors_and_all_embedding_rows_equal_original(self):
        with MiniWeights(self.source) as old, MiniSafetensorWeights(self.path) as new:
            for name in vector_lengths():
                self.assertEqual(old.vector(name), new.vector(name))
            for token in range(32):
                self.assertEqual(old.embedding(token), new.embedding(token))

    def test_every_weight_scale_pair_crosses_shards(self):
        index = read_index(self.path)
        for name in matrix_shapes():
            weight, scale = matrix_names(name)
            self.assertNotEqual(index["weight_map"][weight], index["weight_map"][scale])
        self.assertEqual(self.exported["tensor_count"], 80)
        self.assertEqual(self.exported["tensor_payload_bytes"], 10072)
        self.assertEqual(self.exported["shard_count"], 4)

    def test_no_payload_read_at_open_no_weight_cache_and_bounded_handles(self):
        with MiniSafetensorWeights(self.path) as reader:
            self.assertEqual(reader.stats()["tensor_read_bytes"], 0)
            for name in matrix_shapes():
                reader.matrix(name)
            stats = reader.stats()
            self.assertEqual(stats["resident_weight_bytes"], 0)
            self.assertEqual(stats["resident_fp8_cache_bytes"], 0)
            self.assertEqual(stats["resident_vector_value_bytes"], 0)
            self.assertLessEqual(stats["peak_open_shards"], 2)
            self.assertLessEqual(stats["max_actual_read_bytes"], 65536)
            self.assertEqual(stats["cross_shard_pairs"], 34)

    def test_embedding_reads_one_row_and_one_scale_not_whole_matrix(self):
        with MiniSafetensorWeights(self.path) as reader:
            reader.embedding(31)
            self.assertEqual(reader.stats()["tensor_read_bytes"], 16 + 4)

    def test_vector_values_are_read_on_demand_not_cached(self):
        with MiniSafetensorWeights(self.path) as reader:
            value = reader.vector("final_norm")
            value[0] = 999.0
            self.assertNotEqual(reader.vector("final_norm")[0], 999.0)
            self.assertEqual(reader.stats()["tensor_read_bytes"], 128)

    def test_invalid_matrix_vector_and_token_inputs(self):
        with MiniSafetensorWeights(self.path) as reader:
            with self.assertRaises(KeyError):
                reader.matrix("unknown")
            with self.assertRaises(KeyError):
                reader.vector("unknown")
            for token in (-1, 32, True, 3.5, "7", None):
                with self.subTest(token=token), self.assertRaises(ValueError):
                    reader.embedding(token)

    def test_weight_dtype_mismatch_is_rejected(self):
        self.replace_tensor(matrix_names("embed")[0], dtype="U8")
        with self.assertRaisesRegex(SafeTensorError, "dtype/shape"):
            MiniSafetensorWeights(self.path)

    def test_weight_shape_mismatch_is_rejected_even_when_nbytes_equal(self):
        self.replace_tensor(matrix_names("embed")[0], shape=(16, 32))
        with self.assertRaisesRegex(SafeTensorError, "dtype/shape"):
            MiniSafetensorWeights(self.path)

    def test_scale_shape_must_be_a_two_dimensional_block_grid(self):
        self.replace_tensor(matrix_names("embed")[1], shape=(1,))
        with self.assertRaisesRegex(SafeTensorError, "dtype/shape"):
            MiniSafetensorWeights(self.path)

    def test_scale_dtype_cannot_be_an_equal_sized_integer(self):
        self.replace_tensor(matrix_names("embed")[1], dtype="U32")
        with self.assertRaisesRegex(SafeTensorError, "dtype/shape"):
            MiniSafetensorWeights(self.path)

    def test_nonfinite_or_nonpositive_scale_fails_when_requested(self):
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            self.replace_tensor(matrix_names("embed")[1], payload=struct.pack("<f", scale))
            with self.subTest(scale=scale), MiniSafetensorWeights(self.path) as reader:
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    reader.matrix("embed")
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    reader.embedding(0)

    def test_fp8_nan_fails_when_matrix_or_embedding_requested(self):
        self.replace_tensor(matrix_names("embed")[0], payload=b"\x7f" + bytes(511))
        with MiniSafetensorWeights(self.path) as reader:
            with self.assertRaisesRegex(ValueError, "NaN"):
                reader.matrix("embed")
            with self.assertRaisesRegex(ValueError, "NaN"):
                reader.embedding(0)

    def test_nonfinite_vector_fails_when_requested(self):
        self.replace_tensor(vector_name("final_norm"), payload=struct.pack("<16f", float("inf"), *([1.] * 15)))
        with MiniSafetensorWeights(self.path) as reader, self.assertRaisesRegex(ValueError, "finite"):
            reader.vector("final_norm")

    def test_missing_scale_name_is_not_guessed(self):
        index = read_index(self.path)
        name = matrix_names("embed")[1]
        shard = index["weight_map"].pop(name)
        index["weight_map"]["renamed_scale"] = shard
        tensors, metadata = read_safe(self.path / shard)
        tensors["renamed_scale"] = tensors.pop(name)
        write_safe(self.path / shard, tensors, metadata)
        write_index(self.path, index)
        with self.assertRaisesRegex(SafeTensorError, "names differ"):
            MiniSafetensorWeights(self.path)

    def test_wrong_index_marker_is_not_a_real_checkpoint_entrypoint(self):
        index = read_index(self.path)
        index["metadata"]["synthetic_fixture"] = "a-real-model"
        write_index(self.path, index)
        with self.assertRaisesRegex(SafeTensorError, "metadata"):
            MiniSafetensorWeights(self.path)

    def test_mixed_shard_markers_are_rejected(self):
        path = self.path / SHARD_NAMES[0]
        tensors, metadata = read_safe(path)
        metadata["seed"] = "123"
        write_safe(path, tensors, metadata)
        with self.assertRaisesRegex(SafeTensorError, "markers disagree"):
            MiniSafetensorWeights(self.path)

    def test_extra_shard_names_rejected_before_reader_opens_them(self):
        index = read_index(self.path)
        index["weight_map"]["embed.weight"] = "unknown.safetensors"
        write_index(self.path, index)
        with patch("glm_local.sharded_safetensors.SafeTensorReader") as opener:
            with self.assertRaisesRegex(SafeTensorError, "required shard"):
                MiniSafetensorWeights(self.path)
            opener.assert_not_called()

    def test_oversize_fixed_shard_rejected_before_header_parse(self):
        with (self.path / SHARD_NAMES[0]).open("wb") as stream:
            stream.truncate(65537)
        with self.assertRaisesRegex(SafeTensorError, "64-KiB"):
            MiniSafetensorWeights(self.path)

    def test_export_never_overwrites_existing_files(self):
        before = {name: (self.path / name).read_bytes() for name in (*SHARD_NAMES, INDEX_NAME)}
        with self.assertRaises(FileExistsError):
            write_mini_shards(self.source, self.path)
        self.assertEqual(before, {name: (self.path / name).read_bytes() for name in before})

    def test_close_rejects_all_future_weight_reads(self):
        reader = MiniSafetensorWeights(self.path)
        reader.close()
        for operation in (lambda: reader.matrix("embed"), lambda: reader.vector("final_norm"),
                          lambda: reader.embedding(0), reader.__enter__):
            with self.assertRaisesRegex(SafeTensorError, "closed"):
                operation()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Official tensor byte comparison needs Torch")
    def test_every_tensor_metadata_and_byte_against_official_safe_open(self):
        import torch
        from safetensors import safe_open
        raw_original = {}
        with MiniWeights(self.source) as source:
            for name in matrix_shapes():
                matrix = source.matrix(name)
                weight, scale = matrix_names(name)
                raw_original[weight] = matrix.weights
                raw_original[scale] = struct.pack("<f", matrix.scale)
            for name, length in vector_lengths().items():
                raw_original[vector_name(name)] = struct.pack(f"<{length}f", *source.vector(name))
        dtypes = {"F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}
        contract = tensor_contract()
        seen = set()
        for filename in SHARD_NAMES:
            with safe_open(str(self.path / filename), framework="pt", device="cpu") as official:
                self.assertEqual(official.metadata()["synthetic_fixture"], MARKER)
                for name in official.keys():
                    value = official.get_tensor(name)
                    self.assertNotIn(name, seen)
                    seen.add(name)
                    dtype, shape = contract[name]
                    self.assertEqual(value.dtype, dtypes[dtype])
                    self.assertEqual(tuple(value.shape), shape)
                    self.assertEqual(value.reshape(-1).view(torch.uint8).numpy().tobytes(), raw_original[name])
        self.assertEqual(seen, set(contract))
