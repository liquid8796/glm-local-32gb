"""Real installed-library checks and explicitly simulated 0.7/0.8 API contracts."""
import ctypes
import gc
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from glm_local.safetensor_serializer import MAX_PAYLOAD_BYTES, serialize_raw_tensors


class TensorSpecDouble:
    """Only models the verified pointer contract; it is not the Rust extension."""
    def __init__(self, *, dtype, shape, data_ptr, data_len):
        self.dtype, self.shape = dtype, list(shape)
        self.data_ptr, self.data_len = data_ptr, data_len


def unpack_specs(tensors):
    if any(not isinstance(value, TensorSpecDouble) for value in tensors.values()):
        raise TypeError("argument 'tensor_dict': 'dict' object is not an instance of 'TensorSpec'")
    return {name: {"dtype": spec.dtype, "shape": spec.shape,
                   "data": ctypes.string_at(spec.data_ptr, spec.data_len)}
            for name, spec in tensors.items()}


def example_tensors():
    return {
        # All byte encodings, including signed zero and NaNs: no numeric conversion.
        "fp8": {"dtype": "float8_e4m3fn", "shape": [16, 16], "data": bytes(range(256))},
        "scale": {"dtype": "float32", "shape": [1, 2], "data": struct.pack("<2f", .125, 1.)},
        "empty": {"dtype": "float32", "shape": [0, 3], "data": b""},
        "scalar": {"dtype": "float32", "shape": [], "data": struct.pack("<f", 1.25)},
    }


class SerializerContractTests(unittest.TestCase):
    def test_legacy_raw_dictionary_is_passed_to_official_serializer(self):
        serializer = Mock(return_value=b"legacy")
        module = SimpleNamespace(serialize=serializer, __version__="not-parsed")
        tensors, metadata = example_tensors(), {"synthetic_fixture": "test"}
        with patch.dict(sys.modules, {"safetensors": module}):
            self.assertEqual(serialize_raw_tensors(tensors, metadata), b"legacy")
        serializer.assert_called_once_with(tensors, metadata=metadata)

    def test_tensor_spec_api_preserves_bytes_and_metadata_with_no_version_parsing(self):
        tensors, metadata = example_tensors(), {"synthetic_fixture": "test"}
        def serialize(specs, metadata=None):
            self.assertEqual(unpack_specs(specs), tensors)
            self.assertEqual(metadata, {"synthetic_fixture": "test"})
            self.assertGreater(specs["empty"].data_ptr, 0)
            self.assertEqual(specs["empty"].data_len, 0)
            return memoryview(b"new-api")
        module = SimpleNamespace(TensorSpec=TensorSpecDouble, serialize=serialize)
        with patch.dict(sys.modules, {"safetensors": module}):
            result = serialize_raw_tensors(tensors, metadata)
        self.assertIs(type(result), bytes)
        self.assertEqual(result, b"new-api")

    def test_all_pointer_buffers_survive_gc_until_output_is_copied_then_are_released(self):
        refs = []
        create_buffer = ctypes.create_string_buffer
        tensors = example_tensors()
        def track_buffer(*args):
            buffer = create_buffer(*args)
            refs.append(weakref.ref(buffer))
            return buffer
        class DeferredOutput:
            def __bytes__(self):
                gc.collect()
                self_test.assertEqual(len(refs), len(tensors))
                self_test.assertTrue(all(ref() is not None for ref in refs))
                return b"copied"
        self_test = self
        def serialize(specs, metadata=None):
            gc.collect()
            self.assertTrue(all(ref() is not None for ref in refs))
            self.assertEqual(unpack_specs(specs), tensors)
            return DeferredOutput()
        module = SimpleNamespace(TensorSpec=TensorSpecDouble, serialize=serialize)
        with patch.dict(sys.modules, {"safetensors": module}), \
                patch("glm_local.safetensor_serializer.ctypes.create_string_buffer", side_effect=track_buffer):
            self.assertEqual(serialize_raw_tensors(tensors), b"copied")
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_serializer_typeerror_propagates_without_legacy_retry(self):
        serialize = Mock(side_effect=TypeError("invalid dtype from serializer"))
        module = SimpleNamespace(TensorSpec=TensorSpecDouble, serialize=serialize)
        with patch.dict(sys.modules, {"safetensors": module}):
            with self.assertRaisesRegex(TypeError, "invalid dtype"):
                serialize_raw_tensors(example_tensors())
        serialize.assert_called_once()

    def test_buffer_references_are_released_on_serialization_failure(self):
        refs = []
        create_buffer = ctypes.create_string_buffer
        def track(*args):
            buffer = create_buffer(*args)
            refs.append(weakref.ref(buffer))
            return buffer
        def fail(specs, metadata=None):
            self.assertEqual(len(unpack_specs(specs)), 4)
            raise RuntimeError("serializer failed")
        module = SimpleNamespace(TensorSpec=TensorSpecDouble, serialize=fail)
        with patch.dict(sys.modules, {"safetensors": module}), \
                patch("glm_local.safetensor_serializer.ctypes.create_string_buffer", side_effect=track):
            with self.assertRaisesRegex(RuntimeError, "serializer failed"):
                serialize_raw_tensors(example_tensors())
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_non_bytes_payload_is_rejected_before_serialization(self):
        for payload in (bytearray(b"x"), memoryview(b"x"), "x", None):
            with self.subTest(payload=type(payload).__name__):
                serialize = Mock()
                module = SimpleNamespace(TensorSpec=TensorSpecDouble, serialize=serialize)
                with patch.dict(sys.modules, {"safetensors": module}):
                    with self.assertRaisesRegex(TypeError, "immutable bytes"):
                        serialize_raw_tensors({"x": {"dtype": "uint8", "shape": [1], "data": payload}})
                serialize.assert_not_called()

    def test_total_payload_bound_applies_before_allocating_pointer_buffers(self):
        tensors = {
            "a": {"dtype": "uint8", "shape": [MAX_PAYLOAD_BYTES], "data": bytes(MAX_PAYLOAD_BYTES)},
            "b": {"dtype": "uint8", "shape": [1], "data": b"x"},
        }
        for modern in (False, True):
            module = SimpleNamespace(serialize=Mock())
            if modern:
                module.TensorSpec = TensorSpecDouble
            with self.subTest(modern=modern), patch.dict(sys.modules, {"safetensors": module}), \
                    patch("glm_local.safetensor_serializer.ctypes.create_string_buffer") as create:
                with self.assertRaisesRegex(ValueError, "64-KiB"):
                    serialize_raw_tensors(tensors)
                module.serialize.assert_not_called()
                create.assert_not_called()

    def test_exact_payload_bound_is_accepted(self):
        tensors = {"x": {"dtype": "uint8", "shape": [MAX_PAYLOAD_BYTES], "data": bytes(MAX_PAYLOAD_BYTES)}}
        def serialize(specs, metadata=None):
            self.assertEqual(unpack_specs(specs), tensors)
            return b"ok"
        with patch.dict(sys.modules, {"safetensors": SimpleNamespace(TensorSpec=TensorSpecDouble, serialize=serialize)}):
            self.assertEqual(serialize_raw_tensors(tensors), b"ok")

    def test_empty_dictionary_is_delegated_to_official_serializer(self):
        for modern in (False, True):
            serialize = Mock(return_value=b"empty")
            module = SimpleNamespace(serialize=serialize)
            if modern:
                module.TensorSpec = TensorSpecDouble
            with self.subTest(modern=modern), patch.dict(sys.modules, {"safetensors": module}):
                self.assertEqual(serialize_raw_tensors({}, {"fixture": "empty"}), b"empty")
            serialize.assert_called_once_with({}, metadata={"fixture": "empty"})


@unittest.skipUnless(importlib.util.find_spec("safetensors"), "Requires the actual safetensors library")
class InstalledSerializerTests(unittest.TestCase):
    def test_raw_payload_round_trip_through_actual_installed_serializer_and_deserializer(self):
        import safetensors
        tensors = example_tensors()
        tensors["bf16"] = {"dtype": "bfloat16", "shape": [2], "data": struct.pack("<2H", 0x3F80, 0xC020)}
        tensors["integer"] = {"dtype": "uint32", "shape": [1], "data": struct.pack("<I", 42)}
        encoded = serialize_raw_tensors(tensors, {"synthetic_fixture": "test"})
        restored = dict(safetensors.deserialize(encoded))
        tags = {"float8_e4m3fn": "F8_E4M3", "float32": "F32", "bfloat16": "BF16", "uint32": "U32"}
        self.assertEqual(set(restored), set(tensors))
        for name, expected in tensors.items():
            with self.subTest(name=name):
                self.assertEqual(restored[name]["shape"], expected["shape"])
                self.assertEqual(restored[name]["dtype"], tags[expected["dtype"]])
                self.assertEqual(bytes(restored[name]["data"]), expected["data"])

    def test_invalid_byte_count_and_dtype_are_rejected_by_actual_library(self):
        for descriptor in ({"dtype": "float32", "shape": [1], "data": b"x"},
                           {"dtype": "unknown", "shape": [1], "data": b"x"}):
            with self.subTest(descriptor=descriptor), self.assertRaises(Exception):
                serialize_raw_tensors({"bad": descriptor})

    def test_mini_export_handles_tensor_spec_only_api_without_changing_fixture_values(self):
        # Simulate the new argument contract, but generate/decode file bytes using
        # the actual installed extension (whatever version is available).
        import safetensors as actual
        from glm_local.mini_safetensors import MiniSafetensorWeights, write_mini_shards
        from glm_local.mini_spec import matrix_shapes, vector_lengths
        from glm_local.mini_weights import MiniWeights, write_mini_bundle
        serialize = actual.serialize
        actual_spec = getattr(actual, "TensorSpec", None)
        def strict_serialize(specs, metadata=None):
            raw = unpack_specs(specs)  # This fails with the exact pre-fix error on 0.6.0.
            if actual_spec is None:
                return serialize(raw, metadata=metadata)
            real_specs = {name: actual_spec(dtype=spec.dtype, shape=spec.shape,
                                           data_ptr=spec.data_ptr, data_len=spec.data_len)
                          for name, spec in specs.items()}
            return serialize(real_specs, metadata=metadata)
        module = SimpleNamespace(TensorSpec=TensorSpecDouble, serialize=strict_serialize,
                                 __version__="0.8.0-contract-test-double")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mini_bundle(root / "private", seed=19)
            with patch.dict(sys.modules, {"safetensors": module}):
                report = write_mini_shards(root / "private", root / "sharded")
            self.assertEqual(report["serializer_api"], "TensorSpec")
            self.assertEqual(report["tensor_count"], 80)
            with MiniWeights(root / "private") as original, MiniSafetensorWeights(root / "sharded") as new:
                for name in matrix_shapes():
                    self.assertEqual(original.matrix(name), new.matrix(name))
                for name in vector_lengths():
                    self.assertEqual(original.vector(name), new.vector(name))
