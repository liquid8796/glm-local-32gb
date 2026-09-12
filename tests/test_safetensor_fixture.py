import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("GLM_TEST_OFFICIAL") != "1":
    raise unittest.SkipTest("Official safetensors fixtures require the pinned reference environment")

import torch
from safetensors import safe_open
from safetensors.torch import save
from glm_local.safetensor_fixture import create_fixture, reference_fixture, MARKER


class OfficialFixtureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)/"case.safetensors"

    def test_official_format_roundtrip_scalars_and_dtype(self):
        created=create_fixture(self.path,129,131,7)
        ref=reference_fixture(self.path)
        self.assertEqual(created["scale_grid"],[2,2])
        self.assertEqual(ref["tensors"]["weight"]["dtype"],"F8_E4M3")
        self.assertEqual(ref["tensors"]["empty_probe"]["raw_bytes"],b"")
        self.assertEqual(ref["tensors"]["scalar_probe"]["shape"],[])
        self.assertEqual(len(set(x for row in ref["scale_values"] for x in row)),4)

    def test_exclusive_creation_and_same_seed(self):
        first=create_fixture(self.path,1,1,7)
        before=self.path.read_bytes()
        with self.assertRaises(FileExistsError): create_fixture(self.path,1,1,9)
        self.assertEqual(self.path.read_bytes(),before)
        other=self.path.with_name("other.safetensors")
        second=create_fixture(other,1,1,7)
        # The official serializer may reorder metadata keys; tensor content is
        # deterministic, but byte-identical whole-file headers are not promised.
        left,right=reference_fixture(self.path),reference_fixture(other)
        for name in left["tensors"]:
            self.assertEqual(left["tensors"][name]["raw_bytes"],right["tensors"][name]["raw_bytes"])
        self.assertEqual(left["expected_output"],right["expected_output"])
        self.assertEqual(first["file_bytes"],second["file_bytes"])

    def test_invalid_dimensions_seed_before_creation(self):
        for rows,cols,seed in ((0,1,1),(1025,1,1),(1,1,-1),(True,1,1),(1,1,2**32)):
            with self.assertRaises(ValueError): create_fixture(self.path,rows,cols,seed)
            self.assertFalse(self.path.exists())

    def test_scale_is_multiplied_not_divided(self):
        create_fixture(self.path,1,1,7)
        ref=reference_fixture(self.path)
        with safe_open(str(self.path),framework="pt",device="cpu") as file:
            q=file.get_tensor("weight").float().item()
            scale=file.get_tensor("weight_scale_inv").item()
            v=file.get_tensor("vector").item()
        self.assertEqual(ref["expected_output"],[q*scale*v])
        self.assertNotEqual(ref["expected_output"],[q/scale*v])

    def test_nonfixture_and_invalid_scale_are_rejected(self):
        self.path.write_bytes(save({"weight":torch.ones((1,1))}))
        with self.assertRaises(ValueError): reference_fixture(self.path)
        self.path.unlink()
        create_fixture(self.path,1,1,7)
        with safe_open(str(self.path),framework="pt",device="cpu") as file:
            tensors={name:file.get_tensor(name) for name in file.keys()}
        tensors["weight_scale_inv"].fill_(0)
        invalid = self.path.with_name("invalid-scale.safetensors")
        invalid.write_bytes(save(tensors,metadata={"synthetic_fixture":MARKER,"seed":"7"}))
        with self.assertRaises(ValueError): reference_fixture(invalid)
