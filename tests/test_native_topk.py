import os
import random
import unittest

from glm_local.native_topk import NativeTopK


class NativeTopKValidationTests(unittest.TestCase):
    def test_invalid_inputs_fail_before_native_access(self):
        selector = NativeTopK.__new__(NativeTopK)
        for values, k in (([], 1), ([0.0]*129, 1), ([0.0], 2), ([0.0], True),
                          ([float("nan")], 1), ([float("inf")], 1), ([True], 1), ([1e100], 1)):
            with self.subTest(values=values, k=k), self.assertRaises(ValueError):
                selector(values, k)


@unittest.skipUnless(os.environ.get("GLM_TEST_OFFICIAL") == "1", "Requires pinned official reference and native top-k DLL")
class NativeTopKOfficialTests(unittest.TestCase):
    def test_cpu_stl_topk_matches_pinned_torch_ties_and_unique_values(self):
        import torch
        selector = NativeTopK()
        rng = random.Random(419)
        for length in range(1, 129):
            patterns = [[0.0]*length, [float(i % 3) for i in range(length)],
                        [rng.choice([-1.0, -0.0, 0.0, 0.5, 1.0]) for _ in range(length)],
                        [rng.uniform(-2, 2) for _ in range(length)]]
            for values in patterns:
                for count in range(1, min(4, length)+1):
                    expected = torch.topk(torch.tensor(values, dtype=torch.float32), count).indices.tolist()
                    with self.subTest(length=length, count=count):
                        self.assertEqual(selector(values, count), expected)
