"""Selected payload transport uses strict HTTP ranges without any network access."""

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from glm_local.checkpoint_http import HttpMetadataSource
from glm_local.execution import build_projection_descriptor, execute_projection, reference_projection, compare_projection
from glm_local.projection_remote import RemoteProjectionReader
from checkpoint_test_helpers import MODEL, REVISION, NAME, SCALE, ranged, settings
from test_execution import Kernel, projection_fixture


class ShardOpener:
    def __init__(self, directory, change=None):
        self.directory, self.change = directory, change
        self.requests, self.responses = [], []

    def open(self, request, timeout):
        name = request.full_url.rsplit("/", 1)[-1]
        self.requests.append(request)
        first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        data = (self.directory / name).read_bytes()
        response = ranged(data[first:last+1], first, len(data), ETag='"invented"')
        response.url = request.full_url
        if self.change:
            self.change(len(self.requests), response)
        self.responses.append(response)
        return response


class RemoteProjectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory, _, _ = projection_fixture(self.root)
        self.descriptor = build_projection_descriptor(self.root, settings(), NAME)
        self.destination = self.root / "download"

    def source(self, change=None):
        opener = ShardOpener(self.directory, change)
        return HttpMetadataSource(MODEL, REVISION, opener=opener), opener

    def test_downloads_only_selected_ranges_and_compares_through_actual_reader(self):
        source, opener = self.source()
        with RemoteProjectionReader(self.descriptor, self.destination, _transport=source) as reader:
            vector = [0.125] * 259
            output, _ = execute_projection(reader, self.descriptor, vector, cpu=Kernel())
            reference = reference_projection(reader, self.descriptor, vector)
            self.assertTrue(compare_projection(output, reference)["passed"])
            self.assertEqual(len(opener.requests), 4)
            for request, tensor in zip(opener.requests[2:], (self.descriptor.weight, self.descriptor.scale)):
                proof = next(item for item in self.descriptor.shards if item.name == tensor.shard)
                start = proof.header_bytes + tensor.data_offsets[0]
                self.assertEqual(request.get_header("Range"), f"bytes={start}-{start+tensor.nbytes-1}")
                self.assertEqual(request.get_header("If-match"), '"invented"')
            stats = reader.stats()
            self.assertEqual(stats["network"]["tensor_payload_bytes_requested"], 257*259+36)
            self.assertEqual(stats["network"]["body_bytes_read"], 257*259+36+sum(s.header_bytes for s in self.descriptor.shards))
            self.assertLessEqual(max(size for response in opener.responses for size in response.requested_reads), 65536)
            self.assertEqual(stats["persistent_decoded_weight_bytes"], 0)
            receipt = json.loads((self.destination / "receipt.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertFalse(receipt["full_checkpoint_downloaded"])
            self.assertEqual({path.suffix for path in self.destination.iterdir()}, {".bin", ".json"})
            for item in receipt["tensors"]:
                self.assertEqual(hashlib.sha256((self.destination/item["file"]).read_bytes()).hexdigest(), item["sha256"])

    def test_header_mismatch_stops_before_payload_download(self):
        source, opener = self.source()
        path = self.directory / self.descriptor.scale.shard
        path.write_bytes(path.read_bytes().replace(b'"format":"pt"', b'"format":"xx"', 1))
        with self.assertRaisesRegex(ValueError, "Live shard header"):
            RemoteProjectionReader(self.descriptor, self.destination, _transport=source)
        self.assertEqual(len(opener.requests), 1)
        self.assertFalse((self.destination / "receipt.json").exists())

    def test_whole_file_response_bad_range_and_etag_change_fail_without_body_reads(self):
        def status(number, response):
            if number == 3:
                response.status = 200

        def content_range(number, response):
            if number == 3:
                response.headers.replace_header("Content-Range", "bytes 0-2/3")

        def etag(number, response):
            if number == 3:
                response.headers.replace_header("ETag", '"changed"')

        for index, change in enumerate((status, content_range, etag)):
            source, opener = self.source(change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                RemoteProjectionReader(self.descriptor, self.root / f"bad-{index}", _transport=source)
            self.assertEqual(opener.responses[-1].requested_reads, [])
            self.assertTrue(all(response.closed for response in opener.responses))
            self.assertFalse((self.root/f"bad-{index}"/"receipt.json").exists())

    def test_budget_is_enforced_before_requests_and_output_directory(self):
        source, opener = self.source()
        with self.assertRaisesRegex(ValueError, "budget"):
            RemoteProjectionReader(self.descriptor, self.destination, budget_bytes=1, _transport=source)
        self.assertEqual(opener.requests, [])
        self.assertFalse(self.destination.exists())

    def test_cached_slice_mutation_bounds_alignment_and_closed_reader_fail(self):
        source, _ = self.source()
        with RemoteProjectionReader(self.descriptor, self.destination, _transport=source) as reader:
            for operation in (lambda: reader.read_bytes(NAME, 0, 65537),
                              lambda: reader.read_bytes(SCALE, 1, 4),
                              lambda: reader.read_bytes(SCALE, 0, 1),
                              lambda: reader.read_matrix_tile(NAME, 0, 0, 129, 1)):
                with self.assertRaises(ValueError):
                    operation()
            path = self.destination / "tensor-0.bin"
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "changed"):
                reader.read_bytes(NAME, 0, 1)
        with self.assertRaisesRegex(ValueError, "closed"):
            reader.read_bytes(NAME, 0, 1)
        with self.assertRaisesRegex(ValueError, "closed"):
            reader.__enter__()


if __name__ == "__main__":
    unittest.main()
