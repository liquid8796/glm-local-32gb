import io
import json
import struct
import unittest
from unittest.mock import patch
import urllib.error
from email.message import Message

from glm_local.checkpoint_http import (FetchLimits, HttpMetadataSource, MetadataError, ReadBudget,
                                      _NoRedirect, _trusted_url, header_length, strict_json,
                                      validate_target, shard_filename)
from glm_local.safetensor_reader import MAX_READ_BYTES, MAX_HEADER_BYTES
from checkpoint_test_helpers import MODEL, REVISION, Opener, Response, ranged


class CheckpointTransportTests(unittest.TestCase):
    def source(self, responses, **kwargs):
        opener = Opener(responses)
        return HttpMetadataSource(MODEL, REVISION, opener=opener, **kwargs), opener

    def test_only_the_three_pinned_json_endpoints_are_fetched(self):
        source, opener = self.source([Response(b"{}", headers={"Content-Length": 2}) for _ in range(3)])
        for kind in ("model", "config", "index"):
            self.assertEqual(source.json_bytes(kind), b"{}")
        self.assertEqual([r.full_url for r, _ in opener.requests], [
            f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true",
            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/config.json",
            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors.index.json"])
        for request, timeout in opener.requests:
            self.assertEqual(request.get_header("Accept-encoding"), "identity")
            self.assertIsNone(request.get_header("Authorization"))
            self.assertGreater(timeout, 0)

    def test_reads_only_prefix_and_header_with_exact_ranges(self):
        header = b'{"__metadata__":{}}'
        prefix = struct.pack("<Q", len(header))
        size = 9000000000
        source, opener = self.source([ranged(prefix, 0, size, ETag='"same"'),
                                      ranged(header, 8, size, ETag='"same"')])
        self.assertEqual(source.header_bytes("part.safetensors", size), prefix + header)
        self.assertEqual([r.get_header("Range") for r, _ in opener.requests],
                         ["bytes=0-7", f"bytes=8-{7 + len(header)}"])
        self.assertEqual(opener.requests[1][0].get_header("If-match"), '"same"')
        self.assertEqual(source.stats()["body_bytes_read"], len(prefix + header))
        self.assertEqual(source.stats()["tensor_payload_bytes_requested"], 0)

    def test_large_header_is_read_in_64_kib_chunks(self):
        header = b"{" + b" " * (MAX_READ_BYTES * 2)
        prefix = struct.pack("<Q", len(header))
        size = 8 + len(header) + 4096
        response = ranged(header, 8, size)
        source, _ = self.source([ranged(prefix, 0, size), response])
        self.assertEqual(source.header_bytes("part.safetensors", size), prefix + header)
        self.assertLessEqual(max(response.requested_reads), MAX_READ_BYTES)

    def test_ignored_range_rejected_before_any_body_read(self):
        for status in (200, 201):
            response = Response(b"would be a huge checkpoint", status=status,
                                headers={"Content-Length": 755000000000})
            source, _ = self.source([response])
            with self.subTest(status=status), self.assertRaisesRegex(MetadataError, "HTTP 206"):
                source.header_bytes("part.safetensors", 755000000000)
            self.assertEqual(response.requested_reads, [])
            self.assertTrue(response.closed)

    def test_wrong_missing_or_multipart_content_range_is_rejected_before_read(self):
        for content_range in (None, "bytes 0-8/9000", "bytes 1-8/9000", "bytes 0-7/*",
                              "bytes 0-7/9001", "items 0-7/9000", "bytes 0-7/9000,9-10/9000"):
            headers = {} if content_range is None else {"Content-Range": content_range}
            response = Response(b"12345678", status=206, headers=headers)
            source, _ = self.source([response])
            with self.subTest(value=content_range), self.assertRaisesRegex(MetadataError, "Content-Range"):
                source.header_bytes("part.safetensors", 9000)
            self.assertFalse(response.requested_reads)

    def test_wrong_range_content_length_rejected_before_read(self):
        response = ranged(b"12345678", 0, 1000)
        response.headers.replace_header("Content-Length", "9")
        source, _ = self.source([response])
        with self.assertRaisesRegex(MetadataError, "Content-Length"):
            source.header_bytes("part.safetensors", 1000)
        self.assertFalse(response.requested_reads)

    def test_truncated_range_raises(self):
        response = Response(b"123", status=206, headers={"Content-Range": "bytes 0-7/1000", "Content-Length": 8})
        source, _ = self.source([response])
        with self.assertRaisesRegex(MetadataError, "Truncated"):
            source.header_bytes("part.safetensors", 1000)
        self.assertEqual(source.stats()["body_bytes_read"], 3)

    def test_invalid_prefix_never_requests_header(self):
        for length in (0, MAX_HEADER_BYTES + 1, 2**64 - 1, 993):
            source, opener = self.source([ranged(struct.pack("<Q", length), 0, 1000)])
            with self.subTest(length=length), self.assertRaises(MetadataError):
                source.header_bytes("part.safetensors", 1000)
            self.assertEqual(len(opener.requests), 1)

    def test_range_budget_is_checked_before_read(self):
        response = ranged(b"12345678", 0, 1000)
        source, _ = self.source([response], limits=FetchLimits(total_body_bytes=7))
        with self.assertRaisesRegex(MetadataError, "budget"):
            source.header_bytes("part.safetensors", 1000)
        self.assertFalse(response.requested_reads)
        self.assertEqual(source.budget.bytes, 0)

    def test_etag_change_and_disappearance_rejected(self):
        prefix = struct.pack("<Q", 2)
        for etag in ('"changed"', None):
            headers = {"ETag": etag} if etag else {}
            second = ranged(b"{}", 8, 1000, **headers)
            source, _ = self.source([ranged(prefix, 0, 1000, ETag='"original"'), second])
            with self.subTest(etag=etag), self.assertRaisesRegex(MetadataError, "ETag"):
                source.header_bytes("part.safetensors", 1000)
            self.assertFalse(second.requested_reads)

    def test_weak_etag_is_compared_but_not_used_for_if_match(self):
        source, opener = self.source([ranged(struct.pack("<Q", 2), 0, 1000, ETag='W/"same"'),
                                      ranged(b"{}", 8, 1000, ETag='W/"same"')])
        source.header_bytes("part.safetensors", 1000)
        self.assertIsNone(opener.requests[1][0].get_header("If-match"))

    def test_missing_etag_does_not_invent_payload_authentication(self):
        source, _ = self.source([ranged(struct.pack("<Q", 2), 0, 1000), ranged(b"{}", 8, 1000)])
        source.header_bytes("part.safetensors", 1000)
        self.assertNotIn("payload_authenticated", source.stats())

    def test_encoded_response_is_rejected_before_read(self):
        for encoding in ("gzip", "br"):
            response = Response(b"{}", headers={"Content-Encoding": encoding})
            source, _ = self.source([response])
            with self.subTest(encoding=encoding), self.assertRaisesRegex(MetadataError, "Compressed"):
                source.json_bytes("config")
            self.assertFalse(response.requested_reads)

    def test_revision_header_mismatch_is_rejected_before_read(self):
        response = Response(b"{}", headers={"X-Repo-Commit": "b" * 40})
        source, _ = self.source([response])
        with self.assertRaisesRegex(MetadataError, "revision"):
            source.json_bytes("config")
        self.assertFalse(response.requested_reads)

    def test_matching_revision_header_is_recorded(self):
        source, _ = self.source([Response(b"{}", headers={"X-Repo-Commit": REVISION})])
        source.json_bytes("config")
        self.assertEqual(source.stats()["matching_repo_commit_headers"], 1)

    def test_json_oversize_known_length_rejected_before_read(self):
        response = Response(b"{}", headers={"Content-Length": 1024**2 + 1})
        source, _ = self.source([response])
        with self.assertRaisesRegex(MetadataError, "bounded size"):
            source.json_bytes("config")
        self.assertFalse(response.requested_reads)

    def test_unknown_json_length_is_bounded_even_without_eof(self):
        response = Response(b"x" * 17)
        source, _ = self.source([response])
        with patch.dict("glm_local.checkpoint_http.JSON_LIMITS", {"config": 16}), \
                self.assertRaisesRegex(MetadataError, "bounded size"):
            source.json_bytes("config")
        self.assertEqual(response.requested_reads, [17])
        self.assertEqual(source.budget.bytes, 17)

    def test_json_without_content_length_can_succeed(self):
        source, _ = self.source([Response(b'{"ok":true}')])
        self.assertEqual(json.loads(source.json_bytes("config")), {"ok": True})

    def test_json_partial_status_is_not_accepted(self):
        response = Response(b"{}", status=206)
        source, _ = self.source([response])
        with self.assertRaisesRegex(MetadataError, "HTTP 200"):
            source.json_bytes("index")
        self.assertFalse(response.requested_reads)

    def test_invalid_content_length(self):
        for length in ("-1", "1.0", "NaN", "999999999999999999999"):
            response = Response(b"{}", headers={"Content-Length": length})
            source, _ = self.source([response])
            with self.subTest(length=length), self.assertRaisesRegex(MetadataError, "Content-Length"):
                source.json_bytes("config")
            self.assertFalse(response.requested_reads)

    def test_safe_redirect_preserves_range_and_never_reads_redirect_body(self):
        body = Response(b"do not read redirect body")
        headers = Message()
        headers["Location"] = "https://cas-bridge.xethub.hf.co/public?secret=signed-value"
        error = urllib.error.HTTPError("https://huggingface.co/start", 302, "Found", headers, body)
        source, opener = self.source([error, ranged(struct.pack("<Q", 2), 0, 1000), ranged(b"{}", 8, 1000)])
        source.header_bytes("part.safetensors", 1000)
        self.assertTrue(body.closed)
        self.assertFalse(body.requested_reads)
        self.assertEqual(opener.requests[1][0].get_header("Range"), "bytes=0-7")
        self.assertNotIn("signed-value", json.dumps(source.stats()))
        self.assertEqual(source.stats()["redirects"], 1)

    def test_unsafe_redirect_is_closed_and_not_followed(self):
        for location in ("http://huggingface.co/file", "https://evil.invalid/x", "https://huggingface.co.evil.invalid/x"):
            body = Response(b"do not read")
            headers = Message(); headers["Location"] = location
            source, opener = self.source([urllib.error.HTTPError("https://huggingface.co", 302, "Found", headers, body)])
            with self.subTest(location=location), self.assertRaises(MetadataError):
                source.json_bytes("config")
            self.assertEqual(len(opener.requests), 1)
            self.assertFalse(body.requested_reads)
            self.assertTrue(body.closed)

    def test_http_error_is_not_retried_or_logged_with_signed_url(self):
        body = Response(b"private error details")
        error = urllib.error.HTTPError("https://cas-bridge.xethub.hf.co/x?secret=abc", 403, "No", Message(), body)
        source, opener = self.source([error])
        with self.assertRaisesRegex(MetadataError, "HTTP 403") as caught:
            source.json_bytes("config")
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(len(opener.requests), 1)
        self.assertFalse(body.requested_reads)

    def test_dns_error_is_reported_without_original_url(self):
        source, _ = self.source([urllib.error.URLError("https://example.invalid/?secret=abc")])
        with self.assertRaisesRegex(MetadataError, "DNS") as error:
            source.json_bytes("config")
        self.assertNotIn("secret", str(error.exception))

    def test_no_redirect_handler_declines_automatic_body_consumption(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://huggingface.co/x"))

    def test_request_and_elapsed_time_limits(self):
        source, opener = self.source([Response(b"{}")], limits=FetchLimits(max_requests=1))
        source.json_bytes("config")
        with self.assertRaisesRegex(MetadataError, "request budget"):
            source.json_bytes("index")
        self.assertEqual(len(opener.requests), 1)
        source, opener = self.source([])
        source._started -= source.limits.total_seconds + 1
        with self.assertRaisesRegex(MetadataError, "elapsed-time"):
            source.json_bytes("config")
        self.assertFalse(opener.requests)


class CheckpointValidationTests(unittest.TestCase):
    def test_json_rejects_duplicate_nonfinite_and_bad_unicode(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'{'):
            with self.subTest(raw=raw), self.assertRaises(MetadataError):
                strict_json(raw)

    def test_pins_and_names_are_validated_without_network(self):
        for model in ("../test", "owner/..", "owner/model/extra", "https://huggingface.co/x", None):
            with self.subTest(model=model), self.assertRaises(MetadataError):
                validate_target(model, REVISION)
        for revision in ("main", "a" * 39, "A" * 40, None):
            with self.subTest(revision=revision), self.assertRaises(MetadataError):
                validate_target(MODEL, revision)
        for filename in ("../x.safetensors", "/x.safetensors", "weights/x.safetensors", "CON.safetensors",
                         "a\\x.safetensors", "x:s.safetensors", "x.safetensors?download=1"):
            with self.subTest(filename=filename), self.assertRaises(MetadataError):
                shard_filename(filename)

    def test_invalid_limits_rejected(self):
        for values in ({"total_body_bytes": True}, {"total_body_bytes": 128 * 1024**2 + 1},
                       {"timeout_seconds": 0}, {"max_redirects": 6}, {"max_requests": -1}):
            with self.subTest(values=values), self.assertRaises(MetadataError):
                FetchLimits(**values)

    def test_trusted_url_validation(self):
        for url in ("file:///x", "https://user:password@huggingface.co/x", "https://hf.co:4433/x",
                    "https://huggingface.co/x#fragment", "https://hf.co:notaport/x"):
            with self.subTest(url=url), self.assertRaises(MetadataError):
                _trusted_url(url)

    def test_header_length_and_read_budget(self):
        for prefix in (b"", b"1234567", b"123456789"):
            with self.subTest(prefix=prefix), self.assertRaises(MetadataError):
                header_length(prefix, 1000)
        budget = ReadBudget(4)
        self.assertEqual(budget.read(io.BytesIO(b"1234"), 4), b"1234")
        with self.assertRaises(MetadataError):
            budget.read(io.BytesIO(b"5"), 1)
        self.assertEqual(budget.bytes, 4)
