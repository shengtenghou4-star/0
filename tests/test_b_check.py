import io
import tarfile
import unittest

from scripts import b_check as verifier


class PublicFetchVerifierTests(unittest.TestCase):
    def test_chunk_rows_must_be_contiguous(self):
        receipt = {
            "chunks": [
                {
                    "name": "result.tar.gz.part0000",
                    "bytes": 10,
                    "sha256": "a" * 64,
                },
                {
                    "name": "result.tar.gz.part0001",
                    "bytes": 20,
                    "sha256": "b" * 64,
                },
            ]
        }
        self.assertEqual(verifier.validate_chunk_rows(receipt), receipt["chunks"])
        receipt["chunks"][1]["name"] = "result.tar.gz.part0002"
        with self.assertRaises(verifier.VerificationError):
            verifier.validate_chunk_rows(receipt)

    def test_archive_member_rejects_parent_traversal(self):
        member = tarfile.TarInfo("../escape")
        with self.assertRaises(verifier.VerificationError):
            verifier.validate_member(member)

    def test_hash_stream_returns_size_digest_and_prefix(self):
        size, digest, prefix = verifier.hash_stream(io.BytesIO(b"abcdef"), prefix_bytes=3)
        self.assertEqual(size, 6)
        self.assertEqual(digest, "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721")
        self.assertEqual(prefix, b"abc")

    def test_manifest_verification_contract_is_exact(self):
        manifest = {
            "schema": 1,
            "capsule_id": "a" * 32,
            "operation": "public_https_fetch",
            "files": [
                {
                    "name": "archive.rar",
                    "min_bytes": 10,
                    "max_bytes": 100,
                    "magic_hex": "526172211a070100",
                }
            ],
        }
        rows = verifier.validate_manifest(manifest, "a" * 32)
        self.assertEqual(rows["archive.rar"]["min_bytes"], 10)
        manifest["files"][0]["name"] = "../archive.rar"
        with self.assertRaises(verifier.VerificationError):
            verifier.validate_manifest(manifest, "a" * 32)

    def test_large_contents_use_raw_media_type(self):
        captured = {}
        original = verifier.api_request

        def fake_api_request(url, token, **kwargs):
            captured.update(url=url, token=token, kwargs=kwargs)
            return b"raw-bytes"

        verifier.api_request = fake_api_request
        try:
            value = verifier.fetch_contents_bytes("owner/repo", "path/file.bin", "results", "token")
        finally:
            verifier.api_request = original
        self.assertEqual(value, b"raw-bytes")
        self.assertEqual(captured["kwargs"]["accept"], "application/vnd.github.raw+json")
        self.assertIn("ref=results", captured["url"])


if __name__ == "__main__":
    unittest.main()
