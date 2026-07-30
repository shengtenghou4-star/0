import json
import tempfile
import unittest
from pathlib import Path

from scripts import store_fixed_tool, verify_fixed_tool


class FixedToolContractTests(unittest.TestCase):
    def test_binary_requires_elf_and_minimum_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool"
            path.write_bytes(b"\x7fELF" + b"x" * 10_000)
            data, digest = store_fixed_tool.validate_binary(path)
            self.assertEqual(len(data), 10_004)
            self.assertEqual(len(digest), 64)
            path.write_bytes(b"not-elf" + b"x" * 10_000)
            with self.assertRaises(store_fixed_tool.ToolStoreError):
                store_fixed_tool.validate_binary(path)

    def test_version_receipt_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version.txt"
            path.write_text("965\n", encoding="utf-8")
            self.assertEqual(store_fixed_tool.validate_version(path), "965")
            path.write_bytes(b"x" * 5000)
            with self.assertRaises(store_fixed_tool.ToolStoreError):
                store_fixed_tool.validate_version(path)

    def test_verifier_uses_raw_media_for_binary(self):
        captured = {}
        original = verify_fixed_tool.api_request

        def fake_api_request(url, token, **kwargs):
            captured.update(url=url, token=token, kwargs=kwargs)
            return b"\x7fELFbinary"

        verify_fixed_tool.api_request = fake_api_request
        try:
            value = verify_fixed_tool.fetch_bytes("owner/repo", "path/binary", "results", "token")
        finally:
            verify_fixed_tool.api_request = original
        self.assertEqual(value, b"\x7fELFbinary")
        self.assertEqual(captured["kwargs"]["accept"], "application/vnd.github.raw+json")

    def test_latest_pointer_is_private_and_run_bound(self):
        certificate = {
            "schema": 1,
            "tool_id": "c" * 32,
            "run_id": "123",
            "run_attempt": "1",
            "operation": "fixed_tool_independent_verification",
            "receipt_sha256": "a" * 64,
            "binary_bytes": 10004,
            "binary_sha256": "b" * 64,
            "elf_magic": "7f454c46",
            "source_commit": "d" * 40,
            "version": "965.1",
            "verdict": "VERIFIED",
        }
        prefix = f"relay/fixed-tools/{certificate['tool_id']}/123-1"
        latest = verify_fixed_tool.build_latest(certificate, prefix)
        self.assertEqual(latest["result_prefix"], prefix)
        self.assertEqual(latest["verification_path"], f"{prefix}/verification.json")
        serialized = json.dumps(latest)
        for forbidden in ("http://", "https://", "picosat", "PercoGuard", "Lazarus"):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
