from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "index_public_fetch_result", ROOT / "scripts" / "index_public_fetch_result.py"
)
assert SPEC and SPEC.loader
indexer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(indexer)

CAPSULE = "0123456789abcdef0123456789abcdef"


def receipt(run_id: int, attempt: int, return_code: int = 0) -> dict:
    return {
        "schema": 1,
        "capsule_id": CAPSULE,
        "operation": "public_https_fetch",
        "run_id": str(run_id),
        "run_attempt": str(attempt),
        "container_image": "python:3.12-bookworm",
        "network": "bridge",
        "return_code": return_code,
        "timed_out": False,
        "started_unix": 1,
        "finished_unix": 2,
        "result_bytes": 123,
        "result_sha256": "a" * 64,
        "chunks": [{"name": "result.tar.gz.part0000", "bytes": 123, "sha256": "b" * 64}],
    }


class PublicFetchIndexTests(unittest.TestCase):
    def test_directory_listing_accepts_only_frozen_run_paths(self):
        raw = json.dumps([
            {"type": "file", "name": "latest.json", "path": f"relay/public-fetch-results/{CAPSULE}/latest.json"},
            {"type": "dir", "name": "305-1", "path": f"relay/public-fetch-results/{CAPSULE}/305-1"},
            {"type": "dir", "name": "306-2", "path": f"relay/public-fetch-results/{CAPSULE}/306-2"},
        ]).encode()
        self.assertEqual(indexer.list_result_directories(raw, CAPSULE), [
            (305, 1, f"relay/public-fetch-results/{CAPSULE}/305-1"),
            (306, 2, f"relay/public-fetch-results/{CAPSULE}/306-2"),
        ])

    def test_directory_listing_rejects_path_mismatch(self):
        raw = json.dumps([{"type": "dir", "name": "305-1", "path": "wrong/305-1"}]).encode()
        with self.assertRaises(indexer.IndexError):
            indexer.list_result_directories(raw, CAPSULE)

    def test_receipt_identity_is_bound_to_directory(self):
        indexer.validate_receipt(receipt(305, 1), CAPSULE, 305, 1)
        with self.assertRaises(indexer.IndexError):
            indexer.validate_receipt(receipt(305, 1), CAPSULE, 306, 1)

    def test_latest_index_contains_only_private_result_metadata(self):
        value = receipt(305, 1)
        raw = (json.dumps(value, sort_keys=True) + "\n").encode()
        path = f"relay/public-fetch-results/{CAPSULE}/305-1"
        latest = indexer.build_latest_index(CAPSULE, 305, 1, path, raw, value)
        self.assertEqual(latest["receipt_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(latest["result_prefix"], path)
        serialized = json.dumps(latest)
        for forbidden in ("http://", "https://", "databank", "Illinois", "PercoGuard"):
            self.assertNotIn(forbidden, serialized)

    def test_receipt_rejects_noncanonical_chunk_sequence(self):
        value = receipt(305, 1)
        value["chunks"][0]["name"] = "unexpected.bin"
        with self.assertRaises(indexer.IndexError):
            indexer.validate_receipt(value, CAPSULE, 305, 1)


if __name__ == "__main__":
    unittest.main()
