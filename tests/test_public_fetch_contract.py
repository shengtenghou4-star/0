import json
import unittest
from pathlib import Path

from scripts import public_fetch_runner, public_fetch_worker


CAPSULE = "a" * 32


def good_manifest():
    return {
        "schema": 1,
        "capsule_id": CAPSULE,
        "operation": "public_https_fetch",
        "timeout_minutes": 25,
        "allowed_hosts": ["data.example.org"],
        "landing_url": "https://data.example.org/landing",
        "max_total_bytes": 20_000_000,
        "files": [
            {
                "name": "archive.rar",
                "urls": ["https://data.example.org/files/archive"],
                "min_bytes": 1_000,
                "max_bytes": 20_000_000,
                "magic_hex": "526172211a070100",
            }
        ],
    }


class PublicFetchContractTests(unittest.TestCase):
    def test_manifest_accepts_frozen_schema(self):
        manifest = good_manifest()
        self.assertEqual(public_fetch_runner.validate_manifest(manifest, CAPSULE), manifest)

    def test_manifest_rejects_private_or_unlisted_url(self):
        manifest = good_manifest()
        manifest["files"][0]["urls"] = ["https://127.0.0.1/private"]
        with self.assertRaises(public_fetch_runner.RelayError):
            public_fetch_runner.validate_manifest(manifest, CAPSULE)

    def test_manifest_rejects_path_output(self):
        manifest = good_manifest()
        manifest["files"][0]["name"] = "../escape"
        with self.assertRaises(public_fetch_runner.RelayError):
            public_fetch_runner.validate_manifest(manifest, CAPSULE)

    def test_manifest_rejects_extra_command(self):
        manifest = good_manifest()
        manifest["command"] = "curl unsafe"
        with self.assertRaises(public_fetch_runner.RelayError):
            public_fetch_runner.validate_manifest(manifest, CAPSULE)

    def test_container_is_networked_but_secretless_and_hardened(self):
        command = public_fetch_runner.build_container_command(
            Path("/tmp/manifest"),
            Path("/tmp/worker"),
            Path("/tmp/result"),
            "opaque-public-fetch-1-1",
        )
        joined = "\n".join(command)
        for required in (
            "--network\nbridge",
            "--read-only",
            "--cap-drop\nALL",
            "--security-opt\nno-new-privileges",
            "/request/manifest.json,readonly",
            "/worker/public_fetch_worker.py,readonly",
        ):
            self.assertIn(required, joined)
        for forbidden in (
            "LAB_QUEUE_TOKEN",
            "shengtenghou4-star/00",
            "GITHUB_TOKEN",
            "/var/run/docker.sock",
        ):
            self.assertNotIn(forbidden, joined)

    def test_split_bytes_is_stable_and_complete(self):
        data = b"abcdefghij"
        chunks = public_fetch_runner.split_bytes(data, 4)
        self.assertEqual(chunks, [b"abcd", b"efgh", b"ij"])
        self.assertEqual(b"".join(chunks), data)

    def test_private_failure_diagnostics_are_machine_readable(self):
        attempts = [
            {
                "attempt": 1,
                "url": "https://data.example.org/a",
                "final_url": "https://data.example.org/b",
                "status": 403,
                "bytes": 0,
                "ok": False,
                "error_type": "http",
                "error": "HTTP 403: Forbidden",
            }
        ]
        value = json.loads(public_fetch_worker.format_failure("archive.rar", attempts))
        self.assertEqual(value["output"], "archive.rar")
        self.assertEqual(value["attempts"], attempts)


if __name__ == "__main__":
    unittest.main()
