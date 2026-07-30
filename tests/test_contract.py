import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts import relay_runner
from scripts.relay_runner import (
    RelayError,
    build_container_command,
    decode_payload_text,
    safe_extract,
    validate_manifest,
    validate_run_number,
)


class ContractTests(unittest.TestCase):
    def test_manifest_accepts_only_frozen_schema(self):
        manifest = {
            "schema": 1,
            "capsule_id": "a" * 32,
            "payload_sha256": "b" * 64,
            "timeout_minutes": 30,
        }
        self.assertEqual(validate_manifest(manifest, "a" * 32), manifest)
        bad = dict(manifest, command="echo unsafe")
        with self.assertRaises(RelayError):
            validate_manifest(bad, "a" * 32)

    def test_run_numbers_are_digits_only(self):
        self.assertEqual(validate_run_number("run", "123"), "123")
        with self.assertRaises(RelayError):
            validate_run_number("run", "12-3")

    def test_payload_wrapper_is_strict_base64(self):
        self.assertEqual(decode_payload_text(b"YWJj"), b"abc")
        with self.assertRaises(RelayError):
            decode_payload_text(b"not base64!")

    def test_container_command_has_no_queue_secret_or_repository(self):
        command = build_container_command(
            Path("/tmp/payload"),
            Path("/tmp/result"),
            "c" * 32,
            "opaque-relay-1-1",
        )
        joined = "\n".join(command)
        self.assertNotIn("LAB_QUEUE_TOKEN", joined)
        self.assertNotIn("shengtenghou4-star/00", joined)
        self.assertNotIn("secret", joined)

    def test_container_command_is_hardened_and_offline(self):
        command = build_container_command(
            Path("/tmp/payload"),
            Path("/tmp/result"),
            "c" * 32,
            "opaque-relay-1-1",
        )
        joined = "\n".join(command)
        for required in (
            "--network\nnone",
            "--read-only",
            "--cap-drop\nALL",
            "--security-opt\nno-new-privileges",
            "--pids-limit\n512",
            "/capsule,readonly",
        ):
            self.assertIn(required, joined)

    def test_put_file_binds_private_result_branch(self):
        captured = {}
        original = relay_runner.api_request

        def fake_api_request(url, token, **kwargs):
            captured.update(url=url, token=token, kwargs=kwargs)
            return b"{}"

        relay_runner.api_request = fake_api_request
        try:
            relay_runner.put_file(
                "owner/queue",
                "relay/results/x/file",
                "results",
                "token",
                b"abc",
                "receipt",
            )
        finally:
            relay_runner.api_request = original
        payload = __import__("json").loads(captured["kwargs"]["data"])
        self.assertEqual(payload["branch"], "results")
        self.assertEqual(payload["content"], "YWJj")
        self.assertEqual(captured["kwargs"]["method"], "PUT")

    def test_archive_rejects_parent_traversal(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            info = tarfile.TarInfo("../escape")
            data = b"x"
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RelayError):
                safe_extract(stream.getvalue(), Path(directory))

    def test_archive_accepts_root_entrypoint(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            data = b'#!/bin/bash\nmkdir -p "$1"\necho ok > "$1/result.txt"\n'
            info = tarfile.TarInfo("run.sh")
            info.size = len(data)
            info.mode = 0o700
            archive.addfile(info, io.BytesIO(data))
        with tempfile.TemporaryDirectory() as directory:
            safe_extract(stream.getvalue(), Path(directory))
            self.assertTrue((Path(directory) / "run.sh").is_file())


if __name__ == "__main__":
    unittest.main()
