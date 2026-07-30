#!/usr/bin/env python3
"""Fail-closed runner for an opaque capsule stored in a dedicated private queue."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

CAPSULE_RE = re.compile(r"^[0-9a-f]{32}$")
RUN_NUMBER_RE = re.compile(r"^[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_PAYLOAD_BYTES = 75 * 1024 * 1024
MAX_RESULT_BYTES = 90 * 1024 * 1024
API_ROOT = "https://api.github.com"
CONTAINER_IMAGE = "python:3.12-bookworm"


class RelayError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RelayError(f"missing required environment variable: {name}")
    return value


def validate_run_number(name: str, value: str) -> str:
    if not RUN_NUMBER_RE.fullmatch(value):
        raise RelayError(f"{name} must contain decimal digits only")
    return value


def api_request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    accept: str = "application/vnd.github+json",
) -> bytes:
    request = urllib.request.Request(url, method=method, data=data)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", accept)
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "opaque-task-relay")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RelayError(f"queue API request failed with HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise RelayError("queue API request failed") from None


def contents_url(repository: str, path: str, ref: str | None = None) -> str:
    safe_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    url = f"{API_ROOT}/repos/{repository}/contents/{safe_path}"
    if ref:
        url += "?" + urllib.parse.urlencode({"ref": ref})
    return url


def fetch_contents_bytes(repository: str, path: str, ref: str, token: str) -> bytes:
    raw = api_request(contents_url(repository, path, ref), token)
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        raise RelayError("queue API response is not valid JSON") from None
    if envelope.get("type") != "file" or envelope.get("encoding") != "base64":
        raise RelayError("queue response is not a base64 file")
    try:
        return base64.b64decode(envelope["content"], validate=False)
    except (KeyError, ValueError):
        raise RelayError("queue file envelope is malformed") from None


def fetch_json(repository: str, path: str, ref: str, token: str) -> dict[str, Any]:
    decoded = fetch_contents_bytes(repository, path, ref, token)
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError:
        raise RelayError("manifest is not valid JSON") from None
    if not isinstance(value, dict):
        raise RelayError("manifest must be a JSON object")
    return value


def decode_payload_text(encoded: bytes) -> bytes:
    try:
        compact = b"".join(encoded.split())
        payload = base64.b64decode(compact, validate=True)
    except ValueError:
        raise RelayError("payload wrapper is not valid base64") from None
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise RelayError("payload exceeds the 75 MiB request limit")
    return payload


def validate_manifest(manifest: dict[str, Any], capsule_id: str) -> dict[str, Any]:
    allowed = {"schema", "capsule_id", "payload_sha256", "timeout_minutes"}
    if set(manifest) != allowed:
        raise RelayError("manifest fields do not match the frozen schema")
    if manifest.get("schema") != 1:
        raise RelayError("unsupported manifest schema")
    if manifest.get("capsule_id") != capsule_id:
        raise RelayError("manifest capsule ID mismatch")
    digest = manifest.get("payload_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise RelayError("invalid payload SHA-256")
    timeout = manifest.get("timeout_minutes")
    if not isinstance(timeout, int) or not 1 <= timeout <= 330:
        raise RelayError("timeout_minutes must be an integer from 1 to 330")
    return manifest


def safe_extract(archive_bytes: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RelayError("payload archive is empty")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RelayError("payload archive contains unsafe paths")
            if member.issym() or member.islnk() or member.isdev():
                raise RelayError("payload archive contains prohibited link or device entries")
        archive.extractall(destination, filter="data")
    entrypoint = destination / "run.sh"
    if not entrypoint.is_file():
        raise RelayError("payload archive lacks root-level run.sh")
    entrypoint.chmod(0o700)


def build_container_command(
    payload_root: Path,
    result_root: Path,
    capsule_id: str,
    container_name: str,
) -> list[str]:
    uid = os.getuid()
    gid = os.getgid()
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--memory",
        "6g",
        "--cpus",
        "2",
        "--user",
        f"{uid}:{gid}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
        "--mount",
        f"type=bind,src={payload_root},dst=/capsule,readonly",
        "--mount",
        f"type=bind,src={result_root},dst=/result",
        "--workdir",
        "/capsule",
        "--env",
        f"CAPSULE_ID={capsule_id}",
        CONTAINER_IMAGE,
        "/bin/bash",
        "/capsule/run.sh",
        "/result/output",
    ]


def stop_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )


def pack_results(result_root: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(result_root.rglob("*")):
            if path.is_symlink():
                raise RelayError("result directory contains a symbolic link")
            archive.add(path, arcname=path.relative_to(result_root), recursive=False)
    data = stream.getvalue()
    if len(data) > MAX_RESULT_BYTES:
        raise RelayError("result package exceeds the 90 MiB private-return limit")
    return data


def put_file(
    repository: str,
    path: str,
    branch: str,
    token: str,
    content: bytes,
    message: str,
) -> None:
    payload = json.dumps(
        {
            "message": message,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": branch,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    api_request(contents_url(repository, path), token, method="PUT", data=payload)


def main() -> int:
    capsule_id = required_env("RELAY_CAPSULE_ID")
    if not CAPSULE_RE.fullmatch(capsule_id):
        raise RelayError("capsule ID must be 32 lowercase hexadecimal characters")

    repository = required_env("LAB_QUEUE_REPOSITORY")
    ref = required_env("LAB_QUEUE_REF")
    write_branch = required_env("LAB_QUEUE_WRITE_BRANCH")
    token = required_env("LAB_QUEUE_TOKEN")
    run_id = validate_run_number("RELAY_RUN_ID", required_env("RELAY_RUN_ID"))
    run_attempt = validate_run_number(
        "RELAY_RUN_ATTEMPT", required_env("RELAY_RUN_ATTEMPT")
    )

    request_root = f"relay/requests/{capsule_id}"
    manifest = validate_manifest(
        fetch_json(repository, f"{request_root}/manifest.json", ref, token),
        capsule_id,
    )
    payload_text = fetch_contents_bytes(
        repository, f"{request_root}/payload.tar.gz.b64", ref, token
    )
    payload = decode_payload_text(payload_text)
    if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
        raise RelayError("payload SHA-256 mismatch")

    with tempfile.TemporaryDirectory(prefix="opaque-relay-") as temporary:
        root = Path(temporary)
        payload_root = root / "payload"
        result_root = root / "result"
        payload_root.mkdir()
        result_root.mkdir()
        safe_extract(payload, payload_root)

        container_name = f"opaque-relay-{run_id}-{run_attempt}"
        started = int(time.time())
        stdout_path = result_root / "stdout.txt"
        stderr_path = result_root / "stderr.txt"
        return_code = 124
        timed_out = False
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            try:
                completed = subprocess.run(
                    build_container_command(
                        payload_root, result_root, capsule_id, container_name
                    ),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=manifest["timeout_minutes"] * 60,
                    check=False,
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_container(container_name)

        finished = int(time.time())
        receipt = {
            "schema": 1,
            "capsule_id": capsule_id,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "payload_sha256": manifest["payload_sha256"],
            "container_image": CONTAINER_IMAGE,
            "network": "none",
            "return_code": return_code,
            "timed_out": timed_out,
            "started_unix": started,
            "finished_unix": finished,
        }
        (result_root / "receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        package = pack_results(result_root)
        receipt["result_sha256"] = hashlib.sha256(package).hexdigest()
        receipt_bytes = (
            json.dumps(receipt, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")

        result_prefix = f"relay/results/{capsule_id}/{run_id}-{run_attempt}"
        put_file(
            repository,
            f"{result_prefix}/result.tar.gz",
            write_branch,
            token,
            package,
            "Store opaque relay result",
        )
        put_file(
            repository,
            f"{result_prefix}/receipt.json",
            write_branch,
            token,
            receipt_bytes,
            "Store opaque relay receipt",
        )

    print("Opaque capsule completed; private result receipt stored.")
    return 0 if return_code == 0 and not timed_out else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RelayError as exc:
        print(f"Relay failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
