#!/usr/bin/env python3
"""Fail-closed orchestrator for opaque public HTTPS acquisition capsules."""

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
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HOST_RE = re.compile(r"^[a-z0-9.-]+$")
API_ROOT = "https://api.github.com"
CONTAINER_IMAGE = "python:3.12-bookworm"
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 8
RETURN_CHUNK_BYTES = 48 * 1024 * 1024


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
    request.add_header("User-Agent", "opaque-public-fetch-relay")
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


def validate_https_url(value: Any, allowed_hosts: set[str]) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise RelayError("public fetch URL is invalid")
    parsed = urllib.parse.urlsplit(value)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or host not in allowed_hosts
        or parsed.fragment
    ):
        raise RelayError("public fetch URL violates the HTTPS host contract")
    return value


def validate_manifest(value: Any, capsule_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RelayError("public fetch manifest must be a JSON object")
    allowed = {
        "schema",
        "capsule_id",
        "operation",
        "timeout_minutes",
        "allowed_hosts",
        "landing_url",
        "max_total_bytes",
        "files",
    }
    if set(value) != allowed:
        raise RelayError("public fetch manifest fields do not match the frozen schema")
    if value.get("schema") != 1 or value.get("operation") != "public_https_fetch":
        raise RelayError("unsupported public fetch manifest")
    if value.get("capsule_id") != capsule_id:
        raise RelayError("public fetch capsule ID mismatch")
    timeout = value.get("timeout_minutes")
    if not isinstance(timeout, int) or not 1 <= timeout <= 120:
        raise RelayError("timeout_minutes must be an integer from 1 to 120")
    max_total = value.get("max_total_bytes")
    if not isinstance(max_total, int) or not 1 <= max_total <= MAX_TOTAL_BYTES:
        raise RelayError("max_total_bytes exceeds the relay contract")
    hosts = value.get("allowed_hosts")
    if (
        not isinstance(hosts, list)
        or not 1 <= len(hosts) <= 8
        or len(set(hosts)) != len(hosts)
    ):
        raise RelayError("allowed_hosts must be a unique non-empty list")
    normalized_hosts: list[str] = []
    for host in hosts:
        if not isinstance(host, str):
            raise RelayError("allowed host is invalid")
        normalized = host.lower()
        if (
            normalized != host
            or not HOST_RE.fullmatch(normalized)
            or normalized.startswith(".")
            or normalized.endswith(".")
            or ".." in normalized
        ):
            raise RelayError("allowed host is invalid")
        normalized_hosts.append(normalized)
    allowed_host_set = set(normalized_hosts)
    landing = value.get("landing_url")
    if landing is not None:
        validate_https_url(landing, allowed_host_set)
    files = value.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise RelayError("files must contain between 1 and 8 entries")
    seen_names: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "urls",
            "min_bytes",
            "max_bytes",
            "magic_hex",
        }:
            raise RelayError("public fetch file entry violates the frozen schema")
        name = entry.get("name")
        if not isinstance(name, str) or not SAFE_NAME_RE.fullmatch(name) or name in seen_names:
            raise RelayError("public fetch output name is unsafe or duplicated")
        seen_names.add(name)
        urls = entry.get("urls")
        if not isinstance(urls, list) or not 1 <= len(urls) <= 8:
            raise RelayError("each file requires between 1 and 8 candidate URLs")
        for url in urls:
            validate_https_url(url, allowed_host_set)
        minimum = entry.get("min_bytes")
        maximum = entry.get("max_bytes")
        if (
            not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 1
            or maximum < minimum
            or maximum > MAX_TOTAL_BYTES
        ):
            raise RelayError("public fetch file size bounds are invalid")
        magic_hex = entry.get("magic_hex")
        if (
            not isinstance(magic_hex, str)
            or len(magic_hex) > 64
            or len(magic_hex) % 2
            or not re.fullmatch(r"[0-9a-f]*", magic_hex)
        ):
            raise RelayError("magic_hex is invalid")
    return value


def fetch_manifest(repository: str, path: str, ref: str, token: str, capsule_id: str) -> dict[str, Any]:
    decoded = fetch_contents_bytes(repository, path, ref, token)
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError:
        raise RelayError("public fetch manifest is not valid JSON") from None
    return validate_manifest(value, capsule_id)


def build_container_command(
    manifest_path: Path,
    worker_path: Path,
    result_root: Path,
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
        "bridge",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--memory",
        "4g",
        "--cpus",
        "2",
        "--user",
        f"{uid}:{gid}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
        "--mount",
        f"type=bind,src={manifest_path},dst=/request/manifest.json,readonly",
        "--mount",
        f"type=bind,src={worker_path},dst=/worker/public_fetch_worker.py,readonly",
        "--mount",
        f"type=bind,src={result_root},dst=/result",
        CONTAINER_IMAGE,
        "python",
        "/worker/public_fetch_worker.py",
        "/request/manifest.json",
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
                raise RelayError("public fetch result directory contains a symbolic link")
            archive.add(path, arcname=path.relative_to(result_root), recursive=False)
    return stream.getvalue()


def split_bytes(data: bytes, chunk_bytes: int = RETURN_CHUNK_BYTES) -> list[bytes]:
    if chunk_bytes < 1:
        raise RelayError("invalid return chunk size")
    return [data[start : start + chunk_bytes] for start in range(0, len(data), chunk_bytes)] or [b""]


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

    request_path = f"relay/public-fetch/{capsule_id}/manifest.json"
    manifest = fetch_manifest(repository, request_path, ref, token, capsule_id)
    worker_path = Path(__file__).with_name("public_fetch_worker.py").resolve()
    if not worker_path.is_file():
        raise RelayError("public fetch worker is missing")

    with tempfile.TemporaryDirectory(prefix="opaque-public-fetch-") as temporary:
        root = Path(temporary)
        result_root = root / "result"
        result_root.mkdir()
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o444)

        container_name = f"opaque-public-fetch-{run_id}-{run_attempt}"
        started = int(time.time())
        stdout_path = result_root / "stdout.txt"
        stderr_path = result_root / "stderr.txt"
        return_code = 124
        timed_out = False
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            try:
                completed = subprocess.run(
                    build_container_command(
                        manifest_path, worker_path, result_root, container_name
                    ),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=manifest["timeout_minutes"] * 60,
                    check=False,
                    env={"PATH": os.environ.get("PATH", "")},
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_container(container_name)

        finished = int(time.time())
        internal_receipt = {
            "schema": 1,
            "capsule_id": capsule_id,
            "operation": "public_https_fetch",
            "run_id": run_id,
            "run_attempt": run_attempt,
            "container_image": CONTAINER_IMAGE,
            "network": "bridge",
            "return_code": return_code,
            "timed_out": timed_out,
            "started_unix": started,
            "finished_unix": finished,
        }
        (result_root / "relay_receipt.json").write_text(
            json.dumps(internal_receipt, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        package = pack_results(result_root)
        chunks = split_bytes(package)
        chunk_rows = []
        result_prefix = f"relay/public-fetch-results/{capsule_id}/{run_id}-{run_attempt}"
        for index, chunk in enumerate(chunks):
            name = f"result.tar.gz.part{index:04d}"
            put_file(
                repository,
                f"{result_prefix}/{name}",
                write_branch,
                token,
                chunk,
                "Store opaque public fetch result chunk",
            )
            chunk_rows.append(
                {
                    "name": name,
                    "bytes": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                }
            )

        external_receipt = dict(internal_receipt)
        external_receipt.update(
            {
                "result_bytes": len(package),
                "result_sha256": hashlib.sha256(package).hexdigest(),
                "chunks": chunk_rows,
            }
        )
        put_file(
            repository,
            f"{result_prefix}/receipt.json",
            write_branch,
            token,
            (json.dumps(external_receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            "Store opaque public fetch receipt",
        )

    print("Opaque public acquisition completed; private receipt stored.")
    return 0 if return_code == 0 and not timed_out else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RelayError as exc:
        print(f"Public acquisition failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
