#!/usr/bin/env python3
"""Create a stable private pointer to the latest opaque public-fetch result."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

CAPSULE_RE = re.compile(r"^[0-9a-f]{32}$")
RUN_DIR_RE = re.compile(r"^([0-9]+)-([0-9]+)$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
API_ROOT = "https://api.github.com"


class IndexError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise IndexError(f"missing required environment variable: {name}")
    return value


def api_request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
) -> bytes:
    request = urllib.request.Request(url, method=method, data=data)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "opaque-public-fetch-indexer")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise IndexError(f"queue API request failed with HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise IndexError("queue API request failed") from None


def contents_url(repository: str, path: str, ref: str | None = None) -> str:
    safe_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    url = f"{API_ROOT}/repos/{repository}/contents/{safe_path}"
    if ref:
        url += "?" + urllib.parse.urlencode({"ref": ref})
    return url


def decode_file_envelope(raw: bytes) -> tuple[bytes, str]:
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        raise IndexError("queue API response is not valid JSON") from None
    if not isinstance(envelope, dict) or envelope.get("type") != "file":
        raise IndexError("queue response is not a file")
    sha = envelope.get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise IndexError("queue file envelope has an invalid blob SHA")
    if envelope.get("encoding") != "base64" or not isinstance(envelope.get("content"), str):
        raise IndexError("queue response is not a base64 file")
    try:
        content = base64.b64decode(envelope["content"], validate=False)
    except ValueError:
        raise IndexError("queue file envelope is malformed") from None
    return content, sha


def fetch_file(repository: str, path: str, ref: str, token: str) -> tuple[bytes, str]:
    return decode_file_envelope(api_request(contents_url(repository, path, ref), token))


def list_result_directories(raw: bytes, capsule_id: str) -> list[tuple[int, int, str]]:
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        raise IndexError("result directory response is not valid JSON") from None
    if not isinstance(entries, list):
        raise IndexError("result path is not a directory listing")
    expected_prefix = f"relay/public-fetch-results/{capsule_id}/"
    rows: list[tuple[int, int, str]] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "dir":
            continue
        name = entry.get("name")
        path = entry.get("path")
        if not isinstance(name, str) or not isinstance(path, str):
            raise IndexError("result directory entry is malformed")
        match = RUN_DIR_RE.fullmatch(name)
        if not match or path != expected_prefix + name:
            raise IndexError("result directory entry violates the frozen path contract")
        rows.append((int(match.group(1)), int(match.group(2)), path))
    if not rows:
        raise IndexError("no completed public-fetch result directory was found")
    return sorted(rows)


def validate_receipt(
    value: Any,
    capsule_id: str,
    run_id: int,
    run_attempt: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IndexError("public-fetch receipt is not a JSON object")
    required = {
        "schema",
        "capsule_id",
        "operation",
        "run_id",
        "run_attempt",
        "container_image",
        "network",
        "return_code",
        "timed_out",
        "started_unix",
        "finished_unix",
        "result_bytes",
        "result_sha256",
        "chunks",
    }
    if set(value) != required:
        raise IndexError("public-fetch receipt fields do not match the frozen schema")
    if value.get("schema") != 1 or value.get("operation") != "public_https_fetch":
        raise IndexError("unsupported public-fetch receipt")
    if value.get("capsule_id") != capsule_id:
        raise IndexError("public-fetch receipt capsule mismatch")
    if value.get("run_id") != str(run_id) or value.get("run_attempt") != str(run_attempt):
        raise IndexError("public-fetch receipt run identity mismatch")
    if not isinstance(value.get("return_code"), int) or not isinstance(value.get("timed_out"), bool):
        raise IndexError("public-fetch receipt completion state is invalid")
    if not isinstance(value.get("result_bytes"), int) or value["result_bytes"] < 1:
        raise IndexError("public-fetch receipt result size is invalid")
    if not isinstance(value.get("result_sha256"), str) or not HEX64_RE.fullmatch(value["result_sha256"]):
        raise IndexError("public-fetch receipt result hash is invalid")
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise IndexError("public-fetch receipt chunk list is invalid")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or set(chunk) != {"name", "bytes", "sha256"}:
            raise IndexError("public-fetch receipt chunk is malformed")
        if chunk.get("name") != f"result.tar.gz.part{index:04d}":
            raise IndexError("public-fetch receipt chunk sequence is invalid")
        if not isinstance(chunk.get("bytes"), int) or chunk["bytes"] < 1:
            raise IndexError("public-fetch receipt chunk size is invalid")
        if not isinstance(chunk.get("sha256"), str) or not HEX64_RE.fullmatch(chunk["sha256"]):
            raise IndexError("public-fetch receipt chunk hash is invalid")
    return value


def build_latest_index(
    capsule_id: str,
    run_id: int,
    run_attempt: int,
    result_path: str,
    receipt_raw: bytes,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "capsule_id": capsule_id,
        "result_prefix": result_path,
        "receipt_path": f"{result_path}/receipt.json",
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "run_id": str(run_id),
        "run_attempt": str(run_attempt),
        "return_code": receipt["return_code"],
        "timed_out": receipt["timed_out"],
        "result_bytes": receipt["result_bytes"],
        "result_sha256": receipt["result_sha256"],
        "chunks": receipt["chunks"],
    }


def put_latest(
    repository: str,
    path: str,
    branch: str,
    token: str,
    content: bytes,
) -> None:
    existing_sha: str | None = None
    try:
        _, existing_sha = fetch_file(repository, path, branch, token)
    except IndexError as exc:
        if "HTTP 404" not in str(exc):
            raise
    payload: dict[str, Any] = {
        "message": "Index latest opaque public fetch result",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": branch,
    }
    if existing_sha is not None:
        payload["sha"] = existing_sha
    api_request(
        contents_url(repository, path),
        token,
        method="PUT",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def main() -> int:
    capsule_id = required_env("RELAY_CAPSULE_ID")
    if not CAPSULE_RE.fullmatch(capsule_id):
        raise IndexError("capsule ID must be 32 lowercase hexadecimal characters")
    repository = required_env("LAB_QUEUE_REPOSITORY")
    ref = required_env("LAB_QUEUE_RESULTS_REF")
    token = required_env("LAB_QUEUE_TOKEN")
    root = f"relay/public-fetch-results/{capsule_id}"
    listing = api_request(contents_url(repository, root, ref), token)
    rows = list_result_directories(listing, capsule_id)

    valid: list[tuple[int, int, str, bytes, dict[str, Any]]] = []
    for run_id, run_attempt, path in rows:
        try:
            receipt_raw, _ = fetch_file(repository, f"{path}/receipt.json", ref, token)
            receipt_value = json.loads(receipt_raw)
            receipt = validate_receipt(receipt_value, capsule_id, run_id, run_attempt)
        except (IndexError, json.JSONDecodeError):
            continue
        valid.append((run_id, run_attempt, path, receipt_raw, receipt))
    if not valid:
        raise IndexError("no valid completed public-fetch receipt was found")

    run_id, run_attempt, path, receipt_raw, receipt = max(valid, key=lambda row: (row[0], row[1]))
    latest = build_latest_index(
        capsule_id, run_id, run_attempt, path, receipt_raw, receipt
    )
    put_latest(
        repository,
        f"{root}/latest.json",
        ref,
        token,
        (json.dumps(latest, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print("Opaque public-fetch result indexed privately.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IndexError as exc:
        print(f"Public-fetch indexing failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
