#!/usr/bin/env python3
"""Re-download and independently verify a private fixed-tool binary."""

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

ID_RE = re.compile(r"^[0-9a-f]{32}$")
NUMBER_RE = re.compile(r"^[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
API_ROOT = "https://api.github.com"
MAX_BINARY_BYTES = 20 * 1024 * 1024


class ToolVerificationError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ToolVerificationError(f"missing required environment variable: {name}")
    return value


def validate_number(name: str, value: str) -> str:
    if not NUMBER_RE.fullmatch(value):
        raise ToolVerificationError(f"{name} must contain decimal digits only")
    return value


def contents_url(repository: str, path: str, ref: str | None = None) -> str:
    safe_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    url = f"{API_ROOT}/repos/{repository}/contents/{safe_path}"
    if ref:
        url += "?" + urllib.parse.urlencode({"ref": ref})
    return url


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
    request.add_header("User-Agent", "opaque-fixed-tool-verifier")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise ToolVerificationError(f"private queue API request failed with HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise ToolVerificationError("private queue API request failed") from None


def fetch_bytes(repository: str, path: str, branch: str, token: str) -> bytes:
    return api_request(
        contents_url(repository, path, branch),
        token,
        accept="application/vnd.github.raw+json",
    )


def fetch_json(repository: str, path: str, branch: str, token: str) -> tuple[dict[str, Any], bytes]:
    raw = fetch_bytes(repository, path, branch, token)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ToolVerificationError("private fixed-tool receipt is malformed") from None
    if not isinstance(value, dict):
        raise ToolVerificationError("private fixed-tool receipt must be an object")
    return value, raw


def put_file(repository: str, path: str, branch: str, token: str, content: bytes, message: str) -> None:
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
    tool_id = required_env("FIXED_TOOL_ID")
    if not ID_RE.fullmatch(tool_id):
        raise ToolVerificationError("fixed tool ID is invalid")
    run_id = validate_number("RELAY_RUN_ID", required_env("RELAY_RUN_ID"))
    run_attempt = validate_number("RELAY_RUN_ATTEMPT", required_env("RELAY_RUN_ATTEMPT"))
    repository = required_env("LAB_QUEUE_REPOSITORY")
    branch = required_env("LAB_QUEUE_WRITE_BRANCH")
    token = required_env("LAB_QUEUE_TOKEN")
    prefix = f"relay/fixed-tools/{tool_id}/{run_id}-{run_attempt}"

    receipt, receipt_bytes = fetch_json(repository, f"{prefix}/receipt.json", branch, token)
    if (
        receipt.get("schema") != 1
        or receipt.get("tool_id") != tool_id
        or str(receipt.get("run_id")) != run_id
        or str(receipt.get("run_attempt")) != run_attempt
        or receipt.get("elf_magic") != "7f454c46"
    ):
        raise ToolVerificationError("fixed-tool receipt identity binding is invalid")
    expected_bytes = receipt.get("binary_bytes")
    expected_sha = receipt.get("binary_sha256")
    if (
        not isinstance(expected_bytes, int)
        or not 10_000 <= expected_bytes <= MAX_BINARY_BYTES
        or not isinstance(expected_sha, str)
        or not SHA256_RE.fullmatch(expected_sha)
    ):
        raise ToolVerificationError("fixed-tool receipt binary binding is invalid")
    binary = fetch_bytes(repository, f"{prefix}/binary", branch, token)
    if len(binary) != expected_bytes or hashlib.sha256(binary).hexdigest() != expected_sha:
        raise ToolVerificationError("re-downloaded fixed-tool binary differs from receipt")
    if not binary.startswith(b"\x7fELF"):
        raise ToolVerificationError("re-downloaded fixed-tool binary lacks ELF magic")

    certificate = {
        "schema": 1,
        "tool_id": tool_id,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "operation": "fixed_tool_independent_verification",
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "binary_bytes": len(binary),
        "binary_sha256": expected_sha,
        "elf_magic": "7f454c46",
        "source_commit": receipt.get("source_commit"),
        "version": receipt.get("version"),
        "verdict": "VERIFIED",
    }
    put_file(
        repository,
        f"{prefix}/verification.json",
        branch,
        token,
        (json.dumps(certificate, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        "Store independent fixed tool verification",
    )
    print("Private fixed-tool verification stored.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ToolVerificationError as exc:
        print(f"Fixed tool verification failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
