#!/usr/bin/env python3
"""Validate and store a fixed public-source tool build in the dedicated private queue."""

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
from pathlib import Path

ID_RE = re.compile(r"^[0-9a-f]{32}$")
NUMBER_RE = re.compile(r"^[0-9]+$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
API_ROOT = "https://api.github.com"
MIN_BINARY_BYTES = 10_000
MAX_BINARY_BYTES = 20 * 1024 * 1024


class ToolStoreError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ToolStoreError(f"missing required environment variable: {name}")
    return value


def validate_number(name: str, value: str) -> str:
    if not NUMBER_RE.fullmatch(value):
        raise ToolStoreError(f"{name} must contain decimal digits only")
    return value


def contents_url(repository: str, path: str, ref: str | None = None) -> str:
    safe_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    url = f"{API_ROOT}/repos/{repository}/contents/{safe_path}"
    if ref:
        url += "?" + urllib.parse.urlencode({"ref": ref})
    return url


def api_request(url: str, token: str, *, method: str = "GET", data: bytes | None = None) -> bytes:
    request = urllib.request.Request(url, method=method, data=data)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "opaque-fixed-tool-store")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise ToolStoreError(f"private queue API request failed with HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise ToolStoreError("private queue API request failed") from None


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


def validate_binary(path: Path) -> tuple[bytes, str]:
    if not path.is_file() or path.is_symlink():
        raise ToolStoreError("fixed tool binary is missing or unsafe")
    data = path.read_bytes()
    if not MIN_BINARY_BYTES <= len(data) <= MAX_BINARY_BYTES:
        raise ToolStoreError("fixed tool binary violates the size gate")
    if not data.startswith(b"\x7fELF"):
        raise ToolStoreError("fixed tool binary lacks the ELF magic")
    return data, hashlib.sha256(data).hexdigest()


def validate_version(path: Path) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 4096:
        raise ToolStoreError("fixed tool version receipt is missing or oversized")
    value = path.read_text(encoding="utf-8", errors="strict").strip()
    if not value or any(ord(char) < 32 and char not in "\t" for char in value):
        raise ToolStoreError("fixed tool version receipt is invalid")
    return value


def main() -> int:
    if len(sys.argv) != 3:
        raise ToolStoreError("store requires binary and version paths")
    tool_id = required_env("FIXED_TOOL_ID")
    if not ID_RE.fullmatch(tool_id):
        raise ToolStoreError("fixed tool ID is invalid")
    source_commit = required_env("FIXED_TOOL_SOURCE_COMMIT")
    if not SHA_RE.fullmatch(source_commit):
        raise ToolStoreError("fixed tool source commit is invalid")
    run_id = validate_number("RELAY_RUN_ID", required_env("RELAY_RUN_ID"))
    run_attempt = validate_number("RELAY_RUN_ATTEMPT", required_env("RELAY_RUN_ATTEMPT"))
    repository = required_env("LAB_QUEUE_REPOSITORY")
    branch = required_env("LAB_QUEUE_WRITE_BRANCH")
    token = required_env("LAB_QUEUE_TOKEN")
    image = required_env("FIXED_TOOL_BUILD_IMAGE")

    binary, digest = validate_binary(Path(sys.argv[1]))
    version = validate_version(Path(sys.argv[2]))
    prefix = f"relay/fixed-tools/{tool_id}/{run_id}-{run_attempt}"
    put_file(repository, f"{prefix}/binary", branch, token, binary, "Store opaque fixed tool binary")
    receipt = {
        "schema": 1,
        "tool_id": tool_id,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "source_commit": source_commit,
        "build_image": image,
        "binary_bytes": len(binary),
        "binary_sha256": digest,
        "elf_magic": "7f454c46",
        "version": version,
    }
    put_file(
        repository,
        f"{prefix}/receipt.json",
        branch,
        token,
        (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        "Store opaque fixed tool receipt",
    )
    print("Private fixed-tool receipt stored.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ToolStoreError as exc:
        print(f"Fixed tool store failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
