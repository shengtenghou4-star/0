#!/usr/bin/env python3
"""Independently verify a private public-fetch package and store a compact certificate."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

CAPSULE_RE = re.compile(r"^[0-9a-f]{32}$")
RUN_NUMBER_RE = re.compile(r"^[0-9]+$")
CHUNK_RE = re.compile(r"^result\.tar\.gz\.part([0-9]{4})$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
API_ROOT = "https://api.github.com"
MAX_PACKAGE_BYTES = 600 * 1024 * 1024
MAX_CHUNKS = 16
MAX_MEMBERS = 128
MAX_RECEIPT_BYTES = 2 * 1024 * 1024


class VerificationError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise VerificationError(f"missing required environment variable: {name}")
    return value


def validate_number(name: str, value: str) -> str:
    if not RUN_NUMBER_RE.fullmatch(value):
        raise VerificationError(f"{name} must contain decimal digits only")
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
    request.add_header("User-Agent", "opaque-public-fetch-verifier")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise VerificationError(f"private queue API request failed with HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise VerificationError("private queue API request failed") from None


def contents_url(repository: str, path: str, ref: str | None = None) -> str:
    safe_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    url = f"{API_ROOT}/repos/{repository}/contents/{safe_path}"
    if ref:
        url += "?" + urllib.parse.urlencode({"ref": ref})
    return url


def fetch_contents_bytes(repository: str, path: str, ref: str, token: str) -> bytes:
    return api_request(
        contents_url(repository, path, ref),
        token,
        accept="application/vnd.github.raw+json",
    )


def fetch_json(repository: str, path: str, ref: str, token: str) -> tuple[dict[str, Any], bytes]:
    raw = fetch_contents_bytes(repository, path, ref, token)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise VerificationError("private JSON file is malformed") from None
    if not isinstance(value, dict):
        raise VerificationError("private JSON file must be an object")
    return value, raw


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


def validate_manifest(manifest: dict[str, Any], capsule_id: str) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != 1 or manifest.get("operation") != "public_https_fetch":
        raise VerificationError("manifest operation is not public_https_fetch schema 1")
    if manifest.get("capsule_id") != capsule_id:
        raise VerificationError("manifest capsule ID mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise VerificationError("manifest files are missing")
    rows: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise VerificationError("manifest file entry is malformed")
        name = entry.get("name")
        minimum = entry.get("min_bytes")
        maximum = entry.get("max_bytes")
        magic_hex = entry.get("magic_hex")
        if (
            not isinstance(name, str)
            or not SAFE_NAME_RE.fullmatch(name)
            or name in rows
            or not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 1
            or maximum < minimum
            or not isinstance(magic_hex, str)
            or len(magic_hex) % 2
            or not re.fullmatch(r"[0-9a-f]*", magic_hex)
        ):
            raise VerificationError("manifest file verification contract is invalid")
        rows[name] = {
            "min_bytes": minimum,
            "max_bytes": maximum,
            "magic_hex": magic_hex,
        }
    return rows


def validate_chunk_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = receipt.get("chunks")
    if not isinstance(chunks, list) or not 1 <= len(chunks) <= MAX_CHUNKS:
        raise VerificationError("result chunk list is invalid")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(chunks):
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            raise VerificationError("result chunk entry is malformed")
        name = row.get("name")
        size = row.get("bytes")
        digest = row.get("sha256")
        match = CHUNK_RE.fullmatch(name) if isinstance(name, str) else None
        if (
            match is None
            or int(match.group(1)) != index
            or not isinstance(size, int)
            or not 0 <= size <= 50 * 1024 * 1024
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
        ):
            raise VerificationError("result chunks are not contiguous and bounded")
        normalized.append(row)
    return normalized


def validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or member.issym()
        or member.islnk()
        or member.isdev()
    ):
        raise VerificationError("result archive contains an unsafe member")


def hash_stream(stream: BinaryIO, *, prefix_bytes: int = 0) -> tuple[int, str, bytes]:
    digest = hashlib.sha256()
    size = 0
    prefix = b""
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        if prefix_bytes and len(prefix) < prefix_bytes:
            prefix += block[: prefix_bytes - len(prefix)]
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest(), prefix


def verify_archive(
    package_path: Path,
    manifest_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    try:
        archive = tarfile.open(package_path, mode="r:gz")
    except tarfile.TarError:
        raise VerificationError("result package is not a valid gzip tar archive") from None
    with archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_MEMBERS:
            raise VerificationError("result archive member count is invalid")
        by_name: dict[str, tarfile.TarInfo] = {}
        for member in members:
            validate_member(member)
            if member.name in by_name:
                raise VerificationError("result archive contains duplicate members")
            by_name[member.name] = member

        worker_receipt_name = "output/receipts/acquisition_receipt.json"
        worker_member = by_name.get(worker_receipt_name)
        if worker_member is None or not worker_member.isfile() or worker_member.size > MAX_RECEIPT_BYTES:
            raise VerificationError("worker acquisition receipt is missing or oversized")
        worker_stream = archive.extractfile(worker_member)
        if worker_stream is None:
            raise VerificationError("worker acquisition receipt cannot be read")
        worker_bytes = worker_stream.read(MAX_RECEIPT_BYTES + 1)
        if len(worker_bytes) > MAX_RECEIPT_BYTES:
            raise VerificationError("worker acquisition receipt exceeds the size cap")
        try:
            worker_receipt = json.loads(worker_bytes)
        except json.JSONDecodeError:
            raise VerificationError("worker acquisition receipt is malformed") from None
        worker_files = worker_receipt.get("files") if isinstance(worker_receipt, dict) else None
        if not isinstance(worker_files, list):
            raise VerificationError("worker acquisition receipt lacks file rows")
        worker_map: dict[str, dict[str, Any]] = {}
        for row in worker_files:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                raise VerificationError("worker acquisition file row is malformed")
            if row["name"] in worker_map:
                raise VerificationError("worker acquisition receipt duplicates a file")
            worker_map[row["name"]] = row
        if set(worker_map) != set(manifest_rows):
            raise VerificationError("worker acquisition receipt file set differs from manifest")

        raw_members = {
            name.removeprefix("output/raw/"): member
            for name, member in by_name.items()
            if name.startswith("output/raw/") and member.isfile()
        }
        if set(raw_members) != set(manifest_rows):
            raise VerificationError("result archive raw file set differs from manifest")

        verified: list[dict[str, Any]] = []
        for name in sorted(manifest_rows):
            contract = manifest_rows[name]
            member = raw_members[name]
            stream = archive.extractfile(member)
            if stream is None:
                raise VerificationError("raw result file cannot be read")
            expected_magic = bytes.fromhex(contract["magic_hex"])
            size, digest, prefix = hash_stream(stream, prefix_bytes=len(expected_magic))
            if not contract["min_bytes"] <= size <= contract["max_bytes"]:
                raise VerificationError("raw result file violates manifest byte bounds")
            if expected_magic and prefix != expected_magic:
                raise VerificationError("raw result file violates manifest magic bytes")
            worker_row = worker_map[name]
            if worker_row.get("bytes") != size or worker_row.get("sha256") != digest:
                raise VerificationError("host hash differs from worker acquisition receipt")
            verified.append(
                {
                    "name": name,
                    "bytes": size,
                    "sha256": digest,
                    "magic_hex": contract["magic_hex"],
                    "worker_receipt_match": True,
                }
            )
        return verified, hashlib.sha256(worker_bytes).hexdigest()


def main() -> int:
    capsule_id = required_env("RELAY_CAPSULE_ID")
    if not CAPSULE_RE.fullmatch(capsule_id):
        raise VerificationError("capsule ID is invalid")
    run_id = validate_number("RELAY_RUN_ID", required_env("RELAY_RUN_ID"))
    run_attempt = validate_number(
        "RELAY_RUN_ATTEMPT", required_env("RELAY_RUN_ATTEMPT")
    )
    repository = required_env("LAB_QUEUE_REPOSITORY")
    request_ref = required_env("LAB_QUEUE_REF")
    result_branch = required_env("LAB_QUEUE_WRITE_BRANCH")
    token = required_env("LAB_QUEUE_TOKEN")

    manifest, manifest_bytes = fetch_json(
        repository,
        f"relay/public-fetch/{capsule_id}/manifest.json",
        request_ref,
        token,
    )
    manifest_rows = validate_manifest(manifest, capsule_id)
    prefix = f"relay/public-fetch-results/{capsule_id}/{run_id}-{run_attempt}"
    receipt, receipt_bytes = fetch_json(
        repository,
        f"{prefix}/receipt.json",
        result_branch,
        token,
    )
    if (
        receipt.get("schema") != 1
        or receipt.get("capsule_id") != capsule_id
        or receipt.get("operation") != "public_https_fetch"
        or str(receipt.get("run_id")) != run_id
        or str(receipt.get("run_attempt")) != run_attempt
        or receipt.get("return_code") != 0
        or receipt.get("timed_out") is not False
    ):
        raise VerificationError("result receipt does not describe a successful matching run")
    expected_size = receipt.get("result_bytes")
    expected_sha = receipt.get("result_sha256")
    if (
        not isinstance(expected_size, int)
        or not 1 <= expected_size <= MAX_PACKAGE_BYTES
        or not isinstance(expected_sha, str)
        or not SHA256_RE.fullmatch(expected_sha)
    ):
        raise VerificationError("result receipt package binding is invalid")
    chunks = validate_chunk_rows(receipt)

    with tempfile.TemporaryDirectory(prefix="verify-public-fetch-") as temporary:
        package_path = Path(temporary) / "result.tar.gz"
        package_digest = hashlib.sha256()
        package_size = 0
        with package_path.open("wb") as output:
            for row in chunks:
                chunk = fetch_contents_bytes(
                    repository,
                    f"{prefix}/{row['name']}",
                    result_branch,
                    token,
                )
                if len(chunk) != row["bytes"] or hashlib.sha256(chunk).hexdigest() != row["sha256"]:
                    raise VerificationError("result chunk bytes do not match the receipt")
                package_size += len(chunk)
                if package_size > MAX_PACKAGE_BYTES:
                    raise VerificationError("reconstructed result package exceeds the cap")
                package_digest.update(chunk)
                output.write(chunk)
        if package_size != expected_size or package_digest.hexdigest() != expected_sha:
            raise VerificationError("reconstructed result package does not match the receipt")
        verified_files, worker_receipt_sha = verify_archive(package_path, manifest_rows)

    certificate = {
        "schema": 1,
        "capsule_id": capsule_id,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "operation": "public_https_fetch_independent_verification",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "relay_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "worker_receipt_sha256": worker_receipt_sha,
        "result_bytes": package_size,
        "result_sha256": expected_sha,
        "chunk_count": len(chunks),
        "files": verified_files,
        "verdict": "VERIFIED",
    }
    put_file(
        repository,
        f"{prefix}/verification.json",
        result_branch,
        token,
        (json.dumps(certificate, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        "Store independent public fetch verification",
    )
    print("Private public-fetch verification stored.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"Private result verification failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
