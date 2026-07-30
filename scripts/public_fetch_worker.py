#!/usr/bin/env python3
"""Networked but secretless worker for validated public HTTPS file acquisition."""

from __future__ import annotations

import hashlib
import http.cookiejar
import ipaddress
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class FetchError(RuntimeError):
    pass


def validate_host_addresses(host: str) -> None:
    try:
        rows = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise FetchError("allowed public host could not be resolved") from None
    addresses = {row[4][0] for row in rows}
    if not addresses:
        raise FetchError("allowed public host resolved to no addresses")
    for value in addresses:
        address = ipaddress.ip_address(value.split("%", 1)[0])
        if not address.is_global:
            raise FetchError("allowed public host resolved to a non-global address")


def validate_url(value: str, allowed_hosts: set[str]) -> str:
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
        detail = json.dumps(
            {
                "url": value,
                "scheme": parsed.scheme,
                "host": host,
                "port": parsed.port,
                "has_userinfo": parsed.username is not None or parsed.password is not None,
                "has_fragment": bool(parsed.fragment),
                "allowed_hosts": sorted(allowed_hosts),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        raise FetchError(f"download URL violates the HTTPS host contract: {detail}")
    validate_host_addresses(host)
    return value


class ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        validate_url(absolute, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


def build_opener(allowed_hosts: set[str]) -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        ValidatingRedirectHandler(allowed_hosts),
    )


def request(url: str, referer: str | None = None) -> urllib.request.Request:
    req = urllib.request.Request(url)
    req.add_header(
        "User-Agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    )
    req.add_header("Accept", "*/*")
    if referer:
        req.add_header("Referer", referer)
    return req


def fetch_landing(
    opener: urllib.request.OpenerDirector,
    landing_url: str,
    allowed_hosts: set[str],
) -> None:
    validate_url(landing_url, allowed_hosts)
    try:
        with opener.open(request(landing_url), timeout=90) as response:
            if getattr(response, "status", 200) >= 400:
                raise FetchError("landing request failed")
            response.read(2 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        raise FetchError(f"landing request failed with HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError):
        raise FetchError("landing request failed") from None


def format_failure(output_name: str, attempts: list[dict[str, Any]]) -> str:
    """Serialize detailed diagnostics for the private result package only."""
    return json.dumps(
        {"output": output_name, "attempts": attempts},
        sort_keys=True,
        separators=(",", ":"),
    )


def download_one(
    opener: urllib.request.OpenerDirector,
    entry: dict[str, Any],
    output_root: Path,
    allowed_hosts: set[str],
    referer: str | None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    magic = bytes.fromhex(entry["magic_hex"])
    for ordinal, url in enumerate(entry["urls"], start=1):
        validate_url(url, allowed_hosts)
        temp_path = output_root / f".{entry['name']}.attempt{ordinal}.tmp"
        digest = hashlib.sha256()
        size = 0
        status = None
        final_url = None
        try:
            with opener.open(request(url, referer), timeout=120) as response:
                status = getattr(response, "status", 200)
                final_url = response.geturl()
                validate_url(final_url, allowed_hosts)
                if status != 200:
                    raise FetchError("candidate returned a non-200 response")
                with temp_path.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        size += len(block)
                        if size > entry["max_bytes"]:
                            raise FetchError("candidate exceeded the per-file byte cap")
                        digest.update(block)
                        handle.write(block)
            if size < entry["min_bytes"]:
                raise FetchError("candidate was smaller than the minimum byte gate")
            if magic:
                with temp_path.open("rb") as handle:
                    prefix = handle.read(len(magic))
                if prefix != magic:
                    raise FetchError("candidate failed the frozen magic-byte gate")
            final_path = output_root / entry["name"]
            os.replace(temp_path, final_path)
            attempts.append(
                {
                    "attempt": ordinal,
                    "url": url,
                    "final_url": final_url,
                    "status": status,
                    "bytes": size,
                    "ok": True,
                }
            )
            return {
                "name": entry["name"],
                "bytes": size,
                "sha256": digest.hexdigest(),
                "selected_url": final_url,
                "attempt": ordinal,
                "attempts": attempts,
            }
        except urllib.error.HTTPError as exc:
            status = exc.code
            final_url = exc.geturl()
            attempts.append(
                {
                    "attempt": ordinal,
                    "url": url,
                    "final_url": final_url,
                    "status": status,
                    "bytes": size,
                    "ok": False,
                    "error_type": "http",
                    "error": f"HTTP {exc.code}: {exc.reason}",
                }
            )
        except (urllib.error.URLError, TimeoutError, FetchError, OSError) as exc:
            attempts.append(
                {
                    "attempt": ordinal,
                    "url": url,
                    "final_url": final_url,
                    "status": status,
                    "bytes": size,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    raise FetchError(format_failure(entry["name"], attempts))


def main() -> int:
    if len(sys.argv) != 3:
        raise FetchError("worker requires manifest and output paths")
    manifest_path = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_hosts = set(manifest["allowed_hosts"])
    for host in allowed_hosts:
        validate_host_addresses(host)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = output_root / "raw"
    raw_root.mkdir()
    receipt_root = output_root / "receipts"
    receipt_root.mkdir()
    opener = build_opener(allowed_hosts)
    landing = manifest.get("landing_url")
    if landing:
        fetch_landing(opener, landing, allowed_hosts)
    rows = []
    total = 0
    for entry in manifest["files"]:
        row = download_one(opener, entry, raw_root, allowed_hosts, landing)
        total += row["bytes"]
        if total > manifest["max_total_bytes"]:
            raise FetchError("downloads exceeded the global byte cap")
        rows.append(row)
    receipt = {
        "schema": 1,
        "capsule_id": manifest["capsule_id"],
        "operation": manifest["operation"],
        "files": rows,
        "total_bytes": total,
    }
    (receipt_root / "acquisition_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Validated public files acquired.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FetchError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Public fetch worker failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
