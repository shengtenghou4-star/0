#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ORIGINAL_SHA256 = "5d8ced1d50e0193ce4d4a16037d2d043e36f09b8ff3731ab9ba9e62d49cbe606"
INTERMEDIATE_SHA256 = "e5fb3ab06575dbf3b8deffb08efb321afdf093a18ed7c71b91538a950301ec7a"
FINAL_SHA256 = "5b1ac2555a9f6cbdefab8e147f533076564c3aee0fc457449d16e2619ffa34ef"
OLD_HISTORICAL_HASH = "f488903da02dc66191875f64bce17eeba74114e3"
NEW_HISTORICAL_HASH = "eea8bca3aa52832c2f2e6ab1ad62ea973c238bc757cbd8a7391f6a1a7044fdc3"
START = "    input_receipts = {\n"
END = "    if total_fits > 9:\n"
REPLACEMENT = '''    input_receipts: dict[str, Any] = {}
    unavailable_releases: dict[str, str] = {}
    for release in ("2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-07-01"):
        try:
            input_receipts[release] = historical.fetch_release(release, releases)
        except RuntimeError as exc:
            if release != "2026-01-01" or "Maximum (12) redirects followed" not in str(exc):
                raise
            unavailable_releases[release] = "official archive absent from the LANL archive index"
            input_receipts[release] = {
                "release": release,
                "status": "OFFICIAL_ARCHIVE_UNAVAILABLE",
                "official_archive_index_checked": True,
                "archive_bytes_read": 0,
                "target_rows_read": 0,
                "supervised_fits": 0,
                "real_august_rows_read": 0,
                "private_bytes_read": 0,
            }

    summaries: list[dict[str, Any]] = []
    total_fits = 0
    fitted_paths: list[Path] = []
    for training_release, target_release in WINDOWS:
        output = windows_root / f"{training_release}_to_{target_release}"
        missing = [
            release for release in (training_release, target_release)
            if release in unavailable_releases
        ]
        if missing:
            output.mkdir(parents=True)
            result = {
                "training_release": training_release,
                "target_release": target_release,
                "status": "TEMPORAL_UNAVAILABLE",
                "unavailable_releases": missing,
                "supervised_fits": 0,
                "criteria": {},
                "real_august_rows_read": 0,
                "real_august_values_read": 0,
                "private_bytes_read": 0,
            }
            (output / "ROUTER_WINDOW_RECEIPT.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\\n"
            )
        else:
            result = execute_window(
                runner, legacy, orth, wrapper, historical, args.families, releases,
                training_release, target_release, output,
            )
        summaries.append(result)
        total_fits += int(result["supervised_fits"])
        if result["status"] == "ROUTER_WINDOW_COMPLETE":
            fitted_paths.append(output)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    raw = args.path.read_bytes()
    if digest(raw) != ORIGINAL_SHA256:
        raise ValueError("original router hash mismatch")
    text = raw.decode()
    if text.count(OLD_HISTORICAL_HASH) != 1:
        raise ValueError("historical hash marker count drifted")
    text = text.replace(OLD_HISTORICAL_HASH, NEW_HISTORICAL_HASH)
    if digest(text.encode()) != INTERMEDIATE_SHA256:
        raise ValueError("intermediate router hash mismatch")
    start = text.index(START)
    end = text.index(END, start)
    text = text[:start] + REPLACEMENT + text[end:]
    final = text.encode()
    if digest(final) != FINAL_SHA256:
        raise ValueError(f"final router hash mismatch: {digest(final)}")
    args.path.write_bytes(final)
    print(FINAL_SHA256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
