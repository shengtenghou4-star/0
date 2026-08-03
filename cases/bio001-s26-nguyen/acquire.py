from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

PREFIX = "https://pmc-oa-opendata.s3.amazonaws.com/PMC4776509.1"
ALLOWED = [f"pnas.1507110112.sd{i:02d}.csv" for i in range(2, 11)]
FORBIDDEN = {f"pnas.1507110112.sd{i:02d}.csv" for i in range(11, 14)}
EXCLUDED = {f"pnas.1507110112.sd{i:02d}.csv" for i in range(14, 17)}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def acquire(name: str) -> dict:
    if name not in ALLOWED or name in FORBIDDEN or name in EXCLUDED:
        raise RuntimeError(f"acquisition rejected by frozen allowlist: {name}")
    url = f"{PREFIX}/{name}"
    target = RAW / name
    with requests.get(url, stream=True, timeout=(30, 300), allow_redirects=True,
                      headers={"User-Agent": "BIO-001-S26-Nguyen-frozen-acquisition/1.0"}) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        with target.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    prefix = target.read_bytes()[:256]
    if not prefix or b"<html" in prefix.lower() or b"<?xml" in prefix.lower():
        raise RuntimeError(f"non-CSV response for {name}: {prefix[:80]!r}")
    return {
        "filename": name,
        "url": url,
        "final_url": response.url,
        "status": response.status_code,
        "content_type": content_type,
        "content_length_header": response.headers.get("content-length"),
        "bytes": target.stat().st_size,
        "sha256": digest(target, "sha256"),
        "md5": digest(target, "md5"),
        "first_line": target.open("r", encoding="utf-8-sig", errors="replace").readline().rstrip("\r\n")[:2000],
    }


def main() -> None:
    if set(ALLOWED) & (FORBIDDEN | EXCLUDED):
        raise RuntimeError("allowlist overlaps forbidden files")
    records = [acquire(name) for name in ALLOWED]
    dump("source_manifest.json", {
        "schema": "bio-001-s26-nguyen-source-manifest-v1",
        "pmcid": "PMC4776509",
        "doi": "10.1073/pnas.1507110112",
        "cloud_prefix": PREFIX,
        "allowed_downloaded": ALLOWED,
        "forbidden_not_requested": sorted(FORBIDDEN),
        "excluded_not_requested": sorted(EXCLUDED),
        "holdout_requested": False,
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
    })
    dump("acquisition_receipt.json", {
        "status": "PASS_REAL_SOURCE_BYTES_ACQUIRED",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "holdout_requested": False,
        "source_manifest_sha256": digest(OUT / "source_manifest.json", "sha256"),
        "next_step": "Inspect only sd02-sd10 schemas and execute the frozen Nguyen train-development model.",
    })
    print(json.dumps({"status": "PASS", "files": len(records),
                      "bytes": sum(record["bytes"] for record in records),
                      "holdout_requested": False}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        dump("failure_receipt.json", {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "holdout_requested": False,
            "allowed": ALLOWED,
            "forbidden": sorted(FORBIDDEN),
        })
        raise
