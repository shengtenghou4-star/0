from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

PMCID = "PMC4776509"
INSTANCE = "4776509"
ALLOWED = [f"pnas.1507110112.sd{i:02d}.csv" for i in range(2, 11)]
FORBIDDEN = {f"pnas.1507110112.sd{i:02d}.csv" for i in range(11, 14)}
USER_AGENT = "BIO-001-S26-Nguyen-frozen-acquisition/3.0"


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def candidates(name: str) -> list[str]:
    encoded = quote(name, safe="")
    bases = [
        f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/bin/{encoded}",
        f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{INSTANCE}/bin/{encoded}",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{PMCID}/bin/{encoded}",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/instance/{INSTANCE}/bin/{encoded}",
    ]
    return bases + [url + "?download=1" for url in bases]


def valid_csv(path: Path) -> tuple[bool, str]:
    prefix = path.read_bytes()[:1024]
    if not prefix or b"<html" in prefix.lower() or b"<!doctype" in prefix.lower() or b"<?xml" in prefix.lower():
        return False, prefix[:160].decode("utf-8", errors="replace")
    line = path.open("r", encoding="utf-8-sig", errors="replace").readline().rstrip("\r\n")
    return ("," in line or "\t" in line), line[:4000]


def acquire(name: str) -> dict:
    if name not in ALLOWED or name in FORBIDDEN:
        raise RuntimeError(f"allowlist rejection: {name}")
    target = RAW / name
    attempts = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
        "Referer": f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/",
    })
    for url in candidates(name):
        target.unlink(missing_ok=True)
        try:
            with session.get(url, stream=True, timeout=(30, 300), allow_redirects=True) as response:
                attempt = {
                    "url": url,
                    "status": response.status_code,
                    "final_url": response.url,
                    "content_type": response.headers.get("content-type"),
                    "content_length": response.headers.get("content-length"),
                }
                attempts.append(attempt)
                if response.status_code >= 400:
                    continue
                with target.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
            ok, first_line = valid_csv(target)
            attempt["bytes"] = target.stat().st_size
            attempt["prefix_or_first_line"] = first_line[:500]
            if not ok:
                continue
            return {
                "filename": name,
                "selected_url": url,
                "final_url": attempt["final_url"],
                "content_type": attempt["content_type"],
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "md5": digest(target, "md5"),
                "first_line": first_line,
                "attempts": attempts,
            }
        except Exception as exc:
            attempts.append({"url": url, "exception": type(exc).__name__, "message": str(exc)})
    target.unlink(missing_ok=True)
    raise RuntimeError(json.dumps({"filename": name, "attempts": attempts}, indent=2))


def main() -> None:
    if set(ALLOWED) & FORBIDDEN:
        raise RuntimeError("allowlist overlaps holdout")
    records = [acquire(name) for name in ALLOWED]
    dump("source_manifest.json", {
        "schema": "bio-001-s26-nguyen-source-manifest-v2",
        "pmcid": PMCID,
        "doi": "10.1073/pnas.1507110112",
        "allowed_downloaded": ALLOWED,
        "forbidden_not_requested": sorted(FORBIDDEN),
        "holdout_requested": False,
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
    })
    dump("acquisition_receipt.json", {
        "status": "PASS_REAL_SOURCE_BYTES_ACQUIRED",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "holdout_requested": False,
        "source_manifest_sha256": digest(OUT / "source_manifest.json"),
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
        })
        raise
