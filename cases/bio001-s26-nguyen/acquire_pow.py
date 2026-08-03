from __future__ import annotations

import hashlib
import json
import re
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
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/127 Safari/537.36"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def solve_pow(challenge: str, difficulty: int) -> tuple[int, str]:
    prefix = "0" * difficulty
    nonce = 0
    while True:
        value = f"{challenge}{nonce}".encode("utf-8")
        digest = hashlib.sha256(value).hexdigest()
        if digest.startswith(prefix):
            return nonce, digest
        nonce += 1


def is_csv_response(response: requests.Response) -> bool:
    prefix = response.content[:1024].lower()
    if not prefix or b"<html" in prefix or b"<!doctype" in prefix or b"<?xml" in prefix:
        return False
    first_line = response.content[:4096].decode("utf-8", errors="replace").splitlines()[0]
    return "," in first_line or "\t" in first_line


def parse_challenge(text: str) -> tuple[str, int]:
    challenge_match = re.search(r'const POW_CHALLENGE = "([^"]+)"', text)
    difficulty_match = re.search(r'const POW_DIFFICULTY = "([^"]+)"', text)
    if not challenge_match or not difficulty_match:
        raise RuntimeError("PMC response is neither CSV nor recognized PoW page")
    return challenge_match.group(1), int(difficulty_match.group(1))


def acquire_one(session: requests.Session, name: str) -> dict:
    if name not in ALLOWED or name in FORBIDDEN:
        raise RuntimeError(f"allowlist rejection: {name}")
    encoded = quote(name, safe="")
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{INSTANCE}/bin/{encoded}?download=1"
    response = session.get(url, timeout=180, allow_redirects=True)
    steps = [{
        "phase": "initial",
        "status": response.status_code,
        "final_url": response.url,
        "content_type": response.headers.get("content-type"),
        "bytes": len(response.content),
    }]
    response.raise_for_status()

    pow_record = None
    if not is_csv_response(response):
        challenge, difficulty = parse_challenge(response.text)
        nonce, digest = solve_pow(challenge, difficulty)
        cookie_value = f"{challenge},{nonce}"
        # The official JavaScript writes this secure host cookie at path '/'.
        session.cookies.set(
            "cloudpmc-viewer-pow",
            cookie_value,
            domain="pmc.ncbi.nlm.nih.gov",
            path="/",
            secure=True,
        )
        pow_record = {
            "challenge": challenge,
            "difficulty": difficulty,
            "nonce": nonce,
            "verification_sha256": digest,
            "verification_pass": digest.startswith("0" * difficulty),
        }
        response = session.get(url, timeout=300, allow_redirects=True)
        steps.append({
            "phase": "after_official_pow",
            "status": response.status_code,
            "final_url": response.url,
            "content_type": response.headers.get("content-type"),
            "content_disposition": response.headers.get("content-disposition"),
            "bytes": len(response.content),
        })
        response.raise_for_status()

    if not is_csv_response(response):
        raise RuntimeError(json.dumps({
            "filename": name,
            "steps": steps,
            "pow": pow_record,
            "body_prefix": response.text[:1000],
            "cookies": session.cookies.get_dict(),
        }, indent=2))

    target = RAW / name
    target.write_bytes(response.content)
    first_line = target.open("r", encoding="utf-8-sig", errors="replace").readline().rstrip("\r\n")
    return {
        "filename": name,
        "source_url": url,
        "final_url": response.url,
        "content_type": response.headers.get("content-type"),
        "content_disposition": response.headers.get("content-disposition"),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "first_line": first_line[:4000],
        "pow": pow_record,
        "steps": steps,
    }


def main() -> None:
    if set(ALLOWED) & FORBIDDEN:
        raise RuntimeError("allowlist overlaps holdout")
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/csv,application/octet-stream,text/html;q=0.9,*/*;q=0.8",
        "Referer": f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/",
    })
    records = [acquire_one(session, name) for name in ALLOWED]
    downloaded = {record["filename"] for record in records}
    if downloaded != set(ALLOWED) or downloaded & FORBIDDEN:
        raise RuntimeError("downloaded file set violates frozen split")
    total_bytes = sum(record["bytes"] for record in records)
    dump("source_manifest.json", {
        "schema": "bio-001-s26-nguyen-source-manifest-v3",
        "pmcid": PMCID,
        "doi": "10.1073/pnas.1507110112",
        "transport": "official PMC PoW challenge solved exactly as published JavaScript",
        "allowed_downloaded": sorted(downloaded),
        "forbidden_not_requested": sorted(FORBIDDEN),
        "holdout_requested": False,
        "files": records,
        "total_bytes": total_bytes,
    })
    dump("acquisition_receipt.json", {
        "status": "PASS_REAL_SOURCE_BYTES_ACQUIRED",
        "file_count": len(records),
        "total_bytes": total_bytes,
        "holdout_requested": False,
        "source_manifest_sha256": sha256_file(OUT / "source_manifest.json"),
    })
    print(json.dumps({
        "status": "PASS_REAL_SOURCE_BYTES_ACQUIRED",
        "files": len(records),
        "total_bytes": total_bytes,
        "holdout_requested": False,
    }, indent=2))


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
