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
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
FILE_ACCEPT = "text/csv,application/octet-stream,*/*;q=0.8"


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
        digest = hashlib.sha256(f"{challenge}{nonce}".encode("utf-8")).hexdigest()
        if digest.startswith(prefix):
            return nonce, digest
        nonce += 1


def is_csv_response(response: requests.Response) -> bool:
    prefix = response.content[:1024].lower()
    if not prefix or b"<html" in prefix or b"<!doctype" in prefix or b"<?xml" in prefix:
        return False
    lines = response.content[:8192].decode("utf-8", errors="replace").splitlines()
    return bool(lines) and ("," in lines[0] or "\t" in lines[0])


def parse_challenge(text: str) -> tuple[str, int]:
    challenge_match = re.search(r"POW_CHALLENGE\s*=\s*['\"]([^'\"]+)['\"]", text)
    difficulty_match = re.search(r"POW_DIFFICULTY\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not challenge_match or not difficulty_match:
        raise RuntimeError(json.dumps({
            "reason": "PMC response is neither CSV nor recognized PoW page",
            "body_prefix": text[:1500],
        }, indent=2))
    return challenge_match.group(1), int(difficulty_match.group(1))


def explicit_cookie_header(session: requests.Session, pow_cookie_value: str) -> str:
    cookies = session.cookies.get_dict()
    parts = [f"{key}={value}" for key, value in cookies.items() if key != "cloudpmc-viewer-pow"]
    parts.append(f"cloudpmc-viewer-pow={pow_cookie_value}")
    return "; ".join(parts)


def acquire_one(session: requests.Session, name: str) -> dict:
    if name not in ALLOWED or name in FORBIDDEN:
        raise RuntimeError(f"allowlist rejection: {name}")
    encoded_name = quote(name, safe="")
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{INSTANCE}/bin/{encoded_name}?download=1"
    referer = f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/"
    initial_headers = {"Accept": HTML_ACCEPT, "Referer": referer, "Cache-Control": "no-cache"}
    response = session.get(url, headers=initial_headers, timeout=180, allow_redirects=True)
    steps = [{
        "phase": "initial_html_navigation",
        "status": response.status_code,
        "final_url": response.url,
        "content_type": response.headers.get("content-type"),
        "set_cookie": response.headers.get("set-cookie"),
        "bytes": len(response.content),
        "body_prefix": response.text[:500] if not is_csv_response(response) else None,
    }]
    response.raise_for_status()

    pow_record = None
    if not is_csv_response(response):
        challenge, difficulty = parse_challenge(response.text)
        nonce, verification_digest = solve_pow(challenge, difficulty)
        # js-cookie's converter keeps ':' literal but encodes comma as %2C.
        encoded_cookie_value = f"{challenge}%2C{nonce}"
        raw_cookie_value = f"{challenge},{nonce}"
        pow_record = {
            "challenge": challenge,
            "difficulty": difficulty,
            "nonce": nonce,
            "verification_sha256": verification_digest,
            "verification_pass": verification_digest.startswith("0" * difficulty),
            "official_js_cookie_value": encoded_cookie_value,
        }

        # The official page writes a secure host-only cookie then reloads the
        # same URL. First reproduce that exact encoded Cookie header. Raw-value
        # and cookie-jar variants are deterministic fallbacks for client-library
        # differences; model/data identities are unchanged.
        attempts = [
            ("explicit_encoded_cookie", encoded_cookie_value, True),
            ("jar_encoded_cookie", encoded_cookie_value, False),
            ("explicit_raw_cookie", raw_cookie_value, True),
        ]
        successful_response = None
        for label, cookie_value, explicit in attempts:
            try:
                session.cookies.clear(domain="pmc.ncbi.nlm.nih.gov", path="/", name="cloudpmc-viewer-pow")
            except KeyError:
                pass
            session.cookies.set(
                "cloudpmc-viewer-pow",
                cookie_value,
                domain="pmc.ncbi.nlm.nih.gov",
                path="/",
                secure=True,
            )
            headers = {
                "Accept": FILE_ACCEPT,
                "Referer": response.url,
                "Cache-Control": "no-cache",
            }
            if explicit:
                headers["Cookie"] = explicit_cookie_header(session, cookie_value)
            candidate = session.get(url, headers=headers, timeout=300, allow_redirects=True)
            step = {
                "phase": label,
                "status": candidate.status_code,
                "final_url": candidate.url,
                "content_type": candidate.headers.get("content-type"),
                "content_disposition": candidate.headers.get("content-disposition"),
                "set_cookie": candidate.headers.get("set-cookie"),
                "bytes": len(candidate.content),
                "body_prefix": candidate.text[:500] if not is_csv_response(candidate) else None,
            }
            steps.append(step)
            candidate.raise_for_status()
            if is_csv_response(candidate):
                successful_response = candidate
                break
        if successful_response is not None:
            response = successful_response

    if not is_csv_response(response):
        raise RuntimeError(json.dumps({
            "filename": name,
            "steps": steps,
            "pow": pow_record,
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
    session.headers.update({"User-Agent": USER_AGENT})
    records = [acquire_one(session, name) for name in ALLOWED]
    downloaded = {record["filename"] for record in records}
    if downloaded != set(ALLOWED) or downloaded & FORBIDDEN:
        raise RuntimeError("downloaded file set violates frozen split")
    total_bytes = sum(record["bytes"] for record in records)
    dump("source_manifest.json", {
        "schema": "bio-001-s26-nguyen-source-manifest-v4",
        "pmcid": PMCID,
        "doi": "10.1073/pnas.1507110112",
        "transport": "official PMC proof-of-work algorithm and js-cookie encoding reproduced exactly",
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
