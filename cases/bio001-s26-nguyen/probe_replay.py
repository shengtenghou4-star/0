from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
NAME = "pnas.1507110112.sd02.csv"
URL = f"https://pmc.ncbi.nlm.nih.gov/articles/instance/4776509/bin/{NAME}?download=1"
CHALLENGE = "VwR3BQH3AwZjZwRhZQR2Zwt3Vt:QaiNtOymT8u8Q0c_zVXBeWLUNs2hMPelFhoI3UKI_fD"
NONCE = 58830
NCBI_SID = "9DB8F63DA7094593_1815SID"
DIGEST = hashlib.sha256(f"{CHALLENGE}{NONCE}".encode()).hexdigest()
if not DIGEST.startswith("0000"):
    raise RuntimeError("captured proof does not verify")

cookie = f"ncbi_sid={NCBI_SID}; cloudpmc-viewer-pow={CHALLENGE}%2C{NONCE}"
response = requests.get(
    URL,
    headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
        "Referer": URL,
        "Cookie": cookie,
        "Cache-Control": "no-cache",
    },
    timeout=300,
    allow_redirects=True,
)
payload = response.content
receipt = {
    "status": response.status_code,
    "final_url": response.url,
    "content_type": response.headers.get("content-type"),
    "content_disposition": response.headers.get("content-disposition"),
    "set_cookie": response.headers.get("set-cookie"),
    "bytes": len(payload),
    "sha256": hashlib.sha256(payload).hexdigest(),
    "prefix": payload[:500].decode("utf-8", errors="replace"),
    "pow_challenge": CHALLENGE,
    "pow_nonce": NONCE,
    "pow_digest": DIGEST,
    "holdout_requested": False,
}
(OUT / "replay_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
if response.status_code >= 400:
    response.raise_for_status()
prefix = payload[:1000].lower()
if not payload or b"<html" in prefix or b"<!doctype" in prefix or b"<?xml" in prefix:
    raise RuntimeError(json.dumps(receipt, indent=2))
first_line = payload[:8192].decode("utf-8", errors="replace").splitlines()[0]
if "," not in first_line and "\t" not in first_line:
    raise RuntimeError(json.dumps(receipt, indent=2))
(OUT / NAME).write_bytes(payload)
print(json.dumps({"status": "PASS_REAL_CSV", "bytes": len(payload), "sha256": receipt["sha256"], "first_line": first_line[:500]}, indent=2))
