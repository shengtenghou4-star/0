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

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/127 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4776509/",
})
response = session.get(URL, timeout=120, allow_redirects=True)
body = response.content
(OUT / "preparing_download.html").write_bytes(body)
(OUT / "preparing_download_receipt.json").write_text(json.dumps({
    "url": URL,
    "status": response.status_code,
    "final_url": response.url,
    "headers": dict(response.headers),
    "cookies": session.cookies.get_dict(),
    "bytes": len(body),
    "sha256": hashlib.sha256(body).hexdigest(),
    "holdout_requested": False,
}, indent=2), encoding="utf-8")
print(body.decode("utf-8", errors="replace"))
