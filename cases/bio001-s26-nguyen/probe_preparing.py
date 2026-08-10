from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin

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
text = body.decode("utf-8", errors="replace")
assets = []
for raw_src in re.findall(r'(?:src|href)=["\']([^"\']+\.js)["\']', text, flags=re.I):
    asset_url = urljoin(response.url, raw_src)
    asset_response = session.get(asset_url, timeout=120, allow_redirects=True)
    asset_response.raise_for_status()
    filename = asset_url.rsplit("/", 1)[-1]
    path = OUT / filename
    path.write_bytes(asset_response.content)
    assets.append({
        "url": asset_url,
        "final_url": asset_response.url,
        "filename": filename,
        "bytes": len(asset_response.content),
        "sha256": hashlib.sha256(asset_response.content).hexdigest(),
        "content_type": asset_response.headers.get("content-type"),
    })
challenge = re.search(r'const POW_CHALLENGE = "([^"]+)"', text)
difficulty = re.search(r'const POW_DIFFICULTY = "([^"]+)"', text)
(OUT / "preparing_download_receipt.json").write_text(json.dumps({
    "url": URL,
    "status": response.status_code,
    "final_url": response.url,
    "headers": dict(response.headers),
    "cookies": session.cookies.get_dict(),
    "bytes": len(body),
    "sha256": hashlib.sha256(body).hexdigest(),
    "pow_challenge": challenge.group(1) if challenge else None,
    "pow_difficulty": difficulty.group(1) if difficulty else None,
    "assets": assets,
    "holdout_requested": False,
}, indent=2), encoding="utf-8")
print(json.dumps({"challenge": challenge.group(1) if challenge else None,
                  "difficulty": difficulty.group(1) if difficulty else None,
                  "assets": assets, "holdout_requested": False}, indent=2))
