from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

import run

DATASET_PAGE = "https://datadryad.org/dataset/doi%3A10.5061/dryad.w9ghx3g4v"
COMMON_CANDIDATES = [
    "https://datadryad.org/stash/downloads/file_stream/{file_id}",
    "https://datadryad.org/stash/downloads/file_stream/{file_id}?filename={name}",
]


def safe_text(response: requests.Response, limit: int = 2000) -> str:
    try:
        return response.text[:limit]
    except Exception:
        return "<binary-or-undecodable>"


def page_candidates(page_text: str, name: str, file_id: int) -> list[str]:
    decoded = html.unescape(page_text)
    links = re.findall(r'(?:href|data-url|download-url)=["\']([^"\']+)["\']', decoded, flags=re.I)
    selected = []
    for link in links:
        if str(file_id) in link or name in link:
            selected.append(urljoin(DATASET_PAGE, link))
    # Some frontend payloads escape slash characters inside JSON.
    normalized = decoded.replace("\\/", "/")
    for match in re.findall(r'https?://[^"\'<>\s]+', normalized):
        if str(file_id) in match or name in match:
            selected.append(match)
    for pattern in COMMON_CANDIDATES:
        selected.append(pattern.format(file_id=file_id, name=name))
    # Stable de-duplication.
    return list(dict.fromkeys(selected))


def download_one(session: requests.Session, role: str, name: str, file_id: int, page_text: str) -> dict:
    if name in run.FORBIDDEN or name not in run.ALLOWED_NAMES or file_id not in run.ALLOWED_IDS:
        raise RuntimeError(f"transport allowlist rejection: {name}/{file_id}")
    destination = run.RAW / name
    destination.unlink(missing_ok=True)
    attempts = []

    # Public metadata can reveal the active web-download URL even when the API
    # binary endpoint requires an API bearer token.
    metadata_url = f"https://datadryad.org/api/v2/files/{file_id}"
    metadata = session.get(metadata_url, timeout=60, allow_redirects=True)
    attempts.append({
        "url": metadata_url,
        "status": metadata.status_code,
        "content_type": metadata.headers.get("content-type"),
        "location": metadata.headers.get("location"),
        "body_prefix": safe_text(metadata),
    })
    candidates = page_candidates(page_text, name, file_id)
    if metadata.ok:
        try:
            payload = metadata.json()
            for key in ("downloadURL", "downloadUrl", "download_url", "url"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    candidates.insert(0, urljoin(DATASET_PAGE, value))
            links = payload.get("_links", {})
            if isinstance(links, dict):
                for value in links.values():
                    if isinstance(value, dict) and isinstance(value.get("href"), str):
                        href = value["href"]
                        if "download" in href or str(file_id) in href:
                            candidates.insert(0, urljoin(DATASET_PAGE, href))
        except Exception:
            pass

    headers = {
        "Referer": DATASET_PAGE,
        "Accept": "application/octet-stream,*/*;q=0.9",
        "User-Agent": "Mozilla/5.0 BIO-001-S26-public-data-validation/1.0",
    }
    for url in list(dict.fromkeys(candidates)):
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
                probe = response.raw.read(8, decode_content=True)
                attempts.append({
                    "url": url,
                    "final_url": response.url,
                    "status": response.status_code,
                    "content_type": response.headers.get("content-type"),
                    "content_length": response.headers.get("content-length"),
                    "location": response.headers.get("location"),
                    "prefix_hex": probe.hex(),
                })
                if response.status_code >= 400 or probe != run.SIG:
                    continue
                with destination.open("wb") as output:
                    output.write(probe)
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            output.write(chunk)
                if destination.read_bytes()[:8] != run.SIG:
                    destination.unlink(missing_ok=True)
                    continue
                return {
                    "role": role,
                    "name": name,
                    "file_id": file_id,
                    "selected_url": url,
                    "final_url": response.url,
                    "bytes": destination.stat().st_size,
                    "sha256": run.sha(destination),
                    "attempts": attempts,
                }
        except Exception as exc:
            attempts.append({"url": url, "exception": type(exc).__name__, "message": str(exc)})
    raise RuntimeError(json.dumps({"failed_name": name, "file_id": file_id, "attempts": attempts}, indent=2))


def download_all() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 BIO-001-S26-public-data-validation/1.0"})
    page = session.get(DATASET_PAGE, timeout=120, allow_redirects=True)
    page.raise_for_status()
    page_text = page.text
    transport = {
        "dataset_page": DATASET_PAGE,
        "dataset_page_status": page.status_code,
        "dataset_page_final_url": page.url,
        "session_cookie_names": sorted(session.cookies.get_dict().keys()),
        "holdout_requested": False,
        "files": [],
    }
    for role, name, file_id in run.FILES:
        transport["files"].append(download_one(session, role, name, file_id, page_text))
    run.dump("transport_receipt.json", transport)


if __name__ == "__main__":
    download_all()
