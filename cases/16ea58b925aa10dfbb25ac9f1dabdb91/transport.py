from __future__ import annotations

import html
import json
import re
from urllib.parse import quote, urljoin

import requests

import run

DATASET_PAGE = "https://datadryad.org/dataset/doi%3A10.5061/dryad.w9ghx3g4v"
SHARE_TOKEN = "L731GV6Ab6VebP9u4_jPxOkUCbFtk7tTrbLZrgP4BXU"
SHARE_PAGE = f"https://datadryad.org/stash/share/{SHARE_TOKEN}"
DATASET_RESOURCE_ID = 183044
S3_BUCKET = "dryad-assetstore-merritt-west"
COMMON_CANDIDATES = [
    "https://datadryad.org/downloads/file_stream/{file_id}?share=" + SHARE_TOKEN,
    "https://datadryad.org/stash/downloads/file_stream/{file_id}?share=" + SHARE_TOKEN,
    "https://datadryad.org/downloads/file_stream/{file_id}?secret_id=" + SHARE_TOKEN,
    "https://datadryad.org/stash/downloads/file_stream/{file_id}",
]


def direct_object_candidates(name: str) -> list[str]:
    encoded = quote(name, safe="")
    keys = [f"v3/{DATASET_RESOURCE_ID}/data/{encoded}", f"{DATASET_RESOURCE_ID}/data/{encoded}"]
    result = []
    for key in keys:
        result.extend([
            f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com/{key}",
            f"https://{S3_BUCKET}.s3.amazonaws.com/{key}",
            f"https://s3.us-west-2.amazonaws.com/{S3_BUCKET}/{key}",
        ])
    return result


def page_candidates(page_text: str, name: str, file_id: int) -> list[str]:
    decoded = html.unescape(page_text).replace("\\/", "/")
    selected = [pattern.format(file_id=file_id) for pattern in COMMON_CANDIDATES]
    selected.extend(direct_object_candidates(name))
    for link in re.findall(r'(?:href|data-url|download-url)=["\']([^"\']+)["\']', decoded, flags=re.I):
        if str(file_id) in link or name in link:
            selected.append(urljoin(DATASET_PAGE, link))
    for match in re.findall(r'https?://[^"\'<>\s]+', decoded):
        if str(file_id) in match or name in match:
            selected.append(match)
    return list(dict.fromkeys(selected))


def download_one(session: requests.Session, role: str, name: str, file_id: int, page_text: str) -> dict:
    if name in run.FORBIDDEN or name not in run.ALLOWED_NAMES or file_id not in run.ALLOWED_IDS:
        raise RuntimeError(f"transport allowlist rejection: {name}/{file_id}")
    destination = run.RAW / name
    destination.unlink(missing_ok=True)
    metadata_url = f"https://datadryad.org/api/v2/files/{file_id}"
    metadata = session.get(metadata_url, timeout=60)
    metadata.raise_for_status()
    payload = metadata.json()
    expected_size = int(payload["size"])
    expected_digest = str(payload["digest"]).lower()
    if payload.get("path") != name or payload.get("digestType") != "sha-256":
        raise RuntimeError(f"metadata identity mismatch for {name}")
    attempts = [{"url": metadata_url, "status": metadata.status_code,
                 "expected_size": expected_size, "expected_sha256": expected_digest}]
    links = payload.get("_links", {})
    candidates = page_candidates(page_text, name, file_id)
    if isinstance(links, dict):
        for value in links.values():
            if isinstance(value, dict) and isinstance(value.get("href"), str) and "download" in value["href"]:
                candidates.append(urljoin(DATASET_PAGE, value["href"]))
    headers = {
        "Referer": SHARE_PAGE,
        "Accept": "application/octet-stream,*/*;q=0.9",
        "User-Agent": "Mozilla/5.0 BIO-001-S26-public-data-validation/1.0",
    }
    for url in list(dict.fromkeys(candidates)):
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
                probe = response.raw.read(8, decode_content=True)
                attempt = {
                    "url": url, "final_url": response.url, "status": response.status_code,
                    "content_type": response.headers.get("content-type"),
                    "content_length": response.headers.get("content-length"),
                    "location": response.headers.get("location"), "prefix_hex": probe.hex(),
                }
                attempts.append(attempt)
                if response.status_code >= 400 or probe != run.SIG:
                    continue
                with destination.open("wb") as output:
                    output.write(probe)
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            output.write(chunk)
                actual_size = destination.stat().st_size
                actual_digest = run.sha(destination)
                attempt.update(actual_size=actual_size, actual_sha256=actual_digest)
                if actual_size != expected_size or actual_digest != expected_digest:
                    destination.unlink(missing_ok=True)
                    continue
                return {
                    "role": role, "name": name, "file_id": file_id,
                    "selected_url": url, "final_url": response.url,
                    "bytes": actual_size, "sha256": actual_digest,
                    "official_bytes": expected_size, "official_sha256": expected_digest,
                    "official_identity_match": True, "attempts": attempts,
                }
        except Exception as exc:
            attempts.append({"url": url, "exception": type(exc).__name__, "message": str(exc)})
    raise RuntimeError(json.dumps({"failed_name": name, "file_id": file_id,
                                   "official_size": expected_size,
                                   "official_sha256": expected_digest,
                                   "attempts": attempts}, indent=2))


def download_all() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 BIO-001-S26-public-data-validation/1.0"})
    share_response = session.get(SHARE_PAGE, timeout=120, allow_redirects=True)
    share_response.raise_for_status()
    dataset_response = session.get(DATASET_PAGE, timeout=120, allow_redirects=True)
    dataset_response.raise_for_status()
    transport = {
        "dataset_page": DATASET_PAGE,
        "public_share_page": SHARE_PAGE,
        "share_final_url": share_response.url,
        "share_status": share_response.status_code,
        "dataset_resource_id": DATASET_RESOURCE_ID,
        "production_bucket": S3_BUCKET,
        "session_cookie_names": sorted(session.cookies.get_dict().keys()),
        "holdout_requested": False,
        "files": [],
    }
    combined_page = share_response.text + "\n" + dataset_response.text
    for role, name, file_id in run.FILES:
        transport["files"].append(download_one(session, role, name, file_id, combined_page))
    run.dump("transport_receipt.json", transport)


if __name__ == "__main__":
    download_all()
