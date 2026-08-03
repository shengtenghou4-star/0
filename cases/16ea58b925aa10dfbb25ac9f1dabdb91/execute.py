from __future__ import annotations

import traceback

import requests

import run
import transport


def preloaded_acquire(role: str, name: str, file_id: int):
    if name in run.FORBIDDEN or name not in run.ALLOWED_NAMES or file_id not in run.ALLOWED_IDS:
        raise RuntimeError(f"preloaded acquisition rejected: {name}/{file_id}")
    path = run.RAW / name
    if not path.exists() or path.read_bytes()[:8] != run.SIG:
        raise RuntimeError(f"preloaded HDF5 missing or invalid: {name}")
    return path


def capture_api_metadata() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "BIO-001-S26-metadata-probe/1.0"})
    endpoints = [
        "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.w9ghx3g4v",
        "https://datadryad.org/api/v2/versions/436401",
        "https://datadryad.org/api/v2/versions/436401/files",
        "https://datadryad.org/api/v2/files/4707115",
    ]
    records = []
    for endpoint in endpoints:
        response = session.get(endpoint, timeout=120, allow_redirects=True)
        record = {
            "url": endpoint,
            "status": response.status_code,
            "final_url": response.url,
            "content_type": response.headers.get("content-type"),
            "headers": {
                key: value for key, value in response.headers.items()
                if key.lower() in {"content-length", "location", "etag", "last-modified", "link"}
            },
        }
        try:
            record["json"] = response.json()
        except Exception:
            record["body_prefix"] = response.text[:5000]
        records.append(record)
    run.dump("api_metadata_probe.json", {
        "schema": "bio-001-s26-dryad-api-metadata-probe-v1",
        "holdout_requested": False,
        "records": records,
    })


if __name__ == "__main__":
    try:
        capture_api_metadata()
        transport.download_all()
        run.acquire = preloaded_acquire
        run.main()
    except Exception as exc:
        run.dump("failure_receipt.json", {
            "type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "holdout_requested": False,
        })
        raise
