from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
NODE = "dpr3h"
API = "https://api.osf.io/v2"
USER_AGENT = "BIO-001-S26-OSF-metadata-only/1.0"


def dump(name: str, payload: Any) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def get_json(session: requests.Session, url: str) -> dict:
    response = session.get(url, timeout=120, allow_redirects=True)
    response.raise_for_status()
    return response.json()


def walk_collection(session: requests.Session, url: str, parent: str = "") -> list[dict]:
    records: list[dict] = []
    next_url: str | None = url
    while next_url:
        payload = get_json(session, next_url)
        for item in payload.get("data", []):
            attrs = item.get("attributes", {})
            links = item.get("links", {})
            kind = attrs.get("kind")
            name = attrs.get("name")
            logical_path = f"{parent}/{name}" if parent else str(name)
            record = {
                "id": item.get("id"),
                "kind": kind,
                "name": name,
                "logical_path": logical_path,
                "size": attrs.get("size"),
                "date_created": attrs.get("date_created"),
                "date_modified": attrs.get("date_modified"),
                "extra": attrs.get("extra"),
                "hashes": attrs.get("extra", {}).get("hashes") if isinstance(attrs.get("extra"), dict) else None,
                "download_url": links.get("download"),
                "related_files": item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href"),
            }
            records.append(record)
            if kind == "folder" and record["related_files"]:
                records.extend(walk_collection(session, record["related_files"], logical_path))
        next_url = payload.get("links", {}).get("next")
    return records


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/vnd.api+json"})
    node = get_json(session, f"{API}/nodes/{NODE}/")
    providers = get_json(session, f"{API}/nodes/{NODE}/files/")
    provider_records = []
    all_files = []
    for provider in providers.get("data", []):
        attrs = provider.get("attributes", {})
        provider_name = attrs.get("name")
        provider_id = provider.get("id")
        related = provider.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
        provider_record = {
            "id": provider_id,
            "name": provider_name,
            "node": provider.get("relationships", {}).get("node", {}).get("data"),
            "files_url": related,
        }
        provider_records.append(provider_record)
        if related:
            for record in walk_collection(session, related):
                record["provider_id"] = provider_id
                record["provider_name"] = provider_name
                all_files.append(record)

    archive_candidates = [
        record for record in all_files
        if record["kind"] == "file"
        and isinstance(record["name"], str)
        and record["name"].lower().endswith((".tar.gz", ".tgz", ".zip", ".txt"))
    ]
    payload = {
        "schema": "bio-001-s26-osf-dpr3h-metadata-v1",
        "node_id": NODE,
        "node_title": node.get("data", {}).get("attributes", {}).get("title"),
        "node_date_modified": node.get("data", {}).get("attributes", {}).get("date_modified"),
        "providers": provider_records,
        "file_count": len([record for record in all_files if record["kind"] == "file"]),
        "folder_count": len([record for record in all_files if record["kind"] == "folder"]),
        "files": all_files,
        "archive_candidates": archive_candidates,
        "content_bytes_downloaded": 0,
    }
    dump("osf_metadata.json", payload)
    digest = hashlib.sha256((OUT / "osf_metadata.json").read_bytes()).hexdigest()
    dump("metadata_receipt.json", {
        "status": "PASS_METADATA_ONLY",
        "file_count": payload["file_count"],
        "folder_count": payload["folder_count"],
        "archive_candidate_count": len(archive_candidates),
        "osf_metadata_sha256": digest,
        "content_bytes_downloaded": 0,
    })
    print(json.dumps({
        "status": "PASS_METADATA_ONLY",
        "title": payload["node_title"],
        "files": payload["file_count"],
        "archive_candidates": [
            {"name": item["name"], "size": item["size"], "id": item["id"]}
            for item in archive_candidates
        ],
        "content_bytes_downloaded": 0,
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
            "content_bytes_downloaded": 0,
        })
        raise
