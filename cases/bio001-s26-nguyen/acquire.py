from __future__ import annotations

import hashlib
import json
import re
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote, unquote

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
PMCID = "PMC4776509"
ALLOWED = [f"pnas.1507110112.sd{i:02d}.csv" for i in range(2, 11)]
FORBIDDEN = {f"pnas.1507110112.sd{i:02d}.csv" for i in range(11, 14)}
EXCLUDED = {f"pnas.1507110112.sd{i:02d}.csv" for i in range(14, 17)}
USER_AGENT = "BIO-001-S26-Nguyen-frozen-acquisition/2.0"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def list_objects(prefix: str) -> list[dict]:
    records: list[dict] = []
    continuation: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if continuation:
            params["continuation-token"] = continuation
        response = requests.get(
            BUCKET + "/",
            params=params,
            timeout=120,
            headers={"User-Agent": USER_AGENT, "Accept": "application/xml"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}", 1)[0] + "}"
        for item in root.findall(f"{namespace}Contents"):
            key = item.findtext(f"{namespace}Key")
            size = item.findtext(f"{namespace}Size")
            etag = item.findtext(f"{namespace}ETag")
            if key:
                records.append({
                    "key": key,
                    "size": int(size) if size is not None else None,
                    "etag": etag.strip('"') if etag else None,
                })
        truncated = root.findtext(f"{namespace}IsTruncated") == "true"
        continuation = root.findtext(f"{namespace}NextContinuationToken")
        if not truncated:
            break
        if not continuation:
            raise RuntimeError("truncated S3 listing without continuation token")
    return records


def normalized_name(value: str) -> str:
    return unquote(value).split("/")[-1].lower()


def resolve_allowed_keys() -> tuple[dict[str, str], dict]:
    # The current PMC cloud has changed object layouts over time. We therefore
    # resolve the actual keys from world-readable S3 listings rather than
    # guessing a display-name path.
    article_records = list_objects(PMCID)
    metadata_records = list_objects(f"metadata/{PMCID}")
    all_records = article_records + metadata_records

    key_map: dict[str, str] = {}
    for record in article_records:
        basename = normalized_name(record["key"])
        for allowed in ALLOWED:
            if basename == allowed.lower() or allowed.lower() in basename:
                if allowed in key_map and key_map[allowed] != record["key"]:
                    raise RuntimeError(f"ambiguous object keys for {allowed}: {key_map[allowed]} vs {record['key']}")
                key_map[allowed] = record["key"]

    metadata_payloads = []
    # If the data objects are not named directly in the article prefix, PMC's
    # metadata JSON contains media_urls. Reading metadata is allowed; it does
    # not acquire any held-out CSV bytes.
    if len(key_map) < len(ALLOWED):
        for record in metadata_records:
            if not record["key"].lower().endswith(".json"):
                continue
            url = f"{BUCKET}/{quote(record['key'], safe='/')}"
            response = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            payload = response.json()
            metadata_payloads.append({"key": record["key"], "sha256": hashlib.sha256(response.content).hexdigest()})
            candidates = []
            if isinstance(payload, dict):
                for field in ("media_urls", "mediaUrls", "files", "supplementary_files"):
                    value = payload.get(field)
                    if isinstance(value, list):
                        candidates.extend(value)
            for candidate in candidates:
                if isinstance(candidate, str):
                    candidate_values = [candidate]
                elif isinstance(candidate, dict):
                    candidate_values = [
                        candidate.get(field)
                        for field in ("key", "path", "url", "href", "filename", "name")
                        if isinstance(candidate.get(field), str)
                    ]
                else:
                    continue
                for value in candidate_values:
                    basename = normalized_name(value)
                    for allowed in ALLOWED:
                        if basename == allowed.lower() or allowed.lower() in basename:
                            if value.startswith("http://") or value.startswith("https://"):
                                key_map[allowed] = value
                            else:
                                key_map[allowed] = value.lstrip("/")

    unresolved = [name for name in ALLOWED if name not in key_map]
    discovery = {
        "schema": "bio-001-s26-nguyen-pmc-object-discovery-v1",
        "bucket": BUCKET,
        "article_prefix_query": PMCID,
        "metadata_prefix_query": f"metadata/{PMCID}",
        "article_object_count": len(article_records),
        "metadata_object_count": len(metadata_records),
        "article_objects": all_records[:2000],
        "metadata_payloads_read": metadata_payloads,
        "resolved_allowed_keys": key_map,
        "unresolved_allowed": unresolved,
        "forbidden_content_requested": False,
    }
    dump("object_discovery.json", discovery)
    if unresolved:
        raise RuntimeError(f"unable to resolve allowed PMC objects: {unresolved}")
    return key_map, discovery


def object_url(key_or_url: str) -> str:
    if key_or_url.startswith("http://") or key_or_url.startswith("https://"):
        return key_or_url
    return f"{BUCKET}/{quote(key_or_url, safe='/')}"


def acquire(name: str, key_or_url: str) -> dict:
    if name not in ALLOWED or name in FORBIDDEN or name in EXCLUDED:
        raise RuntimeError(f"acquisition rejected by frozen allowlist: {name}")
    # A resolved URL/key is accepted only if it identifies this exact allowed
    # display filename. This prevents a metadata mix-up from opening holdout.
    resolved_text = unquote(key_or_url).lower()
    if name.lower() not in resolved_text:
        raise RuntimeError(f"resolved object does not identify {name}: {key_or_url}")
    if any(forbidden.lower() in resolved_text for forbidden in FORBIDDEN):
        raise RuntimeError(f"forbidden holdout name present in resolved object: {key_or_url}")

    url = object_url(key_or_url)
    target = RAW / name
    with requests.get(
        url,
        stream=True,
        timeout=(30, 300),
        allow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv,application/octet-stream,*/*;q=0.8"},
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        with target.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    prefix = target.read_bytes()[:512]
    if not prefix or b"<html" in prefix.lower() or b"<?xml" in prefix.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError(f"non-CSV response for {name}: {prefix[:100]!r}")
    first_line = target.open("r", encoding="utf-8-sig", errors="replace").readline().rstrip("\r\n")
    if "," not in first_line and "\t" not in first_line:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"no delimited header in {name}: {first_line[:160]!r}")
    return {
        "filename": name,
        "resolved_key_or_url": key_or_url,
        "url": url,
        "final_url": response.url,
        "status": response.status_code,
        "content_type": content_type,
        "content_length_header": response.headers.get("content-length"),
        "bytes": target.stat().st_size,
        "sha256": digest(target, "sha256"),
        "md5": digest(target, "md5"),
        "first_line": first_line[:4000],
    }


def main() -> None:
    if set(ALLOWED) & (FORBIDDEN | EXCLUDED):
        raise RuntimeError("allowlist overlaps forbidden files")
    key_map, _ = resolve_allowed_keys()
    records = [acquire(name, key_map[name]) for name in ALLOWED]
    dump("source_manifest.json", {
        "schema": "bio-001-s26-nguyen-source-manifest-v1",
        "pmcid": PMCID,
        "doi": "10.1073/pnas.1507110112",
        "allowed_downloaded": ALLOWED,
        "forbidden_not_requested": sorted(FORBIDDEN),
        "excluded_not_requested": sorted(EXCLUDED),
        "holdout_requested": False,
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
    })
    dump("acquisition_receipt.json", {
        "status": "PASS_REAL_SOURCE_BYTES_ACQUIRED",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "holdout_requested": False,
        "source_manifest_sha256": digest(OUT / "source_manifest.json", "sha256"),
        "next_step": "Inspect only sd02-sd10 schemas and execute the frozen Nguyen train-development model.",
    })
    print(json.dumps({
        "status": "PASS",
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
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
