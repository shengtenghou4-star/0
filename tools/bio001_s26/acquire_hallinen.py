from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import requests

FROZEN_ALIASES = {
    "AML310_moving": {"train": ["A", "B"], "dev": ["C"], "holdout": ["D"]},
    "AML32_moving": {"train": ["A", "B", "C", "D"], "dev": ["E"], "holdout": ["F", "G"]},
}


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(session: requests.Session, url: str) -> dict:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def iter_pages(session: requests.Session, url: str):
    while url:
        payload = get_json(session, url)
        yield from payload.get("data", [])
        url = payload.get("links", {}).get("next")


def discover(session: requests.Session) -> list[dict]:
    inventory = []
    queue = ["https://api.osf.io/v2/nodes/dpr3h/files/osfstorage/"]
    seen = set()
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        for item in iter_pages(session, url):
            attrs = item.get("attributes", {})
            record = {
                "id": item.get("id"),
                "name": attrs.get("name", ""),
                "kind": attrs.get("kind"),
                "path": attrs.get("materialized_path"),
                "size": attrs.get("size"),
                "download": item.get("links", {}).get("download"),
            }
            inventory.append(record)
            if record["kind"] == "folder":
                related = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
                if related:
                    queue.append(related)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("train-dev", "holdout"), required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    raw = root / "raw"
    extracted = root / "extracted"
    results = root / "results"
    for directory in (raw, extracted, results):
        directory.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "BIO-001-S26-Hallinen/1.0"
    inventory = discover(session)
    (results / "osf_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    targets = [
        record
        for record in inventory
        if record["kind"] == "file"
        and any(token in record["name"].lower() for token in ("aml310_moving", "aml32_moving"))
        and record["name"].lower().endswith((".tar.gz", ".tgz"))
    ]
    if len(targets) != 2:
        raise RuntimeError(f"expected exactly two moving archives, found {targets}")

    archive_receipts = []
    for record in sorted(targets, key=lambda item: item["name"]):
        destination = raw / record["name"]
        digest = hashlib.sha256()
        with session.get(record["download"], stream=True, timeout=300) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
        archive_receipts.append(
            {
                "name": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": digest.hexdigest(),
                "osf_file_id": record["id"],
            }
        )
    (results / "archive_receipts.json").write_text(json.dumps(archive_receipts, indent=2), encoding="utf-8")

    split_receipt = {}
    extracted_identifiers = []
    for archive in sorted(raw.iterdir()):
        condition = "AML310_moving" if "aml310_moving" in archive.name.lower() else "AML32_moving"
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            lists = [member for member in members if member.isfile() and member.name.endswith("_datasets.txt")]
            if len(lists) != 1:
                raise RuntimeError(f"{condition}: expected one dataset list, found {len(lists)}")
            lines = tar.extractfile(lists[0]).read().decode("utf-8").splitlines()
            identifiers = sorted(
                {
                    line.split("#", 1)[0].strip().split()[0]
                    for line in lines
                    if line.split("#", 1)[0].strip()
                }
            )
            alias_map = {chr(65 + index): identifier for index, identifier in enumerate(identifiers)}
            required = sum(FROZEN_ALIASES[condition].values(), [])
            if any(alias not in alias_map for alias in required):
                raise RuntimeError(f"{condition}: insufficient recording list {identifiers}")
            split = {
                role: [alias_map[alias] for alias in aliases]
                for role, aliases in FROZEN_ALIASES[condition].items()
            }
            split_receipt[condition] = {
                "identifiers_lexicographic": identifiers,
                "alias_map": alias_map,
                "split": split,
            }
            selected = set(split["holdout"] if args.mode == "holdout" else split["train"] + split["dev"])
            extracted_identifiers.extend(sorted(selected))
            for member in members:
                if member.isfile() and (member == lists[0] or any(identifier in member.name for identifier in selected)):
                    tar.extract(member, extracted)
    (results / "split_receipt.json").write_text(json.dumps(split_receipt, indent=2), encoding="utf-8")

    forbidden_roles = ("train", "dev") if args.mode == "holdout" else ("holdout",)
    forbidden = [
        identifier
        for receipt in split_receipt.values()
        for role in forbidden_roles
        for identifier in receipt["split"][role]
    ]
    extracted_paths = [str(path) for path in extracted.rglob("*")]
    leaked = [identifier for identifier in forbidden if any(identifier in path for path in extracted_paths)]
    if leaked:
        raise RuntimeError(f"forbidden recording leak: {leaked}")
    custody = {
        "mode": args.mode,
        "extracted_identifiers": sorted(extracted_identifiers),
        "forbidden_identifiers": sorted(forbidden),
        "leaked_identifiers": leaked,
    }
    (results / "custody_receipt.json").write_text(json.dumps(custody, indent=2), encoding="utf-8")
    print(json.dumps({"archive_receipts": archive_receipts, "split_receipt": split_receipt, "custody": custody}, indent=2))


if __name__ == "__main__":
    main()
