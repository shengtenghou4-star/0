from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "s34_artifacts"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {
        "role": "training",
        "filename": "AML310_moving.tar.gz",
        "url": "https://osf.io/download/evhrg/",
        "bytes": 348_444_164,
        "sha256": "144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a",
    },
    {
        "role": "development",
        "filename": "AML310_transition.tar.gz",
        "url": "https://osf.io/download/4u8h2/",
        "bytes": 486_820_852,
        "sha256": "c524561ae35c09e65b9c0cd4b8ac84df01bf40cbf067c036a3e2a4a5b59fe63e",
    },
]
FORBIDDEN = {"AML32_chip.tar.gz"}
EXPECTED_TOTAL = 835_265_016


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_contract() -> None:
    names = [str(t["filename"]) for t in TARGETS]
    if [t["role"] for t in TARGETS] != ["training", "development"]:
        raise RuntimeError("frozen role order changed")
    if len(set(names)) != 2:
        raise RuntimeError("duplicate target name")
    if FORBIDDEN.intersection(names):
        raise RuntimeError("sealed holdout leaked into executable targets")
    if sum(int(t["bytes"]) for t in TARGETS) != EXPECTED_TOTAL:
        raise RuntimeError("frozen total byte count changed")


def acquire(target: dict[str, Any]) -> dict[str, Any]:
    dst = RAW / str(target["filename"])
    if dst.exists():
        dst.unlink()
    cmd = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "8", "--retry-delay", "5", "--retry-all-errors",
        "--connect-timeout", "30", "--max-time", "7200",
        "--user-agent", "BIO-001-S34-OSF/1.0",
        "--output", str(dst), str(target["url"]),
    ]
    subprocess.run(cmd, check=True)
    observed_bytes = dst.stat().st_size
    observed_sha = digest(dst)
    if observed_bytes != int(target["bytes"]):
        raise RuntimeError(
            f"{dst.name}: byte mismatch expected={target['bytes']} observed={observed_bytes}"
        )
    if observed_sha != str(target["sha256"]):
        raise RuntimeError(
            f"{dst.name}: sha256 mismatch expected={target['sha256']} observed={observed_sha}"
        )
    return {
        "role": target["role"],
        "filename": dst.name,
        "bytes": observed_bytes,
        "sha256": observed_sha,
        "status": "VERIFIED",
    }


def inspect_archive(target: dict[str, Any]) -> dict[str, Any]:
    path = RAW / str(target["filename"])
    with tarfile.open(path, "r:gz") as tf:
        members = tf.getmembers()
        files = [m for m in members if m.isfile()]
        dataset_lists = [
            m for m in files
            if m.name.lower().endswith("_datasets.txt")
            or ("dataset" in m.name.lower() and m.name.lower().endswith(".txt"))
        ]
        if not dataset_lists:
            raise RuntimeError(f"{path.name}: no official dataset-list text")
        receipts: list[dict[str, Any]] = []
        for member in dataset_lists:
            handle = tf.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot open {member.name}")
            content = handle.read()
            active = [
                line.split("#", 1)[0].strip()
                for line in content.decode("utf-8", errors="replace").splitlines()
                if line.split("#", 1)[0].strip()
            ]
            receipts.append({
                "member": member.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "active_line_count": len(active),
                "active_lines": active,
            })
        return {
            "role": target["role"],
            "filename": path.name,
            "member_count": len(members),
            "file_count": len(files),
            "dataset_lists": receipts,
            "archive_readable": True,
        }


def main() -> None:
    validate_contract()
    source = [acquire(t) for t in TARGETS]
    observed_total = sum(int(x["bytes"]) for x in source)
    if observed_total != EXPECTED_TOTAL:
        raise RuntimeError(f"verified total mismatch: {observed_total}")
    inventory = [inspect_archive(t) for t in TARGETS]
    write_json(OUT / "source_manifest.json", {
        "schema": "bio-001-s34-osf-source-manifest-v1",
        "osf_node": "dpr3h",
        "base_state_seq": 33,
        "targets": source,
        "expected_total_bytes": EXPECTED_TOTAL,
        "observed_total_bytes": observed_total,
        "sealed_holdout_requested": False,
        "sealed_holdout_downloaded": False,
        "CeRSI_v2_delta": 0.0,
    })
    write_json(OUT / "archive_inventory.json", {
        "schema": "bio-001-s34-osf-archive-inventory-v1",
        "archives": inventory,
        "sealed_holdout_requested": False,
    })
    write_json(OUT / "holdout_nonaccess_receipt.json", {
        "schema": "bio-001-s34-holdout-nonaccess-v1",
        "forbidden_filenames": sorted(FORBIDDEN),
        "executable_target_filenames": [t["filename"] for t in TARGETS],
        "intersection": sorted(FORBIDDEN.intersection({str(t["filename"]) for t in TARGETS})),
        "sealed_holdout_requested": False,
        "sealed_holdout_downloaded": False,
        "sealed_holdout_listed_or_opened": False,
    })
    print(json.dumps({
        "status": "PASS_VERIFIED_TRAIN_DEV_BYTES",
        "verified_bytes": observed_total,
        "training_sha256": source[0]["sha256"],
        "development_sha256": source[1]["sha256"],
        "sealed_holdout_requested": False,
    }, indent=2))


if __name__ == "__main__":
    main()
