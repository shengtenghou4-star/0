#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SOURCE = Path("source12/relay_jobs/k4")
PLAN = SOURCE / "sixteen-additional-controls.json"
PIPELINE = SOURCE / "run_matched_control_pipeline.py"
OUT = Path("p495_historical_south_evidence")

WANTED = [
    "KIDSCTRL-00003-02", "KIDSCTRL-00003-03",
    "KIDSCTRL-00004-02", "KIDSCTRL-00004-03",
    "KIDSCTRL-00008-02", "KIDSCTRL-00008-03",
    "KIDSCTRL-00041-02", "KIDSCTRL-00041-03",
    "KIDSCTRL-00113-02", "KIDSCTRL-00113-03",
]
EXPECTED_CUBE = {
    "KIDSCTRL-00003-02": "6b5a6a3c7137986d59b0900941e9169ad04cf2f0f67efbe1bedbae7c0cd54dea",
    "KIDSCTRL-00003-03": "1604b575fd1304560619827e78cec0aae0f494789a18b9a6c99cc9e2b235818e",
    "KIDSCTRL-00004-02": "b393712a0c4019d2788660d99583d7fac7f21081dcdc56036c429536e4ef2bf7",
    "KIDSCTRL-00004-03": "3f498db0333d7920bfdd9f42ec4cf56dd78bb4d5ff94d9d826a861282014f711",
    "KIDSCTRL-00008-02": "e9975598a0a421be05c83c21746a365c0ed0e9c0e019b96491e3a9e20492d4d1",
    "KIDSCTRL-00008-03": "2547afe3d830bc836d2acca90f6340c4146fcb4de54b0a0b8360596376328aa0",
    "KIDSCTRL-00041-02": "c586403d0ba34ff0f190f3b4908210cd98630fb28cfd15ecc6ec7814396a6356",
    "KIDSCTRL-00041-03": "5306b4ab79a7984bb3862d5a384d53c806451b6451cafa0218fb8342af8bddfc",
    "KIDSCTRL-00113-02": "34ee245dcfecbdcd6bdf01fbed45e3f396b0aa6e41135819b8d9a690bdbde21e",
    "KIDSCTRL-00113-03": "186694452f96fd86ded56037aad07b724279f5b44f44840f9b77786ed3c014af",
}
EXPECTED_BLOB = {
    "sixteen-additional-controls.json": "ccee7820e5b08717074009697d53b7418ba9b3f7",
    "run_matched_control_pipeline.py": "09142b8f3ac615a4eed15b7c4712d8fe0667846b",
    "run_selected_target_pipeline.py": "f72edeab83732ef1a05672d8d8e880dcb23c5e6d",
    "kids_imaging.py": "583b5929c05f46d671228cbccc52af5709a2caa0",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    source_audit = {}
    for name, expected in EXPECTED_BLOB.items():
        path = SOURCE / name
        got = git_blob_sha(path)
        source_audit[name] = {"expected_git_blob_sha": expected, "got_git_blob_sha": got, "match": got == expected}
    if not all(x["match"] for x in source_audit.values()):
        raise RuntimeError("historical source blob mismatch: " + json.dumps(source_audit, indent=2))

    plan = json.loads(PLAN.read_text())
    rows = {x["target_id"]: x for x in plan["targets"]}
    if plan.get("source_controls_sha256") != "b40553f6ceb6e918852cba3c5c478e7bcb9dd150325e2595aa6c57818b7e0a56":
        raise RuntimeError("source controls authority drift")
    for tid in WANTED:
        row = rows.get(tid)
        if row is None:
            raise RuntimeError(f"locked target missing: {tid}")
        if row.get("control_rank") not in (2, 3) or row.get("sample") != "matched_control" or row.get("split") != "development":
            raise RuntimeError(f"identity drift: {tid}")
        if set(row.get("science_dataset_ids", {})) != {"u", "g", "r", "i1", "i2"}:
            raise RuntimeError(f"five-band product drift: {tid}")

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    recovered = []
    for idx, tid in enumerate(WANTED, 1):
        target = rows[tid]
        td = OUT / tid
        print(f"=== {idx}/10 exact historical byte replay: {tid} ===", flush=True)
        subprocess.run([
            sys.executable, str(PIPELINE),
            "--targets", str(PLAN), "--target-id", tid,
            "--output", str(td), "--radius", "0.01",
            "--attempts", "5", "--timeout", "240",
        ], check=True)

        rp = td / "receipt.json"
        r = json.loads(rp.read_text())
        products = r.get("products", [])
        product_rows = []
        for p in products:
            band = p["band_label"]
            fp = td / f"{band}_science.fits"
            history = p.get("request_history", [])
            product_rows.append({
                "band": band,
                "dataset_id": p.get("dataset_id"),
                "expected_dataset_id": target["science_dataset_ids"][band],
                "dataset_id_match": p.get("dataset_id") == target["science_dataset_ids"][band],
                "request_url": p.get("request_url"),
                "attempt_count": len(history),
                "terminal_http_status": history[-1].get("status_code") if history else None,
                "receipt_byte_size": p.get("byte_size"),
                "file_byte_size": fp.stat().st_size if fp.exists() else None,
                "receipt_sha256": p.get("sha256"),
                "file_sha256": sha256(fp) if fp.exists() else None,
                "fits_shape": p.get("fits", {}).get("shape"),
                "finite_fraction": p.get("fits", {}).get("finite_fraction"),
            })
        got_cube = r.get("output_sha256", {}).get("raw_scoring_cube")
        raw_cube = td / "psf_matched_ugri_scoring_cube.npz"
        checks = {
            "receipt_pass": r.get("status") == "PASS",
            "stage_exact": r.get("stage") == "kids_dr5_phase4_matched_control_science_pipeline",
            "target_exact": r.get("object", {}).get("target_id") == tid,
            "sample_exact": r.get("object", {}).get("sample") == "matched_control",
            "split_exact": r.get("object", {}).get("split") == "development",
            "matched_positive_exact": r.get("object", {}).get("matched_positive_id") == target["matched_positive_id"],
            "cutout_count_exact": r.get("cutout_count") == 5 and len(product_rows) == 5,
            "raw_cube_shape_exact": r.get("raw_cube_shape") == [4, 360, 360],
            "raw_cube_file_sha_matches_receipt": raw_cube.exists() and sha256(raw_cube) == got_cube,
            "historical_cube_sha_exact": got_cube == EXPECTED_CUBE[tid],
            "all_dataset_ids_exact": all(x["dataset_id_match"] for x in product_rows),
            "all_fits_present": all(x["file_sha256"] is not None for x in product_rows),
            "all_fits_sha_exact": all(x["file_sha256"] == x["receipt_sha256"] for x in product_rows),
            "all_fits_bytes_exact": all(x["file_byte_size"] == x["receipt_byte_size"] for x in product_rows),
            "all_http_200": all(x["terminal_http_status"] == 200 for x in product_rows),
            "all_shapes_360": all(x["fits_shape"] == [360, 360] for x in product_rows),
            "all_finite_gt_099": all(float(x["finite_fraction"] or 0) > 0.99 for x in product_rows),
        }
        if not all(checks.values()):
            raise RuntimeError(json.dumps({"target_id": tid, "expected_cube": EXPECTED_CUBE[tid], "got_cube": got_cube, "checks": checks, "products": product_rows}, indent=2))
        recovered.append({
            "target_id": tid,
            "matched_positive_id": target["matched_positive_id"],
            "control_rank": target["control_rank"],
            "ra_deg": target["ra_deg"],
            "dec_deg": target["dec_deg"],
            "tile_id": target["tile_id"],
            "match_distance": target["match_distance"],
            "expected_raw_cube_sha256": EXPECTED_CUBE[tid],
            "recovered_raw_cube_sha256": got_cube,
            "receipt_sha256": sha256(rp),
            "checks": checks,
            "products": product_rows,
        })
        print(f"PASS {tid} historical cube SHA={got_cube}", flush=True)

    file_rows = []
    for p in sorted(x for x in OUT.rglob("*") if x.is_file()):
        file_rows.append({"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)})
    file_set_sha = hashlib.sha256((json.dumps(file_rows, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    aggregate = {
        "schema_version": 1,
        "protocol": "HOU-LENS-P4.9.5-HISTORICAL-SOUTH-EXACT-BYTE-RECOVERY",
        "created_utc": now(),
        "status": "PASS_EXACT_10_OF_10_HISTORICAL_BYTE_AND_CUBE_CLOSURE",
        "source_repository": "shengtenghou4-star/12",
        "source_ref": "relay/kids-k4-multiband-canary",
        "source_blob_audit": source_audit,
        "source_controls_sha256": plan["source_controls_sha256"],
        "original_acquisition_run": 30023955747,
        "target_count": len(recovered),
        "logical_fits_cutout_count": sum(len(x["products"]) for x in recovered),
        "historical_cube_sha_matches": sum(x["checks"]["historical_cube_sha_exact"] for x in recovered),
        "files_before_aggregate": file_rows,
        "file_set_sha256_before_aggregate": file_set_sha,
        "targets": recovered,
        "scores_computed": False,
        "blind_search_performed": False,
        "unknown_targets_queried": False,
        "future_validation_touched": False,
        "claim_boundary": "Exact byte recovery of ten already-frozen historical South development controls only. No matching, reselection, scoring, unknown target or future-validation access.",
    }
    (OUT / "aggregate_receipt.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    with (OUT / "SHA256SUMS.txt").open("w") as fh:
        for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name != "SHA256SUMS.txt"):
            fh.write(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}\n")
    print(json.dumps({
        "status": aggregate["status"],
        "target_count": aggregate["target_count"],
        "fits_cutouts": aggregate["logical_fits_cutout_count"],
        "historical_cube_matches": aggregate["historical_cube_sha_matches"],
        "file_set_sha256": aggregate["file_set_sha256_before_aggregate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
