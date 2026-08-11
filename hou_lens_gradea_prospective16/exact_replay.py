#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import requests

MANIFEST_SHA = "2dd8ac0ca209693207451319ec1a7df92466582ebc295d601b6358faf8d70ad0"
RESULT_CERT_SHA = "7903eccd7c1861ae9bd5f908225fd91da6aca077c8aa94a1961ba5590835e5a5"
PIPELINE_BLOB = "f72edeab83732ef1a05672d8d8e880dcb23c5e6d"
IMAGING_BLOB = "583b5929c05f46d671228cbccc52af5709a2caa0"
SOURCE_ASSET_SHA = "bc60c052791683f022754c2d70b5c0a0d21106baffa662a87962c2aaa068bd75"
TAP = "https://archive.eso.org/tap_obs/sync"
BANDS = ("u", "g", "r", "i1", "i2")
TOP = 20
PRIVATE_IN = pathlib.Path("private_input")
PRIVATE_OUT = pathlib.Path("private_output")
PIPELINE = pathlib.Path("hou_lens_p495_historical_south/source12_exact/run_selected_target_pipeline.py")
IMAGING = pathlib.Path("hou_lens_p495_historical_south/source12_exact/kids_imaging.py")


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def blob_sha(path: pathlib.Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fail(message: str) -> None:
    raise RuntimeError(message)


def tap_query(tile: str) -> str:
    return f"""SELECT TOP {TOP}
  obs_publisher_did,
  obs_id,
  target_name,
  dataproduct_type,
  t_min,
  t_max,
  filter
FROM ivoa.ObsCore
WHERE target_name='{tile}'
  AND dataproduct_type='image'
  AND filter IN ('u_SDSS','g_SDSS','r_SDSS','i_SDSS')
ORDER BY filter, t_min, obs_publisher_did
"""


def resolve_tile(session: requests.Session, tile: str) -> tuple[dict[str, str], dict]:
    query = tap_query(tile)
    response = session.post(TAP, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": query}, timeout=180)
    raw = response.content
    if response.status_code != 200:
        fail("TAP non-200")
    payload = json.loads(raw)
    names = [str(item.get("name", "")) for item in payload.get("metadata", [])]
    expected = ["obs_publisher_did", "obs_id", "target_name", "dataproduct_type", "t_min", "t_max", "filter"]
    if names != expected:
        fail("TAP column drift")
    data = payload.get("data", [])
    if not data or len(data) >= TOP:
        fail("TAP zero/truncation ambiguity")
    rows = [dict(zip(names, row)) for row in data]
    if any(row["target_name"] != tile or row["dataproduct_type"] != "image" for row in rows):
        fail("TAP scope drift")
    by = {f: [row for row in rows if row["filter"] == f] for f in ("u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS")}
    if any(len(by[f]) != 1 for f in ("u_SDSS", "g_SDSS", "r_SDSS")) or len(by["i_SDSS"]) != 2:
        fail("deterministic product-count gate failed")
    irows = sorted(by["i_SDSS"], key=lambda row: (float(row["t_min"]), float(row["t_max"]), str(row["obs_id"]), str(row["obs_publisher_did"])))
    selected = {
        "u": by["u_SDSS"][0]["obs_publisher_did"],
        "g": by["g_SDSS"][0]["obs_publisher_did"],
        "r": by["r_SDSS"][0]["obs_publisher_did"],
        "i1": irows[0]["obs_publisher_did"],
        "i2": irows[1]["obs_publisher_did"],
    }
    if any(not str(value).startswith("ivo://eso.org/ID?ADP.") for value in selected.values()):
        fail("invalid ESO product ID")
    return selected, {
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
        "row_count": len(rows),
    }


def main() -> None:
    manifest_path = PRIVATE_IN / "GRADEA_PROSPECTIVE16_EXACT_REPLAY_MANIFEST.json"
    cert_path = PRIVATE_IN / "result-recipient-cert.pem"
    if sha(manifest_path) != MANIFEST_SHA:
        fail("private manifest SHA gate failed")
    if sha(cert_path) != RESULT_CERT_SHA:
        fail("result cert SHA gate failed")
    if blob_sha(PIPELINE) != PIPELINE_BLOB or blob_sha(IMAGING) != IMAGING_BLOB:
        fail("historical source blob gate failed")

    manifest = json.loads(manifest_path.read_text())
    objects = manifest["objects"]
    if manifest.get("source_asset_sha256") != SOURCE_ASSET_SHA:
        fail("source asset SHA gate failed")
    if len(objects) != 64 or manifest.get("group_count") != 16 or manifest.get("positive_count") != 16 or manifest.get("control_count") != 48:
        fail("manifest structural gate failed")
    if manifest.get("scores_included") is not False:
        fail("score exclusion gate failed")
    groups = sorted({obj["group_id"] for obj in objects})
    tiles = sorted({obj["tile_id"] for obj in objects})
    if len(groups) != 16 or len(tiles) != 16:
        fail("group/tile count gate failed")
    for group_id in groups:
        members = [obj for obj in objects if obj["group_id"] == group_id]
        if len(members) != 4 or sum(obj["role"] == "positive" for obj in members) != 1 or sum(obj["role"] == "control" for obj in members) != 3 or len({obj["tile_id"] for obj in members}) != 1:
            fail("group structure gate failed")

    PRIVATE_OUT.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "HOU-LENS-GradeA-Prospective16-exact-replay/1.0"
    products: dict[str, dict[str, str]] = {}
    tap_receipts = []
    for index, tile in enumerate(tiles, 1):
        products[tile], receipt = resolve_tile(session, tile)
        tap_receipts.append({"tile_index": index, "receipt": receipt, "products": products[tile]})
        print(f"tile {index:02d}/16 product mapping closed", flush=True)
    (PRIVATE_OUT / "tile_product_receipts.json").write_bytes(canonical(tap_receipts))

    plan_rows = []
    for obj in objects:
        plan_rows.append({
            "target_id": obj["target_id"],
            "ra_deg": obj["ra_deg"],
            "dec_deg": obj["dec_deg"],
            "tile_id": obj["tile_id"],
            "benchmark_tier": "grade_a_candidate" if obj["role"] == "positive" else "matched_control",
            "split": "development",
            "science_dataset_ids": products[obj["tile_id"]],
        })
    replay_plan = {
        "schema_version": 1,
        "targets": plan_rows,
        "split": "development",
        "holdout_targets_included": 0,
        "blind_search_performed": False,
        "unknown_targets_queried": False,
    }
    replay_plan_path = PRIVATE_OUT / "resolved_replay_plan.json"
    replay_plan_path.write_bytes(canonical(replay_plan))

    results = []
    for index, obj in enumerate(objects, 1):
        output_dir = PRIVATE_OUT / f"o{index:03d}"
        private_log = PRIVATE_OUT / f"o{index:03d}.pipeline.log"
        with private_log.open("wb") as log_handle:
            process = subprocess.run([
                sys.executable, str(PIPELINE),
                "--targets", str(replay_plan_path),
                "--target-id", obj["target_id"],
                "--output", str(output_dir),
                "--radius", "0.01",
                "--attempts", "5",
                "--timeout", "240",
            ], stdout=log_handle, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            fail(f"historical pipeline technical failure at opaque object index {index}")
        receipt_path = output_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        raw_sha = receipt.get("output_sha256", {}).get("raw_scoring_cube")
        normalized_sha = receipt.get("output_sha256", {}).get("normalized_model_cube")
        raw_path = output_dir / "psf_matched_ugri_scoring_cube.npz"
        normalized_path = output_dir / "normalized_ugri_model_cube.npz"
        checks = {
            "receipt_pass": receipt.get("status") == "PASS",
            "five_cutouts": receipt.get("cutout_count") == 5,
            "raw_shape": receipt.get("raw_cube_shape") == [4, 360, 360],
            "normalized_shape": receipt.get("model_cube_shape") == [4, 360, 360],
            "raw_file_sha_matches_receipt": raw_path.exists() and sha(raw_path) == raw_sha,
            "normalized_file_sha_matches_receipt": normalized_path.exists() and sha(normalized_path) == normalized_sha,
            "historical_raw_sha_exact": raw_sha == obj["expected_raw_cube_sha256"],
            "historical_normalized_sha_exact": normalized_sha == obj["expected_normalized_cube_sha256"],
        }
        if not all(checks.values()):
            fail(f"historical cube SHA gate failed at opaque object index {index}")
        results.append({
            "opaque_index": f"o{index:03d}",
            "target_id": obj["target_id"],
            "authoritative_target_id": obj["authoritative_target_id"],
            "group_id": obj["group_id"],
            "role": obj["role"],
            "source_id": obj["source_id"],
            "tile_id": obj["tile_id"],
            "ra_deg": obj["ra_deg"],
            "dec_deg": obj["dec_deg"],
            "raw_sha256": raw_sha,
            "normalized_sha256": normalized_sha,
            "receipt_sha256": sha(receipt_path),
            "pipeline_log_sha256": sha(private_log),
            "checks": checks,
        })
        print(f"object {index:02d}/64 exact raw+normalized SHA PASS", flush=True)

    files = []
    for path in sorted(item for item in PRIVATE_OUT.rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(PRIVATE_OUT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    aggregate = {
        "schema_version": 1,
        "protocol": "HOU-LENS-P4.9.1-GRADEA-PROSPECTIVE16-EXACT-PAYLOAD-REPLAY",
        "status": "PASS_64_OF_64_RAW_AND_NORMALIZED_HISTORICAL_SHA_REPLAY",
        "manifest_sha256": MANIFEST_SHA,
        "source_asset_sha256": SOURCE_ASSET_SHA,
        "object_count": 64,
        "group_count": 16,
        "positive_count": 16,
        "control_count": 48,
        "tile_count": 16,
        "historical_raw_sha_matches": 64,
        "historical_normalized_sha_matches": 64,
        "results": results,
        "file_set_before_aggregate": files,
        "file_set_sha256_before_aggregate": hashlib.sha256(canonical(files)).hexdigest(),
        "scores_included_in_manifest": False,
        "scores_computed": False,
        "model_code_executed": False,
        "scenario_generation": False,
        "blind_search": False,
        "future_validation_touched": False,
    }
    (PRIVATE_OUT / "aggregate_receipt.json").write_bytes(canonical(aggregate))
    print(json.dumps({
        "status": aggregate["status"],
        "objects": 64,
        "raw_matches": 64,
        "normalized_matches": 64,
        "file_set_sha256": aggregate["file_set_sha256_before_aggregate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
