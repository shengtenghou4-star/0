#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time

import requests

MANIFEST_SHA = "ee41c5312b2a6918b0241acbe235970f825e6b5712af392be85dafce9cf524ef"
RESULT_CERT_SHA = "151e539be07262be6d6c4abd0776d791db10b73521b5daf1c408d71b3b3865e1"
PIPELINE_BLOB = "f72edeab83732ef1a05672d8d8e880dcb23c5e6d"
IMAGING_BLOB = "583b5929c05f46d671228cbccc52af5709a2caa0"
TAP = "https://archive.eso.org/tap_obs/sync"
BANDS = ("u", "g", "r", "i1", "i2")
TOP = 20
PRIVATE_IN = pathlib.Path("private_input")
PRIVATE_OUT = pathlib.Path("private_output")
PIPELINE = pathlib.Path("hou_lens_p495_historical_south/source12_exact/run_selected_target_pipeline.py")
IMAGING = pathlib.Path("hou_lens_p495_historical_south/source12_exact/kids_imaging.py")


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blob_sha(path: pathlib.Path) -> str:
    b = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()


def canonical(x) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fail(msg: str):
    raise RuntimeError(msg)


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
    q = tap_query(tile)
    r = session.post(TAP, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": q}, timeout=180)
    raw = r.content
    if r.status_code != 200:
        fail("TAP non-200")
    p = json.loads(raw)
    names = [str(x.get("name", "")) for x in p.get("metadata", [])]
    expected = ["obs_publisher_did", "obs_id", "target_name", "dataproduct_type", "t_min", "t_max", "filter"]
    if names != expected:
        fail("TAP column drift")
    data = p.get("data", [])
    if not data or len(data) >= TOP:
        fail("TAP zero/truncation ambiguity")
    rows = [dict(zip(names, x)) for x in data]
    if any(x["target_name"] != tile or x["dataproduct_type"] != "image" for x in rows):
        fail("TAP scope drift")
    by = {f: [x for x in rows if x["filter"] == f] for f in ("u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS")}
    if any(len(by[f]) != 1 for f in ("u_SDSS", "g_SDSS", "r_SDSS")) or len(by["i_SDSS"]) != 2:
        fail("deterministic product-count gate failed")
    ir = sorted(by["i_SDSS"], key=lambda x: (float(x["t_min"]), float(x["t_max"]), str(x["obs_id"]), str(x["obs_publisher_did"])))
    out = {"u": by["u_SDSS"][0]["obs_publisher_did"], "g": by["g_SDSS"][0]["obs_publisher_did"], "r": by["r_SDSS"][0]["obs_publisher_did"], "i1": ir[0]["obs_publisher_did"], "i2": ir[1]["obs_publisher_did"]}
    if any(not str(v).startswith("ivo://eso.org/ID?ADP.") for v in out.values()):
        fail("invalid ESO product ID")
    return out, {"query_sha256": hashlib.sha256(q.encode()).hexdigest(), "response_sha256": hashlib.sha256(raw).hexdigest(), "response_bytes": len(raw), "row_count": len(rows)}


def main():
    mp = PRIVATE_IN / "P492_MIXED16_EXACT_REPLAY_MANIFEST.json"
    cert = PRIVATE_IN / "result-recipient-cert.pem"
    if sha(mp) != MANIFEST_SHA:
        fail("private manifest SHA gate failed")
    if sha(cert) != RESULT_CERT_SHA:
        fail("result cert SHA gate failed")
    if blob_sha(PIPELINE) != PIPELINE_BLOB or blob_sha(IMAGING) != IMAGING_BLOB:
        fail("historical source blob gate failed")
    manifest = json.loads(mp.read_text())
    objects = manifest["objects"]
    if len(objects) != 64 or manifest.get("group_count") != 16 or manifest.get("positive_count") != 16 or manifest.get("control_count") != 48:
        fail("manifest structural gate failed")
    groups = sorted({x["group_id"] for x in objects})
    tiles = sorted({x["tile_id"] for x in objects})
    if len(groups) != 16 or len(tiles) != 16:
        fail("group/tile count gate failed")
    for gid in groups:
        rr = [x for x in objects if x["group_id"] == gid]
        if len(rr) != 4 or sum(x["role"] == "positive" for x in rr) != 1 or sum(x["role"] == "control" for x in rr) != 3 or len({x["tile_id"] for x in rr}) != 1:
            fail("group structure gate failed")

    PRIVATE_OUT.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "HOU-LENS-P492-Mixed16-exact-replay/1.0"
    products = {}
    tap_receipts = []
    for i, tile in enumerate(tiles, 1):
        products[tile], rec = resolve_tile(session, tile)
        tap_receipts.append({"tile_index": i, "receipt": rec, "products": products[tile]})
        print(f"tile {i:02d}/16 product mapping closed", flush=True)
    (PRIVATE_OUT / "tile_product_receipts.json").write_bytes(canonical(tap_receipts))

    plan_rows = []
    for x in objects:
        plan_rows.append({
            "target_id": x["target_id"],
            "ra_deg": x["ra_deg"],
            "dec_deg": x["dec_deg"],
            "tile_id": x["tile_id"],
            "benchmark_tier": "high_quality_candidate" if x["role"] == "positive" else "matched_control",
            "split": "development",
            "science_dataset_ids": products[x["tile_id"]],
        })
    plan = {"schema_version": 1, "targets": plan_rows, "split": "development", "holdout_targets_included": 0, "blind_search_performed": False, "unknown_targets_queried": False}
    plan_path = PRIVATE_OUT / "resolved_replay_plan.json"
    plan_path.write_bytes(canonical(plan))

    results = []
    for idx, x in enumerate(objects, 1):
        od = PRIVATE_OUT / f"o{idx:03d}"
        log = PRIVATE_OUT / f"o{idx:03d}.pipeline.log"
        with log.open("wb") as lf:
            proc = subprocess.run([
                sys.executable, str(PIPELINE), "--targets", str(plan_path), "--target-id", x["target_id"],
                "--output", str(od), "--radius", "0.01", "--attempts", "5", "--timeout", "240"
            ], stdout=lf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(json.dumps({"opaque_index": idx, "diagnostic": "pipeline_technical_failure", "returncode": proc.returncode}, sort_keys=True), flush=True)
            fail(f"historical pipeline technical failure at opaque object index {idx}")
        rp = od / "receipt.json"
        rec = json.loads(rp.read_text())
        got_raw = rec.get("output_sha256", {}).get("raw_scoring_cube")
        got_norm = rec.get("output_sha256", {}).get("normalized_model_cube")
        rawp = od / "psf_matched_ugri_scoring_cube.npz"
        normp = od / "normalized_ugri_model_cube.npz"
        checks = {
            "receipt_pass": rec.get("status") == "PASS",
            "five_cutouts": rec.get("cutout_count") == 5,
            "raw_shape": rec.get("raw_cube_shape") == [4, 360, 360],
            "normalized_shape": rec.get("model_cube_shape") == [4, 360, 360],
            "raw_file_sha_matches_receipt": rawp.exists() and sha(rawp) == got_raw,
            "normalized_file_sha_matches_receipt": normp.exists() and sha(normp) == got_norm,
            "historical_raw_sha_exact": got_raw == x["expected_raw_cube_sha256"],
            "historical_normalized_sha_exact": got_norm == x["expected_normalized_cube_sha256"],
        }
        if not all(checks.values()):
            print(json.dumps({
                "opaque_index": idx,
                "diagnostic": "historical_cube_sha_gate_failed",
                "checks": checks,
                "actual_raw_sha256": got_raw,
                "actual_normalized_sha256": got_norm,
                "actual_receipt_sha256": sha(rp),
                "expected_receipt_sha256": None,
            }, sort_keys=True), flush=True)
            fail(f"historical cube SHA gate failed at opaque object index {idx}")
        results.append({
            "opaque_index": f"o{idx:03d}", "target_id": x["target_id"], "group_id": x["group_id"], "role": x["role"],
            "source_id": x["source_id"], "tile_id": x["tile_id"], "ra_deg": x["ra_deg"], "dec_deg": x["dec_deg"],
            "raw_sha256": got_raw, "normalized_sha256": got_norm, "checks": checks,
            "receipt_sha256": sha(rp), "pipeline_log_sha256": sha(log)
        })
        print(f"object {idx:02d}/64 exact raw+normalized SHA PASS", flush=True)

    files = []
    for p in sorted(x for x in PRIVATE_OUT.rglob("*") if x.is_file()):
        files.append({"path": p.relative_to(PRIVATE_OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)})
    aggregate = {
        "schema_version": 1,
        "protocol": "HOU-LENS-P4.9.2-MIXED16-EXACT-PAYLOAD-REPLAY",
        "status": "PASS_64_OF_64_RAW_AND_NORMALIZED_HISTORICAL_SHA_REPLAY",
        "manifest_sha256": MANIFEST_SHA,
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
        "scores_computed": False,
        "model_code_executed": False,
        "scenario_generation": False,
        "blind_search": False,
        "future_validation_touched": False,
    }
    (PRIVATE_OUT / "aggregate_receipt.json").write_bytes(canonical(aggregate))
    print(json.dumps({"status": aggregate["status"], "objects": 64, "raw_matches": 64, "normalized_matches": 64, "file_set_sha256": aggregate["file_set_sha256_before_aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
