#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import pathlib
import time
import warnings
from datetime import UTC, datetime

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

TAP = "https://archive.eso.org/tap_obs/sync"
SODA = "https://dataportal.eso.org/dataPortal/soda/sync"
TARGET_SHA = "ea77f181dca2174cbb85bd77ba5fd072e79b437572c2d69cce76abdcd747c4bc"
H24_MANIFEST_SHA = "9954d6f20a0f9280c0dd41f77775854660c6bc01327807878812105ba760b64b"
RESULT_CERT_SHA = "37571101784b05edfcbb7e697685e1d6c69356c2c44c86a5c3a3158f6cccee16"
BANDS = ("u", "g", "r", "i1", "i2")
MAX_ATTEMPTS = 3
TIMEOUT = 180
RADIUS = 0.01
TOP_LIMIT = 20

PRIVATE_IN = pathlib.Path("private_input")
PRIVATE_OUT = pathlib.Path("private_result")
PUBLIC_OUT = pathlib.Path("public_result")


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fail(msg: str) -> None:
    # Never echo private identifiers or request URLs to public logs.
    raise RuntimeError(msg)


def query_for_tile(tile: str) -> str:
    return f"""SELECT TOP {TOP_LIMIT}
  obs_publisher_did,
  obs_id,
  target_name,
  obs_collection,
  instrument_name,
  facility_name,
  dataproduct_type,
  access_url,
  access_format,
  calib_level,
  s_ra,
  s_dec,
  em_min,
  em_max,
  t_min,
  t_max,
  filter
FROM ivoa.ObsCore
WHERE target_name='{tile}'
  AND dataproduct_type='image'
  AND filter IN ('u_SDSS','g_SDSS','r_SDSS','i_SDSS')
ORDER BY filter, t_min, obs_publisher_did
"""


def tap_rows(session: requests.Session, tile: str) -> tuple[list[dict], dict]:
    q = query_for_tile(tile)
    started = time.monotonic()
    try:
        r = session.post(TAP, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": q}, timeout=TIMEOUT)
        raw = r.content
    except Exception:
        fail("TAP transport failure")
    receipt = {
        "http_status": r.status_code,
        "elapsed_seconds": time.monotonic() - started,
        "response_bytes": len(raw),
        "response_sha256": sha_bytes(raw),
        "query_sha256": sha_bytes(q.encode()),
    }
    if r.status_code != 200:
        fail("TAP non-200 response")
    try:
        payload = json.loads(raw)
    except Exception:
        fail("TAP JSON parse failure")
    fields = [str(x.get("name", "")) for x in payload.get("metadata", [])]
    expected = [
        "obs_publisher_did", "obs_id", "target_name", "obs_collection", "instrument_name",
        "facility_name", "dataproduct_type", "access_url", "access_format", "calib_level",
        "s_ra", "s_dec", "em_min", "em_max", "t_min", "t_max", "filter",
    ]
    if fields != expected:
        fail("TAP response-column drift")
    data = payload.get("data", [])
    if not data or len(data) >= TOP_LIMIT:
        fail("TAP zero-row or truncation ambiguity")
    rows = [dict(zip(fields, row)) for row in data]
    for row in rows:
        if row.get("target_name") != tile or row.get("dataproduct_type") != "image":
            fail("TAP scope drift")
        if row.get("filter") not in {"u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS"}:
            fail("TAP filter drift")
        if not str(row.get("obs_publisher_did", "")).startswith("ivo://eso.org/ID?ADP."):
            fail("TAP invalid ESO dataset ID")
    return rows, receipt


def select_products(rows: list[dict]) -> dict[str, dict]:
    by = {f: [r for r in rows if r["filter"] == f] for f in ("u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS")}
    if any(len(by[f]) != 1 for f in ("u_SDSS", "g_SDSS", "r_SDSS")) or len(by["i_SDSS"]) != 2:
        fail("deterministic product-count gate failed")
    ir = sorted(by["i_SDSS"], key=lambda r: (float(r["t_min"]), float(r["t_max"]), str(r["obs_id"]), str(r["obs_publisher_did"])))
    return {"u": by["u_SDSS"][0], "g": by["g_SDSS"][0], "r": by["r_SDSS"][0], "i1": ir[0], "i2": ir[1]}


def download_one(session: requests.Session, did: str, ra: float, dec: float) -> tuple[bytes | None, list[dict]]:
    params = {"ID": did, "POS": f"CIRCLE {ra:.10f} {dec:.10f} {RADIUS:.10f}"}
    history = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            r = session.get(SODA, params=params, timeout=TIMEOUT, allow_redirects=True)
            body = r.content
            history.append({
                "attempt": attempt,
                "http_status": r.status_code,
                "elapsed_seconds": time.monotonic() - started,
                "content_type": r.headers.get("content-type"),
                "bytes": len(body),
                "sha256": sha_bytes(body),
                "request_url": r.url,
            })
            if r.status_code == 200 and body.startswith(b"SIMPLE  ="):
                return body, history
        except Exception as exc:
            history.append({"attempt": attempt, "elapsed_seconds": time.monotonic() - started, "error_class": type(exc).__name__})
        if attempt < MAX_ATTEMPTS:
            time.sleep(2 ** (attempt - 1))
    return None, history


def inspect_fits(body: bytes, ra: float, dec: float, expected_band: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(io.BytesIO(body), memmap=False, checksum=True) as hdul:
            hdu = hdul[0]
            if hdu.data is None or hdu.data.ndim != 2:
                fail("FITS primary-image dimensionality failure")
            image = np.asarray(hdu.data, dtype=np.float32)
            wcs = WCS(hdu.header)
            x, y = wcs.world_to_pixel_values(ra, dec)
            if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                fail("FITS target-outside-cutout failure")
            centre = float(np.hypot(x - (image.shape[1] - 1) / 2, y - (image.shape[0] - 1) / 2))
            if centre > 2.0:
                fail("FITS target-centering failure")
            matrix = wcs.pixel_scale_matrix
            scales = [float(np.hypot(matrix[0, 0], matrix[1, 0]) * 3600.0), float(np.hypot(matrix[0, 1], matrix[1, 1]) * 3600.0)]
            if max(abs(v - 0.2) for v in scales) > 1e-5:
                fail("FITS pixel-scale failure")
            filt = str(hdu.header.get("FILTER", ""))
            if not filt.lower().startswith(expected_band[0]):
                fail("FITS filter-label failure")
            finite = float(np.isfinite(image).mean())
            if finite <= 0.99:
                fail("FITS finite-coverage failure")
            psf = float(hdu.header.get("PSF_FWHM", np.nan))
            if not np.isfinite(psf) or psf <= 0:
                fail("FITS PSF metadata failure")
            return {
                "shape": list(image.shape),
                "finite_fraction": finite,
                "pixel_scale_arcsec": scales,
                "target_centre_distance_pixels": centre,
                "filter": filt,
                "psf_fwhm_arcsec": psf,
            }


def main() -> None:
    targets_path = PRIVATE_IN / "P495_TARGETS.csv"
    h24_path = PRIVATE_IN / "P495_H24_TILE_PRODUCT_MANIFEST.csv"
    result_cert = PRIVATE_IN / "result-recipient-cert.pem"
    if sha_file(targets_path) != TARGET_SHA:
        fail("target-manifest SHA gate failed")
    if sha_file(h24_path) != H24_MANIFEST_SHA:
        fail("H24 product-manifest SHA gate failed")
    if sha_file(result_cert) != RESULT_CERT_SHA:
        fail("result-recipient certificate SHA gate failed")

    targets = list(csv.DictReader(targets_path.open()))
    if len(targets) != 38:
        fail("target-count gate failed")
    roles = {"positive": 0, "control": 0}
    for row in targets:
        if row.get("role") not in roles:
            fail("target-role gate failed")
        roles[row["role"]] += 1
    if roles != {"positive": 5, "control": 33}:
        fail("target-role-count gate failed")
    tiles = sorted({r["tile_id"] for r in targets})
    if len(tiles) != 11:
        fail("target-tile-count gate failed")

    h24_rows = list(csv.DictReader(h24_path.open()))
    if len(h24_rows) != 30:
        fail("H24 product-count gate failed")
    h24_map = {(r["tile_id"], r["band"]): r["obs_publisher_did"] for r in h24_rows}
    h24_tiles = sorted({r["tile_id"] for r in h24_rows})
    if len(h24_tiles) != 6 or len(h24_map) != 30:
        fail("H24 product-structure gate failed")
    private_tiles = sorted(set(tiles) - set(h24_tiles))
    if len(private_tiles) != 5:
        fail("private-tile-count gate failed")

    PRIVATE_OUT.mkdir(exist_ok=True)
    PUBLIC_OUT.mkdir(exist_ok=True)
    fits_root = PRIVATE_OUT / "fits"
    fits_root.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "HOU-LENS-P4.9.5-explicit-recovery/1.0"

    tile_products: dict[str, dict[str, dict]] = {}
    tap_receipts = []
    h24_exact = 0
    for idx, tile in enumerate(tiles, 1):
        rows, tr = tap_rows(session, tile)
        selected = select_products(rows)
        tile_products[tile] = selected
        tap_receipts.append({"tile_index": idx, "tile_id": tile, "receipt": tr, "selected": {b: selected[b]["obs_publisher_did"] for b in BANDS}})
        if tile in h24_tiles:
            if any(selected[b]["obs_publisher_did"] != h24_map[(tile, b)] for b in BANDS):
                fail("H24 frozen product identity mismatch")
            h24_exact += 5
    if h24_exact != 30:
        fail("H24 30/30 frozen product gate failed")

    # Full private product manifest, never emitted to public logs/artifact in plaintext.
    product_manifest = []
    for tile in tiles:
        for band in BANDS:
            r = tile_products[tile][band]
            product_manifest.append({
                "tile_id": tile, "band": band, "obs_publisher_did": r["obs_publisher_did"],
                "obs_id": r["obs_id"], "filter": r["filter"], "t_min": r["t_min"], "t_max": r["t_max"],
            })
    (PRIVATE_OUT / "tile_product_manifest.json").write_bytes(canonical(product_manifest))
    (PRIVATE_OUT / "tap_receipts.json").write_bytes(canonical(tap_receipts))

    object_receipts = []
    successful_requests = 0
    terminal_requests = 0
    complete_objects = 0
    total_fits_bytes = 0
    for oi, row in enumerate(targets, 1):
        ra, dec = float(row["ra_deg"]), float(row["dec_deg"])
        opaque = f"o{oi:03d}"
        odir = fits_root / opaque
        odir.mkdir(exist_ok=True)
        band_receipts = []
        object_ok = True
        for bi, band in enumerate(BANDS, 1):
            did = tile_products[row["tile_id"]][band]["obs_publisher_did"]
            body, history = download_one(session, did, ra, dec)
            terminal_requests += 1
            item = {"band": band, "dataset_id": did, "history": history, "success": body is not None}
            if body is None:
                object_ok = False
                band_receipts.append(item)
                continue
            try:
                meta = inspect_fits(body, ra, dec, band)
            except Exception:
                object_ok = False
                item["success"] = False
                item["validation_error"] = "FITS_VALIDATION_FAILURE"
                band_receipts.append(item)
                continue
            fp = odir / f"{band}.fits"
            fp.write_bytes(body)
            item.update({"fits": meta, "file": fp.relative_to(PRIVATE_OUT).as_posix(), "file_bytes": len(body), "file_sha256": sha_bytes(body)})
            band_receipts.append(item)
            successful_requests += 1
            total_fits_bytes += len(body)
        if object_ok and len(band_receipts) == 5 and all(x.get("success") for x in band_receipts):
            complete_objects += 1
        object_receipts.append({
            "opaque_index": opaque,
            "object_key": row["object_key"],
            "group_id": row["group_id"],
            "role": row["role"],
            "source_id": row["source_id"],
            "tile_id": row["tile_id"],
            "ra_deg": ra,
            "dec_deg": dec,
            "complete": bool(object_ok and len(band_receipts) == 5 and all(x.get("success") for x in band_receipts)),
            "bands": band_receipts,
        })
        print(f"object {oi:02d}/38 terminal; complete_count={complete_objects}; successful_fits={successful_requests}", flush=True)

    (PRIVATE_OUT / "object_receipts.json").write_bytes(canonical(object_receipts))
    private_files = []
    for p in sorted(x for x in PRIVATE_OUT.rglob("*") if x.is_file()):
        private_files.append({"path": p.relative_to(PRIVATE_OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    file_set_sha = sha_bytes(canonical(private_files))
    aggregate = {
        "schema_version": 1,
        "protocol": "HOU-LENS-P4.9.5-EXPLICIT-ACQUISITION-RECOVERY-SUPERSESSION",
        "created_utc": now(),
        "target_manifest_sha256": TARGET_SHA,
        "target_count": 38,
        "positive_count": 5,
        "control_count": 33,
        "tile_count": 11,
        "h24_frozen_product_matches": h24_exact,
        "private_tile_product_count": len(private_tiles) * 5,
        "logical_request_count": 190,
        "terminal_request_count": terminal_requests,
        "successful_fits_count": successful_requests,
        "complete_object_count": complete_objects,
        "total_fits_bytes": total_fits_bytes,
        "maximum_attempts_per_request": MAX_ATTEMPTS,
        "timeout_seconds": TIMEOUT,
        "cutout_radius_deg": RADIUS,
        "private_file_set_before_aggregate": private_files,
        "private_file_set_sha256_before_aggregate": file_set_sha,
        "scores_computed": False,
        "model_code_executed": False,
        "scenario_generation": False,
        "blind_search": False,
        "unknown_dr5_only_targets_queried": False,
        "future_validation_touched": False,
        "claim_boundary": "Raw baseline FITS byte/provenance recovery only for the exact frozen 38-object P4.9.5 manifest."
    }
    aggregate["status"] = "PASS_38_OF_38_190_OF_190_BYTE_ACQUISITION" if complete_objects == 38 and successful_requests == 190 else "PARTIAL_TECHNICAL_ACQUISITION"
    (PRIVATE_OUT / "aggregate_receipt.json").write_bytes(canonical(aggregate))

    public_summary = {
        "schema_version": 1,
        "protocol": aggregate["protocol"],
        "status": aggregate["status"],
        "target_manifest_sha256": TARGET_SHA,
        "target_count": 38,
        "positive_count": 5,
        "control_count": 33,
        "tile_count": 11,
        "h24_frozen_product_matches": h24_exact,
        "private_tile_product_count": len(private_tiles) * 5,
        "logical_request_count": 190,
        "terminal_request_count": terminal_requests,
        "successful_fits_count": successful_requests,
        "complete_object_count": complete_objects,
        "total_fits_bytes": total_fits_bytes,
        "private_file_set_sha256_before_aggregate": file_set_sha,
        "scores_computed": False,
        "model_code_executed": False,
        "blind_search": False,
        "unknown_dr5_only_targets_queried": False,
        "future_validation_touched": False,
        "private_plaintext_emitted": False,
    }
    (PUBLIC_OUT / "public_summary.pre_encryption.json").write_bytes(canonical(public_summary))
    print(json.dumps({k: public_summary[k] for k in ["status", "target_count", "logical_request_count", "successful_fits_count", "complete_object_count", "total_fits_bytes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
