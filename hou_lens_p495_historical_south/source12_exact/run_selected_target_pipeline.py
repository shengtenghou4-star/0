#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import kids_imaging  # noqa: E402

SODA_URL = "https://dataportal.eso.org/dataPortal/soda/sync"
BANDS = ("u", "g", "r", "i1", "i2")


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_target(path: Path, target_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("split") != "development" or payload.get("holdout_targets_included") != 0:
        raise RuntimeError("selected-target manifest violates split contract")
    matches = [row for row in payload["targets"] if row["target_id"] == target_id]
    if len(matches) != 1:
        raise RuntimeError(f"target {target_id} is not uniquely selected")
    target = matches[0]
    if target.get("split") != "development":
        raise RuntimeError("non-development target requested")
    if set(target.get("science_dataset_ids", {})) != set(BANDS):
        raise RuntimeError("target lacks exactly five science dataset IDs")
    return target


def download(session: requests.Session, did: str, target: dict[str, Any], radius: float, attempts: int, timeout: int):
    history = []
    last = "not attempted"
    params = {
        "ID": did,
        "POS": f"CIRCLE {float(target['ra_deg']):.10f} {float(target['dec_deg']):.10f} {radius:.10f}",
    }
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            response = session.get(SODA_URL, params=params, timeout=timeout, allow_redirects=True)
            body = response.content
            history.append({
                "attempt": attempt,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "byte_size": len(body),
                "request_url": response.url,
            })
            if response.status_code == 200 and body.startswith(b"SIMPLE  ="):
                return body, response.url, history
            last = f"HTTP {response.status_code}, type={response.headers.get('content-type')}, bytes={len(body)}"
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
            history.append({"attempt": attempt, "elapsed_seconds": round(time.monotonic() - started, 3), "error": last})
        if attempt < attempts:
            time.sleep(min(16, 2 ** (attempt - 1)))
    raise RuntimeError(f"SODA download failed after {attempts} attempts: {last}")


def inspect_fits(body: bytes, target: dict[str, Any], expected_band: str) -> tuple[np.ndarray, dict[str, Any]]:
    with fits.open(io.BytesIO(body), memmap=False, checksum=True) as hdul:
        hdu = hdul[0]
        if hdu.data is None or hdu.data.ndim != 2:
            raise RuntimeError("expected a 2-D primary science image")
        image = np.asarray(hdu.data, dtype=np.float32)
        wcs = WCS(hdu.header)
        x, y = wcs.world_to_pixel_values(float(target["ra_deg"]), float(target["dec_deg"]))
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            raise RuntimeError(f"target outside cutout: {(x, y)} {image.shape}")
        centre = float(np.hypot(x - (image.shape[1] - 1) / 2, y - (image.shape[0] - 1) / 2))
        if centre > 2.0:
            raise RuntimeError(f"target not centred: {centre}")
        matrix = wcs.pixel_scale_matrix
        scales = [
            float(np.hypot(matrix[0, 0], matrix[1, 0]) * 3600.0),
            float(np.hypot(matrix[0, 1], matrix[1, 1]) * 3600.0),
        ]
        if max(abs(v - 0.2) for v in scales) > 1e-5:
            raise RuntimeError(f"unexpected pixel scale: {scales}")
        psf = float(hdu.header.get("PSF_FWHM", np.nan))
        if not np.isfinite(psf) or psf <= 0:
            raise RuntimeError("missing positive PSF_FWHM")
        filt = str(hdu.header.get("FILTER", ""))
        if not filt.lower().startswith(expected_band[0]):
            raise RuntimeError(f"filter mismatch for {expected_band}: {filt}")
        finite_fraction = float(np.isfinite(image).mean())
        if finite_fraction <= 0.99:
            raise RuntimeError(f"insufficient finite coverage: {finite_fraction}")
        return image, {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "filter": filt,
            "object": hdu.header.get("OBJECT"),
            "exptime_seconds": hdu.header.get("EXPTIME"),
            "psf_fwhm_arcsec": psf,
            "pixel_scale_arcsec": scales,
            "target_pixel": {"x": float(x), "y": float(y)},
            "target_centre_distance_pixels": centre,
            "finite_fraction": finite_fraction,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    target = load_target(args.targets, args.target_id)
    args.output.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "hou-lens-eight-target-science-pipeline/1.0"})
    images: dict[str, np.ndarray] = {}
    psf: dict[str, float] = {}
    products = []
    shapes = set()
    scales = []
    for band in BANDS:
        did = target["science_dataset_ids"][band]
        body, request_url, history = download(session, did, target, args.radius, args.attempts, args.timeout)
        image, meta = inspect_fits(body, target, band)
        images[band] = image
        psf[band] = float(meta["psf_fwhm_arcsec"])
        shapes.add(tuple(image.shape))
        scales.extend(float(v) for v in meta["pixel_scale_arcsec"])
        path = args.output / f"{band}_science.fits"
        path.write_bytes(body)
        products.append({
            "band_label": band,
            "dataset_id": did,
            "request_url": request_url,
            "request_history": history,
            "byte_size": len(body),
            "sha256": sha256_bytes(body),
            "fits": meta,
        })
    if len(shapes) != 1 or max(scales) - min(scales) > 1e-6:
        raise RuntimeError(f"inconsistent input image geometry: shapes={shapes}, scales={scales}")
    pixscale = float(np.median(scales))

    ugri, ugri_psf, i_meta = kids_imaging.prepare_ugri_from_kids_visits(images, psf, pixscale_arcsec=pixscale)
    matched, target_psf, sigmas = kids_imaging.psf_match_bands(ugri, ugri_psf, pixscale_arcsec=pixscale)
    raw = np.stack([matched[b] for b in ("u", "g", "r", "i")], axis=0).astype(np.float32)
    raw_path = args.output / "psf_matched_ugri_scoring_cube.npz"
    np.savez_compressed(raw_path, image=raw, bands=np.asarray(["u", "g", "r", "i"]), target_id=np.asarray(args.target_id), tile_id=np.asarray(target["tile_id"]), pixel_scale_arcsec=np.asarray(pixscale), target_psf_fwhm_arcsec=np.asarray(target_psf))

    normalized, norm_meta = {}, {}
    for b in ("u", "g", "r", "i"):
        normalized[b], norm_meta[b] = kids_imaging.robust_background_normalize(matched[b])
    model = np.stack([normalized[b] for b in ("u", "g", "r", "i")], axis=0).astype(np.float16)
    model_path = args.output / "normalized_ugri_model_cube.npz"
    np.savez_compressed(model_path, image=model, bands=np.asarray(["u", "g", "r", "i"]), target_id=np.asarray(args.target_id), tile_id=np.asarray(target["tile_id"]), pixel_scale_arcsec=np.asarray(pixscale), target_psf_fwhm_arcsec=np.asarray(target_psf))

    receipt = {
        "schema_version": 1,
        "created_utc": now(),
        "stage": "kids_dr5_phase4_eight_target_science_pipeline",
        "status": "PASS",
        "published_reference": {k: target[k] for k in ("target_id", "ra_deg", "dec_deg", "tile_id", "benchmark_tier", "split")},
        "cutout_radius_deg": args.radius,
        "cutout_count": 5,
        "input_band_labels": list(BANDS),
        "output_band_labels": ["u", "g", "r", "i"],
        "common_shape": list(next(iter(shapes))),
        "pixel_scale_arcsec": pixscale,
        "input_psf_fwhm_arcsec": psf,
        "i_visit_combination": i_meta.to_dict(),
        "ugri_input_psf_fwhm_arcsec": ugri_psf,
        "target_psf_fwhm_arcsec": target_psf,
        "convolution_sigma_pixels": sigmas,
        "raw_cube_shape": list(raw.shape),
        "model_cube_shape": list(model.shape),
        "normalization": norm_meta,
        "products": products,
        "output_sha256": {
            "raw_scoring_cube": sha256_file(raw_path),
            "normalized_model_cube": sha256_file(model_path),
        },
        "auxiliary_policy": "science-only SODA route; weight/mask HTTP-204 limitation remains explicit",
        "blind_search_performed": False,
        "unknown_targets_queried": False,
        "claim_boundary": "Acquires and preprocesses one previously published development reference only; no holdout or blind target is queried or scored.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "cutout_count", "common_shape", "raw_cube_shape", "target_psf_fwhm_arcsec")}, indent=2))


if __name__ == "__main__":
    main()
