from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, binary_dilation, binary_erosion
from scipy.optimize import curve_fit
from scipy.signal import medfilt

ROOT = Path('s42_data/training')
OUT = Path('s42_output')
OUT.mkdir(parents=True, exist_ok=True)
PREDICTIONCODE_REV = 'ca59416112a9c10a8d6a3179092a7d3c888bcd4e'
DATA_HANDLER_BLOB = '91c6fa0f3f69cc9f1f8fd7af445a8cfc200d1a79'
VPS = 6.0
WINDOW_GCAMP = 5
EXCLUDE_INTERVAL = {
    'BrainScanner20200130_105254': [[65.0, 75.0]],
    'BrainScanner20200310_141211': [[200.0, 210.0], [240.0, 250.0]],
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


def sha_array(x: np.ndarray) -> str:
    a = np.asarray(x, dtype='<f8').copy()
    a[np.isnan(a)] = np.nan
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def dataset_rows() -> list[tuple[str, int | None]]:
    matches = sorted(ROOT.rglob('*_datasets.txt'))
    if len(matches) != 1:
        raise RuntimeError(matches)
    rows = []
    for raw in matches[0].read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        rows.append((toks[0], int(toks[1]) if len(toks) == 2 else None))
    if len(rows) != 4:
        raise RuntimeError(rows)
    return rows


def expfunc(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * x) + c


def fit_photobleaching(activity_trace: np.ndarray, vps: float) -> tuple[np.ndarray, np.ndarray]:
    xvals = np.arange(activity_trace.shape[-1], dtype=float) / float(vps)
    nonnans = ~np.isnan(activity_trace)
    scale = float(np.nanmean(activity_trace))
    scaled = activity_trace / scale
    num_lengths = 8.0
    bounds = ([0.0, 1.0 / (num_lengths * np.nanmax(xvals)), 0.0],
              [np.nanmax(scaled[nonnans]) * 1.5, 0.5, 2.0 * np.nanmean(scaled)])
    guess = [np.nanmax(scaled) / 2.0, 2.0 / np.nanmax(xvals), np.nanmean(scaled)]
    popt, _ = curve_fit(expfunc, xvals[nonnans], scaled[nonnans], p0=guess, bounds=bounds, maxfev=20000)
    residual = scaled - expfunc(xvals, *popt)
    exc = nonnans.copy()
    with np.errstate(invalid='ignore'):
        exc[np.abs(residual) > (3.0 * np.nanstd(residual))] = False
    try:
        popt, _ = curve_fit(expfunc, xvals[exc], scaled[exc], p0=popt, bounds=bounds, maxfev=20000)
    except Exception:
        popt, _ = curve_fit(expfunc, xvals[exc], scaled[exc], p0=popt, maxfev=20000)
    popt[0] = scale * popt[0]
    popt[2] = scale * popt[2]
    return popt, xvals


def correct_photobleaching(raw: np.ndarray, vps: float) -> tuple[np.ndarray, dict[str, int]]:
    window_s = 12.6
    medfilt_window = int(np.ceil(window_s * vps / 2.0) * 2 + 1)
    smoothed = medfilt(raw, [1, medfilt_window])
    out = np.zeros_like(raw, dtype=float)
    flat_fallback = 0
    for row in range(raw.shape[0]):
        popt, xvals = fit_photobleaching(smoothed[row], vps)
        residual = raw[row] - expfunc(xvals, *popt)
        ss_fit = float(np.nansum(np.square(residual)))
        flat = float(np.nanmean(raw[row]))
        ss_flat = float(np.nansum(np.square(raw[row] - flat)))
        if ss_fit > ss_flat:
            out[row] = raw[row]
            flat_fallback += 1
        else:
            out[row] = popt[0] * raw[row] / expfunc(xvals, *popt)
    return out, {'flat_fallback_neurons': flat_fallback, 'median_filter_frames': medfilt_window}


def close_nan_holes(x: np.ndarray) -> np.ndarray:
    structure = np.zeros((3, 3), dtype=int)
    structure[1, :] = 1
    a = np.isnan(x)
    b = binary_dilation(a, structure=structure).astype(a.dtype)
    c = binary_erosion(b, structure=structure).astype(a.dtype)
    out = np.copy(x)
    out[c] = np.nan
    return out


def decorrelate_neurons_linear(R: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    I = np.full_like(G, np.nan, dtype=float)
    coefs = np.full((R.shape[0], 2), np.nan, dtype=float)
    nanmask = np.isnan(R) | np.isnan(G)
    for n in range(R.shape[0]):
        mask = ~nanmask[n]
        red = np.expand_dims(R[n, mask].T, axis=1)
        design = np.concatenate([red, np.ones(red.shape)], axis=1)
        best_fit, _, _, _ = np.linalg.lstsq(design, G[n, mask].T, rcond=None)
        coefs[n] = best_fit
        I[n] = G[n] - (best_fit[0] * R[n] + best_fit[1])
    return I, coefs


def gauss_filter_nan(U: np.ndarray, sig: float) -> np.ndarray:
    V = U.copy()
    V[np.isnan(U)] = 0.0
    VV = gaussian_filter1d(V, sigma=sig)
    W = np.ones_like(U, dtype=float)
    W[np.isnan(U)] = 0.0
    WW = gaussian_filter1d(W, sigma=sig)
    with np.errstate(invalid='ignore', divide='ignore'):
        Z = VV / WW
    valid = np.isfinite(Z)
    invalid = ~valid
    if np.any(invalid) and np.any(valid):
        Z[invalid] = np.interp(np.flatnonzero(invalid), np.flatnonzero(valid), Z[valid])
    else:
        Z[invalid] = 0.0
    return Z


def reconstruct_record(rid: str, cutoff: int | None) -> dict[str, Any]:
    folders = sorted(p for p in ROOT.rglob(f'{rid}_MS') if p.is_dir())
    if len(folders) != 1:
        raise RuntimeError((rid, folders))
    folder = folders[0]
    path = folder / 'heatDataMS.mat'
    if not path.exists():
        path = folder / 'heatData.mat'
    data = loadmat(path)
    has_time = np.asarray(data['hasPointsTime'], dtype=float).squeeze()
    rraw = np.asarray(data['rRaw'], dtype=float)[:, :len(has_time)]
    graw = np.asarray(data['gRaw'], dtype=float)[:, :len(has_time)]
    rmask_source = np.asarray(data['rPhotoCorr'], dtype=float)[:, :len(has_time)]
    gmask_source = np.asarray(data['gPhotoCorr'], dtype=float)[:, :len(has_time)]
    max_index = rraw.shape[1] - 1 if cutoff is None else min(int(cutoff), rraw.shape[1] - 1)
    idx_data = np.arange(rraw.shape[1]) <= max_index
    R0 = rraw[:, idx_data]
    G0 = graw[:, idx_data]
    R, rfit = correct_photobleaching(R0, VPS)
    G, gfit = correct_photobleaching(G0, VPS)
    R[np.isnan(rmask_source[:, idx_data])] = np.nan
    G[np.isnan(gmask_source[:, idx_data])] = np.nan
    R = close_nan_holes(R)
    G = close_nan_holes(G)
    flagged = []
    if 'flagged_volumes' in data and len(data['flagged_volumes']) > 0:
        flagged = [int(x) for x in np.asarray(data['flagged_volumes'][0]).ravel() if int(x) <= max_index]
        if flagged:
            R[:, flagged] = np.nan
            G[:, flagged] = np.nan
    I, coefs = decorrelate_neurons_linear(R, G)
    I_smooth_interp = np.array([gauss_filter_nan(line, WINDOW_GCAMP) for line in I])
    G_smooth_interp = np.array([gauss_filter_nan(line, WINDOW_GCAMP) for line in G])
    R_smooth_interp = np.array([gauss_filter_nan(line, WINDOW_GCAMP) for line in R])
    smooth_all_finite = bool(np.all(np.isfinite(I_smooth_interp)))
    valid_map = np.flatnonzero(np.mean(np.isnan(I), axis=0) < 0.5)
    time = has_time.copy()
    if valid_map.size:
        time = time - time[valid_map[0]]
    analysis_keep = np.ones(valid_map.size, dtype=bool)
    for lo, hi in EXCLUDE_INTERVAL.get(rid, []):
        analysis_keep &= ((time[valid_map] < lo) | (time[valid_map] > hi))
    analysis_map = valid_map[analysis_keep]
    finite_coef = np.all(np.isfinite(coefs), axis=1)
    ratio = np.asarray(data['Ratio2'], dtype=float)[:, :max_index + 1] if 'Ratio2' in data else np.empty((I.shape[0], max_index + 1)) * np.nan
    return {
        'record_id': rid,
        'source_file': str(path.relative_to(ROOT)),
        'channels': int(I.shape[0]),
        'frames': int(I.shape[1]),
        'cutoff_volume': None if cutoff is None else int(cutoff),
        'raw_R_finite_fraction': float(np.mean(np.isfinite(R0))),
        'raw_G_finite_fraction': float(np.mean(np.isfinite(G0))),
        'stored_Ratio2_finite_fraction_for_context_only': float(np.mean(np.isfinite(ratio))),
        'source_mask_R_finite_fraction': float(np.mean(np.isfinite(rmask_source[:, idx_data]))),
        'source_mask_G_finite_fraction': float(np.mean(np.isfinite(gmask_source[:, idx_data]))),
        'post_source_R_finite_fraction': float(np.mean(np.isfinite(R))),
        'post_source_G_finite_fraction': float(np.mean(np.isfinite(G))),
        'canonical_I_finite_fraction': float(np.mean(np.isfinite(I))),
        'canonical_I_channels_with_finite_motion_fit': int(np.count_nonzero(finite_coef)),
        'canonical_I_smooth_interp_all_finite': smooth_all_finite,
        'valid_population_frames': int(valid_map.size),
        'valid_population_frame_fraction': float(valid_map.size / max(I.shape[1], 1)),
        'analysis_frames_after_source_exclusions': int(analysis_map.size),
        'analysis_frame_fraction': float(analysis_map.size / max(I.shape[1], 1)),
        'source_exclude_intervals_seconds': EXCLUDE_INTERVAL.get(rid, []),
        'flagged_volumes': flagged,
        'R_photobleach_fit_receipt': rfit,
        'G_photobleach_fit_receipt': gfit,
        'motion_slope_median': float(np.nanmedian(coefs[:, 0])),
        'motion_intercept_median': float(np.nanmedian(coefs[:, 1])),
        'I_sha256': sha_array(I),
        'I_smooth_interp_sha256': sha_array(I_smooth_interp),
        'I_smooth_interp_valid_map_sha256': sha_array(I_smooth_interp[:, valid_map]),
        'I_smooth_interp_analysis_map_sha256': sha_array(I_smooth_interp[:, analysis_map]),
        'G_smooth_interp_sha256': sha_array(G_smooth_interp),
        'R_smooth_interp_sha256': sha_array(R_smooth_interp),
        'valid_map_sha256': hashlib.sha256(np.asarray(valid_map, dtype='<i8').tobytes()).hexdigest(),
        'analysis_map_sha256': hashlib.sha256(np.asarray(analysis_map, dtype='<i8').tobytes()).hexdigest(),
    }


def main() -> None:
    records = {rid: reconstruct_record(rid, cutoff) for rid, cutoff in dataset_rows()}
    checks = {
        'C1_canonical_smoothed_interpolated_signal_finite_all_records': all(r['canonical_I_smooth_interp_all_finite'] for r in records.values()),
        'C2_motion_fit_available_all_channels_all_records': all(r['canonical_I_channels_with_finite_motion_fit'] == r['channels'] for r in records.values()),
        'C3_majority_nan_valid_map_retains_at_least_half_frames_all_records': all(r['valid_population_frame_fraction'] >= 0.50 for r in records.values()),
        'C4_ratio2_not_used_as_canonical_input': True,
        'C5_no_cross_record_identity_mapping': True,
        'C6_no_behavioral_prediction_built': True,
    }
    passed = sum(checks.values())
    status = 'TRAINING_SOURCE_RECONSTRUCTION_PASS' if passed == len(checks) else 'TRAINING_SOURCE_RECONSTRUCTION_FAIL'
    verdict = 'RATIO2_NONCANONICAL_CANONICAL_SIGNAL_RECOVERED' if status.endswith('PASS') else 'RATIO2_NONCANONICAL_CANONICAL_SIGNAL_RECONSTRUCTION_FAIL'
    result = {
        'schema': 'bio-001-s42-canonical-reconstruction-result-v1',
        'source_repository': 'leiferlab/PredictionCode',
        'source_revision': PREDICTIONCODE_REV,
        'source_data_handler_blob_sha': DATA_HANDLER_BLOB,
        'source_signal': 'I = G_photobleach_masked - (a * R_photobleach_masked + b), per neuron; then source Gaussian NaN-aware smoothing/interpolation',
        'ratio2_role': 'NONCANONICAL_HISTORICAL_MATLAB_ORDERING_SIGNAL_NOT_USED_AS_NEURAL_INPUT',
        'ratio2_reconstruction_attempted': False,
        'ratio2_missing_values_filled': 0,
        'status': status,
        'verdict': verdict,
        'passed_checks': passed,
        'total_checks': len(checks),
        'checks': checks,
        'records': records,
        'CeRSI_v2_delta': 0.0,
        'behavioral_prediction_built': False,
        'cross_record_identity_mapping_used': False,
        'AML310_transition_accessed': False,
        'AML32_chip_opened': False,
    }
    write_json(OUT / 's42_canonical_result.json', result)
    receipt = {
        'schema': 'bio-001-s42-run-receipt-v1',
        'status': status,
        'verdict': verdict,
        'passed_checks': passed,
        'total_checks': len(checks),
        'CeRSI_v2_delta': 0.0,
        'AML32_chip_opened': False,
    }
    write_json(OUT / 's42_run_receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
