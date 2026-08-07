from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

import s42_canonical_reconstruction as canon

ROOT = Path('s43_data/training')
OUT = Path('s43_output')
OUT.mkdir(parents=True, exist_ok=True)
WARM_FRAC = 0.40
EMBARGO = 24
FULL_LAGS = (1, 2, 3, 5, 8, 13)
FAMILIES = {'AR1': (1,), 'AR3': (1, 2, 3), 'AR6': FULL_LAGS}
ALPHAS = (0.1, 1.0, 10.0, 100.0)
EPS = 1e-12
EXPECTED_S42_I_SHA = {
    'BrainScanner20200130_105254': '5cf0e603e6a885933bd4a50458060798aff0e96c94eb79a35e6b7623ac7ec5ab',
    'BrainScanner20200130_110803': '21d543119bc18cb622d5db415abd05b9bb8f2cad80b187de102eafbb43069ab4',
    'BrainScanner20200310_141211': '0c076029bcb0916e0763ce0f38eb377e1eb991c37d8f81418d3415c7dacc98ab',
    'BrainScanner20200310_142022': 'a983f51247b13d588bc48420c1bb21736b7c0a0f4761e7cec4198ebda99a2f0c',
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


def robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, 1.4826 * mad


def reconstruct_unsmoothed_I(rid: str, cutoff: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    folders = sorted(p for p in ROOT.rglob(f'{rid}_MS') if p.is_dir())
    if len(folders) != 1:
        raise RuntimeError((rid, folders))
    folder = folders[0]
    path = folder / 'heatDataMS.mat'
    if not path.exists():
        path = folder / 'heatData.mat'
    data = loadmat(path)
    has_time = np.asarray(data['hasPointsTime'], float).squeeze()
    rraw = np.asarray(data['rRaw'], float)[:, :len(has_time)]
    graw = np.asarray(data['gRaw'], float)[:, :len(has_time)]
    rmask = np.asarray(data['rPhotoCorr'], float)[:, :len(has_time)]
    gmask = np.asarray(data['gPhotoCorr'], float)[:, :len(has_time)]
    max_index = rraw.shape[1] - 1 if cutoff is None else min(int(cutoff), rraw.shape[1] - 1)
    keep = np.arange(rraw.shape[1]) <= max_index
    R, rfit = canon.correct_photobleaching(rraw[:, keep], canon.VPS)
    G, gfit = canon.correct_photobleaching(graw[:, keep], canon.VPS)
    R[np.isnan(rmask[:, keep])] = np.nan
    G[np.isnan(gmask[:, keep])] = np.nan
    R = canon.close_nan_holes(R)
    G = canon.close_nan_holes(G)
    flagged = []
    if 'flagged_volumes' in data and len(data['flagged_volumes']) > 0:
        flagged = [int(x) for x in np.asarray(data['flagged_volumes'][0]).ravel() if int(x) <= max_index]
        if flagged:
            R[:, flagged] = np.nan
            G[:, flagged] = np.nan
    I, coefs = canon.decorrelate_neurons_linear(R, G)
    observed_sha = canon.sha_array(I)
    if observed_sha != EXPECTED_S42_I_SHA[rid]:
        raise RuntimeError(f'S42 canonical I drift {rid}: {observed_sha}')
    valid_map = np.flatnonzero(np.mean(np.isnan(I), axis=0) < 0.5)
    time = has_time[:I.shape[1]].copy()
    if valid_map.size:
        time = time - time[valid_map[0]]
    quality = np.ones(I.shape[1], dtype=bool)
    for lo, hi in canon.EXCLUDE_INTERVAL.get(rid, []):
        quality &= ((time < lo) | (time > hi))
    receipt = {
        'I_sha256': observed_sha,
        'valid_map_sha256': __import__('hashlib').sha256(np.asarray(valid_map, dtype='<i8').tobytes()).hexdigest(),
        'channels': int(I.shape[0]), 'frames': int(I.shape[1]), 'flagged_volumes': flagged,
        'R_photobleach_fit_receipt': rfit, 'G_photobleach_fit_receipt': gfit,
        'finite_motion_fits': int(np.count_nonzero(np.all(np.isfinite(coefs), axis=1))),
        'source_quality_excluded_frames': int(np.count_nonzero(~quality)),
    }
    return I, time, quality, receipt


def common_ar6_rows(x: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    y = []
    idx = []
    for t in range(max(FULL_LAGS), len(x)):
        req = [t] + [t - lag for lag in FULL_LAGS]
        if not all(quality[z] for z in req):
            continue
        vals = np.asarray([x[t - lag] for lag in FULL_LAGS], float)
        if np.isfinite(x[t]) and np.all(np.isfinite(vals)):
            rows.append(vals); y.append(float(x[t])); idx.append(t)
    return np.asarray(rows, float), np.asarray(y, float), np.asarray(idx, int)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xa = np.column_stack([np.ones(len(X)), X])
    reg = np.eye(Xa.shape[1]) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y)


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def normalize_fit(Xfit: np.ndarray, yfit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xm = np.median(Xfit, axis=0)
    xmad = np.median(np.abs(Xfit - xm), axis=0)
    xs = 1.4826 * xmad
    xs[~np.isfinite(xs) | (xs < EPS)] = 1.0
    ym, ys = robust_center_scale(yfit)
    if ys < EPS:
        raise RuntimeError('degenerate target scale')
    return xm, xs, np.asarray(ym), float(ys), ys


def select_and_score(x: np.ndarray, quality: np.ndarray, warm_end: int, score_start: int) -> dict[str, Any] | None:
    Xall, yall, idx = common_ar6_rows(x, quality)
    warm = idx <= warm_end
    score = idx >= score_start
    if np.count_nonzero(warm) < 100 or np.count_nonzero(score) < 200:
        return None
    Xw, yw, iw = Xall[warm], yall[warm], idx[warm]
    Xs, ys, iscore = Xall[score], yall[score], idx[score]
    cut = int(np.floor(0.70 * len(Xw)))
    if cut < 70 or len(Xw) - cut < 30:
        return None
    candidates = []
    for family, lags in FAMILIES.items():
        cols = [FULL_LAGS.index(lag) for lag in lags]
        Xtr0, Xval0 = Xw[:cut, cols], Xw[cut:, cols]
        ytr0, yval0 = yw[:cut], yw[cut:]
        xm = np.median(Xtr0, axis=0)
        xmad = np.median(np.abs(Xtr0 - xm), axis=0)
        xsc = 1.4826 * xmad
        xsc[~np.isfinite(xsc) | (xsc < EPS)] = 1.0
        ym, ysc = robust_center_scale(ytr0)
        if ysc < EPS:
            continue
        Xtr = (Xtr0 - xm) / xsc
        Xval = (Xval0 - xm) / xsc
        ytr = (ytr0 - ym) / ysc
        yval = (yval0 - ym) / ysc
        for alpha in ALPHAS:
            beta = ridge_fit(Xtr, ytr, alpha)
            mse = float(np.mean((yval - predict(beta, Xval)) ** 2))
            candidates.append({'mse': mse, 'family': family, 'lags': lags, 'alpha': alpha})
    if not candidates:
        return None
    best_mse = min(c['mse'] for c in candidates)
    tied = [c for c in candidates if c['mse'] <= best_mse + 1e-12]
    selected = sorted(tied, key=lambda c: (len(c['lags']), -c['alpha'], c['family']))[0]
    cols = [FULL_LAGS.index(lag) for lag in selected['lags']]
    Xwf = Xw[:, cols]
    Xsf = Xs[:, cols]
    xm = np.median(Xwf, axis=0)
    xmad = np.median(np.abs(Xwf - xm), axis=0)
    xsc = 1.4826 * xmad
    xsc[~np.isfinite(xsc) | (xsc < EPS)] = 1.0
    ym, ysc = robust_center_scale(yw)
    if ysc < EPS:
        return None
    beta = ridge_fit((Xwf - xm) / xsc, (yw - ym) / ysc, selected['alpha'])
    pred_norm = predict(beta, (Xsf - xm) / xsc)
    pred = pred_norm * ysc + ym
    innov = ys - pred
    if not np.all(np.isfinite(innov)):
        return None
    return {
        'selected_family': selected['family'], 'selected_lags': list(selected['lags']), 'selected_alpha': selected['alpha'],
        'validation_mse_normalized': selected['mse'], 'score_idx': iscore, 'raw': ys, 'innovation': innov,
        'warm_common_rows': int(len(Xw)), 'scored_common_rows': int(len(Xs)),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def acf_abs(values: np.ndarray, idx: np.ndarray, lag: int) -> float:
    targets = idx + lag
    pos = np.searchsorted(idx, targets)
    valid = (pos < len(idx))
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return 1.0
    valid_idx = valid_idx[idx[pos[valid_idx]] == targets[valid_idx]]
    if valid_idx.size < 10:
        return 1.0
    return abs(corr(values[valid_idx], values[pos[valid_idx]]))


def event_count(values: np.ndarray, idx: np.ndarray, scale: float) -> int:
    if scale < EPS:
        return 0
    hits = idx[np.abs(values) > 3.0 * scale]
    if hits.size == 0:
        return 0
    count = 1
    last = int(hits[0])
    for h in hits[1:]:
        if int(h) - last >= 2:
            count += 1
            last = int(h)
    return count


def channel_metrics(result: dict[str, Any], time: np.ndarray) -> dict[str, Any]:
    raw = result['raw']; innov = result['innovation']; idx = result['score_idx']
    raw_acf = {str(l): acf_abs(raw, idx, l) for l in FULL_LAGS}
    inn_acf = {str(l): acf_abs(innov, idx, l) for l in FULL_LAGS}
    raw_max = max(raw_acf.values()); inn_max = max(inn_acf.values())
    _, scale = robust_center_scale(innov)
    half = len(innov) // 2
    _, s1 = robust_center_scale(innov[:half]); _, s2 = robust_center_scale(innov[half:])
    split = s2 / max(s1, EPS)
    retained = float(np.var(innov) / max(np.var(raw), EPS))
    events = event_count(innov, idx, scale)
    duration_min = 0.0
    if len(idx) > 1 and time[idx[-1]] > time[idx[0]]:
        duration_min = float((time[idx[-1]] - time[idx[0]]) / 60.0)
    duration_min = max(duration_min, EPS)
    return {
        'selected_family': result['selected_family'], 'selected_lags': result['selected_lags'], 'selected_alpha': result['selected_alpha'],
        'validation_mse_normalized': result['validation_mse_normalized'], 'warm_common_rows': result['warm_common_rows'],
        'scored_common_rows': result['scored_common_rows'], 'raw_abs_acf': raw_acf, 'innovation_abs_acf': inn_acf,
        'raw_max_abs_acf': raw_max, 'innovation_max_abs_acf': inn_max,
        'raw_lag1_abs_acf': raw_acf['1'], 'innovation_lag1_abs_acf': inn_acf['1'],
        'whitening_ratio': inn_max / max(raw_max, EPS), 'split_half_innovation_scale_ratio': split,
        'retained_innovation_variance': retained, 'innovation_event_rate_per_minute': events / duration_min,
        'innovation_events': events, 'scored_duration_minutes': duration_min,
    }


def median(rows: list[dict[str, Any]], key: str, default: float = 999.0) -> float:
    vals = [float(r[key]) for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.median(vals)) if vals else default


def analyze_record(rid: str, cutoff: int | None) -> dict[str, Any]:
    I, time, quality, source_receipt = reconstruct_unsmoothed_I(rid, cutoff)
    warm_end = int(np.floor(WARM_FRAC * (I.shape[1] - 1)))
    score_start = warm_end + EMBARGO
    channels = []
    for j in range(I.shape[0]):
        r = select_and_score(I[j], quality, warm_end, score_start)
        if r is None:
            channels.append({'channel': j, 'admissible': False})
        else:
            m = channel_metrics(r, time)
            m.update({'channel': j, 'admissible': True})
            channels.append(m)
    adm = [c for c in channels if c['admissible']]
    summary = {
        'structural_channels': int(I.shape[0]), 'admissible_channels': len(adm),
        'admissible_fraction': len(adm) / max(I.shape[0], 1),
        'median_raw_lag1_abs_acf': median(adm, 'raw_lag1_abs_acf'),
        'median_innovation_lag1_abs_acf': median(adm, 'innovation_lag1_abs_acf'),
        'median_raw_max_abs_acf': median(adm, 'raw_max_abs_acf'),
        'median_innovation_max_abs_acf': median(adm, 'innovation_max_abs_acf'),
        'median_whitening_ratio': median(adm, 'whitening_ratio'),
        'median_split_half_innovation_scale_ratio': median(adm, 'split_half_innovation_scale_ratio'),
        'median_retained_innovation_variance': median(adm, 'retained_innovation_variance'),
        'fraction_channels_event_rate_ge_0_5_per_minute': float(np.mean([c['innovation_event_rate_per_minute'] >= 0.5 for c in adm])) if adm else 0.0,
        'selected_family_counts': {family: sum(c.get('selected_family') == family for c in adm) for family in FAMILIES},
    }
    return {'record_id': rid, 'warm_end': warm_end, 'score_start': score_start, 'source_receipt': source_receipt,
            'summary': summary, 'channels': channels}


def main() -> None:
    records = {rid: analyze_record(rid, cutoff) for rid, cutoff in canon.dataset_rows()}
    sums = [records[r]['summary'] for r in sorted(records)]
    qualified = [s for s in sums if s['admissible_fraction'] >= 0.50]
    g1 = len(qualified) >= 3
    g2 = sum(s['median_innovation_lag1_abs_acf'] <= 0.35 for s in sums) >= 3
    g3 = sum(s['median_innovation_max_abs_acf'] <= 0.40 for s in sums) >= 3
    g4 = sum(s['median_whitening_ratio'] <= 0.50 for s in sums) >= 3
    g5 = sum(0.67 <= s['median_split_half_innovation_scale_ratio'] <= 1.50 for s in sums) >= 3
    g6 = sum(0.02 <= s['median_retained_innovation_variance'] <= 0.80 for s in sums) >= 3
    g7 = sum(s['fraction_channels_event_rate_ge_0_5_per_minute'] >= 0.25 for s in sums) >= 3
    g8 = all(s['median_innovation_max_abs_acf'] <= s['median_raw_max_abs_acf'] + 1e-15 for s in qualified) if qualified else False
    checks = {
        'G1_three_of_four_records_ge_0_50_admissible_channels': g1,
        'G2_three_of_four_median_innovation_lag1_le_0_35': g2,
        'G3_three_of_four_median_innovation_max_acf_le_0_40': g3,
        'G4_three_of_four_median_whitening_ratio_le_0_50': g4,
        'G5_three_of_four_split_half_ratio_in_range': g5,
        'G6_three_of_four_retained_variance_in_range': g6,
        'G7_three_of_four_event_support': g7,
        'G8_no_qualified_record_whitening_worse_than_raw': g8,
    }
    passed = sum(checks.values())
    candidate = passed == len(checks)
    result = {
        'schema': 'bio-001-s43-calcium-innovation-result-v1',
        'source_canonical_revision': canon.PREDICTIONCODE_REV,
        'S42_canonical_generator_integrity': 'PASS',
        'status_pre_external_audit': 'TRAINING_PASS_CANDIDATE' if candidate else 'TRAINING_FAIL',
        'verdict_pre_external_audit': 'CANONICAL_NEURAL_INNOVATION_IDENTIFIABLE_CANDIDATE' if candidate else 'CANONICAL_NEURAL_INNOVATION_NOT_IDENTIFIABLE',
        'passed_scientific_checks': passed, 'total_scientific_checks': 8, 'checks': checks,
        'records': records, 'audit_gate_G9_pending': True,
        'CeRSI_v2_delta': 0.0, 'behavioral_prediction_built': False,
        'cross_record_identity_mapping_used': False, 'AML310_transition_accessed': False, 'AML32_chip_opened': False,
    }
    write_json(OUT / 's43_result.json', result)
    receipt = {
        'schema': 'bio-001-s43-run-receipt-v1',
        'status_pre_external_audit': result['status_pre_external_audit'],
        'verdict_pre_external_audit': result['verdict_pre_external_audit'],
        'passed_scientific_checks': passed, 'total_scientific_checks': 8,
        'CeRSI_v2_delta': 0.0, 'AML32_chip_opened': False,
    }
    write_json(OUT / 's43_run_receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
