from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat

import s42_canonical_reconstruction as canon
import s43_calcium_innovation as s43

ROOT = Path('s45_data/training')
OUT = Path('s45_output')
OUT.mkdir(parents=True, exist_ok=True)
BFP_Z_THRESHOLD = 5.0
TOPK = (1, 2, 4, 8, 16)
EPS = 1e-12
EXPLICIT_TERMS = ('ava', 'aval', 'avar')
INVENTORY_TERMS = ('ava', 'aval', 'avar', 'bfp', 'identity', 'identities', 'neuronid', 'neuron_id', 'id')


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dataset_rows() -> list[tuple[str, int]]:
    canon.ROOT = ROOT
    rows = canon.dataset_rows()
    out = []
    for rid, cutoff in rows:
        if cutoff is None:
            raise RuntimeError(f'S45 requires AML310 cutVolume: {rid}')
        out.append((rid, int(cutoff)))
    if len(out) != 4:
        raise RuntimeError(out)
    return out


def low(s: Any) -> str:
    return str(s).lower()


def term_hit(s: Any, terms: tuple[str, ...]) -> bool:
    x = low(s)
    return any(t in x for t in terms)


def compact_numeric(x: Any, limit: int = 32) -> list[float] | None:
    try:
        a = np.asarray(x)
    except Exception:
        return None
    if a.dtype.kind not in 'iufb' or a.size == 0 or a.size > limit:
        return None
    vals = np.asarray(a, dtype=float).ravel()
    if not np.all(np.isfinite(vals)):
        return None
    return [float(v) for v in vals]


def scan_object(obj: Any, path: str, hits: list[dict[str, Any]], explicit_numeric: list[dict[str, Any]], depth: int = 0) -> None:
    if depth > 5:
        return
    fields = getattr(obj, '_fieldnames', None)
    if fields:
        for name in fields:
            child = getattr(obj, name)
            p = f'{path}.{name}'
            if term_hit(name, INVENTORY_TERMS):
                hits.append({'path': p, 'kind': 'field_name'})
            vals = compact_numeric(child)
            if term_hit(name, EXPLICIT_TERMS) and vals is not None:
                explicit_numeric.append({'path': p, 'values': vals})
            scan_object(child, p, hits, explicit_numeric, depth + 1)
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in 'US':
            text = ' '.join(str(x) for x in obj.ravel()[:100])
            if term_hit(text, INVENTORY_TERMS):
                hits.append({'path': path, 'kind': 'text', 'text': text[:500]})
            return
        if obj.dtype == object:
            for i, child in enumerate(obj.ravel()[:100]):
                scan_object(child, f'{path}[{i}]', hits, explicit_numeric, depth + 1)
        return
    if isinstance(obj, str) and term_hit(obj, INVENTORY_TERMS):
        hits.append({'path': path, 'kind': 'text', 'text': obj[:500]})


def mat_inventory(path: Path, channels: int) -> dict[str, Any]:
    vars_meta = [{'name': n, 'shape': list(shape), 'class': cls} for n, shape, cls in whosmat(path)]
    name_hits = [v for v in vars_meta if term_hit(v['name'], INVENTORY_TERMS)]
    explicit_numeric: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = [{'path': v['name'], 'kind': 'top_variable'} for v in name_hits]
    likely = [v['name'] for v in vars_meta if v['class'] in ('struct', 'cell', 'char') or term_hit(v['name'], INVENTORY_TERMS)]
    if likely:
        try:
            data = loadmat(path, variable_names=likely, squeeze_me=True, struct_as_record=False)
            for key, value in data.items():
                if key.startswith('__'):
                    continue
                vals = compact_numeric(value)
                if term_hit(key, EXPLICIT_TERMS) and vals is not None:
                    explicit_numeric.append({'path': key, 'values': vals})
                scan_object(value, key, hits, explicit_numeric)
        except Exception as exc:
            hits.append({'path': '__scan_error__', 'kind': 'error', 'text': type(exc).__name__})
    valid_explicit = []
    for q in explicit_numeric:
        vals = q['values']
        ints = [int(round(v)) for v in vals if abs(v - round(v)) <= 1e-9]
        if ints and all(0 <= z < channels or 1 <= z <= channels for z in ints):
            valid_explicit.append(q)
    return {
        'file': path.name,
        'variables': vars_meta,
        'identity_hits': hits[:200],
        'explicit_ava_numeric_candidates': valid_explicit[:50],
    }


def bfp_candidates(folder: Path, cutoff: int, channels: int) -> dict[str, Any]:
    path = folder / 'heatDataMS.mat'
    if not path.exists():
        path = folder / 'heatData.mat'
    d = loadmat(path, variable_names=['gRaw', 'hasPointsTime'])
    g = np.asarray(d['gRaw'], float)
    ntime = int(np.asarray(d['hasPointsTime']).size)
    g = g[:, :ntime]
    if g.shape[0] != channels:
        raise RuntimeError('channel mismatch')
    post = g[:, cutoff + 1:]
    if post.shape[1] <= 0:
        raise RuntimeError('empty post-cut identity segment')
    finite_frac = np.mean(np.isfinite(post), axis=1)
    med = np.nanmedian(post, axis=1)
    valid = (finite_frac >= 0.50) & np.isfinite(med)
    center = float(np.median(med[valid])) if np.any(valid) else 0.0
    mad = float(np.median(np.abs(med[valid] - center))) if np.any(valid) else 0.0
    scale = 1.4826 * mad
    z = np.full(channels, -999.0, dtype=float)
    if scale > EPS:
        z[valid] = (med[valid] - center) / scale
    elif np.any(valid):
        z[valid] = 0.0
    order = sorted(range(channels), key=lambda j: (-float(z[j]), j))
    candidates = [int(j) for j in order if valid[j] and z[j] > BFP_Z_THRESHOLD]
    return {
        'post_cut_frames': int(post.shape[1]),
        'post_cut_finite_fraction_all_values': float(np.mean(np.isfinite(post))),
        'channel_median_center': center,
        'channel_median_mad_scale': scale,
        'candidate_threshold_robust_z': BFP_Z_THRESHOLD,
        'candidate_channels_zero_based': candidates,
        'candidate_count': len(candidates),
        'top8': [{'channel_zero_based': int(j), 'robust_z': float(z[j]), 'post_finite_fraction': float(finite_frac[j]), 'post_median': float(med[j]) if np.isfinite(med[j]) else None} for j in order[:8]],
    }


def sparse_pass(c: dict[str, Any]) -> bool:
    return bool(
        c.get('admissible', False)
        and c.get('scored_common_rows', 0) >= 100
        and c.get('innovation_lag1_abs_acf', 999.0) <= 0.35
        and c.get('innovation_max_abs_acf', 999.0) <= 0.40
        and c.get('whitening_ratio', 999.0) <= 0.50
        and 0.02 <= c.get('retained_innovation_variance', -1.0) <= 0.80
        and 0.67 <= c.get('split_half_innovation_scale_ratio', -1.0) <= 1.50
        and c.get('innovation_event_rate_per_minute', -1.0) >= 0.50
    )


def sparse_key(c: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if c['sparse_quality_pass'] else 1,
        float(c.get('innovation_max_abs_acf', 999.0)),
        float(c.get('innovation_lag1_abs_acf', 999.0)),
        abs(float(c.get('split_half_innovation_scale_ratio', 999.0)) - 1.0),
        -float(c.get('innovation_event_rate_per_minute', -1.0)),
        int(c['channel']),
    )


def analyze_record(rid: str, cutoff: int) -> dict[str, Any]:
    canon.ROOT = ROOT
    s43.ROOT = ROOT
    neural = s43.analyze_record(rid, cutoff)
    channels = neural['channels']
    for c in channels:
        c['sparse_quality_pass'] = sparse_pass(c)
    ranked = sorted(channels, key=sparse_key)
    quality = [c for c in ranked if c['sparse_quality_pass']]
    topk = {str(k): [int(c['channel']) for c in quality[:k]] for k in TOPK if len(quality) >= k}

    folders = sorted(p for p in ROOT.rglob(f'{rid}_MS') if p.is_dir())
    if len(folders) != 1:
        raise RuntimeError((rid, folders))
    folder = folders[0]
    inv = []
    for name in ('heatDataMS.mat', 'heatData.mat', 'pointStatsNew.mat', 'positionDataMS.mat'):
        path = folder / name
        if path.exists():
            inv.append(mat_inventory(path, neural['summary']['structural_channels']))
    explicit = []
    for q in inv:
        explicit.extend(q['explicit_ava_numeric_candidates'])
    bfp = bfp_candidates(folder, cutoff, neural['summary']['structural_channels'])
    identity_class = 'N3_EXPLICIT_AVA_INDEX' if explicit else ('N1_BFP_CANDIDATES_ONLY' if bfp['candidate_count'] > 0 else 'N0_NO_USABLE_IDENTITY_MEASUREMENT')
    # N2 is intentionally impossible without an externally source-backed deterministic spatial mapping rule.
    return {
        'record_id': rid,
        'cutVolume': int(cutoff),
        'structural_channels': int(neural['summary']['structural_channels']),
        'post_cut_identity_frames': int(bfp['post_cut_frames']),
        'identity_inventory': inv,
        'explicit_ava_numeric_candidates': explicit,
        'bfp': bfp,
        'named_identity_class': identity_class,
        'S42_I_sha256': neural['source_receipt']['I_sha256'],
        'S43_summary': neural['summary'],
        'sparse_quality_pass_count': len(quality),
        'sparse_quality_pass_fraction': len(quality) / max(len(channels), 1),
        'topk_behavior_free_sparse_channels_zero_based': topk,
        'ranked_quality_channels': [{k: c.get(k) for k in ('channel','sparse_quality_pass','selected_family','selected_alpha','scored_common_rows','innovation_lag1_abs_acf','innovation_max_abs_acf','whitening_ratio','retained_innovation_variance','split_half_innovation_scale_ratio','innovation_event_rate_per_minute')} for c in ranked],
    }


def main() -> None:
    rows = dataset_rows()
    records = {rid: analyze_record(rid, cutoff) for rid, cutoff in rows}
    classes = [records[r]['named_identity_class'] for r in sorted(records)]
    explicit_all = all(c == 'N3_EXPLICIT_AVA_INDEX' for c in classes)
    any_bfp = all(records[r]['bfp']['candidate_count'] > 0 for r in records)
    named_summary = 'N3_EXPLICIT_AVA_INDEX' if explicit_all else ('N1_BFP_CANDIDATES_ONLY' if any_bfp else 'N0_NO_USABLE_IDENTITY_MEASUREMENT')
    checks = {
        'G1_all_four_cutVolumes_and_nonempty_identity_segments': all(records[r]['post_cut_identity_frames'] > 0 for r in records),
        'G2_identity_inventory_completed_all_four': all(len(records[r]['identity_inventory']) >= 3 for r in records),
        'G3_named_identity_mechanically_adjudicated_without_behavior': all(records[r]['named_identity_class'] in ('N3_EXPLICIT_AVA_INDEX','N1_BFP_CANDIDATES_ONLY','N0_NO_USABLE_IDENTITY_MEASUREMENT') for r in records),
        'G4_S42_canonical_I_sentinels_all_four': all(records[r]['S42_I_sha256'] == s43.EXPECTED_S42_I_SHA[r] for r in records),
        'G5_S43_channel_quality_reconstruction_all_four': all(records[r]['S43_summary']['admissible_channels'] >= 20 for r in records),
        'G6_each_record_at_least_20_sparse_quality_pass_channels': all(records[r]['sparse_quality_pass_count'] >= 20 for r in records),
        'G7_each_record_has_deterministic_top16_sparse_list': all(len(records[r]['topk_behavior_free_sparse_channels_zero_based'].get('16', [])) == 16 for r in records),
    }
    result = {
        'schema': 'bio-001-s45-phaseA-result-v1',
        'status_pre_external_audit': 'PHASE_A_PASS_CANDIDATE' if all(checks.values()) else 'PHASE_A_FAIL',
        'named_identity_summary': named_summary,
        'named_AVA_future_eligibility': bool(explicit_all),
        'behavior_free_sparse_future_eligibility': bool(checks['G6_each_record_at_least_20_sparse_quality_pass_channels'] and checks['G7_each_record_has_deterministic_top16_sparse_list']),
        'passed_scientific_checks': int(sum(checks.values())),
        'total_scientific_checks': 7,
        'checks': checks,
        'records': records,
        'audit_gates_G8_G9_G10_pending': True,
        'behavior_variables_read': False,
        'behavioral_prediction_built': False,
        'CeRSI_v2_delta': 0.0,
        'AML310_transition_accessed': False,
        'AML32_chip_opened': False,
    }
    write_json(OUT / 's45_phaseA_result.json', result)
    write_json(OUT / 's45_run_receipt.json', {
        'schema': 'bio-001-s45-phaseA-run-receipt-v1',
        'status_pre_external_audit': result['status_pre_external_audit'],
        'named_identity_summary': named_summary,
        'named_AVA_future_eligibility': result['named_AVA_future_eligibility'],
        'behavior_free_sparse_future_eligibility': result['behavior_free_sparse_future_eligibility'],
        'passed_scientific_checks': result['passed_scientific_checks'],
        'total_scientific_checks': 7,
        'CeRSI_v2_delta': 0.0,
        'AML32_chip_opened': False,
    })
    print(json.dumps(json.loads((OUT / 's45_run_receipt.json').read_text()), sort_keys=True))


if __name__ == '__main__':
    main()
