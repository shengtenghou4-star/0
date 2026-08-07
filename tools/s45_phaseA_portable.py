from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

import s42_canonical_reconstruction as canon
import s43_calcium_innovation as s43
import s45_phaseA as base

ROOT = Path('s45_data/training')
OUT = Path('s45_output')
OUT.mkdir(parents=True, exist_ok=True)
TOPK = base.TOPK

S42_REF = {
    'BrainScanner20200130_105254': {'channels':128,'frames':1525,'finite_fraction':0.7568698770491803,'valid_frames':1150,'valid_map_sha':'3139a2c8b1d89563f346bcc88121de6f4d31cbc9bf5aba716d4f58e4acffaec6','finite_motion_fits':128,'R_flat':31,'G_flat':48},
    'BrainScanner20200130_110803': {'channels':134,'frames':1466,'finite_fraction':0.9428284905621959,'valid_frames':1433,'valid_map_sha':'957fae480f98bf4ad0d7442bca95d6e49f8671b82eadcde756b5264fdf077efe','finite_motion_fits':134,'R_flat':3,'G_flat':7},
    'BrainScanner20200310_141211': {'channels':116,'frames':1495,'finite_fraction':0.7362933917656557,'valid_frames':1181,'valid_map_sha':'1079f35255b7d9bb2d5a0b191370c2fb1e0ea03347f6371021c8958d25b783bd','finite_motion_fits':116,'R_flat':3,'G_flat':27},
    'BrainScanner20200310_142022': {'channels':97,'frames':1480,'finite_fraction':0.929834215658958,'valid_frames':1396,'valid_map_sha':'bcea5145321ea2eb2bb47e044ae092606d3f543e059408dc994b372f915246b1','finite_motion_fits':97,'R_flat':7,'G_flat':27},
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


def reconstruct_portable(rid: str, cutoff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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
    max_index = min(int(cutoff), rraw.shape[1] - 1)
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
    valid_map = np.flatnonzero(np.mean(np.isnan(I), axis=0) < 0.5)
    valid_sha = hashlib.sha256(np.asarray(valid_map, dtype='<i8').tobytes()).hexdigest()
    time = has_time[:I.shape[1]].copy()
    if valid_map.size:
        time = time - time[valid_map[0]]
    quality = np.ones(I.shape[1], dtype=bool)
    for lo, hi in canon.EXCLUDE_INTERVAL.get(rid, []):
        quality &= ((time < lo) | (time > hi))
    ref = S42_REF[rid]
    observed = {
        'channels': int(I.shape[0]),
        'frames': int(I.shape[1]),
        'finite_fraction': float(np.mean(np.isfinite(I))),
        'valid_frames': int(valid_map.size),
        'valid_map_sha': valid_sha,
        'finite_motion_fits': int(np.count_nonzero(np.all(np.isfinite(coefs), axis=1))),
        'R_flat': int(rfit['flat_fallback_neurons']),
        'G_flat': int(gfit['flat_fallback_neurons']),
        'I_float64_byte_sha256_informational_only': canon.sha_array(I),
    }
    checks = {
        'channels': observed['channels'] == ref['channels'],
        'frames': observed['frames'] == ref['frames'],
        'finite_fraction': abs(observed['finite_fraction'] - ref['finite_fraction']) <= 1e-12,
        'valid_frames': observed['valid_frames'] == ref['valid_frames'],
        'valid_map_sha': observed['valid_map_sha'] == ref['valid_map_sha'],
        'finite_motion_fits': observed['finite_motion_fits'] == ref['finite_motion_fits'],
        'R_flat': observed['R_flat'] == ref['R_flat'],
        'G_flat': observed['G_flat'] == ref['G_flat'],
    }
    if not all(checks.values()):
        raise RuntimeError(f'S42 portable sentinel drift {rid}: {checks} {observed}')
    receipt = {'portable_sentinel': 'PASS', 'portable_checks': checks, 'observed': observed, 'reference': ref, 'flagged_volumes': flagged}
    return I, time, quality, receipt


def neural_record(rid: str, cutoff: int) -> dict[str, Any]:
    I, time, quality, source_receipt = reconstruct_portable(rid, cutoff)
    warm_end = int(np.floor(s43.WARM_FRAC * (I.shape[1] - 1)))
    score_start = warm_end + s43.EMBARGO
    channels = []
    for j in range(I.shape[0]):
        r = s43.select_and_score(I[j], quality, warm_end, score_start)
        if r is None:
            channels.append({'channel': j, 'admissible': False})
        else:
            m = s43.channel_metrics(r, time)
            m.update({'channel': j, 'admissible': True})
            channels.append(m)
    adm = [c for c in channels if c['admissible']]
    summary = {
        'structural_channels': int(I.shape[0]),
        'admissible_channels': len(adm),
        'admissible_fraction': len(adm) / max(I.shape[0], 1),
        'median_raw_lag1_abs_acf': s43.median(adm, 'raw_lag1_abs_acf'),
        'median_innovation_lag1_abs_acf': s43.median(adm, 'innovation_lag1_abs_acf'),
        'median_raw_max_abs_acf': s43.median(adm, 'raw_max_abs_acf'),
        'median_innovation_max_abs_acf': s43.median(adm, 'innovation_max_abs_acf'),
        'median_whitening_ratio': s43.median(adm, 'whitening_ratio'),
        'median_split_half_innovation_scale_ratio': s43.median(adm, 'split_half_innovation_scale_ratio'),
        'median_retained_innovation_variance': s43.median(adm, 'retained_innovation_variance'),
        'fraction_channels_event_rate_ge_0_5_per_minute': float(np.mean([c['innovation_event_rate_per_minute'] >= 0.5 for c in adm])) if adm else 0.0,
        'selected_family_counts': {family: sum(c.get('selected_family') == family for c in adm) for family in s43.FAMILIES},
    }
    return {'record_id': rid, 'warm_end': warm_end, 'score_start': score_start, 'source_receipt': source_receipt, 'summary': summary, 'channels': channels}


def analyze_record(rid: str, cutoff: int) -> dict[str, Any]:
    neural = neural_record(rid, cutoff)
    channels = neural['channels']
    for c in channels:
        c['sparse_quality_pass'] = base.sparse_pass(c)
    ranked = sorted(channels, key=base.sparse_key)
    quality = [c for c in ranked if c['sparse_quality_pass']]
    topk = {str(k): [int(c['channel']) for c in quality[:k]] for k in TOPK if len(quality) >= k}
    folders = sorted(p for p in ROOT.rglob(f'{rid}_MS') if p.is_dir())
    if len(folders) != 1:
        raise RuntimeError((rid, folders))
    folder = folders[0]
    inv = []
    for name in ('heatDataMS.mat','heatData.mat','pointStatsNew.mat','positionDataMS.mat'):
        path = folder / name
        if path.exists():
            inv.append(base.mat_inventory(path, neural['summary']['structural_channels']))
    explicit = []
    for q in inv:
        explicit.extend(q['explicit_ava_numeric_candidates'])
    bfp = base.bfp_candidates(folder, cutoff, neural['summary']['structural_channels'])
    identity_class = 'N3_EXPLICIT_AVA_INDEX' if explicit else ('N1_BFP_CANDIDATES_ONLY' if bfp['candidate_count'] > 0 else 'N0_NO_USABLE_IDENTITY_MEASUREMENT')
    return {
        'record_id': rid, 'cutVolume': int(cutoff), 'structural_channels': int(neural['summary']['structural_channels']),
        'post_cut_identity_frames': int(bfp['post_cut_frames']), 'identity_inventory': inv,
        'explicit_ava_numeric_candidates': explicit, 'bfp': bfp, 'named_identity_class': identity_class,
        'S42_portable_sentinel': neural['source_receipt'], 'S43_summary': neural['summary'],
        'sparse_quality_pass_count': len(quality), 'sparse_quality_pass_fraction': len(quality)/max(len(channels),1),
        'topk_behavior_free_sparse_channels_zero_based': topk,
        'ranked_quality_channels': [{k:c.get(k) for k in ('channel','sparse_quality_pass','selected_family','selected_alpha','scored_common_rows','innovation_lag1_abs_acf','innovation_max_abs_acf','whitening_ratio','retained_innovation_variance','split_half_innovation_scale_ratio','innovation_event_rate_per_minute')} for c in ranked],
    }


def main() -> None:
    base.ROOT = ROOT
    canon.ROOT = ROOT
    s43.ROOT = ROOT
    rows = base.dataset_rows()
    records = {rid: analyze_record(rid, int(cutoff)) for rid, cutoff in rows}
    classes = [records[r]['named_identity_class'] for r in sorted(records)]
    explicit_all = all(c == 'N3_EXPLICIT_AVA_INDEX' for c in classes)
    any_bfp_all = all(records[r]['bfp']['candidate_count'] > 0 for r in records)
    named_summary = 'N3_EXPLICIT_AVA_INDEX' if explicit_all else ('N1_BFP_CANDIDATES_ONLY' if any_bfp_all else 'N0_NO_USABLE_IDENTITY_MEASUREMENT')
    checks = {
        'G1_all_four_cutVolumes_and_nonempty_identity_segments': all(records[r]['post_cut_identity_frames'] > 0 for r in records),
        'G2_identity_inventory_completed_all_four': all(len(records[r]['identity_inventory']) >= 3 for r in records),
        'G3_named_identity_mechanically_adjudicated_without_behavior': all(records[r]['named_identity_class'] in ('N3_EXPLICIT_AVA_INDEX','N1_BFP_CANDIDATES_ONLY','N0_NO_USABLE_IDENTITY_MEASUREMENT') for r in records),
        'G4_S42_portable_canonical_sentinel_all_four': all(records[r]['S42_portable_sentinel']['portable_sentinel'] == 'PASS' for r in records),
        'G5_S43_channel_quality_reconstruction_all_four': all(records[r]['S43_summary']['admissible_channels'] >= 20 for r in records),
        'G6_each_record_at_least_20_sparse_quality_pass_channels': all(records[r]['sparse_quality_pass_count'] >= 20 for r in records),
        'G7_each_record_has_deterministic_top16_sparse_list': all(len(records[r]['topk_behavior_free_sparse_channels_zero_based'].get('16',[])) == 16 for r in records),
    }
    result = {
        'schema': 'bio-001-s45-phaseA-portable-result-v1',
        'status_pre_external_audit': 'PHASE_A_PASS_CANDIDATE' if all(checks.values()) else 'PHASE_A_FAIL',
        'numeric_portability_erratum_applied': True,
        'named_identity_summary': named_summary,
        'named_AVA_future_eligibility': bool(explicit_all),
        'behavior_free_sparse_future_eligibility': bool(checks['G6_each_record_at_least_20_sparse_quality_pass_channels'] and checks['G7_each_record_has_deterministic_top16_sparse_list']),
        'passed_scientific_checks': int(sum(checks.values())), 'total_scientific_checks': 7, 'checks': checks,
        'records': records, 'audit_gates_G8_G9_G10_pending': True,
        'behavior_variables_read': False, 'behavioral_prediction_built': False, 'CeRSI_v2_delta': 0.0,
        'AML310_transition_accessed': False, 'AML32_chip_opened': False,
    }
    write_json(OUT/'s45_phaseA_result.json', result)
    receipt = {
        'schema':'bio-001-s45-phaseA-portable-run-receipt-v1', 'status_pre_external_audit':result['status_pre_external_audit'],
        'named_identity_summary':named_summary, 'named_AVA_future_eligibility':result['named_AVA_future_eligibility'],
        'behavior_free_sparse_future_eligibility':result['behavior_free_sparse_future_eligibility'],
        'passed_scientific_checks':result['passed_scientific_checks'], 'total_scientific_checks':7,
        'CeRSI_v2_delta':0.0, 'AML32_chip_opened':False,
    }
    write_json(OUT/'s45_run_receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
