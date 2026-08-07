from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('.')
EXPECTED_I = {
    'BrainScanner20200130_105254': '5cf0e603e6a885933bd4a50458060798aff0e96c94eb79a35e6b7623ac7ec5ab',
    'BrainScanner20200130_110803': '21d543119bc18cb622d5db415abd05b9bb8f2cad80b187de102eafbb43069ab4',
    'BrainScanner20200310_141211': '0c076029bcb0916e0763ce0f38eb377e1eb991c37d8f81418d3415c7dacc98ab',
    'BrainScanner20200310_142022': 'a983f51247b13d588bc48420c1bb21736b7c0a0f4761e7cec4198ebda99a2f0c',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def qpass(c: dict) -> bool:
    return bool(c.get('sparse_quality_pass', False))


def rank_key(c: dict) -> tuple:
    return (
        0 if qpass(c) else 1,
        float(c.get('innovation_max_abs_acf', 999.0)),
        float(c.get('innovation_lag1_abs_acf', 999.0)),
        abs(float(c.get('split_half_innovation_scale_ratio', 999.0)) - 1.0),
        -float(c.get('innovation_event_rate_per_minute', -1.0)),
        int(c['channel']),
    )


p1 = json.loads((ROOT / 's45_primary/s45_phaseA_result.json').read_text())
p2 = json.loads((ROOT / 's45_rerun/s45_phaseA_result.json').read_text())
r1 = json.loads((ROOT / 's45_primary/s45_run_receipt.json').read_text())
r2 = json.loads((ROOT / 's45_rerun/s45_run_receipt.json').read_text())
source = json.loads((ROOT / 's45_artifacts/source_manifest.json').read_text())
assert p1 == p2 and r1 == r2
assert sha(ROOT / 's45_primary/s45_phaseA_result.json') == sha(ROOT / 's45_rerun/s45_phaseA_result.json')
assert p1['behavior_variables_read'] is False and p1['behavioral_prediction_built'] is False
assert p1['AML310_transition_accessed'] is False and p1['AML32_chip_opened'] is False
assert source['development_source_accessed'] is False
assert source['sealed_holdout_requested'] is False and source['sealed_holdout_listed'] is False and source['sealed_holdout_downloaded'] is False and source['sealed_holdout_opened'] is False
records = p1['records']
assert set(records) == set(EXPECTED_I)
for rid, rec in records.items():
    assert rec['S42_I_sha256'] == EXPECTED_I[rid]
    ranked = rec['ranked_quality_channels']
    assert ranked == sorted(ranked, key=rank_key)
    quality = [c for c in ranked if qpass(c)]
    assert len(quality) == rec['sparse_quality_pass_count']
    for k in (1,2,4,8,16):
        if len(quality) >= k:
            assert rec['topk_behavior_free_sparse_channels_zero_based'][str(k)] == [int(c['channel']) for c in quality[:k]]
    if rec['named_identity_class'] == 'N3_EXPLICIT_AVA_INDEX':
        assert len(rec['explicit_ava_numeric_candidates']) > 0
    if rec['named_identity_class'] == 'N1_BFP_CANDIDATES_ONLY':
        assert rec['bfp']['candidate_count'] > 0 and len(rec['explicit_ava_numeric_candidates']) == 0

recalc = {
    'G1_all_four_cutVolumes_and_nonempty_identity_segments': all(records[r]['post_cut_identity_frames'] > 0 for r in records),
    'G2_identity_inventory_completed_all_four': all(len(records[r]['identity_inventory']) >= 3 for r in records),
    'G3_named_identity_mechanically_adjudicated_without_behavior': all(records[r]['named_identity_class'] in ('N3_EXPLICIT_AVA_INDEX','N1_BFP_CANDIDATES_ONLY','N0_NO_USABLE_IDENTITY_MEASUREMENT') for r in records),
    'G4_S42_canonical_I_sentinels_all_four': all(records[r]['S42_I_sha256'] == EXPECTED_I[r] for r in records),
    'G5_S43_channel_quality_reconstruction_all_four': all(records[r]['S43_summary']['admissible_channels'] >= 20 for r in records),
    'G6_each_record_at_least_20_sparse_quality_pass_channels': all(records[r]['sparse_quality_pass_count'] >= 20 for r in records),
    'G7_each_record_has_deterministic_top16_sparse_list': all(len(records[r]['topk_behavior_free_sparse_channels_zero_based'].get('16', [])) == 16 for r in records),
}
assert recalc == p1['checks']
assert sum(recalc.values()) == p1['passed_scientific_checks'] == r1['passed_scientific_checks']
G8 = True
G9 = True
G10 = True
formal_passed = sum(recalc.values()) + int(G8) + int(G9) + int(G10)
formal_status = 'PHASE_A_PASS' if formal_passed == 10 else 'PHASE_A_FAIL'
out = {
    'schema': 'bio-001-s45-phaseA-independent-audit-v1',
    'status': 'PASS',
    'formal_status': formal_status,
    'formal_passed_checks': formal_passed,
    'formal_total_checks': 10,
    'scientific_checks': recalc,
    'G8_exact_rerun': True,
    'G9_independent_ranking_arithmetic': True,
    'G10_nonbehavior_and_holdout_nonaccess': True,
    'exact_rerun': 'PASS',
    'ranking_recalculation': 'PASS',
    'nonbehavior_sentinel': 'PASS',
    'holdout_nonaccess': 'PASS',
    'named_identity_summary': p1['named_identity_summary'],
    'named_AVA_future_eligibility': p1['named_AVA_future_eligibility'],
    'behavior_free_sparse_future_eligibility': p1['behavior_free_sparse_future_eligibility'],
    'result_sha256': sha(ROOT / 's45_primary/s45_phaseA_result.json'),
    'receipt_sha256': sha(ROOT / 's45_primary/s45_run_receipt.json'),
}
(ROOT / 's45_independent_audit.json').write_text(json.dumps(out, indent=2, sort_keys=True), encoding='utf-8')
print('BIO001_S45_PHASEA_AUDIT=' + json.dumps(out, sort_keys=True))
