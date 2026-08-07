from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path('.')

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

p1=json.loads((ROOT/'s42_primary/s42_canonical_result.json').read_text())
p2=json.loads((ROOT/'s42_rerun/s42_canonical_result.json').read_text())
r1=json.loads((ROOT/'s42_primary/s42_run_receipt.json').read_text())
r2=json.loads((ROOT/'s42_rerun/s42_run_receipt.json').read_text())
source=json.loads((ROOT/'s42_artifacts/source_manifest.json').read_text())
assert p1==p2 and r1==r2
assert sha(ROOT/'s42_primary/s42_canonical_result.json')==sha(ROOT/'s42_rerun/s42_canonical_result.json')
assert sha(ROOT/'s42_primary/s42_run_receipt.json')==sha(ROOT/'s42_rerun/s42_run_receipt.json')
assert p1['source_repository']=='leiferlab/PredictionCode'
assert p1['source_revision']=='ca59416112a9c10a8d6a3179092a7d3c888bcd4e'
assert p1['source_data_handler_blob_sha']=='91c6fa0f3f69cc9f1f8fd7af445a8cfc200d1a79'
assert p1['ratio2_role']=='NONCANONICAL_HISTORICAL_MATLAB_ORDERING_SIGNAL_NOT_USED_AS_NEURAL_INPUT'
assert p1['ratio2_reconstruction_attempted'] is False
assert p1['ratio2_missing_values_filled']==0
assert p1['behavioral_prediction_built'] is False
assert p1['cross_record_identity_mapping_used'] is False
assert source['development_source_accessed'] is False
assert source['sealed_holdout_requested'] is False
assert source['sealed_holdout_listed'] is False
assert source['sealed_holdout_downloaded'] is False
assert source['sealed_holdout_opened'] is False
records=p1['records']
recalc={
 'C1_canonical_smoothed_interpolated_signal_finite_all_records':all(r['canonical_I_smooth_interp_all_finite'] for r in records.values()),
 'C2_motion_fit_available_all_channels_all_records':all(r['canonical_I_channels_with_finite_motion_fit']==r['channels'] for r in records.values()),
 'C3_majority_nan_valid_map_retains_at_least_half_frames_all_records':all(r['valid_population_frame_fraction']>=0.50 for r in records.values()),
 'C4_ratio2_not_used_as_canonical_input':True,
 'C5_no_cross_record_identity_mapping':True,
 'C6_no_behavioral_prediction_built':True,
}
assert recalc==p1['checks']
assert sum(recalc.values())==p1['passed_checks']==r1['passed_checks']
assert p1['total_checks']==6==r1['total_checks']
status='TRAINING_SOURCE_RECONSTRUCTION_PASS' if all(recalc.values()) else 'TRAINING_SOURCE_RECONSTRUCTION_FAIL'
verdict='RATIO2_NONCANONICAL_CANONICAL_SIGNAL_RECOVERED' if status.endswith('PASS') else 'RATIO2_NONCANONICAL_CANONICAL_SIGNAL_RECONSTRUCTION_FAIL'
assert p1['status']==status==r1['status']
assert p1['verdict']==verdict==r1['verdict']
for rid,r in records.items():
    assert len(r['I_sha256'])==64 and len(r['I_smooth_interp_sha256'])==64
    assert r['valid_population_frames']<=r['frames']
    assert r['analysis_frames_after_source_exclusions']<=r['valid_population_frames']
out={
 'schema':'bio-001-s42-independent-audit-v1',
 'status':'PASS',
 'scientific_status':status,
 'verdict':verdict,
 'exact_rerun':'PASS',
 'source_revision_sentinel':'PASS',
 'ratio2_noncanonical_sentinel':'PASS',
 'gate_recalculation':'PASS',
 'holdout_nonaccess':'PASS',
 'result_sha256':sha(ROOT/'s42_primary/s42_canonical_result.json'),
 'receipt_sha256':sha(ROOT/'s42_primary/s42_run_receipt.json'),
}
(ROOT/'s42_independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
print('BIO001_S42_AUDIT='+json.dumps(out,sort_keys=True))
