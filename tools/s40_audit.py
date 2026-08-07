from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('.')

def sha(p: Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

p1=json.loads((ROOT/'s40_phaseB_primary/s40_phaseB_result.json').read_text())
p2=json.loads((ROOT/'s40_phaseB_rerun/s40_phaseB_result.json').read_text())
r1=json.loads((ROOT/'s40_phaseB_primary/s40_phaseB_run_receipt.json').read_text())
r2=json.loads((ROOT/'s40_phaseB_rerun/s40_phaseB_run_receipt.json').read_text())
phaseA=json.loads((ROOT/'s40_phaseA/s40_phaseA_identity_inventory.json').read_text())
source=json.loads((ROOT/'s40_artifacts/source_manifest.json').read_text())

assert sha(ROOT/'s40_phaseB_primary/s40_phaseB_result.json')==sha(ROOT/'s40_phaseB_rerun/s40_phaseB_result.json')
assert sha(ROOT/'s40_phaseB_primary/s40_phaseB_run_receipt.json')==sha(ROOT/'s40_phaseB_rerun/s40_phaseB_run_receipt.json')
assert p1==p2 and r1==r2
assert phaseA['identity_class']=='I1_WITHIN_RECORD_STABLE_IDENTITY_ONLY'
assert phaseA['cross_record_biological_identity_claim_allowed'] is False
assert p1['identity_class']=='I1_WITHIN_RECORD_STABLE_IDENTITY_ONLY'
assert p1['cross_record_biological_identity_claim_allowed'] is False
assert source['development_source_accessed'] is False
assert source['sealed_holdout_requested'] is False
assert source['sealed_holdout_listed'] is False
assert source['sealed_holdout_downloaded'] is False
assert source['sealed_holdout_opened'] is False

sums=[p1['records'][rid]['summary'] for rid in sorted(p1['records'])]
rb_split=float(np.median([s['median_split_half_residual_scale_ratio'] for s in sums]))
rb_lag=float(np.median([s['median_abs_lag1_residual_autocorrelation'] for s in sums]))
r2wins=sum((s['median_abs_corr_residual_R2']<=0.20 and s['median_abs_corr_residual_R2']<s['median_abs_corr_raw_R2']) for s in sums)
popwins=sum((s['median_abs_corr_residual_population']<=0.25 and s['median_abs_corr_residual_population']<s['median_abs_corr_raw_population']) for s in sums)
eventwins=sum(s['fraction_channels_event_rate_ge_0_5_per_minute']>=0.25 for s in sums)
permwins=sum(s['median_matched_to_permuted_residual_variance_ratio']<=0.98 for s in sums)
recalc={
 'G1_identity_I1_or_higher':True,
 'G2_each_record_admissible_fraction_ge_0_75':all(s['admissible_fraction']>=0.75 for s in sums),
 'G3_at_least_three_records_ge_20_admissible':sum(s['admissible_channels']>=20 for s in sums)>=3,
 'G4_balanced_split_half_ratio_in_range':0.67<=rb_split<=1.50,
 'G5_balanced_abs_lag1_le_0_35':rb_lag<=0.35,
 'G6_R2_deconfounding_three_of_four':r2wins>=3,
 'G7_population_deconfounding_three_of_four':popwins>=3,
 'G8_event_support_three_of_four':eventwins>=3,
 'G9_matched_R2_beats_permuted_three_of_four':permwins>=3,
 'G10_no_record_gt_0_25_nonadmissible':all(s['nonadmissible_fraction']<=0.25 for s in sums),
}
assert recalc==p1['checks']
assert sum(recalc.values())==p1['passed_checks']==r1['passed_checks']
assert p1['total_checks']==10==r1['total_checks']
for rid,rec in p1['records'].items():
    assert rec['channel_count']==rec['summary']['structurally_stable_channels']
    adm=[c for c in rec['channels'] if c.get('admissible')]
    assert len(adm)==rec['summary']['admissible_channels']
    for c in adm:
        for key in ('innovation_fraction','matched_to_permuted_residual_variance_ratio','split_half_residual_scale_ratio',
                    'abs_lag1_residual_autocorrelation','abs_corr_residual_R2','abs_corr_raw_R2',
                    'abs_corr_residual_population','abs_corr_raw_population','event_rate_per_minute'):
            assert np.isfinite(c[key])
        assert c['scored_samples']>=150
        assert c['warm_finite_rate']>=0.80 and c['score_finite_rate']>=0.80

status='TRAINING_PASS' if all(recalc.values()) else 'TRAINING_FAIL'
verdict='WITHIN_RECORD_NEURAL_INNOVATION_IDENTIFIABLE' if status=='TRAINING_PASS' else 'NEURAL_INNOVATION_NOT_IDENTIFIABLE_UNDER_S40'
assert p1['status']==status==r1['status'] and p1['verdict']==verdict==r1['verdict']
out={
 'schema':'bio-001-s40-independent-audit-v1','status':'PASS','scientific_status':status,'verdict':verdict,
 'exact_rerun':'PASS','phaseA_identity_class':'I1_WITHIN_RECORD_STABLE_IDENTITY_ONLY',
 'gate_recalculation':'PASS','cross_record_identity_claim_prohibited':'PASS','holdout_nonaccess':'PASS',
 'result_sha256':sha(ROOT/'s40_phaseB_primary/s40_phaseB_result.json'),
 'run_receipt_sha256':sha(ROOT/'s40_phaseB_primary/s40_phaseB_run_receipt.json'),
 'record_balanced':{'split_half':rb_split,'abs_lag1':rb_lag,'R2_pass_records':r2wins,'population_pass_records':popwins,
                    'event_pass_records':eventwins,'matched_R2_pass_records':permwins},
}
(ROOT/'s40_independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
print('BIO001_S40_AUDIT='+json.dumps(out,sort_keys=True))
