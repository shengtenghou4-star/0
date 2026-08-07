from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path('.')
EXPECTED_I_SHA={
'BrainScanner20200130_105254':'5cf0e603e6a885933bd4a50458060798aff0e96c94eb79a35e6b7623ac7ec5ab',
'BrainScanner20200130_110803':'21d543119bc18cb622d5db415abd05b9bb8f2cad80b187de102eafbb43069ab4',
'BrainScanner20200310_141211':'0c076029bcb0916e0763ce0f38eb377e1eb991c37d8f81418d3415c7dacc98ab',
'BrainScanner20200310_142022':'a983f51247b13d588bc48420c1bb21736b7c0a0f4761e7cec4198ebda99a2f0c'}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
p1=json.loads((ROOT/'s43_primary/s43_result.json').read_text())
p2=json.loads((ROOT/'s43_rerun/s43_result.json').read_text())
r1=json.loads((ROOT/'s43_primary/s43_run_receipt.json').read_text())
r2=json.loads((ROOT/'s43_rerun/s43_run_receipt.json').read_text())
source=json.loads((ROOT/'s43_artifacts/source_manifest.json').read_text())
assert p1==p2 and r1==r2
assert sha(ROOT/'s43_primary/s43_result.json')==sha(ROOT/'s43_rerun/s43_result.json')
assert p1['source_canonical_revision']=='ca59416112a9c10a8d6a3179092a7d3c888bcd4e'
assert p1['S42_canonical_generator_integrity']=='PASS'
assert p1['behavioral_prediction_built'] is False and p1['cross_record_identity_mapping_used'] is False
assert source['development_source_accessed'] is False and source['sealed_holdout_requested'] is False and source['sealed_holdout_listed'] is False and source['sealed_holdout_downloaded'] is False and source['sealed_holdout_opened'] is False
for rid,rec in p1['records'].items(): assert rec['source_receipt']['I_sha256']==EXPECTED_I_SHA[rid]
sums=[p1['records'][r]['summary'] for r in sorted(p1['records'])]
qualified=[s for s in sums if s['admissible_fraction']>=0.50]
recalc={
'G1_three_of_four_records_ge_0_50_admissible_channels':len(qualified)>=3,
'G2_three_of_four_median_innovation_lag1_le_0_35':sum(s['median_innovation_lag1_abs_acf']<=0.35 for s in sums)>=3,
'G3_three_of_four_median_innovation_max_acf_le_0_40':sum(s['median_innovation_max_abs_acf']<=0.40 for s in sums)>=3,
'G4_three_of_four_median_whitening_ratio_le_0_50':sum(s['median_whitening_ratio']<=0.50 for s in sums)>=3,
'G5_three_of_four_split_half_ratio_in_range':sum(0.67<=s['median_split_half_innovation_scale_ratio']<=1.50 for s in sums)>=3,
'G6_three_of_four_retained_variance_in_range':sum(0.02<=s['median_retained_innovation_variance']<=0.80 for s in sums)>=3,
'G7_three_of_four_event_support':sum(s['fraction_channels_event_rate_ge_0_5_per_minute']>=0.25 for s in sums)>=3,
'G8_no_qualified_record_whitening_worse_than_raw':all(s['median_innovation_max_abs_acf']<=s['median_raw_max_abs_acf']+1e-15 for s in qualified) if qualified else False}
assert recalc==p1['checks']; assert sum(recalc.values())==p1['passed_scientific_checks']==r1['passed_scientific_checks']
g9=True
formal_passed=sum(recalc.values())+int(g9); formal_total=9
formal_status='TRAINING_PASS' if formal_passed==formal_total else 'TRAINING_FAIL'
formal_verdict='CANONICAL_NEURAL_INNOVATION_IDENTIFIABLE' if formal_status=='TRAINING_PASS' else 'CANONICAL_NEURAL_INNOVATION_NOT_IDENTIFIABLE'
out={'schema':'bio-001-s43-independent-audit-v1','status':'PASS','formal_status':formal_status,'formal_verdict':formal_verdict,
'formal_passed_checks':formal_passed,'formal_total_checks':formal_total,'scientific_checks':recalc,
'G9_exact_rerun_source_sentinel_and_nonaccess':True,'exact_rerun':'PASS','S42_canonical_generator_sentinel':'PASS','gate_recalculation':'PASS','holdout_nonaccess':'PASS',
'result_sha256':sha(ROOT/'s43_primary/s43_result.json'),'receipt_sha256':sha(ROOT/'s43_primary/s43_run_receipt.json')}
(ROOT/'s43_independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
print('BIO001_S43_AUDIT='+json.dumps(out,sort_keys=True))
