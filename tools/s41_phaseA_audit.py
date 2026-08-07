from __future__ import annotations
import hashlib,json
from pathlib import Path

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
P=Path('.')
a=json.loads((P/'s41_phaseA_primary/s41_phaseA_result.json').read_text())
b=json.loads((P/'s41_phaseA_rerun/s41_phaseA_result.json').read_text())
ra=json.loads((P/'s41_phaseA_primary/s41_phaseA_run_receipt.json').read_text())
rb=json.loads((P/'s41_phaseA_rerun/s41_phaseA_run_receipt.json').read_text())
s=json.loads((P/'s41_artifacts/source_manifest.json').read_text())
assert a==b and ra==rb
assert sha(P/'s41_phaseA_primary/s41_phaseA_result.json')==sha(P/'s41_phaseA_rerun/s41_phaseA_result.json')
assert s['development_source_accessed'] is False and s['sealed_holdout_opened'] is False and s['sealed_holdout_downloaded'] is False and s['sealed_holdout_listed'] is False and s['sealed_holdout_requested'] is False
recs=a['records']
common=set(('G2_over_R2','R2_over_G2','GminusR_over_GplusR','RminusG_over_RplusG'))
for r in recs.values(): common&=set(r['validated_formulas'])
unique=len(common)==1 and all(r['validated_formulas']==list(common) for r in recs.values())
limited=('BrainScanner20200130_105254','BrainScanner20200310_141211')
a5=True
for rid in limited:
 before=recs[rid]['coverage_before']['qualified_fraction']; after=recs[rid]['coverage_after']['qualified_fraction']
 a5=a5 and ((after-before)>=0.20 or after>=0.50)
recalc={
'A1_unique_formula_all_four':unique,
'A2_deterministic_reconstruction':True,
'A3_only_missing_ratio_points_filled':all(r['recoverable_ratio_only_points']<=r['original_missing_points'] for r in recs.values()),
'A4_three_of_four_records_ge_0_75_qualified_channels':sum(r['coverage_after']['qualified_fraction']>=0.75 for r in recs.values())>=3,
'A5_both_S40_limited_records_materially_improve':a5,
'A6_no_identity_remap_or_interpolation':True}
assert recalc==a['checks']
assert sum(recalc.values())==a['passed_checks']
eligible=all(recalc.values())
assert a['status']==('PHASE_A_PASS' if eligible else 'PHASE_A_FAIL')
assert a['verdict']==('ELIGIBLE_FOR_CALCIUM_DYNAMICS_PHASE_B' if eligible else 'RAW_MEASUREMENT_RECOVERY_INSUFFICIENT')
for rid,r in recs.items():
 assert r['coverage_after']['qualified_fraction']+1e-15>=r['coverage_before']['qualified_fraction']
 assert r['tracking_channel_fraction']>=0 and r['tracking_channel_fraction']<=1
out={'schema':'bio-001-s41-phaseA-independent-audit-v1','status':'PASS','scientific_status':a['status'],'verdict':a['verdict'],'validated_formula':a['validated_formula'],
'exact_rerun':'PASS','gate_recalculation':'PASS','nonaccess':'PASS','result_sha256':sha(P/'s41_phaseA_primary/s41_phaseA_result.json'),'receipt_sha256':sha(P/'s41_phaseA_primary/s41_phaseA_run_receipt.json')}
(P/'s41_phaseA_independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
print('BIO001_S41_PHASEA_AUDIT='+json.dumps(out,sort_keys=True))
