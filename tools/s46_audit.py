from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path('.')
EXPECTED_TOP16={
 'BrainScanner20200130_105254':[44,69,51,96,105,20,119,32,123,93,126,60,89,48,49,61],
 'BrainScanner20200130_110803':[39,107,103,111,28,84,89,58,51,27,76,66,77,38,17,62],
 'BrainScanner20200310_141211':[62,66,59,24,81,60,16,18,28,13,49,12,71,75,25,17],
 'BrainScanner20200310_142022':[8,68,36,6,90,18,32,51,39,17,31,47,21,57,0,23]}
ALLOWED={1,2,4,8,16}; ENDPOINTS=('ANY_TRANSITION_10','ANY_TRANSITION_20','TURN_INITIATION_20'); PRIMARY='TURN_INITIATION_20'; SECONDARY='ANY_TRANSITION_20'; SHORT='ANY_TRANSITION_10'; SOURCES=('REAL','TIME_REVERSED','PHASE_SCRAMBLED','INDEPENDENT_CIRCULAR_SHIFT'); CONTROLS=SOURCES[1:]

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

p1=json.loads((ROOT/'s46_primary/s46_training_result.json').read_text()); p2=json.loads((ROOT/'s46_rerun/s46_training_result.json').read_text())
r1=json.loads((ROOT/'s46_primary/s46_run_receipt.json').read_text()); r2=json.loads((ROOT/'s46_rerun/s46_run_receipt.json').read_text())
source=json.loads((ROOT/'s46_artifacts/source_manifest.json').read_text())
assert p1==p2 and r1==r2
assert sha(ROOT/'s46_primary/s46_training_result.json')==sha(ROOT/'s46_rerun/s46_training_result.json')
assert p1['frozen_sparse_top16_zero_based']==EXPECTED_TOP16 and set(p1['allowed_k'])==ALLOWED
assert p1['S45_sparse_authority']=='afb56c7b84ef56b8a75663a0567e1ecdaf1e350e'
assert p1['named_AVA_used'] is False and p1['global_POP_PCA_used'] is False and p1['channel_membership_selected_with_behavior'] is False
assert p1['development_evaluation_executed'] is False and p1['AML32_chip_opened'] is False
assert source['development_source_accessed'] is False and source['sealed_holdout_requested'] is False and source['sealed_holdout_listed'] is False and source['sealed_holdout_downloaded'] is False and source['sealed_holdout_opened'] is False
ids=sorted(p1['outer'])
for rid in ids:
 for ep in ENDPOINTS:
  assert set(p1['outer'][rid]['endpoints'][ep]['sources'])==set(SOURCES)
  for s in SOURCES:
   cfg=p1['outer'][rid]['endpoints'][ep]['sources'][s]['selected_config']
   if cfg['off']: assert cfg['k']==0
   else: assert int(cfg['k']) in ALLOWED
   assert float(p1['outer'][rid]['endpoints'][ep]['sources'][s]['selection_receipt']['inner_max_model_to_B1'])<=1.10+1e-15
summ=p1['summaries']; comp=p1['control_comparisons']
max_real=max(summ['REAL'][ep]['maximum_outer_model_to_B1_ratio'] for ep in ENDPOINTS)
recalc={
 'all_real_probabilities_finite_and_clipped':all(p1['outer'][r]['endpoints'][ep]['sources']['REAL']['metric']['finite'] and p1['outer'][r]['endpoints'][ep]['sources']['REAL']['metric']['probability_min']>=0.01-1e-15 and p1['outer'][r]['endpoints'][ep]['sources']['REAL']['metric']['probability_max']<=0.99+1e-15 for r in ids for ep in ENDPOINTS),
 'all_records_all_endpoints_event_support':all(p1['outer'][r]['endpoints'][ep]['support_ok'] for r in ids for ep in ENDPOINTS),
 'primary_real_to_B1_at_most_0_95':summ['REAL'][PRIMARY]['recording_balanced_mean_model_to_B1_ratio']<=0.95,
 'primary_real_to_best_control_at_most_0_98':comp[PRIMARY]['real_to_best_control_ratio']<=0.98,
 'primary_at_least_three_of_four_beats_B1':summ['REAL'][PRIMARY]['recordings_model_beats_B1']>=3,
 'primary_at_least_three_of_four_beats_per_record_best_control':comp[PRIMARY]['real_recordings_beating_per_record_best_control']>=3,
 'secondary_real_to_B1_at_most_0_97':summ['REAL'][SECONDARY]['recording_balanced_mean_model_to_B1_ratio']<=0.97,
 'secondary_real_to_best_control_at_most_0_99':comp[SECONDARY]['real_to_best_control_ratio']<=0.99,
 'secondary_at_least_three_of_four_beats_B1':summ['REAL'][SECONDARY]['recordings_model_beats_B1']>=3,
 'short_real_to_B1_at_most_0_98':summ['REAL'][SHORT]['recording_balanced_mean_model_to_B1_ratio']<=0.98,
 'no_real_outer_any_endpoint_above_1_10':max_real<=1.10,
 'all_selected_configs_inner_safety_pass':all(p1['outer'][r]['endpoints'][ep]['sources'][s]['selection_receipt']['inner_max_model_to_B1']<=1.10+1e-15 for r in ids for ep in ENDPOINTS for s in SOURCES)}
assert recalc==p1['checks'] and sum(recalc.values())==p1['passed_checks']==r1['passed_checks']
if not recalc['all_records_all_endpoints_event_support']: status='TRAINING_FAIL'; verdict='INSUFFICIENT_EVENT_SUPPORT'
elif sum(recalc.values())==12: status='TRAINING_PASS'; verdict='FROZEN_SPARSE_BEHAVIORAL_SPECIFICITY_TRAINING_PASS'
elif summ['REAL'][PRIMARY]['recording_balanced_mean_model_to_B1_ratio']<=0.95 and comp[PRIMARY]['real_to_best_control_ratio']>0.98: status='TRAINING_FAIL'; verdict='NON_SPECIFIC_FROZEN_SPARSE_SIGNAL'
else: status='TRAINING_FAIL'; verdict='NO_FROZEN_SPARSE_BEHAVIORAL_SPECIFICITY'
assert p1['status']==status==r1['status'] and p1['verdict']==verdict==r1['verdict']
src=(ROOT/'tools/s46_pipeline.py').read_text(encoding='utf-8')
base=(ROOT/'tools/s44_pipeline.py').read_text(encoding='utf-8')
assert 'matured=[q for q in pending if q[0]<=f]' in base and 'p=model.predict(x)' in base
assert 'FROZEN_TOP16' in src and 'READOUT_TO_K' in src and 's44.readout_features=sparse_features' in src
out={'schema':'bio-001-s46-independent-audit-v1','status':'PASS','scientific_status':status,'verdict':verdict,
 'exact_rerun':'PASS','gate_recalculation':'PASS','frozen_sparse_membership_sentinel':'PASS','delayed_feedback_code_sentinel':'PASS','equal_source_candidate_structure':'PASS','holdout_nonaccess':'PASS',
 'passed_checks':sum(recalc.values()),'total_checks':12,'checks':recalc,
 'result_sha256':sha(ROOT/'s46_primary/s46_training_result.json'),'receipt_sha256':sha(ROOT/'s46_primary/s46_run_receipt.json')}
(ROOT/'s46_independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
print('BIO001_S46_AUDIT='+json.dumps(out,sort_keys=True))
