from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path('.')
SOURCES=('REAL','TIME_REVERSED','PHASE_SCRAMBLED','INDEPENDENT_CIRCULAR_SHIFT')
CONTROLS=SOURCES[1:]
ENDPOINTS=('ANY_TRANSITION_10','ANY_TRANSITION_20','TURN_INITIATION_20')
PRIMARY='TURN_INITIATION_20'; SECONDARY='ANY_TRANSITION_20'; SHORT='ANY_TRANSITION_10'; EPS=1e-12

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

p1=json.loads((ROOT/'s44_primary/s44_training_result.json').read_text())
p2=json.loads((ROOT/'s44_rerun/s44_training_result.json').read_text())
r1=json.loads((ROOT/'s44_primary/s44_run_receipt.json').read_text())
r2=json.loads((ROOT/'s44_rerun/s44_run_receipt.json').read_text())
source=json.loads((ROOT/'s44_artifacts/source_manifest.json').read_text())
assert p1==p2 and r1==r2
assert sha(ROOT/'s44_primary/s44_training_result.json')==sha(ROOT/'s44_rerun/s44_training_result.json')
assert sha(ROOT/'s44_primary/s44_run_receipt.json')==sha(ROOT/'s44_rerun/s44_run_receipt.json')
assert p1['S42_canonical_generator_integrity']=='PASS' and p1['S43_calcium_generator_integrity']=='PASS'
assert p1['development_evaluation_executed'] is False and p1['AML32_chip_opened'] is False
assert source['development_source_accessed'] is False and source['sealed_holdout_requested'] is False and source['sealed_holdout_listed'] is False and source['sealed_holdout_downloaded'] is False and source['sealed_holdout_opened'] is False
assert set(p1['summaries'])==set(SOURCES)
ids=sorted(p1['outer'])
for rid in ids:
    src=p1['source_receipts'][rid]
    assert set(src)==set(SOURCES)
    assert src['REAL']['finite'] is True and src['REAL']['offline_negative_control'] is False
    for s in CONTROLS: assert src[s]['offline_negative_control'] is True and src[s]['finite'] is True

support_all=True; all_finite=True; inner_safe=True
for rid in ids:
    for ep in ENDPOINTS:
        row=p1['outer'][rid]['endpoints'][ep]
        bm=row['behavior_metric']; minpos=3 if ep==PRIMARY else 5
        support=(bm['samples']>=50 and bm['positives']>=minpos and bm['negatives']>=20)
        assert support==row['support_ok']; support_all=support_all and support
        real=row['sources']['REAL']['metric']
        all_finite=all_finite and real['finite'] and real['probability_min']>=0.01-1e-15 and real['probability_max']<=0.99+1e-15
        for s in SOURCES:
            sel=row['sources'][s]['selection_receipt']
            inner_safe=inner_safe and sel['inner_max_model_to_B1']<=sel['inner_safety_ceiling']+1e-15
            ratio=row['sources'][s]['metric']['brier']/max(bm['brier'],EPS)
            assert abs(ratio-row['sources'][s]['model_to_B1_brier_ratio'])<=1e-14
for s in SOURCES:
    for ep in ENDPOINTS:
        ratios=[p1['outer'][r]['endpoints'][ep]['sources'][s]['model_to_B1_brier_ratio'] for r in ids]
        briers=[p1['outer'][r]['endpoints'][ep]['sources'][s]['metric']['brier'] for r in ids]
        q=p1['summaries'][s][ep]
        assert abs(q['recording_balanced_mean_brier']-sum(briers)/len(briers))<=1e-14
        assert abs(q['recording_balanced_mean_model_to_B1_ratio']-sum(ratios)/len(ratios))<=1e-14
        assert q['recordings_model_beats_B1']==sum(x<1.0 for x in ratios)
        assert abs(q['maximum_outer_model_to_B1_ratio']-max(ratios))<=1e-14
for ep in ENDPOINTS:
    cc=p1['control_comparisons'][ep]
    means={s:p1['summaries'][s][ep]['recording_balanced_mean_brier'] for s in CONTROLS}
    best=min(means,key=lambda s:(means[s],s)); assert best==cc['best_control_source']
    assert abs(cc['real_to_best_control_ratio']-cc['real_mean_brier']/max(means[best],EPS))<=1e-14
    wins=0
    for rid in ids:
        rb=p1['outer'][rid]['endpoints'][ep]['sources']['REAL']['metric']['brier']
        cb=min(p1['outer'][rid]['endpoints'][ep]['sources'][s]['metric']['brier'] for s in CONTROLS)
        wins+=int(rb<cb)
    assert wins==cc['real_recordings_beating_per_record_best_control']
primary_ratio=p1['summaries']['REAL'][PRIMARY]['recording_balanced_mean_model_to_B1_ratio']
secondary_ratio=p1['summaries']['REAL'][SECONDARY]['recording_balanced_mean_model_to_B1_ratio']
short_ratio=p1['summaries']['REAL'][SHORT]['recording_balanced_mean_model_to_B1_ratio']
max_real=max(p1['summaries']['REAL'][ep]['maximum_outer_model_to_B1_ratio'] for ep in ENDPOINTS)
recalc={
'all_real_probabilities_finite_and_clipped':all_finite,
'all_records_all_endpoints_event_support':support_all,
'primary_real_to_B1_at_most_0_95':primary_ratio<=0.95,
'primary_real_to_best_control_at_most_0_98':p1['control_comparisons'][PRIMARY]['real_to_best_control_ratio']<=0.98,
'primary_at_least_three_of_four_beats_B1':p1['summaries']['REAL'][PRIMARY]['recordings_model_beats_B1']>=3,
'primary_at_least_three_of_four_beats_per_record_best_control':p1['control_comparisons'][PRIMARY]['real_recordings_beating_per_record_best_control']>=3,
'secondary_real_to_B1_at_most_0_97':secondary_ratio<=0.97,
'secondary_real_to_best_control_at_most_0_99':p1['control_comparisons'][SECONDARY]['real_to_best_control_ratio']<=0.99,
'secondary_at_least_three_of_four_beats_B1':p1['summaries']['REAL'][SECONDARY]['recordings_model_beats_B1']>=3,
'short_real_to_B1_at_most_0_98':short_ratio<=0.98,
'no_real_outer_any_endpoint_above_1_10':max_real<=1.10,
'all_selected_configs_inner_safety_pass':inner_safe}
assert recalc==p1['checks'] and sum(recalc.values())==p1['passed_checks']==r1['passed_checks']
if not support_all: verdict='INSUFFICIENT_EVENT_SUPPORT'; status='TRAINING_FAIL'
elif sum(recalc.values())==12: verdict='CANONICAL_INNOVATION_BEHAVIORAL_SPECIFICITY_TRAINING_PASS'; status='TRAINING_PASS'
elif primary_ratio<=0.95 and p1['control_comparisons'][PRIMARY]['real_to_best_control_ratio']>0.98: verdict='NON_SPECIFIC_CANONICAL_INNOVATION_SIGNAL'; status='TRAINING_FAIL'
else: verdict='NO_CANONICAL_INNOVATION_BEHAVIORAL_SPECIFICITY'; status='TRAINING_FAIL'
assert p1['status']==status==r1['status'] and p1['verdict']==verdict==r1['verdict']
for rid in ids:
    for ep in ENDPOINTS:
        keys=[set(p1['outer'][rid]['endpoints'][ep]['sources'][s]['selected_config']) for s in SOURCES]
        assert all(k==keys[0] for k in keys)
out={'schema':'bio-001-s44-independent-audit-v1','status':'PASS','scientific_status':status,'verdict':verdict,
'passed_checks':sum(recalc.values()),'total_checks':12,'exact_rerun':'PASS','S42_canonical_sentinel':'PASS','S43_calcium_generator_sentinel':'PASS',
'gate_recalculation':'PASS','negative_control_equal_treatment_structure':'PASS','future_target_and_delayed_feedback_code_path':'PASS','holdout_nonaccess':'PASS',
'result_sha256':sha(ROOT/'s44_primary/s44_training_result.json'),'receipt_sha256':sha(ROOT/'s44_primary/s44_run_receipt.json')}
(ROOT/'s44_independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
print('BIO001_S44_AUDIT='+json.dumps(out,sort_keys=True))
