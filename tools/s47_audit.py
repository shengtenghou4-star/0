from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path('.')

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

p1=json.loads((ROOT/'s47_primary/s47_result.json').read_text()); p2=json.loads((ROOT/'s47_rerun/s47_result.json').read_text()); r1=json.loads((ROOT/'s47_primary/s47_run_receipt.json').read_text()); r2=json.loads((ROOT/'s47_rerun/s47_run_receipt.json').read_text()); src=json.loads((ROOT/'s47_artifacts/source_manifest.json').read_text())
assert p1==p2 and r1==r2 and sha(ROOT/'s47_primary/s47_result.json')==sha(ROOT/'s47_rerun/s47_result.json')
assert p1['behavior_variables_loaded'] is False and p1['behavior_variables_used'] is False and p1['behavioral_prediction_built'] is False
assert p1['AML310_transition_accessed'] is False and p1['AML32_chip_opened'] is False
assert src['development_source_accessed'] is False and src['sealed_holdout_requested'] is False and src['sealed_holdout_listed'] is False and src['sealed_holdout_downloaded'] is False and src['sealed_holdout_opened'] is False
rows=list(p1['records'].values())
for rec in rows:
 assert rec['total_pairs']==120 and len(rec['pairs'])==120 and len(rec['channel_receipts'])==16
 stable=[]
 for q in rec['pairs']:
  if q['stable']:
   kind=q['stable_type']; assert kind in ('SAME','OPPOSITE')
   h1=q['half1'][kind]; h2=q['half2'][kind]
   assert h1['log2_lift']>=0.25 and h2['log2_lift']>=0.25
   assert h1['margin_over_max_null']>=0.10 and h2['margin_over_max_null']>=0.10
   assert h1['joint_events']>=5 and h2['joint_events']>=5
   assert len(h1['null_lifts'])==4 and len(h2['null_lifts'])==4
   stable.append(q)
 assert len(stable)==rec['stable_edge_count']
 expected=sorted(stable,key=lambda p:(-p['min_half_margin'],-p['min_half_lift'],-p['min_half_joint_events'],p['rank_pair_one_based']))[:20]
 assert expected==rec['top20_stable_edges']
recalc={
 'G1_portable_S42_sentinels_all_four':all(r['portable_S42_sentinel']['portable_sentinel']=='PASS' for r in rows),
 'G2_all_16_frozen_channels_reconstruct_all_four':all(len(r['channel_receipts'])==16 and all(q['score_rows']>=100 for q in r['channel_receipts']) for r in rows),
 'G3_at_least_100_support_pairs_each_record':all(r['support_qualified_pairs']>=100 for r in rows),
 'G4_at_least_3_stable_edges_each_record':all(r['stable_edge_count']>=3 for r in rows),
 'G5_stable_edge_fraction_at_least_0_025_each_record':all(r['stable_edge_fraction']>=0.025 for r in rows),
 'G6_at_least_one_SAME_stable_edge_each_record':all(r['stable_same_count']>=1 for r in rows),
 'G7_three_of_four_have_OPPOSITE_stable_edge':sum(r['stable_opposite_count']>=1 for r in rows)>=3,
 'G8_all_stable_edges_half_lift_at_least_0_25':all(q['half1'][q['stable_type']]['log2_lift']>=0.25 and q['half2'][q['stable_type']]['log2_lift']>=0.25 for r in rows for q in r['pairs'] if q['stable']),
 'G9_all_stable_edges_margin_at_least_0_10':all(q['half1'][q['stable_type']]['margin_over_max_null']>=0.10 and q['half2'][q['stable_type']]['margin_over_max_null']>=0.10 for r in rows for q in r['pairs'] if q['stable'])}
assert recalc==p1['checks'] and sum(recalc.values())==p1['passed_scientific_checks']==r1['passed_scientific_checks']
formal=sum(recalc.values())+3; status='PHASE_A_PASS' if formal==12 else 'PHASE_A_FAIL'
out={'schema':'bio-001-s47-independent-audit-v1','status':'PASS','formal_status':status,'formal_passed_checks':formal,'formal_total_checks':12,'scientific_checks':recalc,'G10_exact_rerun':True,'G11_independent_pair_arithmetic_and_ranking':True,'G12_nonbehavior_and_holdout_nonaccess':True,'exact_rerun':'PASS','pair_arithmetic_recalculation':'PASS','ranking_recalculation':'PASS','nonbehavior_sentinel':'PASS','holdout_nonaccess':'PASS','result_sha256':sha(ROOT/'s47_primary/s47_result.json'),'receipt_sha256':sha(ROOT/'s47_primary/s47_run_receipt.json')}
(ROOT/'s47_independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8'); print('BIO001_S47_AUDIT='+json.dumps(out,sort_keys=True))
