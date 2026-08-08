#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
PIPELINE=Path('hou_lens_p495_historical_south/source12_exact/run_selected_target_pipeline.py')
OUT=Path('p495_h24_primary_replay_evidence'); PLAN=Path('/tmp/p495-h24-primary-plan.json')
ROWS=[
{"target_id":"H24GOLD-CXCOJ100201+020330","ra_deg":150.5063,"dec_deg":2.0581,"tile_id":"KIDS_150.1_2.2","benchmark_tier":"external_confirmed_lensed_quasar","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.399","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.403","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.407","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.411","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.415"}},
{"target_id":"H24GOLD-J0907+0003","ra_deg":136.7937,"dec_deg":0.0559,"tile_id":"KIDS_137.0_0.5","benchmark_tier":"external_confirmed_lensed_quasar","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:30.779","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:30.783","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:30.787","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:30.791","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:30.795"}},
{"target_id":"H24GOLD-J1037+0018","ra_deg":159.3665,"dec_deg":0.3057,"tile_id":"KIDS_159.0_0.5","benchmark_tier":"external_confirmed_lensed_quasar","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.663","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.667","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.671","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.675","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.679"}},
{"target_id":"H24GOLD-J1233-0227","ra_deg":188.4219,"dec_deg":-2.4604,"tile_id":"KIDS_188.5_-2.5","benchmark_tier":"external_confirmed_lensed_quasar","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-13T18:13:34.937","g":"ivo://eso.org/ID?ADP.2024-12-13T18:13:35.546","r":"ivo://eso.org/ID?ADP.2024-12-13T18:13:34.941","i1":"ivo://eso.org/ID?ADP.2024-12-13T18:13:34.945","i2":"ivo://eso.org/ID?ADP.2024-12-13T18:13:34.949"}},
{"target_id":"H24GOLD-J1335+0118","ra_deg":203.894979,"dec_deg":1.301553,"tile_id":"KIDS_204.0_1.5","benchmark_tier":"external_confirmed_lensed_quasar","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-13T18:13:33.347","g":"ivo://eso.org/ID?ADP.2024-12-13T18:13:33.351","r":"ivo://eso.org/ID?ADP.2024-12-13T18:13:33.355","i1":"ivo://eso.org/ID?ADP.2024-12-13T18:13:33.359","i2":"ivo://eso.org/ID?ADP.2024-12-13T18:13:33.363"}},
{"target_id":"H24GOLD-KIDS1042+0023","ra_deg":160.6553,"dec_deg":0.3839,"tile_id":"KIDS_161.0_0.5","benchmark_tier":"external_confirmed_lensed_quasar","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.743","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.747","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.751","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.767","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:31.771"}}
]
EXPECTED={"H24GOLD-CXCOJ100201+020330":"ae3b75c15cd35275b58569f081a6b78c7738fdb4181c32017e20a606740fbb1b","H24GOLD-J0907+0003":"62b6887d6eb75e543bddc46ab9213bf59663275cb039509e57f91db1731d1e25","H24GOLD-J1037+0018":"01c5ae295fc18ebff4623a8c14c89a3be7700444b983e8cb34f3167a5c71271a","H24GOLD-J1233-0227":"9711189ea09541c26d240dcfea872a1f6fd6543ca49bde4508fbc7d9a5d022ca","H24GOLD-J1335+0118":"f5f59b212711d41124d8f92504fbbea67d089e610abde18fe7c2f11ea7e524b9","H24GOLD-KIDS1042+0023":"55dc40cc6815eaebc7565ed52f0d3ebb602c5331c9e07ef12603662639280256"}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 PLAN.write_text(json.dumps({"schema_version":1,"targets":ROWS,"split":"development","holdout_targets_included":0},separators=(',',':'))+'\n')
 shutil.rmtree(OUT,ignore_errors=True); OUT.mkdir()
 results=[]
 for i,row in enumerate(ROWS,1):
  tid=row['target_id']; od=OUT/tid; print(f'[{i}/6] exact replay {tid}',flush=True)
  subprocess.run([sys.executable,str(PIPELINE),'--targets',str(PLAN),'--target-id',tid,'--output',str(od),'--radius','0.01','--attempts','5','--timeout','240'],check=True)
  rec=json.loads((od/'receipt.json').read_text()); raw=rec['output_sha256']['raw_scoring_cube']; model=rec['output_sha256']['normalized_model_cube']
  checks={'receipt_pass':rec.get('status')=='PASS','raw_historical_sha_exact':raw==EXPECTED[tid],'raw_file_sha_exact':sha(od/'psf_matched_ugri_scoring_cube.npz')==raw,'model_file_sha_exact':sha(od/'normalized_ugri_model_cube.npz')==model,'five_cutouts':rec.get('cutout_count')==5,'shape_raw':rec.get('raw_cube_shape')==[4,360,360],'shape_model':rec.get('model_cube_shape')==[4,360,360]}
  if not all(checks.values()): raise SystemExit(json.dumps({'target':tid,'checks':checks,'got_raw':raw,'expected_raw':EXPECTED[tid]},indent=2))
  results.append({'target_id':tid,'raw_sha256':raw,'normalized_sha256':model,'checks':checks}); print(f'PASS {tid}',flush=True)
 files=[]
 for p in sorted(x for x in OUT.rglob('*') if x.is_file()): files.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 agg={'schema_version':1,'protocol':'HOU-LENS-P4.9.5-H24-PRIMARY-PAYLOAD-REPLAY','status':'PASS_6_OF_6_HISTORICAL_RAW_SHA_REPLAY','target_count':6,'historical_raw_sha_matches':6,'results':results,'file_set':files,'file_set_sha256':hashlib.sha256((json.dumps(files,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest(),'scores_computed':False,'model_code_executed':False,'scenario_generation':False,'future_validation_touched':False}
 (OUT/'aggregate_receipt.json').write_text(json.dumps(agg,sort_keys=True,separators=(',',':'))+'\n'); print(json.dumps({k:agg[k] for k in ['status','target_count','historical_raw_sha_matches','file_set_sha256']},indent=2),flush=True)
if __name__=='__main__': main()
