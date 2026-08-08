#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PIPELINE = Path('hou_lens_p495_historical_south/source12_exact/run_selected_target_pipeline.py')
OUT = Path('p495_eight_primary_replay_evidence')
PLAN = Path('/tmp/p495-eight-primary-plan.json')

ROWS = [
 {"target_id":"KIDSOBJ-00003","ra_deg":0.050133,"dec_deg":-31.162044,"tile_id":"KIDS_0.0_-31.2","benchmark_tier":"high_quality_candidate","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.525","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.529","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.533","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.537","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.541"}},
 {"target_id":"KIDSOBJ-00004","ra_deg":0.0599,"dec_deg":-28.1933,"tile_id":"KIDS_0.0_-28.2","benchmark_tier":"grade_a_candidate","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.927","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.931","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.935","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.939","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.943"}},
 {"target_id":"KIDSOBJ-00008","ra_deg":0.124587,"dec_deg":-33.523305,"tile_id":"KIDS_0.0_-33.1","benchmark_tier":"grade_a_candidate","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.565","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.569","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.573","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.577","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.581"}},
 {"target_id":"KIDSOBJ-00041","ra_deg":1.322827,"dec_deg":-35.395134,"tile_id":"KIDS_1.2_-35.1","benchmark_tier":"high_quality_candidate","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.827","g":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.831","r":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.835","i1":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.839","i2":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.883"}},
 {"target_id":"KIDSOBJ-00113","ra_deg":4.54318,"dec_deg":-28.935984,"tile_id":"KIDS_4.6_-29.2","benchmark_tier":"high_quality_candidate","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.059","g":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.063","r":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.067","i1":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.071","i2":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.075"}},
 {"target_id":"KIDSCTRL-00003-01","ra_deg":0.45636,"dec_deg":-31.653014,"tile_id":"KIDS_0.0_-31.2","benchmark_tier":"matched_control","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.525","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.529","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.533","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.537","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.541"}},
 {"target_id":"KIDSCTRL-00004-01","ra_deg":0.096259,"dec_deg":-27.957635,"tile_id":"KIDS_0.0_-28.2","benchmark_tier":"matched_control","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.927","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.931","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.935","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.939","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:29.943"}},
 {"target_id":"KIDSCTRL-00008-01","ra_deg":359.956003,"dec_deg":-32.915916,"tile_id":"KIDS_0.0_-33.1","benchmark_tier":"matched_control","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.565","g":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.569","r":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.573","i1":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.577","i2":"ivo://eso.org/ID?ADP.2024-12-16T08:02:33.581"}},
 {"target_id":"KIDSCTRL-00041-01","ra_deg":0.87026,"dec_deg":-35.183434,"tile_id":"KIDS_1.2_-35.1","benchmark_tier":"matched_control","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.827","g":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.831","r":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.835","i1":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.839","i2":"ivo://eso.org/ID?ADP.2024-12-13T18:13:31.883"}},
 {"target_id":"KIDSCTRL-00113-01","ra_deg":4.367493,"dec_deg":-28.788501,"tile_id":"KIDS_4.6_-29.2","benchmark_tier":"matched_control","split":"development","science_dataset_ids":{"u":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.059","g":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.063","r":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.067","i1":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.071","i2":"ivo://eso.org/ID?ADP.2024-12-13T18:08:21.075"}},
]
EXPECTED_RAW={
 "KIDSOBJ-00003":"e1cb3459f65cd10a513ed768b6638514ecf38c70e715a59e095b57ac6c9528d9",
 "KIDSOBJ-00004":"a20f6ce7b9afd8206cb7236792208f98ab5fe33a684f1eec71fcccf2563458fb",
 "KIDSOBJ-00008":"0c30f8f5e1127981837873cc7147ebbe53b28ecc97d67f04677591adf75a6f18",
 "KIDSOBJ-00041":"c6712a135c0d459d56bbb06f72783700ce57f7e62500ab4438d4b80bbf7bc4f2",
 "KIDSOBJ-00113":"993609dc6199c5dad3023297e7f504abce1ac142346b8e6ed67dc4f44185dd8b",
 "KIDSCTRL-00003-01":"4c2d357e3f4245e009f8504d8753058059e57b19bb0fc4fc482df72719c12ee4",
 "KIDSCTRL-00004-01":"fd743de75d0b64c31870f7065abba9d10bb58f21d9adc7580ec44788ab5c55b0",
 "KIDSCTRL-00008-01":"a6d2ca9324600abb820de1e47a8ba4632501e92b1f8deeaefe761c25df35c230",
 "KIDSCTRL-00041-01":"482c47c6686b0e39dcb8a5f9410bb5ba67965208115d84eb54dd1eed5609e540",
 "KIDSCTRL-00113-01":"f8773df3efd1cf7cf614fe42fc0bd73530103f809411032473e4bce657dec8ff",
}
EXPECTED_MODEL={
 "KIDSOBJ-00003":"2f9ede48ee52dd97a1bf86f4ec6eeab6c2603ccaf5863b642c246e87e50abf82",
 "KIDSOBJ-00004":"5efbf2dd0d0dd91d638a00bbf70c9b0889bfe9e0110a4f3832a1308d0bb9eabf",
 "KIDSOBJ-00008":"1846a5b5ba483cb21395f38e850b92aa71ecb6cfc81149351136786e8dd8d3d1",
 "KIDSOBJ-00041":"0ca0fecbb14a74af557dec9ff048f82b7f78cf0e5707e0d75f45b03a1d8b9c99",
 "KIDSOBJ-00113":"2f730fe34557d0b060ed7c528cd643989de4aa18694eccd2e36bcd0afd570d03",
}

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 if not PIPELINE.exists(): raise SystemExit('historical pipeline missing')
 payload={"schema_version":1,"targets":ROWS,"split":"development","holdout_targets_included":0,"blind_search_performed":False,"unknown_targets_queried":False}
 PLAN.write_text(json.dumps(payload,separators=(',',':'))+'\n')
 shutil.rmtree(OUT,ignore_errors=True); OUT.mkdir()
 results=[]
 for i,row in enumerate(ROWS,1):
  tid=row['target_id']; od=OUT/tid
  print(f'[{i}/10] exact replay {tid}',flush=True)
  subprocess.run([sys.executable,str(PIPELINE),'--targets',str(PLAN),'--target-id',tid,'--output',str(od),'--radius','0.01','--attempts','5','--timeout','240'],check=True)
  rec=json.loads((od/'receipt.json').read_text())
  got_raw=rec['output_sha256']['raw_scoring_cube']; got_model=rec['output_sha256']['normalized_model_cube']
  checks={
   'receipt_pass':rec.get('status')=='PASS',
   'raw_historical_sha_exact':got_raw==EXPECTED_RAW[tid],
   'raw_file_sha_exact':sha(od/'psf_matched_ugri_scoring_cube.npz')==got_raw,
   'model_file_sha_exact':sha(od/'normalized_ugri_model_cube.npz')==got_model,
   'five_cutouts':rec.get('cutout_count')==5,
   'shape_raw':rec.get('raw_cube_shape')==[4,360,360],
   'shape_model':rec.get('model_cube_shape')==[4,360,360],
  }
  if tid in EXPECTED_MODEL: checks['model_historical_sha_exact']=got_model==EXPECTED_MODEL[tid]
  if not all(checks.values()): raise SystemExit(json.dumps({'target':tid,'checks':checks,'got_raw':got_raw,'expected_raw':EXPECTED_RAW[tid],'got_model':got_model,'expected_model':EXPECTED_MODEL.get(tid)},indent=2))
  results.append({'target_id':tid,'role':'positive' if tid.startswith('KIDSOBJ') else 'control','raw_sha256':got_raw,'normalized_sha256':got_model,'checks':checks})
  print(f'PASS {tid}',flush=True)
 files=[]
 for p in sorted(x for x in OUT.rglob('*') if x.is_file()): files.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 aggregate={'schema_version':1,'protocol':'HOU-LENS-P4.9.5-EIGHT-STRICT-PRIMARY-PAYLOAD-REPLAY','status':'PASS_10_OF_10_HISTORICAL_RAW_SHA_REPLAY','target_count':10,'positive_count':5,'control_count':5,'historical_raw_sha_matches':10,'historical_positive_normalized_sha_matches':5,'results':results,'file_set':files,'file_set_sha256':hashlib.sha256((json.dumps(files,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest(),'scores_computed':False,'model_code_executed':False,'scenario_generation':False,'future_validation_touched':False}
 (OUT/'aggregate_receipt.json').write_text(json.dumps(aggregate,sort_keys=True,separators=(',',':'))+'\n')
 print(json.dumps({k:aggregate[k] for k in ['status','target_count','historical_raw_sha_matches','historical_positive_normalized_sha_matches','file_set_sha256']},indent=2),flush=True)
if __name__=='__main__': main()
