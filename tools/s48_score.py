from __future__ import annotations

import hashlib, json
from pathlib import Path
from typing import Any

import numpy as np

from s48_target_common import parse_worm,endpoints

ROOT=Path('s48_target/extracted')
OUT=Path('s48_output'); OUT.mkdir(parents=True,exist_ok=True)
SEED=480048; BOOT=10000
MODEL={
 '0.4.0':{'frequency_ratio':4.333333333333333,'amplitude_ratio':0.41444955584195425,'spatial_cycles_ratio':0.9772152041902191,'rhythmicity_ratio':0.38068856588497335},
 '0.5.0':{'frequency_ratio':4.0,'amplitude_ratio':0.30467058711464157,'spatial_cycles_ratio':1.0456442795867726,'rhythmicity_ratio':1.0205503687967137}}
EP={'frequency_ratio':'dominant_frequency_hz','amplitude_ratio':'rms_angle_amplitude','spatial_cycles_ratio':'spatial_cycles_per_body','rhythmicity_ratio':'rhythmicity'}
FOLDERS={'crawl':'N2_A1_crawling','swim':'N2_A1_swimming'}


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def find_folder(name:str)->Path:
 hits=[p for p in ROOT.rglob('*') if p.is_dir() and p.name.lower()==name.lower()]
 if len(hits)!=1: raise RuntimeError((name,[str(x) for x in hits]))
 return hits[0]

def percentile_ci(x:np.ndarray)->list[float]:
 return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]

def main()->None:
 manifest=[]; gait_values={g:{k:[] for k in EP} for g in FOLDERS}; gait_files={g:{k:[] for k in EP} for g in FOLDERS}
 for gait,folder_name in FOLDERS.items():
  folder=find_folder(folder_name); files=sorted(p for p in folder.rglob('*.txt') if p.is_file())
  if not files: raise RuntimeError(f'no txt {folder_name}')
  for path in files:
   row={'gait':gait,'path':str(path.relative_to(ROOT)),'file_sha256':sha(path)}
   try:
    t,x,pre=parse_worm(path); e=endpoints(t,x); row.update({'included_parser':True,'parser':pre,'endpoints':e})
    for ratio_name,metric in EP.items():
     eligible=True
     if ratio_name=='spatial_cycles_ratio' and float(e['spatial_phase_r2'])<0.25: eligible=False
     if eligible and np.isfinite(float(e[metric])):
      gait_values[gait][ratio_name].append(float(e[metric])); gait_files[gait][ratio_name].append(str(path.relative_to(ROOT)))
    row['spatial_endpoint_eligible']=bool(float(e['spatial_phase_r2'])>=0.25)
   except Exception as exc:
    row.update({'included_parser':False,'exclude_reason':str(exc)})
   manifest.append(row)
 rng=np.random.default_rng(SEED); targets={}; scores={}; credits=[]
 for ratio_name,metric in EP.items():
  c=np.asarray(gait_values['crawl'][ratio_name],float); s=np.asarray(gait_values['swim'][ratio_name],float)
  eligible=len(c)>=5 and len(s)>=5 and np.all(np.isfinite(c)) and np.all(np.isfinite(s)) and float(np.median(c))>0
  target=None; boot_ci=None; gait_ci={}
  if eligible:
   cm=float(np.median(c)); sm=float(np.median(s)); target=sm/cm
   bc=np.empty(BOOT); bs=np.empty(BOOT); br=np.empty(BOOT)
   for i in range(BOOT):
    mc=float(np.median(c[rng.integers(0,len(c),len(c))])); ms=float(np.median(s[rng.integers(0,len(s),len(s))])); bc[i]=mc; bs[i]=ms; br[i]=ms/mc if mc>0 else np.nan
   br=br[np.isfinite(br)]
   gait_ci={'crawl_median_95ci':percentile_ci(bc),'swim_median_95ci':percentile_ci(bs)}; boot_ci=percentile_ci(br)
   C=float(MODEL['0.4.0'][ratio_name]); M=float(MODEL['0.5.0'][ratio_name]); E0=abs(C-target); E1=abs(M-target); credit=0.0 if E0<=1e-12 else float(np.clip(1.0-E1/E0,0.0,1.0))
  else:
   cm=float(np.median(c)) if len(c) else None; sm=float(np.median(s)) if len(s) else None; C=float(MODEL['0.4.0'][ratio_name]); M=float(MODEL['0.5.0'][ratio_name]); E0=E1=None; credit=0.0
  targets[ratio_name]={'eligible':bool(eligible),'crawl_n':int(len(c)),'swim_n':int(len(s)),'crawl_median':cm,'swim_median':sm,'target_swim_to_crawl_ratio':target,'ratio_95ci':boot_ci,'gait_median_95ci':gait_ci,'crawl_files':gait_files['crawl'][ratio_name],'swim_files':gait_files['swim'][ratio_name]}
  scores[ratio_name]={'comparator_0_4':C,'candidate_0_5':M,'target':target,'comparator_error':E0,'candidate_error':E1,'credit':credit}; credits.append(credit)
 total=float(sum(credits)); elig=sum(int(v['eligible']) for v in targets.values()); pos=sum(int(v['credit']>0) for v in scores.values()); no_big_degrade=all((not targets[k]['eligible']) or scores[k]['comparator_error']<=1e-12 or scores[k]['candidate_error']<=1.25*scores[k]['comparator_error'] for k in EP)
 if elig==4 and pos>=3 and total>=2.0 and no_big_degrade: label='RAW_TRAJECTORY_EXTERNAL_PASS'
 elif elig>=2 and total>0: label='RAW_TRAJECTORY_EXTERNAL_PARTIAL'
 else: label='RAW_TRAJECTORY_EXTERNAL_FAIL'
 result={'schema':'bio-001-s48-result-v1','source_doi':'10.5061/dryad.stqjq2c8p','archive_manifest_sha256':json.loads(Path('s48_target/archive_manifest.json').read_text())['sha256'],'folders':FOLDERS,'inclusion_manifest':manifest,'targets':targets,'frozen_model_ratios':MODEL,'scores':scores,'total_CeRSI_credit':total,'validation_label':label,'eligible_endpoints':elig,'positive_credit_endpoints':pos,'no_endpoint_degradation_gt_25pct':bool(no_big_degrade),'CeRSI_v2_previous':5.05777447992075,'CeRSI_v2_current':5.05777447992075+total,'neural_branch_accessed':False,'AML310_transition_accessed':False,'AML32_chip_opened':False}
 Path(OUT/'s48_result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
 receipt={'schema':'bio-001-s48-run-receipt-v1','validation_label':label,'eligible_endpoints':elig,'positive_credit_endpoints':pos,'total_CeRSI_credit':total,'CeRSI_v2_current':result['CeRSI_v2_current'],'AML32_chip_opened':False}; Path(OUT/'s48_run_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8'); print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__': main()
