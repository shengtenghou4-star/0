from __future__ import annotations

import hashlib,io,json,math
from pathlib import Path
import numpy as np
from scipy.signal import periodogram

ROOT=Path('s48_target/extracted'); BAND=(0.10,5.00); MODEL={'0.4.0':{'frequency_ratio':4.333333333333333,'amplitude_ratio':0.41444955584195425,'spatial_cycles_ratio':0.9772152041902191,'rhythmicity_ratio':0.38068856588497335},'0.5.0':{'frequency_ratio':4.0,'amplitude_ratio':0.30467058711464157,'spatial_cycles_ratio':1.0456442795867726,'rhythmicity_ratio':1.0205503687967137}}; EP={'frequency_ratio':'dominant_frequency_hz','amplitude_ratio':'rms_angle_amplitude','spatial_cycles_ratio':'spatial_cycles_per_body','rhythmicity_ratio':'rhythmicity'}

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def folder(name):
 h=[p for p in ROOT.rglob('*') if p.is_dir() and p.name.lower()==name.lower()]; assert len(h)==1; return h[0]

def parse2(p:Path):
 lines=p.read_text(encoding='utf-8',errors='replace').splitlines(); st=next(i for i,s in enumerate(lines) if s.lstrip().lower().startswith('time')); rows=[]
 for s in lines[st+1:]:
  if not s.strip(): continue
  vals=[float(z) for z in s.replace(',',' ').split()]
  if len(vals)<11: raise ValueError('cols')
  rows.append(vals[:11])
 a=np.asarray(rows,float); m=np.all(np.isfinite(a),axis=1); a=a[m]; order=np.argsort(a[:,0],kind='stable'); a=a[order]; keep=np.r_[True,np.diff(a[:,0])!=0]; a=a[keep]
 if len(a)<200 or a[-1,0]-a[0,0]<10: raise ValueError('support')
 t=a[:,0]; x=a[:,1:11]; d=np.diff(t); pos=d[d>0]; dt=float(np.median(pos)); assert np.mean(pos<=5*dt)>=.95
 grid=np.arange(t[0],t[-1]+.5*dt,dt); xx=np.column_stack([np.interp(grid,t,x[:,j]) for j in range(10)]); xx-=np.median(xx,axis=0,keepdims=True); return grid,xx

def ep2(t,x):
 dt=float(np.median(np.diff(t))); fs=1/dt; mats=[]
 for j in range(2,8):
  f,p=periodogram(x[:,j],fs=fs,window='hann',detrend='constant',scaling='spectrum'); m=(f>=.10)&(f<=5.0); q=p[m]; mats.append(q/q.sum() if q.sum()>0 else np.zeros_like(q)); fb=f[m]
 av=np.mean(np.asarray(mats),axis=0); ii=int(np.flatnonzero(av==av.max())[0]); f0=float(fb[ii]); amp=float(np.sqrt(np.mean(np.square(x)))); n=len(t); w=np.hanning(n); freq=np.fft.rfftfreq(n,d=dt); kk=int(np.argmin(np.abs(freq-f0))); ph=np.unwrap(np.angle([np.fft.rfft(x[:,j]*w)[kk] for j in range(10)])); q=np.arange(10,dtype=float); A=np.column_stack([q,np.ones(10)]); beta=np.linalg.lstsq(A,ph,rcond=None)[0]; pred=A@beta; sst=float(np.sum((ph-ph.mean())**2)); r2=1-float(np.sum((ph-pred)**2))/sst if sst>0 else 0.; spatial=float(abs(beta[0])*9/(2*math.pi)); width=max(.10,.2*f0); near=(fb>=max(.10,f0-width))&(fb<=min(5.0,f0+width)); rhythmic=float(av[near].sum()/av.sum()) if av.sum()>0 else 0.; return {'dominant_frequency_hz':f0,'rms_angle_amplitude':amp,'spatial_cycles_per_body':spatial,'spatial_phase_r2':r2,'rhythmicity':rhythmic}

p1=json.loads(Path('s48_primary/s48_result.json').read_text()); p2=json.loads(Path('s48_rerun/s48_result.json').read_text()); r1=json.loads(Path('s48_primary/s48_run_receipt.json').read_text()); r2=json.loads(Path('s48_rerun/s48_run_receipt.json').read_text()); arch=json.loads(Path('s48_target/archive_manifest.json').read_text())
assert p1==p2 and r1==r2 and sha(Path('s48_primary/s48_result.json'))==sha(Path('s48_rerun/s48_result.json'))
assert p1['frozen_model_ratios']==MODEL and p1['archive_manifest_sha256']==arch['sha256']
assert p1['neural_branch_accessed'] is False and p1['AML310_transition_accessed'] is False and p1['AML32_chip_opened'] is False
# Independently recalc every included worm endpoint.
manifest={row['path']:row for row in p1['inclusion_manifest']}
vals={g:{k:[] for k in EP} for g in ('crawl','swim')}
for gait,name in [('crawl','N2_A1_crawling'),('swim','N2_A1_swimming')]:
 for f in sorted(folder(name).rglob('*.txt')):
  key=str(f.relative_to(ROOT)); row=manifest[key]
  try:
   t,x=parse2(f); e=ep2(t,x); assert row['included_parser'] is True
   for metric in ('dominant_frequency_hz','rms_angle_amplitude','spatial_cycles_per_body','spatial_phase_r2','rhythmicity'):
    assert abs(float(e[metric])-float(row['endpoints'][metric]))<=1e-9*max(1,abs(float(e[metric])))
   for k,m in EP.items():
    if k!='spatial_cycles_ratio' or e['spatial_phase_r2']>=.25: vals[gait][k].append(float(e[m]))
  except Exception:
   assert row['included_parser'] is False
# Independently recalc target medians and scoring, bootstrap excluded because it is reporting-only.
credits=[]; eligible_count=0; pos=0
for k in EP:
 c=np.asarray(vals['crawl'][k]); s=np.asarray(vals['swim'][k]); eligible=len(c)>=5 and len(s)>=5 and np.median(c)>0
 assert bool(p1['targets'][k]['eligible'])==bool(eligible)
 if eligible:
  target=float(np.median(s)/np.median(c)); assert abs(target-float(p1['targets'][k]['target_swim_to_crawl_ratio']))<=1e-12*max(1,abs(target)); C=MODEL['0.4.0'][k]; M=MODEL['0.5.0'][k]; E0=abs(C-target); E1=abs(M-target); cr=0. if E0<=1e-12 else float(np.clip(1-E1/E0,0,1)); eligible_count+=1
 else: cr=0.
 assert abs(cr-float(p1['scores'][k]['credit']))<=1e-12; credits.append(cr); pos+=int(cr>0)
total=float(sum(credits)); assert abs(total-float(p1['total_CeRSI_credit']))<=1e-12 and eligible_count==p1['eligible_endpoints'] and pos==p1['positive_credit_endpoints']
no_big=all((not p1['targets'][k]['eligible']) or p1['scores'][k]['comparator_error']<=1e-12 or p1['scores'][k]['candidate_error']<=1.25*p1['scores'][k]['comparator_error'] for k in EP)
if eligible_count==4 and pos>=3 and total>=2 and no_big: label='RAW_TRAJECTORY_EXTERNAL_PASS'
elif eligible_count>=2 and total>0: label='RAW_TRAJECTORY_EXTERNAL_PARTIAL'
else: label='RAW_TRAJECTORY_EXTERNAL_FAIL'
assert label==p1['validation_label']==r1['validation_label']
out={'schema':'bio-001-s48-independent-audit-v1','status':'PASS','validation_label':label,'exact_rerun':'PASS','independent_raw_parser_and_E1_E4':'PASS','target_and_score_recalculation':'PASS','archive_hash_sentinel':'PASS','frozen_model_endpoint_sentinel':'PASS','neural_and_holdout_nonaccess':'PASS','total_CeRSI_credit':total,'result_sha256':sha(Path('s48_primary/s48_result.json')),'receipt_sha256':sha(Path('s48_primary/s48_run_receipt.json'))}
Path('s48_independent_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8'); print('BIO001_S48_AUDIT='+json.dumps(out,sort_keys=True))
