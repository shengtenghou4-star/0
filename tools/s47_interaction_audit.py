from __future__ import annotations

import hashlib, json, math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

import s42_canonical_reconstruction as canon
import s43_calcium_innovation as s43

ROOT=Path('s47_data/training')
OUT=Path('s47_output'); OUT.mkdir(parents=True,exist_ok=True)
TOP16={
 'BrainScanner20200130_105254':(44,69,51,96,105,20,119,32,123,93,126,60,89,48,49,61),
 'BrainScanner20200130_110803':(39,107,103,111,28,84,89,58,51,27,76,66,77,38,17,62),
 'BrainScanner20200310_141211':(62,66,59,24,81,60,16,18,28,13,49,12,71,75,25,17),
 'BrainScanner20200310_142022':(8,68,36,6,90,18,32,51,39,17,31,47,21,57,0,23)}
S42_REF={
 'BrainScanner20200130_105254':{'channels':128,'frames':1525,'finite_fraction':0.7568698770491803,'valid_frames':1150,'valid_map_sha':'3139a2c8b1d89563f346bcc88121de6f4d31cbc9bf5aba716d4f58e4acffaec6','finite_motion_fits':128,'R_flat':31,'G_flat':48},
 'BrainScanner20200130_110803':{'channels':134,'frames':1466,'finite_fraction':0.9428284905621959,'valid_frames':1433,'valid_map_sha':'957fae480f98bf4ad0d7442bca95d6e49f8671b82eadcde756b5264fdf077efe','finite_motion_fits':134,'R_flat':3,'G_flat':7},
 'BrainScanner20200310_141211':{'channels':116,'frames':1495,'finite_fraction':0.7362933917656557,'valid_frames':1181,'valid_map_sha':'1079f35255b7d9bb2d5a0b191370c2fb1e0ea03347f6371021c8958d25b783bd','finite_motion_fits':116,'R_flat':3,'G_flat':27},
 'BrainScanner20200310_142022':{'channels':97,'frames':1480,'finite_fraction':0.929834215658958,'valid_frames':1396,'valid_map_sha':'bcea5145321ea2eb2bb47e044ae092606d3f543e059408dc994b372f915246b1','finite_motion_fits':97,'R_flat':7,'G_flat':27}}
EPS=1e-12


def write(path:Path,p:Any): path.write_text(json.dumps(p,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')

def dataset_rows():
 canon.ROOT=ROOT
 rows=canon.dataset_rows()
 if len(rows)!=4 or any(c is None for _,c in rows): raise RuntimeError(rows)
 return [(r,int(c)) for r,c in rows]

def neural_only_reconstruct(rid:str,cutoff:int):
 folder=sorted(p for p in ROOT.rglob(f'{rid}_MS') if p.is_dir())[0]; path=folder/'heatDataMS.mat'
 d=loadmat(path,variable_names=['hasPointsTime','rRaw','gRaw','rPhotoCorr','gPhotoCorr','flagged_volumes'])
 t=np.asarray(d['hasPointsTime'],float).squeeze(); rraw=np.asarray(d['rRaw'],float)[:,:len(t)]; graw=np.asarray(d['gRaw'],float)[:,:len(t)]
 rm=np.asarray(d['rPhotoCorr'],float)[:,:len(t)]; gm=np.asarray(d['gPhotoCorr'],float)[:,:len(t)]
 mx=min(int(cutoff),rraw.shape[1]-1); keep=np.arange(rraw.shape[1])<=mx
 R,rfit=canon.correct_photobleaching(rraw[:,keep],canon.VPS); G,gfit=canon.correct_photobleaching(graw[:,keep],canon.VPS)
 R[np.isnan(rm[:,keep])]=np.nan; G[np.isnan(gm[:,keep])]=np.nan; R=canon.close_nan_holes(R); G=canon.close_nan_holes(G)
 flagged=[]
 if 'flagged_volumes' in d and np.asarray(d['flagged_volumes']).size:
  flagged=[int(x) for x in np.asarray(d['flagged_volumes']).ravel() if int(x)<=mx]
  if flagged: R[:,flagged]=np.nan; G[:,flagged]=np.nan
 I,coef=canon.decorrelate_neurons_linear(R,G); valid=np.flatnonzero(np.mean(np.isnan(I),axis=0)<0.5)
 vsha=hashlib.sha256(np.asarray(valid,dtype='<i8').tobytes()).hexdigest(); ref=S42_REF[rid]
 obs={'channels':int(I.shape[0]),'frames':int(I.shape[1]),'finite_fraction':float(np.mean(np.isfinite(I))),'valid_frames':int(valid.size),'valid_map_sha':vsha,'finite_motion_fits':int(np.count_nonzero(np.all(np.isfinite(coef),axis=1))),'R_flat':int(rfit['flat_fallback_neurons']),'G_flat':int(gfit['flat_fallback_neurons'])}
 checks={'channels':obs['channels']==ref['channels'],'frames':obs['frames']==ref['frames'],'finite_fraction':abs(obs['finite_fraction']-ref['finite_fraction'])<=1e-12,'valid_frames':obs['valid_frames']==ref['valid_frames'],'valid_map_sha':obs['valid_map_sha']==ref['valid_map_sha'],'finite_motion_fits':obs['finite_motion_fits']==ref['finite_motion_fits'],'R_flat':obs['R_flat']==ref['R_flat'],'G_flat':obs['G_flat']==ref['G_flat']}
 if not all(checks.values()): raise RuntimeError((rid,checks,obs))
 time=t[:I.shape[1]].copy()
 if valid.size: time-=time[valid[0]]
 quality=np.ones(I.shape[1],dtype=bool)
 for lo,hi in canon.EXCLUDE_INTERVAL.get(rid,[]): quality &= ((time<lo)|(time>hi))
 return I,time,quality,{'portable_sentinel':'PASS','checks':checks,'observed':obs,'reference':ref,'behavior_mat_variable_loaded':False}

def channel_z(I:np.ndarray,quality:np.ndarray,j:int,warm_end:int,score_start:int):
 ref=s43.select_and_score(I[j],quality,warm_end,score_start)
 if ref is None: raise RuntimeError(f'inadmissible frozen channel {j}')
 X,y,idx=s43.common_ar6_rows(I[j],quality); warm=idx<=warm_end; score=idx>=score_start
 lags=tuple(ref['selected_lags']); cols=[s43.FULL_LAGS.index(l) for l in lags]; Xw=X[warm][:,cols]; yw=y[warm]
 xm=np.median(Xw,axis=0); xmad=np.median(np.abs(Xw-xm),axis=0); xsc=1.4826*xmad; xsc[~np.isfinite(xsc)|(xsc<EPS)]=1.0
 ym,ys=s43.robust_center_scale(yw); beta=s43.ridge_fit((Xw-xm)/xsc,(yw-ym)/ys,float(ref['selected_alpha'])); pred=s43.predict(beta,(X[:,cols]-xm)/xsc)*ys+ym; innov=y-pred
 med,scale=s43.robust_center_scale(innov[warm]); z=(innov-med)/scale
 dense=np.full(I.shape[1],np.nan); dense[idx[score]]=z[score]
 return dense,{'channel_zero_based':j,'selected_family':ref['selected_family'],'selected_alpha':float(ref['selected_alpha']),'score_rows':int(score.sum()),'warm_scale':float(scale)}

def states(z:np.ndarray):
 a=np.zeros(len(z),dtype=np.int8); a[z>=2]=1; a[z<=-2]=-1; return a

def lift(a:np.ndarray,b:np.ndarray,kind:str):
 n=len(a); pa_pos=np.mean(a==1); pa_neg=np.mean(a==-1); pb_pos=np.mean(b==1); pb_neg=np.mean(b==-1)
 if kind=='SAME': mask=((a==1)&(b==1))|((a==-1)&(b==-1)); exp=pa_pos*pb_pos+pa_neg*pb_neg
 else: mask=((a==1)&(b==-1))|((a==-1)&(b==1)); exp=pa_pos*pb_neg+pa_neg*pb_pos
 cnt=int(mask.sum()); rate=cnt/max(n,1); smooth=0.5/max(n,1); val=math.log2((rate+smooth)/(exp+smooth))
 return {'n':n,'joint_events':cnt,'rate':float(rate),'expected':float(exp),'log2_lift':float(val)}

def half_metrics(a:np.ndarray,b:np.ndarray):
 out={}
 for kind in ('SAME','OPPOSITE'):
  real=lift(a,b,kind); null=[]; n=len(a)
  for frac in (0.2,0.4,0.6,0.8):
   sh=max(1,int(math.floor(n*frac)))%n
   if sh==0: sh=1
   null.append(lift(a,np.roll(b,sh),kind)['log2_lift'])
  real['null_lifts']=[float(x) for x in null]; real['max_null_lift']=float(max(null)); real['margin_over_max_null']=float(real['log2_lift']-max(null)); out[kind]=real
 return out

def pair_metric(za:np.ndarray,zb:np.ndarray,ra:int,rb:int,ca:int,cb:int):
 common=np.flatnonzero(np.isfinite(za)&np.isfinite(zb)); n=len(common)
 base={'rank_pair_one_based':[ra+1,rb+1],'channel_pair_zero_based':[ca,cb],'common_score_samples':n,'support_ok':False,'stable':False,'stable_type':None}
 if n<200: return base
 a=states(za[common]); b=states(zb[common]); cut=n//2
 if cut<80 or n-cut<80: return base
 h1=half_metrics(a[:cut],b[:cut]); h2=half_metrics(a[cut:],b[cut:]); base.update({'support_ok':True,'half1':h1,'half2':h2})
 passing=[]
 for kind in ('SAME','OPPOSITE'):
  x,y=h1[kind],h2[kind]
  ok=x['log2_lift']>=0.25 and y['log2_lift']>=0.25 and x['margin_over_max_null']>=0.10 and y['margin_over_max_null']>=0.10 and x['joint_events']>=5 and y['joint_events']>=5
  if ok: passing.append((min(x['margin_over_max_null'],y['margin_over_max_null']),min(x['log2_lift'],y['log2_lift']),kind))
 if passing:
  passing.sort(key=lambda q:(-q[0],-q[1],0 if q[2]=='SAME' else 1)); _,_,kind=passing[0]; base['stable']=True; base['stable_type']=kind; base['min_half_margin']=float(min(h1[kind]['margin_over_max_null'],h2[kind]['margin_over_max_null'])); base['min_half_lift']=float(min(h1[kind]['log2_lift'],h2[kind]['log2_lift'])); base['min_half_joint_events']=int(min(h1[kind]['joint_events'],h2[kind]['joint_events']))
 return base

def analyze_record(rid:str,cutoff:int):
 I,time,quality,sentinel=neural_only_reconstruct(rid,cutoff); warm_end=int(np.floor(0.40*(I.shape[1]-1))); score_start=warm_end+24
 zs=[]; receipts=[]
 for j in TOP16[rid]:
  z,r=channel_z(I,quality,j,warm_end,score_start); zs.append(z); receipts.append(r)
 pairs=[]
 for a in range(16):
  for b in range(a+1,16): pairs.append(pair_metric(zs[a],zs[b],a,b,TOP16[rid][a],TOP16[rid][b]))
 stable=[p for p in pairs if p['stable']]; stable.sort(key=lambda p:(-p['min_half_margin'],-p['min_half_lift'],-p['min_half_joint_events'],p['rank_pair_one_based']))
 return {'record_id':rid,'portable_S42_sentinel':sentinel,'channel_receipts':receipts,'total_pairs':120,'support_qualified_pairs':sum(p['support_ok'] for p in pairs),'stable_edge_count':len(stable),'stable_edge_fraction':len(stable)/120.0,'stable_same_count':sum(p.get('stable_type')=='SAME' for p in stable),'stable_opposite_count':sum(p.get('stable_type')=='OPPOSITE' for p in stable),'stable_min_margin':float(min((p['min_half_margin'] for p in stable),default=0.0)),'stable_median_margin':float(np.median([p['min_half_margin'] for p in stable])) if stable else 0.0,'top20_stable_edges':stable[:20],'pairs':pairs}

def main():
 records={rid:analyze_record(rid,c) for rid,c in dataset_rows()}; rows=list(records.values())
 checks={
  'G1_portable_S42_sentinels_all_four':all(r['portable_S42_sentinel']['portable_sentinel']=='PASS' for r in rows),
  'G2_all_16_frozen_channels_reconstruct_all_four':all(len(r['channel_receipts'])==16 and all(q['score_rows']>=100 for q in r['channel_receipts']) for r in rows),
  'G3_at_least_100_support_pairs_each_record':all(r['support_qualified_pairs']>=100 for r in rows),
  'G4_at_least_3_stable_edges_each_record':all(r['stable_edge_count']>=3 for r in rows),
  'G5_stable_edge_fraction_at_least_0_025_each_record':all(r['stable_edge_fraction']>=0.025 for r in rows),
  'G6_at_least_one_SAME_stable_edge_each_record':all(r['stable_same_count']>=1 for r in rows),
  'G7_three_of_four_have_OPPOSITE_stable_edge':sum(r['stable_opposite_count']>=1 for r in rows)>=3,
  'G8_all_stable_edges_half_lift_at_least_0_25':all(p['half1'][p['stable_type']]['log2_lift']>=0.25 and p['half2'][p['stable_type']]['log2_lift']>=0.25 for r in rows for p in r['pairs'] if p['stable']),
  'G9_all_stable_edges_margin_at_least_0_10':all(p['half1'][p['stable_type']]['margin_over_max_null']>=0.10 and p['half2'][p['stable_type']]['margin_over_max_null']>=0.10 for r in rows for p in r['pairs'] if p['stable'])}
 result={'schema':'bio-001-s47-result-v1','status_pre_external_audit':'PHASE_A_PASS_CANDIDATE' if all(checks.values()) else 'PHASE_A_FAIL','passed_scientific_checks':int(sum(checks.values())),'total_scientific_checks':9,'checks':checks,'records':records,'behavior_variables_loaded':False,'behavior_variables_used':False,'behavioral_prediction_built':False,'CeRSI_v2_delta':0.0,'AML310_transition_accessed':False,'AML32_chip_opened':False,'audit_gates_G10_G11_G12_pending':True}
 write(OUT/'s47_result.json',result); receipt={'schema':'bio-001-s47-run-receipt-v1','status_pre_external_audit':result['status_pre_external_audit'],'passed_scientific_checks':result['passed_scientific_checks'],'total_scientific_checks':9,'CeRSI_v2_delta':0.0,'AML32_chip_opened':False}; write(OUT/'s47_run_receipt.json',receipt); print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__': main()
