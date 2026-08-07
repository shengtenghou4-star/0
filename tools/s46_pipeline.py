from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

import s44_pipeline as s44
import s45_phaseA_portable as s45p

ROOT=Path('s46_data/training')
OUT=Path('s46_output')
OUT.mkdir(parents=True,exist_ok=True)
FROZEN_TOP16={
 'BrainScanner20200130_105254':(44,69,51,96,105,20,119,32,123,93,126,60,89,48,49,61),
 'BrainScanner20200130_110803':(39,107,103,111,28,84,89,58,51,27,76,66,77,38,17,62),
 'BrainScanner20200310_141211':(62,66,59,24,81,60,16,18,28,13,49,12,71,75,25,17),
 'BrainScanner20200310_142022':(8,68,36,6,90,18,32,51,39,17,31,47,21,57,0,23)}
READOUT_TO_K={'K01':1,'K02':2,'K04':4,'K08':8,'K16':16}
ALLOWED_K=(1,2,4,8,16)
LAST_CALCIUM_RECEIPTS:dict[str,Any]={}


def sparse_calcium(I:np.ndarray,quality:np.ndarray,warm_end:int,score_start:int,channels:tuple[int,...])->tuple[np.ndarray,dict[str,Any]]:
    dense=np.zeros((16,I.shape[1]),dtype=float); receipts=[]; counts={'AR1':0,'AR3':0,'AR6':0}
    for rank,j in enumerate(channels):
        if not 0<=j<I.shape[0]: raise RuntimeError(f'frozen channel out of range {j}')
        ref=s44.s43.select_and_score(I[j],quality,warm_end,score_start)
        if ref is None: raise RuntimeError(f'frozen S45 channel inadmissible {j}')
        Xall,yall,idx=s44.s43.common_ar6_rows(I[j],quality); warm=idx<=warm_end; score=idx>=score_start
        family=ref['selected_family']; lags=tuple(ref['selected_lags']); alpha=float(ref['selected_alpha']); cols=[s44.s43.FULL_LAGS.index(l) for l in lags]
        Xw=Xall[warm][:,cols]; yw=yall[warm]; xm=np.median(Xw,axis=0); xmad=np.median(np.abs(Xw-xm),axis=0); xsc=1.4826*xmad
        xsc[~np.isfinite(xsc)|(xsc<s44.EPS)]=1.0; ym,ysc=s44.s43.robust_center_scale(yw)
        if ysc<s44.EPS: raise RuntimeError('degenerate frozen sparse channel')
        beta=s44.s43.ridge_fit((Xw-xm)/xsc,(yw-ym)/ysc,alpha); pred=s44.s43.predict(beta,(Xall[:,cols]-xm)/xsc)*ysc+ym; innovation=yall-pred
        if not np.array_equal(idx[score],ref['score_idx']): raise RuntimeError('S43 score support drift')
        if not np.allclose(innovation[score],ref['innovation'],rtol=0,atol=1e-12): raise RuntimeError('S43 innovation drift')
        med,scale=s44.robust_center_scale(innovation[warm]);
        if scale<s44.EPS: raise RuntimeError('innovation warm scale degenerate')
        dense[rank,idx]=(innovation-med)/scale; counts[family]+=1
        receipts.append({'rank_one_based':rank+1,'channel_zero_based':int(j),'selected_family':family,'selected_alpha':alpha,'warm_rows':int(warm.sum()),'score_rows':int(score.sum()),'warm_innovation_scale':float(scale)})
    return dense,{'frozen_sparse_channels_zero_based':list(channels),'selected_family_counts':counts,'channel_receipts':receipts}


def load_record(rid:str,cutoff:int|None)->s44.Record:
    s45p.ROOT=ROOT; s44.canon.ROOT=ROOT; s44.s43.ROOT=ROOT
    I,time,quality,portable=s45p.reconstruct_portable(rid,int(cutoff)); n=I.shape[1]; warm_end=int(np.floor(s44.WARM_FRAC*(n-1))); score_start=warm_end+s44.EMBARGO
    sparse,receipt=sparse_calcium(I,quality,warm_end,score_start,FROZEN_TOP16[rid]); receipt['S42_portable_sentinel']=portable; LAST_CALCIUM_RECEIPTS[rid]=receipt
    folder=sorted(p for p in ROOT.rglob(f'{rid}_MS') if p.is_dir())[0]; heat=loadmat(folder/'heatDataMS.mat'); behavior=heat['behavior'][0,0]
    v=np.asarray(behavior['v'],float)[:n,0]; k=s44.centerline_curvature_metric(folder,heat)[:n]
    if np.any(~np.isfinite(v)):
        good=np.isfinite(v)
        if not np.any(good): raise RuntimeError('no finite velocity')
        v[~good]=np.interp(np.flatnonzero(~good),np.flatnonzero(good),v[good])
    return s44.Record(rid,np.arange(n,dtype=int),v,k,warm_end,quality,sparse,np.ones(n,dtype=float),receipt)


def load_records()->list[s44.Record]:
    s44.canon.ROOT=ROOT; rows=s44.canon.dataset_rows()
    if set(r for r,_ in rows)!=set(FROZEN_TOP16): raise RuntimeError('record set drift')
    return [load_record(r,c) for r,c in rows]


def sparse_features(rec:s44.Record,source_matrix:np.ndarray,readout:str)->np.ndarray:
    k=READOUT_TO_K[readout]; base=source_matrix[:k].T; parts=[]
    for lag in s44.NEURAL_LAGS:
        shifted=np.zeros_like(base); shifted[lag:]=base[:len(base)-lag] if lag else base; parts.append(shifted)
    return np.column_stack(parts)


def cfg_key(cfg:dict[str,Any])->tuple:
    if cfg['off']: return (0,0,float(cfg['alpha']))
    return (1,READOUT_TO_K[str(cfg['readout'])],float(cfg['alpha']))


def main()->None:
    s44.ROOT=ROOT; s44.OUT=OUT; s44.READOUTS=tuple(READOUT_TO_K); s44.load_records=load_records; s44.readout_features=sparse_features; s44.cfg_key=cfg_key
    # S44 source transforms, behavior states, delayed feedback, nested selection and twelve gates are intentionally reused unchanged.
    s44.main()
    src=OUT/'s44_training_result.json'; rr=OUT/'s44_run_receipt.json'; result=json.loads(src.read_text()); receipt=json.loads(rr.read_text())
    for rid,row in result['outer'].items():
        for ep,erow in row['endpoints'].items():
            for source,srow in erow['sources'].items():
                cfg=srow['selected_config']; cfg['k']=0 if cfg['off'] else READOUT_TO_K[cfg['readout']]
    if result['status']=='TRAINING_PASS': result['verdict']='FROZEN_SPARSE_BEHAVIORAL_SPECIFICITY_TRAINING_PASS'
    elif result['verdict']=='NON_SPECIFIC_CANONICAL_INNOVATION_SIGNAL': result['verdict']='NON_SPECIFIC_FROZEN_SPARSE_SIGNAL'
    elif result['verdict']!='INSUFFICIENT_EVENT_SUPPORT': result['verdict']='NO_FROZEN_SPARSE_BEHAVIORAL_SPECIFICITY'
    receipt['verdict']=result['verdict']; result['schema']='bio-001-s46-result-v1'; receipt['schema']='bio-001-s46-run-receipt-v1'
    result['frozen_sparse_top16_zero_based']={k:list(v) for k,v in FROZEN_TOP16.items()}; result['allowed_k']=list(ALLOWED_K); result['S45_sparse_authority']='afb56c7b84ef56b8a75663a0567e1ecdaf1e350e'; result['S42_portable_sentinel']='PASS'; result['sparse_calcium_receipts']=LAST_CALCIUM_RECEIPTS
    result['named_AVA_used']=False; result['global_POP_PCA_used']=False; result['channel_membership_selected_with_behavior']=False
    (OUT/'s46_training_result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8'); (OUT/'s46_run_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
    src.unlink(); rr.unlink(); print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__': main()
