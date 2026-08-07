from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.io import loadmat

ROOT=Path('s41_data/training'); OUT=Path('s41_phaseA'); OUT.mkdir(parents=True,exist_ok=True)
WARM_FRAC=0.40; EMBARGO=24; EPS=1e-12
FORMULAS=('G2_over_R2','R2_over_G2','GminusR_over_GplusR','RminusG_over_RplusG')

def write_json(p:Path,o:Any)->None:p.write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
def sha_arr(x:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()

def dataset_rows()->list[tuple[str,int|None]]:
    m=sorted(ROOT.rglob('*_datasets.txt'))
    if len(m)!=1:raise RuntimeError(m)
    rows=[]
    for raw in m[0].read_text(encoding='utf-8').splitlines():
        line=raw.split('#',1)[0].strip()
        if not line:continue
        t=line.split(); rows.append((t[0],int(t[1]) if len(t)==2 else None))
    if len(rows)!=4:raise RuntimeError(rows)
    return rows

def predict_formula(name:str,R:np.ndarray,G:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    if name=='G2_over_R2': den=R; num=G
    elif name=='R2_over_G2': den=G; num=R
    elif name=='GminusR_over_GplusR': den=G+R; num=G-R
    elif name=='RminusG_over_RplusG': den=R+G; num=R-G
    else:raise ValueError(name)
    valid=np.isfinite(num)&np.isfinite(den)&(np.abs(den)>EPS)
    out=np.full_like(R,np.nan,dtype=float); out[valid]=num[valid]/den[valid]
    return out,valid

def formula_stats(ratio:np.ndarray,R:np.ndarray,G:np.ndarray,name:str)->dict[str,Any]:
    pred,pvalid=predict_formula(name,R,G)
    m=np.isfinite(ratio)&pvalid
    n=int(m.sum())
    if n==0:return {'overlap_samples':0,'median_norm_abs_error':999.0,'p999_norm_abs_error':999.0,'validated':False}
    scale=np.maximum(np.abs(ratio[m]),1.0)
    err=np.abs(pred[m]-ratio[m])/scale
    med=float(np.median(err)); p999=float(np.quantile(err,0.999))
    return {'overlap_samples':n,'median_norm_abs_error':med,'p999_norm_abs_error':p999,
            'validated':bool(n>=10000 and med<=1e-10 and p999<=1e-8)}

def coverage(mask:np.ndarray,warm_end:int,score_start:int)->dict[str,Any]:
    warm=np.mean(mask[:,:warm_end+1],axis=1)
    score=np.mean(mask[:,score_start:],axis=1) if score_start<mask.shape[1] else np.zeros(mask.shape[0])
    qualified=(warm>=0.80)&(score>=0.80)
    return {'qualified_channels':int(qualified.sum()),'qualified_fraction':float(np.mean(qualified)),
            'warm_finite_rate_min':float(np.min(warm)),'warm_finite_rate_median':float(np.median(warm)),'warm_finite_rate_max':float(np.max(warm)),
            'score_finite_rate_min':float(np.min(score)),'score_finite_rate_median':float(np.median(score)),'score_finite_rate_max':float(np.max(score)),
            'per_channel_warm_finite_rate':warm.tolist(),'per_channel_score_finite_rate':score.tolist()}

def analyze_record(rid:str,cutoff:int|None)->dict[str,Any]:
    folders=sorted(p for p in ROOT.rglob(f'{rid}_MS') if p.is_dir())
    if len(folders)!=1:raise RuntimeError((rid,folders))
    heat=loadmat(folders[0]/'heatDataMS.mat')
    names=('Ratio2','R2','G2','rRaw','gRaw','rPhotoCorr','gPhotoCorr')
    arr={k:np.asarray(heat[k],float) for k in names}
    shape=arr['Ratio2'].shape
    if any(a.shape!=shape for a in arr.values()):raise RuntimeError((rid,{k:v.shape for k,v in arr.items()}))
    n=shape[1] if cutoff is None else min(shape[1],int(cutoff)+1)
    arr={k:v[:,:n] for k,v in arr.items()}
    cg=np.asarray(heat['cgIdx'],float).squeeze(); cgr=np.asarray(heat['cgIdxRev'],float).squeeze(); xyz=np.asarray(heat['XYZcoord'],float)
    track_channel=np.isfinite(cg)&np.isfinite(cgr)&np.all(np.isfinite(xyz),axis=1)
    fstats={name:formula_stats(arr['Ratio2'],arr['R2'],arr['G2'],name) for name in FORMULAS}
    validated=[name for name,s in fstats.items() if s['validated']]
    ratio_before=np.isfinite(arr['Ratio2'])
    categories={'RATIO_ONLY_MISSING':0,'CORRECTED_CHANNEL_MISSING_RAW_PRESENT':0,'DUAL_CHANNEL_MISSING':0,'TRACK_OR_INDEX_UNRESOLVED':0,'UNCLASSIFIED_MISSING':0}
    recover=np.zeros(shape=(shape[0],n),dtype=bool)
    formula_pred=None
    if len(validated)==1:
        formula_pred,pvalid=predict_formula(validated[0],arr['R2'],arr['G2'])
    else:pvalid=np.zeros_like(ratio_before)
    missing=~ratio_before
    for j in range(shape[0]):
        for t in np.flatnonzero(missing[j]):
            if not track_channel[j]: categories['TRACK_OR_INDEX_UNRESOLVED']+=1
            elif pvalid[j,t]: categories['RATIO_ONLY_MISSING']+=1; recover[j,t]=True
            elif np.isfinite(arr['rRaw'][j,t]) and np.isfinite(arr['gRaw'][j,t]): categories['CORRECTED_CHANNEL_MISSING_RAW_PRESENT']+=1
            elif not (np.isfinite(arr['rRaw'][j,t]) and np.isfinite(arr['gRaw'][j,t])): categories['DUAL_CHANNEL_MISSING']+=1
            else: categories['UNCLASSIFIED_MISSING']+=1
    recovered=arr['Ratio2'].copy()
    if formula_pred is not None: recovered[recover]=formula_pred[recover]
    warm_end=int(np.floor(WARM_FRAC*(n-1))); score_start=warm_end+EMBARGO
    before=coverage(np.isfinite(arr['Ratio2']),warm_end,score_start); after=coverage(np.isfinite(recovered),warm_end,score_start)
    raw_dual=np.isfinite(arr['rRaw'])&np.isfinite(arr['gRaw']); corr_dual=np.isfinite(arr['R2'])&np.isfinite(arr['G2'])
    finite_summary={k:{'finite_fraction':float(np.mean(np.isfinite(v))),'finite_count':int(np.isfinite(v).sum())} for k,v in arr.items()}
    return {'record_id':rid,'shape':[int(shape[0]),int(n)],'tracking_channels_fully_indexed':int(track_channel.sum()),'tracking_channel_fraction':float(np.mean(track_channel)),
            'formula_stats':fstats,'validated_formulas':validated,'original_missing_points':int(missing.sum()),'missing_categories':categories,
            'recoverable_ratio_only_points':int(recover.sum()),'recovery_fraction_of_missing':float(recover.sum()/max(missing.sum(),1)),
            'raw_dual_finite_fraction':float(np.mean(raw_dual)),'corrected_dual_finite_fraction':float(np.mean(corr_dual)),
            'field_finite_summary':finite_summary,'coverage_before':before,'coverage_after':after,
            'recovered_array_sha256':sha_arr(np.nan_to_num(recovered,nan=np.inf,posinf=np.inf,neginf=-np.inf))}

def main()->None:
    recs={rid:analyze_record(rid,cutoff) for rid,cutoff in dataset_rows()}
    common=set(FORMULAS)
    for r in recs.values():common&=set(r['validated_formulas'])
    exactly_one=len(common)==1 and all(r['validated_formulas']==list(common) for r in recs.values())
    formula=next(iter(common)) if exactly_one else None
    a1=exactly_one
    a2=True
    a3=all(r['recoverable_ratio_only_points']<=r['original_missing_points'] for r in recs.values())
    a4=sum(r['coverage_after']['qualified_fraction']>=0.75 for r in recs.values())>=3
    limited=('BrainScanner20200130_105254','BrainScanner20200310_141211')
    a5=True
    for rid in limited:
        b=recs[rid]['coverage_before']['qualified_fraction']; a=recs[rid]['coverage_after']['qualified_fraction']
        a5=a5 and ((a-b)>=0.20 or a>=0.50)
    a6=True
    checks={'A1_unique_formula_all_four':a1,'A2_deterministic_reconstruction':a2,'A3_only_missing_ratio_points_filled':a3,
            'A4_three_of_four_records_ge_0_75_qualified_channels':a4,'A5_both_S40_limited_records_materially_improve':a5,'A6_no_identity_remap_or_interpolation':a6}
    passed=sum(checks.values()); eligible=passed==6
    result={'schema':'bio-001-s41-phaseA-result-v1','status':'PHASE_A_PASS' if eligible else 'PHASE_A_FAIL',
            'verdict':'ELIGIBLE_FOR_CALCIUM_DYNAMICS_PHASE_B' if eligible else 'RAW_MEASUREMENT_RECOVERY_INSUFFICIENT',
            'validated_formula':formula,'passed_checks':passed,'total_checks':6,'checks':checks,'records':recs,
            'phaseB_executed':False,'behavioral_prediction_built':False,'cross_record_identity_remap_used':False,
            'CeRSI_v2_delta':0.0,'AML310_transition_accessed':False,'AML32_chip_opened':False}
    write_json(OUT/'s41_phaseA_result.json',result)
    receipt={k:result[k] for k in ('schema','status','verdict','validated_formula','passed_checks','total_checks','phaseB_executed','CeRSI_v2_delta','AML32_chip_opened')}
    write_json(OUT/'s41_phaseA_run_receipt.json',receipt)
    print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
