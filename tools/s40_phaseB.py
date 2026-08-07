from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

ROOT = Path('s40_data/training')
OUT = Path('s40_phaseB')
OUT.mkdir(parents=True, exist_ok=True)
LAGS = (0,1,2,3,5,8,13)
AR_LAGS = (1,2,3,5,8,13)
ALPHAS = (1.0,10.0,100.0,1000.0)
WARM_FRAC = 0.40
EMBARGO = 24
EMA_SPAN = 5
EPS = 1e-12
STATES = ('QUIET','FORWARD','REVERSE','TURN_LEFT','TURN_RIGHT')


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


def parse_dataset_list(root: Path) -> list[tuple[str,int|None]]:
    matches=sorted(root.rglob('*_datasets.txt'))
    if len(matches)!=1:
        raise RuntimeError(f'expected one dataset list, got {matches}')
    rows=[]
    for raw in matches[0].read_text(encoding='utf-8').splitlines():
        line=raw.split('#',1)[0].strip()
        if not line: continue
        toks=line.split(); rows.append((toks[0], int(toks[1]) if len(toks)==2 else None))
    if len(rows)!=4: raise RuntimeError(rows)
    return rows


def robust_center_scale(x: np.ndarray) -> tuple[float,float]:
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if x.size==0: return 0.0,0.0
    med=float(np.median(x)); mad=float(np.median(np.abs(x-med)))
    return med, 1.4826*mad


def causal_ema(x: np.ndarray, span: int=EMA_SPAN) -> np.ndarray:
    x=np.asarray(x,float)
    out=np.empty_like(x)
    a=2.0/(span+1.0)
    out[0]=x[0]
    for i in range(1,len(x)):
        xi=x[i] if np.isfinite(x[i]) else out[i-1]
        out[i]=(1-a)*out[i-1]+a*xi
    return out


def centerline_curvature(folder: Path, heat: dict[str,Any]) -> np.ndarray:
    center=loadmat(folder/'centerline.mat')
    cl=np.rollaxis(np.asarray(center['centerline'],dtype=float),2,0)
    npoints=cl.shape[1]
    d=np.diff(cl,axis=1)
    tangent=np.unwrap(np.arctan2(-d[:,:,1],d[:,:,0]),axis=-1)
    curvature=np.unwrap(np.diff(tangent,axis=1),axis=-1)*npoints
    metric=np.mean(curvature[:,15:80],axis=1)
    mu=float(np.nanmean(metric)); sd=float(np.nanstd(metric))
    if sd>0: metric[np.abs(metric-mu)>6*sd]=np.nan
    bad=~np.isfinite(metric); good=~bad
    if np.any(bad) and np.any(good):
        metric[bad]=np.interp(np.flatnonzero(bad),np.flatnonzero(good),metric[good])
    clt=np.asarray(heat['clTime'],float).squeeze()
    volt=np.asarray(heat['hasPointsTime'],float).squeeze()
    idx=np.rint(np.interp(volt,clt,np.arange(clt.size))).astype(int)
    idx=np.clip(idx,0,metric.size-1)
    return metric[idx]


def state_features(v: np.ndarray,k: np.ndarray,warm_end: int) -> tuple[np.ndarray,np.ndarray,dict[str,float]]:
    ve=causal_ema(v); ke=causal_ema(k)
    warm=np.arange(len(v))<=warm_end
    vpool=np.abs(ve[warm & np.isfinite(ve)])
    vthr=float(np.quantile(vpool,0.35)) if vpool.size else EPS
    moving=warm & np.isfinite(ke) & (np.abs(ve)>=max(vthr,EPS))
    kpool=np.abs(ke[moving])
    if kpool.size<20: kpool=np.abs(ke[warm & np.isfinite(ke)])
    kthr=float(np.quantile(kpool,0.75)) if kpool.size else EPS
    s=np.empty(len(v),dtype=int)
    for i in range(len(v)):
        if not np.isfinite(ve[i]) or not np.isfinite(ke[i]): s[i]=0
        elif abs(ve[i])<max(vthr,EPS): s[i]=0
        elif ke[i]>=max(kthr,EPS): s[i]=3
        elif ke[i]<=-max(kthr,EPS): s[i]=4
        elif ve[i]>0: s[i]=1
        else: s[i]=2
    dur=np.ones(len(v),dtype=float)
    for i in range(1,len(v)):
        dur[i]=dur[i-1]+1 if s[i]==s[i-1] else 1
    onehot=np.eye(5,dtype=float)[s]
    return onehot,dur,{'movement_threshold':vthr,'turn_threshold':kthr}


def loo_population_median(ratio: np.ndarray,j: int) -> np.ndarray:
    tmp=ratio.copy(); tmp[j,:]=np.nan
    return np.nanmedian(tmp,axis=0)


def build_design(target: np.ndarray,r2ref: np.ndarray,pop: np.ndarray,v: np.ndarray,k: np.ndarray,
                 state: np.ndarray,dur: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    n=len(target); maxlag=max(max(LAGS),max(AR_LAGS))
    rows=[]; ys=[]; idx=[]
    tnorm=np.linspace(-1.0,1.0,n)
    for t in range(maxlag,n):
        feat=[tnorm[t],tnorm[t]*tnorm[t]]
        for lag in LAGS:
            tt=t-lag
            feat.extend([pop[tt],r2ref[tt],v[tt],k[tt]])
        feat.extend(state[t].tolist()); feat.append(dur[t])
        feat.extend(target[t-l] for l in AR_LAGS)
        arr=np.asarray(feat,float)
        if np.isfinite(target[t]) and np.all(np.isfinite(arr)):
            rows.append(arr); ys.append(float(target[t])); idx.append(t)
    return np.asarray(rows,float),np.asarray(ys,float),np.asarray(idx,int)


def ridge_fit(X: np.ndarray,y: np.ndarray,alpha: float) -> np.ndarray:
    Xa=np.column_stack([np.ones(len(X)),X])
    reg=np.eye(Xa.shape[1])*alpha; reg[0,0]=0.0
    return np.linalg.solve(Xa.T@Xa+reg,Xa.T@y)


def predict(beta: np.ndarray,X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)),X])@beta


def choose_and_score(target: np.ndarray,r2ref: np.ndarray,pop: np.ndarray,v: np.ndarray,k: np.ndarray,
                     state: np.ndarray,dur: np.ndarray,warm_end: int,score_start: int) -> dict[str,Any] | None:
    X,y,idx=build_design(target,r2ref,pop,v,k,state,dur)
    wm=idx<=warm_end; sm=idx>=score_start
    if wm.sum()<80 or sm.sum()<150: return None
    Xw,yw=X[wm],y[wm]; Xs,ys=X[sm],y[sm]; iscore=idx[sm]
    cut=max(40,int(np.floor(0.70*len(Xw))))
    if len(Xw)-cut<20: return None
    Xtr0,Xval0=Xw[:cut],Xw[cut:]; ytr0,yval0=yw[:cut],yw[cut:]
    xm=np.nanmedian(Xtr0,axis=0); xmad=np.nanmedian(np.abs(Xtr0-xm),axis=0); xsc=1.4826*xmad
    xsc[~np.isfinite(xsc)|(xsc<EPS)]=1.0
    ym,ysc=robust_center_scale(ytr0)
    if ysc<EPS: return None
    Xtr=(Xtr0-xm)/xsc; Xval=(Xval0-xm)/xsc; ytr=(ytr0-ym)/ysc; yval=(yval0-ym)/ysc
    scored=[]
    for a in ALPHAS:
        b=ridge_fit(Xtr,ytr,a); mse=float(np.mean((yval-predict(b,Xval))**2)); scored.append((mse,-a,a))
    alpha=min(scored)[2]
    xm=np.nanmedian(Xw,axis=0); xmad=np.nanmedian(np.abs(Xw-xm),axis=0); xsc=1.4826*xmad
    xsc[~np.isfinite(xsc)|(xsc<EPS)]=1.0
    ym,ysc=robust_center_scale(yw)
    if ysc<EPS: return None
    Xwn=(Xw-xm)/xsc; Xsn=(Xs-xm)/xsc; ywn=(yw-ym)/ysc; ysn=(ys-ym)/ysc
    beta=ridge_fit(Xwn,ywn,alpha); pred=predict(beta,Xsn); resid=ysn-pred
    if not (np.all(np.isfinite(resid)) and len(resid)>=150): return None
    return {'alpha':alpha,'score_idx':iscore,'raw_norm':ysn,'residual':resid,'r2ref':r2ref[iscore],
            'pop':pop[iscore],'raw_original':ys,'target_center':ym,'target_scale':ysc}


def corr(a: np.ndarray,b: np.ndarray) -> float:
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<10: return 0.0
    aa=a[m]; bb=b[m]
    if np.std(aa)<EPS or np.std(bb)<EPS: return 0.0
    return float(np.corrcoef(aa,bb)[0,1])


def robust_scale(x: np.ndarray) -> float:
    _,s=robust_center_scale(x); return s


def lag1_abs(x: np.ndarray,idx: np.ndarray) -> float:
    if len(x)<3: return 1.0
    m=(idx[1:]==idx[:-1]+1)
    if m.sum()<10: return 1.0
    return abs(corr(x[:-1][m],x[1:][m]))


def event_count(x: np.ndarray,scale: float) -> tuple[int,int]:
    if scale<EPS: return 0,0
    pos=np.flatnonzero(x>3*scale); neg=np.flatnonzero(x<-3*scale)
    def spaced(a: np.ndarray) -> int:
        if len(a)==0: return 0
        c=1; last=int(a[0])
        for z in a[1:]:
            if int(z)-last>=2: c+=1; last=int(z)
        return c
    return spaced(pos),spaced(neg)


def record_analysis(folder: Path,rid: str,cutoff: int|None) -> dict[str,Any]:
    heat=loadmat(folder/'heatDataMS.mat')
    ratio=np.asarray(heat['Ratio2'],float); r2=np.asarray(heat['R2'],float)
    if ratio.ndim!=2 or r2.shape!=ratio.shape: raise RuntimeError((rid,ratio.shape,r2.shape))
    n=min(ratio.shape[1],np.asarray(heat['behavior'][0,0]['v']).shape[0])
    if cutoff is not None: n=min(n,int(cutoff)+1)
    ratio=ratio[:,:n]; r2=r2[:,:n]
    behavior=heat['behavior'][0,0]; v=np.asarray(behavior['v'],float)[:n,0]
    k=centerline_curvature(folder,heat)[:n]
    times=np.asarray(heat['hasPointsTime'],float).squeeze()[:n]
    warm_end=int(np.floor(WARM_FRAC*(n-1))); score_start=warm_end+EMBARGO
    state,dur,thresholds=state_features(v,k,warm_end)
    seed=int.from_bytes(hashlib.sha256(f'BIO001-S40-R2PERM::{rid}'.encode()).digest()[:8],'big')%(2**32)
    rng=np.random.default_rng(seed); perm=rng.permutation(ratio.shape[0])
    if np.any(perm==np.arange(len(perm))): perm=np.roll(perm,1)
    channels=[]; structurally_stable=ratio.shape[0]
    for j in range(ratio.shape[0]):
        warm_rate=float(np.mean(np.isfinite(ratio[j,:warm_end+1])))
        score_rate=float(np.mean(np.isfinite(ratio[j,score_start:]))) if score_start<n else 0.0
        _,tsc=robust_center_scale(ratio[j,:warm_end+1])
        base={'channel':j,'warm_finite_rate':warm_rate,'score_finite_rate':score_rate,'target_warm_scale':tsc}
        if warm_rate<0.80 or score_rate<0.80 or tsc<EPS:
            base['admissible']=False; channels.append(base); continue
        pop=loo_population_median(ratio,j)
        matched=choose_and_score(ratio[j],r2[j],pop,v,k,state,dur,warm_end,score_start)
        control=choose_and_score(ratio[j],r2[perm[j]],pop,v,k,state,dur,warm_end,score_start)
        if matched is None or control is None:
            base['admissible']=False; channels.append(base); continue
        common=np.intersect1d(matched['score_idx'],control['score_idx'])
        if len(common)<150:
            base['admissible']=False; channels.append(base); continue
        mm=np.isin(matched['score_idx'],common); cm=np.isin(control['score_idx'],common)
        resid=matched['residual'][mm]; cres=control['residual'][cm]; idx=matched['score_idx'][mm]
        raw=matched['raw_norm'][mm]; rr=matched['r2ref'][mm]; popv=matched['pop'][mm]
        rs=robust_scale(resid); half=len(resid)//2
        s1=robust_scale(resid[:half]); s2=robust_scale(resid[half:]); split=s2/max(s1,EPS)
        pos,neg=event_count(resid,rs)
        if len(idx)>1 and np.isfinite(times[idx[0]]) and np.isfinite(times[idx[-1]]) and times[idx[-1]]>times[idx[0]]:
            minutes=float((times[idx[-1]]-times[idx[0]])/60.0)
        else:
            minutes=float(len(idx)/300.0)
        minutes=max(minutes,EPS)
        rv=float(np.var(resid)); rawv=float(np.var(raw)); crv=float(np.var(cres))
        base.update({
            'admissible':True,'selected_alpha':matched['alpha'],'permuted_selected_alpha':control['alpha'],
            'scored_samples':int(len(resid)),'innovation_fraction':rv/max(rawv,EPS),
            'matched_to_permuted_residual_variance_ratio':rv/max(crv,EPS),
            'split_half_residual_scale_ratio':split,'abs_lag1_residual_autocorrelation':lag1_abs(resid,idx),
            'abs_corr_residual_R2':abs(corr(resid,rr)),'abs_corr_raw_R2':abs(corr(raw,rr)),
            'abs_corr_residual_population':abs(corr(resid,popv)),'abs_corr_raw_population':abs(corr(raw,popv)),
            'positive_events':pos,'negative_events':neg,'event_rate_per_minute':(pos+neg)/minutes,
            'scored_minutes':minutes,'permuted_r2_channel':int(perm[j]),
        })
        channels.append(base)
    adm=[c for c in channels if c.get('admissible')]
    def med(key:str,default:float=0.0)->float:
        vals=[float(c[key]) for c in adm if np.isfinite(c.get(key,np.nan))]
        return float(np.median(vals)) if vals else default
    summary={
        'structurally_stable_channels':structurally_stable,'admissible_channels':len(adm),
        'admissible_fraction':len(adm)/max(structurally_stable,1),
        'median_split_half_residual_scale_ratio':med('split_half_residual_scale_ratio',999),
        'median_abs_lag1_residual_autocorrelation':med('abs_lag1_residual_autocorrelation',999),
        'median_abs_corr_residual_R2':med('abs_corr_residual_R2',999),
        'median_abs_corr_raw_R2':med('abs_corr_raw_R2',999),
        'median_abs_corr_residual_population':med('abs_corr_residual_population',999),
        'median_abs_corr_raw_population':med('abs_corr_raw_population',999),
        'median_matched_to_permuted_residual_variance_ratio':med('matched_to_permuted_residual_variance_ratio',999),
        'median_innovation_fraction':med('innovation_fraction',999),
        'fraction_channels_event_rate_ge_0_5_per_minute':float(np.mean([c['event_rate_per_minute']>=0.5 for c in adm])) if adm else 0.0,
        'nonadmissible_fraction':1.0-len(adm)/max(structurally_stable,1),
        'permutation_seed':seed,'thresholds':thresholds,
    }
    return {'record_id':rid,'channel_count':ratio.shape[0],'frame_count':n,'warm_end':warm_end,'score_start':score_start,
            'summary':summary,'channels':channels}


def main() -> None:
    rows=parse_dataset_list(ROOT); recs={}
    for rid,cutoff in rows:
        folders=sorted(p for p in ROOT.rglob(f'{rid}_MS') if p.is_dir())
        if len(folders)!=1: raise RuntimeError((rid,folders))
        recs[rid]=record_analysis(folders[0],rid,cutoff)
    sums=[r['summary'] for r in recs.values()]
    g1=True
    g2=all(s['admissible_fraction']>=0.75 for s in sums)
    g3=sum(s['admissible_channels']>=20 for s in sums)>=3
    rb_split=float(np.median([s['median_split_half_residual_scale_ratio'] for s in sums])); g4=0.67<=rb_split<=1.50
    rb_lag=float(np.median([s['median_abs_lag1_residual_autocorrelation'] for s in sums])); g5=rb_lag<=0.35
    r2wins=sum((s['median_abs_corr_residual_R2']<=0.20 and s['median_abs_corr_residual_R2']<s['median_abs_corr_raw_R2']) for s in sums); g6=r2wins>=3
    popwins=sum((s['median_abs_corr_residual_population']<=0.25 and s['median_abs_corr_residual_population']<s['median_abs_corr_raw_population']) for s in sums); g7=popwins>=3
    eventwins=sum(s['fraction_channels_event_rate_ge_0_5_per_minute']>=0.25 for s in sums); g8=eventwins>=3
    permwins=sum(s['median_matched_to_permuted_residual_variance_ratio']<=0.98 for s in sums); g9=permwins>=3
    g10=all(s['nonadmissible_fraction']<=0.25 for s in sums)
    checks={'G1_identity_I1_or_higher':g1,'G2_each_record_admissible_fraction_ge_0_75':g2,
            'G3_at_least_three_records_ge_20_admissible':g3,'G4_balanced_split_half_ratio_in_range':g4,
            'G5_balanced_abs_lag1_le_0_35':g5,'G6_R2_deconfounding_three_of_four':g6,
            'G7_population_deconfounding_three_of_four':g7,'G8_event_support_three_of_four':g8,
            'G9_matched_R2_beats_permuted_three_of_four':g9,'G10_no_record_gt_0_25_nonadmissible':g10}
    passed=sum(bool(v) for v in checks.values())
    status='TRAINING_PASS' if passed==len(checks) else 'TRAINING_FAIL'
    verdict='WITHIN_RECORD_NEURAL_INNOVATION_IDENTIFIABLE' if status=='TRAINING_PASS' else 'NEURAL_INNOVATION_NOT_IDENTIFIABLE_UNDER_S40'
    result={'schema':'bio-001-s40-phaseB-result-v1','identity_class':'I1_WITHIN_RECORD_STABLE_IDENTITY_ONLY',
            'cross_record_biological_identity_claim_allowed':False,'status':status,'verdict':verdict,
            'passed_checks':passed,'total_checks':len(checks),'checks':checks,
            'record_balanced':{'median_split_half_residual_scale_ratio':rb_split,'median_abs_lag1_residual_autocorrelation':rb_lag,
                               'records_R2_deconfounding_pass':r2wins,'records_population_deconfounding_pass':popwins,
                               'records_event_support_pass':eventwins,'records_matched_R2_control_pass':permwins},
            'records':recs,'CeRSI_v2_delta':0.0,'development_evaluation_executed':False,'AML32_chip_opened':False}
    write_json(OUT/'s40_phaseB_result.json',result)
    receipt={'schema':'bio-001-s40-phaseB-run-receipt-v1','status':status,'verdict':verdict,
             'passed_checks':passed,'total_checks':len(checks),'CeRSI_v2_delta':0.0,
             'development_evaluation_executed':False,'AML32_chip_opened':False}
    write_json(OUT/'s40_phaseB_run_receipt.json',receipt)
    print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__':
    main()
