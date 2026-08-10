from __future__ import annotations

import io, math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import periodogram

BAND=(0.10,5.00)


def parse_worm(path:Path)->tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
    start=None
    for i,line in enumerate(lines):
        if line.lstrip().lower().startswith('time'):
            start=i; break
    if start is None: raise ValueError('NO_TIME_HEADER')
    try:
        a=np.loadtxt(io.StringIO('\n'.join(lines[start+1:])),dtype=float,ndmin=2)
    except Exception as exc:
        raise ValueError('NUMERIC_PARSE') from exc
    if a.ndim!=2 or a.shape[1]<11: raise ValueError('LT_11_NUMERIC_COLUMNS')
    t=a[:,0]; x=a[:,1:11]
    finite=np.isfinite(t)&np.all(np.isfinite(x),axis=1); t=t[finite]; x=x[finite]
    order=np.argsort(t,kind='stable'); t=t[order]; x=x[order]
    if len(t):
        keep=np.r_[True,np.diff(t)!=0]; t=t[keep]; x=x[keep]
    if len(t)<200: raise ValueError('LT_200_ROWS')
    if t[-1]-t[0]<10.0: raise ValueError('LT_10_SECONDS')
    gaps=np.diff(t); pos=gaps[gaps>0]
    if not len(pos): raise ValueError('NO_POSITIVE_DT')
    dt0=float(np.median(pos))
    if not np.isfinite(dt0) or dt0<=0: raise ValueError('BAD_MEDIAN_DT')
    if float(np.mean(pos<=5.0*dt0))<0.95: raise ValueError('EXCESS_LARGE_GAPS')
    grid=np.arange(t[0],t[-1]+0.5*dt0,dt0)
    xu=np.column_stack([np.interp(grid,t,x[:,j]) for j in range(10)])
    xu=xu-np.median(xu,axis=0,keepdims=True)
    return grid,xu,{'raw_numeric_rows':int(a.shape[0]),'finite_unique_rows':int(len(t)),'duration_s':float(t[-1]-t[0]),'median_dt_s':dt0,'uniform_rows':int(len(grid))}


def endpoints(time:np.ndarray,x:np.ndarray)->dict[str,float|int]:
    t=np.asarray(time,float); x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=10 or len(t)!=len(x): raise ValueError('SHAPE')
    dt=float(np.median(np.diff(t))); fs=1.0/dt
    specs=[]; fref=None
    for j in range(2,8):
        f,p=periodogram(x[:,j],fs=fs,window='hann',detrend='constant',scaling='spectrum')
        band=(f>=BAND[0])&(f<=BAND[1]); fb=f[band]; pb=p[band]
        if not len(fb): raise ValueError('EMPTY_FREQ_BAND')
        total=float(np.sum(pb)); pn=pb/total if total>0 else np.zeros_like(pb)
        if fref is None: fref=fb
        elif not np.allclose(fref,fb,rtol=0,atol=1e-12): raise ValueError('FREQ_GRID_DRIFT')
        specs.append(pn)
    avg=np.mean(np.vstack(specs),axis=0)
    mx=np.max(avg); imax=int(np.flatnonzero(avg==mx)[0]); fstar=float(fref[imax])
    amp=float(np.sqrt(np.mean(x*x)))
    n=len(t); win=np.hanning(n); ff=np.fft.rfftfreq(n,d=dt); k=int(np.argmin(np.abs(ff-fstar)))
    coeff=np.array([np.fft.rfft(x[:,j]*win)[k] for j in range(10)])
    phase=np.unwrap(np.angle(coeff)); idx=np.arange(10,dtype=float)
    slope,inter=np.polyfit(idx,phase,1); pred=slope*idx+inter
    ssr=float(np.sum((phase-pred)**2)); sst=float(np.sum((phase-np.mean(phase))**2)); r2=1.0-ssr/sst if sst>0 else 0.0
    spatial=float(abs(slope)*9.0/(2.0*math.pi))
    width=max(0.10,0.20*fstar); near=(fref>=max(BAND[0],fstar-width))&(fref<=min(BAND[1],fstar+width))
    den=float(np.sum(avg)); rhythmic=float(np.sum(avg[near])/den) if den>0 else 0.0
    return {'dominant_frequency_hz':fstar,'rms_angle_amplitude':amp,'spatial_cycles_per_body':spatial,'spatial_phase_r2':float(r2),'rhythmicity':rhythmic,'uniform_dt_s':dt,'uniform_samples':int(n)}
