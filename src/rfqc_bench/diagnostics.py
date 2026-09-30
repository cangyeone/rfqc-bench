"""RF stack and cross-filter diagnostics from the frozen benchmark equations.

These operational extrema are not a replacement for undocumented provider
feature definitions, azimuth measurements, or structural inversion.
"""
import numpy as np
from scipy.signal import find_peaks
TIME=np.arange(501,dtype=float)/10-10
WINDOWS={'whole':(-5,20),'p':(-.3,1.6),'ps_candidate':(2.5,7.5),'post_p':(2.5,15)}

def window(name):
    lo,hi=WINDOWS[name];return (TIME>=lo-1e-8)&(TIME<=hi+1e-8)

def corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a/max(np.max(np.abs(a)),1e-100);b=b/max(np.max(np.abs(b)),1e-100)
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/den,-1,1)) if den>1e-15 else np.nan

def stack_features(x):
    p=np.flatnonzero(window('p'));s=np.flatnonzero(window('ps_candidate'))
    extrema=np.sort(np.r_[find_peaks(x)[0],find_peaks(-x)[0]])
    extrema=extrema[np.isin(extrema,s)]
    pick=int(extrema[np.argmax(x[extrema])]) if len(extrema) else int(s[np.argmax(x[s])])
    pi=int(p[np.argmax(x[p])]);mi=int(np.argmax(x))
    neg=abs(float(x[(TIME>=-10)&(TIME<=5)].min()))
    return dict(P_amp=float(x[pi]),P_time=float(TIME[pi]),Ps_amp=float(x[pick]),Ps_time=float(TIME[pick]),
        MaxAmp=float(x[mi]),MaxTime=float(TIME[mi]),NegMinAmp=neg,
        NegPosRatio=neg/x[mi] if x[mi]>0 else np.nan,
        PPsRatio=x[mi]/x[pick] if x[pick]>0 else np.nan,PsPeaks=len(extrema),
        Ps_fallback=int(not len(extrema)),P_boundary=int(pi in [p[0],p[-1]]))

def agreement(x,ref):
    out={}
    for name in ['whole','post_p','ps_candidate']:
        mask=window(name);a=x[mask];b=ref[mask]
        scale=max(np.abs(b).max(),1e-100)
        den=np.sqrt(np.mean((b/scale)**2))
        out['corr_'+name]=corr(a,b)
        out['nrmse_'+name]=float(np.sqrt(np.mean(((a-b)/scale)**2))/den) if den>1e-15 else np.nan
    fx=stack_features(x);fr=stack_features(ref)
    for name,value in fx.items():
        out[name]=value;out['manual_'+name]=fr[name];out['delta_'+name]=value-fr[name]
    return out


def _finite(report):
    return {k: float(v) if np.isfinite(v) else None for k,v in report.items()}


def stack_comparison(waveforms,manual_good,predicted_good):
    """Same-pool AG3 arithmetic stacks, raw and per-record peak normalized."""
    x=np.asarray(waveforms,dtype=float)
    if x.ndim!=2 or x.shape[1]!=501 or not np.isfinite(x).all():
        raise ValueError('Expected N x 501 finite AG3 RFs')
    masks=[np.asarray(m,dtype=bool) for m in [manual_good,predicted_good]]
    if any(m.shape!=(len(x),) or not m.any() for m in masks):
        raise ValueError('Both selections must be nonempty and match N')
    raw=[x[m].mean(0) for m in masks]
    norm=x/np.maximum(np.abs(x).max(1,keepdims=True),1e-100)
    shapes=[norm[m].mean(0) for m in masks]
    return dict(manual_count=int(masks[0].sum()),automatic_count=int(masks[1].sum()),
                raw=_finite(agreement(raw[1],raw[0])),shape=_finite(agreement(shapes[1],shapes[0])),
                interpretation='Descriptive same-pool stacks; Ps candidate is not an independently identified phase.')


def cross_filter_association(data):
    """Mean of finite within-record filter-pair Pearson correlations, 2.5..15 s."""
    if data.waveforms is None:raise ValueError('Waveforms required')
    result=[]
    for i,n in enumerate(data.lengths):
        x=np.asarray(data.waveforms[i,:n][:,window('post_p')],dtype=float)
        values=[corr(x[a],x[b]) for a in range(n) for b in range(a+1,n)]
        finite=[v for v in values if np.isfinite(v)]
        result.append(float(np.mean(finite)) if finite else None)
    return result
