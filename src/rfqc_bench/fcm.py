"""Registered Xiong feature and fuzzy-clustering equations."""
import numpy as np
from scipy.signal import find_peaks
from .io import atomic_json

def features(waves,stations):
    x=np.asarray(waves,dtype=np.float64).reshape(len(waves),501)
    # Scale first to avoid overflow in squared amplitudes; then paper's unit norm.
    x=x/np.maximum(np.max(np.abs(x),axis=1,keepdims=True),1e-30)
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30)
    result=np.zeros((len(x),4),dtype=np.float64)
    for i,w in enumerate(x):
        p=w[find_peaks(w)[0]]
        p=p[p>0]
        if len(p):
            largest=p.max(); second=p[p<largest]
            result[i,0]=second.max()/largest if len(second) else 0.
    a=np.abs(x); count=50  # floor(501*.1), explicit integer convention
    result[:,1]=np.partition(a,-count,axis=1)[:,-count:].sum(1)/np.maximum(a.sum(1),1e-30)
    # Xiong Eq.3 is noise RMS / signal RMS, despite its SNR name.
    result[:,2]=np.sqrt(np.mean(x[:,:71]**2,axis=1))/np.maximum(np.sqrt(np.mean(x[:,100:201]**2,axis=1)),1e-30)
    signal=x[:,100:201]; signal=signal-signal.mean(1,keepdims=True)
    signal=signal/np.maximum(np.linalg.norm(signal,axis=1,keepdims=True),1e-30)
    for station in sorted(set(stations)):
        idx=np.flatnonzero(stations==station); n=len(idx)
        if n<2: continue
        k=int(np.ceil((n-1)/2)); s=signal[idx]
        for start in range(0,n,128):
            stop=min(start+128,n)
            cc=np.clip(s[start:stop]@s.T,-1,1)
            cc[np.arange(stop-start),np.arange(start,stop)]=-np.inf
            result[idx[start:stop],3]=np.partition(cc,n-k,axis=1)[:,-k:].mean(1)
    if not np.isfinite(result).all(): raise ValueError('Non-finite FCM features')
    return result

def membership(x,centers):
    d2=((x[:,None,:]-centers[None,:,:])**2).sum(2)
    zero=d2<=1e-30
    inv=1/np.maximum(d2,1e-30)
    u=inv/inv.sum(1,keepdims=True)
    hit=zero.any(1)
    u[hit]=zero[hit]/zero[hit].sum(1,keepdims=True)
    return u

def fit(x,seed,state=None,path=None):
    rng=np.random.default_rng(seed)
    centers=x[rng.choice(len(x),2,replace=False)].copy() if state is None else np.array(state['centers'])
    start=0 if state is None else state['iteration']
    if start>=300 or (state is not None and state.get('change',1)<1e-5):
        return centers,start
    for step in range(start,300):
        u=membership(x,centers); w=u*u
        update=(w.T@x)/np.maximum(w.sum(0)[:,None],1e-30)
        change=float(np.max(np.abs(update-centers)));centers=update
        if path and ((step+1)%10==0 or change<1e-5):
            atomic_json(path,dict(iteration=step+1,centers=centers.tolist(),change=change))
        if change<1e-5: break
    return centers,step+1
