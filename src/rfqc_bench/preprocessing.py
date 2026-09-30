"""Frozen training-only transformations used by the benchmark."""
import numpy as np
import torch
from .data import GAUSSIANS


class Preprocessor:
    def __init__(self, state):
        self.state = dict(state)
        self.mean = np.asarray(state.get('feature_mean',[0.]*6),dtype=float)
        self.std = np.asarray(state.get('feature_std',[1.]*6),dtype=float)
        self.scale = float(state.get('wave_scale',1.))
        if self.mean.shape!=(6,) or self.std.shape!=(6,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0) or not np.isfinite(self.scale) or self.scale<=0:
            raise ValueError('Invalid preprocessing parameters')

    @classmethod
    def fit(cls, data, image_adapter=False):
        total=np.zeros(6); squares=np.zeros(6); count=0; peaks=[]
        for start in range(0,len(data),512):
            d=data.subset(slice(start,start+512))
            valid=np.arange(d.gaussians.shape[1])[None,:] < d.lengths[:,None]
            if d.features is not None:
                f=np.asarray(d.features,dtype=float)[valid]
                f=np.sign(f)*np.log1p(np.abs(f))
                total+=f.sum(0);squares+=(f*f).sum(0);count+=len(f)
            if d.waveforms is not None:
                peaks.append(np.abs(np.asarray(d.waveforms,dtype=float)[valid]).max(-1))
        mean=total/max(count,1)
        std=np.maximum(np.sqrt(np.maximum(squares/max(count,1)-mean*mean,0)),1e-6) if count else np.ones(6)
        return cls(dict(feature_mean=mean.tolist(),feature_std=std.tolist(),
                        wave_scale=max(float(np.median(np.concatenate(peaks))),1e-6) if peaks else 1.,
                        image_adapter=bool(image_adapter),fit_split='train',preprocessing_version=2))

    def transform(self,data,device='cpu'):
        n,k=data.gaussians.shape;valid=np.arange(k)[None,:]<data.lengths[:,None]
        wave=np.zeros((n,k,501),dtype=np.float32)
        if data.waveforms is not None:
            raw=np.asarray(data.waveforms[valid],dtype=np.float64)
            if self.state.get('image_adapter'):
                w=raw/np.maximum(np.max(np.abs(raw),axis=-1,keepdims=True),1e-30)
            else: w=np.arcsinh(raw/self.scale)
            wave[valid]=w.astype(np.float32)
        features=np.zeros((n,k,6),dtype=np.float32)
        if data.features is not None:
            # Match the original neural RFDataset's float32 feature arithmetic.
            f=np.asarray(data.features[valid],dtype=np.float32)
            features[valid]=(np.sign(f)*np.log1p(np.abs(f))-self.mean.astype(np.float32))/self.std.astype(np.float32)
        if not np.isfinite(wave).all() or not np.isfinite(features).all():
            raise ValueError('Nonfinite transformed input')
        gaussians=np.where(valid,data.gaussians,0).astype(np.float32)
        return tuple(torch.as_tensor(a,device=device) for a in [wave,features,gaussians])+ (torch.as_tensor(data.lengths,dtype=torch.long),)

    def linear_features(self,data,multifilter):
        if data.features is None: raise ValueError('Six supplied descriptors are required')
        f=np.asarray(data.features,dtype=np.float64)
        f=(np.sign(f)*np.log1p(np.abs(f))-self.mean)/self.std
        valid=np.arange(f.shape[1])[None,:]<data.lengths[:,None]
        f[~valid]=0
        if not multifilter: return f.reshape(len(f),6)
        ordered=np.zeros((len(f),7,6),dtype=float)
        for slot,g in enumerate(GAUSSIANS):
            ii,jj=np.where((data.gaussians==g)&valid);ordered[ii,slot]=f[ii,jj]
        return ordered.reshape(len(f),42)
