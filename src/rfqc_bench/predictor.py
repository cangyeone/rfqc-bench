"""Uniform trained inference for the neural, FCM and logistic adapters."""
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
import torch
from safetensors.torch import load_file
from .data import RFData
from .fcm import features as fcm_features, membership
from .io import digest
from .models import create_model
from .preprocessing import Preprocessor
from .registry import get_spec
from .zoo import download_model


@dataclass
class Prediction:
    p_good: np.ndarray
    prediction: np.ndarray
    threshold: float
    method: str
    seed: int

    def to_dict(self):
        return dict(p_good=self.p_good.tolist(),prediction=self.prediction.tolist(),threshold=self.threshold,method=self.method,seed=self.seed)

    def to_csv(self,path,sample_ids=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        ids=np.arange(len(self.p_good)).astype(str) if sample_ids is None else sample_ids
        if len(ids)!=len(self.p_good): raise ValueError('sample_ids length mismatch')
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['sample_id','p_good','prediction','threshold','method','seed'])
            writer.writerows((i,float(p),int(y),self.threshold,self.method,self.seed) for i,p,y in zip(ids,self.p_good,self.prediction))


class RFQCPredictor:
    def __init__(self,bundle,network=None,device='cpu'):
        if bundle.get('schema')!=1: raise ValueError('Unsupported model bundle schema')
        self.bundle=bundle;self.spec=get_spec(bundle['method']);self.device=torch.device(device)
        self.threshold=float(bundle['threshold']);self.seed=int(bundle['seed'])
        if not np.isfinite(self.threshold) or not 0<=self.threshold<=1: raise ValueError('Invalid threshold')
        self.preprocessor=Preprocessor(bundle.get('preprocessing',{}))
        self.network=network
        if network is not None: self.network.to(self.device).eval()
        elif self.spec.kind=='neural': raise ValueError('Trained neural weights required')

    @classmethod
    def from_directory(cls,path,device='cpu'):
        path=Path(path);bundle=json.loads((path/'bundle.json').read_text());spec=get_spec(bundle['method'])
        network=None
        if spec.kind=='neural':
            weight=path/'weights.safetensors'
            if digest(weight)!=bundle['weights_sha256']: raise ValueError('Neural weight checksum mismatch')
            network=create_model(spec.name)
            network.load_state_dict(load_file(str(weight),device='cpu'),strict=True)
        return cls(bundle,network,device)

    @classmethod
    def from_pretrained(cls,name='reference_multifilter',seed=None,device='cpu',cache_dir=None):
        """Load RF-trained benchmark weights; no phase-task transfer is implied."""
        return cls.from_directory(download_model(name,seed,cache_dir),device)

    def predict(self,data=None,*,waveforms=None,features=None,gaussians=None,lengths=None,stations=None,batch_size=32):
        if batch_size<1: raise ValueError('batch_size must be positive')
        if data is not None and any(x is not None for x in [waveforms,features,gaussians,lengths,stations]):
            raise ValueError('Pass RFData or array keyword arguments, not both')
        data=RFData(waveforms,gaussians,lengths,features,stations) if data is None else data
        if not isinstance(data,RFData): raise TypeError('data must be RFData')
        data=data.select(self.spec.gaussian)
        if self.spec.mode!='features' and data.waveforms is None: raise ValueError('This model requires waveforms')
        if self.spec.mode!='waveform' and data.features is None: raise ValueError('This model requires the six supplied descriptors')
        if self.spec.kind=='neural':
            values=[]
            with torch.inference_mode():
                for start in range(0,len(data),batch_size):
                    args=self.preprocessor.transform(data.subset(slice(start,start+batch_size)),self.device)
                    values.append(self.network(*args).sigmoid().cpu().numpy())
            p=np.concatenate(values).astype(float)
        elif self.spec.kind=='logistic':
            params=self.bundle['parameters'];x=self.preprocessor.linear_features(data,self.spec.gaussian is None)
            p=expit(x@np.asarray(params['coef'])[0]+params['intercept'][0])
        else:
            if data.stations is None: raise ValueError('Xiong-FCM requires stations and a complete same-station context batch')
            f=fcm_features(data.waveforms,data.stations);params=self.bundle['parameters'];prep=self.bundle['preprocessing']
            x=np.clip((f-np.asarray(prep['minimum']))/np.asarray(prep['scale']),0,1)
            p=membership(x,np.asarray(params['centers']))[:,int(params['good_cluster'])]
        if p.shape!=(len(data),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Model returned invalid scores')
        return Prediction(p,(p>=self.threshold).astype(int),self.threshold,self.spec.name,self.seed)
