"""Train registered adapters on explicit user-owned train/validation inputs.

No test input enters fitting or threshold selection. Neural checkpoints save
optimizer, RNG, sampled order and cursor at safe batch boundaries.
"""
import json
import os
from pathlib import Path
import random
import signal
import threading
import time
import warnings
import hashlib
import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.linear_model import LogisticRegression
from .fcm import features as fcm_features, fit as fit_fcm, membership
from .io import atomic_json, digest
from .metrics import evaluate, select_threshold
from .models import create_model
from .predictor import RFQCPredictor
from .preprocessing import Preprocessor
from .registry import get_spec


def save_bundle(output,method,seed,threshold,preprocessing,network=None,parameters=None,training=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    bundle=dict(schema=1,method=method,seed=int(seed),threshold=float(threshold),
                software_version='0.1.0',license='GPL-3.0-only',source_repository='https://github.com/cangyeone/rfqc-bench',
                preprocessing=preprocessing,parameters=parameters,training=training or {},
                input_grid=dict(samples=501,start_time=-10.,sampling_interval=.1),
                labels={'bad':0,'good':1})
    if network is not None:
        weights={k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()}
        tmp=output/'weights.safetensors.tmp';save_file(weights,str(tmp));os.replace(tmp,output/'weights.safetensors')
        bundle['weights_sha256']=digest(output/'weights.safetensors')
    atomic_json(output/'bundle.json',bundle)
    return bundle


def _fingerprint(data):
    h=hashlib.sha256()
    for name in ['waveforms','features','gaussians','lengths','labels','stations','sample_ids']:
        a=getattr(data,name);h.update(name.encode())
        if a is None: continue
        h.update(str((a.shape,a.dtype.str)).encode())
        for i in range(0,len(a),256):h.update(np.ascontiguousarray(a[i:i+256]).tobytes())
    return h.hexdigest()


def fit(method,train,validation,output,*,seed=20260928,epochs=50,patience=10,batch_size=32,device='cpu',resume=False,checkpoint_seconds=60.):
    """Fit one model and return an RFQCPredictor. Statistical defaults stay fixed.

    Train and validation need labels. If station IDs are supplied in both, they
    must be disjoint. Resume reuses the exact inputs/options; it starts no daemon.
    """
    spec=get_spec(method);train=train.select(spec.gaussian);validation=validation.select(spec.gaussian)
    for data in [train,validation]:
        if data.labels is None or set(np.unique(data.labels))!={0,1}:raise ValueError('Both classes are required in train and validation')
        if spec.mode!='features' and data.waveforms is None:raise ValueError('Waveforms required')
        if spec.mode!='waveform' and data.features is None:raise ValueError('Six supplied descriptors required')
    if train.stations is not None and validation.stations is not None and set(train.stations)&set(validation.stations):
        raise ValueError('Train and validation stations overlap; provide station-disjoint inputs')
    if train.sample_ids is not None and validation.sample_ids is not None and set(train.sample_ids)&set(validation.sample_ids):
        raise ValueError('Train and validation sample IDs overlap')
    if min(epochs,patience,batch_size)<1 or checkpoint_seconds<=0:raise ValueError('Invalid training budget')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    contract=dict(method=method,seed=seed,epochs=epochs,patience=patience,batch_size=batch_size,
                  device=str(device),torch_version=str(torch.__version__),
                  train_sha256=_fingerprint(train),validation_sha256=_fingerprint(validation))
    prior=output/'run_config.json'
    if prior.exists():
        if not resume:raise FileExistsError('Output already contains a run; use resume=True or a new directory')
        if json.loads(prior.read_text())!=contract:raise ValueError('Resume inputs or training settings differ')
        if (output/'bundle.json').exists():return RFQCPredictor.from_directory(output,device)
    else:
        if any(output.iterdir()):raise FileExistsError('New training output must be empty')
        atomic_json(prior,contract)
    prep=Preprocessor.fit(train,image_adapter=method=='chen2026_image')
    notes={};network=None;parameters=None
    if spec.kind=='logistic':
        x=prep.linear_features(train,spec.gaussian is None);xv=prep.linear_features(validation,spec.gaussian is None)
        estimator=LogisticRegression(C=1.,class_weight='balanced',random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');estimator.fit(x,train.labels)
        pv=estimator.predict_proba(xv)[:,1]
        parameters=dict(coef=estimator.coef_.tolist(),intercept=estimator.intercept_.tolist(),classes=estimator.classes_.tolist())
        notes=dict(iterations=estimator.n_iter_.tolist(),warnings=[str(w.message) for w in caught],max_iter=100)
    elif spec.kind=='fcm':
        if train.stations is None or validation.stations is None:raise ValueError('FCM requires same-station context identifiers')
        f=fcm_features(train.waveforms,train.stations);v=fcm_features(validation.waveforms,validation.stations)
        mean=f.mean(0);std=np.maximum(f.std(0),1e-12);keep=np.all(np.abs((f-mean)/std)<3,axis=1)
        if keep.sum()<2:raise ValueError('Too few inlier training records')
        lo=f[keep].min(0);scale=np.maximum(f[keep].max(0)-lo,1e-12);z=np.clip((f-lo)/scale,0,1)
        latest=output/'fcm_checkpoint.json';state=json.loads(latest.read_text()) if latest.exists() else None
        centers,iterations=fit_fcm(z[keep],seed,state,latest)
        u=membership(z,centers);good=int(np.argmax((u.T@train.labels)/np.maximum(u.sum(0),1e-30)))
        parameters=dict(centers=centers.tolist(),good_cluster=good)
        prep=Preprocessor(dict(minimum=lo.tolist(),scale=scale.tolist(),fit_split='train'))
        pv=membership(np.clip((v-lo)/scale,0,1),centers)[:,good];notes=dict(iterations=iterations)
    else:
        network,pv,notes=_fit_neural(spec,train,validation,prep,output,seed,epochs,patience,batch_size,device,checkpoint_seconds)
    threshold=select_threshold(validation.labels,pv)
    atomic_json(output/'validation_metrics.json',evaluate(validation.labels,pv,threshold))
    save_bundle(output,method,seed,threshold,prep.state,network,parameters,dict(**notes,phase_task_transfer=False))
    atomic_json(output/'status.json',dict(state='completed',method=method,seed=seed))
    return RFQCPredictor.from_directory(output,device)


def _fit_neural(spec,train,val,prep,out,seed,epochs,patience,batch,device,interval):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    network=create_model(spec.name).to(device)
    optimizer=torch.optim.AdamW(network.parameters(),lr=spec.learning_rate,weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed)
    state=dict(epoch=0,cursor=0,order=None,stale=0,best_ap=-1.,best_weights=None,history=[])
    checkpoint=out/'latest_checkpoint.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        network.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer']);state=saved['state']
        torch.set_rng_state(saved['rng']);sampler.set_state(saved['sampler_rng']);random.setstate(saved['python_rng'])
        n=saved['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        if saved.get('cuda_rng'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if saved.get('mps_rng') is not None:torch.mps.set_rng_state(saved['mps_rng'])
    counts=np.bincount(train.labels.astype(int),minlength=2)
    weights=torch.tensor(1./counts[train.labels.astype(int)],dtype=torch.double)
    stop=[False];old={};last_save=time.monotonic()

    def persist(phase):
        n=np.random.get_state();temp=out/'latest_checkpoint.pt.tmp'
        torch.save(dict(weights=network.state_dict(),optimizer=optimizer.state_dict(),state=state,rng=torch.get_rng_state(),
                        sampler_rng=sampler.get_state(),python_rng=random.getstate(),numpy_rng=(n[0],n[1].tolist(),n[2],n[3],n[4]),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.device(device).type=='cuda' else [],
                        mps_rng=torch.mps.get_rng_state() if torch.device(device).type=='mps' else None),temp)
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,checkpoint)
        atomic_json(out/'status.json',dict(state=phase,epoch=state['epoch'],next_record=state['cursor'],history=state['history']))

    def val_probabilities():
        network.eval();values=[]
        with torch.inference_mode():
            for start in range(0,len(val),batch):
                values.append(network(*prep.transform(val.subset(slice(start,start+batch)),device)).sigmoid().cpu().numpy())
        return np.concatenate(values)

    if threading.current_thread() is threading.main_thread():
        for sig in [signal.SIGINT,signal.SIGTERM]:
            old[sig]=signal.getsignal(sig);signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        while state['epoch']<epochs and state['stale']<patience:
            if state['order'] is None:
                state['order']=torch.multinomial(weights,len(train),replacement=True,generator=sampler).tolist();state['cursor']=0
                persist('training')
            network.train()
            while state['cursor']<len(train):
                selected=np.array(state['order'][state['cursor']:state['cursor']+batch]);d=train.subset(selected)
                args=prep.transform(d,device);truth=torch.as_tensor(d.labels,dtype=torch.float32,device=device)
                optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(network(*args),truth)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5);optimizer.step()
                state['cursor']+=len(selected)
                if stop[0] or time.monotonic()-last_save>=interval:
                    persist('paused' if stop[0] else 'training');last_save=time.monotonic()
                if stop[0]:raise KeyboardInterrupt('Saved at a safe batch boundary; repeat with resume=True')
            pv=val_probabilities();ap=evaluate(val.labels,pv)['good_auprc']
            if ap>state['best_ap']:
                state['best_ap']=ap;state['stale']=0
                state['best_weights']={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
            else:state['stale']+=1
            state['history'].append(dict(epoch=state['epoch']+1,validation_ap=ap))
            state['epoch']+=1;state['order']=None;state['cursor']=0;persist('training')
            if stop[0]:persist('paused');raise KeyboardInterrupt('Saved after validation; repeat with resume=True')
        network.load_state_dict(state['best_weights'])
        return network,val_probabilities(),dict(epochs_completed=state['epoch'],history=state['history'],
                                               cuda_trajectory_bitwise_reproducible=False)
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
