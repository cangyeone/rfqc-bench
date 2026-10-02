"""Evaluate frozen model bundles on one explicit user-owned labeled test pool."""
from pathlib import Path
import argparse
import json
import numpy as np
from rfqc_bench import RFData, RFQCPredictor, evaluate, list_models
from rfqc_bench.io import atomic_json, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['primary'])
    p.add_argument('--seeds',type=int,nargs='+',default=[20260928,20260929,20260930])
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--model-root',type=Path,help='Offline folders METHOD-seedSEED')
    a=p.parse_args();registry={x['name']:x for x in list_models()}
    names=[n for n in registry if n not in ['reference_ag1','reference_ag5']] if a.models==['primary'] else a.models
    if set(names)-set(registry) or len(set(names))!=len(names):p.error('Unknown or duplicate model')
    if len(set(a.seeds))!=len(a.seeds):p.error('Duplicate seeds would repeat the same model')
    if a.output.exists():p.error('Output exists; choose a new path')
    data=RFData.load(a.input)
    if data.labels is None:p.error('Accuracy comparison requires your binary labels')
    if set(np.unique(data.labels))!={0,1}:p.error('Comparison requires both good and bad labels')
    if data.sample_ids is None or len(set(data.sample_ids))!=len(data):p.error('Unique sample_ids are required')
    # Fail before downloads if input regimes do not cover exactly the same pool.
    for name in names:
        selected=data.select(registry[name]['gaussian'])
        if registry[name]['mode']!='waveform' and selected.features is None:p.error(f'{name} requires supplied descriptors')
        if registry[name]['mode']!='features' and selected.waveforms is None:p.error(f'{name} requires waveforms')
        if registry[name]['kind']=='fcm' and selected.stations is None:p.error('FCM requires complete station context')
    results=[]
    for name in names:
        for seed in a.seeds:
            device=a.device if registry[name]['kind']=='neural' else 'cpu'
            predictor=(RFQCPredictor.from_directory(a.model_root/f'{name}-seed{seed}',device)
                       if a.model_root else RFQCPredictor.from_pretrained(name,seed,device))
            if predictor.spec.name!=name or predictor.seed!=seed:p.error('Bundle identity differs from requested model/seed')
            pred=predictor.predict(data,batch_size=a.batch_size)
            result=dict(method=name,seed=seed,**evaluate(data.labels,pred.p_good,pred.threshold))
            results.append(result)
            print(json.dumps(dict(method=name,seed=seed,macro_f1=result['macro_f1'])),flush=True)
    summary=[]
    for name in names:
        chosen=[x for x in results if x['method']==name];row=dict(method=name,seeds=len(chosen))
        for key in ['accuracy','macro_f1','good_auprc','good_precision','good_recall']:
            values=np.array([x[key] for x in chosen]);row[key+'_mean']=float(values.mean())
            row[key+'_sample_sd']=float(values.std(ddof=1)) if len(values)>1 else None
        summary.append(row)
    atomic_json(a.output,dict(input_sha256=digest(a.input) if a.input.is_file() else None,
                records=len(data),runs=results,summary=summary,
                scope='Your fixed test labels and existing bundle thresholds; no fitting, threshold selection or silent missing-view exclusion. Not the published scores unless the exact original cohort and environment are supplied.'))


if __name__=='__main__':main()
