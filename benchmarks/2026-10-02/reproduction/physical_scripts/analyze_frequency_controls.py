"""Only report AG1/AG3/AG5/multi when both new fits have verified completion."""
import argparse,csv,json
import numpy as np
from pathlib import Path
from analyze_physical import ROOT,sha,table,agreement,window,pair_correlations
from classification_metrics import metrics

def main(workspace):
    cache=workspace/'rf_quality_control/cache'
    ix=np.load(cache/'test_indices.npy');length=np.load(cache/'lengths.npy')[ix]
    common=length==7;ix=ix[common]
    ids=np.load(cache/'sample_ids.npy')[ix];labels=np.load(cache/'labels.npy')[ix];stations=np.load(cache/'stations.npy')[ix]
    assert len(ids)==24107 and len(set(stations))==25
    refs=workspace/'rfqc_benchmark_20260929/results/bands'
    runs={'ag1':ROOT/'runs/ag1_waveform_random_seed20260929','ag3':refs/'ag3/waveform_random_seed20260929',
        'ag5':ROOT/'runs/ag5_waveform_random_seed20260929','multiband':refs/'multiband/waveform_random_seed20260929'}
    pending=[k for k,p in runs.items() if not (p/'completion.json').exists()]
    if pending:raise RuntimeError('Frequency comparison incomplete; no scores published: '+', '.join(pending))
    choices={};receipts=[];overall=[];by_station=[]
    settings={'seed':20260929,'arm':'waveform_random','epochs':50,'patience':10,'batch_size':32,'lr':.0003,'input_samples':2048,'balance':'sampler','pretrained':False,'device':'cuda','trainable_parameters':395393,'torch_version':'2.8.0+cu128'}
    for band,p in runs.items():
        manifest=json.loads((p/'completion.json').read_text());assert all(sha(p/n)==h for n,h in manifest['sha256'].items())
        c=json.loads((p/'config.json').read_text());assert all(c[k]==v for k,v in settings.items()),band
        assert c['actual_split_sizes']['train']==101852 and c['actual_split_sizes']['val']==21309 and c['smoke_samples']==0
        if band in ['ag1','ag5']:assert c['resume_contract']['protocol_sha256']==sha(ROOT/'protocol.json')
        with (p/'test_predictions.csv').open() as f:rows=list(csv.DictReader(f))
        assert len({r['sample_id'] for r in rows})==len(rows)
        original=json.loads((p/'metrics.json').read_text())['test'];threshold=original['threshold']
        full=metrics(np.array([int(r['label']) for r in rows]),np.array([float(r['p_good']) for r in rows]),threshold)
        for k in ['accuracy','macro_f1','good_auprc','auroc']:assert abs(full[k]-original[k])<1e-10,(band,k)
        lookup={r['sample_id']:r for r in rows};chosen=[lookup[x] for x in ids]
        np.testing.assert_array_equal(labels,[int(r['label']) for r in chosen]);np.testing.assert_array_equal(stations,[r['station'] for r in chosen])
        scores=np.array([float(r['p_good']) for r in chosen]);keep=scores>=threshold
        np.testing.assert_array_equal(keep,[int(r['prediction']) for r in chosen]);choices[band]=keep
        overall.append(dict(band=band,seed=20260929,**metrics(labels,scores,threshold)))
        for station in sorted(set(stations)):
            sel=stations==station;by_station.append(dict(band=band,station=station,**metrics(labels[sel],scores[sel],threshold)))
        receipts.append(dict(band=band,path=str(p),completion=manifest))
    # Diagnose all four classifications on identical raw AG3 waveforms and native seven-band pairs.
    waves=np.asarray(np.load(cache/'waveforms.npy',mmap_mode='r')[ix],dtype=np.float64)
    normalized=waves/np.maximum(np.abs(waves).max(-1,keepdims=True),1e-100)
    pair,_=pair_correlations(waves,np.ones(waves.shape[:2],bool),window('post_p'));association=pair.mean(1)
    stacks=[]
    for station in sorted(set(stations)):
        pool=np.flatnonzero(stations==station);manual=pool[labels[pool]==1]
        for band,keep in choices.items():
            accepted=pool[keep[pool]]
            for norm,data in [('raw',waves),('peak_normalized',normalized)]:
                base=dict(band=band,station=station,normalization=norm,n_input=len(pool),n_selected=len(accepted),n_manual=len(manual))
                if len(accepted) and len(manual):base.update(agreement(data[accepted,4].mean(0),data[manual,4].mean(0)))
                else:base.update({k:np.nan for k in agreement(np.ones(501),np.ones(501))})
                base['mean_crossband_correlation']=float(association[accepted].mean()) if len(accepted) else np.nan
                stacks.append(base)
    output=ROOT/'results/frequency_control';output.mkdir(exist_ok=True)
    table(output/'classification.csv',overall);table(output/'station_classification.csv',by_station);table(output/'stack_and_association.csv',stacks)
    np.savez_compressed(output/'selection.npz',sample_ids=ids,labels=labels,stations=stations,methods=np.array(list(choices)),selected=np.stack(list(choices.values())))
    (output/'verification.json').write_text(json.dumps(dict(complete=True,common_test_n=len(ids),stations=len(set(stations)),
        seed=20260929,records_excluded=426,excluded_station='YP_NE53',protocol_sha256=sha(ROOT/'protocol.json'),
        comparison='One fixed seed; no training-seed uncertainty estimated. AG3 and multi rescored on common subset; no threshold retuning.',sources=receipts),indent=2)+'\n')
    print(json.dumps(overall,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True,type=Path);main(p.parse_args().workspace.resolve())
