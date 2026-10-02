"""RF-specific diagnostics from verified frozen classifications, not new thresholds."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]
TIME=np.arange(501,dtype=float)/10-10
BANDS=[1.,1.5,2.,2.5,3.,4.,5.]
WINDOWS={'whole':(-5,20),'p':(-.3,1.6),'ps_candidate':(2.5,7.5),'post_p':(2.5,15)}
SPECS=[('li2021_cnn','Li 2021','extension','li2021_cnn'),
 ('gan2021_cnn','Gan 2021','extension','gan2021_cnn'),('gong_cnn','Gong CNN','literature','gong_cnn'),
 ('gong_cnn_bilstm','Gong CNN-BiLSTM','literature','gong_cnn_bilstm'),
 ('gan_cnn','Gan 2023','literature','gan_cnn'),('deeprfqc','DeepRFQC','literature','deeprfqc'),
 ('hegaz_capsule','RF-Capsule','literature','hegaz_capsule'),('chen2026_image','Chen 2026','extension','chen2026_image'),
 ('ours_ag3','Ours AG3','bands/ag3','waveform_random'),
 ('ours_multiband','Ours multi-band','bands/multiband','waveform_random'),
 ('xiong2025_fcm','Xiong FCM','fcm','xiong2025_fcm')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


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


def permutation(ids,seed,station):
    return np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(f'{seed}|{station}|{ids[i]}'.encode()).digest()))


def load_predictions(workspace,ids,labels,stations,seed):
    choices={'all':np.ones(len(ids),bool),'manual':labels==1};receipt=[]
    root=workspace/'rfqc_benchmark_20260929/results'
    for key,label,sub,arm in SPECS:
        p=root/sub/f'{arm}_seed{seed}'
        complete=json.loads((p/'completion.json').read_text())
        assert all(sha(p/n)==h for n,h in complete['sha256'].items()),str(p)
        config=json.loads((p/'config.json').read_text());assert not config['pretrained'] and config['seed']==seed
        with (p/'test_predictions.csv').open() as f:rows=list(csv.DictReader(f))
        np.testing.assert_array_equal([r['sample_id'] for r in rows],ids)
        np.testing.assert_array_equal([int(r['label']) for r in rows],labels)
        np.testing.assert_array_equal([r['station'] for r in rows],stations)
        values=np.array([float(r['p_good']) for r in rows]);threshold=json.loads((p/'metrics.json').read_text())['test']['threshold']
        selected=values>=threshold
        np.testing.assert_array_equal(selected,[int(r['prediction']) for r in rows])
        choices[key]=selected
        receipt.append(dict(method=key,source=str(p),threshold=threshold,completion_sha256=sha(p/'completion.json'),files=complete['sha256']))
    return choices,receipt


def pair_correlations(waves,valid,mask):
    x=waves[...,mask].copy()
    x/=np.maximum(np.max(np.abs(x),axis=-1,keepdims=True),1e-100)
    x-=x.mean(-1,keepdims=True)
    norms=np.linalg.norm(x,axis=-1)
    x/=np.maximum(norms[...,None],1e-100)
    pairs=[];names=[]
    for i in range(7):
        for j in range(i+1,7):
            v=np.clip(np.sum(x[:,i]*x[:,j],axis=-1),-1,1)
            v[~(valid[:,i]&valid[:,j]&(norms[:,i]>1e-15)&(norms[:,j]>1e-15))]=np.nan
            pairs.append(v);names.append(f'{BANDS[i]:g}-{BANDS[j]:g}')
    return np.stack(pairs,axis=1),names


def main(workspace):
    protocol=json.loads((ROOT/'protocol.json').read_text());seed=protocol['diagnostic_seed']
    cache=workspace/'rf_quality_control/cache';ix=np.load(cache/'test_indices.npy')
    ids=np.load(cache/'sample_ids.npy')[ix];labels=np.load(cache/'labels.npy')[ix];stations=np.load(cache/'stations.npy')[ix]
    gauss=np.load(cache/'gaussians.npy')[ix];length=np.load(cache/'lengths.npy')[ix]
    source=np.load(cache/'waveforms.npy',mmap_mode='r')
    waves=np.asarray(source[ix],dtype=np.float64);valid=np.arange(7)[None,:]<length[:,None]
    assert np.all(gauss[valid]==np.broadcast_to(np.array(BANDS),(len(ix),7))[valid])
    choices,receipts=load_predictions(workspace,ids,labels,stations,seed)
    target=ROOT/'results';target.mkdir(exist_ok=True)
    normalized=waves/np.maximum(np.max(np.abs(waves),axis=-1,keepdims=True),1e-100)
    np.savez_compressed(target/'selection_masks.npz',sample_ids=ids,labels=labels,stations=stations,
        methods=np.array(list(choices)),selected=np.stack(list(choices.values())))
    rows=[];saved_stacks=[];stack_keys=[];subsamples=[]
    for station in sorted(set(stations)):
        where=np.flatnonzero(stations==station)
        for sampling_seed in protocol['subsampling']['seeds']:
            ordered=where[permutation(ids[where],sampling_seed,station)]
            for fraction in protocol['subsampling']['fractions']:
                if fraction==1 and sampling_seed!=protocol['subsampling']['seeds'][0]:continue
                sample=ordered[:max(1,int(len(ordered)*fraction))]
                subsamples.append(dict(station=str(station),seed=sampling_seed,fraction=fraction,
                                       sample_ids=ids[sample].tolist()))
                for band in [3.]:
                    bi=BANDS.index(band)
                    for normalization,data in [('raw',waves),('peak_normalized',normalized)]:
                        manual=sample[choices['manual'][sample]]
                        full_manual=where[choices['manual'][where]]
                        reference=data[manual,bi].mean(0) if len(manual) else None
                        full_reference=data[full_manual,bi].mean(0) if len(full_manual) else None
                        for method,keep in choices.items():
                            selected=sample[keep[sample]]
                            base=dict(station=str(station),method=method,model_seed=seed,sampling_seed=sampling_seed,
                                fraction=fraction,band=band,normalization=normalization,n_input=len(sample),
                                n_selected=len(selected),n_manual=len(manual),retention=len(selected)/len(sample),
                                comparable=bool(len(selected)>=3 and len(manual)>=3))
                            if len(selected) and reference is not None:
                                stack=data[selected,bi].mean(0)
                                base.update(agreement(stack,reference))
                                base['corr_full_manual']=corr(stack[window('whole')],full_reference[window('whole')])
                                if fraction==1:
                                    stack_keys.append('|'.join([str(station),method,normalization]))
                                    saved_stacks.append(stack)
                            else:
                                empty=agreement(np.ones(501),np.ones(501));base.update({k:np.nan for k in empty})
                                base['corr_full_manual']=np.nan
                            rows.append(base)
    table(target/'stack_metrics.csv',rows)
    np.savez_compressed(target/'stacks.npz',keys=np.array(stack_keys),waveforms=np.stack(saved_stacks),time=TIME)
    dump(target/'subsample_membership.json',subsamples)
    pair_rows=[];assoc_rows=[];pairs_all={}
    for name in ['whole','post_p','ps_candidate']:
        scores,pair_names=pair_correlations(waves,valid,window(name));pairs_all[name]=scores
        event_mean=np.nanmean(scores,axis=1)
        for station in sorted(set(stations)):
            where=np.flatnonzero(stations==station)
            order=where[permutation(ids[where],20260929,str(station))]
            for method,keep in choices.items():
                selected=where[keep[where]];reference=order[:len(selected)]
                vals=event_mean[selected];basevals=event_mean[reference]
                assoc_rows.append(dict(station=str(station),method=method,window=name,n_input=len(where),n_selected=len(selected),
                    retention=len(selected)/len(where),n_finite=int(np.isfinite(vals).sum()),
                    mean_correlation=float(np.nanmean(vals)) if len(selected) else np.nan,
                    count_matched_random_mean=float(np.nanmean(basevals)) if len(selected) else np.nan))
                for col,pair in enumerate(pair_names):
                    v=scores[selected,col]
                    pair_rows.append(dict(station=str(station),method=method,window=name,pair=pair,
                        n_valid=int(np.isfinite(v).sum()),mean_correlation=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan))
    table(target/'crossband_by_station.csv',assoc_rows);table(target/'crossband_by_pair.csv',pair_rows)
    np.savez_compressed(target/'crossband_per_record.npz',sample_ids=ids,pair_names=np.array(pair_names),**pairs_all)
    raw=np.array([r['nrmse_whole'] for r in rows if r['fraction']==1 and r['normalization']=='raw' and r['method'] not in ['all','manual']],float)
    audit=dict(protocol_sha256=sha(ROOT/'protocol.json'),analysis_sha256=sha(Path(__file__)),
        records=len(ids),stations=len(set(stations)),diagnostic_seed=seed,methods=len(SPECS),
        source_receipts=receipts,cache_sha256={n:sha(cache/n) for n in ['metadata.json','waveforms.npy','sample_ids.npy','labels.npy','test_indices.npy','gaussians.npy','lengths.npy']},
        input_amplitude_peak_max=float(np.abs(waves).max()),raw_nrmse_max=float(np.nanmax(raw)),
        observed_backazimuth_available=False,observed_hk_executed=False,small_magnitude_executed=False,
        missing_AG5_records=int((~valid[:,6]).sum()),no_waveforms_clipped_or_removed=True,
        assertions=['Every method has exact same test IDs and labels','All source completion hashes checked',
                    'One fixed completed seed for all methods; not best-test seed',
                    'Subsampling fractions fixed before output inspection; identical input subset for all methods'],
        output_sha256={name:sha(target/name) for name in ['selection_masks.npz','stack_metrics.csv','stacks.npz','subsample_membership.json','crossband_by_station.csv','crossband_by_pair.csv','crossband_per_record.npz']})
    dump(ROOT/'audit/analysis_verification.json',audit)
    print(json.dumps({k:audit[k] for k in ['records','stations','methods','diagnostic_seed','input_amplitude_peak_max','raw_nrmse_max','missing_AG5_records']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    main(parser.parse_args().workspace.resolve())
