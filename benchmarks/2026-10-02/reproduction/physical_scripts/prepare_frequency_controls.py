"""Extract AG1/AG5 on a common observed-band cohort; preserve train/val IDs."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpointing import digest,atomic_json


def main(source):
    arrays={n:np.load(source/(n+'.npy'),mmap_mode='r') for n in
        ['waveforms','features','gaussians','lengths','sample_ids','labels','stations','train_indices','val_indices','test_indices']}
    meta=json.loads((source/'metadata.json').read_text());assert meta['samples']==147694
    valid=np.arange(7)[None,:]<arrays['lengths'][:,None]
    matched={b:((arrays['gaussians']==b)&valid) for b in [1.,3.,5.]}
    eligible=np.flatnonzero(np.logical_and.reduce([m.sum(1)==1 for m in matched.values()]))
    index=np.full(meta['samples'],-1);index[eligible]=np.arange(len(eligible))
    splits={s:index[arrays[s+'_indices']] for s in ['train','val','test']}
    for s in ['train','val']:assert (splits[s]>=0).all(),'Availability changes training/validation cohort'
    splits={s:v[v>=0] for s,v in splits.items()}
    for s in ['train','val']:
        np.testing.assert_array_equal(arrays['sample_ids'][eligible[splits[s]]],arrays['sample_ids'][arrays[s+'_indices']])
    audit=dict(original_records=meta['samples'],common_records=len(eligible),
        excluded_records=meta['samples']-len(eligible),excluded_stations=sorted(set(arrays['stations'][index<0].tolist())),
        split_sizes={s:len(v) for s,v in splits.items()},train_val_sample_ids_and_order_unchanged=True,
        selection='Every row has real AG1, AG3 and AG5; no missing-band imputation',
        source_metadata_sha256=digest(source/'metadata.json'),source_waveform_sha256=digest(source/'waveforms.npy'))
    assert audit['excluded_records']==426 and audit['excluded_stations']==['YP_NE53']
    for band in [1.,5.]:
        out=ROOT/'cache'/f'ag{int(band)}';out.mkdir(parents=True,exist_ok=False)
        pos=matched[band].argmax(1)[eligible]
        for name in ['waveforms','features']:
            values=np.asarray(arrays[name][eligible,pos])[:,None,:]
            assert np.isfinite(values).all();np.save(out/(name+'.npy'),values)
        for name in ['sample_ids','labels','stations']:np.save(out/(name+'.npy'),arrays[name][eligible])
        for s,v in splits.items():np.save(out/(s+'_indices.npy'),v)
        np.save(out/'gaussians.npy',np.full((len(eligible),1),band,dtype=np.float32))
        np.save(out/'lengths.npy',np.ones(len(eligible),dtype=np.int64))
        seen={};duplicates=[];waves=np.load(out/'waveforms.npy',mmap_mode='r')
        split=np.empty(len(eligible),dtype='U5')
        for s,v in splits.items():split[v]=s
        for i,w in enumerate(waves):
            h=hashlib.sha256(w.tobytes()).hexdigest()
            if h in seen:
                j=seen[h];duplicates.append([j,i]);assert split[i]==split[j] and arrays['labels'][eligible[i]]==arrays['labels'][eligible[j]],'Cross-split duplicate or label conflict'
            else:seen[h]=i
        m=dict(meta);m.update(samples=len(eligible),stations=len(set(arrays['stations'][eligible])),max_bands=1,
            variable_bands=False,band_counts={'1':len(eligible)},retained_gaussian_coefficients=[band],
            control_type='AG1/AG3/AG5 observed intersection; one fixed seed',
            station_split={k:v for k,v in meta['station_split'].items() if k not in audit['excluded_stations']},
            split_counts={s:dict(total=len(v),good=int(arrays['labels'][eligible[v]].sum()),bad=int(len(v)-arrays['labels'][eligible[v]].sum())) for s,v in splits.items()},
            exact_singleband_duplicates=len(duplicates))
        atomic_json(out/'metadata.json',m);atomic_json(out/'derivation_audit.json',dict(**audit,band=band,duplicates=duplicates,
             cache_sha256={p.name:digest(p) for p in sorted(out.glob('*')) if p.is_file()}))
    atomic_json(ROOT/'audit/frequency_cache_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);main(p.parse_args().source)
