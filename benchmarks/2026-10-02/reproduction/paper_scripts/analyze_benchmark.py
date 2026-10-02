"""Scratch-only eight-paper benchmark; verify predictions before writing a draft.

Incomplete configurations have no reported mean. --require-complete enforces all
45 results. No training, thresholds, seed selection, or legacy results are altered.
"""
from pathlib import Path
import argparse,csv,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_results import metrics,check,sha

SEEDS=[20260928,20260929,20260930]
METRICS=['accuracy','macro_f1','good_auprc','good_precision','good_recall','balanced_accuracy']
LITERATURE=[('li2021_cnn','Li-CNN (2021)','extension'),('gan2021_cnn','Gan-CNN (2021)','extension'),
 ('gong_cnn','Gong-CNN (2022)','literature'),('gong_cnn_bilstm','Gong-CNN-BiLSTM (2022)','literature'),
 ('gan_cnn','Gan-CNN (2023)','literature'),('deeprfqc','DeepRFQC (2024)','literature'),
 ('hegaz_capsule','RF-Capsule (2025)','literature'),('chen2026_image','Chen-AlexNet (2026)','extension')]


def tex(s):return s.replace('_',r'\_').replace('%',r'\%')


def table(path,rows):
    if not rows:raise ValueError('Refusing an empty result table')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def run(a):
    root=a.workspace/'rfqc_benchmark_20260929';paper=Path(__file__).resolve().parents[1]
    cache=a.workspace/'rf_quality_control/cache_ag3_v1'
    ix=np.load(cache/'test_indices.npy');ids=np.load(cache/'sample_ids.npy')[ix]
    y=np.load(cache/'labels.npy')[ix];stations=np.load(cache/'stations.npy')[ix]
    assert len(y)==24533 and int(y.sum())==2392 and len(set(stations))==26
    specs=[dict(key=arm,label=label,group='primary',source=source,arm=arm,input='AG3 image' if 'image' in arm else 'AG3')
           for arm,label,source in LITERATURE]
    for mode in ['ag3','multiband']:
        specs.append(dict(key='ours_'+mode,label='Ours '+('AG3' if mode=='ag3' else 'multi-band'),group='primary',
             source=mode,arm='waveform_random',input=mode))
    for mode in ['ag3','multiband']:
        for arm,label in [('features_only','Descriptors only'),('combined_random','Waveform + descriptors')]:
            specs.append(dict(key=arm+'_'+mode,label=label+' / '+('AG3' if mode=='ag3' else 'multi-band'),group='features',
                source=mode,arm=arm,input=mode))
    specs.append(dict(key='xiong2025_fcm',label='Xiong-FCM (2025)',group='features',source='fcm',arm='xiong2025_fcm',input='AG3 + station context'))
    roots={n:root/'results'/n for n in ['literature','extension','fcm']}
    roots.update({n:root/'results/bands'/n for n in ['ag3','multiband']})
    protocols={
      'literature':a.workspace/'rf_quality_control/baselines_20260929/protocol.json',
      'extension':root/'protocol.json', 'fcm':root/'protocol.json',
      'ag3':root/'results/bands/protocol.json','multiband':root/'results/bands/protocol.json'}
    if a.require_complete:
        for spec in specs:
            count=sum((roots[spec['source']]/f"{spec['arm']}_seed{s}"/'completion.json').exists() for s in SEEDS)
            if count!=3:raise RuntimeError(f"{spec['key']}: only {count}/3 completed")
    rows=[];receipts=[];summary=[];station_rows=[];probs=[];names=[];status=[];lookup={}
    output=paper/'SourceData/benchmark_v1';output.mkdir(parents=True,exist_ok=True)
    for spec in specs:
        runs=[roots[spec['source']]/f"{spec['arm']}_seed{s}" for s in SEEDS]
        count=sum((p/'completion.json').exists() for p in runs)
        item={**spec,'completed_seeds':count,'required_seeds':3};status.append(item)
        if count<3:
            if a.require_complete:raise RuntimeError(f"{spec['key']}: only {count}/3 completed")
            continue
        values=[]
        for seed,path in zip(SEEDS,runs):
            receipt=json.loads((path/'completion.json').read_text());config=json.loads((path/'config.json').read_text())
            expected={'metrics.json','station_metrics.json','test_predictions.csv','best_model.pt','config.json','preprocessing.json'}
            if spec['source']=='fcm':expected={'config.json','metrics.json','station_metrics.json','model.json','preprocessing.json','test_predictions.csv','validation_predictions.csv'}
            assert set(receipt['sha256'])==expected
            for f,h in receipt['sha256'].items():assert sha(path/f)==h,(path,f)
            assert config['arm']==spec['arm'] and config['seed']==seed and not config['pretrained']
            protocol_path=protocols[spec['source']]
            frozen=json.loads(protocol_path.read_text())
            if spec['source']!='fcm':
                assert config['torch_version']=='2.8.0+cu128' and config['device']=='cuda'
                assert config['epochs']==50 and config['patience']==10 and config['batch_size']==32 and config['balance']=='sampler'
                assert config['actual_split_sizes']==dict(train=101852,val=21309,test=24533)
                assert config['lr']==(1e-5 if spec['arm']=='chen2026_image' else 3e-4)
                assert config['input_samples']==(2048 if spec['source'] in ['ag3','multiband'] else 501)
                assert config['cache_metadata']['split_counts']==json.loads((cache/'metadata.json').read_text())['split_counts']
                assert config['cache_metadata']['station_split']==json.loads((cache/'metadata.json').read_text())['station_split']
                source_hashes=frozen.get('trainer_sources_sha256',frozen.get('source_sha256',frozen.get('sources_sha256')))
                for filename,h in config['resume_contract']['source_sha256'].items():assert source_hashes[filename]==h
                if spec['source'] in ['extension','literature']:assert config['protocol_sha256']==sha(protocol_path)
            else:
                assert config['contract']['protocol_sha256']==sha(protocol_path)
                assert config['contract']['source_sha256']==sha(root/'src/fcm.py')
            with (path/'test_predictions.csv').open() as f:pred=list(csv.DictReader(f))
            np.testing.assert_array_equal([p['sample_id'] for p in pred],ids)
            np.testing.assert_array_equal([int(float(p['label'])) for p in pred],y)
            np.testing.assert_array_equal([p['station'] for p in pred],stations)
            p=np.array([float(v['p_good']) for v in pred]);assert np.isfinite(p).all() and np.min(p)>=0 and np.max(p)<=1
            saved=json.loads((path/'metrics.json').read_text());assert not saved['smoke_only']
            threshold=saved['test']['threshold'];assert threshold==saved['validation']['threshold']
            assert np.min(np.abs(np.linspace(.01,.99,99)-threshold))<1e-12
            np.testing.assert_array_equal([int(v['prediction']) for v in pred],p>=threshold)
            m=metrics(y,p,threshold);check(m,saved['test'],str(path))
            check(metrics(y,p,.5),saved['test_at_05'],str(path)+'/.5')
            station_saved=json.loads((path/'station_metrics.json').read_text());assert set(station_saved)==set(stations)
            for station in sorted(set(stations)):
                keep=stations==station;sm=metrics(y[keep],p[keep],threshold);check(sm,station_saved[str(station)],str(path)+'/'+str(station))
                station_rows.append(dict(key=spec['key'],seed=seed,station=str(station),n=int(keep.sum()),**{k:sm[k] for k in METRICS}))
            value=dict(key=spec['key'],label=spec['label'],group=spec['group'],seed=seed,threshold=threshold,
                       parameters=config.get('trainable_parameters',8),**{k:m[k] for k in METRICS})
            values.append(value);rows.append(value);lookup[spec['key'],seed]=value
            receipts.append(dict(key=spec['key'],seed=seed,source=str(path),sha256=receipt['sha256']))
            folder=output/spec['key']/str(seed);folder.mkdir(parents=True,exist_ok=True)
            for f in ['config.json','metrics.json','station_metrics.json','preprocessing.json','completion.json']:
                shutil.copy2(path/f,folder/f)
            names.append(spec['key']+'/'+str(seed));probs.append(p)
        sm={**spec,'seeds':3,'parameters':values[0]['parameters']}
        for metric in METRICS:
            numbers=np.array([v[metric] for v in values]);sm[metric+'_mean']=float(numbers.mean());sm[metric+'_std']=float(numbers.std(ddof=1))
        summary.append(sm)
    table(output/'run_metrics.csv',rows);table(output/'summary.csv',summary);table(output/'station_metrics.csv',station_rows)
    table(output/'status.csv',status)
    np.savez_compressed(output/'predictions.npz',run_names=np.array(names),sample_ids=ids,labels=y,stations=stations,probabilities=np.stack(probs))
    complete=len(rows)==45
    audit=dict(complete=complete,eligible_completed_runs=len(rows),target_runs=45,
               completion_files_present=sum(s['completed_seeds'] for s in status),
               minimum_unit_for_reporting='all three fixed seeds for a configuration',test_n=len(y),test_good=int(y.sum()),
               test_id_sha256=sha(cache/'sample_ids.npy'),test_indices_sha256=sha(cache/'test_indices.npy'),
               no_pretrained_models=True,no_mps_results=True,metrics_independently_recomputed=True,receipts=receipts)
    (output/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    (paper/'tables/benchmark_status.tex').write_text(
        '\\newcommand{\\BenchmarkVerifiedRuns}{'+str(len(rows))+'}\n'+
        '\\newcommand{\\BenchmarkTargetRuns}{45}\n'+
        '\\newcommand{\\BenchmarkComplete}{'+('1' if complete else '0')+'}\n')
    bykey={v['key']:v for v in summary}
    for group in ['primary','features']:
        body=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Method & Accuracy & Macro-$F_1$ & Good AP & Precision & Recall\\',r'\midrule']
        for s in specs:
            if s['group']!=group:continue
            v=bykey.get(s['key'])
            cells=[]
            for k in ['accuracy','macro_f1','good_auprc','good_precision','good_recall']:
                if not v:cells.append(r'\textit{pending}');continue
                value=f"{100*v[k+'_mean']:.2f}\\pm{100*v[k+'_std']:.2f}"
                # Ranking is reported only after the full matrix is verified.
                if complete and v[k+'_mean']==max(q[k+'_mean'] for q in summary if q['group']==group):
                    value=r'\mathbf{'+value+'}'
                cells.append('$'+value+'$')
            label=s['label'].replace('Waveform + descriptors','Waveform + desc.').replace('Descriptors only','Descriptors')
            body.append(tex(label)+' & '+' & '.join(cells)+r'\\')
        body += [r'\bottomrule',r'\end{tabular}']
        (paper/'tables'/('benchmark_'+group+'.tex')).write_text('\n'.join(body)+'\n')
    pairs=[]
    for arm in ['waveform_random','combined_random','features_only']:
        keys=['ours_'+m if arm=='waveform_random' else arm+'_'+m for m in ['ag3','multiband']]
        if not all(k in bykey for k in keys):continue
        for metric in METRICS:
            d=np.array([lookup[keys[1],s][metric]-lookup[keys[0],s][metric] for s in SEEDS])*100
            pairs.append(dict(arm=arm,metric=metric,mean_pp=float(d.mean()),sd_pp=float(d.std(ddof=1)),**{str(s):float(v) for s,v in zip(SEEDS,d)}))
    table(output/'band_paired_differences.csv',pairs)
    primary=[v for v in summary if v['group']=='primary']
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,k,title in zip(axes,['macro_f1','good_auprc'],['Macro F1 (%)','Good average precision (%)']):
        for i,v in enumerate(primary):
            color='#D55E00' if v['key']=='ours_multiband' else '#0072B2' if v['key']=='ours_ag3' else '#777777'
            ax.errorbar(100*v[k+'_mean'],i,xerr=100*v[k+'_std'],fmt='o',color=color,capsize=3)
        ax.set_yticks(range(len(primary)),[v['label'] for v in primary]);ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Three-seed mean and sample SD'+('' if complete else ' — completed configurations only; draft'))
    fig.savefig(paper/'figures/benchmark_primary.pdf');fig.savefig(paper/'figures/benchmark_primary.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
    station_distributions=[]
    for i,v in enumerate(primary):
        records=[r for r in station_rows if r['key']==v['key']]
        values=np.array([np.mean([r['macro_f1'] for r in records if r['station']==station]) for station in sorted(set(stations))])*100
        station_distributions.append(values)
        color='#D55E00' if v['key']=='ours_multiband' else '#0072B2' if v['key']=='ours_ag3' else '#777777'
        axes[0].plot(values,np.full(len(values),i),'.',color=color,alpha=.45)
        axes[0].plot(values.mean(),i,'D',color=color,markersize=5)
        axes[1].errorbar(100*v['good_recall_mean'],100*v['good_precision_mean'],
             xerr=100*v['good_recall_std'],yerr=100*v['good_precision_std'],fmt='o',color=color,capsize=2)
        axes[1].annotate(str(i+1),(100*v['good_recall_mean'],100*v['good_precision_mean']),xytext=(5,5),textcoords='offset points',fontsize=9)
    axes[0].set_yticks(range(len(primary)),[str(i+1)+'. '+v['label'] for i,v in enumerate(primary)])
    axes[0].invert_yaxis();axes[0].set_xlabel('Station macro F1 (%), averaged over 3 seeds')
    axes[1].set_xlabel('Good recall (%)');axes[1].set_ylabel('Good precision (%)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Held-out station variation and screening trade-offs'+('' if complete else ' — draft'))
    fig.savefig(paper/'figures/benchmark_stations.pdf');fig.savefig(paper/'figures/benchmark_stations.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in audit.items() if k!='receipts'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--require-complete',action='store_true')
    run(p.parse_args())
