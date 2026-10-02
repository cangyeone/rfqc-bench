"""Verify aggregate/timing sources and rebuild the public comparison report.

No private RFs, trained weights, GPU, downloads or retraining are required.
Use --plots with the optional matplotlib extra to regenerate the figure.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {'ours_ag3':'reference_ag3', 'ours_multiband':'reference_multifilter',
           'features_only_ag3':'descriptors_ag3', 'features_only_multiband':'descriptors_multifilter',
           'combined_random_ag3':'combined_ag3', 'combined_random_multiband':'combined_multifilter'}
LABELS = {'reference_ag3':'Reference–AG3','reference_multifilter':'Reference–multi-filter',
          'gong_cnn':'Gong-CNN','gong_cnn_bilstm':'Gong-CNN–BiLSTM'}


def rows(path):return list(csv.DictReader(path.open()))
def f(x):return float(x)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, body):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(map(str,r))+' |' for r in body])+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=ROOT/'benchmarks/2026-10-02')
    p.add_argument('--output',type=Path,help='Output directory, default SNAPSHOT/generated')
    p.add_argument('--plots',action='store_true')
    a=p.parse_args();r=a.snapshot/'results';out=a.output or a.snapshot/'generated';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.snapshot/'manifest.json').read_text())
    for name,info in manifest['files'].items():assert sha(a.snapshot/name)==info['public_sha256'],name
    runs=rows(r/'original_runs.csv');original=rows(r/'original_summary.csv')
    assert len(runs)==45 and len(original)==15
    metrics=['accuracy','macro_f1','good_auprc','good_precision','good_recall']
    accuracy=[]
    for x in original:
        selected=[s for s in runs if s['key']==x['key']];assert len(selected)==3
        item=dict(method=ALIASES.get(x['key'],x['key']),label=x['label'],group=x['group'])
        item['label']=LABELS.get(item['method'],item['label']).replace('Ours','Reference')
        for key in metrics:
            values=np.array([f(y[key]) for y in selected]);mean=float(values.mean());sd=float(values.std(ddof=1))
            np.testing.assert_allclose([mean,sd],[f(x[key+'_mean']),f(x[key+'_std'])],atol=1e-12,rtol=1e-10)
            item[key+'_mean']=100*mean;item[key+'_sd']=100*sd
        accuracy.append(item)
    log=rows(r/'logistic_runs.csv');assert len(log)==6
    for variant in ['ag3','multiband']:
        selected=[x for x in log if x['variant']==variant];assert len(selected)==3
        name='logreg_'+('ag3' if variant=='ag3' else 'multifilter')
        item=dict(method=name,label='LogReg–'+('AG3' if variant=='ag3' else 'multi-filter'),group='features')
        for key in metrics:
            vals=np.array([f(x[key]) for x in selected])*100
            item[key+'_mean']=float(vals.mean());item[key+'_sd']=float(vals.std(ddof=1))
        accuracy.append(item)
    timings={}
    timing_files=list((a.snapshot/'timings').glob('*.json'));assert len(timing_files)==57
    hashes=set()
    for file in timing_files:
        d=json.loads(file.read_text());key=(d['method'],d['round']);assert key not in timings
        if d['kind']=='fcm':
            np.testing.assert_allclose(d['records']/d['pool_seconds'],d['pool_records_per_second'],rtol=1e-12)
        else:
            hashes.add(d['sample_ids_sha256']);single=np.array(d['latency_seconds']);bulk=np.array(d['batch32_seconds'])
            assert len(single)==100 and len(bulk)==32 and d['records']==512
            np.testing.assert_allclose([1000*np.median(single),1000*np.percentile(single,95),1024/bulk.sum()],
                                      [d['latency_p50_ms'],d['latency_p95_ms'],d['batch32_records_per_second']],rtol=1e-12)
            if d['kind']=='neural':
                np.testing.assert_allclose(1024/sum(d['forward_batch32_seconds']),d['forward_batch32_records_per_second'],rtol=1e-12)
        timings[key]=d
    assert len(hashes)==1
    summary=rows(r/'inference_summary.csv');cost={x['method']:x for x in summary};assert len(cost)==19
    for x in summary:
        selected=[timings[x['method'],i] for i in range(3)]
        keys=['pool_seconds','pool_records_per_second'] if x['method']=='xiong2025_fcm' else ['latency_p50_ms','latency_p95_ms','batch32_records_per_second']
        for key in keys:
            vals=[d[key] for d in selected]
            np.testing.assert_allclose([np.median(vals),min(vals),max(vals)],[f(x[key]),f(x[key+'_min']),f(x[key+'_max'])],rtol=1e-12)
    seedrows=rows(r/'eight_seed_metrics.csv');assert len(seedrows)==32
    values={n:np.array([f(x['macro_f1_percent']) for x in sorted(seedrows,key=lambda z:int(z['seed'])) if x['method']==n]) for n in LABELS}
    assert all(len(v)==8 for v in values.values())
    pairs=rows(r/'eight_seed_summary.csv');assert len(pairs)==6
    for x in pairs:
        d=values[x['first']]-values[x['second']];sd=d.std(ddof=1);half=student_t.ppf(.975,7)*sd/np.sqrt(8)
        np.testing.assert_allclose([d.mean(),sd,d.mean()-half,d.mean()+half],
                                  [f(x[k]) for k in ['mean_pp','sample_sd_pp','lower_pp','upper_pp']],atol=1e-11)
    cross=rows(r/'cross_protocol_metrics.csv');counts=rows(r/'cross_protocol_counts.csv');assert len(cross)==6 and len(counts)==12
    for x in counts:
        tn,fp,fn,tp=[int(x[k]) for k in ['tn','fp','fn','tp']];assert tn+fp+fn+tp==18305
        val=50*(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn))
        run=next(s for s in cross if x['run']==s['direction']+'_seed'+s['seed'])
        np.testing.assert_allclose(val,f(run[x['evaluation_labels']+'_macro_f1_percent']),rtol=1e-12)
    for x in rows(r/'label_agreement.csv'):
        aa,ab,ba,bb=[int(x[k]) for k in ['strict_bad_ag3_bad','strict_bad_ag3_good','strict_good_ag3_bad','strict_good_ag3_good']]
        n=aa+ab+ba+bb;observed=(aa+bb)/n;expected=((aa+ab)*(aa+ba)+(ba+bb)*(ab+bb))/n**2
        np.testing.assert_allclose([observed,(observed-expected)/(1-expected)],[f(x['agreement']),f(x['kappa'])],rtol=1e-12)

    lines=['# Completed RFQC comparisons — 2026-10-02\n',
           'Source manuscript commit: `'+manifest['paper_commit']+'`. These are documented adaptations under a fixed task, not each source paper’s optimal pipeline.\n',
           '## Original three-seed classification and measured deployment cost\n',
           'The ten waveform/image configurations below share 24,533 held-out RFs. Scores are mean ± sample SD across three training seeds. Timing uses fixed seed 20260929, one RTX 5090, FP32 eager execution and three measurement rounds. RF/s includes API preprocessing and transfers at batch 32. Bold identifies the highest original mean in each score column, not significance. Additional views distinguish the multi-filter reference.\n']
    primary=[x for x in accuracy if x['group']=='primary'];assert len(primary)==10
    heads=['Configuration','Accuracy (%)','Macro-F1 (%)','Good AP (%)','p50 (ms)','API RF/s']
    body=[]
    for x in primary:
        vals=[]
        for k in metrics[:3]:
            value=f"{x[k+'_mean']:.2f} ± {x[k+'_sd']:.2f}"
            if x[k+'_mean']==max(y[k+'_mean'] for y in primary):value='**'+value+'**'
            vals.append(value)
        c=cost[x['method']];body.append([x['label'],*vals,f"{f(c['latency_p50_ms']):.2f}",f"{f(c['batch32_records_per_second']):,.0f}"])
    lines += [table(heads,body),'## Descriptor and station-context controls\n',
              'These input regimes are separate from waveform-only prediction. FCM uses a complete station pool on CPU; its rate is not independent-record batch throughput. LogReg runs on CPU. Supplied-descriptor extraction is excluded from timing.\n']
    body=[]
    for x in accuracy:
        if x in primary:continue
        c=cost[x['method']];pool=x['method']=='xiong2025_fcm';rate=f(c['pool_records_per_second'] if pool else c['batch32_records_per_second'])
        body.append([x['label'],*[f"{x[k+'_mean']:.2f} ± {x[k+'_sd']:.2f}" for k in metrics[:3]],c['device'],f'{rate:,.0f}'+(' (full pool)' if pool else '')])
    lines += [table(['Configuration','Accuracy (%)','Macro-F1 (%)','Good AP (%)','Device','RF/s'],body),
              '## Appended eight-seed results\n',
              'The original three-seed table above remains unchanged. Five additional fits for each of four configurations provide eight seeds. Other methods retain three fits; no nonexistent extra seeds are inferred.\n',
              table(['Configuration','Macro-F1 (%)'],[[LABELS[k],f'{v.mean():.3f} ± {v.std(ddof=1):.3f}'] for k,v in values.items()]),
              table(['Paired contrast','Mean difference (pp)','SD (pp)','95% interval (pp)'],
                    [[LABELS[x['first']]+' − '+LABELS[x['second']],f"{f(x['mean_pp']):+.3f}",f"{f(x['sample_sd_pp']):.3f}",f"[{f(x['lower_pp']):+.3f}, {f(x['upper_pp']):+.3f}]"] for x in pairs]),
              'Intervals are paired Student-t intervals, unadjusted for multiple comparisons and conditional on the fixed split. The multi-filter-minus-AG3 and BiLSTM-minus-CNN intervals cross zero; both Gong-minus-multi-filter intervals lie just above zero. This is neither global equivalence nor a general multi-filter advantage.\n',
              '## Label protocols and perturbation controls\n']
    body=[]
    for direction in ['strict_to_ag3','ag3_to_strict']:
        group=[x for x in cross if x['direction']==direction]
        fields=[]
        for k in ['source_macro_f1_percent','target_macro_f1_percent','target_ap_percent']:
            v=np.array([f(x[k]) for x in group]);fields.append(f'{v.mean():.3f} ± {v.std(ddof=1):.3f}')
        d=np.array([f(x['target_macro_f1_percent'])-f(x['source_macro_f1_percent']) for x in group])
        body.append([direction.replace('_',' '),*fields,f'{d.mean():+.3f} ± {d.std(ddof=1):.3f}'])
    lines += [table(['Source → target','Source F1 (%)','Target F1 (%)','Target AP (%)','F1 change (pp)'],body),
              'Both directions use the same 18,305 paired test records at 22 stations. Achieved scores are not a theoretical label-system ceiling and cannot be pooled with the original test cohort. Labels agree on 98.04% of 116,722 paired records, with 2,287 strict-bad/AG3-good changes and zero reverse changes.\n',
              'Noise and synthetic-label perturbation curves are in `results/input_perturbation.csv` and `results/label_perturbation.csv`. Smallest detected effects are specific to the tested grid, not universal resolution limits. Their station-bootstrap intervals require record-level inputs to recompute; those data are not distributed here.\n',
              '## RF waveform diagnostics\n']
    physical=rows(r/'physical_summary.csv')
    lines += [table(['Selection','Retained RFs','Median shape r','Median raw NRMSE','P amplitude error (%)','Ps-candidate error (%)','Cross-filter r'],
                    [[x['label'],x['retained_total'],f"{f(x['median_shape_correlation']):.4f}",f"{f(x['median_raw_nrmse']):.4g}",f"{f(x['median_raw_P_amplitude_relative_error_pct']):.3g}",f"{f(x['median_raw_Ps_amplitude_relative_error_pct']):.3g}",f"{f(x['median_crossband_correlation']):.4f}"] for x in physical]),
              'These fixed-seed diagnostics summarize station medians and shared-record selections. Ps is an operational waveform candidate, not an independently identified Moho phase. Similar stacks do not establish improved H–κ or structural inversion; no such inversion was performed. Half/quarter-pool summaries are provided separately.\n',
              '## Single-filter sensitivity on the common-availability cohort\n']
    freq=rows(r/'frequency_control.csv');assert len(freq)==4 and {int(x['n']) for x in freq}=={24107}
    lines += [table(['Input','Accuracy (%)','Macro-F1 (%)','Good AP (%)'],
                    [[x['band'].replace('multiband','multi-filter'),*[f'{100*f(x[k]):.2f}' for k in metrics[:3]]] for x in freq]),
              'This is one fixed seed on 24,107 common test records. It is not a three- or eight-seed result and does not choose a universally optimal Gaussian coefficient. AG denotes a Gaussian coefficient, not hertz.\n',
              '## Full inference comparison\n',
              table(['Configuration','Device','p50 / p95 (ms)','API RF/s','Round range (RF/s)'],
                    [[x['label'].replace('--','–'),x['device'],('--' if x['method']=='xiong2025_fcm' else f"{f(x['latency_p50_ms']):.3f} / {f(x['latency_p95_ms']):.3f}"),
                      f"{f(x['pool_records_per_second'] if x['method']=='xiong2025_fcm' else x['batch32_records_per_second']):,.0f}"+(' (pool)' if x['method']=='xiong2025_fcm' else ''),
                      f"[{f(x['pool_records_per_second_min'] if x['method']=='xiong2025_fcm' else x['batch32_records_per_second_min']):,.0f}, {f(x['pool_records_per_second_max'] if x['method']=='xiong2025_fcm' else x['batch32_records_per_second_max']):,.0f}]"] for x in summary]),
              'Timing excludes disk/model loading, initial RFData construction, RF production, supplied-descriptor extraction and HTTP. FCM times all 24,533 records at 26 stations after one warm-up. The other methods share 512 complete-view records, 100 single calls and 32 batch-32 calls per round after 10 warm-ups per workload. Forward-only rates and peak allocated GPU memory are retained in the CSV. Timing ranges are not training-seed uncertainty.\n']
    ratio=f(cost['gong_cnn']['batch32_records_per_second'])/f(cost['gong_cnn_bilstm']['batch32_records_per_second'])
    lines += [f'Gong-CNN has {ratio:.1f}× the measured API batch throughput of Gong-CNN–BiLSTM on this machine. The API throughput of the multi-filter reference is {100*f(cost["reference_multifilter"]["batch32_records_per_second"])/f(cost["reference_ag3"]["batch32_records_per_second"]):.1f}% of its AG3 counterpart. No timing result changes the archived accuracy scores.\n']
    (out/'COMPARISON.md').write_text('\n'.join(lines))
    with (out/'classification_summary.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(accuracy[0]),lineterminator='\n');w.writeheader();w.writerows(accuracy)
    if a.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True,layout='constrained')
        for i,x in enumerate(primary):
            c=cost[x['method']];color='#ba573a' if x['method']=='reference_multifilter' else '#246887'
            axes[0].errorbar(x['macro_f1_mean'],i,xerr=x['macro_f1_sd'],fmt='o',color=color,capsize=3)
            mean=f(c['batch32_records_per_second'])
            axes[1].errorbar(mean,i,xerr=[[mean-f(c['batch32_records_per_second_min'])],[f(c['batch32_records_per_second_max'])-mean]],fmt='o',color=color,capsize=3)
        axes[0].set_yticks(range(len(primary)),[x['label'] for x in primary]);axes[0].invert_yaxis()
        axes[0].set_xlim(85.5,90);axes[0].set_xlabel('Macro-F1 (%)\nThree training seeds: mean ± SD')
        axes[1].set_xscale('log');axes[1].set_xlabel('API RF/s, batch 32\nThree timing rounds: median and range')
        for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
        fig.savefig(out/'accuracy_speed.png',dpi=170);plt.close(fig)
    report=dict(passed=True,source_files_hashed=len(manifest['files']),original_fits=51,appended_seed_fits=20,
                cross_protocol_fits=6,frequency_fits=2,timing_records=57,paired_seed_contrasts=6,
                raw_data_included=False,scope='Aggregate means/SDs/paired t intervals, confusion-count F1, label agreement and raw-duration summaries verified; no recomputation of per-record accuracy or station-bootstrap draws without data.')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
