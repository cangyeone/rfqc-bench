#!/usr/bin/env python3
"""Verify the frozen 25-run local matrix and regenerate paper results.

Default: recompute from the portable SourceData bundle (NumPy + matplotlib).
--workspace PATH: first audit original runs, cache cohort, and completion SHA256,
then refresh the portable bundle. Does not train or modify original run artifacts.
"""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,platform
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER=Path(__file__).resolve().parents[1]
OUT=PAPER/'SourceData'/'formal_local_v2'
ARMS=['waveform_random','waveform_pretrained','combined_random','combined_pretrained','features_only']
SHORT=dict(zip(ARMS,['W-R','W-P','WC-R','WC-P','C']))
SEEDS=[20260928,20260929,20260930,20261001,20261002]
METRICS=['accuracy','balanced_accuracy','macro_f1','good_precision','good_recall','bad_recall','good_auprc','auroc']
CONTRASTS=[('combined_pretrained','waveform_pretrained'),('combined_random','waveform_random'),('combined_pretrained','combined_random'),('waveform_pretrained','waveform_random'),('combined_pretrained','features_only')]
COLORS=['#666666','#0072B2','#D55E00','#009E73','#CC79A7']
TOL=1e-12

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(path,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def tex_table(name,spec,header,lines):
 text=r'\begin{tabular}{'+spec+'}\n'+r'\toprule'+'\n'+header+r' \\'+'\n'+r'\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
 (PAPER/'tables'/name).write_text(text)
def ratio(a,b):return float(a/b) if b else 0.
def metrics(y,p,t):
 """Independent confusion-count, grouped-score AP, and pair-count AUC formulas."""
 pred=p>=t
 tn=int(np.sum((y==0)&~pred));fp=int(np.sum((y==0)&pred))
 fn=int(np.sum((y==1)&~pred));tp=int(np.sum((y==1)&pred))
 n1=tp+fn;n0=tn+fp
 ap=auc=None
 if n1 and n0:
  order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
  ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
  cum=np.cumsum(ys)[ends]
  ap=float(np.sum(np.diff(np.r_[0,cum])/n1*cum/(ends+1)))
  order=np.argsort(p,kind='stable');ys=y[order];ps=p[order]
  ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
  pos=np.diff(np.r_[0,np.cumsum(ys)[ends]])
  neg=np.diff(np.r_[0,np.cumsum(1-ys)[ends]])
  auc=float(np.sum(pos*(np.cumsum(neg)-neg+0.5*neg))/(n1*n0))
 return dict(accuracy=ratio(tp+tn,len(y)),balanced_accuracy=(ratio(tp,n1)+ratio(tn,n0))/2 if n1 and n0 else None,
  macro_f1=(ratio(2*tp,2*tp+fp+fn)+ratio(2*tn,2*tn+fp+fn))/2,
  good_precision=ratio(tp,tp+fp),good_recall=ratio(tp,n1),bad_recall=ratio(tn,n0),
  good_auprc=ap,auroc=auc,confusion_matrix_bad_good=[[tn,fp],[fn,tp]],threshold=float(t),n=len(y))
def check(a,b,where):
 if isinstance(a,dict):
  for k,v in a.items():check(v,b[k],where+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),where
  for i,(x,z) in enumerate(zip(a,b)):check(x,z,where+'/'+str(i))
 elif a is None:assert b is None,(where,a,b)
 else:assert abs(a-b)<TOL,(where,a,b)

def bundle(workspace):
 root=workspace/'rf_quality_control/runs/formal_local_v2';cache=workspace/'rf_quality_control/cache'
 expected={f'{a}_seed{s}' for a in ARMS for s in SEEDS}
 actual={f.parent.name for f in root.glob('*/completion.json')}
 assert actual==expected,(actual-expected,expected-actual)
 status=read(root/'status.json');assert status['state']=='completed' and status['completed_runs']==25 and not status['failed_runs']
 ids=np.load(cache/'test_indices.npy');sample=np.load(cache/'sample_ids.npy',mmap_mode='r')[ids]
 y=np.load(cache/'labels.npy',mmap_mode='r')[ids].astype(int);station=np.load(cache/'stations.npy',mmap_mode='r')[ids]
 lengths=np.load(cache/'lengths.npy',mmap_mode='r')[ids]
 assert len(ids)==24533 and y.sum()==2392 and len(set(station))==26 and len(set(sample))==len(sample)
 meta=read(cache/'metadata.json');source_hashes=None;prep=None;probs=[];names=[];receipts=[]
 OUT.mkdir(parents=True,exist_ok=True)
 for arm in ARMS:
  for seed in SEEDS:
   name=f'{arm}_seed{seed}';run=root/name;manifest=read(run/'completion.json')
   assert len(manifest['sha256'])==6
   for fn,h in manifest['sha256'].items():assert sha(run/fn)==h,(name,fn)
   config=read(run/'config.json');m=read(run/'metrics.json');history=read(run/'history.json')
   assert config['arm']==m['arm']==arm and config['seed']==m['seed']==seed
   assert not m['smoke_only'] and config['device']=='mps' and config['torch_version']=='2.11.0'
   assert config['cache_metadata']==meta and config['actual_split_sizes']==dict(train=101852,val=21309,test=24533)
   for k,v in dict(epochs=50,patience=10,batch_size=32,lr=3e-4,input_samples=2048,balance='sampler',smoke_samples=0).items():assert config[k]==v,(name,k)
   sources=config['resume_contract']['source_sha256']
   if source_hashes is None:source_hashes=sources
   assert sources==source_hashes
   for fn,h in sources.items():assert sha(workspace/'rf_quality_control'/fn)==h,(name,fn)
   ph=sha(run/'preprocessing.json')
   if prep is None:prep=ph
   assert prep==ph
   assert len(history)==m['epochs_completed'] and [r['epoch'] for r in history]==list(range(1,len(history)+1))
   best=max(history,key=lambda r:r['validation']['good_auprc'])
   assert abs(best['validation']['good_auprc']-m['validation']['good_auprc'])<TOL
   with (run/'test_predictions.csv').open() as f:rows=list(csv.DictReader(f))
   np.testing.assert_array_equal([r['sample_id'] for r in rows],sample)
   np.testing.assert_array_equal([r['station'] for r in rows],station)
   np.testing.assert_array_equal([int(r['label']) for r in rows],y)
   p=np.array([float(r['p_good']) for r in rows],dtype=np.float64)
   assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
   np.testing.assert_array_equal([int(r['prediction']) for r in rows],p>=m['test']['threshold'])
   assert abs(m['validation']['threshold']-m['test']['threshold'])<TOL
   assert np.min(abs(np.linspace(.01,.99,99)-m['test']['threshold']))<TOL
   probs.append(p);names.append(name)
   dst=OUT/name;dst.mkdir(exist_ok=True)
   for fn in ['metrics.json','station_metrics.json','config.json','preprocessing.json','completion.json','history.json']:
    shutil.copyfile(run/fn,dst/fn)
   receipts.append(dict(run=name,source=str(run),sha256=manifest['sha256'],history_sha256=sha(run/'history.json'),best_epoch=best['epoch']))
 np.savez_compressed(OUT/'predictions.npz',run_names=np.array(names),sample_ids=sample,stations=station,labels=y,lengths=lengths,probabilities=np.stack(probs))
 for fn in ['summary.csv','paired_differences.json','status.json']:shutil.copyfile(root/fn,OUT/('original_'+fn))
 dump(OUT/'source_audit.json',dict(verified_at=datetime.now(timezone.utc).isoformat(),source_root=str(root),run_count=25,
  test_records=len(ids),test_good=int(y.sum()),test_bad=int((y==0).sum()),test_stations=len(set(station)),
  cache_sha256={f:sha(cache/f) for f in ['metadata.json','test_indices.npy','sample_ids.npy','labels.npy','stations.npy','lengths.npy']},
  prediction_bundle_sha256=sha(OUT/'predictions.npz'),source_sha256=source_hashes,runs=receipts,
  scope='All original six-file completion hashes and exact ordered cache test IDs/labels checked. Portable bundle stores the exact CSV float64 scores. Validation thresholds checked against stored reports and trainer code; validation inference not repeated.'))

def analyze():
 data=np.load(OUT/'predictions.npz',allow_pickle=False);y=data['labels'];station=data['stations'];names=data['run_names'];ps=data['probabilities']
 assert sha(OUT/'predictions.npz')==read(OUT/'source_audit.json')['prediction_bundle_sha256']
 assert len(names)==25 and len(set(names))==25 and len(set(data['sample_ids']))==24533
 results=[];station_rows=[];lookup={};pr={};audit_deltas=[]
 for name,p in zip(names,ps):
  run=OUT/str(name);saved=read(run/'metrics.json');arm=saved['arm'];seed=saved['seed'];t=saved['test']['threshold']
  for fn,h in read(run/'completion.json')['sha256'].items():
   if (run/fn).exists():assert sha(run/fn)==h
  m=metrics(y,p,t);check(m,saved['test'],str(name)+'/test');check(metrics(y,p,.5),saved['test_at_05'],str(name)+'/test_at_05')
  sm=read(run/'station_metrics.json');assert set(sm)==set(station)
  for st in sorted(set(station)):
   take=station==st;x=metrics(y[take],p[take],t);check(x,sm[st],str(name)+'/'+str(st))
   station_rows.append(dict(arm=arm,seed=seed,station=str(st),n=int(take.sum()),n_good=int(y[take].sum()),n_bands=int(data['lengths'][take][0]),**{k:x[k] for k in METRICS}))
  station_f1=float(np.mean([x['macro_f1'] for x in sm.values()]));assert abs(station_f1-saved['station_macro_f1'])<TOL
  history=read(run/'history.json');best=max(history,key=lambda r:r['validation']['good_auprc'])
  row=dict(arm=arm,seed=seed,threshold=t,epochs=saved['epochs_completed'],best_epoch=best['epoch'],station_macro_f1=station_f1,**{k:m[k] for k in METRICS})
  results.append(row);lookup[arm,seed]=row;pr[arm,seed]=p
 table(OUT/'metrics_recomputed.csv',results);table(OUT/'station_metrics_recomputed.csv',station_rows)
 aggregate=[]
 for arm in ARMS:
  rows=[r for r in results if r['arm']==arm];row=dict(arm=arm,seeds=5)
  for k in METRICS+['station_macro_f1']:
   v=np.array([r[k] for r in rows]);row[k+'_mean']=float(v.mean());row[k+'_std']=float(v.std(ddof=1))
  aggregate.append(row)
 table(OUT/'summary_recomputed.csv',aggregate)
 with (OUT/'original_summary.csv').open() as f:
  for orig in csv.DictReader(f):
   row=next(r for r in aggregate if r['arm']==orig['arm']);assert orig['seeds']=='5' and orig['smoke_only']=='False'
   for k in METRICS:
    for stat in ['mean','std']:assert abs(row[k+'_'+stat]-float(orig[k+'_'+stat]))<TOL
 paired=[];orig_pairs=read(OUT/'original_paired_differences.json')
 for left,right in CONTRASTS:
  original=next(r for r in orig_pairs if r['left']==left and r['right']==right)
  for k in METRICS+['station_macro_f1']:
   d=np.array([lookup[left,s][k]-lookup[right,s][k] for s in SEEDS])
   if k in METRICS:
    check(dict(paired_differences=d.tolist(),mean=float(d.mean()),std=float(d.std(ddof=1))),original[k],left+'-'+right+'/'+k)
   paired.append(dict(left=left,right=right,metric=k,mean_pp=float(d.mean()*100),std_pp=float(d.std(ddof=1)*100),positive_seeds=int((d>0).sum()),**{str(s):float(x*100) for s,x in zip(SEEDS,d)}))
 table(OUT/'paired_differences_recomputed.csv',paired)
 station_agg=[]
 for st in sorted(set(station)):
  for arm in ARMS:
   rows=[r for r in station_rows if r['arm']==arm and r['station']==st]
   row={k:rows[0][k] for k in ['arm','station','n','n_good','n_bands']}
   for k in METRICS:
    v=[r[k] for r in rows];row[k+'_mean']=float(np.mean(v)) if all(x is not None for x in v) else None;row[k+'_std']=float(np.std(v,ddof=1)) if all(x is not None for x in v) else None
   station_agg.append(row)
 table(OUT/'station_summary.csv',station_agg)
 figs(aggregate,paired,station_agg,results)
 latex_tables(aggregate,paired,station_agg)
 wc=[r for r in results if r['arm']=='combined_pretrained']
 wc05=[metrics(y,pr['combined_pretrained',s],.5) for s in SEEDS]
 confusions=[];confusion_lines=[]
 for label,vals in [('Validation-selected',[metrics(y,pr['combined_pretrained',s],lookup['combined_pretrained',s]['threshold']) for s in SEEDS]),('Fixed 0.5',wc05)]:
  mats=np.array([r['confusion_matrix_bad_good'] for r in vals])
  mean=mats.mean(axis=0);sd=mats.std(axis=0,ddof=1)
  confusions.append(dict(threshold_rule=label,mean=mean.tolist(),sample_sd=sd.tolist()))
  confusion_lines.append(' & '.join([label]+[f'${v:.1f} \\pm {e:.1f}$' for v,e in zip(mean.ravel(),sd.ravel())])+r' \\')
 tex_table('confusion.tex','@{}lrrrr@{}','Threshold rule & TN & FP & FN & TP',confusion_lines)
 details=dict(class_prevalence=float(y.mean()),all_bad=metrics(y,np.zeros(len(y)),.5),
  wc_p_confusion_matrices=confusions,
  wc_p_threshold_range=[min(r['threshold'] for r in wc),max(r['threshold'] for r in wc)],
  wc_p_at_05={k:dict(mean=float(np.mean([r[k] for r in wc05])),std=float(np.std([r[k] for r in wc05],ddof=1))) for k in METRICS},
  wc_p_station_range={k:[min(r[k+'_mean'] for r in station_agg if r['arm']=='combined_pretrained'),max(r[k+'_mean'] for r in station_agg if r['arm']=='combined_pretrained')] for k in ['macro_f1','good_auprc']},
  missing_band_station=[r for r in station_agg if r['station']=='YP_NE53'],
  epochs={a:[min(r['epochs'] for r in results if r['arm']==a),max(r['epochs'] for r in results if r['arm']==a)] for a in ARMS})
 dump(OUT/'analysis_details.json',details)
 dump(OUT/'verification.json',dict(verified_at=datetime.now(timezone.utc).isoformat(),runs=25,records_per_run=len(y),stations=26,
  checked_metric_sets=25*(2+26),tolerance=TOL,method='Independent NumPy confusion counts, tied-score AP and positive-negative pair AUC; no trainer metrics helper or sklearn metric calls.',
  checks=['25 unique required arm/seed pairs','exact shared sample IDs, station names and labels checked against cache during bundling','6 original artifacts hashed per completion manifest','all test metrics at validation-selected and 0.5 thresholds','all per-station metrics','original mean and sample SD summary','all original same-seed paired differences','stored best validation AP equals history maximum'],
  limitations=['No repeat validation inference; saved thresholds verified via code, grid membership and report consistency.','Seed SD describes optimization variation on one fixed split, not a generalization confidence interval.','No AG3-only control or structure inversion.'],
  prediction_bundle_sha256=sha(OUT/'predictions.npz'),python=platform.python_version(),numpy=np.__version__))
 print(json.dumps(dict(summary=aggregate,paired=[r for r in paired if r['metric'] in ['macro_f1','good_auprc']],details=details),indent=2))

def figs(aggregate,paired,stations,results):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fd=PAPER/'figures';fd.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(7,3.5),layout='constrained')
 for ax,k,title in zip(axes,['macro_f1','good_auprc'],['(a) Macro-F1','(b) Accepted-class AP']):
  for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
   vals=np.array([r[k] for r in results if r['arm']==arm])*100
   ax.scatter(np.full(5,i)+np.linspace(-.12,.12,5),vals,s=17,color=color,alpha=.6)
   ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='D',markersize=5,color=color,capsize=4)
  ax.set_xticks(range(5),[SHORT[a] for a in ARMS]);ax.set_ylabel('Score (%)');ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.2);ax.set_ylim(73,94)
 fig.savefig(fd/'ablation_results.pdf');fig.savefig(fd/'ablation_results.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(7,3.5),layout='constrained')
 for ax,k,title in zip(axes,['macro_f1','good_auprc'],['(a) Macro-F1 difference','(b) AP difference']):
  for i,(left,right) in enumerate(CONTRASTS[:4]):
   r=next(x for x in paired if (x['left'],x['right'],x['metric'])==(left,right,k))
   ax.scatter([r[str(s)] for s in SEEDS],np.full(5,i)+np.linspace(-.10,.10,5),s=15,color='#0072B2',alpha=.6)
   ax.errorbar(r['mean_pp'],i,xerr=r['std_pp'],fmt='D',markersize=5,color='#222222',capsize=4)
  ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(4),[SHORT[a]+' − '+SHORT[b] for a,b in CONTRASTS[:4]]);ax.invert_yaxis();ax.set_xlim(-2.5,3.7);ax.set_xlabel('Paired difference (percentage points)');ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.2)
 fig.savefig(fd/'paired_ablation.pdf');fig.savefig(fd/'paired_ablation.png',dpi=200);plt.close(fig)
 order=sorted(set(r['station'] for r in stations));mat=np.array([[next(r['macro_f1_mean'] for r in stations if r['station']==s and r['arm']==a) for a in ARMS] for s in order])*100
 fig,ax=plt.subplots(figsize=(7,7.2),layout='constrained');im=ax.imshow(mat,cmap='cividis',vmin=50,vmax=100,aspect='auto')
 labels=[]
 for s in order:
  r=next(r for r in stations if r['station']==s);labels.append(f"{s}{'*' if r['n_bands']==6 else ''}  ({r['n_good']}/{r['n']})")
 ax.set_yticks(range(len(order)),labels);ax.set_xticks(range(5),[SHORT[a] for a in ARMS]);ax.tick_params(length=0);ax.set_title('Mean test macro-F1 by station (%)\nStation (accepted / total records)',loc='left',pad=10)
 for i in range(len(order)):
  for j in range(5):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]<77 else 'black')
 fig.colorbar(im,ax=ax,fraction=.04,pad=.03,label='Mean macro-F1 (%)');fig.savefig(fd/'station_results.pdf');fig.savefig(fd/'station_results.png',dpi=180);plt.close(fig)

def latex_tables(aggregate,paired,stations):
 td=PAPER/'tables';td.mkdir(exist_ok=True)
 lines=[]
 for r in aggregate:
  cells=[SHORT[r['arm']]]
  for k in ['accuracy','balanced_accuracy','macro_f1','good_auprc','good_precision','good_recall']:
   cells.append(f"${100*r[k+'_mean']:.2f} \\pm {100*r[k+'_std']:.2f}$")
  lines.append(' & '.join(cells)+r' \\')
 tex_table('main_results.tex','@{}lrrrrrr@{}',r'Arm & Accuracy & Balanced acc. & Macro-$F_1$ & AP & Precision & Recall',lines)
 lines=[]
 for left,right in CONTRASTS:
  cells=[SHORT[left]+' -- '+SHORT[right]]
  for k in ['macro_f1','good_auprc']:
   r=next(x for x in paired if (x['left'],x['right'],x['metric'])==(left,right,k))
   cells.extend([f"${r['mean_pp']:+.2f} \\pm {r['std_pp']:.2f}$",f"{r['positive_seeds']}/5"])
  lines.append(' & '.join(cells)+r' \\')
 tex_table('paired_results.tex','@{}lrrrr@{}',r'Comparison & $\Delta$ Macro-$F_1$ & Positive & $\Delta$ AP & Positive',lines)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--workspace',type=Path);args=a.parse_args()
 if args.workspace:bundle(args.workspace.resolve())
 analyze()
