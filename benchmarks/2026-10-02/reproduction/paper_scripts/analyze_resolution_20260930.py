"""Append-only perturbation, cross-protocol and eight-seed analyses."""
from pathlib import Path
import argparse,csv,json,hashlib,datetime,itertools,os
import numpy as np
from scipy.stats import t as student_t
P=Path(__file__).resolve().parents[1];W=P.parent;R=Path(os.environ.get('RFQC_RESOLUTION_ROOT',W/'rfqc_resolution_20260930'));O=P/'SourceData/resolution_20260930';BASE_CACHE=Path(os.environ.get('RFQC_BASE_CACHE',W/'rf_quality_control/cache_ag3_v1'))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
 with p.open('w',newline='') as f:
  q=csv.DictWriter(f,fieldnames=list(rows[0]));q.writeheader();q.writerows(rows)
def rows(p):return list(csv.DictReader(p.open()))
def macro(c):
 c=np.asarray(c,dtype=float);tn,fp,fn,tp=np.moveaxis(c,-1,0)
 return .5*(np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=2*tn+fp+fn>0)+np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=2*tp+fp+fn>0))
def counts(y,p,s):return np.stack([np.bincount(2*y[s==a].astype(int)+p[s==a].astype(int),minlength=4) for a in np.unique(s)])
def draws(s):
 n=len(np.unique(s));a=np.random.default_rng(20260930).integers(0,n,(10000,n));return np.stack([np.bincount(b,minlength=n) for b in a])
def start(name):
 receipt=json.loads((O/'registration_receipt.json').read_text());assert sha(O/'registration.json')==receipt['registration_sha256']
 marker=O/(name+'_started.json')
 if not marker.exists():write(marker,{'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registration_sha256':receipt['registration_sha256']})
 return json.loads((O/'registration.json').read_text())
def effect(clean,perturb,boot):
 drops=np.array([100*(macro(c.sum(0))-macro(p.sum(0))) for c,p in zip(clean,perturb)])
 distribution=np.mean([100*(macro(boot@c)-macro(boot@p)) for c,p in zip(clean,perturb)],axis=0)
 lo,hi=np.percentile(distribution,[2.5,97.5]);mean=float(drops.mean())
 return {'mean_drop_pp':mean,'lower_pp':float(lo),'upper_pp':float(hi),'all_realizations_drop':bool(np.all(drops>0)),'operationally_detected':bool(lo>0 and np.all(drops>0)),'min_realization_drop_pp':float(drops.min()),'max_realization_drop_pp':float(drops.max())},distribution,drops

def original():
 z=np.load(P/'SourceData/benchmark_v1/predictions.npz',allow_pickle=False);lookup=dict(zip(z['run_names'],z['probabilities']));metrics=rows(P/'SourceData/benchmark_v1/run_metrics.csv');threshold={(r['key'],int(r['seed'])):float(r['threshold']) for r in metrics};return z,lookup,metrics,threshold

def label_flips():
 reg=start('label_perturbation');z,lookup,metrics,threshold=original();y=z['labels'].astype(int);s=z['stations'];boot=draws(s);output=[];details=[];masks=[];distributions=[]
 uniforms=[np.random.default_rng(seed).random(len(y)) for seed in reg['label_flip_seeds']]
 for eps in reg['label_flip_proportions']:
  changed=[u<eps for u in uniforms]
  for k,rs in enumerate(reg['label_flip_seeds']):masks.append(changed[k]);details.append({'epsilon':eps,'seed':rs,'flipped_records':int(changed[k].sum())})
  for variant,key in [('ag3','ours_ag3'),('multiband','ours_multiband')]:
   clean=[];corrupted=[]
   for seed in reg['original_seeds']:
    pred=lookup[f'{key}/{seed}']>=threshold[key,seed];c=counts(y,pred,s)
    for mask in changed:clean.append(c);corrupted.append(counts(np.bitwise_xor(y,mask.astype(int)),pred,s))
   report,distribution,drop=effect(clean,corrupted,boot);output.append({'variant':variant,'epsilon':eps,**report});distributions.append(distribution)
 csvwrite(O/'label_perturbation.csv',output);csvwrite(O/'label_flip_counts.csv',details);np.savez_compressed(O/'label_flip_masks.npz',masks=np.stack(masks),bootstrap_weights=boot,bootstrap_drops_pp=np.stack(distributions))
 write(O/'label_perturbation_verification.json',{'reported_cells':len(output),'replicates_per_cell':60,'test_records':len(y),'test_stations':len(np.unique(s)),'label_flip_counts_retained':True,'registered_before_execution':True,'new_fits':0})
 print(json.dumps(output,indent=2))

def noise():
 reg=start('input_perturbation_analysis');d=R/'positive_controls';comp=json.loads((d/'completion.json').read_text());assert comp['evaluations']==78;assert all(sha(d/n)==h for n,h in comp['sha256'].items())
 allreports=[json.loads(p.read_text()) for p in d.glob('*_noise*.json')];assert len(allreports)==78
 by={(r['variant'],r['seed'],r['snr_db'],r['noise_seed']):r for r in allreports};output=[];distributions=[];archive_drift=[]
 z,lookup,met,threshold=original()
 for variant,key in [('ag3','ours_ag3'),('multiband','ours_multiband')]:
  for seed in reg['original_seeds']:
   cr=by[variant,seed,None,0];a=np.load(d/cr['file'],allow_pickle=False)
   assert np.array_equal(a['sample_ids'],z['sample_ids']) and np.array_equal(a['labels'],z['labels'])
   archive_drift.append({'variant':variant,'seed':seed,'max_abs_score_difference':float(np.max(np.abs(a['probabilities']-lookup[f'{key}/{seed}']))),'decisions_changed':int(np.sum((a['probabilities']>=cr['threshold'])!=(lookup[f'{key}/{seed}']>=threshold[key,seed])))})
  for snr in reg['noise_snr_db']:
   clean=[];corrupted=[]
   for seed in reg['original_seeds']:
    cr=by[variant,seed,None,0];a=np.load(d/cr['file'],allow_pickle=False);boot=draws(a['stations']);c=counts(a['labels'],a['probabilities']>=cr['threshold'],a['stations'])
    for rs in reg['noise_seeds']:
     r=by[variant,seed,snr,rs];b=np.load(d/r['file'],allow_pickle=False);assert np.array_equal(a['sample_ids'],b['sample_ids']);clean.append(c);corrupted.append(counts(b['labels'],b['probabilities']>=r['threshold'],b['stations']))
   report,distribution,drop=effect(clean,corrupted,boot);output.append({'variant':variant,'snr_db':snr,**report});distributions.append(distribution)
 csvwrite(O/'input_perturbation.csv',output);write(O/'clean_cuda_replay.json',archive_drift);np.savez_compressed(O/'input_perturbation_draws.npz',bootstrap_drops_pp=np.stack(distributions),bootstrap_weights=boot)
 allrows=[dict(family='input',**r) for r in output]+[dict(family='labels',**r) for r in rows(O/'label_perturbation.csv')]
 detected=[]
 for family in ['input','labels']:
  for variant in ['ag3','multiband']:
   options=[r for r in allrows if r['family']==family and r['variant']==variant and str(r['operationally_detected']).lower()=='true'];best=min(options,key=lambda r:float(r['mean_drop_pp'])) if options else None;detected.append({'family':family,'variant':variant,'smallest_observed_detected_drop':best})
 keys=['li2021_cnn','gan2021_cnn','gong_cnn','gong_cnn_bilstm','gan_cnn','deeprfqc','hegaz_capsule','chen2026_image'];means=sorted([(k,100*np.mean([float(r['macro_f1']) for r in met if r['key']==k])) for k in keys],key=lambda x:x[1],reverse=True)
 assert all(np.isfinite(a[1]) for a in means)
 write(O/'positive_control_summary.json',{'detected_grid_effects':detected,'literature_means_percent':means,'top_five_range_pp':means[0][1]-means[4][1],'all_eight_range_pp':means[0][1]-means[-1][1],'fit_count':0})
 print('Positive-control summary complete')

def verified_run(path):
 c=json.loads((path/'completion.json').read_text());assert all(sha(path/n)==h for n,h in c['sha256'].items());config=json.loads((path/'config.json').read_text());assert config['epochs']==50 and config['patience']==10 and config['batch_size']==32 and config['lr']==.0003 and config['balance']=='sampler' and config['torch_version']=='2.8.0+cu128';r=rows(path/'test_predictions.csv');y=np.array([int(x['label']) for x in r]);p=np.array([float(x['p_good']) for x in r]);s=np.array([x['station'] for x in r]);ids=np.array([x['sample_id'] for x in r]);m=json.loads((path/'metrics.json').read_text());threshold=m['test']['threshold'];calc=float(macro(counts(y,p>=threshold,s).sum(0)));np.testing.assert_allclose(calc,m['test']['macro_f1'],rtol=0,atol=1e-12);return y,p,s,ids,threshold,config,calc

def seeds():
 reg=start('eight_seed_analysis');z,lookup,metrics,threshold=original();names=['reference_ag3','reference_multifilter','gong_cnn','gong_cnn_bilstm'];oldkeys=['ours_ag3','ours_multiband','gong_cnn','gong_cnn_bilstm'];values={};output=[]
 for name,key in zip(names,oldkeys):
  for seed in reg['original_seeds']:
   m=next(r for r in metrics if r['key']==key and int(r['seed'])==seed);value=100*float(m['macro_f1']);values[name,seed]=value;output.append({'method':name,'seed':seed,'macro_f1_percent':value,'source':'frozen_original'})
  for seed in reg['new_seeds']:
   y,p,s,ids,th,c,value=verified_run(R/f'runs/seed_extension/{name}_seed{seed}');assert np.array_equal(ids,z['sample_ids']) and np.array_equal(y,z['labels']);values[name,seed]=100*value;output.append({'method':name,'seed':seed,'macro_f1_percent':100*value,'source':'appended'})
 diffs=[];summaries=[];allseeds=reg['original_seeds']+reg['new_seeds']
 # Named primary direction is multi-filter minus AG3; all unordered pairs are reported.
 for b,a in itertools.combinations(names,2):
  d=np.array([values[a,s]-values[b,s] for s in allseeds]);sd=float(d.std(ddof=1));half=float(student_t.ppf(.975,7)*sd/np.sqrt(8));mean=float(d.mean())
  for s,v in zip(allseeds,d):diffs.append({'first':a,'second':b,'seed':s,'difference_pp':float(v)})
  summaries.append({'first':a,'second':b,'n':8,'mean_pp':mean,'sample_sd_pp':sd,'half_width_pp':half,'lower_pp':mean-half,'upper_pp':mean+half,'contains_zero':bool(mean-half<=0<=mean+half)})
 csvwrite(O/'eight_seed_metrics.csv',output);csvwrite(O/'eight_seed_differences.csv',diffs);csvwrite(O/'eight_seed_summary.csv',summaries);write(O/'eight_seed_verification.json',{'new_fits':20,'reused_fits':12,'configurations':4,'seeds':allseeds,'original_test_ids_and_labels_match':True,'all_new_completion_hashes_verified':True});print(json.dumps(summaries,indent=2))

def cross():
 reg=start('cross_protocol_analysis');z=np.load(R/'matched_index.npz',allow_pickle=False);meta=json.loads((BASE_CACHE/'metadata.json').read_text());mask=np.array([meta['station_split'][str(s)]=='test' for s in z['stations']]);assert mask.sum()==18305;summary=[];deltas=[]
 from sklearn.metrics import average_precision_score
 for direction in ['strict_to_ag3','ag3_to_strict']:
  for seed in reg['original_seeds']:
   y,p,s,ids,th,c,val=verified_run(R/f'runs/cross_protocol/{direction}_seed{seed}');assert np.array_equal(ids,z['sample_ids'][mask]);source=z['strict_labels' if direction=='strict_to_ag3' else 'ag3_labels'][mask];target=z['ag3_labels' if direction=='strict_to_ag3' else 'strict_labels'][mask];assert np.array_equal(y,source);pred=p>=th;a=counts(source,pred,s);b=counts(target,pred,s);boot=draws(s);d=100*(macro(boot@b)-macro(boot@a));lo,med,hi=np.percentile(d,[2.5,50,97.5]);deltas.append(d)
   summary.append({'direction':direction,'seed':seed,'n':len(y),'stations':len(np.unique(s)),'source_macro_f1_percent':100*val,'target_macro_f1_percent':100*float(macro(b.sum(0))),'source_ap_percent':100*float(average_precision_score(source,p)),'target_ap_percent':100*float(average_precision_score(target,p)),'target_minus_source_lower_pp':float(lo),'target_minus_source_median_pp':float(med),'target_minus_source_upper_pp':float(hi),'threshold':th})
 csvwrite(O/'cross_protocol_metrics.csv',summary);np.savez_compressed(O/'cross_protocol_draws.npz',differences_pp=np.stack(deltas));write(O/'cross_protocol_verification.json',{'new_fits':6,'records':116722,'test_records':18305,'test_stations':22,'both_directions_complete':True,'all_completion_hashes_verified':True,'same_ids_source_labels_and_split_verified':True});print(json.dumps(summary,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['labels','noise','seeds','cross']);a=p.parse_args();globals()[{'labels':'label_flips','noise':'noise','seeds':'seeds','cross':'cross'}[a.action]]()
