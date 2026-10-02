"""Independently audit completed appended fits and create a portable prediction pack.

Raw weights remain outside the paper repository; all completion hashes are checked
when --raw-root is supplied. Existing primary tables/results are never changed.
"""
from pathlib import Path
import argparse,csv,json,hashlib,shutil,datetime
import numpy as np
from scipy.stats import t as student_t
from analyze_results import metrics,check,sha,table,dump
P=Path(__file__).resolve().parents[1];O=P/'SourceData/resolution_20260930'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--raw-root',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True);a=parser.parse_args();R=a.raw_root
 reg=json.loads((O/'registration.json').read_text());plan=json.loads((R/'execution_plan.json').read_text())
 assert sha(O/'registration.json')==sha(R/'protocol.json')==plan['registered_sha256']
 for name,h in reg['frozen_inputs'].items():assert sha(P/name)==h,name
 old=np.load(P/'SourceData/benchmark_v1/predictions.npz',allow_pickle=False);paired=np.load(R/'matched_index.npz',allow_pickle=False)
 meta=json.loads((a.cache/'metadata.json').read_text());mask=np.array([meta['station_split'][str(s)]=='test' for s in paired['stations']]);assert mask.sum()==18305
 expected={(j['kind'],j['name']) for j in plan['jobs']};actual={(p.parent.parent.name,p.parent.name) for p in (R/'runs').glob('*/*/completion.json')};assert expected==actual and len(expected)==26
 hashes=[];source_hashes={};runs=[];vectors={'seed_extension':[],'cross_protocol':[]};names={'seed_extension':[],'cross_protocol':[]};station_rows=[];cross_rows=[];preps={};pairs={}
 for j in plan['jobs']:
  root=R/'runs'/j['kind']/j['name'];receipt=json.loads((root/'completion.json').read_text());assert len(receipt['sha256'])==6
  for name,h in receipt['sha256'].items():assert sha(root/name)==h,(j['name'],name);hashes.append(dict(run=j['name'],file=name,sha256=h))
  c=json.loads((root/'config.json').read_text());m=json.loads((root/'metrics.json').read_text());prep=json.loads((root/'preprocessing.json').read_text());hist=json.loads((root/'history.json').read_text())
  for k,v in dict(epochs=50,patience=10,batch_size=32,lr=.0003,balance='sampler',torch_version='2.8.0+cu128',device='cuda',pretrained=False,smoke_samples=0).items():assert c[k]==v,(j['name'],k)
  assert c['seed']==j['seed'] and c['arm']==j['command'][j['command'].index('--arm')+1]
  assert c['cuda_visible_devices']==plan['gpu_uuids'][j['gpu']]
  source='literature' if j['name'].startswith('gong') else 'reference'
  for name,h in c['resume_contract']['source_sha256'].items():assert sha(R/'source_snapshot'/source/name)==h;source_hashes[source+'/'+name]=h
  assert prep['fit_split']=='train'
  group=j['name'].split('_seed')[0];preps.setdefault(group,sha(root/'preprocessing.json'));assert preps[group]==sha(root/'preprocessing.json')
  assert [r['epoch'] for r in hist]==list(range(1,m['epochs_completed']+1)) and len(hist)<=50
  best=max(hist,key=lambda r:r['validation']['good_auprc']);assert abs(m['validation']['good_auprc']-best['validation']['good_auprc'])<1e-12
  with (root/'test_predictions.csv').open() as f:rows=list(csv.DictReader(f))
  ids=np.array([r['sample_id'] for r in rows]);stations=np.array([r['station'] for r in rows]);y=np.array([int(r['label']) for r in rows]);prob=np.array([float(r['p_good']) for r in rows]);threshold=m['test']['threshold'];assert threshold==m['validation']['threshold'];assert np.min(np.abs(np.linspace(.01,.99,99)-threshold))<1e-12
  assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1));np.testing.assert_array_equal([int(r['prediction']) for r in rows],prob>=threshold)
  if j['kind']=='seed_extension':
   for key,array in [('sample_ids',ids),('labels',y),('stations',stations)]:np.testing.assert_array_equal(array,old[key])
   assert c['cache_metadata']['station_split']==meta['station_split'];assert c['actual_split_sizes']==dict(train=101852,val=21309,test=24533)
  else:
   np.testing.assert_array_equal(ids,paired['sample_ids'][mask]);np.testing.assert_array_equal(stations,paired['stations'][mask])
   direction=group;source_label=paired['strict_labels' if direction=='strict_to_ag3' else 'ag3_labels'][mask];target=paired['ag3_labels' if direction=='strict_to_ag3' else 'strict_labels'][mask];np.testing.assert_array_equal(y,source_label)
   assert c['actual_split_sizes']==dict(train=78887,val=19530,test=18305)
   other=metrics(target,prob,threshold);cross_rows.append(dict(direction=direction,seed=j['seed'],source_macro_f1_percent=100*metrics(y,prob,threshold)['macro_f1'],target_macro_f1_percent=100*other['macro_f1'],target_ap_percent=100*other['good_auprc']))
  recalculated=metrics(y,prob,threshold);check(recalculated,m['test'],j['name']);check(metrics(y,prob,.5),m['test_at_05'],j['name']+'/.5')
  saved_station=json.loads((root/'station_metrics.json').read_text());assert set(saved_station)==set(stations)
  for station in sorted(set(stations)):
   ix=stations==station;values=metrics(y[ix],prob[ix],threshold);check(values,saved_station[station],j['name']+'/'+station);station_rows.append(dict(run=j['name'],station=station,**{k:values[k] for k in ['n','macro_f1','accuracy','good_auprc']}))
  runs.append(dict(run=j['name'],group=group,kind=j['kind'],seed=j['seed'],threshold=threshold,**{k:recalculated[k] for k in ['n','accuracy','macro_f1','good_auprc','good_precision','good_recall','auroc']}))
  vectors[j['kind']].append(prob);names[j['kind']].append(j['name'])
  dest=O/'runs'/j['kind']/j['name'];dest.mkdir(parents=True,exist_ok=True)
  for name in ['config.json','preprocessing.json','metrics.json','station_metrics.json','completion.json','history.json']:shutil.copy2(root/name,dest/name)
 assert preps['strict_to_ag3']==preps['ag3_to_strict']
 for kind in vectors:
  arrays=dict(run_names=np.array(names[kind]),probabilities=np.stack(vectors[kind]))
  if kind=='seed_extension':arrays.update({k:old[k] for k in ['sample_ids','labels','stations']})
  else:arrays.update(sample_ids=paired['sample_ids'][mask],stations=paired['stations'][mask],strict_labels=paired['strict_labels'][mask],ag3_labels=paired['ag3_labels'][mask])
  np.savez_compressed(O/(kind+'_predictions.npz'),**arrays)
 table(O/'independent_metrics.csv',runs);table(O/'independent_station_metrics.csv',station_rows)
 saved_cross=list(csv.DictReader((O/'cross_protocol_metrics.csv').open()))
 for row in cross_rows:
  saved=next(r for r in saved_cross if r['direction']==row['direction'] and int(r['seed'])==row['seed'])
  for k in ['source_macro_f1_percent','target_macro_f1_percent','target_ap_percent']:assert abs(row[k]-float(saved[k]))<1e-10
 # Recompute reused seeds from original probability vectors, not summary scores.
 keys={'reference_ag3':'ours_ag3','reference_multifilter':'ours_multiband','gong_cnn':'gong_cnn','gong_cnn_bilstm':'gong_cnn_bilstm'}
 old_metrics=list(csv.DictReader((P/'SourceData/benchmark_v1/run_metrics.csv').open()));all_values={}
 for name,key in keys.items():
  for seed in reg['original_seeds']:
   original=next(r for r in old_metrics if r['key']==key and int(r['seed'])==seed);ix=old['run_names'].tolist().index(f'{key}/{seed}');all_values[name,seed]=100*metrics(old['labels'],old['probabilities'][ix],float(original['threshold']))['macro_f1']
  for row in runs:
   if row['group']==name:all_values[name,row['seed']]=100*row['macro_f1']
 for r in csv.DictReader((O/'eight_seed_metrics.csv').open()):assert abs(float(r['macro_f1_percent'])-all_values[r['method'],int(r['seed'])])<1e-10
 for r in csv.DictReader((O/'eight_seed_summary.csv').open()):
  diff=np.array([all_values[r['first'],s]-all_values[r['second'],s] for s in reg['original_seeds']+reg['new_seeds']]);half=student_t.ppf(.975,7)*diff.std(ddof=1)/np.sqrt(8)
  for k,v in [('mean_pp',diff.mean()),('sample_sd_pp',diff.std(ddof=1)),('lower_pp',diff.mean()-half),('upper_pp',diff.mean()+half)]:assert abs(float(r[k])-v)<1e-10
 for directory in ['source_snapshot']:
  shutil.copytree(R/directory,O/directory,dirs_exist_ok=True)
 for name in ['execution_plan.json','execution_started.json','matched_index_verification.json','status.json','analysis_status.json','user_pause_20260930.json','user_resume_20260930.json']:shutil.copy2(R/name,O/name)
 dump(O/'completed_verification_20261002.json',dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=26,completion_files_hashed=len(hashes),completion_manifest=hashes,source_sha256=source_hashes,independent_threshold_metrics=len(runs)*2,independent_station_metrics=len(station_rows),reused_original_scores_recomputed=12,all_six_eight_seed_contrasts_recomputed=True,cross_protocol_source_target_metrics_recomputed=True,same_ordered_ids_labels_stations_verified=True,training_only_preprocessing_verified=True,validation_inference_repeated=False,portable_predictions=True,raw_root=str(R)))
 print(json.dumps({'verified_fits':len(runs),'hashes':len(hashes),'station_metrics':len(station_rows),'portable_predictions':'written'}))

if __name__=='__main__':main()
