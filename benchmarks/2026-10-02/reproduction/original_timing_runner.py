"""Fixed-protocol inference measurements; never fits models or selects thresholds."""
from pathlib import Path
import argparse,csv,gc,hashlib,json,os,platform,resource,subprocess,sys,time
import numpy as np
import torch
from rfqc_bench import RFData,RFQCPredictor
from rfqc_bench.models import create_model
from rfqc_bench.registry import get_spec
from threadpoolctl import threadpool_limits,threadpool_info
R=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def gpu():
 return subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader'],text=True).strip()
def prepared_data(cache,spec,n,all_context):
 arrays={k:np.load(cache/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['waveforms','features','gaussians','lengths','stations','sample_ids']}
 test=np.load(cache/'test_indices.npy',allow_pickle=False)
 if all_context:idx=test
 else:
  eligible=test[arrays['lengths'][test]==7]
  idx=np.array(sorted(eligible,key=lambda i:hashlib.sha256(str(arrays['sample_ids'][i]).encode()).hexdigest())[:n])
 want_wave=spec.mode!='features';want_features=spec.mode!='waveform'
 if spec.gaussian is not None:
  rows=np.arange(len(idx));positions=np.argmax(arrays['gaussians'][idx]==spec.gaussian,axis=1)
  assert np.all(arrays['gaussians'][idx,positions]==spec.gaussian)
  wave=np.array(arrays['waveforms'][idx,positions,None,:]) if want_wave else None
  features=np.array(arrays['features'][idx,positions,None,:]) if want_features else None
  g=np.full((len(idx),1),spec.gaussian);lengths=np.ones(len(idx),dtype=np.int64)
 else:
  wave=np.array(arrays['waveforms'][idx]) if want_wave else None
  features=np.array(arrays['features'][idx]) if want_features else None
  g=np.array(arrays['gaussians'][idx]);lengths=np.array(arrays['lengths'][idx])
 return RFData(waveforms=wave,features=features,gaussians=g,lengths=lengths,stations=np.array(arrays['stations'][idx]),sample_ids=np.array(arrays['sample_ids'][idx]))

def worker(name,repeat):
 protocol=json.loads((R/'protocol.json').read_text());entry=next(x for x in protocol['methods'] if x['name']==name);spec=get_spec(name)
 torch.set_num_threads(protocol['cpu_threads']);torch.set_num_interop_threads(protocol['interop_threads'])
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed'])
 cuda=spec.kind=='neural';device='cuda:0' if cuda else 'cpu'
 if cuda:assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 5090'
 bundle_path=R/'bundles'/(name+'.json');assert sha(bundle_path)==entry['bundle_sha256'];bundle=json.loads(bundle_path.read_text())
 network=None;params=None;source_hash=None
 if cuda:
  source=Path(entry['source_directory']);weight=source/'best_model.pt';source_hash=sha(weight)
  assert source_hash==entry['source_model_sha256'];assert json.loads((source/'completion.json').read_text())['sha256']['best_model.pt']==source_hash
  network=create_model(name);network.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
  params=sum(x.numel() for x in network.parameters())
 elif spec.kind=='logistic':params=int(np.asarray(bundle['parameters']['coef']).size+np.asarray(bundle['parameters']['intercept']).size)
 else:params=int(np.asarray(bundle['parameters']['centers']).size)
 pred=RFQCPredictor(bundle,network,device=device)
 data=prepared_data(Path(protocol['cache']),spec,protocol['test_sample_size'],spec.kind=='fcm')
 ids_sha=hashlib.sha256('\n'.join(data.sample_ids.tolist()).encode()).hexdigest()
 sync=lambda:torch.cuda.synchronize() if cuda else None
 measure=lambda fn:timed(fn,sync)
 record=dict(method=name,round=repeat,seed=protocol['seed'],kind=spec.kind,device=device,parameters=params,records=len(data),views=sorted(set(data.lengths.tolist())),sample_ids_sha256=ids_sha,bundle_sha256=sha(bundle_path),source_model_sha256=source_hash,protocol_sha256=sha(R/'protocol.json'),script_sha256=sha(__file__),hardware_before=gpu(),torch=str(torch.__version__),numpy=np.__version__,python=sys.version,platform=platform.platform(),processor=platform.processor(),cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),''),threadpools=threadpool_info())
 if spec.kind=='fcm':
  assert len(data)==24533 and len(set(data.stations))==26
  pred.predict(data);seconds,p=measure(lambda:pred.predict(data))
  record.update(pool_seconds=seconds,pool_records_per_second=len(data)/seconds,mean_ms_per_record=1000*seconds/len(data),warmup_calls=1,predicted_good=int(p.prediction.sum()),latency_scope='Complete station-context pool; no independent-record latency')
 else:
  one=[data.subset(slice(i,i+1)) for i in range(len(data))]
  batches=[data.subset(slice(i,i+32)) for i in range(0,len(data),32)]
  for i in range(protocol['warmup_calls']):pred.predict(one[i],batch_size=1)
  single=[]
  for i in range(protocol['latency_calls']):
   sec,p=measure(lambda i=i:pred.predict(one[(repeat*100+i)%len(one)],batch_size=1));single.append(sec)
  for i in range(protocol['warmup_calls']):pred.predict(batches[i%len(batches)],batch_size=32)
  if cuda:torch.cuda.reset_peak_memory_stats()
  bulk=[]
  for i in range(protocol['throughput_calls']):
   sec,p=measure(lambda i=i:pred.predict(batches[(repeat+i)%len(batches)],batch_size=32));bulk.append(sec)
  peak=torch.cuda.max_memory_allocated() if cuda else None
  record.update(latency_seconds=single,latency_p50_ms=1000*float(np.median(single)),latency_p95_ms=1000*float(np.percentile(single,95)),batch32_seconds=bulk,batch32_records_per_second=32*len(bulk)/sum(bulk),pipeline_peak_gpu_allocated_bytes=peak)
  if cuda:
   args=[pred.preprocessor.transform(x,device) for x in batches]
   with torch.inference_mode():
    for i in range(protocol['warmup_calls']):pred.network(*args[i%len(args)]).sigmoid()
    net=[]
    for i in range(protocol['throughput_calls']):
     sec,_=measure(lambda i=i:pred.network(*args[(repeat+i)%len(args)]).sigmoid());net.append(sec)
   record.update(forward_batch32_seconds=net,forward_batch32_records_per_second=32*len(net)/sum(net))
  if repeat==0:
   replay=pred.predict(data,batch_size=32);record['finite_replay']=bool(np.isfinite(replay.p_good).all())
   if entry['source_directory']:
    with (Path(entry['source_directory'])/'test_predictions.csv').open() as f:expected={x['sample_id']:x for x in csv.DictReader(f)}
    original=np.array([float(expected[i]['p_good']) for i in data.sample_ids]);decisions=np.array([int(expected[i]['prediction']) for i in data.sample_ids])
    record.update(archive_max_abs_score_difference=float(np.max(np.abs(original-replay.p_good))),archive_decisions_changed=int(np.sum(decisions!=replay.prediction)))
  save(R/'sample_ids'/f'{name}.json',data.sample_ids.tolist())
 record.update(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,hardware_after=gpu(),finished_at=time.time())
 save(R/'results'/f'{name}_round{repeat}.json',record)
 print(json.dumps({k:record[k] for k in ['method','round','kind','records']},ensure_ascii=False),flush=True)

def timed(fn,sync):
 sync();start=time.perf_counter_ns();result=fn();sync();return (time.perf_counter_ns()-start)/1e9,result

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--round',type=int,default=0);a=p.parse_args()
 if a.worker:
  with threadpool_limits(limits=4):worker(a.worker,a.round)
  return
 protocol=json.loads((R/'protocol.json').read_text());(R/'logs').mkdir(exist_ok=True)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==protocol['gpu_uuid']
 assert sha(R/'rfqc_bench-0.1.0-py3-none-any.whl')==protocol['wheel_sha256']
 names=[x['name'] for x in protocol['methods']];complete=[]
 for repeat in range(protocol['repeated_rounds']):
  order=sorted(names,key=lambda x:hashlib.sha256(f'{repeat}/{x}'.encode()).hexdigest())
  for name in order:
   path=R/'results'/f'{name}_round{repeat}.json'
   if path.exists():
    old=json.loads(path.read_text());assert old['protocol_sha256']==sha(R/'protocol.json') and old['script_sha256']==sha(__file__)
   else:
    save(R/'status.json',dict(state='running',current=name,round=repeat,completed=len(complete),expected=len(names)*3))
    with (R/'logs'/f'{name}_round{repeat}.log').open('a') as f:
     run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name,'--round',str(repeat)],stdout=f,stderr=f)
    if run.returncode:
     save(R/'status.json',dict(state='failed',current=name,round=repeat,returncode=run.returncode,completed=len(complete)));raise SystemExit(run.returncode)
   complete.append(str(path.relative_to(R)));print('DONE',name,repeat,len(complete),flush=True)
 save(R/'completion.json',dict(completed=len(complete),sha256={p:sha(R/p) for p in complete},protocol_sha256=sha(R/'protocol.json'),script_sha256=sha(__file__)))
 save(R/'status.json',dict(state='completed',completed=len(complete),expected=len(names)*3))

if __name__=='__main__':main()
