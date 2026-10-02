"""Finite manually started two-fit queue; waits for existing GPU capacity."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpointing import atomic_json,completed,digest


def busy(uuids):
    out={u:0 for u in uuids}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if not any(a.endswith((b'/train.py',b'/evaluate_cross.py')) for a in args):continue
            if (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
            u=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
            if u in out:out[u]+=1
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return out


def unlocked(run):
    folder=run.parent/'.job_locks';folder.mkdir(exist_ok=True)
    with (folder/(run.name+'.lock')).open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());uuids=protocol['runtime']['gpu_uuids']
    lock=(ROOT/'audit/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in json.loads((ROOT/'audit/training_sources.json').read_text()).items():assert digest(ROOT/name)==h
    out=ROOT/'runs';out.mkdir(exist_ok=True)
    active={};done=set();failed={};stopping=False
    def stop(sig,frame):
        nonlocal stopping
        stopping=True
        for p,u in active.values():p.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for band in ['ag1','ag5']:
        if completed(out/f'{band}_waveform_random_seed20260929'):done.add(band)
    while True:
        for band,(p,u) in list(active.items()):
            if p.poll() is None:continue
            del active[band]
            if p.returncode==0 and completed(out/f'{band}_waveform_random_seed20260929'):done.add(band)
            elif not stopping:failed[band]=p.returncode
        used=busy(uuids)
        if not stopping and not failed:
            for band in ['ag1','ag5']:
                if band in active or band in done:continue
                candidate=out/f'{band}_waveform_random_seed20260929'
                if completed(candidate):done.add(band);continue
                if not unlocked(candidate):continue
                free=[u for u in uuids if used[u]<2 and u not in [v[1] for v in active.values()]]
                if not free:continue
                u=min(free,key=lambda x:used[x]);run=out/f'{band}_waveform_random_seed20260929'
                cmd=[sys.executable,'-u',str(ROOT/'src/train.py'),'--cache',str(ROOT/'cache'/band),'--output',str(run),
                    '--arm','waveform_random','--seed','20260929','--device','cuda','--input-samples','2048',
                    '--epochs','50','--patience','10','--batch-size','32','--threads','4','--balance','sampler',
                    '--lr','0.0003','--resume','--checkpoint-seconds','120']
                with (out/(band+'.log')).open('a') as f:
                    p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':u,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},start_new_session=True)
                active[band]=(p,u);used[u]+=1;print('Started',band,p.pid,u,flush=True)
        state='completed' if len(done)==2 else 'failed' if failed else 'paused' if stopping else 'running' if active else 'waiting_for_capacity'
        atomic_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=time.time(),planned_runs=2,
            completed=sorted(done),failed=failed,active=[dict(band=b,pid=p.pid,gpu=u) for b,(p,u) in active.items()]))
        if not active and (stopping or failed or len(done)==2):break
        time.sleep(10)
    if failed:raise RuntimeError(failed)


if __name__=='__main__':main()
