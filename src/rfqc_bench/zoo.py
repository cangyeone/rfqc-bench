"""Opt-in, checksummed model downloads. No dataset is downloaded."""
from importlib.resources import files
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile
from .io import digest
from .registry import get_spec


def available_weights():
    return json.loads(files('rfqc_bench').joinpath('model_zoo.json').read_text())['models']


def download_model(name, seed=None, cache_dir=None):
    get_spec(name)
    options=[m for m in available_weights() if m['name']==name]
    if seed is None and options: seed=min(m['seed'] for m in options)
    entry=next((m for m in options if m['seed']==seed),None)
    if entry is None: raise ValueError(f'No released weights for {name}, seed {seed}')
    base=Path(cache_dir or os.environ.get('RFQC_CACHE',Path.home()/'.cache/rfqc-bench')).expanduser()
    base.mkdir(parents=True,exist_ok=True)
    target=base/f'{name}-seed{seed}-{entry["sha256"][:12]}'
    if target.is_dir():
        for n,h in entry['files'].items():
            if not (target/n).is_file() or digest(target/n)!=h: raise ValueError(f'Corrupt cached model: remove {target} and download again')
        return target
    staging=Path(tempfile.mkdtemp(prefix='.download-',dir=base))
    try:
        archive=staging/'model.zip'
        with urlopen(entry['url'],timeout=60) as response,archive.open('wb') as stream:
            copied=0
            while block:=response.read(1024*1024):
                copied+=len(block)
                if copied>entry['size_bytes']: raise ValueError('Model download exceeds registered size')
                stream.write(block)
        if digest(archive)!=entry['sha256']: raise ValueError('Model archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            if sorted(z.namelist())!=sorted(entry['files']): raise ValueError('Unexpected model archive entries')
            for name in entry['files']:
                if name not in {'bundle.json','weights.safetensors'}: raise ValueError('Invalid model file')
                with z.open(name) as src,(staging/name).open('wb') as dst: shutil.copyfileobj(src,dst)
                if digest(staging/name)!=entry['files'][name]: raise ValueError('Model file checksum mismatch')
        archive.unlink()
        try: staging.rename(target)
        except OSError:
            if not target.is_dir(): raise
            for name,h in entry['files'].items():
                if digest(target/name)!=h: raise ValueError('Concurrent model cache differs')
        return target
    finally:
        if staging.exists(): shutil.rmtree(staging)
