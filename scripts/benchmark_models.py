"""Measure selected public model bundles in fresh, sequential worker processes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

from rfqc_bench import __version__, list_models
import rfqc_bench.benchmarking as timing_code
from rfqc_bench.io import atomic_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Your NPZ test pool; no dataset is downloaded')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=['all'])
    p.add_argument('--device', default='cpu', help='Neural device; statistical models always use CPU')
    p.add_argument('--seed', type=int, default=20260929)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--sample-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--latency-calls', type=int, default=100)
    p.add_argument('--throughput-calls', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--model-root', type=Path, help='Offline folders named METHOD-seedSEED; otherwise explicit model download')
    p.add_argument('--resume', action='store_true', help='Skip hash-verified completed timing rounds')
    a = p.parse_args()
    if not a.input.is_file():p.error('--input must be an NPZ file')
    if a.rounds < 1:p.error('--rounds must be positive')
    registry = {x['name']:x for x in list_models()}
    names = list(registry) if a.models == ['all'] else a.models
    if len(set(names)) != len(names) or set(names) - set(registry):p.error('Unknown or duplicate model name')
    local_bundles={}
    if a.model_root:
        for name in names:
            folder=a.model_root/f'{name}-seed{a.seed}'
            metadata=folder/'bundle.json'
            if not metadata.is_file():p.error(f'Missing bundle for {name}')
            local_bundles[name]={'bundle_sha256':digest(metadata)}
            weight=folder/'weights.safetensors'
            if weight.exists():local_bundles[name]['weights_sha256']=digest(weight)
    protocol = dict(software_version=__version__, runner_sha256=digest(Path(__file__)),
                    timing_implementation_sha256=digest(Path(timing_code.__file__)),
                    model_catalog_sha256=digest(Path(timing_code.__file__).with_name('model_zoo.json')),
                    local_bundle_hashes=local_bundles,
                    input_sha256=digest(a.input), models=names, device=a.device, seed=a.seed,
                    rounds=a.rounds, sample_size=a.sample_size, batch_size=a.batch_size,
                    warmup=a.warmup, latency_calls=a.latency_calls, throughput_calls=a.throughput_calls,
                    threads=a.threads, complete_views=True,
                    model_root=str(a.model_root.resolve()) if a.model_root else None,
                    scope='Fresh worker per method/round; complete-view common ID selection except full-pool FCM; no training')
    config = a.output/'protocol.json'
    a.output.mkdir(parents=True, exist_ok=True)
    if config.exists():
        if not a.resume:p.error('Output already exists; use --resume or a new directory')
        if json.loads(config.read_text()) != protocol:p.error('Resume protocol or input differs')
    else:
        if any(a.output.iterdir()):p.error('New output directory must be empty')
        atomic_json(config, protocol)
    manifest_path = a.output/'completed_rounds.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    env = os.environ.copy()
    env.update({k:str(a.threads) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    for repeat in range(a.rounds):
        order = sorted(names, key=lambda n:hashlib.sha256(f'{repeat}/{n}'.encode()).hexdigest())
        for name in order:
            target = a.output/'results'/f'{name}_round{repeat}.json'
            relative = str(target.relative_to(a.output))
            if target.exists():
                if manifest.get(relative) != digest(target):p.error(f'Unverified or changed result: {relative}')
                continue
            command = [sys.executable, '-m', 'rfqc_bench', 'benchmark', '--input', str(a.input.resolve()),
                       '--output', str(target.resolve()), '--device', a.device if registry[name]['kind']=='neural' else 'cpu',
                       '--sample-size', str(a.sample_size), '--batch-size', str(a.batch_size), '--complete-views',
                       '--warmup', str(a.warmup), '--latency-calls', str(a.latency_calls),
                       '--throughput-calls', str(a.throughput_calls), '--threads', str(a.threads), '--round-index', str(repeat)]
            if a.model_root:
                command += ['--model-dir', str((a.model_root/f'{name}-seed{a.seed}').resolve())]
            else:
                command += ['--model', name, '--seed', str(a.seed)]
            atomic_json(a.output/'status.json', dict(state='running', method=name, round=repeat, completed=len(manifest)))
            result = subprocess.run(command, env=env, check=False)
            if result.returncode:
                atomic_json(a.output/'status.json', dict(state='stopped', method=name, round=repeat, returncode=result.returncode))
                raise SystemExit(result.returncode)
            report=json.loads(target.read_text())
            if report['method']!=name or report['seed']!=a.seed or report['round']!=repeat:
                p.error('Bundle identity or round does not match requested protocol')
            manifest[relative] = digest(target)
            atomic_json(manifest_path, manifest)
    atomic_json(a.output/'status.json', dict(state='completed', completed=len(manifest), expected=len(names)*a.rounds))
    print(json.dumps(dict(state='completed', measurement_rounds=len(manifest), output=str(a.output))))


if __name__ == '__main__':
    main()
