"""Maintainer-only: export verified benchmark weights, never samples or labels.

Requires the original local experiment workspace. Outputs are outside the git
tree. Run before building distributions so the wheel embeds checksums and URLs.
"""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rfqc_bench.models import create_model
from rfqc_bench.registry import get_spec
from rfqc_bench.training import save_bundle
from rfqc_bench.io import digest, atomic_json


def sources(work):
    results=work/'rfqc_benchmark_20260929/results'
    for seed in [20260928,20260929,20260930]:
        for name in ['gong_cnn','gong_cnn_bilstm','gan_cnn','deeprfqc','hegaz_capsule']:
            yield name,seed,results/f'literature/{name}_seed{seed}',None
        for name in ['li2021_cnn','gan2021_cnn','chen2026_image']:
            yield name,seed,results/f'extension/{name}_seed{seed}',None
        for mode,arm in [('reference','waveform_random'),('descriptors','features_only'),('combined','combined_random')]:
            for variant,folder in [('ag3','ag3'),('multifilter','multiband')]:
                yield f'{mode}_{variant}',seed,results/f'bands/{folder}/{arm}_seed{seed}',None
        yield 'xiong2025_fcm',seed,results/f'fcm/xiong2025_fcm_seed{seed}',None
        paper=work/'overleaf-rf-multiband'
        for variant,folder in [('ag3','ag3'),('multifilter','multiband')]:
            yield f'logreg_{variant}',seed,paper/f'SourceData/revision_20260930/logistic/{folder}_seed{seed}',paper/f'SourceData/benchmark_v1/features_only_{folder}/20260928/preprocessing.json'
    for g in [1,5]:
        yield f'reference_ag{g}',20260929,work/f'rfqc_physical_validation_20260929/runs/ag{g}_waveform_random_seed20260929',None


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);entries=[];audit=[]
    torch.set_num_threads(1)
    for name,seed,source,scaler_source in sources(a.workspace):
        spec=get_spec(name)
        required=['model.json'] if spec.kind!='neural' else ['best_model.pt']
        required+=['metrics.json']
        if scaler_source is None:required+=['preprocessing.json']
        completion=json.loads((source/'completion.json').read_text())['sha256']
        for file in required:
            if digest(source/file)!=completion[file]:raise ValueError(f'Completion mismatch: {source/file}')
        prep=json.loads((scaler_source or source/'preprocessing.json').read_text())
        if scaler_source:
            assert digest(scaler_source)==json.loads((source/'config.json').read_text())['preprocessing_sha256']
        # Publish only fitted transforms, not source paths or sample statistics.
        keep=['feature_mean','feature_std','wave_scale','image_adapter','preprocessing_version','fit_split','minimum','scale']
        prep={k:v for k,v in prep.items() if k in keep}
        metric=json.loads((source/'metrics.json').read_text())
        threshold=metric['validation']['threshold']
        network=None;parameters=None
        if spec.kind=='neural':
            network=create_model(name)
            network.load_state_dict(torch.load(source/'best_model.pt',map_location='cpu',weights_only=True),strict=True)
        else:parameters=json.loads((source/'model.json').read_text())
        info=dict(training_seed=seed,phase_task_transfer=False,source='Registered RFQC benchmark adaptation',
                  selection='Fixed first seed by default; all available registered seeds are exposed, not selected by test score.')
        if spec.kind=='logistic':
            config=json.loads((source/'config.json').read_text())
            info.update(sklearn_version=config['sklearn_version'],iterations=config['iterations'],warnings=config['warnings'])
        bundle_dir=a.output/f'{name}-seed{seed}'
        save_bundle(bundle_dir,name,seed,threshold,prep,network,parameters,info)
        files={f.name:digest(f) for f in sorted(bundle_dir.iterdir()) if f.is_file()}
        assert set(files)<={'bundle.json','weights.safetensors'}
        archive=a.output/f'{name}-seed{seed}.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for file in files:
                zi=zipfile.ZipInfo(file,date_time=(2026,9,30,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(zi,(bundle_dir/file).read_bytes(),compresslevel=1)
        entries.append(dict(name=name,seed=seed,url=f'https://github.com/cangyeone/rfqc-bench/releases/download/v0.1.0/{archive.name}',
                            sha256=digest(archive),size_bytes=archive.stat().st_size,files=files))
        audit.append(dict(name=name,seed=seed,source_completion_verified=True,
                          source_weights_sha256=digest(source/required[0]),exported_files=files))
        print(name,seed,archive.stat().st_size,flush=True)
        del network
    assert len(entries)==53
    atomic_json(ROOT/'src/rfqc_bench/model_zoo.json',dict(schema=1,release='v0.1.0',models=entries))
    atomic_json(ROOT/'validation/model_export.json',dict(models=53,dataset_files_included=False,models_verified=audit))


if __name__=='__main__':main()
