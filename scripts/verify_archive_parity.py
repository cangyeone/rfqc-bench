"""Maintainer-only numerical replay. Publish aggregate errors, never RF records."""
import argparse
import csv
import json
from pathlib import Path
import sys
import ast
import importlib.util
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rfqc_bench import RFData,RFQCPredictor
from rfqc_bench.io import atomic_json
from export_model_zoo import sources


def original_cpu_probabilities(work,name,source,data,seed):
    """Load the old adapter and its original Dataset, outside the new package."""
    from rfqc_bench.registry import get_spec
    spec=get_spec(name)
    if name.startswith(('reference','descriptors','combined')):
        root=work/'rf_quality_control';mode=spec.mode;size=2048
    elif name in ['li2021_cnn','gan2021_cnn','chen2026_image']:
        root=work/'rfqc_benchmark_20260929/src';mode=name;size=501
    else:
        root=work/'rf_quality_control/baselines_20260929/src';mode=name;size=501
    module_spec=importlib.util.spec_from_file_location('original_adapter',root/'models.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    model=module.MultiBandQC(mode,False,size).eval()
    model.load_state_dict(torch.load(source/'best_model.pt',map_location='cpu',weights_only=True),strict=True)
    original_source=(root/'train.py').read_text();tree=ast.parse(original_source)
    parts=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['signed_log','transform_waveform','RFDataset']]
    context=dict(np=np,torch=torch,Dataset=torch.utils.data.Dataset)
    exec(compile(ast.Module(body=parts,type_ignores=[]),str(root/'train.py'),'exec'),context)
    selected=data.select(spec.gaussian)
    dd={n:np.array(getattr(selected,n)) for n in ['waveforms','features','gaussians','lengths']}
    dd['gaussians']=dd['gaussians'].astype(np.float32)
    dd.update(test_indices=np.arange(len(selected)),labels=np.zeros(len(selected)))
    prep=json.loads((source/'preprocessing.json').read_text())
    dataset=context['RFDataset'](dd,'test',prep)
    result=[]
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(dataset,batch_size=32):result.append(model(*batch[:4]).sigmoid().numpy())
    return np.concatenate(result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--bundles',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    cache=a.workspace/'rf_quality_control/cache'
    arrays={n:np.load(cache/f'{n}.npy',allow_pickle=False,mmap_mode='r') for n in
            ['waveforms','features','gaussians','lengths','stations','sample_ids']}
    indices={str(s):i for i,s in enumerate(arrays['sample_ids'])};checks=[]
    for name,seed,source,_ in sources(a.workspace):
        with (source/'test_predictions.csv').open() as f:rows=list(csv.DictReader(f))
        if name!='xiong2025_fcm':rows=[rows[i] for i in np.linspace(0,len(rows)-1,128,dtype=int)]
        idx=np.array([indices[r['sample_id']] for r in rows])
        data=RFData(**{n:np.array(v[idx]) for n,v in arrays.items()})
        predictor=RFQCPredictor.from_directory(a.bundles/f'{name}-seed{seed}')
        result=predictor.predict(data,batch_size=32)
        expected=np.array([float(r['p_good']) for r in rows]);error=np.abs(result.p_good-expected)
        # Keep the initial cross-platform tolerance as a diagnostic, not an
        # adjustable acceptance gate. Isolate packaging via original code on CPU.
        cpu_error=None
        if predictor.spec.kind=='neural':
            original=original_cpu_probabilities(a.workspace,name,source,data,seed)
            cpu_error=float(np.max(np.abs(result.p_good-original)))
            np.testing.assert_allclose(result.p_good,original,rtol=0,atol=1e-7,err_msg=f'packaging changed {name}/{seed}')
        else:
            np.testing.assert_allclose(result.p_good,expected,rtol=0,atol=1e-10)
        changed=int(np.sum(result.prediction!=np.array([int(r['prediction']) for r in rows])))
        checks.append(dict(method=name,seed=seed,records_compared=len(rows),max_absolute_score_error=float(error.max()),
                           archived_decisions_changed=changed,cross_platform_within_initial_tolerance=bool(error.max()<=1e-4),
                           original_cpu_vs_package_max_error=cpu_error,full_station_context=(name=='xiong2025_fcm')))
        print(name,seed,len(rows),float(error.max()),'original_cpu_error',cpu_error,'changed_decisions',changed,flush=True)
    atomic_json(ROOT/'validation/archive_parity.json',dict(schema=1,models_verified=len(checks),
        observed_data_distributed=False,torch_validation=str(torch.__version__),device='cpu',initial_cross_platform_tolerance=1e-4,
        acceptance='Package vs original code on the same CPU <=1e-7; statistical archive replay <=1e-10. Cross-platform drift is separately retained, including failures of the initial 1e-4 tolerance.',
        sample_selection='128 evenly spaced positions in each ordered archive, or the full test set for same-station FCM context',
        scope='Deployment consistency check, not new accuracy estimates or a re-evaluation of model selection',checks=checks))


if __name__=='__main__':main()
