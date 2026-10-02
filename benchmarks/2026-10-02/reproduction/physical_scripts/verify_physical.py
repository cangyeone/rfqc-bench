"""Independent numpy/scipy reconstruction of full-input stacks and diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from analyze_physical import ROOT,sha

def main(workspace):
    cache=workspace/'rf_quality_control/cache';ix=np.load(cache/'test_indices.npy')
    waves=np.asarray(np.load(cache/'waveforms.npy',mmap_mode='r')[ix],float)
    z=np.load(ROOT/'results/selection_masks.npz');stations=z['stations'];ids=z['sample_ids']
    np.testing.assert_array_equal(ids,np.load(cache/'sample_ids.npy')[ix])
    saved=np.load(ROOT/'results/stacks.npz');stored=dict(zip(saved['keys'],saved['waveforms']));t=saved['time']
    masks=dict(zip(z['methods'],z['selected']));shape=waves/np.maximum(np.abs(waves).max(-1,keepdims=True),1e-100)
    rebuilt={}
    for station in np.unique(stations):
        for method,keep in masks.items():
            selected=(stations==station)&keep
            for norm,data in [('raw',waves),('peak_normalized',shape)]:
                key=f'{station}|{method}|{norm}';value=np.mean(data[selected,4],axis=0)
                np.testing.assert_allclose(value,stored[key],rtol=1e-11,atol=1e-12);rebuilt[key]=value
    with (ROOT/'results/stack_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    checked=0
    for r in rows:
        if float(r['fraction'])!=1:continue
        x=rebuilt[f"{r['station']}|{r['method']}|{r['normalization']}"]
        y=rebuilt[f"{r['station']}|manual|{r['normalization']}"]
        for name,lo,hi in [('whole',-5,20),('post_p',2.5,15),('ps_candidate',2.5,7.5)]:
            m=(t>=lo-1e-8)&(t<=hi+1e-8)
            np.testing.assert_allclose(pearsonr(x[m],y[m]).statistic,float(r['corr_'+name]),rtol=1e-11,atol=1e-12)
            e=np.sqrt(np.mean((x[m]-y[m])**2))/np.sqrt(np.mean(y[m]**2))
            np.testing.assert_allclose(e,float(r['nrmse_'+name]),rtol=1e-10,atol=1e-12);checked+=2
    cross=np.load(ROOT/'results/crossband_per_record.npz');length=np.load(cache/'lengths.npy')[ix];pair_checks=0
    m=(t>=2.5)&(t<=15)
    for row in range(0,len(ix),257):
        col=0
        for a in range(7):
            for b in range(a+1,7):
                value=cross['post_p'][row,col];col+=1
                if b>=length[row]:assert np.isnan(value);continue
                reference=np.corrcoef(waves[row,a,m],waves[row,b,m])[0,1]
                np.testing.assert_allclose(reference,value,rtol=1e-11,atol=1e-12);pair_checks+=1
    report=dict(passed=True,reconstructed_stacks=len(rebuilt),full_input_correlation_nrmse_checks=checked,
        independent_crossband_pair_checks=pair_checks,method='Direct numpy mean/RMSE, scipy.stats.pearsonr and numpy.corrcoef; fixed every-257th-record cross-band audit.',
        verified_outputs={n:sha(ROOT/'results'/n) for n in ['stacks.npz','stack_metrics.csv','crossband_per_record.npz']},script_sha256=sha(__file__))
    (ROOT/'audit/independent_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);main(p.parse_args().workspace)
