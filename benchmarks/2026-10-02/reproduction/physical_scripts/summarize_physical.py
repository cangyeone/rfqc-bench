"""Aggregate station diagnostics without pooling station and subsample repetitions."""
import csv,json
import numpy as np
from analyze_physical import ROOT,SPECS,table,sha

def read(name):
    with (ROOT/'results'/name).open() as f:return list(csv.DictReader(f))
def finite(x):return np.asarray([v for v in x if np.isfinite(v)],float)
def med(rows,key,absolute=False):
    a=finite([float(r[key]) for r in rows]);return float(np.median(np.abs(a) if absolute else a)) if len(a) else np.nan

def main():
    stack=read('stack_metrics.csv');cross=read('crossband_by_station.csv');out=[];features=[]
    names=[('all','All RFs'),('manual','Manual good')]+[(s[0],s[1]) for s in SPECS]
    for method,label in names:
        shape=[r for r in stack if r['method']==method and float(r['fraction'])==1 and r['normalization']=='peak_normalized' and r['comparable']=='True']
        raw=[r for r in stack if r['method']==method and float(r['fraction'])==1 and r['normalization']=='raw' and r['comparable']=='True']
        cc=[r for r in cross if r['method']==method and r['window']=='post_p' and int(r['n_selected'])>=3]
        row=dict(method=method,label=label,n_stations=len(shape),retained_total=sum(int(r['n_selected']) for r in shape),
            median_shape_correlation=med(shape,'corr_post_p'),median_raw_nrmse=med(raw,'nrmse_whole'),
            median_raw_P_amplitude_relative_error_pct=float(np.median([100*abs(float(r['delta_P_amp'])/float(r['manual_P_amp'])) for r in raw])),
            median_raw_Ps_amplitude_relative_error_pct=float(np.median([100*abs(float(r['delta_Ps_amp'])/float(r['manual_Ps_amp'])) for r in raw])),
            median_shape_P_time_abs_error_s=med(shape,'delta_P_time',True),
            median_shape_Ps_time_abs_error_s=med(shape,'delta_Ps_time',True),
            median_crossband_correlation=med(cc,'mean_correlation'),
            median_count_control_difference=float(np.median([float(r['mean_correlation'])-float(r['count_matched_random_mean']) for r in cc])))
        out.append(row)
        for norm in ['raw','peak_normalized']:
            data=raw if norm=='raw' else shape
            for key in ['P_amp','P_time','Ps_amp','Ps_time','MaxAmp','MaxTime','NegMinAmp','NegPosRatio','PPsRatio','PsPeaks']:
                vals=finite([abs(float(r['delta_'+key])) for r in data])
                features.append(dict(method=method,normalization=norm,descriptor=key,stations=len(vals),
                    median_absolute_delta=float(np.median(vals)),q25=float(np.quantile(vals,.25)),q75=float(np.quantile(vals,.75)),
                    max_absolute_delta=float(vals.max()),nonzero_stations=int((vals>1e-8).sum())))
    table(ROOT/'results/summary.csv',out);table(ROOT/'results/feature_summary.csv',features)
    sampled=[]
    for method,label in names:
        for fraction in [1.,.5,.25]:
            rows=[r for r in stack if r['method']==method and float(r['fraction'])==fraction and r['normalization']=='peak_normalized' and r['comparable']=='True']
            vals=finite([float(r['corr_post_p']) for r in rows])
            sampled.append(dict(method=method,fraction=fraction,comparable_station_subsets=len(vals),
                median_correlation=float(np.median(vals)),q25=float(np.quantile(vals,.25)),q75=float(np.quantile(vals,.75)),min=float(vals.min())))
    table(ROOT/'results/subsample_summary.csv',sampled)
    associations=np.load(ROOT/'results/crossband_per_record.npz');selections=np.load(ROOT/'results/selection_masks.npz')
    np.testing.assert_array_equal(associations['sample_ids'],selections['sample_ids'])
    pooled=[]
    for window_name in ['whole','post_p','ps_candidate']:
        values=np.nanmean(associations[window_name],axis=1)
        for method,mask in zip(selections['methods'],selections['selected']):
            v=values[mask];pooled.append(dict(method=str(method),window=window_name,n_selected=int(mask.sum()),n_finite=int(np.isfinite(v).sum()),mean_correlation=float(np.nanmean(v))))
    table(ROOT/'results/crossband_pooled.csv',pooled)

    tex=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Method & Retained & Shape $r$ & Raw NRMSE & $E_P$ (\%) & $E_{Ps}$ (\%) & Cross-band $r$\\',r'\midrule']
    for r in out:
        value=r['median_raw_nrmse']; exponent=int(np.floor(np.log10(value))) if value else 0
        err=f'{value:.4f}' if value<100 else '$'+f'{value/10**exponent:.2f}'+r'\times10^{'+str(exponent)+'}$'
        ep=f"{r['median_raw_P_amplitude_relative_error_pct']:.2f}" if r['median_raw_P_amplitude_relative_error_pct']<1e5 else f"{r['median_raw_P_amplitude_relative_error_pct']:.2e}"
        es=f"{r['median_raw_Ps_amplitude_relative_error_pct']:.2f}" if r['median_raw_Ps_amplitude_relative_error_pct']<1e5 else f"{r['median_raw_Ps_amplitude_relative_error_pct']:.2e}"
        tex.append(f"{r['label']} & {r['retained_total']:,} & {r['median_shape_correlation']:.4f} & {err} & {ep} & {es} & {r['median_crossband_correlation']:.4f}"+r'\\')
    tex.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'results/physical_summary.tex').write_text('\n'.join(tex)+'\n')
    (ROOT/'audit/summary_verification.json').write_text(json.dumps(dict(
        statistic='Median over 26 test stations. Subsample summaries show station-subset distribution, not independent replicates or CIs.',
        definitions={'raw_amplitude_error_pct':'100*abs(auto_peak-manual_peak)/abs(manual_peak); same raw stack windows',
        'crossband':'Station mean of per-RF means across observed native Gaussian pairs; then station median'},
        script_sha256=sha(__file__),output_sha256={n:sha(ROOT/'results'/n) for n in ['summary.csv','feature_summary.csv','subsample_summary.csv','physical_summary.tex','crossband_pooled.csv']}),indent=2)+'\n')
    print(json.dumps([r for r in out if r['method'].startswith('ours')],indent=2))
if __name__=='__main__':main()
