"""Render only completed, verified appended analyses; never edit original tables."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];O=P/'SourceData/resolution_20260930'
def rows(name):return list(csv.DictReader((O/name).open()))
def f(x):return float(x)
def write(name,text):(O/name).write_text('\n'.join(line.rstrip() for line in text.rstrip().splitlines())+'\n')
def tabular(headers,body,fmt):return '\\begin{tabular}{'+fmt+'}\n\\toprule\n'+' & '.join(headers)+r'\\'+'\n\\midrule\n'+'\n'.join(' & '.join(r)+r'\\' for r in body)+'\n\\bottomrule\n\\end{tabular}\n'
def wrap_table(caption,label,contents):return '\n\\begin{table}[!htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n'+contents+'\\end{table}\n'

def positive():
 if not (O/'positive_control_summary.json').exists():return
 summary=json.loads((O/'positive_control_summary.json').read_text());noise=rows('input_perturbation.csv');labels=rows('label_perturbation.csv');body=[];phrases=[]
 for cell in summary['detected_grid_effects']:
  r=cell['smallest_observed_detected_drop'];family='Input noise' if cell['family']=='input' else 'Label flips';model='AG3' if cell['variant']=='ag3' else 'Multi-filter'
  if r:
   level=(f"{f(r['snr_db']):g} dB" if cell['family']=='input' else f"{100*f(r['epsilon']):g}\\%")
   body.append([family,model,level,f"{f(r['mean_drop_pp']):.3f}",f"[{f(r['lower_pp']):.3f}, {f(r['upper_pp']):.3f}]"])
  else:body.append([family,model,'None','--','--'])
 t=tabular(['Perturbation','Reference','Level','Drop (pp)','95\\% interval'],body,'lllrr')
 top=f(summary['top_five_range_pp']);allrange=f(summary['all_eight_range_pp'])
 text=(r"The perturbation controls produced the sensitivity curves in Figure~\ref{fig:resolution-controls}. Table~\ref{tab:resolution-controls} reports the smallest observed macro-$F_1$ drop meeting the fixed detection rule for each perturbation family and reference input. The unchanged top-five literature-neural mean-score range was "+f'{top:.3f}'+r" percentage points, compared with "+f'{allrange:.3f}'+r" points across all eight literature neural models. The comparison places each observed perturbation effect on the same macro-$F_1$ scale as the model differences.")
 text+=wrap_table(r'Smallest operationally detected effects on the tested perturbation grid. Drops average all three training seeds and all perturbation realizations. Intervals are paired station-bootstrap percentiles; every individual drop must also be positive. Noise level is power SNR and flip level is a Bernoulli probability; realized counts are supplied separately. These are grid-specific stress responses.', 'tab:resolution-controls',t)
 text+=r'''
\begin{figure}[!htbp]
\centering\includegraphics[width=\linewidth]{figures/resolution_controls.pdf}
\caption{Frozen-model perturbation sensitivity. Left: added raw-waveform Gaussian noise before preprocessing; right: synthetic label flips with scores held fixed. Points give mean macro-$F_1$ drops and bars paired station-bootstrap 95\% intervals. Each input regime includes three trained seeds, three noise realizations or twenty label-flip realizations per level. Labels are perturbed solely for this synthetic sensitivity calculation.}
\label{fig:resolution-controls}
\end{figure}
'''
 write('positive_results.tex',text)
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42});fig,axes=plt.subplots(1,2,figsize=(9.6,3.7),layout='constrained')
 for variant,color,label in [('ag3','#236493','Reference–AG3'),('multiband','#b94e32','Reference–multi-filter')]:
  for ax,data,xkey in [(axes[0],noise,'snr_db'),(axes[1],labels,'epsilon')]:
   selected=[r for r in data if r['variant']==variant];x=np.array([f(r[xkey]) for r in selected]);x=x*100 if xkey=='epsilon' else x;y=np.array([f(r['mean_drop_pp']) for r in selected]);lo=np.array([f(r['lower_pp']) for r in selected]);hi=np.array([f(r['upper_pp']) for r in selected]);ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),color=color,marker='o',lw=1.5,capsize=3,label=label)
 for ax in axes:ax.axhline(0,color='.6',lw=.8);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_ylabel('Macro-F1 drop (percentage points)')
 axes[0].set_xlabel('Added-noise power SNR (dB)');axes[0].invert_xaxis();axes[0].set_title('(a) Input degradation');axes[1].set_xlabel('Label-flip probability (%)');axes[1].set_title('(b) Metric sensitivity');axes[1].legend(frameon=False,fontsize=9)
 fig.savefig(P/'figures/resolution_controls.pdf');fig.savefig(P/'figures/resolution_controls.png',dpi=180);plt.close(fig)

def seeds():
 if not (O/'eight_seed_verification.json').exists():return
 source=rows('eight_seed_metrics.csv');summary=rows('eight_seed_summary.csv');names={'reference_ag3':'Reference--AG3','reference_multifilter':'Reference--multi-filter','gong_cnn':'Gong-CNN','gong_cnn_bilstm':'Gong-CNN--BiLSTM'};body=[]
 for r in summary:body.append([names[r['first']]+r' $-$ '+names[r['second']],f"{f(r['mean_pp']):+.3f}",f"{f(r['sample_sd_pp']):.3f}",f"[{f(r['lower_pp']):+.3f}, {f(r['upper_pp']):+.3f}]",f"{f(r['half_width_pp']):.3f}"])
 primary=next(r for r in summary if r['first']=='reference_multifilter' and r['second']=='reference_ag3');lit=next(r for r in summary if r['first']=='gong_cnn_bilstm' and r['second']=='gong_cnn')
 text=(r"Eight-seed paired comparisons quantified the precision of the selected contrasts (Table~\ref{tab:eight-seeds}). The half-width was "+f"{f(primary['half_width_pp']):.3f}"+r" percentage points for multi-filter minus AG3 and "+f"{f(lit['half_width_pp']):.3f}"+r" points for Gong-CNN--BiLSTM minus Gong-CNN. ")
 if (O/'positive_control_summary.json').exists():
  grid=json.loads((O/'positive_control_summary.json').read_text());v=f(lit['half_width_pp']);span=f(grid['all_eight_range_pp']);small=f(grid['top_five_range_pp']);text+=f"The latter half-width was {'larger' if v>span else 'smaller'} than the original eight-literature mean-score range ({span:.3f} points) and {'larger' if v>small else 'smaller'} than its top-five range ({small:.3f} points)."
 write('seed_precision_results.tex',text)
 text=(r"With eight seeds, the paired multi-filter-minus-AG3 mean difference was "+f"${f(primary['mean_pp']):+.3f}\\pm{f(primary['sample_sd_pp']):.3f}$"+r" percentage points (sample SD), with a 95\% interval of "+f"$[{f(primary['lower_pp']):+.3f},{f(primary['upper_pp']):+.3f}]$"+r". The Gong-CNN--BiLSTM-minus-Gong-CNN mean difference was "+f"{f(lit['mean_pp']):+.3f}"+r" points, with interval "+f"$[{f(lit['lower_pp']):+.3f},{f(lit['upper_pp']):+.3f}]$"+r". ")
 for r,label in [(primary,'The input contrast'),(lit,'The selected literature contrast')]:text+=label+(' had an interval spanning both signs. ' if r['contains_zero']=='True' else ' had an interval entirely '+('above' if f(r['lower_pp'])>0 else 'below')+' zero. ')
 text+=wrap_table(r'All six paired contrasts for the four configurations with eight training seeds. Every value is in macro-$F_1$ percentage points. Half-widths use the paired Student-$t$ interval with seven degrees of freedom, without multiplicity adjustment. The three-seed primary table is unchanged; the append adds twenty fits. These intervals describe training-seed variation on the fixed split.', 'tab:eight-seeds','\\resizebox{\\linewidth}{!}{'+tabular(['Contrast','Mean','SD','95\\% interval','Half-width'],body,'lrrrr')+'}')
 write('seed_contrast_results.tex',text)
 allrows=[]
 for n,label in names.items():
  vals=sorted((int(r['seed']),f(r['macro_f1_percent'])) for r in source if r['method']==n)
  for seed,val in vals:allrows.append([label,str(seed),f'{val:.5f}'])
 write('eight_seed_individuals.tex',tabular(['Configuration','Seed','Macro-$F_1$ (\\%)'],allrows,'llr'))

def cross():
 if not (O/'cross_protocol_verification.json').exists():return
 r=rows('cross_protocol_metrics.csv');old=list(csv.DictReader((P/'SourceData/benchmark_v1/run_metrics.csv').open()));keys=['li2021_cnn','gan2021_cnn','gong_cnn','gong_cnn_bilstm','gan_cnn','deeprfqc','hegaz_capsule','chen2026_image'];means=[100*np.mean([f(x['macro_f1']) for x in old if x['key']==k]) for k in keys];low,high=min(means),max(means);text='The bidirectional matched-cohort experiment yielded different within- and cross-protocol agreement levels (Table~\\ref{tab:cross-protocol}). ';body=[];interpret=[]
 for direction,label in [('strict_to_ag3',r'Strict $\rightarrow$ AG3 protocol'),('ag3_to_strict',r'AG3 protocol $\rightarrow$ strict')]:
  selected=[v for v in r if v['direction']==direction];assert len(selected)==3
  source=np.array([f(v['source_macro_f1_percent']) for v in selected]);target=np.array([f(v['target_macro_f1_percent']) for v in selected]);ap=np.array([f(v['target_ap_percent']) for v in selected]);delta=target-source;mean=target.mean();where='above' if mean>high else 'below' if mean<low else 'within';positions=['above' if v>high else 'below' if v<low else 'within' for v in target]
  body.append([label,f'${source.mean():.3f}\\pm{source.std(ddof=1):.3f}$',f'${mean:.3f}\\pm{target.std(ddof=1):.3f}$',f'${delta.mean():+.3f}\\pm{delta.std(ddof=1):.3f}$',f'${ap.mean():.3f}\\pm{ap.std(ddof=1):.3f}$'])
  text+=label+f" achieved {mean:.3f}\\% mean target-protocol macro-$F_1$, {where} the original literature-neural range ({low:.3f}--{high:.3f}\\%). "
  text+=('All three seed scores had the same range placement. ' if len(set(positions))==1 else 'The three seed scores crossed a range boundary. ')
  interpret.append({'direction':direction,'mean_target_macro_f1_percent':float(mean),'band_placement':where,'individual_band_placements':positions,'target_minus_source_mean_pp':float(delta.mean())})
 text+='The source-to-target score changes quantify screening-protocol sensitivity for fixed model decisions on the same matched test records.'
 text+=wrap_table(r'Bidirectional screening-protocol prediction on the same 18,305 test records at 22 held-out stations. Values are mean $\pm$ sample SD over three seeds; classification metrics are percentages and the difference is in percentage points. Training, checkpoint selection and threshold selection use source-protocol labels only. Both source and target test scores use the same predictions. The strict/AG3 protocols contain 2,086/2,367 accepted test RFs, respectively.','tab:cross-protocol','\\resizebox{\\linewidth}{!}{'+tabular(['Train/validation $\\rightarrow$ test','Source macro-$F_1$','Target macro-$F_1$','Target $-$ source','Target AP'],body,'lrrrr')+'}')
 write('cross_protocol_results.tex',text);write('cross_protocol_interpretation.json',json.dumps(interpret,indent=2))

if __name__=='__main__':positive();seeds();cross()
