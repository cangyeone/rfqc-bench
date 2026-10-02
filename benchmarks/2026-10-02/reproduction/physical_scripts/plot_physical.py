"""Static publication figures; every point is a station or a prespecified sample."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_physical import SPECS,ROOT,sha

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':230})
LABELS={'all':'All RFs','manual':'Manual good',**{s[0]:s[1] for s in SPECS}}
METHODS=['all']+[s[0] for s in SPECS]


def load(name):
    with (ROOT/'results'/name).open() as f:return list(csv.DictReader(f))


def save(fig,name):
    fig.savefig(ROOT/'figures'/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(ROOT/'figures'/(name+'.png'),bbox_inches='tight')
    plt.close(fig)


def boxes(ax,values,methods,xlabel):
    b=ax.boxplot(values,orientation='horizontal',tick_labels=[LABELS[m] for m in methods],
        patch_artist=True,widths=.54,showfliers=True,
        flierprops={'marker':'.','markersize':3,'alpha':.55},medianprops={'color':'#252525','linewidth':1})
    for m,box in zip(methods,b['boxes']):
        box.set_facecolor('#D99648' if m.startswith('ours') else '#E1E1E1' if m in ['all','manual'] else '#8DB0C7')
    ax.invert_yaxis();ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.18)


def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    rows=load('stack_metrics.csv');cross=load('crossband_by_station.csv')
    full=[r for r in rows if float(r['fraction'])==1 and r['comparable']=='True']
    shape=[r for r in full if r['normalization']=='peak_normalized']
    raw=[r for r in full if r['normalization']=='raw']
    z=np.load(ROOT/'results/stacks.npz');stacks=dict(zip(z['keys'],z['waveforms']));t=z['time']
    stations=['DB_EW27','YP_NE8A','YP_NE69']
    groups=[[s[0] for s in SPECS[:4]],[s[0] for s in SPECS[4:8]],[s[0] for s in SPECS[8:]]]
    colors=['#21618C','#B56821','#775B9A','#647443']
    fig=plt.figure(figsize=(8.2,10.4),layout='constrained')
    grid=fig.add_gridspec(3,3)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color='#161616',lw=1.6,label='Manual good'),Line2D([0],[0],color='#888888',ls=':',label='All RFs')],loc='outside upper center',ncol=2,frameon=False,fontsize=10)
    for i,station in enumerate(stations):
        for j,group in enumerate(groups):
            sub=grid[i,j].subgridspec(2,1,height_ratios=[3.8,1.3],hspace=.03)
            ax=fig.add_subplot(sub[0]);legend_ax=fig.add_subplot(sub[1]);legend_ax.axis('off')
            method_handles=[]
            manual=stacks[f'{station}|manual|peak_normalized']
            ax.plot(t,manual,color='#161616',lw=1.6,label='Manual good')
            ax.plot(t,stacks[f'{station}|all|peak_normalized'],color='#888888',ls=':',lw=1,label='All RFs')
            for k,method in enumerate(group):
                w=stacks.get(f'{station}|{method}|peak_normalized')
                if w is None:continue
                n=next(int(r['n_selected']) for r in shape if r['station']==station and r['method']==method)
                line,=ax.plot(t,w,lw=1,color=colors[k],ls=['-','--','-.',':'][k],label=f'{LABELS[method]} (n={n})');method_handles.append(line)
            ng=next(int(r['n_manual']) for r in shape if r['station']==station)
            ax.set_title(f'{chr(97+i*3+j)}) {station} | manual n={ng}',loc='left',fontsize=10)
            ax.axvspan(2.5,7.5,color='#BBBBBB',alpha=.13);ax.axvline(0,color='#999999',lw=.5)
            legend_ax.legend(handles=method_handles,fontsize=8.5,loc='upper left',frameon=False,handlelength=1.8,borderaxespad=0,labelspacing=.2);ax.set_xlim(-5,20)
            ax.tick_params(labelsize=9)
            if j:ax.tick_params(labelleft=False)
            ax.set_ylim(-.55,1.08)
            if j==0:ax.set_ylabel('Mean peak-normalized RF')
            if i<2:ax.tick_params(labelbottom=False)
    fig.supxlabel('Time relative to P (s)',fontsize=10)
    save(fig,'stack_waveform_comparison')

    fig,axes=plt.subplots(2,2,figsize=(9,8.8),layout='constrained')
    specs=[(shape,'corr_post_p',lambda x:x,'a) Stack shape: 2.5-15 s','Zero-lag Pearson r'),
           (raw,'nrmse_whole',lambda x:np.log10(1+x),'b) Raw-amplitude error: -5 to 20 s','log10(1 + NRMSE to manual stack)'),
           (shape,'delta_P_time',abs,'c) P-window maximum time','Absolute time difference (s)'),
           (shape,'delta_Ps_time',abs,'d) Ps-candidate time: 2.5-7.5 s','Absolute time difference (s)')]
    for ax,(data,key,transform,title,xlabel) in zip(axes.flat,specs):
        metric_methods=[s[0] for s in SPECS]
        values=[[transform(float(r[key])) for r in data if r['method']==m and np.isfinite(float(r[key]))] for m in metric_methods]
        boxes(ax,values,metric_methods,xlabel);ax.set_title(title,loc='left')
    save(fig,'stack_metric_comparison')

    methods=['all','manual']+[s[0] for s in SPECS]
    fig,axes=plt.subplots(1,2,figsize=(9.5,6.5),layout='constrained')
    selected=[r for r in cross if r['window']=='post_p' and int(r['n_selected'])>=3]
    values=[[float(r['mean_correlation']) for r in selected if r['method']==m and np.isfinite(float(r['mean_correlation']))] for m in methods]
    boxes(axes[0],values,methods,'Mean within-RF cross-band Pearson r')
    axes[0].set_title('a) Native filter views, 2.5-15 s',loc='left');axes[0].set_xlim(.72,.89)
    values=[[float(r['mean_correlation'])-float(r['count_matched_random_mean']) for r in selected if r['method']==m and np.isfinite(float(r['mean_correlation']))] for m in methods]
    boxes(axes[1],values,methods,'Cross-band r minus random control')
    axes[1].axvline(0,color='#666666',lw=.7,ls='--');axes[1].set_title('b) Same station and retained count',loc='left')
    save(fig,'crossband_association')

    fig,axes=plt.subplots(1,3,figsize=(9.5,6.5),sharex=True,sharey=True,layout='constrained')
    for ax,fraction in zip(axes,[1,.5,.25]):
        data=[r for r in rows if float(r['fraction'])==fraction and r['normalization']=='peak_normalized' and r['comparable']=='True']
        values=[[float(r['corr_post_p']) for r in data if r['method']==m and np.isfinite(float(r['corr_post_p']))] for m in METHODS]
        boxes(ax,values,METHODS,'Stack-to-manual Pearson r (2.5-15 s)')
        ax.set_title(f'{int(fraction*100)}% of input RFs',loc='left');ax.set_xlim(0,1.005);ax.set_xlabel('')
    fig.supxlabel('Stack-to-manual Pearson r (2.5-15 s)',fontsize=10.5)
    save(fig,'subsample_stack_comparison')
    assert min(float(r['corr_post_p']) for r in rows if r['normalization']=='peak_normalized' and r['comparable']=='True')>=0, 'Subsample axis would clip values'
    assert all(.72<=float(r['mean_correlation'])<=.89 for r in selected), 'Cross-band axis would clip values'
    meta=dict(figure_count=4,model_seed=20260929,axis_clipping_checks_passed=True,
        statistic='Boxes show station distributions; sampled cases additionally include 3 fixed subset seeds. Not confidence intervals.',
        units='Shape curves use each-trace absolute-peak normalization. Raw-amplitude metrics preserve original data.',
        source_sha256={n:sha(ROOT/'results'/n) for n in ['stack_metrics.csv','stacks.npz','crossband_by_station.csv']},
        output_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (ROOT/'audit/figure_verification.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
