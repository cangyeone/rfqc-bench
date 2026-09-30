"""Public evaluation helpers; probabilities always refer to good=1."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score)


def evaluate(labels, p_good, threshold=.5):
    y=np.asarray(labels);p=np.asarray(p_good,dtype=float)
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected equally sized binary labels and finite probabilities')
    if not 0<=threshold<=1: raise ValueError('threshold must be in [0,1]')
    predicted=p>=threshold;both=len(np.unique(y))==2
    return dict(accuracy=float(accuracy_score(y,predicted)),macro_f1=float(f1_score(y,predicted,labels=[0,1],average='macro',zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y,predicted)),
                good_precision=float(precision_score(y,predicted,zero_division=0)),good_recall=float(recall_score(y,predicted,zero_division=0)),
                bad_recall=float(recall_score(y,predicted,pos_label=0,zero_division=0)),
                good_auprc=float(average_precision_score(y,p)) if np.any(y==1) else None,
                auroc=float(roc_auc_score(y,p)) if both else None,threshold=float(threshold),n=len(y))


def select_threshold(labels, probabilities):
    """Validation-only maximum macro F1; choose first maximum on .01..99 grid."""
    y=np.asarray(labels)
    if set(np.unique(y))!={0,1}: raise ValueError('Threshold selection requires both validation classes')
    grid=np.linspace(.01,.99,99)
    return float(grid[np.argmax([f1_score(y,np.asarray(probabilities)>=t,average='macro',zero_division=0) for t in grid])])


def station_bootstrap(labels, first, second, stations, thresholds=(.5,.5), draws=10000, seed=20260930):
    """Paired station-cluster bootstrap of pooled macro F1 (percentage points)."""
    y=np.asarray(labels);stations=np.asarray(stations)
    evaluate(y,first,thresholds[0]);evaluate(y,second,thresholds[1])
    if stations.shape!=y.shape or draws<1: raise ValueError('Invalid station vector or draw count')
    unique=np.unique(stations);idx=np.random.default_rng(seed).integers(0,len(unique),(draws,len(unique)))
    multiplicity=np.stack([(idx==i).sum(1) for i in range(len(unique))],axis=1)
    scores=[]
    for p,t in zip([first,second],thresholds):
        codes=2*y.astype(int)+(np.asarray(p)>=t).astype(int)
        c=np.array([np.bincount(codes[stations==s],minlength=4) for s in unique])
        tn,fp,fn,tp=(multiplicity@c).T
        a=np.divide(tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
        b=np.divide(tn,2*tn+fp+fn,out=np.zeros_like(tn,dtype=float),where=(2*tn+fp+fn)>0)
        scores.append(a+b)
    d=100*(scores[0]-scores[1]);lo,median,hi=np.percentile(d,[2.5,50,97.5])
    return dict(median_pp=float(median),lower_pp=float(lo),upper_pp=float(hi),seed=seed,draws=draws,stations=len(unique),
                interpretation='does not separate at station level' if lo<=0<=hi else 'separates under this resampling scheme')


def label_agreement(first,second):
    a=np.asarray(first);b=np.asarray(second)
    if a.ndim!=1 or a.shape!=b.shape or not len(a) or not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():
        raise ValueError('Pass aligned nonempty binary label arrays')
    k=cohen_kappa_score(a,b)
    return dict(n=len(a),agreement=float(np.mean(a==b)),kappa=float(k) if np.isfinite(k) else None,
                counts=np.bincount(2*a.astype(int)+b.astype(int),minlength=4).reshape(2,2).tolist())
