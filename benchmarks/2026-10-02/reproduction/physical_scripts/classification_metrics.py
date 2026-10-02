"""Independent confusion-count, grouped-score AP and pair-count AUC formulas.
Copied from the existing audited paper analyzer; no sklearn dependency.
"""
import numpy as np
def ratio(a,b):return float(a/b) if b else 0.
def metrics(y,p,t):
 """Independent confusion-count, grouped-score AP, and pair-count AUC formulas."""
 pred=p>=t
 tn=int(np.sum((y==0)&~pred));fp=int(np.sum((y==0)&pred))
 fn=int(np.sum((y==1)&~pred));tp=int(np.sum((y==1)&pred))
 n1=tp+fn;n0=tn+fp
 ap=auc=None
 if n1 and n0:
  order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
  ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
  cum=np.cumsum(ys)[ends]
  ap=float(np.sum(np.diff(np.r_[0,cum])/n1*cum/(ends+1)))
  order=np.argsort(p,kind='stable');ys=y[order];ps=p[order]
  ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
  pos=np.diff(np.r_[0,np.cumsum(ys)[ends]])
  neg=np.diff(np.r_[0,np.cumsum(1-ys)[ends]])
  auc=float(np.sum(pos*(np.cumsum(neg)-neg+0.5*neg))/(n1*n0))
 return dict(accuracy=ratio(tp+tn,len(y)),balanced_accuracy=(ratio(tp,n1)+ratio(tn,n0))/2 if n1 and n0 else None,
  macro_f1=(ratio(2*tp,2*tp+fp+fn)+ratio(2*tn,2*tn+fp+fn))/2,
  good_precision=ratio(tp,tp+fp),good_recall=ratio(tp,n1),bad_recall=ratio(tn,n0),
  good_auprc=ap,auroc=auc,confusion_matrix_bad_good=[[tn,fp],[fn,tp]],threshold=float(t),n=len(y))
