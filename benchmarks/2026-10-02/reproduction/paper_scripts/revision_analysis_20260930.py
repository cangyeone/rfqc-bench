"""Append-only T4/T5/T6 analyses. Never rewrite registered benchmark artifacts."""
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / 'SourceData/revision_20260930'
WORK = PAPER.parent
SEEDS = [20260928, 20260929, 20260930]
sys.path.insert(0, str(PAPER / 'scripts'))
from analyze_results import metrics, check


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(path, rows):
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def begin(task):
    reg = OUT / 'registration.json'
    receipt = json.loads((OUT / 'registration_receipt.json').read_text())
    assert sha(reg) == receipt['registration_sha256']
    marker = OUT / (task + '_started.json')
    if marker.exists():
        raise FileExistsError(f'{task} already executed; preserve its output')
    dump(marker, dict(started_at=datetime.now(timezone.utc).isoformat(),
                     registration_sha256=sha(reg)))


def macro(c):
    tn, fp, fn, tp = np.moveaxis(np.asarray(c, dtype=float), -1, 0)
    return .5 * (np.divide(2*tn, 2*tn+fp+fn, out=np.zeros_like(tn), where=2*tn+fp+fn>0)
                 + np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp), where=2*tp+fp+fn>0))


def bootstrap():
    begin('T4')
    from sklearn.metrics import f1_score
    source = PAPER / 'SourceData/benchmark_v1'
    z = np.load(source / 'predictions.npz')
    lookup = dict(zip(z['run_names'], z['probabilities']))
    thresholds = {(r['key'], int(r['seed'])): float(r['threshold'])
                  for r in rows(source / 'run_metrics.csv')}
    stations = np.unique(z['stations']); assert len(stations) == 26
    draws = np.random.default_rng(20260930).integers(0, 26, size=(10000, 26))
    weights = np.stack([np.bincount(x, minlength=26) for x in draws])
    station_index = np.searchsorted(stations, z['stations'])
    y = z['labels']; outputs = []; distributions = []
    for name, a, b in [('input', 'ours_multiband', 'ours_ag3'),
                       ('literature', 'gong_cnn_bilstm', 'ours_multiband')]:
        for seed in SEEDS:
            preds = [lookup[f'{key}/{seed}'] >= thresholds[key, seed] for key in [a, b]]
            counts = []
            for pred in preds:
                code = 2*y.astype(int) + pred.astype(int)
                counts.append(np.stack([np.bincount(code[z['stations']==s], minlength=4)
                                        for s in stations]))
            delta = 100 * (macro(weights @ counts[0]) - macro(weights @ counts[1]))
            # Independent implementation at original cohort and 20 bootstrap draws.
            original = 100 * (f1_score(y, preds[0], average='macro')
                              - f1_score(y, preds[1], average='macro'))
            np.testing.assert_allclose(original, 100*(macro(counts[0].sum(0))-macro(counts[1].sum(0))), atol=1e-12)
            for i in range(20):
                w = weights[i, station_index]
                independent = 100*(f1_score(y, preds[0], average='macro', sample_weight=w)
                                   - f1_score(y, preds[1], average='macro', sample_weight=w))
                np.testing.assert_allclose(delta[i], independent, atol=1e-12)
            lo, med, hi = np.percentile(delta, [2.5, 50, 97.5])
            outputs.append(dict(contrast=name, first=a, second=b, seed=seed,
                                original_pp=float(original), median_pp=float(med), lower_pp=float(lo),
                                upper_pp=float(hi), contains_zero=bool(lo<=0<=hi),
                                interpretation='does not separate at station level' if lo<=0<=hi
                                else 'separates under this resampling scheme'))
            distributions.append(delta)
    table(OUT/'station_bootstrap.csv', outputs)
    np.savez_compressed(OUT/'station_bootstrap_draws.npz', stations=stations, draws=draws,
                        differences_pp=np.stack(distributions))
    dump(OUT/'T4_verification.json', dict(seed=20260930, draws=10000, stations=26,
         independent_sklearn_checks=126, source_sha256={n:sha(source/n) for n in ['predictions.npz','run_metrics.csv']},
         registration_sha256=sha(OUT/'registration.json')))
    print(json.dumps(outputs, indent=2))


def label_summary(c):
    c = np.asarray(c, dtype=int).reshape(2,2); n = int(c.sum())
    observed = float(np.trace(c)/n) if n else None
    expected = float(c.sum(0) @ c.sum(1)/n**2) if n else None
    kappa = float((observed-expected)/(1-expected)) if expected is not None and expected<1 else None
    return dict(n=n, strict_bad_ag3_bad=int(c[0,0]), strict_bad_ag3_good=int(c[0,1]),
                strict_good_ag3_bad=int(c[1,0]), strict_good_ag3_good=int(c[1,1]),
                agreement=observed, kappa=kappa)


def agreement():
    begin('T5')
    from sklearn.metrics import cohen_kappa_score, accuracy_score
    audit = WORK/'rf_quality_control/audit'
    inv = json.loads((audit/'file_inventory.json').read_text())
    cohort = json.loads((audit/'resolved_manifest.json').read_text())
    split = json.loads((WORK/'rf_quality_control/cache/metadata.json').read_text())['station_split']
    groups = defaultdict(list)
    for item in inv:
        if item['kind']=='waveform' and item['band']=='AG3':groups[item['station']].append(item)
    comparisons=[]; station_rows=[]; exclusions=[]; sources=[]
    for station in sorted(s['station'] for s in cohort if s['eligible']):
        files=groups[station]; good=sorted([r for r in files if r['label']=='good'],key=lambda r:r['rows'])
        bad=sorted([r for r in files if r['label']=='bad'],key=lambda r:r['rows'],reverse=True)
        if len(good)!=2 or len(bad)!=2:
            exclusions.append(dict(station=station,reason='two versions unavailable'));continue
        if good[0]['rows']+bad[0]['rows'] != good[1]['rows']+bad[1]['rows']:
            exclusions.append(dict(station=station,reason='version totals differ'));continue
        versions=[]
        for v in range(2):
            by_hash=defaultdict(list)
            for file in [bad[v],good[v]]:
                path=WORK/'data_yangruihao/Data_files'/file['path']; raw=path.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==file['sha256']
                arr=np.loadtxt(io.BytesIO(raw),ndmin=2,dtype='<f8')
                assert arr.shape==(file['rows'],504) and np.isfinite(arr).all()
                arr[arr==0]=0
                sources.append(dict(path=file['path'],sha256=file['sha256']))
                for i,wave in enumerate(arr):
                    key=hashlib.sha256(wave.tobytes()).hexdigest()
                    by_hash[key].append((int(file['label']=='good'),file['path'],i+1))
            versions.append(by_hash)
        shared=sorted(set(versions[0]) & set(versions[1])); c=np.zeros((2,2),int)
        matched=[]
        for key in shared:
            if len(versions[0][key])!=1 or len(versions[1][key])!=1:continue
            a,b=versions[0][key][0],versions[1][key][0];c[a[0],b[0]]+=1
            row=dict(station=station,split=split[station],waveform_sha256=key,
                     strict_label=a[0],ag3_label=b[0],strict_file=a[1],strict_row=a[2],
                     ag3_file=b[1],ag3_row=b[2]);comparisons.append(row);matched.append(row)
        report=dict(station=station,split=split[station],**label_summary(c),
            strict_total=good[0]['rows']+bad[0]['rows'],ag3_total=good[1]['rows']+bad[1]['rows'],
            strict_duplicate_rows=sum(len(v) for v in versions[0].values() if len(v)>1),
            ag3_duplicate_rows=sum(len(v) for v in versions[1].values() if len(v)>1),
            strict_unmatched_unique=sum(len(v)==1 and k not in versions[1] for k,v in versions[0].items()),
            ag3_unmatched_unique=sum(len(v)==1 and k not in versions[0] for k,v in versions[1].items()))
        if matched:
            a=[r['strict_label'] for r in matched];b=[r['ag3_label'] for r in matched]
            np.testing.assert_allclose(report['agreement'],accuracy_score(a,b),atol=1e-12)
            if report['kappa'] is not None:np.testing.assert_allclose(report['kappa'],cohen_kappa_score(a,b),atol=1e-12)
        station_rows.append(report)
        if len(station_rows)%25==0:print('T5 stations completed',len(station_rows),flush=True)
    aggregate=[]
    for group in ['all','train','val','test']:
        chosen=[r for r in comparisons if group=='all' or r['split']==group]
        a=np.array([r['strict_label'] for r in chosen]);b=np.array([r['ag3_label'] for r in chosen])
        aggregate.append(dict(split=group,stations=len(set(r['station'] for r in chosen)),
                              **label_summary(np.bincount(2*a+b,minlength=4))))
    table(OUT/'label_version_pairs.csv',comparisons);table(OUT/'label_agreement_stations.csv',station_rows)
    table(OUT/'label_agreement_summary.csv',aggregate)
    dump(OUT/'T5_verification.json',dict(eligible_stations=len(split),paired_stations=len(station_rows),
         excluded_stations=exclusions,matched_records=len(comparisons),
         median_station_agreement=float(np.median([r['agreement'] for r in station_rows if r['n']])),
         median_station_kappa=float(np.median([r['kappa'] for r in station_rows if r['kappa'] is not None])),
         independent_station_checks=len(station_rows),sources=sources,
         registration_sha256=sha(OUT/'registration.json')))
    print(json.dumps(aggregate,indent=2))


def logistic():
    begin('T6')
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score
    from threadpoolctl import threadpool_limits
    output=OUT/'logistic';output.mkdir()
    summary=[];all_scores=[];run_names=[];station_reports=[]
    for variant,cache_name in [('ag3','cache_ag3_v1'),('multiband','cache')]:
        cache=WORK/'rf_quality_control'/cache_name
        arrays={n:np.load(cache/f'{n}.npy',mmap_mode='r') for n in
                ['features','labels','lengths','gaussians','sample_ids','stations','train_indices','val_indices','test_indices']}
        prep_path=PAPER/f'SourceData/benchmark_v1/features_only_{variant}/20260928/preprocessing.json'
        scaler=json.loads(prep_path.read_text())
        x=np.asarray(arrays['features'],dtype=float)
        x=np.sign(x)*np.log1p(np.abs(x))
        valid=np.arange(x.shape[1])[None,:] < arrays['lengths'][:,None]
        train=arrays['train_indices'];val=arrays['val_indices'];test=arrays['test_indices'];y=arrays['labels']
        # Independently confirm archived scaling uses only valid training observations.
        feature_train=x[train][valid[train]]
        np.testing.assert_allclose(feature_train.mean(0),scaler['feature_mean'],atol=1e-10)
        np.testing.assert_allclose(feature_train.std(0),scaler['feature_std'],atol=1e-10)
        x=(x-np.array(scaler['feature_mean']))/np.array(scaler['feature_std'])
        x[~valid]=0
        if variant=='multiband':
            ordered=np.zeros_like(x)
            for slot,g in enumerate([1.,1.5,2.,2.5,3.,4.,5.]):
                ii,jj=np.where((arrays['gaussians']==g)&valid);ordered[ii,slot]=x[ii,jj]
            x=ordered
        x=np.ascontiguousarray(x.reshape(len(x),-1))
        saved=[]
        for seed in SEEDS:
            model=LogisticRegression(C=1.0,class_weight='balanced',random_state=seed)
            with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                warnings.simplefilter('always');model.fit(x[train],y[train])
                vp=model.predict_proba(x[val])[:,1]
                grid=np.linspace(.01,.99,99)
                threshold=float(grid[np.argmax([f1_score(y[val],vp>=t,average='macro') for t in grid])])
                p=model.predict_proba(x[test])[:,1]
            run=output/f'{variant}_seed{seed}';run.mkdir()
            m=metrics(y[test],p,threshold)
            np.testing.assert_allclose([m['accuracy'],m['macro_f1'],m['good_auprc'],m['auroc']],
                [accuracy_score(y[test],p>=threshold),f1_score(y[test],p>=threshold,average='macro'),
                 average_precision_score(y[test],p),roc_auc_score(y[test],p)],atol=1e-12)
            dump(run/'config.json',dict(variant=variant,seed=seed,sklearn_version=sklearn.__version__,
                 parameters=model.get_params(),input_columns=x.shape[1],iterations=model.n_iter_.tolist(),
                 warnings=[str(w.message) for w in caught],training_n=len(train),validation_n=len(val),test_n=len(test),
                 preprocessing_sha256=sha(prep_path),registration_sha256=sha(OUT/'registration.json')))
            dump(run/'metrics.json',dict(validation=metrics(y[val],vp,threshold),test=m))
            dump(run/'model.json',dict(coef=model.coef_.tolist(),intercept=model.intercept_.tolist(),classes=model.classes_.tolist()))
            pred=[dict(sample_id=str(arrays['sample_ids'][j]),station=str(arrays['stations'][j]),label=int(y[j]),
                       p_good=float(score),prediction=int(score>=threshold)) for j,score in zip(test,p)]
            table(run/'test_predictions.csv',pred)
            for s in sorted(set(arrays['stations'][test])):
                mask=arrays['stations'][test]==s;sm=metrics(y[test][mask],p[mask],threshold)
                station_reports.append(dict(variant=variant,seed=seed,station=str(s),**sm))
            dump(run/'completion.json',dict(sha256={f.name:sha(f) for f in run.iterdir() if f.is_file()}))
            summary.append(dict(variant=variant,seed=seed,input_columns=x.shape[1],iterations=int(model.n_iter_[0]),warnings=len(caught),**m))
            all_scores.append(p);run_names.append(f'{variant}/{seed}');saved.append(p)
        identical=all(np.array_equal(saved[0],v) for v in saved[1:])
        print('T6',variant,'identical seed predictions',identical,flush=True)
    table(OUT/'logistic_metrics.csv',summary);table(OUT/'logistic_station_metrics.csv',station_reports)
    np.savez_compressed(OUT/'logistic_predictions.npz',run_names=run_names,probabilities=np.stack(all_scores),
                        sample_ids=arrays['sample_ids'][test],labels=y[test],stations=arrays['stations'][test])
    dump(OUT/'T6_verification.json',dict(fits=6,source_cache_files={name:{f:sha(WORK/'rf_quality_control'/name/(f+'.npy'))
          for f in ['features','labels','gaussians','lengths','sample_ids','stations','train_indices','val_indices','test_indices']}
          for name in ['cache','cache_ag3_v1']},registration_sha256=sha(OUT/'registration.json'),
          independent_metric_checks=24,sklearn_version=sklearn.__version__,original_models_modified=False))
    print(json.dumps([{k:r[k] for k in ['variant','seed','accuracy','macro_f1','good_auprc','iterations','warnings']} for r in summary],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['T4','T5','T6'])
    parser.add_argument('--output', type=Path,
                        help='Separate replay directory; registered outputs are never overwritten.')
    args=parser.parse_args()
    if args.output is not None:
        destination=args.output.expanduser().resolve()
        if destination != OUT.resolve():
            if not destination.exists():
                destination.mkdir(parents=True)
                for name in ['baseline_snapshot.json','registration.json','registration.tex','registration_receipt.json']:
                    shutil.copy2(OUT/name,destination/name)
            assert sha(destination/'registration.json')==sha(OUT/'registration.json')
            OUT=destination
    {'T4':bootstrap,'T5':agreement,'T6':logistic}[args.task]()
