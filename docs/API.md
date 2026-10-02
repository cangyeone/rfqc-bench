# Python API

## Discovery and networks

```python
from rfqc_bench import list_models, available_weights, create_model

print(list_models())       # 19 configurations, input regime, publication DOI
print(available_weights()) # 53 versioned bundles with SHA256 hashes
net = create_model("gong_cnn")
```

`create_model(name)` is a low-level **randomly initialized** PyTorch network,
not a trained classifier. Neural forward arguments are `(waveforms, features,
gaussians, lengths)` after registered preprocessing; output is one good-vs-bad
logit per record. Use a predictor for normal inference. Statistical models use
`fit()` or saved bundles instead of `create_model()`.

## Trained inference

```python
from rfqc_bench import RFData, RFQCPredictor

p = RFQCPredictor.from_pretrained("reference_multifilter", seed=20260928,
                                device="cpu", cache_dir="./model-cache")
data = RFData.load("my_rf.npz")
r = p.predict(data, batch_size=32)
r.to_csv("predictions.csv", data.sample_ids)
print(r.to_dict())
```

`RFQCPredictor.from_directory(path, device='cpu')` loads an offline bundle.
`predict()` also accepts keyword arrays: `waveforms`, `features`, `gaussians`,
`lengths`, `stations`. Do not mix an RFData argument and array keywords.
Outputs contain `p_good`, `prediction`, `threshold`, `method`, `seed`.
Scores are not assumed calibrated across regions or label systems. The saved
validation threshold is applied without inspecting labels in a prediction input.

## Fitting and independent evaluation

```python
from rfqc_bench import fit, RFData, evaluate

p = fit("logreg_ag3", RFData.load("train.npz"),
        RFData.load("validation.npz"), output="runs/logreg", seed=20260928)
test = RFData.load("test.npz")
result = p.predict(test)
print(evaluate(test.labels, result.p_good, result.threshold))
```

`evaluate(labels, p_good, threshold=.5)` returns accuracy, macro-F1 (fixed bad/good
classes), balanced accuracy, good precision/recall/AP, bad recall and AUROC.
Undefined AUROC (one observed class) and AP (no good records) are `None`.
`station_bootstrap(labels, first, second, stations, thresholds=(.5,.5),
draws=10000, seed=20260930)` resamples entire stations, computes paired pooled
macro-F1 differences and returns median/percentile intervals in percentage points.
Supply each model's validation-selected threshold; these intervals do not measure
training-seed uncertainty. `label_agreement(a,b)` requires already aligned binary
labels; it performs no approximate event matching.

## RF diagnostics

```python
from rfqc_bench.diagnostics import stack_comparison, cross_filter_association

ag3 = test.select(3.0)
summary = stack_comparison(ag3.waveforms[:, 0], test.labels == 1,
                           result.prediction == 1)
association = cross_filter_association(test)
```

Stack diagnostics use same-pool arithmetic means, raw and per-record
peak-normalized; report correlation, NRMSE, operational P/Ps-candidate extrema
and timing differences. Cross-filter association returns one finite-pair mean
in 2.5..15 s per record, or `None` if undefined. These are descriptive waveform
diagnostics, not structural inversion or independent physical ground truth.

## Inference timing (v0.1.1)

```python
from rfqc_bench import benchmark_inference
timing = benchmark_inference(model, data, sample_size=512, complete_views=True)
```

This runs one synchronized timing round with a fixed loaded predictor. It returns
raw single/batch durations, latency quantiles, API and network throughput, memory
and environment metadata, without observational arrays or predictions. It does
not fit weights or thresholds. FCM retains the full station pool and returns only
pool timing. For repeated fresh-process rounds and full definitions, see
[BENCHMARK_zh.md](BENCHMARK_zh.md). Timings generated on new hardware are new
measurements and do not replace the published snapshot.
