# RFQC Bench

Receiver-function quality control through one Python API, CLI and optional HTTP
service. This repository contains the **benchmark adaptations** of nine
literature configurations from eight studies, six matched reference/descriptor
configurations, two logistic controls, and the AG1/AG5 sensitivity controls:
**19 configurations and 53 RF-trained weight bundles**.

**No observational waveforms, labels, dataset caches, event metadata or
per-record predictions are distributed here.** Examples generate synthetic
signals at runtime. Installation downloads software dependencies only; trained
weights are downloaded explicitly on first `from_pretrained()` use and cached.

Cross-platform scores are not bitwise identical. The packaging check compares
the original and packaged code on the same CPU; CUDA-to-CPU numerical differences
are retained separately in `validation/archive_parity.json`. Use the frozen
archived predictions for the manuscript's reported scores.

[中文入门](docs/QUICKSTART_zh.md) · [EQR 目录自动筛选](docs/EQR_SCREENING_zh.md) · [Python API](docs/API.md) ·
[Input format](docs/DATA.md) · [Training](docs/TRAINING.md) ·
[HTTP deployment](docs/DEPLOYMENT.md) · [Methods](docs/METHODS.md) ·
[Comparison tables](benchmarks/2026-10-02/generated/COMPARISON.md) · [测速与复现](docs/BENCHMARK_zh.md)

## Install with pip

Python 3.10 or newer. Use a virtual environment. The versioned GitHub wheel does
not require Git:

```bash
python -m pip install https://github.com/cangyeone/rfqc-bench/releases/download/v0.1.2/rfqc_bench-0.1.2-py3-none-any.whl
rfqc-bench doctor
```

Alternatively, install the tagged source (requires Git), with HTTP support:

```bash
python -m pip install "rfqc-bench[api] @ git+https://github.com/cangyeone/rfqc-bench.git@v0.1.2"
```

For development: `python -m pip install -e '.[dev]'`. This release is installable
with pip from GitHub. A plain `pip install rfqc-bench` from PyPI is **not claimed**
until the maintainer publishes there; the [publication workflow](docs/RELEASING.md)
is provided. CUDA users can install their appropriate PyTorch build first; CPU
is the default. `--device cuda:0`, `cuda:1` or `mps` selects an available device.

## Python: trained predictions in a few lines

```python
from rfqc_bench import RFData, RFQCPredictor

data = RFData.load("my_rf.npz")                 # your own records
model = RFQCPredictor.from_pretrained("reference_multifilter")
result = model.predict(data)
print(result.p_good, result.prediction)        # good=1; bad=0
result.to_csv("predictions.csv", data.sample_ids)
```

## SAC EQR directory to a retained-file record

```bash
rfqc-bench screen-eqr "/path/to/all_eqr"
```

This recursively reads binary SAC `.eqr` files, matches identical event filenames
within each station's `AG1`, `AG3`, etc. directories, and writes `all_eqr/record`:
**one retained relative file path per line, without a header**. Original files
are unchanged. The default Reference multi-filter model accepts missing views;
single-filter models list only the view they use. A `bad/` directory is not a label.
Use `--gaussian 3` for flat directories without an AG name, `--model` to select an
existing waveform model, or `--model-dir` to use a completed local training bundle.
Inputs must already be P-referenced radial RFs, sampled at 0.1 s and covering
the observed −10 to 40 s window. No resampling or zero-padding is hidden here.

```python
from rfqc_bench import screen_eqr
report = screen_eqr("/path/to/all_eqr", output="results/record")
print(report["retained_files"], report["outputs"]["record"])
```

Prediction and rejection CSVs plus a JSON provenance report accompany `record`.
Existing outputs are protected; choose another output or explicitly use
`--overwrite`. A preloaded `RFQCPredictor` also exposes `.screen_eqr()`.
For the server-side directory API, configure `serve --eqr-root /srv/rfqc` and
call `POST /screen-eqr` with paths relative to that root. The endpoint is disabled
by default. [Complete directory, CLI, Python and HTTP guide](docs/EQR_SCREENING_zh.md).

The model catalog remains pinned to the existing v0.1.0 weights; this interface
release does not replace them with results from ongoing experiments.

The array-based `.predict()` API supports `gong_cnn_bilstm`, `deeprfqc`,
`xiong2025_fcm`, `logreg_ag3`, or any name returned by `list_models()`.
Single-filter models select their required
Gaussian view from a multi-filter input, and reject records missing that view.
Six supplied descriptors are required only for descriptor/combined/LogReg models.
FCM also requires the **complete same-station input pool** and station IDs.

The default weight is the **first registered seed**, never the best test seed.
Choose another registered seed with `seed=20260929`. All original three seeds
are available for the 17 main configurations; AG1/AG5 each use their one
registered seed, 20260929. These are RF models trained from scratch on the
benchmark cohort, not the source authors' pretrained weights or phase-task
transfer weights. Thresholds and normalization parameters travel with each model.

## CLI and installation check

```bash
rfqc-bench list
rfqc-bench demo --model gong_cnn --output synthetic_predictions.csv
rfqc-bench predict --model reference_multifilter --input my_rf.npz --output predictions.csv
rfqc-bench download --model deeprfqc --seed 20260928
```

The demo is an installation test, **not an accuracy experiment**. For offline
use, pre-download a bundle, then pass `--model-dir /path/to/bundle` or use
`RFQCPredictor.from_directory()`. Set `RFQC_CACHE` to change the download cache.
The Chen model is about 220 MiB before compression; other neural models are
roughly 0.05–6 MiB. The package never downloads all models automatically.

## HTTP API

```bash
rfqc-bench serve --model gong_cnn --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

Open `http://127.0.0.1:8000/docs` for interactive request documentation.
`POST /predict` accepts waveform arrays, Gaussian coefficients and optional
descriptors/lengths/stations. It returns probabilities, decisions, model, seed
and the frozen threshold. See [deployment and curl examples](docs/DEPLOYMENT.md).

## Completed comparisons and inference speed (v0.1.1)

[View all comparison tables](benchmarks/2026-10-02/generated/COMPARISON.md): original
three-seed classification, eight-seed paired contrasts, label-protocol controls,
RF waveform diagnostics, single-filter/multi-filter sensitivity, and 19-model
inference cost. Aggregate tables and raw timing durations are included; no
observational records or per-record predictions are uploaded. The measured
paper timings used **v0.1.0**, whose 53 model bundles remain unchanged.

```bash
# In a clone of this repository; no data/GPU/model downloads needed:
python scripts/reproduce_comparisons.py
# On your own data, one timing round with fixed weights:
rfqc-bench benchmark --model gong_cnn --seed 20260929 --input my_test.npz --output timing.json --device cuda:0 --complete-views
# All 19 configurations, three fresh-process timing rounds each:
python scripts/benchmark_models.py --input my_test.npz --output outputs/timing --device cuda:0
```

For this recorded RTX 5090 / FP32 / batch-32 API workload, Gong-CNN reaches
13,789 RF/s versus 1,079 for Gong-CNN–BiLSTM. The matched multi-filter reference
reaches 5,128 RF/s versus 7,421 for AG3. These include preprocessing/transfers,
not model loading or upstream RF generation. FCM is separately measured on a
complete station pool. The eight-seed multi-filter accuracy contrast spans zero;
read [the report](benchmarks/2026-10-02/generated/COMPARISON.md) for all results,
negative findings and limitations. See [中文操作说明](docs/BENCHMARK_zh.md) for
own-data evaluation, offline use, timing scope and manual recovery.

## Train on your own data

```bash
rfqc-bench train --model reference_multifilter --train train.npz --validation validation.npz --output runs/reference --device cuda:0
# After interruption, repeat the same command with --resume.
```

Neural models use the registered class balancing, AdamW, validation-AP early
stopping and validation macro-F1 threshold grid. Training saves optimizer/RNG
state and sampled batch position; no startup service is installed. `fit()`
accepts train and validation inputs only. Test data enter `evaluate()` separately.
See [training scope and reproducibility limits](docs/TRAINING.md).

## Scientific scope

Inputs are **501 samples, −10 to 40 s relative to P, dt=0.1 s**. AG values
1, 1.5, 2, 2.5, 3, 4, 5 are Gaussian coefficients, not hertz. Different input
time grids are rejected rather than silently reinterpreted. Variable-length
reference inputs omit missing views. RF scores measure agreement with a
screening label system; they do not establish improved H–κ or inversion results.

Published architecture/code discrepancies and all benchmark adaptations are
listed in [METHODS](docs/METHODS.md). The multi-filter logistic weights reached
the registered 100-iteration limit; their convergence warnings are retained.
Descriptor definitions and source labels are distinct from Xiong's four computed
features. The API never fabricates the six provider-supplied descriptors.

## Tests and release provenance

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
python -m twine check dist/*
```

Tests use synthetic inputs and cover all neural architectures, input validation,
missing-filter masking, statistical fitting, HTTP calls and interrupted/resumed
CPU training. Release validation additionally compares exported model predictions
with the frozen experiment archive on local data; only aggregate verification
receipts are published in `validation/`. Weight archives have outer SHA256 checks
and inner file hashes, and use safetensors or JSON rather than arbitrary pickles.

Code is distributed under GPL-3.0-only, with applicable third-party notices and
licenses retained. This software license does not license a dataset. Please cite
the original studies relevant to the models you use and the benchmark manuscript;
see [CITATION.cff](CITATION.cff) and [third-party attribution](THIRD_PARTY_NOTICES.md).
