# Release validation

## v0.1.1: comparison snapshot and inference timing

`comparison_release_20261002.json` records the 34 passing synthetic tests, public
snapshot verification and installed-wheel checks. The wheel was imported outside
the checkout using dependencies from the existing Python environment; this is
not a clean dependency-resolution test. CLI timing, six fresh-process rounds
across neural/LogReg/FCM implementations, hash-verified resume and a three-model
accuracy command passed on runtime-generated synthetic data. Those scores are
installation checks and are not published scientific results.

The public reproduction command verifies 92 source-file hashes, aggregate
comparisons and all 57 original timing records. The 53 model bundles, network
implementations, preprocessing and prediction behavior remain unchanged. The
portable v0.1.1 timing runner was smoke-tested on CPU; the manuscript's recorded
RTX 5090 timings were obtained with the original v0.1.0 implementation.

No observational arrays, individual label vectors, sample/station lists or
per-record predictions are added. See the snapshot manifest for the explicit
projection of timing metadata and the limits of aggregate-only reproduction.

## v0.1.0: model export and prediction APIs

This folder contains aggregate software checks, not RF waveforms, labels,
station/sample metadata, or per-record predictions. Checks were performed on
2026-09-30 with Python 3.14.4 and PyTorch 2.11.0 on macOS/CPU.

- `model_export.json`: 53 completed fits passed their source completion hash
  checks. Exported JSON/safetensors hashes are recorded. The 17 main
  configurations have three seeds each; reference AG1 and AG5 have one each.
- `archive_parity.json`: all 53 bundles were checked on 128 evenly spaced
  archived test positions per model, except FCM, which used the full 24,533-record
  test pool to retain same-station context. All 44 neural models produced
  identical scores in the packaged and original code on the same CPU (maximum
  absolute error 0). Statistical model replay passed the 1e-10 bound.
- **Cross-platform drift is retained:** 42 of 53 checks exceeded the initial
  CUDA-to-CPU diagnostic tolerance of 1e-4. The largest absolute probability
  difference was 0.004481911659240723. No thresholded decisions changed in the
  checked records; this does not establish identical decisions for all neural
  test records. The packaging comparison uses the original CPU implementation
  to distinguish packaging errors from platform differences. Archived CUDA
  predictions remain the source of manuscript metrics.
- `http_smoke.json`: an actual local HTTP server loaded an exported Gong CNN
  bundle and passed health, OpenAPI and synthetic two-record prediction calls.
- `source_provenance.json`: hashes and origins of model source files and a
  summary of the packaging adaptations.

The 25 synthetic pytest cases cover all neural configurations, input validation,
missing-band masking, statistical fitting, HTTP requests, bundle integrity and
interrupted/resumed CPU training. No claim of bitwise CUDA/MPS training recovery
is made. Tests require no private RF data or downloaded trained weights.

Both wheel and source distribution are built using `python -m build` and checked
using `python -m twine check dist/*`. The wheel was installed through pip into a
separate virtual environment (with system site packages available), dependencies
were resolved by pip, and import/prediction were exercised outside the source
checkout. GitHub Actions additionally runs the synthetic test/build workflow on
Linux with Python 3.10 and 3.12; its actual run status is available on GitHub.

Run `scripts/verify_archive_parity.py --workspace /path/to/original/workspace
--bundles /path/to/exported/bundles` to repeat the optional archive comparison.
That script needs the original local benchmark archive; it is not required for
installation, prediction, training, or the public synthetic test suite.
