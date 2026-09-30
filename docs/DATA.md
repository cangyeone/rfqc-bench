# Input contract

All arrays are **user supplied**. `RFData.load()` reads an NPZ (`allow_pickle=False`)
or a directory of named NPY files. It never infers cross-filter event alignment.
Preserve the same record ordering in waveforms, features and metadata.

| Field | Shape | Meaning |
|---|---|---|
| `waveforms` | `(N,K,501)` or `(N,501)` for one view | Raw exported RF amplitudes, not pre-normalized |
| `gaussians` | `(K,)` or `(N,K)` | Gaussian coefficients; default 3 only when K=1 |
| `lengths` | `(N,)` integers | Number of valid leading views; defaults to K |
| `features` | `(N,K,6)` or `(N,6)` | Raw supplied descriptors, required by feature/combined/LogReg models |
| `stations` | `(N,)` strings | Context groups, required for FCM; optional for other inference |
| `sample_ids` | `(N,)` strings | Optional caller identifiers for output; not model inputs |
| `labels` | `(N,)` binary | Required for training/evaluation only; bad=0, good=1 |
| `start_time` | scalar | −10.0 seconds relative to P |
| `sampling_interval` | scalar | 0.1 seconds |

Valid filter coefficients are unique, ascending members of
`[1,1.5,2,2.5,3,4,5]`. Missing views are omitted within each record; pad to a
common K and pass `lengths` if needed. Each multi-view row must refer to the same
event. Padding is ignored, not an observed zero RF. An AG5 classifier rejects a
record lacking AG5; it does not fabricate one.

Six descriptor columns, in order:

1. MaxAmp: supplied maximum amplitude.
2. MaxTime: time of that maximum relative to P.
3. NegMinAmp: absolute negative minimum in −10..5 s.
4. NegPosRatio: NegMinAmp/MaxAmp.
5. PPsRatio: MaxAmp / supplied Ps-window extremum, 2.5..7.5 s.
6. PsPeaks: supplied extremum count in that window.

The provider's precise extraction conventions are not fully documented.
Supply the existing measurements; do not substitute `diagnostics.stack_features`
as an exact regeneration. Xiong's four computed features are a separate definition.

## Save your own batch

```python
import numpy as np

np.savez_compressed("my_rf.npz", waveforms=raw_waveforms,
                    gaussians=np.array([1.,1.5,2.,2.5,3.,4.,5.]),
                    lengths=valid_view_counts,
                    start_time=-10., sampling_interval=.1)
```

The model applies its saved transformations exactly once. One-dimensional neural
models use asinh(raw/training median peak); Chen uses per-trace absolute-peak
normalization followed by its RGB adapter. Neural descriptors use signed-log and
training-only standardization. LogReg uses the registered float64 equivalent;
multi-filter LogReg puts standardized features into a fixed 42-column order,
with absent filters filled by the standardized training mean (zero).

Station strings are never categorical predictors. For Xiong-FCM, context is
computed from the whole station pool supplied in the current call; `batch_size`
does not subdivide that calculation. Streaming that pool in independent API
requests changes the method's input and can change scores. No earthquake IDs,
ray parameters, back azimuths or magnitudes are invented by the package.
