# Training and manual recovery

`fit(method, train, validation, output, seed=20260928, epochs=50, patience=10,
batch_size=32, device='cpu', resume=False, checkpoint_seconds=60)` supports all
19 configurations. Use explicit user-owned train and validation inputs with
both classes. Provided station IDs and sample IDs must be disjoint between
the two inputs. An omitted station vector cannot establish station independence.

Neural defaults match the benchmark: random initialization; inverse-class-count
sampling with replacement; original training sample count per epoch; BCE logits;
AdamW with weight decay 1e-4; gradient-norm clipping at 5; learning rate 3e-4
(Chen: 1e-5); maximum 50 epochs; validation-good-AP patience 10. Thresholds use
the earliest maximum validation macro-F1 on 0.01,0.02,...,0.99. Preprocessing is
fitted on train only. Checkpoints are selected on validation AP, not test scores.

```bash
rfqc-bench train --model gong_cnn_bilstm --train train.npz --validation validation.npz --output runs/gong --device cuda:0
rfqc-bench train --model gong_cnn_bilstm --train train.npz --validation validation.npz --output runs/gong --device cuda:0 --resume
```

Neural training saves at epoch boundaries, approximately every 60 seconds at a
safe completed-batch boundary, and on handled Ctrl+C/SIGTERM. Checkpoints retain
weights, best weights, optimizer, RNG states, sampled order, next batch position
and early-stopping state. A hard power loss can discard work since the latest
checkpoint. Resume checks the input fingerprint and training configuration;
use the same command and output directory. No scheduler or startup service runs.
State recovery does not promise bitwise CUDA/MPS update trajectories.

Outputs: `bundle.json`, optional `weights.safetensors`, `validation_metrics.json`,
`run_config.json`, `status.json` and recovery checkpoints. Serve or predict from
the completed directory without separately loading scalers or choosing thresholds.

LogReg uses L2, C=1, balanced class weights, lbfgs, tolerance 1e-4 and 100
iterations. Neural `epochs`/`patience` do not override those defaults. The original
multi-filter control reaches this iteration limit; convergence warnings are
stored rather than suppressed. Its three seed settings produce identical
solutions in the recorded environment. A interrupted logistic fit is restarted
on manual resume; there is no claimed within-lbfgs checkpoint.

Xiong-FCM computes four features using unlabeled same-station context, excludes
training outliers with |z|>=3, fits training-only min/max scaling and two fuzzy
centers (m=2, maximum 300 iterations, tolerance 1e-5), orients clusters with train
labels, then selects a validation threshold. All validation/test records remain.
FCM checkpoints centers every ten iterations. This is not an independent-record
method; supply complete station pools, including during evaluation.

This portable trainer reproduces the documented recipe, not the original GPU
training trajectory across all software/hardware. The released weights retain
the completed benchmark outcomes. Training on new inputs creates new results;
it does not replace the published frozen experiments. Runtime/framework versions
and any protocol changes must be reported when comparing newly fitted models.
