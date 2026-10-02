# Completed RFQC comparisons — 2026-10-02

Source manuscript commit: `b88a6c13f6ae378d9b276bd19dcb4789af3c446d`. These are documented adaptations under a fixed task, not each source paper’s optimal pipeline.

## Original three-seed classification and measured deployment cost

The ten waveform/image configurations below share 24,533 held-out RFs. Scores are mean ± sample SD across three training seeds. Timing uses fixed seed 20260929, one RTX 5090, FP32 eager execution and three measurement rounds. RF/s includes API preprocessing and transfers at batch 32. Bold identifies the highest original mean in each score column, not significance. Additional views distinguish the multi-filter reference.

| Configuration | Accuracy (%) | Macro-F1 (%) | Good AP (%) | p50 (ms) | API RF/s |
| --- | --- | --- | --- | --- | --- |
| Li-CNN (2021) | 95.61 ± 0.15 | 88.26 ± 0.15 | 86.67 ± 0.25 | 1.06 | 18,967 |
| Gan-CNN (2021) | 95.60 ± 0.06 | 87.99 ± 0.14 | 85.98 ± 0.23 | 1.10 | 18,377 |
| Gong-CNN | 96.11 ± 0.12 | 89.21 ± 0.24 | **88.66 ± 0.21** | 1.63 | 13,789 |
| Gong-CNN–BiLSTM | **96.15 ± 0.09** | **89.27 ± 0.28** | 88.65 ± 0.26 | 26.00 | 1,079 |
| Gan-CNN (2023) | 95.53 ± 0.09 | 87.77 ± 0.28 | 85.64 ± 0.36 | 1.25 | 16,948 |
| DeepRFQC (2024) | 95.94 ± 0.10 | 88.60 ± 0.16 | 87.59 ± 0.69 | 2.22 | 10,696 |
| RF-Capsule (2025) | 94.81 ± 0.16 | 86.12 ± 0.11 | 81.25 ± 0.28 | 1.81 | 13,018 |
| Chen-AlexNet (2026) | 96.03 ± 0.01 | 89.12 ± 0.19 | 88.11 ± 0.23 | 2.13 | 5,977 |
| Reference–AG3 | 95.85 ± 0.18 | 88.56 ± 0.28 | 87.58 ± 0.39 | 3.55 | 7,421 |
| Reference–multi-filter | 96.08 ± 0.10 | 89.19 ± 0.34 | 88.33 ± 0.37 | 3.63 | 5,128 |

## Descriptor and station-context controls

These input regimes are separate from waveform-only prediction. FCM uses a complete station pool on CPU; its rate is not independent-record batch throughput. LogReg runs on CPU. Supplied-descriptor extraction is excluded from timing.

| Configuration | Accuracy (%) | Macro-F1 (%) | Good AP (%) | Device | RF/s |
| --- | --- | --- | --- | --- | --- |
| Descriptors only / AG3 | 90.01 ± 0.10 | 77.18 ± 0.12 | 59.46 ± 0.30 | cuda:0 | 17,681 |
| Waveform + descriptors / AG3 | 95.79 ± 0.18 | 88.32 ± 0.48 | 87.15 ± 0.32 | cuda:0 | 7,198 |
| Descriptors only / multi-band | 94.18 ± 0.25 | 84.71 ± 0.30 | 76.24 ± 1.06 | cuda:0 | 18,506 |
| Waveform + descriptors / multi-band | 96.00 ± 0.16 | 88.87 ± 0.24 | 88.42 ± 0.15 | cuda:0 | 4,834 |
| Xiong-FCM (2025) | 85.97 ± 0.00 | 73.01 ± 0.00 | 38.29 ± 0.00 | cpu | 30,852 (full pool) |
| LogReg–AG3 | 81.27 ± 0.00 | 64.60 ± 0.00 | 26.32 ± 0.00 | cpu | 90,860 |
| LogReg–multi-filter | 89.63 ± 0.00 | 76.75 ± 0.00 | 45.95 ± 0.00 | cpu | 292,621 |

## Appended eight-seed results

The original three-seed table above remains unchanged. Five additional fits for each of four configurations provide eight seeds. Other methods retain three fits; no nonexistent extra seeds are inferred.

| Configuration | Macro-F1 (%) |
| --- | --- |
| Reference–AG3 | 88.143 ± 0.448 |
| Reference–multi-filter | 88.590 ± 0.688 |
| Gong-CNN | 89.248 ± 0.195 |
| Gong-CNN–BiLSTM | 89.260 ± 0.222 |

| Paired contrast | Mean difference (pp) | SD (pp) | 95% interval (pp) |
| --- | --- | --- | --- |
| Reference–multi-filter − Reference–AG3 | +0.447 | 0.615 | [-0.067, +0.961] |
| Gong-CNN − Reference–AG3 | +1.106 | 0.566 | [+0.633, +1.579] |
| Gong-CNN–BiLSTM − Reference–AG3 | +1.117 | 0.474 | [+0.721, +1.514] |
| Gong-CNN − Reference–multi-filter | +0.659 | 0.758 | [+0.025, +1.293] |
| Gong-CNN–BiLSTM − Reference–multi-filter | +0.670 | 0.790 | [+0.010, +1.331] |
| Gong-CNN–BiLSTM − Gong-CNN | +0.012 | 0.341 | [-0.274, +0.297] |

Intervals are paired Student-t intervals, unadjusted for multiple comparisons and conditional on the fixed split. The multi-filter-minus-AG3 and BiLSTM-minus-CNN intervals cross zero; both Gong-minus-multi-filter intervals lie just above zero. This is neither global equivalence nor a general multi-filter advantage.

## Label protocols and perturbation controls

| Source → target | Source F1 (%) | Target F1 (%) | Target AP (%) | F1 change (pp) |
| --- | --- | --- | --- | --- |
| strict to ag3 | 88.139 ± 0.291 | 87.030 ± 0.494 | 86.444 ± 1.197 | -1.109 ± 0.236 |
| ag3 to strict | 87.286 ± 0.440 | 86.920 ± 0.611 | 84.648 ± 1.639 | -0.366 ± 0.171 |

Both directions use the same 18,305 paired test records at 22 stations. Achieved scores are not a theoretical label-system ceiling and cannot be pooled with the original test cohort. Labels agree on 98.04% of 116,722 paired records, with 2,287 strict-bad/AG3-good changes and zero reverse changes.

Noise and synthetic-label perturbation curves are in `results/input_perturbation.csv` and `results/label_perturbation.csv`. Smallest detected effects are specific to the tested grid, not universal resolution limits. Their station-bootstrap intervals require record-level inputs to recompute; those data are not distributed here.

## RF waveform diagnostics

| Selection | Retained RFs | Median shape r | Median raw NRMSE | P amplitude error (%) | Ps-candidate error (%) | Cross-filter r |
| --- | --- | --- | --- | --- | --- | --- |
| All RFs | 24533 | 0.9204 | 5.714e+15 | 2.48e+16 | 4.66e+17 | 0.8317 |
| Manual good | 2392 | 1.0000 | 0 | 0 | 0 | 0.8414 |
| Li 2021 | 2589 | 0.9917 | 0.06784 | 2.78 | 3.88 | 0.8378 |
| Gan 2021 | 2604 | 0.9931 | 0.06405 | 2.1 | 2.66 | 0.8363 |
| Gong CNN | 2421 | 0.9933 | 0.05644 | 1.52 | 3.6 | 0.8386 |
| Gong CNN-BiLSTM | 2493 | 0.9942 | 0.057 | 1.59 | 2.85 | 0.8396 |
| Gan 2023 | 2621 | 0.9921 | 0.06432 | 1.81 | 2.63 | 0.8363 |
| DeepRFQC | 2528 | 0.9936 | 0.07354 | 1.98 | 3.7 | 0.8377 |
| RF-Capsule | 2607 | 0.9917 | 0.06688 | 3.21 | 3.01 | 0.8354 |
| Chen 2026 | 2494 | 0.9930 | 0.06991 | 1.63 | 3.93 | 0.8346 |
| Ours AG3 | 2544 | 0.9934 | 0.05722 | 2.1 | 4.06 | 0.8398 |
| Ours multi-band | 2484 | 0.9918 | 0.05426 | 2.05 | 3.93 | 0.8403 |
| Xiong FCM | 5145 | 0.9787 | 0.2401 | 7.1 | 29.3 | 0.8346 |

These fixed-seed diagnostics summarize station medians and shared-record selections. Ps is an operational waveform candidate, not an independently identified Moho phase. Similar stacks do not establish improved H–κ or structural inversion; no such inversion was performed. Half/quarter-pool summaries are provided separately.

## Single-filter sensitivity on the common-availability cohort

| Input | Accuracy (%) | Macro-F1 (%) | Good AP (%) |
| --- | --- | --- | --- |
| ag1 | 95.75 | 87.86 | 86.05 |
| ag3 | 95.75 | 87.98 | 86.65 |
| ag5 | 95.55 | 87.01 | 84.86 |
| multi-filter | 96.19 | 89.07 | 88.19 |

This is one fixed seed on 24,107 common test records. It is not a three- or eight-seed result and does not choose a universally optimal Gaussian coefficient. AG denotes a Gaussian coefficient, not hertz.

## Full inference comparison

| Configuration | Device | p50 / p95 (ms) | API RF/s | Round range (RF/s) |
| --- | --- | --- | --- | --- |
| Li-CNN (2021) | cuda:0 | 1.062 / 1.171 | 18,967 | [18,810, 18,968] |
| Gan-CNN (2021) | cuda:0 | 1.103 / 1.181 | 18,377 | [18,040, 18,380] |
| Gong-CNN (2022) | cuda:0 | 1.626 / 1.710 | 13,789 | [13,543, 13,935] |
| Gong-CNN–BiLSTM (2022) | cuda:0 | 25.999 / 26.911 | 1,079 | [1,068, 1,082] |
| Gan-CNN (2023) | cuda:0 | 1.250 / 1.374 | 16,948 | [16,894, 16,955] |
| DeepRFQC (2024) | cuda:0 | 2.216 / 2.354 | 10,696 | [10,278, 10,732] |
| RF-Capsule (2025) | cuda:0 | 1.809 / 1.948 | 13,018 | [11,581, 13,087] |
| Chen-AlexNet (2026) | cuda:0 | 2.127 / 2.690 | 5,977 | [5,627, 6,080] |
| Reference–AG3 | cuda:0 | 3.549 / 3.715 | 7,421 | [7,391, 7,514] |
| Reference–multi-filter | cuda:0 | 3.632 / 4.068 | 5,128 | [4,942, 5,149] |
| Descriptors–AG3 | cuda:0 | 1.343 / 1.424 | 17,681 | [17,504, 17,718] |
| Descriptors–multi-filter | cuda:0 | 1.294 / 1.372 | 18,506 | [18,222, 18,775] |
| Combined–AG3 | cuda:0 | 3.650 / 3.925 | 7,198 | [6,170, 7,219] |
| Combined–multi-filter | cuda:0 | 3.729 / 3.964 | 4,834 | [4,655, 5,007] |
| Xiong-FCM (2025) | cpu | -- | 30,852 (pool) | [29,341, 30,869] |
| LogReg–AG3 | cpu | 0.152 / 0.189 | 90,860 | [90,457, 92,239] |
| LogReg–multi-filter | cpu | 0.074 / 0.104 | 292,621 | [272,788, 294,204] |
| Reference–AG1 | cuda:0 | 3.568 / 3.827 | 7,408 | [7,197, 7,429] |
| Reference–AG5 | cuda:0 | 3.558 / 3.882 | 7,303 | [7,279, 7,435] |

Timing excludes disk/model loading, initial RFData construction, RF production, supplied-descriptor extraction and HTTP. FCM times all 24,533 records at 26 stations after one warm-up. The other methods share 512 complete-view records, 100 single calls and 32 batch-32 calls per round after 10 warm-ups per workload. Forward-only rates and peak allocated GPU memory are retained in the CSV. Timing ranges are not training-seed uncertainty.

Gong-CNN has 12.8× the measured API batch throughput of Gong-CNN–BiLSTM on this machine. The API throughput of the multi-filter reference is 69.1% of its AG3 counterpart. No timing result changes the archived accuracy scores.
