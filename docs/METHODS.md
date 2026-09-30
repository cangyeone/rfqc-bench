# Model implementations and reproduction limits

## Coverage and reproduction level

| Supplied paper | Benchmark configuration | Basis and important differences |
|---|---|---|
| Li et al. 2021, doi:10.6038/cjg2021O0378 | `li2021_cnn` | Paper Section 1.2 / Fig.1: two width-5, 16-channel convolutions; SAME padding; two max-pools of 2; 256/60/2 dense units. Original 600 points at 20 Hz over −5..25 s becomes the observed 501 points at 10 Hz over −10..40 s. Dropout 0.5 on the two hidden dense activations is an explicit reconstruction choice because exact insertion points are not fully specified. No author training code located in the checked materials. |
| Gan et al. 2021, doi:10.6038/cjg2021O0141 | `gan2021_cnn` | Section 2.2 / Fig.3: two convolutions (32/64, width 3, strides 1/2), one width-3 stride-2 max-pool, FC100/2. SAME padding yields the published 419→210→105 sizes. Dropout 0.2 from the paper's selected setting is applied to hidden convolution/dense activations. This is distinct from the 2023 network. The supplementary azimuth/distance-conditioned similarity recovery cannot be reproduced without those event attributes; only the CNN component is evaluated. |
| Gong et al. 2022, doi:10.3389/feart.2022.921830 | `gong_cnn`, `gong_cnn_bilstm` | Fixed source-based adaptations; four residual blocks from released code rather than three in text. Recurrent dropout retained. See the source-derived details below. |
| Gan et al. 2023, doi:10.1093/gji/ggac417 | `gan_cnn` | Existing released three-convolution adaptation. This measures RF quality classification, not the inversion component. |
| Sabermahani & Frederiksen 2024, doi:10.26443/seismica.v3i2.1341 | `deeprfqc` | Existing author-source U-net adaptation; common loss/preprocessing and no original augmentation. These changes may affect ranking and are explicitly retained in the comparison limits. |
| Hegazi et al. 2025, doi:10.1016/j.eqs.2024.09.002 | `hegaz_capsule` | Existing released-notebook variant, not the incompatible architecture in the main paper diagram. |
| Xiong et al. 2025, doi:10.1029/2024EA003859 | `xiong2025_fcm` | Equations 1–8 reconstructed; four features and fuzzy c-means. Uses additional unlabeled same-station context. Appears in the separate feature/context section with all changes below. |
| Chen et al. 2026, doi:10.1016/j.eqs.2026.01.003 | `chen2026_image` | Author `RF_classify.py` at commit `6084ce4cccf05bba4cd3bb40b5c355677938583c`: torchvision-style AlexNet (64/192/384/256/256 channels, AdaptiveAvgPool6, FC4096/4096/2) plus a 256-channel 3×3 sigmoid attention block after convolution features. This differs from paper channels 96/256/384/384/256 and FC2048/2048/1000/2. Uses the released structure, initialized from scratch, never author/ImageNet weights. |

These are registered adaptations on a common RF cohort, not claims to reproduce each author's published score or complete original data-processing pipeline. Every supplied paper is accounted for; unavailable components are not replaced by invented results.

## Chen image adapter

The authors' scripts render the first 500 SAC samples as a MATLAB `wiggle(...,'BR')` figure and resize to 448×448 with fixed channel means/stds. Their MATLAB figure margins and maximum-positive-amplitude scale differ from this implementation. We render all 501 observed AG3 samples deterministically on the GPU, with absolute-peak normalization, positive red / negative blue lobes, black waveform outline, white background, 6% horizontal and 10% vertical margins. No sample name, station name, manual class, filename or axes text is included. The layer shapes follow released code but images are not claimed pixel-identical to MATLAB output. Per-channel normalization constants are preprocessing, not pretrained weights.

Chen's learning rate is fixed at 1e-5, as reported in the supplied paper. Li/Gan use the established 3e-4 comparison rate. All three use the same data split, balanced sampler, batch 32, maximum 50 epochs, validation AP early stopping (patience10), validation macro-F1 threshold selection and three seeds. Model-specific image preprocessing and this learning-rate difference must be visible in the manuscript. No post-test hyperparameter selection is allowed. The shared AdamW/BCE optimization is a benchmark adaptation; it is not the authors' full training recipe.

## Xiong feature and clustering adapter

- Work with the full available −10..40 s raw waveform at 10 Hz, scaled to unit norm using overflow-safe intermediate scaling.
- Peak ratio is the second distinct positive local maximum divided by the largest positive local maximum (`scipy.signal.find_peaks`). With fewer than two distinct positive peaks, set the ratio to zero. This edge-case convention is explicit.
- Top-10% amplitude mass uses the 50 largest absolute values of 501 samples, divided by total absolute amplitude. This quantity is called a Gini ratio in the source and is not the classical Gini coefficient.
- Follow the orientation of source Equation3: noise RMS / signal RMS. Signal is [0,10] s. Noise is [−10,−3] s because the original [−13,−3] s is unavailable. No missing samples are invented.
- Correlation uses centered, normalized [0,10] s traces at the same station; exclude self, average the highest ceiling((N−1)/2) correlations. Singleton context defaults to zero. This uses unlabeled held-out-station waveforms together, hence is a station-batch method, not an independent-trace classifier. No station ID is used as a predictive category.
- Fit outlier statistics and min/max scaling on training only. Exclude training rows at |z|≥3 from centroid fitting; retain all validation/test records and clip their scaled features to [0,1]. The source instead discards outliers from its reported cohort. Report this coverage-preserving adaptation.
- Fit two fuzzy clusters on training features, m=2, up to300 iterations, center tolerance1e-5. Choose initial centroids from training samples using each fixed seed. Resolve good/bad cluster orientation using training labels only. Choose acceptance threshold using validation labels; additionally report original t=.75 and t=.5.
- Source package `fcmeans` is named by the paper but its application script is not supplied. The NumPy implementation exposes the formulas and handles exact zero distances explicitly; it is not claimed to follow the package's random initialization trajectory bit for bit.


## Additional source-derived details

Gong uses four residual blocks, each with two 8-channel width-11 convolutions,
dropout 0.1, pooling 2, and dense 32/8/2. The BiLSTM variant retains four hidden
units per direction and separate input/recurrent dropout masks, held fixed over
the sequence. This differs from a plain PyTorch LSTM layer-dropout argument.
Gan 2023 preserves TensorFlow SAME padding and time-major flattening. DeepRFQC
uses 32/64/128/256 channels, nearest-neighbor upsampling, skip connections, and
right padding from 501 to 504 points followed by cropping; the shared BCE loss
and absence of original augmentation are benchmark adaptations. The RF-Capsule
variant is the released notebook's 8-dimensional primary capsules, two
16-dimensional output capsules, three routing iterations and dropout 0.8, not
the different main-paper diagram.

## Reference and descriptor controls

The PnSn-derived encoder operates on each valid filtered view after interpolation
from 501 to 2048 samples. Its within-waveform recurrent representation is pooled
into a 192-dimensional vector; a GRU then fuses ordered filter views.
`reference_ag3` and `reference_multifilter` have identical architectures and
395,393 parameters. No phase-task parameters are loaded. `reference_ag1` and
`reference_ag5` expose the separately completed single-seed sensitivity controls.

`descriptors_ag3`, `descriptors_multifilter`, `combined_ag3` and
`combined_multifilter` cover supplied-descriptor-only and waveform-plus-descriptor
controls. `logreg_ag3` and `logreg_multifilter` are the appended L2 controls
(C=1, balanced class weights, lbfgs, default 100 iterations), using 6 and 42
columns respectively. All three multi-filter logistic fits reached the iteration
limit; this constraint is recorded in their bundles. No claim is made that those
weights are optimally converged linear baselines.

## Registry and seeds

The 17 main configurations have three registered fits each (20260928, 20260929,
20260930), giving 51 bundles. The two AG1/AG5 controls each have one fit with seed
20260929, giving 53 bundles across 19 configuration names. The default download
is the first registered seed, not the best score. Three deterministic LogReg seed
settings can produce identical solutions; duplicate outcomes remain explicit.

The public package changes packaging and calls, not the frozen numerical evidence
in the manuscript. Archived identifiers may contain `multiband`; display labels
use multi-filter to distinguish Gaussian views from disjoint frequency bands.
