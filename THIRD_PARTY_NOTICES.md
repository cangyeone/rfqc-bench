# Attribution and distribution scope

The package distributes the RFQC benchmark's PyTorch adaptations and original
integration/API/training code under GPL-3.0-only. It does not redistribute the
authors' notebooks, paper PDFs, datasets or source-author pretrained weights.
The adaptations are not assertions of reproducing each publication's original
score or entire processing pipeline. See `docs/METHODS.md` for differences.

The PnSn definitions in `src/rfqc_bench/models/pnsn.py` are packaged from the
GPLv3 source snapshot in
[elastic-wave-structure-emerge](https://huggingface.co/cangyeone/elastic-wave-structure-emerge).
Source path: `model_definitions/pnsn/models/BRNNPNSN.py`. The package retains
the encoder/RNN definitions and their source provenance; the unused executable
example is omitted. `models/reference.py` packages the existing RF adaptation
with relative imports and no phase-task checkpoint dependency. Original source
SHA256 values and the packaging changes are in `validation/source_provenance.json`.
PnSn architecture credit: Cai et al. (2025), cited by the benchmark manuscript.

Literature adaptation sources and credits:

- Gong et al. (2022): [Receiver-Function-Quality-Control](https://github.com/gongchang96/Receiver-Function-Quality-Control).
- Gan et al. (2023): [AI_code](https://github.com/Ganlu007/AI_code).
- Sabermahani and Frederiksen (2024): [DeepRFQC archive](https://zenodo.org/records/10087652).
- Hegazi et al. (2025): [RF-Capsule](https://github.com/omarmohamed15/RF-Capsule).
- Chen et al. (2026): author AlexNet/DAB implementation documented in METHODS.
- Li et al. (2021), Gan et al. (2021), Xiong et al. (2025): the equations and
  architecture descriptions identified by DOI in METHODS.

Applicable permissive license notices from the supplied DeepRFQC, RF-Capsule and
Chen source archives are retained under `third_party/`. This notice does not
grant new rights over third-party materials that are not distributed here.
PyTorch, NumPy, SciPy, scikit-learn, safetensors and FastAPI are dependencies with
their own licenses; their distributions are installed separately through pip.

The released model bundles contain this project's RF-trained parameters,
training-only preprocessing parameters and frozen validation thresholds. They
contain no RF records, manual labels, individual predictions or credentials.
The software/model release does not assign a license to the separately managed
benchmark dataset. Transfer-initialized historical experiments are outside the
current benchmark package.
