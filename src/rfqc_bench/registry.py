"""Stable public names; these are benchmark adaptations, not source pipelines."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    kind: str = "neural"
    gaussian: float | None = 3.0
    mode: str = "waveform"
    learning_rate: float = 3e-4
    doi: str | None = None


_SPECS = [
    ModelSpec("li2021_cnn", "Li-CNN (2021)", doi="10.6038/cjg2021O0378"),
    ModelSpec("gan2021_cnn", "Gan-CNN (2021)", doi="10.6038/cjg2021O0141"),
    ModelSpec("gong_cnn", "Gong-CNN (2022)", doi="10.3389/feart.2022.921830"),
    ModelSpec("gong_cnn_bilstm", "Gong-CNN-BiLSTM (2022)", doi="10.3389/feart.2022.921830"),
    ModelSpec("gan_cnn", "Gan-CNN (2023)", doi="10.1093/gji/ggac417"),
    ModelSpec("deeprfqc", "DeepRFQC (2024)", doi="10.26443/seismica.v3i2.1341"),
    ModelSpec("hegaz_capsule", "RF-Capsule (2025)", doi="10.1016/j.eqs.2024.09.002"),
    ModelSpec("chen2026_image", "Chen-AlexNet (2026)", learning_rate=1e-5, doi="10.1016/j.eqs.2026.01.003"),
    ModelSpec("xiong2025_fcm", "Xiong-FCM (2025)", kind="fcm", doi="10.1029/2024EA003859"),
    ModelSpec("reference_ag3", "Reference-AG3"),
    ModelSpec("reference_multifilter", "Reference-multi-filter", gaussian=None),
    ModelSpec("descriptors_ag3", "Descriptors / AG3", mode="features"),
    ModelSpec("descriptors_multifilter", "Descriptors / multi-filter", gaussian=None, mode="features"),
    ModelSpec("combined_ag3", "Waveform + descriptors / AG3", mode="combined"),
    ModelSpec("combined_multifilter", "Waveform + descriptors / multi-filter", gaussian=None, mode="combined"),
    ModelSpec("logreg_ag3", "LogReg / AG3", kind="logistic", mode="features"),
    ModelSpec("logreg_multifilter", "LogReg / multi-filter", kind="logistic", gaussian=None, mode="features"),
    ModelSpec("reference_ag1", "Reference-AG1", gaussian=1.0),
    ModelSpec("reference_ag5", "Reference-AG5", gaussian=5.0),
]
REGISTRY = {s.name: s for s in _SPECS}


def get_spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose one of {', '.join(REGISTRY)}") from None


def list_models() -> list[dict]:
    return [asdict(s) for s in _SPECS]
