"""Low-level neural networks accept preprocessed tensors and return logits."""
from torch import nn
from ..registry import get_spec
from .literature import Gong, Gan, DeepRFQC, Capsule
from .extension import Li2021, Gan2021, Chen2026
from .reference import MultiBandQC


class SingleViewQC(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, waveforms, features=None, gaussians=None, lengths=None):
        if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (1, 501):
            raise ValueError("Single-view neural input must have shape (N, 1, 501)")
        out = self.network(waveforms)
        return out[:, 1] - out[:, 0] if out.ndim == 2 else out


def create_model(name: str) -> nn.Module:
    """Create random neural weights; use RFQCPredictor for trained inference."""
    spec = get_spec(name)
    if spec.kind != "neural":
        raise ValueError("FCM/LogReg are statistical estimators; use fit() or RFQCPredictor")
    constructors = {
        "li2021_cnn": lambda: Li2021(501), "gan2021_cnn": lambda: Gan2021(501),
        "gong_cnn": lambda: Gong(501), "gong_cnn_bilstm": lambda: Gong(501, True),
        "gan_cnn": lambda: Gan(501), "deeprfqc": lambda: DeepRFQC(501),
        "hegaz_capsule": lambda: Capsule(501), "chen2026_image": Chen2026,
    }
    if name in constructors:
        return SingleViewQC(constructors[name]())
    return MultiBandQC(spec.mode, pretrained=False, input_samples=2048)
