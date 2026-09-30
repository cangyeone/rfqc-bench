"""PnSn-derived reference model; architecture reuse without phase-task weights."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from .pnsn import BRNN
GAUSSIANS = (1., 1.5, 2., 2.5, 3., 4., 5.)

def make_pnsn(pretrained=False):
    if pretrained:
        raise ValueError("This benchmark uses scratch RF weights; phase-task transfer is excluded.")
    return BRNN()

class MultiBandQC(nn.Module):
    def __init__(self, mode='combined', pretrained=False, input_samples=2048, hidden=64):
        super().__init__()
        if mode not in {'waveform', 'combined', 'features'}:
            raise ValueError(mode)
        if mode == 'features' and pretrained:
            raise ValueError('Feature-only arm has no PnSn transfer learning.')
        if input_samples < 128:
            raise ValueError('PnSn has seven stride-2 pools; input_samples must be >= 128.')
        self.mode, self.input_samples = mode, input_samples
        width = 1  # Gaussian parameter records the physical filter scale, not a frequency in Hz.
        if mode != 'features':
            upstream = make_pnsn(pretrained)
            self.encoder, self.temporal = upstream.encoder, upstream.rnns
            width += 192  # mean + max of the 96-dimensional projected sequence
        if mode != 'waveform':
            self.character = nn.Sequential(nn.Linear(6, 16), nn.GELU())
            width += 16
        # Resetting seeds in the trainer keeps head initialization paired across transfer arms.
        self.project = nn.Sequential(nn.Linear(width, 96), nn.GELU())
        self.gru = nn.GRU(96, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, waveforms, features, gaussians=None, lengths=None):
        source = features if self.mode == 'features' else waveforms
        batch, bands = source.shape[:2]
        if gaussians is None:
            if bands != len(GAUSSIANS):
                raise ValueError('Supply Gaussian values explicitly for a subset of bands.')
            gaussians = source.new_tensor(GAUSSIANS)
        if gaussians.ndim==1:
            gaussians = gaussians.view(1,-1).expand(batch,-1)
        if gaussians.shape != (batch,bands):
            raise ValueError('Expected Gaussian parameters with shape (batch, bands).')
        if lengths is None:
            lengths = torch.full((batch,),bands,dtype=torch.long)
        if lengths.shape != (batch,) or (lengths<1).any() or (lengths>bands).any():
            raise ValueError('Each sample must have 1..bands valid inputs.')
        valid = torch.arange(bands,device=source.device)[None,:] < lengths.to(source.device)[:,None]
        pieces = [(gaussians / 5).unsqueeze(-1)]
        if self.mode != 'features':
            # Exclude padding BEFORE PnSn, including its batch-normalization layers.
            wave = waveforms[valid].unsqueeze(1)
            # Same adaptation in random and pretrained arms; retains sign and amplitude.
            # Interpolation does not create new frequency information.
            wave = F.interpolate(wave, size=self.input_samples, mode='linear', align_corners=True)
            sequence = self.temporal(self.encoder(wave.repeat(1, 3, 1)))
            vector = torch.cat([sequence.mean(-1), sequence.amax(-1)], dim=1)
            vectors = vector.new_zeros(batch,bands,192)
            vectors[valid] = vector
            pieces.append(vectors)
        if self.mode != 'waveform':
            pieces.append(self.character(features))
        tokens = self.project(torch.cat(pieces, -1))
        packed = pack_padded_sequence(tokens,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.classifier(hidden[-1]).squeeze(-1)
