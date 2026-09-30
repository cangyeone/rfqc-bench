"""Benchmark architecture snapshot. See THIRD_PARTY_NOTICES.md for attribution."""

from pathlib import Path

import math

import torch

from torch import nn

from torch.nn import functional as F

def glorot(layer):
    if isinstance(layer, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def he_truncated(layer):
    if isinstance(layer, (nn.Conv1d, nn.Linear)):
        fan_in = layer.weight[0].numel()
        std = math.sqrt(2. / fan_in) / .87962566103423978
        nn.init.trunc_normal_(layer.weight, std=std, a=-2*std, b=2*std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def flatten_time_channels(x):
    """Keras Flatten uses time-major/channel-last order, not PyTorch C,T order."""
    return x.transpose(1, 2).contiguous().flatten(1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 8, 11, padding=5)
        self.conv2 = nn.Conv1d(8, 8, 11, padding=5)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        first = F.relu(self.conv1(x))
        # Author code adds conv1 and conv2, not the raw block input.
        return self.dropout(first + F.relu(self.conv2(first)))

class VariationalLSTM(nn.Module):
    """Keras-style gates with four input/recurrent dropout masks per sequence.

Gate order i,f,c,o; unit forget bias; masks are reused across time, not across
batches. Explicit recurrence retains recurrent_dropout=.1 from Gong's code.
"""
    def __init__(self, inputs=8, hidden=4, rate=.1):
        super().__init__()
        self.inputs, self.hidden, self.rate = inputs, hidden, rate
        self.kernel = nn.Parameter(torch.empty(inputs, 4*hidden))
        self.recurrent = nn.Parameter(torch.empty(hidden, 4*hidden))
        self.bias = nn.Parameter(torch.zeros(4*hidden))
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent)
        with torch.no_grad():
            self.bias[hidden:2*hidden].fill_(1.)

    def forward(self, x):
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden)
        c = torch.zeros_like(h)
        xm = F.dropout(x.new_ones(4, batch, self.inputs), self.rate, self.training)
        hm = F.dropout(x.new_ones(4, batch, self.hidden), self.rate, self.training)
        # Precompute the input projection for each gate across all timesteps.
        projected = [F.linear(x*xm[i, :, None], k.T, b)
                     for i, (k,b) in enumerate(zip(self.kernel.chunk(4,1), self.bias.chunk(4)))]
        recurrent = self.recurrent.chunk(4, 1)
        outputs = []
        for t in range(x.shape[1]):
            gates = [projected[i][:,t] + F.linear(h*hm[i], recurrent[i].T) for i in range(4)]
            c = gates[1].sigmoid()*c + gates[0].sigmoid()*gates[2].tanh()
            h = gates[3].sigmoid()*c.tanh()
            outputs.append(h)
        return torch.stack(outputs, 1)

class Gong(nn.Module):
    def __init__(self, samples, recurrent=False):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(c) for c in (1,8,8,8)])
        self.forward_lstm = VariationalLSTM() if recurrent else None
        self.backward_lstm = VariationalLSTM() if recurrent else None
        self.dense = nn.Sequential(nn.Linear((samples//16)*8, 32), nn.ReLU(),
                                   nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 2))
        self.apply(glorot)

    def forward(self, x):
        for block in self.blocks:
            x = F.max_pool1d(block(x), 2)
        if self.forward_lstm is not None:
            sequence = x.transpose(1,2)
            forward = self.forward_lstm(sequence)
            backward = self.backward_lstm(sequence.flip(1)).flip(1)
            flat = torch.cat([forward, backward], -1).flatten(1)
        else:
            flat = flatten_time_channels(x)
        return self.dense(flat)

class SameConv1d(nn.Conv1d):
    """TensorFlow SAME for stride > 1: odd total padding goes on the right."""
    def forward(self, x):
        length, stride, kernel = x.shape[-1], self.stride[0], self.kernel_size[0]
        total = max(0, (math.ceil(length/stride)-1)*stride + kernel-length)
        return super().forward(F.pad(x, (total//2, total-total//2)))

class Gan(nn.Module):
    def __init__(self, samples):
        super().__init__()
        self.conv1 = SameConv1d(1, 16, 3, stride=1)
        self.conv2 = SameConv1d(16, 32, 3, stride=2)
        self.conv3 = SameConv1d(32, 64, 3, stride=2)
        self.drop = nn.Dropout(.25)  # keep_prob=.75 in the training notebook
        length = samples//2
        length = math.ceil(length/2)//2
        length = math.ceil(length/2)//2
        self.dense1 = nn.Linear(length*64, 150)
        self.dense2 = nn.Linear(150, 2)
        self.apply(glorot)

    def forward(self, x):
        x = F.max_pool1d(F.elu(self.conv1(x)), 2)
        x = F.max_pool1d(self.drop(F.elu(self.conv2(x))), 2)
        x = F.max_pool1d(self.drop(F.elu(self.conv3(x))), 2)
        x = self.drop(F.elu(self.dense1(flatten_time_channels(x))))
        return self.dense2(x)

def conv_pair(inputs, outputs, kernel):
    return nn.Sequential(nn.Conv1d(inputs, outputs, kernel, padding=kernel//2), nn.ReLU(),
                         nn.Conv1d(outputs, outputs, kernel, padding=kernel//2), nn.ReLU())

class DeepRFQC(nn.Module):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples
        self.enc1, self.enc2, self.enc3 = conv_pair(1,32,5), conv_pair(32,64,5), conv_pair(64,128,5)
        self.middle = conv_pair(128,256,5)
        self.dec3, self.dec2, self.dec1 = conv_pair(384,128,5), conv_pair(192,64,5), conv_pair(96,32,3)
        self.dense = nn.Sequential(nn.Linear(samples*32,32), nn.ReLU(),
                                   nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))
        self.apply(he_truncated)

    def forward(self, x):
        # Preserve the full observed 501-sample window; no synthetic time stretch.
        x = F.pad(x, (0, (-self.samples)%8))
        c1 = self.enc1(x)
        c2 = self.enc2(F.max_pool1d(c1,2))
        c3 = self.enc3(F.max_pool1d(c2,2))
        c4 = self.middle(F.max_pool1d(c3,2))
        c5 = self.dec3(torch.cat([c3, F.interpolate(c4,scale_factor=2,mode='nearest')],1))
        c6 = self.dec2(torch.cat([c2, F.interpolate(c5,scale_factor=2,mode='nearest')],1))
        c7 = self.dec1(torch.cat([c1, F.interpolate(c6,scale_factor=2,mode='nearest')],1))
        return self.dense(flatten_time_channels(c7[...,:self.samples])).squeeze(-1)

def squash(x):
    squared = x.square().sum(-1, keepdim=True)
    return squared/(1+squared)/torch.sqrt(squared+1e-7)*x

class Capsule(nn.Module):
    def __init__(self, samples):
        super().__init__()
        # Released RF-Capsule notebook variant (8x8 primary, 2x16 digit, p=.8).
        self.primary = nn.Conv1d(1,64,3,padding=1)
        self.drop = nn.Dropout(.8)
        self.weight = nn.Parameter(torch.empty(2, samples*8, 16, 8))
        self.apply(glorot)
        # Keras Glorot fan computation for this four-dimensional custom weight.
        fan_in = 2*samples*8*16
        fan_out = 2*samples*8*8
        nn.init.uniform_(self.weight, -math.sqrt(6/(fan_in+fan_out)), math.sqrt(6/(fan_in+fan_out)))

    def lengths(self, x):
        primary = self.primary(x).transpose(1,2).reshape(x.shape[0], -1, 8)
        primary = self.drop(squash(primary))
        votes = torch.einsum('bid,jiod->bjio', primary, self.weight)
        b = votes.new_zeros(votes.shape[:3])
        for i in range(3):
            weights = b.softmax(dim=1)
            output = squash(torch.einsum('bji,bjio->bjo', weights, votes))
            if i < 2:
                b = b + torch.einsum('bjo,bjio->bji', output, votes)
        return torch.sqrt(output.square().sum(-1)+1e-7)

    def forward(self, x):
        lengths = self.lengths(x)
        # sigmoid(log(good/bad)) == good/(good+bad). BCE equals normalized CCE,
        # the actual loss compiled in CapsuleRF.ipynb (unused margin_loss omitted).
        return lengths[:,1].log()-lengths[:,0].log()
