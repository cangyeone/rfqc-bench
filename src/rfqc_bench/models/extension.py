"""Benchmark architecture snapshot. See THIRD_PARTY_NOTICES.md for attribution."""

from pathlib import Path

import math

import torch

from torch import nn

from torch.nn import functional as F

def initialize(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def flatten_keras(x):
    return x.transpose(1, 2).contiguous().flatten(1)

class SameConv(nn.Conv1d):
    def forward(self, x):
        total = max(0, (math.ceil(x.shape[-1]/self.stride[0])-1)*self.stride[0]
                    + self.kernel_size[0]-x.shape[-1])
        return super().forward(F.pad(x, (total//2, total-total//2)))

def same_pool(x, kernel=3, stride=2):
    total = max(0, (math.ceil(x.shape[-1]/stride)-1)*stride+kernel-x.shape[-1])
    return F.max_pool1d(F.pad(x, (total//2, total-total//2), value=-float('inf')), kernel, stride)

class Li2021(nn.Module):
    """Paper Fig.1: 16/16 width-5 conv, pool 2, FC 256/60/2."""
    def __init__(self, samples=501):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)
        self.dense = nn.Sequential(nn.Linear((samples//4)*16,256), nn.ReLU(), nn.Dropout(.5),
                                   nn.Linear(256,60), nn.ReLU(), nn.Dropout(.5), nn.Linear(60,2))
        self.apply(initialize)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)),2)
        x = F.max_pool1d(F.relu(self.conv2(x)),2)
        return self.dense(flatten_keras(x))

class Gan2021(nn.Module):
    """Paper Fig.3: two convolutions 32/64; one pool; FC 100/2.

    This is distinct from the three-convolution 2023 released implementation.
    SAME padding gives the paper's 419 -> 210 -> 105 example dimensions.
    """
    def __init__(self, samples=501):
        super().__init__()
        self.conv1 = SameConv(1,32,3,stride=1)
        self.conv2 = SameConv(32,64,3,stride=2)
        self.dense1 = nn.Linear(math.ceil(math.ceil(samples/2)/2)*64,100)
        self.dense2 = nn.Linear(100,2)
        self.dropout = nn.Dropout(.2)
        self.apply(initialize)

    def forward(self, x):
        x = self.dropout(F.relu(self.conv1(x)))
        x = same_pool(self.dropout(F.relu(self.conv2(x))))
        x = self.dropout(F.relu(self.dense1(flatten_keras(x))))
        return self.dense2(x)

def wiggle_rgb(wave, size=448):
    """Deterministic single-RF image adapter; no station/label/text is rendered.

    Input is per-trace absolute-peak normalized raw AG3, not asinh transformed.
    Red positive lobes, blue negative lobes, black outline, white background;
    6% horizontal/10% vertical margins. This is not pixel-identical MATLAB output.
    """
    if wave.ndim != 3 or wave.shape[1] != 1:
        raise ValueError('Expected B,1,T')
    margin = round(.06*size)
    curve = F.interpolate(wave, size=size-2*margin, mode='linear', align_corners=True)
    curve = F.pad(curve, (margin,margin))[:,0]
    y = torch.linspace(1.25,-1.25,size,device=wave.device,dtype=wave.dtype)[None,:,None]
    inside = torch.ones(size,device=wave.device,dtype=torch.bool)
    inside[:margin] = False; inside[-margin:] = False
    pos = (curve[:,None,:]>0)&(y>=0)&(y<=curve[:,None,:])&inside
    neg = (curve[:,None,:]<0)&(y<=0)&(y>=curve[:,None,:])&inside
    edge = (torch.abs(y-curve[:,None,:])<=1.5*2.5/(size-1))&inside
    rgb = wave.new_ones(wave.shape[0],3,size,size)
    rgb[:,1].masked_fill_(pos|neg,0)
    rgb[:,2].masked_fill_(pos,0)
    rgb[:,0].masked_fill_(neg,0)
    rgb.masked_fill_(edge[:,None],0)
    return rgb

class DynamicAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(256,256,3,padding=1)
    def forward(self,x):
        return x*torch.sigmoid(self.conv(x))

class Chen2026(nn.Module):
    """Exact layer shapes of authors' torchvision AlexNet+DAB inference source.

    Released code has 64/192/384/256/256 channels and 4096/4096 FC,
    unlike the paper's diagram. AdaptiveAvgPool(6,6) is retained.
    No torchvision installation or external weight download is needed.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Sequential(
            nn.Conv2d(3,64,11,stride=4,padding=2),nn.ReLU(inplace=True),nn.MaxPool2d(3,2),
            nn.Conv2d(64,192,5,padding=2),nn.ReLU(inplace=True),nn.MaxPool2d(3,2),
            nn.Conv2d(192,384,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),nn.ReLU(inplace=True),nn.MaxPool2d(3,2)),
            DynamicAttentionBlock())
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(nn.Dropout(.5),nn.Linear(256*6*6,4096),nn.ReLU(inplace=True),
            nn.Dropout(.5),nn.Linear(4096,4096),nn.ReLU(inplace=True),nn.Linear(4096,2))
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('image_std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self, x):
        image=(wiggle_rgb(x)-self.image_mean)/self.image_std
        return self.classifier(self.avgpool(self.features(image)).flatten(1))
