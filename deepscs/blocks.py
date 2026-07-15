"""SE-ResNet building block: ResNeXt-style split-transform-merge (grouped 5x5 conv,
cardinality=4 groups implemented as PyTorch conv groups — the ResNeXt paper itself
notes grouped convolution as an exact equivalent of explicit multi-branch aggregation)
+ squeeze-excite channel attention. Matches the official DeepSC-S repo: 5x5 kernels,
`same` padding, BatchNorm+ReLU, SE reduction ratio r=4, residual `out = in + SE(x)*x`."""
from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    """Global-average-pool -> bottleneck FC -> sigmoid gate per channel."""

    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        hidden = max(channels // r, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        return self.fc(s).view(b, c, 1, 1)


class SEResNetBlock(nn.Module):
    def __init__(self, channels: int, cardinality: int = 4, r: int = 4, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.transition = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=cardinality),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcite(channels, r)
        self.last_se_weights: torch.Tensor | None = None  # (B, C) — cached for visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.transition(x)
        s = self.se(t)
        self.last_se_weights = s.detach().view(s.shape[0], s.shape[1])
        return x + s * t


def conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 5) -> nn.Sequential:
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
