"""SE-ResNet building block, matching the official DeepSC-S repo (Zhenzi-Weng/DeepSC-S,
`models.py`) rather than the paper's one-line description of it:

    split      -> `cardinality` branches, each a full 5x5 conv to `branch_filters`
                  channels + BN + ReLU, concatenated
    transition -> 1x1 conv down to `out_dim` + BN, no activation
    SE         -> global average pool -> FC/r -> ReLU -> FC -> sigmoid, per channel
    output     -> `inputs + SE(t) * t`

The branches each see the *whole* input (they are not grouped/split channel-wise), so
`cardinality` full convs concatenated is exactly one conv to `cardinality*branch_filters`
channels — BatchNorm is per-channel and ReLU is elementwise, so folding them is exact,
not an approximation. Every conv is bias-free, as in the repo.

The residual add requires `in_ch == out_dim`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    """Global-average-pool -> bottleneck FC -> sigmoid gate per channel. Bias-free FCs."""

    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        hidden = max(channels // r, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        return self.fc(s).view(b, c, 1, 1)


class SEResNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_dim: int, cardinality: int = 4,
                 branch_filters: int = 128, r: int = 4, kernel_size: int = 5):
        super().__init__()
        assert in_ch == out_dim, f"residual add needs in_ch == out_dim ({in_ch} vs {out_dim})"
        wide = cardinality * branch_filters
        self.split = nn.Sequential(
            nn.Conv2d(in_ch, wide, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(wide),
            nn.ReLU(inplace=True),
        )
        self.transition = nn.Sequential(          # 1x1, no activation — repo's transition_layer
            nn.Conv2d(wide, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.se = SqueezeExcite(out_dim, r)
        self.last_se_weights: torch.Tensor | None = None  # (B, C) — cached for visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.transition(self.split(x))
        s = self.se(t)
        self.last_se_weights = s.detach().view(s.shape[0], s.shape[1])
        return x + s * t


def conv_bn(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 1) -> nn.Sequential:
    """Repo's conv_bn_layer: bias-free conv + BN. Activation is applied by the caller,
    because the channel encoder deliberately has none."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_ch),
    )


def convtrans_bn(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.Sequential:
    """Repo's convtrans_bn_layer. output_padding restores TF's `same` output size, so a
    stride-2 transpose exactly undoes a stride-2 conv."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride,
                           padding=kernel_size // 2, output_padding=stride - 1, bias=False),
        nn.BatchNorm2d(out_ch),
    )


def conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(conv_bn(in_ch, out_ch, kernel_size, stride), nn.ReLU(inplace=True))
