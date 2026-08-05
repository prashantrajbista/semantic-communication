"""DeepSC-S transceiver: SemanticEncoder -> ChannelEncoder -> [channel] -> ChannelDecoder ->
SemanticDecoder. Feature depth D=32, kernels 5x5 (paper's "4x32" = filter count not kernel
size — see docs/initial_plan.md #1/#3), SE reduction ratio r=4. All conv layers use `same`
padding so spatial dims (F, L) never shrink; only channel depth changes."""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SEResNetBlock, conv_bn_relu

D = 32  # semantic feature depth


class SemanticEncoder(nn.Module):
    """Raw 1-channel framed matrix (B,1,F,L) -> D-channel semantic feature map (B,D,F,L)."""

    def __init__(self, n_blocks: int = 4, cardinality: int = 4, r: int = 4):
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu(1, 16), conv_bn_relu(16, D))
        self.se_blocks = nn.ModuleList(
            [SEResNetBlock(D, cardinality, r) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.se_blocks:
            x = block(x)
        return x

    def se_weights(self) -> list[torch.Tensor]:
        """SE gate values cached from the last forward pass, one (B, D) tensor per block."""
        return [b.last_se_weights for b in self.se_blocks]


class SemanticDecoder(nn.Module):
    """D-channel feature map (B,D,F,L) -> reconstructed 1-channel matrix (B,1,F,L)."""

    def __init__(self, n_blocks: int = 4, cardinality: int = 4, r: int = 4):
        super().__init__()
        self.se_blocks = nn.ModuleList(
            [SEResNetBlock(D, cardinality, r) for _ in range(n_blocks)]
        )
        # Paper's Table: the output layer is "1 x CNN module, 1 kernel, no activation".
        # No Tanh — it saturates on loud frames and silently caps SDR, which is exactly
        # the metric E0 is judged on.
        self.head = nn.Sequential(
            conv_bn_relu(D, 16),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.se_blocks:
            x = block(x)
        return self.head(x)


class ChannelEncoder(nn.Module):
    """D-channel semantic features -> `depth` real channel symbols. `depth` is the
    compression knob (#4 in the plan): channel width in complex symbols per (frame,
    column) position is depth/2. Linear output (no BN/activation) — the values feed
    straight into power normalization, which needs an unconstrained range."""

    def __init__(self, depth: int = 8):
        super().__init__()
        assert depth % 2 == 0, "depth must be even (I/Q pairs)"
        self.depth = depth
        self.net = nn.Sequential(conv_bn_relu(D, 16), nn.Conv2d(16, depth, kernel_size=5, padding=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelDecoder(nn.Module):
    """Inverse of ChannelEncoder: `depth` real channel symbols -> D-channel features."""

    def __init__(self, depth: int = 8):
        super().__init__()
        self.net = nn.Sequential(conv_bn_relu(depth, 16), conv_bn_relu(16, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSC_S(nn.Module):
    """Full transceiver. `channel` is any callable (complex tensor -> complex tensor);
    pass None to disable the channel entirely (stage 2: prove the autoencoder works
    before any noise is introduced)."""

    def __init__(self, depth: int = 8, n_blocks: int = 4, cardinality: int = 4, r: int = 4):
        super().__init__()
        self.semantic_encoder = SemanticEncoder(n_blocks, cardinality, r)
        self.channel_encoder = ChannelEncoder(depth)
        self.channel_decoder = ChannelDecoder(depth)
        self.semantic_decoder = SemanticDecoder(n_blocks, cardinality, r)

    def forward(self, x: torch.Tensor, channel=None):
        """x: (B, 1, F, L). Returns reconstruction (B, 1, F, L)."""
        b = self.semantic_encoder(x)
        z = self.channel_encoder(b)  # (B, depth, F, L) real channel symbols
        z_hat = channel(z) if channel is not None else z
        b_hat = self.channel_decoder(z_hat)
        return self.semantic_decoder(b_hat)
