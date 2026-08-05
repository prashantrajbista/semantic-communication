"""DeepSC-S transceiver, following the authors' TensorFlow reference
(Zhenzi-Weng/DeepSC-S, `models.py` + `main.py` defaults) rather than the paper's
config table, which disagrees with the code the paper's numbers came from:

    paper table               official repo                  used here
    4 x SE-ResNet, 32 kernels 5 x SE-ResNet, out_dim 128     repo
    channel coder: 8 kernels  channel coder: 128 filters     repo
    (no front-end stated)     2 stride-2 convs, 4x downsample  repo
    (no tail stated)          2 stride-2 transposed convs    repo

Shapes, at the paper's W = 16384 = F 128 x L 128:

    (B,1,128,128) -stride-2 conv x2-> (B,128,32,32) -chan enc-> (B,128,32,32)
      -> 128 rows x 512 complex symbols = 65536 symbols = rho 4 per source sample

The two stride-2 convs shrink the feature map 16x while the channel width grows 1 -> 128,
so the symbol budget is unchanged: rho stays 4, exactly what the paper's PCM + turbo 1/3
+ 64-QAM benchmark costs.

Waveform mean/variance normalization happens inside `DeepSC_S.forward` and is undone at
the output, so the MSE loss is still measured against the untouched input (repo's
wav_norm/wav_denorm).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SEResNetBlock, conv_bn, convtrans_bn, init_keras_default

D = 128           # semantic feature depth (repo: sem_enc_outdims[1:])
STEM = 32         # first stride-2 conv width (repo: sem_enc_outdims[0])
N_BLOCKS = 5      # repo: len(sem_enc_outdims[2:])
CHAN_FILTERS = 128  # repo: chan_enc_filters / chan_dec_filters


class SemanticEncoder(nn.Module):
    """(B,1,F,L) raw framed matrix -> (B,D,F/4,L/4) semantic features."""

    def __init__(self, n_blocks: int = N_BLOCKS, cardinality: int = 4, r: int = 4):
        super().__init__()
        self.down1 = conv_bn(1, STEM, stride=2)
        self.down2 = conv_bn(STEM, D, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.se_blocks = nn.ModuleList([SEResNetBlock(D, D, cardinality, r=r) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.down2(self.relu(self.down1(x))))
        for block in self.se_blocks:
            x = self.relu(block(x))
        return x

    def se_weights(self) -> list[torch.Tensor]:
        """SE gate values cached from the last forward pass, one (B, D) tensor per block."""
        return [b.last_se_weights for b in self.se_blocks]


class SemanticDecoder(nn.Module):
    """(B,D,F/4,L/4) -> (B,1,F,L). Final layer is a single 1x1 conv with no activation."""

    def __init__(self, n_blocks: int = N_BLOCKS, cardinality: int = 4, r: int = 4):
        super().__init__()
        self.se_blocks = nn.ModuleList([SEResNetBlock(D, D, cardinality, r=r) for _ in range(n_blocks)])
        self.up1 = convtrans_bn(D, D, stride=2)
        self.up2 = convtrans_bn(D, STEM, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(STEM, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.se_blocks:
            x = self.relu(block(x))
        x = self.relu(self.up2(self.relu(self.up1(x))))
        return self.head(x)


class ChannelEncoder(nn.Module):
    """One 5x5 conv + BN, no activation — the values feed straight into power
    normalization, which needs an unconstrained range (repo: Chan_Enc)."""

    def __init__(self, filters: int = CHAN_FILTERS):
        super().__init__()
        assert filters % 2 == 0, "filters must be even (I/Q pairs)"
        self.filters = filters
        self.net = conv_bn(D, filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelDecoder(nn.Module):
    """One 5x5 conv + BN + ReLU (repo: Chan_Dec)."""

    def __init__(self, filters: int = CHAN_FILTERS):
        super().__init__()
        self.net = nn.Sequential(conv_bn(filters, D), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSC_S(nn.Module):
    """Full transceiver. `channel` is any callable (B,C,H,W) -> (B,C,H,W); pass None to
    disable the channel entirely (stage 2: prove the autoencoder works before noise)."""

    def __init__(self, filters: int = CHAN_FILTERS, n_blocks: int = N_BLOCKS,
                 cardinality: int = 4, r: int = 4):
        super().__init__()
        self.semantic_encoder = SemanticEncoder(n_blocks, cardinality, r)
        self.channel_encoder = ChannelEncoder(filters)
        self.channel_decoder = ChannelDecoder(filters)
        self.semantic_decoder = SemanticDecoder(n_blocks, cardinality, r)
        self.apply(init_keras_default)  # Keras' glorot_uniform, not PyTorch's kaiming

    def forward(self, x: torch.Tensor, channel=None):
        """x: (B, 1, F, L). Returns reconstruction (B, 1, F, L) in the input's own scale."""
        dims = (1, 2, 3)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.var(dim=dims, unbiased=False, keepdim=True).clamp_min(1e-12).sqrt()
        b = self.semantic_encoder((x - mean) / std)
        z = self.channel_encoder(b)
        z_hat = channel(z) if channel is not None else z
        b_hat = self.channel_decoder(z_hat)
        return self.semantic_decoder(b_hat) * std + mean
