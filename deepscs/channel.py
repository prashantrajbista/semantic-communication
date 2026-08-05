"""Differentiable wireless channel layers: power normalization + AWGN + Rayleigh/Rician
fading with perfect-CSI equalization. Noise must be differentiable (added, not looked up)
so gradients flow encoder <- channel <- decoder during training.

Layout, matching the official repo's Chan_Model. The channel encoder emits a real tensor
(B, C, H, W); each of the C feature maps is flattened and read as K = H*W/2 consecutive
I/Q pairs, giving C rows of K complex symbols.

**Fading is per row, not per symbol.** The repo draws `h` with shape [B, C, 1] and
broadcasts it over all K symbols in that row — block fading with a 512-symbol block at
the paper's dimensions. Drawing `h` i.i.d. per symbol instead would hand the system full
diversity the paper never had, and would flatter both DeepSC-S and the benchmark.

Power normalization follows the repo exactly: I and Q are L2-normalized *separately* per
row and scaled by sqrt(K/2), so each real dimension carries average power 0.5 and each
complex symbol carries E|x|^2 = 1.
"""
from __future__ import annotations

import torch


def to_pairs(z: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) real -> (B, C, K, 2) I/Q pairs, K = H*W/2."""
    b, c, h, w = z.shape
    assert (h * w) % 2 == 0, "H*W must be even to pair into I/Q"
    return z.reshape(b, c, (h * w) // 2, 2)


def from_pairs(p: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
    """Exact inverse of to_pairs."""
    return p.reshape(p.shape[0], c, h, w)


def bandwidth_ratio(filters: int = 128, downsample: int = 4) -> float:
    """Complex channel uses per source sample, rho = FN/W.

    The encoder shrinks (F, L) by `downsample` in each dimension and widens the channel
    axis to `filters`, so per source sample the system spends
    filters / (2 * downsample^2) complex symbols. At the repo's 128 filters and 4x
    downsampling that is rho = 4 — exactly what 8-bit PCM + turbo 1/3 + 64-QAM costs
    (see deepscs.baseline). Report it with every result: a number at an unstated rho is
    not a result."""
    return filters / (2 * downsample ** 2)


def power_norm(p: torch.Tensor) -> torch.Tensor:
    """(B, C, K, 2) -> same, with mean square 0.5 per real dimension. I and Q normalized
    independently, as in the repo's `sqrt(K/2) * l2_normalize(x, axis=2)`."""
    k = p.shape[2]
    norm = p.pow(2).sum(dim=2, keepdim=True).sqrt().clamp_min(1e-8)
    return p / norm * (k / 2) ** 0.5


def measured_power(x: torch.Tensor) -> float:
    """Mean |x_k|^2 across all complex symbols — should read ~1.0 after power_norm."""
    return x.abs().pow(2).mean().item()


def snr_db_to_sigma(snr_db: float) -> float:
    """Per-real-dim noise std given unit signal power: sigma^2 = 1/(2*10^(SNR/10))."""
    return (10 ** (-snr_db / 10) / 2) ** 0.5


def _awgn_noise(shape, snr_db: float, device, dtype) -> torch.Tensor:
    sigma = snr_db_to_sigma(snr_db)
    real = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    imag = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(real, imag).to(dtype)


def _block_fading(shape, k_factor: float, device, dtype) -> torch.Tensor:
    """One coefficient per row, broadcast over that row's symbols. `shape` is (B, C, K);
    the draw is (B, C, 1). k_factor 0 gives Rayleigh, -> inf gives AWGN."""
    b, c, _ = shape
    los = (k_factor / (k_factor + 1)) ** 0.5
    scatter = (1 / (2 * (k_factor + 1))) ** 0.5
    real = los + torch.randn((b, c, 1), device=device) * scatter
    imag = torch.randn((b, c, 1), device=device) * scatter
    return torch.complex(real, imag).to(dtype)


class AWGNChannel(torch.nn.Module):
    """y = x + w. Caches last tx/rx complex symbols for visualization."""

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        w = _awgn_noise(x.shape, snr_db, x.device, x.dtype)
        y = x + w
        self.last_tx, self.last_rx, self.last_h = x.detach(), y.detach(), None
        return y


class _FadingChannel(torch.nn.Module):
    """y = h*x + w with block fading, equalized under perfect CSI as x_hat = y/h."""

    k_factor = 0.0

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        h = _block_fading(x.shape, self.k_factor, x.device, x.dtype)
        w = _awgn_noise(x.shape, snr_db, x.device, x.dtype)
        y = h * x + w
        x_hat = y / h  # perfect CSI equalization
        self.last_tx, self.last_rx, self.last_h = x.detach(), y.detach(), h.detach()
        return x_hat


class RayleighChannel(_FadingChannel):
    """No line of sight: h ~ CN(0, 1) per row."""

    k_factor = 0.0


class RicianChannel(_FadingChannel):
    """LoS component sqrt(K/(K+1)) plus Rayleigh scatter. The repo hardcodes the
    equivalent of K = 1 (E|h|^2 = 1, LoS power 0.5); its LoS sits at 45 degrees rather
    than on the real axis, which perfect-CSI equalization makes irrelevant."""

    def __init__(self, k_factor: float = 1.0):
        super().__init__()
        self.k_factor = k_factor


class ChannelLayer(torch.nn.Module):
    """Wraps to_pairs -> power_norm -> {awgn,rayleigh,rician} -> from_pairs so it can be
    dropped straight into DeepSC_S.forward(x, channel=this)."""

    _CHANNELS = {"awgn": AWGNChannel, "rayleigh": RayleighChannel}

    def __init__(self, kind: str = "awgn", snr_db: float = 10.0, rician_k: float = 1.0):
        super().__init__()
        assert kind in ("awgn", "rayleigh", "rician")
        self.kind = kind
        self.snr_db = snr_db
        self.impl = RicianChannel(rician_k) if kind == "rician" else self._CHANNELS[kind]()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        c, h, w = z.shape[1], z.shape[2], z.shape[3]
        # Always fp32 here: torch.complex rejects half/bfloat16 inputs, and the power
        # constraint and noise variance are exactly what must not be quantized away —
        # a low-precision channel is a different channel.
        with torch.autocast(device_type=z.device.type, enabled=False):
            p = power_norm(to_pairs(z.float()))
            x = torch.complex(p[..., 0], p[..., 1])
            x_hat = self.impl(x, self.snr_db)
            out = from_pairs(torch.stack([x_hat.real, x_hat.imag], dim=-1), c, h, w)
        return out.to(z.dtype)
