"""Differentiable wireless channel layers: power normalization + AWGN + Rayleigh/Rician
fading with perfect-CSI equalization. Noise must be differentiable (added, not looked up)
so gradients flow encoder <- channel <- decoder during training.

Layout convention: the channel encoder emits a real tensor (B, depth, F, L) — `depth`
real values per (frame, column) position. We view consecutive pairs as I/Q and transmit
`depth/2` complex symbols per position, `K = L*depth/2` per frame row (matches
docs/initial_plan.md #5)."""
from __future__ import annotations

import torch


def to_iq(z: torch.Tensor) -> torch.Tensor:
    """(B, depth, F, L) real -> (B, F, K) complex, K = L*depth/2."""
    b, depth, f, l = z.shape
    assert depth % 2 == 0
    x = z.permute(0, 2, 3, 1).reshape(b, f, l * depth)  # (B, F, L*depth), channel-last flatten
    pairs = x.reshape(b, f, -1, 2)  # (B, F, K, 2)
    return torch.complex(pairs[..., 0], pairs[..., 1])  # (B, F, K)


def from_iq(x: torch.Tensor, depth: int, f: int, l: int) -> torch.Tensor:
    """Exact inverse of to_iq: (B, F, K) complex -> (B, depth, F, L) real."""
    b = x.shape[0]
    pairs = torch.stack([x.real, x.imag], dim=-1)  # (B, F, K, 2)
    flat = pairs.reshape(b, f, l * depth)  # (B, F, L*depth)
    return flat.reshape(b, f, l, depth).permute(0, 3, 1, 2)  # (B, depth, F, L)


def power_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-frame-row power normalization: scale each (B, F) row of K complex symbols so
    E|x_k|^2 = 1 averaged over that row's K symbols (`sqrt(K)*unit_norm(x)`, docs #5)."""
    k = x.shape[-1]
    norm = x.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    return x / norm * (k ** 0.5)


def measured_power(x: torch.Tensor) -> float:
    """Mean |x_k|^2 across all symbols — should read ~1.0 after power_norm."""
    return x.abs().pow(2).mean().item()


def bandwidth_ratio(depth: int = 8) -> float:
    """Complex channel uses per source sample, rho = FN/W — the number E0 turns on.

    Every conv uses `same` padding, so the encoder emits `depth` real values at each of
    the W = F*L input positions, i.e. depth/2 complex symbols per source sample:

        F rows x K symbols/row = 128 x (L*depth/2) = 128 x 512 = 65536 symbols
        rho = 65536 / 16384 = 4                                     (at depth = 8)

    which is exactly what 8-bit PCM + turbo 1/3 + 64-QAM costs (see deepscs.baseline).
    That the paper's stated 8-kernel channel coder lands on the same rho as its stated
    64-QAM benchmark is the evidence that `depth = 8` is the intended reading of N.

    Report it with every result: a number at an unstated rho is not a result."""
    assert depth % 2 == 0, "depth must be even (I/Q pairs)"
    return depth / 2


def assert_unit_power(depth: int = 8, f: int = 128, l: int = 128, tol: float = 1e-3) -> float:
    """Verify E|x|^2 = 1 empirically rather than trusting power_norm (hygiene item in
    docs/assumption_stripping.md section 4 — a normalization bug is free SNR)."""
    z = torch.randn(2, depth, f, l) * 7.3 + 2.1   # deliberately not already normalized
    p = measured_power(power_norm(to_iq(z)))
    assert abs(p - 1.0) < tol, f"transmit power is {p:.6f}, not 1.0"
    return p


def snr_db_to_sigma(snr_db: float) -> float:
    """Per-complex-dim noise std given unit signal power: sigma^2 = 1/(2*10^(SNR/10))."""
    return (10 ** (-snr_db / 10) / 2) ** 0.5


def _awgn_noise(shape, snr_db: float, device, dtype) -> torch.Tensor:
    sigma = snr_db_to_sigma(snr_db)
    real = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    imag = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(real, imag).to(dtype)


class AWGNChannel(torch.nn.Module):
    """y = x + w. Caches last tx/rx complex symbols for visualization."""

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        w = _awgn_noise(x.shape, snr_db, x.device, x.dtype)
        y = x + w
        self.last_tx, self.last_rx, self.last_h = x.detach(), y.detach(), None
        return y


class RayleighChannel(torch.nn.Module):
    """y = h*x + w, h ~ CN(0,1) i.i.d. per symbol (no line-of-sight). Perfect-CSI
    equalization x_hat = y/h."""

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        h_real = torch.randn(x.shape, device=x.device) / (2 ** 0.5)
        h_imag = torch.randn(x.shape, device=x.device) / (2 ** 0.5)
        h = torch.complex(h_real, h_imag).to(x.dtype)
        w = _awgn_noise(x.shape, snr_db, x.device, x.dtype)
        y = h * x + w
        x_hat = y / h  # perfect CSI equalization
        self.last_tx, self.last_rx, self.last_h = x.detach(), y.detach(), h.detach()
        return x_hat


class RicianChannel(torch.nn.Module):
    """y = h*x + w, h = LoS component (mean sqrt(K/(K+1))) + Rayleigh scatter
    (std sqrt(1/(K+1))). K -> inf recovers AWGN (h=1); K=0 recovers Rayleigh."""

    def __init__(self, k_factor: float = 1.0):
        super().__init__()
        self.k_factor = k_factor

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        k = self.k_factor
        los = (k / (k + 1)) ** 0.5
        scatter_std = (1 / (2 * (k + 1))) ** 0.5
        h_real = los + torch.randn(x.shape, device=x.device) * scatter_std
        h_imag = torch.randn(x.shape, device=x.device) * scatter_std
        h = torch.complex(h_real, h_imag).to(x.dtype)
        w = _awgn_noise(x.shape, snr_db, x.device, x.dtype)
        y = h * x + w
        x_hat = y / h
        self.last_tx, self.last_rx, self.last_h = x.detach(), y.detach(), h.detach()
        return x_hat


class ChannelLayer(torch.nn.Module):
    """Wraps to_iq -> power_norm -> {awgn,rayleigh,rician} -> from_iq so it can be
    dropped straight into DeepSC_S.forward(x, channel=this)."""

    _CHANNELS = {"awgn": AWGNChannel, "rayleigh": RayleighChannel}

    def __init__(self, kind: str = "awgn", snr_db: float = 10.0, rician_k: float = 1.0):
        super().__init__()
        assert kind in ("awgn", "rayleigh", "rician")
        self.kind = kind
        self.snr_db = snr_db
        self.impl = RicianChannel(rician_k) if kind == "rician" else self._CHANNELS[kind]()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        depth, f, l = z.shape[1], z.shape[2], z.shape[3]
        x = power_norm(to_iq(z))
        x_hat = self.impl(x, self.snr_db)
        return from_iq(x_hat, depth, f, l)
