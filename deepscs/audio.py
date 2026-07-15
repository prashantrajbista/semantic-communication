"""Audio I/O + framing. Frame = plain reshape (no STFT/windowing) — that's the whole trick:
turn a 1-D signal into a 2-D "image" so 2-D CNNs apply. Round-trip must be bit-exact."""
from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa

SR = 8000
W = 16384  # frame window: one training example
F = 128    # rows
L = 128    # cols, F*L == W


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Load a wav file, force mono, resample to `sr`. Returns float32 in [-1, 1]."""
    x, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if orig_sr != sr:
        x = librosa.resample(x, orig_sr=orig_sr, target_sr=sr)
    return x.astype(np.float32)


def frame(x: np.ndarray, F: int = F, L: int = L) -> np.ndarray:
    """Reshape a length-(F*L) 1-D signal into an (F, L) matrix, row-major. No overlap."""
    w = F * L
    if x.shape[0] != w:
        raise ValueError(f"frame() expects exactly {w} samples, got {x.shape[0]}")
    return x.reshape(F, L)


def deframe(m: np.ndarray) -> np.ndarray:
    """Exact inverse of frame(): flatten (F, L) back to a 1-D signal."""
    return m.reshape(-1)


def crop_or_pad(x: np.ndarray, w: int = W, rng: np.random.Generator | None = None) -> np.ndarray:
    """Train-time length handling: random w-length crop; zero-pad if shorter than w."""
    if x.shape[0] < w:
        return np.pad(x, (0, w - x.shape[0]))
    if x.shape[0] == w:
        return x
    rng = rng or np.random.default_rng()
    start = rng.integers(0, x.shape[0] - w + 1)
    return x[start : start + w]


def tile_frames(x: np.ndarray, w: int = W) -> list[np.ndarray]:
    """Eval-time length handling: non-overlapping w-length tiles covering all of x,
    zero-padding only the final tile. Use `reassemble` to invert."""
    n = x.shape[0]
    n_tiles = int(np.ceil(n / w)) if n > 0 else 1
    padded_len = n_tiles * w
    xp = np.pad(x, (0, padded_len - n))
    return [xp[i * w : (i + 1) * w] for i in range(n_tiles)]


def reassemble(tiles: list[np.ndarray], orig_len: int) -> np.ndarray:
    """Exact inverse of tile_frames(): concatenate tiles, drop the zero-padding tail."""
    return np.concatenate(tiles)[:orig_len]


def sdr(s: np.ndarray, s_hat: np.ndarray) -> float:
    """Signal-to-Distortion Ratio in dB: 10*log10(||s||^2 / ||s - s_hat||^2)."""
    num = np.sum(s.astype(np.float64) ** 2)
    den = np.sum((s.astype(np.float64) - s_hat.astype(np.float64)) ** 2)
    if den == 0:
        return float("inf")
    return 10 * np.log10(num / den)
