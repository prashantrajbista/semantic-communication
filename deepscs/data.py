"""Toy synthetic data (zero-download, stages 1-3) + real speech dataset (stage 5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import chirp
from torch.utils.data import Dataset

from .audio import SR, W, crop_or_pad, load_audio


def toy_signal(rng: np.random.Generator, length: int = W, sr: int = SR) -> np.ndarray:
    """One synthetic 'speech-like' clip: a chirp sweep + a few formant-like tones with
    an amplitude envelope + a couple of noise bursts (consonant-like transients)."""
    t = np.arange(length) / sr

    f0, f1 = rng.uniform(150, 400), rng.uniform(800, 2000)
    sweep = 0.3 * chirp(t, f0=f0, f1=f1, t1=t[-1], method="linear")

    formants = np.zeros(length)
    for f_center in (300, 900, 2500):
        f = f_center * rng.uniform(0.85, 1.15)
        env = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t + rng.uniform(0, 2 * np.pi)))
        formants += 0.15 * env * np.sin(2 * np.pi * f * t)

    bursts = np.zeros(length)
    n_bursts = rng.integers(2, 6)
    for _ in range(n_bursts):
        start = rng.integers(0, length - length // 20)
        burst_len = rng.integers(length // 100, length // 20)
        end = min(start + burst_len, length)
        window = np.hanning(end - start)
        bursts[start:end] += 0.2 * window * rng.normal(size=end - start)

    x = sweep + formants + bursts
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak * 0.9
    return x.astype(np.float32)


def toy_batch(n: int, length: int = W, seed: int | None = None) -> np.ndarray:
    """n synthetic clips, shape (n, length), float32 in [-1, 1]. Zero download, runs in seconds."""
    rng = np.random.default_rng(seed)
    return np.stack([toy_signal(rng, length) for _ in range(n)])


class ToyDataset(Dataset):
    """Fixed-size toy dataset for quick training loops on stages 1-3."""

    def __init__(self, n: int, length: int = W, seed: int | None = None):
        self.data = toy_batch(n, length, seed)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])


class SpeechDataset(Dataset):
    """Real speech clips (e.g. Edinburgh DataShare 10283/2791 clean_trainset_wav) resampled
    to 8 kHz. Train mode returns a random W-length crop per __getitem__ (matches paper);
    eval mode returns the whole clip uncropped, for use with audio.tile_frames/reassemble."""

    def __init__(self, wav_dir: str | Path, train: bool = True, seed: int | None = None):
        self.paths = sorted(Path(wav_dir).glob("*.wav"))
        if not self.paths:
            raise FileNotFoundError(f"no .wav files under {wav_dir}")
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        x = load_audio(str(self.paths[idx]))
        if self.train:
            return torch.from_numpy(crop_or_pad(x, W, self.rng))
        return torch.from_numpy(x)
