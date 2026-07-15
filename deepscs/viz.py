"""Shared plot helpers so every notebook renders intermediate state the same way."""
from __future__ import annotations

import librosa
import librosa.display
import numpy as np

from .audio import SR


def plot_waveform(ax, x: np.ndarray, sr: int = SR, title: str | None = None):
    t = np.arange(len(x)) / sr
    ax.plot(t, x, linewidth=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    if title:
        ax.set_title(title)


def plot_spectrogram(ax, x: np.ndarray, sr: int = SR, title: str | None = None, n_mels: int = 64):
    mel = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    if title:
        ax.set_title(title)
    return img


def plot_framed_image(ax, m: np.ndarray, title: str | None = None):
    im = ax.imshow(m, aspect="auto", origin="upper", cmap="magma")
    ax.set_xlabel("col (L)")
    ax.set_ylabel("row (F)")
    if title:
        ax.set_title(title)
    return im


def plot_constellation(ax, x: np.ndarray, title: str | None = None, alpha: float = 0.3):
    """x: complex array (or real array shape [..., 2] treated as I/Q)."""
    if np.iscomplexobj(x):
        i, q = x.real, x.imag
    else:
        i, q = x[..., 0], x[..., 1]
    ax.scatter(i.ravel(), q.ravel(), s=4, alpha=alpha)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)


def plot_se_weights(ax, weights: np.ndarray, title: str | None = None):
    ax.bar(np.arange(len(weights)), weights)
    ax.set_xlabel("channel")
    ax.set_ylabel("SE weight")
    if title:
        ax.set_title(title)
