"""Evaluation harness shared by the paper-figure scripts: load the test clips once, load
every seed of a checkpoint, run either system over an SNR sweep, score it.

Both systems are handed the *same* clips and the same channel classes, and rho is
asserted equal before anything is measured — the two things that make the numbers
comparable at all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .audio import F, L, W, crop_or_pad, pesq_score, sdr
from .baseline import transmit as baseline_transmit
from .channel import ChannelLayer
from .data import SpeechDataset, ToyDataset
from .model import DeepSC_S

KINDS = ["awgn", "rayleigh", "rician"]


def load_clips(data_dir: Path, n_clips: int, seed: int = 0) -> np.ndarray:
    """(n, W) float32 test clips — real speech if downloaded, else the toy fallback."""
    test_dir = Path(data_dir) / "clean_testset_wav"
    if test_dir.exists() and any(test_dir.glob("*.wav")):
        ds = SpeechDataset(test_dir, train=False, seed=seed)
        print(f"test set: {test_dir} ({len(ds)} clips, using {min(n_clips, len(ds))})")
    else:
        ds = ToyDataset(n=n_clips, length=W, seed=seed)
        print(f"no real test set under {test_dir} — falling back to toy synthetic clips")
    clips = []
    for i in range(min(n_clips, len(ds))):
        c = ds[i]
        clips.append(crop_or_pad(c.numpy() if torch.is_tensor(c) else c, W))
    return np.stack(clips).astype(np.float32)


def load_seeds(ckpt_dir: Path, kind: str, filters: int = 128, n_blocks: int = 5,
               device="cpu") -> list[DeepSC_S]:
    """Every seed trained for one channel. Falls back to the un-suffixed single-seed
    name written before the seed sweep existed."""
    ckpt_dir = Path(ckpt_dir)
    paths = sorted(ckpt_dir.glob(f"deepsc_s_{kind}_s*_final.pt"))
    if not paths:
        legacy = ckpt_dir / f"deepsc_s_{kind}_final.pt"
        paths = [legacy] if legacy.exists() else []
    models = []
    for path in paths:
        m = DeepSC_S(filters=filters, n_blocks=n_blocks).to(device)
        try:
            m.load_state_dict(torch.load(path, map_location=device))
        except RuntimeError as e:
            # checkpoints from before the architecture was matched to the official repo
            print(f"skipping {path.name}: incompatible with the current model ({str(e)[:80]}...)")
            continue
        m.eval()
        models.append(m)
    return models


def neural_reconstruct(model: DeepSC_S, clips: np.ndarray, kind: str, snr_db: float,
                       rician_k: float = 1.0, device="cpu") -> np.ndarray:
    ch = ChannelLayer(kind, snr_db=snr_db, rician_k=rician_k)
    x = torch.from_numpy(clips).view(-1, 1, F, L).to(device)
    with torch.no_grad():
        return model(x, channel=ch).view(-1, W).cpu().numpy()


def baseline_reconstruct(clips: np.ndarray, kind: str, snr_db: float,
                         rician_k: float = 1.0, n_iter: int = 6, seed: int = 0) -> np.ndarray:
    return np.stack([baseline_transmit(c, kind, snr_db, rician_k, n_iter=n_iter, seed=seed)
                     for c in clips])


def score(clips: np.ndarray, recon: np.ndarray, with_pesq: bool = True) -> tuple[float, float]:
    """(mean SDR in dB, mean PESQ). PESQ is NaN-tolerant — P.862 refuses some degenerate
    clips and dropping them beats poisoning the whole average."""
    sdrs = [sdr(a, b) for a, b in zip(clips, recon)]
    if not with_pesq:
        return float(np.mean(sdrs)), float("nan")
    pesqs = [pesq_score(a, b) for a, b in zip(clips, recon)]
    pesqs = [p for p in pesqs if np.isfinite(p)]
    return float(np.mean(sdrs)), float(np.mean(pesqs)) if pesqs else float("nan")


def sweep_neural(models: list[DeepSC_S], clips: np.ndarray, kind: str, snrs, rician_k=1.0,
                 device="cpu", with_pesq=True, seed=0):
    """(n_seeds, n_snr) SDR and PESQ grids. Every seed sees identical noise draws at a
    given SNR, so seed spread is model variance rather than noise luck."""
    out_sdr = np.full((len(models), len(snrs)), np.nan)
    out_pesq = np.full((len(models), len(snrs)), np.nan)
    for j, snr in enumerate(snrs):
        for i, m in enumerate(models):
            torch.manual_seed(seed + 1000 * j)
            recon = neural_reconstruct(m, clips, kind, float(snr), rician_k, device)
            out_sdr[i, j], out_pesq[i, j] = score(clips, recon, with_pesq)
    return out_sdr, out_pesq


def sweep_baseline(clips: np.ndarray, kind: str, snrs, rician_k=1.0, n_iter=6,
                   with_pesq=True, seed=0):
    """(n_snr,) SDR and PESQ. Nothing is learned here, so there is no seed axis — the
    only randomness is the channel, matched to the neural sweep's draws."""
    s, p = np.full(len(snrs), np.nan), np.full(len(snrs), np.nan)
    for j, snr in enumerate(snrs):
        torch.manual_seed(seed + 1000 * j)
        recon = baseline_reconstruct(clips, kind, float(snr), rician_k, n_iter, seed)
        s[j], p[j] = score(clips, recon, with_pesq)
    return s, p
