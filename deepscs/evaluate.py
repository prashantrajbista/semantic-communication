"""Evaluation harness shared by the paper-figure scripts: load the test clips once, load
every seed of a checkpoint, run either system over an SNR sweep, score it.

Both systems are handed the *same* clips and the same channel classes, and rho is
asserted equal before anything is measured — the two things that make the numbers
comparable at all.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .audio import F, L, W, crop_or_pad, pesq_score, sdr
from .baseline import transmit as baseline_transmit
from .channel import ChannelLayer
from .data import SpeechDataset, ToyDataset
from .model import DeepSC_S

KINDS = ["awgn", "rayleigh", "rician"]

# E0 success criterion (assumption_stripping.md section 2): reproduction curves within
# ~1 dB SDR / ~0.2 PESQ of the published figures.
E0_TOL = {"SDR": 1.0, "PESQ": 0.2}
REFERENCE_CSV = Path(__file__).resolve().parent.parent / "docs/paper_reference.csv"


def load_reference(path: Path = REFERENCE_CSV) -> dict[tuple[str, str, str], dict[float, float]]:
    """The paper's own Fig. 5/6 values, keyed (metric, system, channel) -> {snr: value}.

    Read off the published figures by hand — arXiv:2012.05369 plots them and tabulates
    nothing, so there is no way to fetch them. Rows with a blank value are ignored, so a
    partly-filled file is usable; an absent file just means no comparison is made."""
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str], dict[float, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            metric = (row.get("metric") or "").strip()
            if metric.startswith("#") or not (row.get("value") or "").strip():
                continue
            key = (metric.upper(), row["system"].strip(), row["channel"].strip())
            out.setdefault(key, {})[float(row["snr_db"])] = float(row["value"])
    return out


def e0_verdict(metric: str, system: str, kind: str, snrs, values, reference) -> tuple[float, int]:
    """(largest |ours - paper| over the SNR points both have, number of points compared).
    Returns (nan, 0) when the reference has nothing for this curve."""
    ref = reference.get((metric.upper(), system, kind), {})
    deltas = [abs(v - ref[float(s)]) for s, v in zip(snrs, values)
              if float(s) in ref and np.isfinite(v)]
    return (max(deltas) if deltas else float("nan")), len(deltas)


def report_e0(results: dict, snrs, reference, with_pesq: bool = True) -> bool | None:
    """Print the per-curve gap against the published figures and return the E0 verdict:
    True/False once there is anything to compare against, None while the reference file is
    still empty. `results[kind]` is the (n_sdr, n_pesq, b_sdr, b_pesq) tuple fig05 builds."""
    if not reference:
        print(f"\nno published values in {REFERENCE_CSV.name} — E0 cannot be signed off. "
              "Read Figs. 5/6 off the paper and fill it in.")
        return None
    print(f"\nE0 check vs the published figures (tolerance "
          f"{E0_TOL['SDR']:.1f} dB SDR / {E0_TOL['PESQ']:.2f} PESQ)")
    ok, compared = True, 0
    for metric, idx in (("SDR", 0), ("PESQ", 1)):
        if metric == "PESQ" and not with_pesq:
            continue
        for kind in KINDS:
            for system, arr in (("deepsc", results[kind][idx]), ("benchmark", results[kind][idx + 2])):
                if arr is None:
                    continue
                curve = np.nanmean(arr, axis=0) if arr.ndim > 1 else arr
                gap, n = e0_verdict(metric, system, kind, snrs, curve, reference)
                if not n:
                    continue
                compared += 1
                passed = bool(gap <= E0_TOL[metric])
                ok &= passed
                print(f"  {metric:4s} {system:10s} {kind:9s} max gap {gap:6.3f} over {n} point(s)"
                      f"  {'ok' if passed else 'OUT OF TOLERANCE'}")
    if not compared:
        print("  reference file has no rows matching this SNR grid — nothing compared.")
        return None
    print(f"E0: {'PASS' if ok else 'FAIL'}")
    return ok


def load_clips(data_dir: Path, n_clips: int, seed: int = 0, allow_toy: bool = False) -> np.ndarray:
    """(n, W) float32 test clips from the real test set.

    The toy synthetic fallback renders perfectly plausible curves out of chirps, so it is
    a hard error here unless asked for explicitly — an E0 number measured on toy audio is
    not comparable to the paper, and that is exactly the kind of run worth catching before
    it costs a day rather than after."""
    test_dir = Path(data_dir) / "clean_testset_wav"
    if test_dir.exists() and any(test_dir.glob("*.wav")):
        ds = SpeechDataset(test_dir, train=False, seed=seed)
        print(f"test set: {test_dir} ({len(ds)} clips, using {min(n_clips, len(ds))})")
    elif not allow_toy:
        raise SystemExit(
            f"no test set under {test_dir} — E0 numbers measured on toy synthetic clips are\n"
            f"not comparable to the paper. Fetch it with:\n"
            f"    uv run python scripts/train.py --fetch-only\n"
            f"or pass --allow-toy if you are only smoke-testing the plumbing.")
    else:
        ds = ToyDataset(n=n_clips, length=W, seed=seed)
        print(f"no real test set under {test_dir} — TOY synthetic clips, NOT paper-comparable")
    clips = []
    for i in range(min(n_clips, len(ds))):
        c = ds[i]
        clips.append(crop_or_pad(c.numpy() if torch.is_tensor(c) else c, W))
    return np.stack(clips).astype(np.float32)


def load_seeds(ckpt_dir: Path, kind: str, depth: int = 8, n_blocks: int = 4,
               device="cpu") -> list[DeepSC_S]:
    """Every seed trained for one channel.

    Deliberately no fallback to the old un-suffixed `deepsc_s_<kind>_final.pt` name: those
    were trained with a Tanh output head and a per-batch random SNR, so they load without
    complaint (Tanh carries no parameters) and quietly produce numbers that are not the
    paper's setup. An E0 curve from one of them would be wrong in a way no assertion here
    could catch. Retrain instead."""
    ckpt_dir = Path(ckpt_dir)
    paths = sorted(ckpt_dir.glob(f"deepsc_s_{kind}_s*_final.pt"))
    models = []
    for path in paths:
        m = DeepSC_S(depth=depth, n_blocks=n_blocks).to(device)
        try:
            m.load_state_dict(torch.load(path, map_location=device))
        except RuntimeError as e:
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


def _self_check():
    """uv run python -m deepscs.evaluate — exercises the E0 comparison without a GPU,
    a checkpoint or the paper."""
    snrs = [0.0, 3.0, 6.0]
    ref = {("SDR", "deepsc", "awgn"): {0.0: 10.0, 3.0: 12.0},
           ("PESQ", "deepsc", "awgn"): {0.0: 2.0}}
    gap, n = e0_verdict("SDR", "deepsc", "awgn", snrs, [10.4, 11.3, 99.0], ref)
    assert (round(gap, 6), n) == (0.7, 2), (gap, n)   # 6 dB has no reference, must not count
    assert e0_verdict("SDR", "deepsc", "rician", snrs, [1, 2, 3], ref)[1] == 0
    # a curve with no overlapping SNR points must not be silently counted as passing
    assert e0_verdict("SDR", "deepsc", "awgn", [99.0], [10.0], ref)[1] == 0

    grid = np.array([[10.4, 11.3, 12.0]])
    results = {k: (grid, None, None, None) for k in KINDS}
    assert report_e0(results, snrs, ref, with_pesq=False) is True
    results["awgn"] = (grid + 2.0, None, None, None)   # 2 dB out on SDR
    assert report_e0(results, snrs, ref, with_pesq=False) is False
    assert report_e0(results, snrs, {}, with_pesq=False) is None

    from .channel import assert_unit_power
    assert_unit_power()
    print("evaluate self-check ok")


if __name__ == "__main__":
    _self_check()
