"""Paper Fig. 4 — MSE loss versus SNR, one panel per *testing* channel, one curve per
*training* channel (arXiv:2012.05369, Sec. V-B robustness experiment).

Loads the three final checkpoints written by scripts/train.py (awgn / rayleigh / rician)
and evaluates each one on all three channels over an SNR sweep. MSE is measured on the
framed tensor, which equals sample-domain MSE because framing is a lossless reshape.

Uses data/clean_testset_wav if present, else falls back to the toy synthetic set (curve
shapes hold; absolute values are not paper-comparable without the real set).

Usage:
    uv run python scripts/fig04_mse_vs_snr.py
    uv run python scripts/fig04_mse_vs_snr.py --n-clips 32 --repeats 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as func

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepscs.audio import F, L
from deepscs.channel import ChannelLayer, bandwidth_ratio
from deepscs.evaluate import KINDS, load_clips, load_seeds

ROOT = Path(__file__).resolve().parent.parent
# same channel colors as index.html / the rest of the write-up
COLORS = {"awgn": "#2f6fed", "rayleigh": "#b8672a", "rician": "#23886b"}


def load_models(ckpt_dir: Path, filters: int, n_blocks: int, device) -> dict[str, list]:
    """{train channel: [one model per seed]}."""
    models = {}
    for kind in KINDS:
        seeds = load_seeds(ckpt_dir, kind, filters, n_blocks, device)
        if not seeds:
            print(f"no checkpoint for {kind} — skipping that curve "
                  f"(train it with: uv run python scripts/train.py --channel {kind})")
            continue
        models[kind] = seeds
    if not models:
        raise SystemExit(f"no checkpoints found in {ckpt_dir}")
    return models


def sweep(models, x, snrs, repeats, rician_k, seed, device):
    """{(train_kind, test_kind): [mse per snr]}, averaged over seeds. Every model sees
    identical noise draws at a given (test channel, SNR, repeat), so the curves are
    directly comparable."""
    out = {}
    for test_kind in KINDS:
        ch = ChannelLayer(test_kind, snr_db=float(snrs[0]), rician_k=rician_k)
        for train_kind, seed_models in models.items():
            mses = []
            for i, snr in enumerate(snrs):
                ch.snr_db = float(snr)
                acc = 0.0
                for r in range(repeats):
                    for model in seed_models:
                        torch.manual_seed(seed + 1000 * i + r)
                        with torch.no_grad():
                            acc += func.mse_loss(model(x, channel=ch), x).item()
                mses.append(acc / (repeats * len(seed_models)))
            out[(train_kind, test_kind)] = mses
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out", default=str(ROOT / "docs/paper_figures/fig04_mse_vs_snr.png"))
    p.add_argument("--n-clips", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3, help="noise realizations averaged per point")
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=18.0)
    p.add_argument("--snr-step", type=float, default=3.0)
    p.add_argument("--rician-k", type=float, default=1.0)
    p.add_argument("--chan-filters", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snrs = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    print(f"rho = {bandwidth_ratio(args.chan_filters)} complex channel uses per source sample")

    clips = load_clips(Path(args.data_dir), args.n_clips, args.seed + 1)
    x = torch.from_numpy(clips).view(-1, 1, F, L).to(device)
    models = load_models(Path(args.checkpoint_dir), args.chan_filters, args.n_blocks, device)
    results = sweep(models, x, snrs, args.repeats, args.rician_k, args.seed, device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, test_kind, panel in zip(axes, KINDS, "abc"):
        for train_kind in models:
            ax.plot(snrs, results[(train_kind, test_kind)], marker="o", markersize=4,
                    color=COLORS[train_kind], label=f"trained on {train_kind}")
        ax.set_yscale("log")
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"({panel}) testing over {test_kind} channels")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()
    fig.suptitle("Fig. 4 — MSE loss versus SNR (DeepSC-S reproduction)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    print("\nMSE  (rows: train channel -> test channel)")
    print(f"{'':24s}" + " ".join(f"{s:>9.0f} dB" for s in snrs))
    for (train_kind, test_kind), mses in results.items():
        print(f"train={train_kind:9s} test={test_kind:9s}" + " ".join(f"{v:12.5f}" for v in mses))

    grid = np.array(list(results.values()))
    assert np.isfinite(grid).all() and (grid > 0).all(), "non-finite/degenerate MSE"
    for kind in models:  # matched train/test channel must improve from lowest to highest SNR
        m = results[(kind, kind)]
        assert m[-1] < m[0], f"train={kind} test={kind}: MSE did not drop as SNR rose ({m[0]:.5f} -> {m[-1]:.5f})"
    print("check: all MSE finite, matched-channel MSE falls with SNR.")


if __name__ == "__main__":
    main()
