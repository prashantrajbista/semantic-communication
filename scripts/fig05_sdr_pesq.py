"""Paper Figs. 5 and 6 — SDR versus SNR and PESQ versus SNR, one panel per channel,
DeepSC-S against the traditional PCM + turbo 1/3 + 64-QAM benchmark
(arXiv:2012.05369 Sec. V-C).

Both systems run at the same bandwidth ratio rho = 4 complex channel uses per source
sample; the script refuses to plot anything if that stops being true. Neural curves are
the mean over the trained seeds, shaded by min/max across seeds.

Usage:
    uv run python scripts/fig05_sdr_pesq.py
    uv run python scripts/fig05_sdr_pesq.py --n-clips 32 --no-baseline    # neural only, fast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepscs.baseline import assert_matched_rho
from deepscs.channel import assert_unit_power
from deepscs.evaluate import (KINDS, load_clips, load_reference, load_seeds, report_e0,
                              sweep_baseline, sweep_neural)

ROOT = Path(__file__).resolve().parent.parent
NEURAL_C, BASE_C, REF_C = "#2f6fed", "#b8672a", "#6b6b6b"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default=str(ROOT / "docs/paper_figures"))
    p.add_argument("--n-clips", type=int, default=16)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=18.0)
    p.add_argument("--snr-step", type=float, default=3.0)
    p.add_argument("--rician-k", type=float, default=1.0)
    p.add_argument("--depth", type=int, default=8, help="channel-encoder output depth; rho = depth/2")
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--turbo-iters", type=int, default=6)
    p.add_argument("--no-baseline", action="store_true", help="skip the benchmark chain (much faster)")
    p.add_argument("--no-pesq", action="store_true")
    p.add_argument("--allow-toy", action="store_true",
                   help="run on synthetic clips when the real test set is missing (smoke tests only)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rho = assert_matched_rho(depth=args.depth)
    print(f"matched rho = {rho} complex channel uses per source sample")
    print(f"measured transmit power E|x|^2 = {assert_unit_power(args.depth):.4f}")
    reference = load_reference()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snrs = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    clips = load_clips(Path(args.data_dir), args.n_clips, args.seed + 1, args.allow_toy)
    with_pesq = not args.no_pesq

    results = {}
    for kind in KINDS:
        models = load_seeds(Path(args.checkpoint_dir), kind, args.depth, args.n_blocks, device)
        if not models:
            print(f"no checkpoint for {kind} — train it with: "
                  f"uv run python scripts/train.py --channel {kind}")
        else:
            print(f"{kind}: {len(models)} seed(s)")
        n_sdr, n_pesq = (sweep_neural(models, clips, kind, snrs, args.rician_k, device,
                                      with_pesq, args.seed) if models else (None, None))
        b_sdr, b_pesq = ((None, None) if args.no_baseline else
                         sweep_baseline(clips, kind, snrs, args.rician_k, args.turbo_iters,
                                        with_pesq, args.seed))
        results[kind] = (n_sdr, n_pesq, b_sdr, b_pesq)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fig_no, metric, ylabel, ni, bi in (("05", "SDR", "SDR (dB)", 0, 2),
                                           ("06", "PESQ", "PESQ", 1, 3)):
        if metric == "PESQ" and not with_pesq:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, kind, panel in zip(axes, KINDS, "abc"):
            neural, base = results[kind][ni], results[kind][bi]
            if neural is not None and np.isfinite(neural).any():
                ax.plot(snrs, np.nanmean(neural, axis=0), marker="o", markersize=4,
                        color=NEURAL_C, label="DeepSC-S")
                if neural.shape[0] > 1:
                    ax.fill_between(snrs, np.nanmin(neural, axis=0), np.nanmax(neural, axis=0),
                                    color=NEURAL_C, alpha=0.18, lw=0)
            if base is not None and np.isfinite(base).any():
                ax.plot(snrs, base, marker="s", markersize=4, color=BASE_C,
                        label="PCM + turbo 1/3 + 64-QAM")
            # E0's deliverable is the reproduction overlaid on the paper's own curve, so
            # the gap is visible and not just a number in a table.
            for system, style in (("deepsc", "--"), ("benchmark", ":")):
                ref = reference.get((metric, system, kind), {})
                if ref:
                    xs = sorted(ref)
                    ax.plot(xs, [ref[x] for x in xs], style, marker="x", markersize=5,
                            color=REF_C, lw=1.2,
                            label=f"paper, {'DeepSC-S' if system == 'deepsc' else 'benchmark'}")
            ax.set_xlabel("SNR (dB)")
            ax.set_title(f"({panel}) {kind} channel")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[0].legend()
        fig.suptitle(f"Fig. {int(fig_no)} — {metric} versus SNR at rho = {rho} "
                     f"(DeepSC-S reproduction)")
        fig.tight_layout()
        path = out_dir / f"fig{fig_no}_{metric.lower()}_vs_snr.png"
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")

    hdr = " ".join(f"{s:>8.0f} dB" for s in snrs)
    for metric, ni, bi in (("SDR (dB)", 0, 2), ("PESQ", 1, 3)):
        if metric == "PESQ" and not with_pesq:
            continue
        print(f"\n{metric}")
        print(f"{'':34s}{hdr}")
        for kind in KINDS:
            for label, arr in (("DeepSC-S", results[kind][ni]), ("benchmark", results[kind][bi])):
                if arr is None:
                    continue
                row = np.nanmean(arr, axis=0) if arr.ndim > 1 else arr
                print(f"{kind:10s} {label:22s}" + " ".join(f"{v:11.3f}" for v in row))

    # DeepSC-S must improve with SNR on the channel it was trained for — the cheapest
    # check that a checkpoint actually loaded and the channel is wired the right way round.
    for kind in KINDS:
        n = results[kind][0]
        if n is None:
            continue
        m = np.nanmean(n, axis=0)
        if len(m) < 2:
            continue  # single-point sweep, nothing to compare
        assert m[-1] > m[0], f"{kind}: DeepSC-S SDR did not rise with SNR ({m[0]:.2f} -> {m[-1]:.2f})"
    print("\ncheck: DeepSC-S SDR rises with SNR on every channel.")

    report_e0(results, snrs, reference, with_pesq)


if __name__ == "__main__":
    main()
