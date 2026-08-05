#!/usr/bin/env bash
# Full E0 reproduction: train every checkpoint, then render every results figure the
# paper has (Figs. 4, 5 and 6). Everything lands under results/.
#
#   ./run_all.sh
#
# Resumable — a run whose final checkpoint already exists is skipped, so an interrupted
# run picks up where it left off. Delete results/checkpoints to force a retrain.
#
# Overridable by environment variable:
#   CHANNELS="awgn rayleigh rician"   which channels to train
#   SEEDS="0 1 2"                     seeds per channel
#   EPOCHS=1000                       training epochs (repo default)
#   RESULTS_DIR=./results             output root
#   N_CLIPS=16                        test clips per figure point
#   TRAIN_ARGS / FIG04_ARGS / FIG05_ARGS   extra flags passed through
#   SKIP_FIGURES=1                    train only, render nothing
#   SKIP_TRAIN=1                      render only, from checkpoints already present
#
# A single run uses ~8 GB and does not saturate a big GPU, so fan training out and
# render once at the end (all invocations share RESULTS_DIR; the skip logic keeps
# them from colliding):
#   for ch in awgn rayleigh rician; do SKIP_FIGURES=1 CHANNELS=$ch ./run_all.sh & done
#   wait && SKIP_TRAIN=1 ./run_all.sh
#
# Smoke test the plumbing before committing days of GPU time:
#   EPOCHS=1 SEEDS=0 N_CLIPS=2 ./run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS_DIR:-$ROOT/results}"
CKPT="$RESULTS/checkpoints"
FIGS="$RESULTS/figures"
LOGS="$RESULTS/logs"

read -r -a CHANNELS <<< "${CHANNELS:-awgn rayleigh rician}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-1000}"
N_CLIPS="${N_CLIPS:-16}"

if command -v uv >/dev/null 2>&1; then
    PY=(uv run python)
elif [ -x "$ROOT/.venv/bin/python" ]; then
    PY=("$ROOT/.venv/bin/python")
else
    echo "no uv and no .venv/bin/python — run 'uv sync' first" >&2
    exit 1
fi

mkdir -p "$CKPT" "$FIGS" "$LOGS"
cd "$ROOT"

echo "results  -> $RESULTS"
echo "channels -> ${CHANNELS[*]}   seeds -> ${SEEDS[*]}   epochs -> $EPOCHS"
echo

# ------------------------------------------------------------------ 1. train
# scripts/train.py fetches the dataset itself on the first run.
if [ -n "${SKIP_TRAIN:-}" ]; then echo "SKIP_TRAIN set — using existing checkpoints"; fi
for ch in "${CHANNELS[@]}"; do
    [ -n "${SKIP_TRAIN:-}" ] && break
    for s in "${SEEDS[@]}"; do
        final="$CKPT/deepsc_s_${ch}_s${s}_final.pt"
        if [ -f "$final" ]; then
            echo "== skip  $ch seed $s (already have $(basename "$final"))"
            continue
        fi
        echo "== train $ch seed $s  -> $LOGS/train_${ch}_s${s}.log"
        "${PY[@]}" scripts/train.py \
            --channel "$ch" --seed "$s" --epochs "$EPOCHS" \
            --checkpoint-dir "$CKPT" ${TRAIN_ARGS:-} 2>&1 | tee "$LOGS/train_${ch}_s${s}.log"
    done
done

# The figure scripts fall back to toy synthetic clips when the real test set is missing,
# which yields plausible-looking curves with meaningless numbers. Say so, loudly.
if [ ! -d "$ROOT/data/clean_testset_wav" ]; then
    echo
    echo "WARNING: data/clean_testset_wav is missing — figures will be rendered from toy"
    echo "         synthetic clips and are NOT comparable to the paper. See README."
fi

# ------------------------------------------------------------------ 2. figures
if [ -n "${SKIP_FIGURES:-}" ]; then
    echo
    echo "SKIP_FIGURES set — training done, no figures rendered."
    exit 0
fi

echo
echo "== Fig. 4 (MSE vs SNR)  -> $LOGS/fig04.log"
"${PY[@]}" scripts/fig04_mse_vs_snr.py \
    --checkpoint-dir "$CKPT" --out "$FIGS/fig04_mse_vs_snr.png" \
    --n-clips "$N_CLIPS" ${FIG04_ARGS:-} 2>&1 | tee "$LOGS/fig04.log"

echo
echo "== Figs. 5 and 6 (SDR / PESQ vs SNR)  -> $LOGS/fig05_06.log"
"${PY[@]}" scripts/fig05_sdr_pesq.py \
    --checkpoint-dir "$CKPT" --out-dir "$FIGS" \
    --n-clips "$N_CLIPS" ${FIG05_ARGS:-} 2>&1 | tee "$LOGS/fig05_06.log"

# ------------------------------------------------------------------ 3. summary
echo
echo "done. figures:"
ls -1 "$FIGS"
echo
echo "numeric tables are at the end of $LOGS/fig04.log and $LOGS/fig05_06.log"
