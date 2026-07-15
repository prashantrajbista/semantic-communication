# DeepSC-S Reproduction — Learning-First Plan

A working PyTorch reproduction of **DeepSC-S** (Weng, Qin & Li, 2021 —
[arXiv:2012.05369](https://arxiv.org/abs/2012.05369)), the semantic communication
system for speech. Built as an interactive lab notebook, not a train-and-print-a-number
script — every intermediate stage is visible, inspectable, and explained.

See [`docs/initial_plan.md`](docs/initial_plan.md) for the full design rationale,
open questions, and staged plan this repo implements.

## Setup

```bash
pyenv install 3.11.9   # already pinned via .python-version
uv sync                # installs deps from pyproject.toml/uv.lock into .venv
uv run jupyter lab     # open notebooks/
```

CPU is enough for stages 1-4 (toy synthetic data, small models). Stage 5 wants a real
speech dataset and benefits from a GPU/longer wall-clock (see below).

## Layout

- `deepscs/` — reusable, tested logic (no narrative): `audio.py` (framing/SDR),
  `blocks.py` (SE-ResNet), `model.py` (SemanticEncoder/Decoder + ChannelEnc/Dec +
  `DeepSC_S`), `channel.py` (AWGN/Rayleigh/Rician + power norm), `viz.py` (shared
  plots), `data.py` (toy generator + real dataset loader).
- `notebooks/` — the 5-stage narrative, in order:
  1. `01_audio_framing.ipynb` — load/resample/frame/deframe, lossless round trip.
  2. `02_autoencoder_no_channel.ipynb` — SemanticEncoder/Decoder trained with the
     channel disabled; SE attention visualization.
  3. `03_awgn_power_norm.ipynb` — power constraint + AWGN channel; SDR vs SNR.
  4. `04_fading_cross_channel.ipynb` — Rayleigh/Rician fading; one model tested
     across channels without retraining.
  5. `05_results_sdr_pesq.ipynb` — real-data training, SDR/PESQ curves, robustness
     table. Falls back to toy data automatically if the real dataset isn't present.

Every notebook is self-contained and has been executed end-to-end (outputs saved
in-place) — run cells top to bottom to reproduce.

## Real dataset (stage 5 only)

[Edinburgh DataShare 10283/2791](https://datashare.ed.ac.uk/handle/10283/2791)
(Valentini 2016). Use the **clean** sets — DeepSC-S transmits/reconstructs speech, it
doesn't denoise:

```bash
mkdir -p data && cd data
curl -L -o clean_trainset.zip \
  "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_wav.zip"
curl -L -o clean_testset.zip \
  "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"
unzip clean_trainset.zip && unzip clean_testset.zip
```

Several GB — not downloaded automatically by any notebook. Without it, stage 5 still
runs (and verifies) on the toy synthetic dataset; rerun after downloading for
paper-comparable numbers.

## Key design choices

Flagged as **CHOICE** throughout `docs/initial_plan.md` where the paper underspecifies
something and the official [TensorFlow repo](https://github.com/Zhenzi-Weng/DeepSC-S)
was used as ground truth instead: 5x5 SE-ResNet kernels, SE reduction ratio r=4,
feature depth D=32, channel-encoder output depth (compression knob) default 8, Adam
over the paper's SGD, Rician K=1 default.

## Scope

**In:** neural transceiver, end-to-end MSE training, AWGN + Rayleigh + Rician, SDR
(+ PESQ if installed), rich intermediate visualization.

**Out:** traditional PCM+Turbo/64-QAM baseline, exact numeric paper reproduction, real
RF hardware, estimated-CSI channels, full 10k-clip training.
