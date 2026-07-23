# DeepSC-S Reproduction — Learning-First Plan

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](.python-version)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](pyproject.toml)

A working PyTorch reproduction of **DeepSC-S** (Weng, Qin & Li, 2021 —
[arXiv:2012.05369](https://arxiv.org/abs/2012.05369)), the semantic communication
system for speech. Built as an interactive lab notebook, not a train-and-print-a-number
script — every intermediate stage is visible, inspectable, and explained.

**[Read the staged write-up →](https://prashantrajbista.github.io/semantic-communication/)**
— what was learned at each of the five stages, with results.

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
curl -L -o clean_trainset_28spk_wav.zip \
  "https://datashare.ed.ac.uk/bitstreams/245452b6-6235-44b6-a6f9-e7eb19797769/download"
curl -L -o clean_testset_wav.zip \
  "https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download"
unzip clean_trainset_28spk_wav.zip && unzip clean_testset_wav.zip
```

(DataShare's older `/bitstream/handle/<id>/<name>.zip` URLs no longer serve the file
directly — these bitstream-UUID URLs are the current working ones, found via the
[handle page](https://datashare.ed.ac.uk/handle/10283/2791)'s download links.)

Several GB — not downloaded automatically by any notebook (`scripts/train.py` does
download it automatically, falling back to toy data if the download fails). Without
it, stage 5 still runs (and verifies) on the toy synthetic dataset; rerun after
downloading for paper-comparable numbers.

## Training script

`scripts/train.py` downloads the real dataset (if not already present under `data/`),
trains `DeepSC_S`, saves checkpoints to `checkpoints/`, and logs to Weights & Biases:

```bash
uv run python scripts/train.py --epochs 40 --channel awgn
```

W&B key resolution: put `WANDB_API_KEY=...` in a `.env` file in the project root (not
committed — see `.gitignore`), or export it in the environment. If neither is set,
training continues without logging in — W&B runs in disabled mode instead of
prompting or failing.

Key flags: `--channel {awgn,rayleigh,rician}`, `--subset-size`, `--epochs`,
`--batch-size`, `--lr`, `--depth` (compression knob), `--snr-low`/`--snr-high`.
Run `uv run python scripts/train.py --help` for the full list.

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

## License

[MIT](LICENSE)
