# DeepSC-S — Faithful Reproduction (E0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](.python-version)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](pyproject.toml)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoints-yellow)](https://huggingface.co/prashantrajbista/deepsc-s)

A PyTorch reproduction of **DeepSC-S** (Weng, Qin & Li — [arXiv:2012.05369](https://arxiv.org/abs/2012.05369)),
built to reproduce the paper's numbers rather than its qualitative story: the neural
transceiver **and** the traditional PCM + turbo + 64-QAM benchmark it is compared against,
at a matched bandwidth ratio, over the paper's three channels.

This is stage **E0** of [`docs/assumption_stripping.md`](docs/assumption_stripping.md) —
the faithful baseline everything downstream inherits from.

**[Read the staged write-up →](https://prashantrajbista.github.io/semantic-communication/)**
· **[Pretrained checkpoints →](https://huggingface.co/prashantrajbista/deepsc-s)**

> The write-up page and the published checkpoints predate this branch — they document an
> earlier, structurally different model. **Those checkpoints will not load here**; the
> figure scripts skip them with a message. Retrain with the commands below.

## Ground truth: the repo, not the paper table

The authors' own TensorFlow code ([Zhenzi-Weng/DeepSC-S](https://github.com/Zhenzi-Weng/DeepSC-S))
disagrees with the paper's own configuration table on several counts. This
implementation follows the **code**, because that is what produced the published numbers:

| | Paper text | Official repo (used here) |
|---|---|---|
| Encoder front-end | not stated | 2 × 5×5 conv, **stride 2** → 4× downsample in both dims (32 then 128 filters) |
| SE-ResNet modules | 4 | **5** per side |
| SE-ResNet width | 32 kernels | out_dim **128**; 4 branches × 128 filters, 1×1 transition, r = 4 |
| Channel encoder | 1 CNN, 8 kernels, ReLU | 1 conv + BN, **128** filters, **no activation** |
| Channel decoder | 1 CNN, 8 kernels | 1 conv + BN, 128 filters, + ReLU |
| Decoder tail | not stated | 2 × stride-2 **transposed** conv (128, 32) + ReLU |
| Output layer | 1 CNN, 1 kernel, no activation | 1×1 conv to 1 channel, no activation ✔ agrees |
| Waveform | not stated | per-example mean/var normalize in, denormalize out |
| Optimizer | SGD | **RMSprop** |
| Learning rate | 0.001 | **5e-4** |
| Epochs / batch | not stated | 1000 / 32 |
| Training SNR | fixed 8 dB | fixed 8 dB ✔ agrees |
| Fading `h` | not stated | **one draw per row, held across 512 symbols** (block fading), Rician K = 1 |

Flags exist for the paper-text readings (`--optimizer sgd --lr 1e-3 --n-blocks 4
--chan-filters 8`), but the authors never trained that configuration.

## The bandwidth ratio ρ

Every comparison in the paper is void unless both systems spend the same number of
channel symbols per source sample. The paper says 64-QAM was picked "to make the number
of transmitted symbols in the traditional systems the same as that in DeepSC-S" but never
states the value, so it is derived here and asserted in code
(`deepscs.baseline.assert_matched_rho`):

| | |
|---|---|
| DeepSC-S | `(B,1,128,128)` → 4× downsample → `(B,128,32,32)` → 128 rows × 512 complex symbols = 65536 per 16384 samples → **ρ = filters / (2·4²) = 4** |
| Benchmark | 16384 × 8 bit A-law = 131072 bits → turbo 1/3 = 393216 bits → 64-QAM at 6 bit/symbol = 65536 symbols → **ρ = 4** |

The two stride-2 convs shrink the feature map 16× while the channel width grows 1 → 128,
so the symbol budget is unchanged. Both figure scripts print ρ and refuse to run if the
two ever diverge. Note this is a 4× bandwidth *expansion*, not compression — worth stating
plainly, since it is easy to read DeepSC-S as a compression result.

## Fading is block, not per-symbol

The repo draws `h` with shape `[B, C, 1]` and broadcasts it over all 512 symbols in that
row. Drawing it i.i.d. per symbol instead hands both systems full diversity the paper
never had — and it flatters the benchmark enormously: with per-symbol fading the turbo
chain clears Rayleigh at ~12 dB, with block fading a single deep fade destroys a whole
codeword and it stalls around 10 dB SDR. The block model is what `deepscs/channel.py`
implements, for both systems.

## Setup

```bash
pyenv install 3.11.9   # already pinned via .python-version
uv sync                # installs deps from pyproject.toml/uv.lock into .venv
```

Data (several GB — the paper's set, Edinburgh DataShare 10283/2791, Valentini 2016; use
the **clean** sets, DeepSC-S transmits speech, it does not denoise):

```bash
mkdir -p data && cd data
curl -L -o clean_trainset_28spk_wav.zip \
  "https://datashare.ed.ac.uk/bitstreams/245452b6-6235-44b6-a6f9-e7eb19797769/download"
curl -L -o clean_testset_wav.zip \
  "https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download"
unzip clean_trainset_28spk_wav.zip && unzip clean_testset_wav.zip
```

`scripts/train.py` will fetch this itself if `data/` is empty, and falls back to a toy
synthetic set if the download fails (curve shapes hold; absolute values are not
paper-comparable without the real set). On cloud IPs DataShare's WAF tends to 403 —
pass `--source hf` to pull the same audio from a Hugging Face mirror instead.

## Reproducing the paper

### 1. Train the reference checkpoint set

Defaults are the repo's: RMSprop at lr 5e-4, MSE on the waveform, **fixed 8 dB training
SNR**, batch 32, 1000 epochs, full ~10k-clip training set. Three channels × three seeds:

```bash
for ch in awgn rayleigh rician; do
  for s in 0 1 2; do
    uv run python scripts/train.py --channel $ch --seed $s
  done
done
```

Writes `checkpoints/deepsc_s_{channel}_s{seed}_final.pt`. The model is ~18.6M parameters
and needs a GPU — MPS is not usable (its autograd breaks on the channel layer's
`torch.complex` ops), so this is CUDA or a very long CPU wait.

### 2. Figure 4 — MSE vs SNR, train channel × test channel

```bash
uv run python scripts/fig04_mse_vs_snr.py
```

### 3. Figures 5 and 6 — SDR and PESQ vs SNR, DeepSC-S vs the benchmark

```bash
uv run python scripts/fig05_sdr_pesq.py
```

Neural curves are the mean across seeds, shaded min-to-max. `--no-baseline` skips the
turbo chain when you only want the neural half (much faster); `--n-clips` and
`--turbo-iters` trade accuracy for wall-clock.

**Success criterion for E0:** within ~1 dB SDR / ~0.2 PESQ of the published figures. If
you can't get there, stop and resolve it — every later experiment inherits the
discrepancy.

## The traditional benchmark

`deepscs/baseline.py` implements the paper's comparison system end to end. Nothing in it
is learned, so there is no baseline training step:

| Stage | Implementation |
|---|---|
| Source coding | 8-bit A-law PCM, exact ITU-T G.711 segment codec, vectorized in numpy |
| Channel coding | Turbo rate 1/3 — two LTE constituent RSCs (feedback `1 + D² + D³`, parity `1 + D + D³`, 8 states), random interleaver, max-log-MAP BCJR, 6 iterations |
| Modulation | 64-QAM, Gray-labelled, unit average power, max-log soft demapping |
| Channel | `deepscs.channel`'s own layers — the *same* code path the neural system uses |

Two details worth knowing:

- **Fading LLRs are scaled by 1/\|h\|².** Perfect-CSI equalization `x̂ = y/h` amplifies
  noise as well as signal; feeding that per-symbol variance into the demapper is what
  keeps the benchmark from being quietly handicapped under Rayleigh/Rician.
- **No trellis termination.** Beta is initialized uniformly instead of at the zero state.
  Costs a fraction of a dB at each block tail and keeps the rate at exactly 1/3 — which
  is what makes ρ match the neural system exactly. Marked in the source.

The turbo decoder is vectorized over the block axis (256 blocks of 512 bits per clip), so
a clip decodes in about a second. An off-the-shelf pure-Python BCJR is roughly three
orders of magnitude slower and was not usable here.

Self-check — cross-validates A-law against the stdlib G.711 codec bit for bit, confirms
zero BER at 25 dB, and asserts matched ρ:

```bash
uv run python -m deepscs.baseline
```

## Layout

- `deepscs/` — logic, no narrative:
  `audio.py` (framing, SDR, PESQ), `blocks.py` (SE-ResNet), `model.py` (DeepSC-S),
  `channel.py` (AWGN/Rayleigh/Rician, power norm, ρ), `baseline.py` (G.711 + turbo +
  64-QAM), `evaluate.py` (shared sweep harness), `data.py`, `viz.py`.
- `scripts/` — `train.py`, `fig04_mse_vs_snr.py`, `fig05_sdr_pesq.py`.
- `notebooks/` — the five-stage learning narrative that built this up
  (framing → autoencoder → AWGN → fading → results). Explanatory, not the reproduction
  path; use the scripts above for that.

## Choices neither the paper nor the repo pins down

| Item | Choice | Note |
| --- | --- | --- |
| Clips ≠ W = 16384 | random crop while training; non-overlapping tile + reassemble at eval; zero-pad if short | Repo pre-slices into TFRecords instead |
| Rician K-factor | 1.0 (`--rician-k`) | Matches the repo's hardcoded value; its LoS sits at 45°, which perfect-CSI equalization makes irrelevant |
| AWGN / Rayleigh variants | same layer, K → ∞ and K = 0 | Repo only ships the Rician case; the paper reports all three |
| Multi-branch SE-ResNet split | one conv to `cardinality × branch_filters` | Exactly equal to 4 concatenated full convs — BN is per-channel, ReLU elementwise |
| Turbo interleaver, block length | random permutation, 512 bits | Paper says "turbo, rate 1/3" and nothing more |
| Turbo iterations | 6 (`--turbo-iters`) | — |
| Trellis termination | none, uniform β init | Keeps the rate at exactly 1/3, which is what makes ρ match |
| PCM law | A-law (vs μ-law) | Paper says "PCM"; A-law is the ITU-T G.711 default outside North America |
| PESQ mode | narrowband (P.862) | Correct for 8 kHz |
| SGD momentum | 0.0 (`--momentum`) | Only used if you switch off RMSprop |

Deviating is one flag each — `--optimizer adam` trains faster at small scale,
`--snr-low 0 --snr-high 20` randomizes the training SNR (that is experiment E3, not E0),
`--subset-size 2000` shrinks the training set.

## Scope

**In:** the neural transceiver, the full traditional benchmark, AWGN + Rayleigh + Rician,
SDR + PESQ, matched ρ, multi-seed.

**Out:** everything after E0 in [`docs/assumption_stripping.md`](docs/assumption_stripping.md)
— estimated CSI, digital/constellation-constrained transmitters, hardened channel models,
out-of-distribution content, modern baselines (Opus/EVS/LDPC).

## License

[MIT](LICENSE)
