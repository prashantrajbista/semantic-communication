"""The paper's traditional benchmark: 8-bit A-law PCM (ITU-T G.711) -> turbo rate 1/3
-> 64-QAM -> the same channel the neural system sees -> soft demod -> turbo decode ->
G.711 decode (arXiv:2012.05369 Sec. V — "PCM + Turbo + 64-QAM").

Nothing here is learned; the whole chain is fixed by the standards, so there is no
baseline training step. It is imported by scripts/fig05_sdr_pesq.py.

Bandwidth accounting — the reason the paper picks 64-QAM at all:

    16384 samples x 8 bit A-law  = 131072 information bits
    turbo rate 1/3               = 393216 coded bits
    64-QAM at 6 bit/symbol       =  65536 channel symbols

    rho = 65536 / 16384 = 4 complex channel uses per source sample

which is exactly DeepSC-S's rho (deepscs.channel.bandwidth_ratio). Matched rho is what
makes the two curves comparable at all; `assert_matched_rho` enforces it.

The channel itself is not reimplemented here — deepscs.channel's layers are called
directly so both systems see the identical fading/noise model and the identical
perfect-CSI equalization.
"""
from __future__ import annotations

import numpy as np
import torch

from .channel import AWGNChannel, RayleighChannel, RicianChannel

# ---------------------------------------------------------------- G.711 A-law

# ITU-T G.711 segment ends, on the 13-bit magnitude (pcm >> 3). Straight from the
# reference g711.c; searchsorted reproduces its linear `search()`.
_SEG_AEND = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def alaw_encode(x: np.ndarray) -> np.ndarray:
    """float32 waveform in [-1, 1] -> (n,) uint8 A-law codewords. Exact G.711, vectorized."""
    pcm = np.clip(np.rint(np.asarray(x, dtype=np.float64) * 32767.0), -32768, 32767).astype(np.int32)
    pcm = pcm >> 3
    neg = pcm < 0
    mask = np.where(neg, 0x55, 0xD5).astype(np.int32)
    mag = np.where(neg, -pcm - 1, pcm)

    seg = np.searchsorted(_SEG_AEND, mag, side="left").astype(np.int32)
    shift = np.where(seg < 2, 1, seg)
    aval = (np.minimum(seg, 7) << 4) | ((mag >> shift) & 0xF)
    aval = np.where(seg >= 8, 0x7F, aval)  # overflow clamp, per g711.c
    return (aval ^ mask).astype(np.uint8)


def alaw_decode(a: np.ndarray) -> np.ndarray:
    """(n,) uint8 A-law codewords -> float32 waveform in [-1, 1]. Exact inverse of the
    G.711 quantizer (lossy by construction — that is the codec, not a bug)."""
    a = np.asarray(a, dtype=np.int32) ^ 0x55
    base = (a & 0xF) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, base + 8, (base + 0x108) << np.maximum(seg - 1, 0))
    pcm = np.where(a & 0x80, t, -t)
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


# ---------------------------------------------------------------- turbo, rate 1/3

# LTE constituent RSC: feedback g0 = 1 + D^2 + D^3 (octal 13), parity g1 = 1 + D + D^3
# (octal 15). Memory 3 -> 8 states, state index = r1*4 + r2*2 + r3.
def _trellis():
    nxt = np.zeros((8, 2), dtype=np.int64)
    par = np.zeros((8, 2), dtype=np.int64)
    for s in range(8):
        r1, r2, r3 = (s >> 2) & 1, (s >> 1) & 1, s & 1
        for u in (0, 1):
            a = u ^ r2 ^ r3
            par[s, u] = a ^ r1 ^ r3
            nxt[s, u] = (a << 2) | (r1 << 1) | r2
    prev_s = np.zeros((8, 2), dtype=np.int64)
    prev_u = np.zeros((8, 2), dtype=np.int64)
    fill = np.zeros(8, dtype=np.int64)
    for s in range(8):
        for u in (0, 1):
            ns = nxt[s, u]
            prev_s[ns, fill[ns]] = s
            prev_u[ns, fill[ns]] = u
            fill[ns] += 1
    return nxt, par, prev_s, prev_u


NEXT, PARITY, PREV_S, PREV_U = _trellis()


def _rsc_encode(u: np.ndarray) -> np.ndarray:
    """(B, K) info bits -> (B, K) parity bits. State runs independently per row."""
    b, k = u.shape
    state = np.zeros(b, dtype=np.int64)
    p = np.empty((b, k), dtype=np.uint8)
    for i in range(k):
        ui = u[:, i].astype(np.int64)
        p[:, i] = PARITY[state, ui]
        state = NEXT[state, ui]
    return p


def _bcjr(ls: np.ndarray, lp: np.ndarray, la: np.ndarray) -> np.ndarray:
    """Max-log-MAP APP decoder, vectorized over the block axis. LLR convention
    L = log P(bit=0)/P(bit=1). Returns the full a-posteriori LLR, shape (B, K).

    ponytail: no trellis termination — beta is initialized uniformly instead of at the
    zero state. Costs a fraction of a dB at the tail of each block and keeps the rate
    exactly 1/3, which is what makes rho match the neural system exactly. Add tail bits
    (and the matching rho correction) only if the block length ever gets short."""
    lsys = (ls + la).astype(np.float32)
    lp = lp.astype(np.float32)
    b, k = lsys.shape

    sign_u = np.array([1.0, -1.0], dtype=np.float32)          # (1 - 2u)
    sign_p = (1.0 - 2.0 * PARITY).astype(np.float32)          # (8, 2)
    # gamma[:, k, s, u] = 0.5 * ((1-2u)(Ls+La)_k + (1-2p(s,u)) Lp_k)
    g = 0.5 * (lsys[:, :, None, None] * sign_u[None, None, None, :]
               + lp[:, :, None, None] * sign_p[None, None, :, :])

    alpha = np.full((b, 8), -1e9, dtype=np.float32)
    alpha[:, 0] = 0.0                                          # encoder starts at state 0
    a_hist = np.empty((b, k, 8), dtype=np.float32)
    for i in range(k):
        a_hist[:, i] = alpha
        cand = alpha[:, PREV_S] + g[:, i][:, PREV_S, PREV_U]   # (B, 8, 2)
        alpha = cand.max(axis=2)
        alpha -= alpha.max(axis=1, keepdims=True)              # keep it from drifting

    beta = np.zeros((b, 8), dtype=np.float32)
    out = np.empty((b, k), dtype=np.float32)
    for i in range(k - 1, -1, -1):
        gk = g[:, i]                                           # (B, 8, 2)
        fwd = gk + beta[:, NEXT]                               # (B, 8, 2)
        m = a_hist[:, i][:, :, None] + fwd
        out[:, i] = m[:, :, 0].max(axis=1) - m[:, :, 1].max(axis=1)
        beta = fwd.max(axis=2)
        beta -= beta.max(axis=1, keepdims=True)
    return out


def turbo_encode(u: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """(B, K) info bits -> (B, K, 3) [systematic, parity1, parity2]. Rate exactly 1/3."""
    p1 = _rsc_encode(u)
    p2 = _rsc_encode(u[:, perm])
    return np.stack([u, p1, p2], axis=-1)


def turbo_decode(llr: np.ndarray, perm: np.ndarray, n_iter: int = 6) -> np.ndarray:
    """(B, K, 3) channel LLRs -> (B, K) hard bits. Standard two-decoder extrinsic loop."""
    ls, lp1, lp2 = llr[..., 0], llr[..., 1], llr[..., 2]
    deperm = np.argsort(perm)
    ls_i = ls[:, perm]
    la1 = np.zeros_like(ls)
    l2 = None
    for _ in range(n_iter):
        l1 = _bcjr(ls, lp1, la1)
        la2 = (l1 - ls - la1)[:, perm]
        l2 = _bcjr(ls_i, lp2, la2)
        la1 = (l2 - ls_i - la2)[:, deperm]
    return (l2[:, deperm] < 0).astype(np.uint8)   # L = log P(0)/P(1) -> negative means 1


# ---------------------------------------------------------------- 64-QAM

def _pam8():
    """Gray-labelled 8-PAM: LVL[label] is the amplitude for that 3-bit label."""
    amps = np.arange(-7, 8, 2, dtype=np.float64)          # -7 .. 7, natural order
    lvl = np.empty(8)
    for i, a in enumerate(amps):
        lvl[i ^ (i >> 1)] = a                              # Gray label for index i
    return lvl / np.sqrt(42.0)                             # E|I+jQ|^2 = 1


PAM8 = _pam8()
_BIT_W = np.array([4, 2, 1])                               # 3 bits, MSB first
_IDX0 = [np.where((np.arange(8) & w) == 0)[0] for w in _BIT_W]
_IDX1 = [np.where((np.arange(8) & w) != 0)[0] for w in _BIT_W]


def qam64_modulate(bits: np.ndarray) -> np.ndarray:
    """(6n,) bits -> (n,) complex symbols, unit average power. First 3 bits -> I, last 3 -> Q."""
    t = bits.reshape(-1, 6).astype(np.int64)
    li = t[:, :3] @ _BIT_W
    lq = t[:, 3:] @ _BIT_W
    return PAM8[li] + 1j * PAM8[lq]


def qam64_demodulate(y: np.ndarray, n0: np.ndarray | float) -> np.ndarray:
    """(n,) equalized symbols + per-symbol complex noise variance -> (6n,) LLRs,
    L = log P(bit=0)/P(bit=1). Max-log, done per axis since Gray labelling is separable."""
    n0 = np.broadcast_to(np.asarray(n0, dtype=np.float64), y.shape)
    out = np.empty((y.shape[0], 6))
    for off, axis in ((0, y.real), (3, y.imag)):
        d2 = (axis[:, None] - PAM8[None, :]) ** 2
        for j in range(3):
            out[:, off + j] = (d2[:, _IDX1[j]].min(axis=1) - d2[:, _IDX0[j]].min(axis=1)) / n0
    return out.reshape(-1)


# ---------------------------------------------------------------- channel + end-to-end

_CHANNELS = {"awgn": AWGNChannel, "rayleigh": RayleighChannel, "rician": RicianChannel}


def apply_channel(sym: np.ndarray, kind: str, snr_db: float, rician_k: float = 1.0,
                  fade_block: int = 512):
    """Push symbols through deepscs.channel's layer (same code path as the neural system)
    and report the post-equalization noise variance the demodulator needs.

    Symbols are laid out as rows of `fade_block` so the benchmark sees the *same* block
    fading the neural system does — one h per row, held across the row. 512 is what
    DeepSC-S gets at the paper's dimensions (H*W/2 = 32*32/2).

    Fading equalizes as x_hat = y/h, so the noise is scaled by 1/|h|^2 too — feeding that
    into the LLRs is what keeps the benchmark honest under fading."""
    n = sym.shape[0]
    pad = (-n) % fade_block
    impl = RicianChannel(rician_k) if kind == "rician" else _CHANNELS[kind]()
    x = torch.from_numpy(np.pad(sym, (0, pad)).astype(np.complex64)).view(1, -1, fade_block)
    y = impl(x, snr_db)
    n0 = np.full(x.shape, 10.0 ** (-snr_db / 10.0))           # variance per complex symbol
    if impl.last_h is not None:
        n0 = n0 / impl.last_h.abs().numpy().astype(np.float64) ** 2
    return y.numpy().reshape(-1)[:n].astype(np.complex128), n0.reshape(-1)[:n]


def transmit(x: np.ndarray, kind: str, snr_db: float, rician_k: float = 1.0,
             block_len: int = 512, n_iter: int = 6, seed: int = 0) -> np.ndarray:
    """Full benchmark chain on one waveform. Returns the reconstruction, same length as x."""
    n = x.shape[0]
    bits = np.unpackbits(alaw_encode(x))
    pad = (-bits.shape[0]) % block_len
    u = np.pad(bits, (0, pad)).reshape(-1, block_len)

    perm = np.random.default_rng(seed).permutation(block_len)
    coded = turbo_encode(u, perm).reshape(-1)                  # (B*K*3,)
    qpad = (-coded.shape[0]) % 6
    sym = qam64_modulate(np.pad(coded, (0, qpad)))

    y, n0 = apply_channel(sym, kind, snr_db, rician_k)
    llr = qam64_demodulate(y, n0)
    llr = llr[: coded.shape[0]].reshape(u.shape[0], block_len, 3)

    bits_hat = turbo_decode(llr, perm, n_iter).reshape(-1)[: bits.shape[0]]
    return alaw_decode(np.packbits(bits_hat))[:n]


def bandwidth_ratio(n_samples: int = 16384, block_len: int = 512) -> float:
    """Complex channel uses per source sample for the benchmark chain."""
    bits = n_samples * 8
    blocks = int(np.ceil(bits / block_len))
    coded = blocks * block_len * 3
    return int(np.ceil(coded / 6)) / n_samples


def assert_matched_rho(n_samples: int = 16384, block_len: int = 512, filters: int = 128):
    """Every comparison in the paper is void unless both systems use the same number of
    channel symbols per source sample. Fail loudly rather than publish an unfair curve."""
    from .channel import bandwidth_ratio as neural_rho

    rho_b, rho_n = bandwidth_ratio(n_samples, block_len), neural_rho(filters)
    assert abs(rho_b - rho_n) < 1e-9, f"rho mismatch: benchmark {rho_b} vs DeepSC-S {rho_n}"
    return rho_b


# ---------------------------------------------------------------- self-check

def _demo():
    from .audio import sdr

    rng = np.random.default_rng(0)
    t = np.arange(8000) / 8000.0
    x = (0.6 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1400 * t)).astype(np.float32)

    # 1. A-law matches the stdlib G.711 codec bit for bit (audioop is gone in 3.13, so
    #    this arm just skips there — the numpy path above is the one actually used).
    try:
        import audioop
        pcm = np.clip(np.rint(x * 32767.0), -32768, 32767).astype("<i2")
        ref = np.frombuffer(audioop.lin2alaw(pcm.tobytes(), 2), dtype=np.uint8)
        assert np.array_equal(alaw_encode(x), ref), "A-law encode differs from ITU-T G.711"
        ref_lin = np.frombuffer(audioop.alaw2lin(ref.tobytes(), 2), dtype="<i2")
        assert np.array_equal((alaw_decode(ref) * 32768.0).astype("<i2"), ref_lin), "A-law decode differs"
        print("ok  A-law matches stdlib G.711 exactly")
    except ImportError:
        print("--  audioop unavailable (py>=3.13), skipped the G.711 cross-check")

    q = alaw_decode(alaw_encode(x))
    assert sdr(x, q) > 30, f"A-law quantization SDR too low: {sdr(x, q):.1f} dB"
    print(f"ok  A-law round trip SDR = {sdr(x, q):.1f} dB (source-coding floor)")

    # 2. turbo + 64-QAM recover every bit on a clean channel, and the LLR sign convention
    #    is right (a wrong sign silently decodes to the complement and still "works").
    u = rng.integers(0, 2, size=(4, 512)).astype(np.uint8)
    perm = rng.permutation(512)
    coded = turbo_encode(u, perm).reshape(-1)
    llr = qam64_demodulate(*apply_channel(qam64_modulate(coded), "awgn", 25.0))
    hat = turbo_decode(llr.reshape(4, 512, 3), perm)
    assert np.array_equal(hat, u), f"turbo BER at 25 dB = {(hat != u).mean():.4f}, expected 0"
    print("ok  turbo 1/3 + 64-QAM: zero bit errors at 25 dB AWGN")

    # 3. rho matches DeepSC-S — the check the whole comparison rests on.
    print(f"ok  matched rho = {assert_matched_rho()} channel uses per source sample")

    # 4. end to end: usable at high SNR, and the turbo cliff shows up at low SNR.
    torch.manual_seed(0)
    hi = sdr(x, transmit(x, "awgn", 20.0))
    torch.manual_seed(0)
    lo = sdr(x, transmit(x, "awgn", 0.0))
    assert hi > 25, f"benchmark broken at 20 dB: SDR {hi:.1f} dB"
    assert lo < hi - 10, f"no cliff at 0 dB (SDR {lo:.1f} vs {hi:.1f} dB) — check the LLR scaling"
    print(f"ok  end to end AWGN: SDR {hi:.1f} dB at 20 dB SNR, {lo:.1f} dB at 0 dB (cliff)")


if __name__ == "__main__":
    _demo()
