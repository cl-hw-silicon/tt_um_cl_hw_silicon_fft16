# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact model of the N-point real-input radix-2^2 single-path delay-feedback FFT (DIF).

W-bit two's-complement I/Q datapath, truncating /2 after every butterfly except the last, whose sum and
difference saturate to W bits instead (net gain 2/N; last_div2=True gives the earlier 1/N pipeline,
kept for reference only), Q1.TW twiddles (rounded, +2^TW saturated to 2^TW-1, exponent 0 bypassed),
saturation wherever a negation or a twiddle product can leave the W-bit range. One sample in, one bin out per (datapath) clock;
output slot t carries bin k = bitrev(t); latency N-1 datapath clocks.

Pipeline for N = 2^n (stage s = 1..n, delay D_s = N >> s):
    odd  s: BF2I   control bit  b_{n-s}
    even s: BF2II  control bit  b_{n-s}, -j applied to the incoming element when ~b_{n-s+1} & b_{n-s}
    after every even s < n: twiddle W_M^e, M = 2^(n-s+2),
        e = (k1 + 2*k2) * (cnt mod 2^(n-s)),  k1 = b_{n-s+1} ^ b_{n-s},  k2 = ~b_{n-s}
        (implemented as W_N^E with E = e * 2^(s-2) from one W_N table)
For N = 32: BF2I(16) BF2II(8) xW32 BF2I(4) BF2II(2) xW8 BF2I(1).

The RTL mirrors this file structurally: one function per primitive, delay lines as shift registers,
all control from one n-bit counter (supplied externally in the chip: the low bits of the symbol-phase
counter, see system_model.py). Integer arithmetic only.
"""
import numpy as np


# ------------------------------------------------------------------------------------------------
# helpers
def bitrev(k, n):
    """Reverse the n low bits of k."""
    r = 0
    for _ in range(n):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


def log2i(N):
    n = N.bit_length() - 1
    assert N == 1 << n and n >= 2, "N must be a power of two >= 4"
    return n


def sat(v, bits):
    """Saturate a Python int to the signed `bits` range."""
    hi = (1 << (bits - 1)) - 1
    return -hi - 1 if v < -hi - 1 else hi if v > hi else v


def round_half_up(x):
    return int(np.floor(x + 0.5))


def tw_table(N, TW=7):
    """W_N^E = cos(2*pi*E/N) - j*sin(2*pi*E/N) as (c, s) in Q1.TW, E = 0..N-1.
    Rounded half-up (no ties occur for N <= 1024), clamped to [-2^TW, 2^TW - 1]."""
    lo, hi = -(1 << TW), (1 << TW) - 1
    t = []
    for E in range(N):
        c = round_half_up(np.cos(2 * np.pi * E / N) * (1 << TW))
        s = round_half_up(-np.sin(2 * np.pi * E / N) * (1 << TW))
        t.append((max(lo, min(hi, c)), max(lo, min(hi, s))))
    return t


# ------------------------------------------------------------------------------------------------
# arithmetic primitives (W-bit signed ints in, W-bit signed ints out); 1:1 with the RTL
def bf_add(a, b):
    """(a + b) >> 1 per component, floor. W+1-bit sum, never overflows W bits after the shift."""
    return ((a[0] + b[0]) >> 1, (a[1] + b[1]) >> 1)


def bf_sub(a, b):
    return ((a[0] - b[0]) >> 1, (a[1] - b[1]) >> 1)


def bf_add_sat(a, b, W):
    """Last stage: a + b per component, no shift, saturated to W bits (W+1-bit sum)."""
    return (sat(a[0] + b[0], W), sat(a[1] + b[1], W))


def bf_sub_sat(a, b, W):
    return (sat(a[0] - b[0], W), sat(a[1] - b[1], W))


def neg_j(a, W):
    """-j * (I + jQ) = Q - jI; -(-2^(W-1)) saturates to 2^(W-1) - 1."""
    return (a[1], sat(-a[0], W))


def twiddle(a, E, table, W, TW):
    """(I + jQ) * (c + js), exact integer products, floor(>> TW), saturate to W bits.
    E == 0 is an exact bypass (no 127/128 gain). Any exact multiplier structure (4-mult, 3-mult
    Gauss with full-width intermediates, CSD shift-add) gives these bits."""
    if E == 0:
        return a
    c, s = table[E]
    re = (a[0] * c - a[1] * s) >> TW
    im = (a[0] * s + a[1] * c) >> TW
    return (sat(re, W), sat(im, W))


# ------------------------------------------------------------------------------------------------
def control(cnt, n):
    """Per-stage control derived from the n-bit counter value `cnt` (the counter value of the
    sample entering stage 1 in this clock). Returns lists indexed by stage s = 1..n (index 0 unused):
    sel[s] (butterfly phase), tj[s] (-j on incoming element, BF2II only), and twE[s] (W_N exponent
    applied after stage s, None if no twiddle after s)."""
    b = [(cnt >> i) & 1 for i in range(n)]
    sel = [None] * (n + 1)
    tj = [0] * (n + 1)
    twE = [None] * (n + 1)
    for s in range(1, n + 1):
        sel[s] = b[n - s]
        if s % 2 == 0:
            tj[s] = (1 - b[n - s + 1]) & b[n - s]
            if s < n:
                k1 = b[n - s + 1] ^ b[n - s]
                k2 = 1 - b[n - s]
                e = (k1 + 2 * k2) * (cnt & ((1 << (n - s)) - 1))
                twE[s] = (e << (s - 2)) & ((1 << n) - 1)
    return sel, tj, twE


def round_conv(v, k):
    """v / 2^k rounded to nearest, ties to even (convergent), for k >= 1."""
    q, r, half = v >> k, v & ((1 << k) - 1), 1 << (k - 1)
    return q + 1 if (r > half or (r == half and q & 1)) else q


class SDF:
    """Cycle-accurate N-point real-input R2^2 SDF FFT.

    step(x, cnt=None) is one datapath clock: x is the real input sample (W-bit signed int) present
    in this clock; returns (I, Q) of the output slot (cnt + 1) mod N, i.e. the combinational output
    in this clock, then advances all delay lines. cnt: the frame-counter value for this clock; if
    None the model keeps its own free-running counter (stand-alone use). Intermediates of the last
    step are in self.trace.

    Precision options (precision study; the chip uses last_div2=False, now the default; the defaults are the
    RTL, bit-exact):
      Wi        internal width of delay lines, butterflies, -j and twiddles (default W). The W-bit input is
                placed at the top (x << (Wi - W)); twiddles are Q1.(Wi-1) unless TW is given explicitly
                with Wi; the Wi-bit result is rounded to W bits (convergent, ties to even) and saturated.
      last_div2 False (default, the chip): the last butterfly stage adds/subtracts without the /2 and
                saturates to Wi bits (pipeline gain 2/N); True: the earlier 1/N pipeline (reference only).
    self.sat counts saturation events of the last stage (last_div2 False) and of the output rounding."""

    def __init__(self, N, W=8, TW=7, Wi=None, last_div2=False, TWi=None):
        self.N, self.W = N, W
        self.Wi = W if Wi is None else Wi
        self.TW = TW if Wi is None else (TWi if TWi is not None else self.Wi - 1)
        self.last_div2 = last_div2
        self.sat = 0
        self.n = log2i(N)
        self.table = tw_table(N, self.TW)
        # delay lines: index 0 = newest, -1 = oldest (= the stage's tail); stage 1 holds reals
        # (stored as (I, 0)); None = never written (X in RTL)
        self.d = {s: [None] * (N >> s) for s in range(1, self.n + 1)}
        self.cnt = 0
        self.trace = {}

    def _stage(self, s, x, sel, tj):
        if tj:
            x = neg_j(x, self.Wi)
        line = self.d[s]
        t = line[-1]
        if t is None:                                  # unflushed delay line (X in RTL)
            t = (0, 0)
            self.trace.setdefault('x_touched', []).append(s)
        if sel and s == self.n and not self.last_div2:   # full-gain last butterfly, saturated
            out, push = bf_add_sat(t, x, self.Wi), bf_sub_sat(t, x, self.Wi)
            raw = ((t[0] + x[0], t[1] + x[1]), (t[0] - x[0], t[1] - x[1]))
            self.sat += sum(a != b for p, q in zip(raw, (out, push)) for a, b in zip(p, q))
        elif sel:
            out, push = bf_add(t, x), bf_sub(t, x)
        else:
            out, push = t, x
        if s == 1:                                     # real-only stage: Q path does not exist
            assert out[1] == 0 and push[1] == 0
        self.d[s] = [push] + line[:-1]
        return out

    def step(self, x, cnt=None):
        W, Wi = self.W, self.Wi
        assert -(1 << (W - 1)) <= x < (1 << (W - 1))
        if cnt is None:
            cnt = self.cnt
        cnt &= self.N - 1
        sel, tj, twE = control(cnt, self.n)
        self.trace = {'cnt': cnt, 'slot': (cnt + 1) % self.N, 'sel': sel, 'tj': tj, 'twE': twE}
        v = (x << (Wi - W), 0)
        for s in range(1, self.n + 1):
            v = self._stage(s, v, sel[s], tj[s])
            self.trace[f'o{s}'] = v
            if twE[s] is not None:
                self.trace[f'twin{s}'] = v
                v = twiddle(v, twE[s], self.table, Wi, self.TW)
                self.trace[f'tw{s}'] = v
        if Wi > W:                                     # option (c): convergent rounding to the W-bit output
            r = tuple(round_conv(c, Wi - W) for c in v)
            v = tuple(sat(c, W) for c in r)
            self.sat += sum(a != b for a, b in zip(r, v))
        self.cnt = (cnt + 1) & (self.N - 1)
        return v


# ------------------------------------------------------------------------------------------------
OUT_GAIN = 2            # output = FFT * OUT_GAIN / N (1 for the earlier last_div2=True pipeline)


def reference(frame, N=None, gain=OUT_GAIN, W=8):
    """Ideal FFT * gain/N of one real frame in bit-reversed slot order (complex floats), each component clipped
    to the W-bit range: the chip saturates its last stage, and saturation is 1-Lipschitz, so the model's
    error against this reference is its fixed-point error alone. With gain 1 nothing can clip."""
    frame = np.asarray(frame, dtype=float)
    N = N or len(frame)
    n = log2i(N)
    X = np.fft.fft(frame) * gain / N
    lo, hi = -(1 << (W - 1)), (1 << (W - 1)) - 1
    X = np.clip(X.real, lo, hi) + 1j * np.clip(X.imag, lo, hi)
    return np.array([X[bitrev(t, n)] for t in range(N)])


def run_frames(samples, N, W=8, TW=7, **opt):
    """Feed a stream (frame-aligned at sample 0) through a fresh SDF with its own counter; return
    complex outputs aligned so that out[f*N + t] is frame f, slot t (latency N-1 removed)."""
    m = SDF(N, W, TW, **opt)
    outs = [complex(*m.step(int(s))) for s in samples]
    return np.array(outs[N - 1:])


# ------------------------------------------------------------------------------------------------
class FloatSDF:
    """Same structure and control, float arithmetic, no scaling, exact twiddles. Used to *prove*
    the control schedule: see schedule_proof()."""

    def __init__(self, N):
        self.N, self.n = N, log2i(N)
        self.d = {s: [0j] * (N >> s) for s in range(1, self.n + 1)}
        self.cnt = 0

    def step(self, x):
        sel, tj, twE = control(self.cnt, self.n)
        v = complex(x)
        for s in range(1, self.n + 1):
            if tj[s]:
                v = -1j * v
            t = self.d[s][-1]
            out, push = (t + v, t - v) if sel[s] else (t, v)
            self.d[s] = [push] + self.d[s][:-1]
            v = out
            if twE[s] is not None:
                v = v * np.exp(-2j * np.pi * twE[s] / self.N)
        self.cnt = (self.cnt + 1) % self.N
        return v


def schedule_proof(N, latency=None):
    """Prove the control schedule for N: the map from three consecutive input frames (f-1, f, f+1)
    to the N outputs attributed to frame f is linear; measure its matrix column by column with unit
    impulses (3N runs) and compare with [0 | F_bitrev | 0], F = unscaled DFT in bit-reversed order.
    Returns the max abs deviation (should be ~1e-12) and the latency used (N-1)."""
    n = log2i(N)
    lat = N - 1 if latency is None else latency
    F = np.array([[np.exp(-2j * np.pi * bitrev(t, n) * m / N) for m in range(N)] for t in range(N)])
    expected = np.hstack([np.zeros((N, N)), F, np.zeros((N, N))])
    got = np.zeros((N, 3 * N), dtype=complex)
    for col in range(3 * N):
        x = np.zeros(5 * N)
        x[N + col] = 1.0                       # frames: 0 = warm-up, 1 = f-1, 2 = f, 3 = f+1, 4 = drain
        m = FloatSDF(N)
        y = np.array([m.step(v) for v in x])
        got[:, col] = y[2 * N + lat: 3 * N + lat]
    return float(np.max(np.abs(got - expected))), lat
