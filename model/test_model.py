# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest for the bit-exact FFT model (fft_model.py). Reference: np.fft.fft * 2/N, bit-reversed order,
each component clipped to 8 bits (fm.reference); see model/README.md."""
import numpy as np
import pytest
import fft_model as fm

# Error bounds (complex magnitude, LSB at the 8-bit output) vs the clipped FFT * 2/N, re-derived for the 2/N gain as
# before: measured over >= 2000 random full-scale frames (N = 16 / 32 / 64: max 3.78 / 4.47 / 5.25, rms 1.194 / 1.396 /
# 1.505) plus ~10 % headroom. The last stage no longer truncates but doubles the earlier stages' error, so the bounds are
# ~1.6x the 1/N ones (2.8/1.0, 3.2/1.1, 3.6/1.2) while the error relative to the signal drops (rms / 2: 0.60 vs 0.875).
BOUNDS = {16: (4.5, 1.35), 32: (5.2, 1.55), 64: (5.7, 1.7)}


def frames_error(x, N):
    y = fm.run_frames(x, N)
    nfr = len(x) // N
    return np.concatenate([y[f * N:(f + 1) * N] - fm.reference(x[f * N:(f + 1) * N]) for f in range(nfr - 1)])


# ---- schedule: proven for every N, not just tested -------------------------------------------------
@pytest.mark.parametrize("N", [4, 8, 16, 32, 64, 128])
def test_schedule_proof(N):
    dev, lat = fm.schedule_proof(N)
    assert lat == N - 1
    assert dev < 1e-9, f"control schedule wrong for N={N}: deviation {dev}"


def test_schedule_negative_control():
    """The proof has teeth: a wrong k1 (no XOR) must fail."""
    orig = fm.control
    def bad(cnt, n):
        sel, tj, twE = orig(cnt, n)
        b = [(cnt >> i) & 1 for i in range(n)]
        e = (b[n - 1] + 2 * (1 - b[n - 2])) * (cnt & ((1 << (n - 2)) - 1))
        twE[2] = e & ((1 << n) - 1)
        return sel, tj, twE
    fm.control = bad
    try:
        dev, _ = fm.schedule_proof(32)
    finally:
        fm.control = orig
    assert dev > 1.0


# ---- (a) random frames ---------------------------------------------------------------------------------
@pytest.mark.parametrize("N", [16, 32, 64])
def test_random_frames(N):
    rng = np.random.default_rng(N)
    nfr = 2000 if N <= 32 else 600
    E = frames_error(rng.integers(-128, 128, size=nfr * N), N)
    mx, rms = np.abs(E).max(), np.sqrt(np.mean(np.abs(E) ** 2))
    print(f"N={N}: max {mx:.2f} rms {rms:.3f} LSB over {E.size} bins")
    bmax, brms = BOUNDS[N]
    assert mx <= bmax and rms <= brms


def test_gain_and_order_N32():
    """Least-squares complex gain per slot == 2/N against the raw np.fft.fft (independent of fm.reference's
    gain) and latency N-1."""
    N = 32
    rng = np.random.default_rng(7)
    nfr = 800
    x = rng.integers(-128, 128, size=nfr * N)
    m = fm.SDF(N)
    y = np.array([complex(*m.step(int(v))) for v in x])
    best = min(range(2 * N), key=lambda L: np.abs(y[2 * N + L: 3 * N + L] - fm.reference(x[2 * N:3 * N])).max())
    assert best == N - 1
    n = fm.log2i(N)
    F = np.array([np.fft.fft(x[f * N:(f + 1) * N])[[fm.bitrev(t, n) for t in range(N)]] for f in range(2, nfr - 2)])
    Y = np.array([y[f * N + N - 1: (f + 1) * N + N - 1] for f in range(2, nfr - 2)])
    g = np.array([np.vdot(F[:, t], Y[:, t]) / np.vdot(F[:, t], F[:, t]) for t in range(N)]) * N
    assert np.all(np.abs(np.abs(g) - 2) < 0.02) and np.all(np.abs(np.angle(g)) < np.radians(0.5))


# ---- (b) tone per bin ---------------------------------------------------------------------------------
@pytest.mark.parametrize("N", [16, 32])
@pytest.mark.parametrize("amp", [1, 64, 127])
def test_tones(N, amp):
    n = fm.log2i(N)
    bmax = BOUNDS[N][0]
    nidx = np.arange(N)
    for k in range(N):
        tone = np.clip(np.round(amp * np.cos(2 * np.pi * k * nidx / N + 0.3)), -128, 127).astype(int)
        x = np.tile(tone, 3)
        y = fm.run_frames(x, N)[N:2 * N]
        ref = fm.reference(tone)
        assert np.abs(y - ref).max() <= bmax, (k, amp)
        if amp >= 64 and 0 < k < N // 2:
            even = [t for t in range(0, N, 2)]
            top = max(even, key=lambda t: abs(y[t]))
            assert top == fm.bitrev(k, n)                      # the bin lights its bit-reversed even slot
            assert abs(abs(y[top]) - amp) <= bmax               # |X| = A (gain 2/N)
            odd = fm.bitrev(N - k, n)                          # conjugate on the odd slot
            assert odd % 2 == 1 and abs(y[odd] - np.conj(y[top])) <= 2 * bmax


# ---- (c) structured inputs ------------------------------------------------------------------------------
STRUCT = {
    'zeros': lambda N: np.zeros(N, int),
    'dc+127': lambda N: np.full(N, 127),
    'dc-128': lambda N: np.full(N, -128),
    'nyquist': lambda N: np.array([127, -128] * (N // 2)),
    'impulse+127': lambda N: np.eye(1, N, 0, dtype=int)[0] * 127,
    'impulse-128': lambda N: np.eye(1, N, 0, dtype=int)[0] * -128,
    'impulse-late': lambda N: np.eye(1, N, N - 1, dtype=int)[0] * -128,
    'ramp': lambda N: np.round(np.linspace(-128, 127, N)).astype(int),
    'two-tone': lambda N: np.clip(np.round(60 * np.cos(2 * np.pi * 3 * np.arange(N) / N) +
                                            60 * np.sin(2 * np.pi * 5 * np.arange(N) / N)), -128, 127).astype(int),
    'random+-extremes': lambda N: np.random.default_rng(3).choice([-128, 127], size=N),
}


@pytest.mark.parametrize("N", [16, 32])
@pytest.mark.parametrize("name", list(STRUCT))
def test_structured(N, name):
    f = STRUCT[name](N)
    y = fm.run_frames(np.tile(f, 3), N)[N:2 * N]
    assert np.abs(y - fm.reference(f)).max() <= BOUNDS[N][0], name


@pytest.mark.parametrize("N", [16, 32])
def test_exact_corners(N):
    """Exact results where truncation cannot act: DC lands in slot 0 as 2 * value, saturated."""
    for val in (-128, -65, -64, -1, 0, 1, 63, 64, 127):
        y = fm.run_frames(np.full(3 * N, val), N)[N:2 * N]
        assert y[0] == max(-128, min(127, 2 * val)) and np.all(y[1:] == 0), val


# ---- (d) primitives, exhaustive with corners -----------------------------------------------------------
W, TW = 8, 7
ALL = np.arange(-128, 128)


def test_last_stage_saturating_add_sub_exhaustive():
    """Last stage: a +- b, no shift, saturated to 8 bits; every 8-bit pair, corners forced first."""
    corners = [(-128, -128, -128, 0), (127, 127, 127, 0), (-128, 127, -1, -128), (127, -128, -1, 127),
               (-1, -128, -128, 127), (0, -128, -128, 127), (64, 64, 127, 0), (-64, -64, -128, 0), (63, 64, 127, -1)]
    for a, b, s_exp, d_exp in corners:
        assert fm.bf_add_sat((a, b), (b, a), 8) == (s_exp, s_exp) and fm.bf_sub_sat((a, 0), (b, 0), 8)[0] == d_exp
    A, B = np.meshgrid(ALL, ALL)
    S = np.clip(A + B, -128, 127)
    D = np.clip(A - B, -128, 127)
    for a, b, s, d in zip(A.ravel(), B.ravel(), S.ravel(), D.ravel()):
        assert fm.bf_add_sat((int(a), int(b)), (int(b), int(a)), 8) == (s, s)
        assert fm.bf_sub_sat((int(a), int(b)), (int(b), int(a)), 8) == (d, max(-128, min(127, int(b - a))))
        assert np.sign(fm.bf_add_sat((int(a), 0), (int(b), 0), 8)[0]) == np.sign(a + b)      # no wrap


def test_butterfly_add_sub_exhaustive():
    A, B = np.meshgrid(ALL, ALL)
    for a, b in zip(A.ravel(), B.ravel()):
        s = fm.bf_add((int(a), 0), (int(b), 0))[0]
        d = fm.bf_sub((int(a), 0), (int(b), 0))[0]
        assert s == (a + b) // 2 and d == (a - b) // 2
        assert -128 <= s <= 127 and -128 <= d <= 127


def test_neg_j_exhaustive():
    for i in ALL:
        for q in (-128, -1, 0, 1, 127):
            r = fm.neg_j((int(i), q), W)
            assert r == (q, 127 if i == -128 else -int(i))


def used_exponents(N):
    n = fm.log2i(N)
    s = set()
    for cnt in range(N):
        for E in fm.control(cnt, n)[2]:
            if E is not None:
                s.add(E)
    return sorted(s)


@pytest.mark.parametrize("N", [16, 32])
def test_twiddle_table(N):
    t = fm.tw_table(N, TW)
    for E in range(N):
        c, s = t[E]
        ec, es = np.cos(2 * np.pi * E / N) * 128, -np.sin(2 * np.pi * E / N) * 128
        assert abs(c - min(ec, 127)) <= 0.5 + 1e-9 and abs(s - min(es, 127)) <= 0.5 + 1e-9
    used = used_exponents(N)
    # no used exponent needs the +128 -> +127 clip (only E = 0, bypassed, and E = 3N/4, unused)
    assert all(127 not in (abs(t[E][0]) if t[E][0] > 0 else 0, abs(t[E][1]) if t[E][1] > 0 else 0)
               or E == 0 for E in used)
    if N == 32:
        assert used == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 21]
        assert set(E for cnt in range(32) for s, E in enumerate(fm.control(cnt, 5)[2]) if s == 4 and E is not None) == {0, 4, 8, 12}


@pytest.mark.parametrize("N", [16, 32])
def test_twiddle_exhaustive(N):
    """Every used exponent x every (I, Q) in [-128, 127]^2: the model equals an independent numpy
    floor/saturate implementation, and is within 1 LSB + coefficient error of the exact product."""
    t = fm.tw_table(N, TW)
    I, Q = np.meshgrid(ALL, ALL)
    I, Q = I.ravel(), Q.ravel()
    for E in used_exponents(N):
        c, s = t[E]
        if E == 0:
            ref_re, ref_im = I, Q
        else:
            ref_re = np.clip(np.floor_divide(I * c - Q * s, 128), -128, 127)
            ref_im = np.clip(np.floor_divide(I * s + Q * c, 128), -128, 127)
        got = np.array([fm.twiddle((int(i), int(q)), E, t, W, TW) for i, q in zip(I, Q)])
        assert np.array_equal(got[:, 0], ref_re) and np.array_equal(got[:, 1], ref_im), E
        exact = (I + 1j * Q) * np.exp(-2j * np.pi * E / N)
        inside = (np.abs(exact.real) < 127) & (np.abs(exact.imag) < 127)
        err = np.abs(got[inside, 0] + 1j * got[inside, 1] - exact[inside])
        assert err.max() < 1.0 * np.sqrt(2) + (np.abs(I) + np.abs(Q)).max() * 0.5 / 128 * np.sqrt(2)


def test_twiddle_inputs_in_range_N32():
    """Twiddle inputs reached in the pipeline: report the range and that saturation only happens
    at the -128 corner (I = -128 into -j-like exponents)."""
    N = 32
    rng = np.random.default_rng(11)
    m = fm.SDF(N)
    sat_events = 0
    xs = np.concatenate([rng.integers(-128, 128, 20000), rng.choice([-128, 127], 20000)])
    t = fm.tw_table(N, TW)
    for v in xs:
        m.step(int(v))
        for s in (2, 4):
            if f'twin{s}' in m.trace:
                a, E = m.trace[f'twin{s}'], m.trace['twE'][s]
                if E:
                    c, sn = t[E]
                    raw = ((a[0] * c - a[1] * sn) >> 7, (a[0] * sn + a[1] * c) >> 7)
                    if raw != m.trace[f'tw{s}']:
                        sat_events += 1
                        assert -128 in a, (a, E)
    print("twiddle saturation events:", sat_events)


# ---- (e) Hermitian -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("N", [16, 32])
def test_hermitian_even_odd(N):
    n = fm.log2i(N)
    for t in range(N):                                  # even slots carry exactly the bins k < N/2
        assert (fm.bitrev(t, n) < N // 2) == (t % 2 == 0)
    rng = np.random.default_rng(5)
    x = rng.integers(-128, 128, size=200 * N)
    y = fm.run_frames(x, N)
    worst = 0.0
    for f in range(199):
        o = y[f * N:(f + 1) * N]
        for t in range(2, N, 2):
            k = fm.bitrev(t, n)
            worst = max(worst, abs(o[fm.bitrev(N - k, n)] - np.conj(o[t])))
    print(f"N={N}: max |X[N-k] - conj X[k]| = {worst:.2f} LSB (fixed-point asymmetry; odd slots are never emitted)")
    assert worst <= 2 * BOUNDS[N][0]


def test_external_counter_equivalence():
    """SDF.step(x, cnt) with cnt from outside == internal counter (the chip drives cnt = ph mod N)."""
    rng = np.random.default_rng(9)
    a, b = fm.SDF(32), fm.SDF(32)
    for i, v in enumerate(rng.integers(-128, 128, 3000)):
        assert a.step(int(v)) == b.step(int(v), cnt=i % 32)


# ---- precision options (precision study): the defaults are the RTL; the options behave as specified ----------------
def test_precision_options():
    assert [fm.round_conv(v, 2) for v in (2, 6, -2, -6, 5, 7, -5, -7)] == [0, 2, 0, -2, 1, 2, -1, -2]   # ties to even
    rng = np.random.default_rng(77)
    x = rng.integers(-100, 100, 16 * 800)                    # below option b's saturation level
    def err(**kw):
        m = fm.SDF(16, **kw)
        g = 2 if kw.get('last_div2') is False else 1
        out = np.array([complex(*m.step(int(v))) for v in x])[15:]
        E = np.concatenate([out[f * 16:(f + 1) * 16] - fm.reference(x[f * 16:(f + 1) * 16], 16, gain=g) for f in range(798)])
        return np.sqrt(np.mean(np.abs(E) ** 2)) / g, m.sat
    explicit = fm.SDF(16, W=8, TW=7, Wi=8, last_div2=False, TWi=7)
    default = fm.SDF(16)
    assert all(explicit.step(int(v)) == default.step(int(v)) for v in x[:4000])     # explicit defaults == the RTL path
    e = {k: err(**kw) for k, kw in (('base', dict(last_div2=True)), ('b', dict(last_div2=False)),
                                    ('c9', dict(Wi=9, last_div2=True)), ('c10', dict(Wi=10, last_div2=True)))}
    print({k: f"{v[0]:.3f} LSB rms (rel.), sat {v[1]}" for k, v in e.items()})
    assert e['c10'][0] < e['c9'][0] < e['base'][0] and e['b'][0] < e['base'][0]
    assert all(v[1] == 0 for v in e.values())
