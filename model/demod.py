# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side demodulator for the chip's output pins (bus protocol: docs/info.md).

Input: what a reader samples once per SCLK: the data byte (uo_out), SOF, VALID, LOCK. Knowledge used
about the chip: the bus (byte order I then Q, even byte c carries bin bitrev(c), N bytes per frame,
SOF on byte 0, VALID marks clean frames, one frame per symbol while LOCK, bus don't-care otherwise)
and the timing (FFT window CP/2 into the CP; slips happen in the pause, so a slip shows as an SOF-to-SOF
period of L +- 1). Nothing else about the chip's internals.

Chain (acceptance path: two pilots, no decision feedback):
  pins -> lock intervals -> frames (SOF-led VALID runs, symbol index = SOF count inside the interval)
       -> bins -> PRBS phase per lock interval (joint sign correlation over both pilots and their known
          PRBS offsets; squaring removes each pilot's BPSK sign, the correlation sign resolves pi)
       -> per frame and pilot: h_p = P_p / s_p (complex gain incl. the phase ramp at k_p)
       -> optional averaging of h_p over a centred window of W frames that never crosses an observed slip
       -> slope per bin = theta_d + wrap(arg h_B - arg h_A - (k_B - k_A) theta_d) / (k_B - k_A),
          theta_d = -2*pi*(CP/2)/N (design ramp); unambiguous while |e| < N / (2 (k_B - k_A)) = 8
       -> data bin k: phase arg h_A + (k - k_A) * slope, gain interpolated from |h_A|, |h_B|
       -> Gray slicer -> bits; EVM.
Why: a window starting delta = CP/2 + e samples before the useful part sees the useful-part spectrum
times exp(-j*2*pi*k*(CP/2 + e)/N) (DFT shift theorem). The design part is known; the chip's residual
timing error e is not (sample grid, tracking hysteresis, lock offset, drift between slips) and rotates bin k
against a pilot by 2*pi*(k - k_p)*e/N = 22.5 deg per sample per bin at N = 16. Two pilots measure it.
Single-pilot mode (LEGACY_1P, the legacy signal): design ramp only -> error floor (why v1 has two pilots; kept as
the strict expected failure). Optional dd=True: decision-directed residual rotation per data bin
(sliding 32 frames inside a slip segment, iterated 3x, ramp-aware in single-pilot mode); not used by
the acceptance tests.
"""
import numpy as np

from signal_gen import gray_levels, MODS, prbs7


def bitrev(k, n):
    r = 0
    for _ in range(n):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


def s8(b):
    """Pin byte -> signed 8-bit value."""
    b = int(b) & 0xFF
    return b - 256 if b >= 128 else b


# ----------------------------------------------------------------------------------------------------------
def parse_pins(uo, sof, valid, lock, N, L):
    """Pins (one sample per SCLK) -> list of lock intervals, each a list of frames:
    dict(cycle, index (SOF count in the interval), period (cycles since the previous SOF, None for the
    first), valid (all N bytes VALID), bins {k: complex} (only if valid))."""
    n = N.bit_length() - 1
    uo, sof, valid, lock = map(np.asarray, (uo, sof, valid, lock))
    intervals = []
    cur, idx, last_sof = None, 0, None
    for c in range(len(uo)):
        if lock[c] and cur is None:
            cur, idx, last_sof = [], 0, None
            intervals.append(cur)
        if not lock[c]:
            cur = None
            continue
        if sof[c]:
            f = dict(cycle=c, index=idx, period=None if last_sof is None else c - last_sof, valid=False, bins=None)
            if c + N <= len(uo) and np.all(valid[c:c + N] == 1) and np.all(lock[c:c + N] == 1):
                by = [s8(v) for v in uo[c:c + N]]
                f['valid'] = True
                f['bins'] = {bitrev(t, n): complex(by[t], by[t + 1]) for t in range(0, N, 2)}
            cur.append(f)
            idx += 1
            last_sof = c
    return intervals


def prbs_sync(pilot_frames, offsets, seed=0x7F, period=127):
    """pilot_frames: {k: list of (frame index i, complex P)}; offsets: {k: PRBS offset o_k}.
    Returns (m, score): pilot k's sign in frame i is ps[(i + m + o_k) mod 127]; score in [0, 1]."""
    ps = 1 - 2 * prbs7(period, seed)
    sig = {}
    for k, fr in pilot_frames.items():
        idx = np.array([i for i, _ in fr])
        P = np.array([p for _, p in fr])
        th = 0.5 * np.angle(np.sum(P ** 2))                 # BPSK: squaring removes the sign
        s = np.sign(np.real(P * np.exp(-1j * th)))
        s[s == 0] = 1
        sig[k] = (idx, s)
    def score(m):
        return sum(abs(np.sum(s * ps[(idx + m + offsets[k]) % period])) for k, (idx, s) in sig.items())
    best = max(range(period), key=score)
    return best, score(best) / sum(len(s) for _, s in sig.values())


def slice_qam(c, mod):
    """Hard decision, Gray bits, nearest point (normalised constellation: outer corner |c| = 1)."""
    if mod == 'BPSK':
        b = (np.real(c) < 0).astype(int)[..., None]
        return b, np.where(b[..., 0] == 1, -1.0, 1.0) + 0j
    m = MODS[mod]
    M = 1 << m
    lv = gray_levels(m)                                   # index = gray code -> level
    lv_by_val = {v: g for g, v in enumerate(lv)}
    scale = (M - 1) / np.sqrt(1.0) * np.sqrt(2)           # normalised -> level units
    def axis(x):
        v = np.clip(2 * np.round((x * scale + (M - 1)) / 2) - (M - 1), -(M - 1), M - 1)
        g = np.vectorize(lambda q: lv_by_val[q])(v)
        bits = (g[..., None] >> np.arange(m - 1, -1, -1)) & 1
        return v, bits
    vi, bi = axis(np.real(c))
    vq, bq = axis(np.imag(c))
    dec = (vi + 1j * vq) / scale
    return np.concatenate([bi, bq], axis=-1), dec


# ----------------------------------------------------------------------------------------------------------
def _dd_rotation(c, mod, seg, dd_window, dd_iter):
    """Sliding-window (within slip segments), iterated decision-directed rotation estimate; returns
    the de-rotated points and the total rotation removed per frame (radians)."""
    h2 = dd_window // 2
    total = np.zeros(len(c))
    for _ in range(dd_iter):                               # iterate: de-rotate, re-slice, re-estimate
        _, d = slice_qam(c, mod)
        z = c * np.conj(d)
        rot = np.zeros(len(c))
        for s_id in np.unique(seg):
            ix = np.flatnonzero(seg == s_id)
            cs = np.concatenate([[0], np.cumsum(z[ix])])
            for n_, i in enumerate(ix):
                lo, hi = max(0, n_ - h2), min(len(ix), n_ + h2 + 1)
                rot[i] = np.angle(cs[hi] - cs[lo])
        c = c * np.exp(-1j * rot)
        total += rot
    return c, total


def _window_mean(z, seg, w):
    """Centred moving average of complex z over w frames, never crossing a segment boundary."""
    if w <= 1:
        return z
    out = np.empty_like(z)
    h2 = w // 2
    for s_id in np.unique(seg):
        ix = np.flatnonzero(seg == s_id)
        cs = np.concatenate([[0], np.cumsum(z[ix])])
        for n_, i in enumerate(ix):
            lo, hi = max(0, n_ - h2), min(len(ix), n_ + h2 + 1)
            out[i] = (cs[hi] - cs[lo]) / (hi - lo)
    return out


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def equalize(frames, pilot_signs, cfg, dd=False, segments=None, avg_window=1, dd_window=32, dd_iter=3):
    """frames: list of dict(bins) (all valid); pilot_signs: {k_p: +-1 per frame}; segments: slip-segment
    id per frame. Returns {k: c_hat array} for the data bins. Two pilots -> interpolation (acceptance);
    one pilot -> design ramp only (+ optional DD)."""
    N, CP = cfg.N, cfg.CP
    theta_d = -2 * np.pi * (CP / 2) / N                   # design ramp per bin (window CP/2 into the CP)
    seg = np.zeros(len(frames), int) if segments is None else np.asarray(segments)
    pb = list(cfg.pilot_bins)
    h = {k: np.array([f['bins'][k] for f in frames]) / np.asarray(pilot_signs[k]) for k in pb}
    h = {k: _window_mean(v, seg, avg_window) for k, v in h.items()}
    kA = pb[0]
    phA, gA = np.angle(h[kA]), np.abs(h[kA])
    if len(pb) >= 2:
        kB = pb[-1]
        resid = _wrap(np.angle(h[kB]) - phA - (kB - kA) * theta_d)
        slope = theta_d + resid / (kB - kA)
        gB = np.abs(h[kB])
    else:
        slope = np.full(len(frames), theta_d)
        kB, gB = kA, gA
    c = {}
    for k in cfg.data_bins:
        Y = np.array([f['bins'][k] for f in frames])
        ph = phA + (k - kA) * slope
        g = gA if kB == kA else gA + (k - kA) / (kB - kA) * (gB - gA)
        c[k] = Y / (g * np.exp(1j * ph))
    if dd:
        if len(pb) == 1:                                   # single pilot: ramp-aware pre-rotation
            adj = [k for k in cfg.data_bins if abs(k - kA) == 1]
            if adj and len(cfg.data_bins) > len(adj):
                rho = np.mean([_dd_rotation(c[k], cfg.modulation, seg, dd_window, dd_iter)[1] / (k - kA) for k in adj], axis=0)
                for k in cfg.data_bins:
                    if abs(k - kA) > 1:
                        c[k] = c[k] * np.exp(-1j * (k - kA) * rho)
        for k in cfg.data_bins:
            c[k], _ = _dd_rotation(c[k], cfg.modulation, seg, dd_window, dd_iter)
    return c


AVG_WINDOW = 16          # pilot-averaging window (frames), on the BER plateau (study_results/pilot_avg_window.json)


def demodulate(pins, cfg, dd=False, avg_window=AVG_WINDOW, min_frames=32):
    """pins: dict of arrays uo, sof, valid, lock (one per SCLK). Returns a result dict with, per lock
    interval: PRBS phase, frames (index, cycle), decided bits and c_hat per data bin, slips."""
    N, L = cfg.N, cfg.L
    intervals = parse_pins(pins['uo'], pins['sof'], pins['valid'], pins['lock'], N, L)
    res = dict(intervals=[], first_lock_cycle=None)
    lk = np.asarray(pins['lock'])
    if lk.any():
        res['first_lock_cycle'] = int(np.argmax(lk == 1))
    offsets = dict(zip(cfg.pilot_bins, cfg.pilot_prbs_offsets))
    ps = 1 - 2 * prbs7(127, cfg.prbs_seed)
    for frames in intervals:
        vf = [f for f in frames if f['valid']]
        slips = [(f['index'], f['period'] - L) for f in frames if f['period'] is not None and f['period'] != L]
        iv = dict(start_cycle=frames[0]['cycle'] if frames else None, n_frames=len(frames), n_valid=len(vf),
                  slips=slips, prbs=None)
        if len(vf) >= min_frames:
            m, score = prbs_sync({k: [(f['index'], f['bins'][k]) for f in vf] for k in cfg.pilot_bins},
                                 offsets, cfg.prbs_seed)
            psign = {k: np.array([ps[(f['index'] + m + offsets[k]) % 127] for f in vf]) for k in cfg.pilot_bins}
            seg_of_index = np.cumsum([0] + [1 if (f['period'] is not None and f['period'] != L) else 0 for f in frames[1:]])
            segs = [seg_of_index[f['index']] for f in vf]
            c = equalize(vf, psign, cfg, dd=dd, segments=segs, avg_window=avg_window)
            bits = {k: slice_qam(c[k], cfg.modulation)[0] for k in cfg.data_bins}
            iv.update(prbs=dict(m=m, score=score), index=np.array([f['index'] for f in vf]),
                      cycle=np.array([f['cycle'] for f in vf]), c_hat=c, bits=bits,
                      pilots={k: np.array([f['bins'][k] for f in vf]) for k in cfg.pilot_bins})
        res['intervals'].append(iv)
    return res


def evm_db(c_hat, c_ref):
    """EVM (dB) of equalised points vs the reference points (rms error / rms reference)."""
    e = np.asarray(c_hat) - np.asarray(c_ref)
    return 10 * np.log10(np.mean(np.abs(e) ** 2) / np.mean(np.abs(c_ref) ** 2))


# ----------------------------------------------------------------------------------------------------------
# Bench use (docs/info.md "How to test"): demodulate a capture of the chip's pins taken with a logic analyser,
# an FPGA or RP2040 PIO, one row per SCLK falling edge (the reader's sampling edge).
def read_capture(path):
    """Capture CSV with a header row -> pins dict for demodulate(). Columns (decimal or 0x hex):
    `uo_out` (the data byte) and `uio_out` (SCLK = bit 0, SOF = bit 1, VALID = bit 2, ADC_CLK = bit 3, LOCK =
    bit 4, DBG = bit 7), or `uo_out` with separate `sof`, `valid`, `lock` columns. Extra columns are ignored."""
    import csv
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    num = lambda v: int(str(v).strip(), 0)
    cols = {c.strip().lower(): c for c in rows[0]}
    uo = np.array([num(r[cols['uo_out']]) for r in rows])
    if 'uio_out' in cols:
        uio = np.array([num(r[cols['uio_out']]) for r in rows])
        sof, valid, lock = (uio >> 1) & 1, (uio >> 2) & 1, (uio >> 4) & 1
    else:
        sof, valid, lock = (np.array([num(r[cols[c]]) for r in rows]) for c in ('sof', 'valid', 'lock'))
    return dict(uo=uo, sof=sof, valid=valid, lock=lock)


def bench_ber(res, tx, cfg):
    """Bit errors against the stimulus record (signal_gen.save's _tx.json). The pilot PRBS fixes each lock
    interval's TX symbol index modulo 127 (j = j0 + frame index, j0 = PRBS phase m mod 127); the multiple of 127
    is searched (the capture need not start with the stimulus), and j wraps modulo the stimulus length (playback
    in a loop). Returns (errors, bits, frames)."""
    nsym = len(tx['pilot_sign'])
    errs = nbits = frames = 0
    for iv in res['intervals']:
        if iv['prbs'] is None:
            continue
        best = None
        for j0 in range(iv['prbs']['m'] % 127, max(nsym, 127), 127):
            j = (j0 + iv['index']) % nsym
            e = sum(int(np.sum(iv['bits'][k] != tx['data_bits'][k][j])) for k in cfg.data_bins)
            if best is None or e < best:
                best = e
        errs += best
        nbits += sum(iv['bits'][k].size for k in cfg.data_bins)
        frames += len(iv['index'])
    return errs, nbits, frames


def main(argv=None):
    import argparse
    import signal_gen as sg
    ap = argparse.ArgumentParser(description="Demodulate a capture of the chip's bus (see read_capture) with the "
                                 "acceptance demodulator (two pilots); optional BER against the stimulus.")
    ap.add_argument("capture", help="CSV, one row per SCLK falling edge: uo_out, uio_out")
    ap.add_argument("--tx", help="stimulus prefix written by model/signal_gen.py (reads PREFIX_tx.json)")
    ap.add_argument("--plot", help="write the equalised data-bin constellation to this PNG")
    a = ap.parse_args(argv)
    pins = read_capture(a.capture)
    tx = None
    cfg = sg.V1
    if a.tx:
        import json
        with open(a.tx + '_tx.json') as f:
            js = json.load(f)
        cfg = sg.config_from_dict(js['config'])
        tx = dict(pilot_sign=np.array(js['pilot_prbs']),
                  data_bits={int(k): np.array(v) for k, v in js['data_bits'].items()})
    res = demodulate(pins, cfg)
    lk = np.asarray(pins['lock'])
    print(f"{a.capture}: {len(lk)} rows; LOCK high in {int(lk.sum())} rows, first at row "
          f"{res['first_lock_cycle']}; {len(res['intervals'])} lock interval(s)")
    pts = {k: [] for k in cfg.data_bins}
    for n, iv in enumerate(res['intervals']):
        line = f"  interval {n}: from row {iv['start_cycle']}, {iv['n_frames']} frames, {iv['n_valid']} VALID, " \
               f"{len(iv['slips'])} slip(s)"
        if iv['prbs'] is None:
            line += " -- too few VALID frames to demodulate"
        else:
            line += f", pilot PRBS phase {iv['prbs']['m']} (score {iv['prbs']['score']:.2f})"
            for k in cfg.data_bins:
                pts[k].append(iv['c_hat'][k])
        print(line)
    for k in cfg.data_bins:
        if pts[k]:
            c = np.concatenate(pts[k])
            ref = slice_qam(c, cfg.modulation)[1]
            print(f"  bin {k} ({cfg.modulation}): {len(c)} symbols, EVM {evm_db(c, ref):.1f} dB (vs nearest point)")
    if tx is not None:
        errs, nbits, frames = bench_ber(res, tx, cfg)
        print(f"  BER {errs / nbits if nbits else float('nan'):.2e} ({errs}/{nbits} bits, {frames} frames)")
    if a.plot and any(pts.values()):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        for k in cfg.data_bins:
            c = np.concatenate(pts[k])
            ax.plot(c.real, c.imag, '.', ms=2, alpha=0.5, label=f"bin {k}")
        ax.set_aspect('equal'); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.grid(alpha=0.3)
        ax.set_title(f"{a.capture.split('/')[-1]}: {cfg.modulation}", fontsize=9)
        fig.tight_layout(); fig.savefig(a.plot, dpi=100)
        print(f"  constellation -> {a.plot}")


if __name__ == "__main__":
    main()
