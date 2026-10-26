# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Closed loop: generator -> chip model (pins) -> demodulator, plus the reference paths that separate
quantisation loss from timing loss (shared by test_ofdm_model.py and the BER curves in study_results/).

Paths (same TX bits and the same noise realisation; ppm = 0, integer phase0 so that the chip's ideal
window falls on RX samples):
  A  float   : analog noisy samples y_float, ideal windows (exactly CP/2 into every CP), float FFT/N
  B  quant   : the 8-bit ADC samples, the same ideal windows, the bit-exact SDF model
  C  chip    : the full system model (its own timing recovery, pins), the host demodulator
A -> B = quantisation loss (ADC + fixed-point FFT), B -> C = timing loss (window placement, PRBS sync,
lock gaps). All three use the same equaliser (v1: two-pilot interpolation with pilot averaging, no
decision feedback; the legacy single-pilot signal: design ramp, optional decision-directed).
"""
import numpy as np

import signal_gen as sg
import demod as dm
from fft_model import SDF
from system_model import FFTChipModel


def run_chip(cfg, adc, rst_cycles=3, prec=None):
    """Chip pins, one sample per cycle (the value visible during the following cycle). prec: SDF precision
    options (precision study; None = the RTL)."""
    chip = FFTChipModel(N=cfg.N, cfg_cp=cfg.cfg_cp, prec=prec)
    x = sg.to_signed(adc)
    uo = np.zeros(len(x), int); sof = np.zeros(len(x), int); valid = np.zeros(len(x), int); lock = np.zeros(len(x), int)
    for i, v in enumerate(x):
        o = chip.step(int(v), rst_n=0 if i < rst_cycles else 1)
        uo[i] = 0 if o['uo'] is None else o['uo'] & 0xFF
        sof[i], valid[i], lock[i] = o['sof'], o['valid'], o['lock']
    return dict(uo=uo, sof=sof, valid=valid, lock=lock)


def score_chip(cfg, res, tx, bd):
    """Map demodulated frames to TX symbols and count bit errors. The PRBS phase found by the
    demodulator fixes the symbol index modulo 127; the true timing only picks the multiple of 127
    (and is used to check the PRBS sync)."""
    errs = nbits = 0
    pts = {k: [] for k in cfg.data_bins}
    refs = {k: [] for k in cfg.data_bins}
    n_frames = 0
    for iv in res['intervals']:
        if iv['prbs'] is None:
            continue
        m = iv['prbs']['m']
        for f_i, (i, c) in enumerate(zip(iv['index'], iv['cycle'])):
            j_est = int(bd['sym'][min(len(bd['sym']) - 1, c + 1 - cfg.L + cfg.N // 2)])
            j = j_est + ((i + m - j_est) % 127 + 63) % 127 - 63
            n_frames += 1
            for k in cfg.data_bins:
                b = iv['bits'][k][f_i]
                errs += int(np.sum(b != tx['data_bits'][k][j]))
                nbits += b.size
                pts[k].append(iv['c_hat'][k][f_i])
                refs[k].append(tx['data_points'][k][j])
    evm = {k: (dm.evm_db(pts[k], refs[k]) if pts[k] else None) for k in cfg.data_bins}
    return dict(ber=errs / nbits if nbits else None, errs=errs, nbits=nbits, frames=n_frames, evm_db=evm,
                points={k: np.array(v) for k, v in pts.items()}, refs={k: np.array(v) for k, v in refs.items()})


def chip_run(cfg, n_symbols, seed, snr_db=None, ppm=0.0, phase0=None, dd=False, dc_offset=0.0, avg_window=dm.AVG_WINDOW,
             prec=None):
    adc, tx, bd = sg.generate(cfg, n_symbols, seed, snr_db=snr_db, ppm=ppm, phase0=phase0, dc_offset=dc_offset)
    pins = run_chip(cfg, adc, prec=prec)
    res = dm.demodulate(pins, cfg, dd=dd, avg_window=avg_window)
    sc = score_chip(cfg, res, tx, bd)
    lk = pins['lock']
    sc.update(first_lock_sym=None if res['first_lock_cycle'] is None else res['first_lock_cycle'] / cfg.L,
              intervals=len(res['intervals']), slips=sum(len(iv['slips']) for iv in res['intervals']),
              locked_sym=lk.sum() / cfg.L, prbs=[iv['prbs'] for iv in res['intervals']],
              clip=tx['clip_fraction'], res=res, tx=tx, bd=bd, pins=pins)
    return sc


def ideal_window_bins(cfg, samples, n_symbols, quant):
    """Bins of every symbol from ideal windows (CP/2 into the CP; ppm = 0, phase0 = 0)."""
    N, L, CP = cfg.N, cfg.L, cfg.CP
    start = CP - CP // 2
    frames = []
    for j in range(n_symbols - 1):
        w = samples[j * L + start: j * L + start + N]
        if len(w) < N:
            break
        if quant:
            m = SDF(N)
            out = [m.step(int(v)) for v in list(w) + [0] * N][N - 1: 2 * N - 1]
            n = N.bit_length() - 1
            bins = {dm.bitrev(t, n): complex(out[t][0], out[t][1]) for t in range(0, N, 2)}
        else:
            X = np.fft.fft(w) / N
            bins = {k: X[k] for k in range(N // 2)}
        frames.append(dict(bins=bins))
    return frames


def reference_run(cfg, tx, bd, adc, path, dd=False, avg_window=dm.AVG_WINDOW):
    """Paths A ('float') and B ('quant') on the ideal windows; returns BER and EVM per data bin."""
    n_sym = len(tx['pilot_sign'])
    if path == 'float':
        frames = ideal_window_bins(cfg, bd['y_float'], n_sym, quant=False)
    else:
        frames = ideal_window_bins(cfg, sg.to_signed(adc), n_sym, quant=True)
    ps = {k: tx['pilot_signs'][k][:len(frames)] for k in cfg.pilot_bins}
    c = dm.equalize(frames, ps, cfg, dd=dd, avg_window=avg_window)
    errs = nbits = 0
    evm = {}
    for k in cfg.data_bins:
        bits, _ = dm.slice_qam(c[k], cfg.modulation)
        ref_bits = tx['data_bits'][k][:len(frames)]
        errs += int(np.sum(bits != ref_bits))
        nbits += bits.size
        evm[k] = dm.evm_db(c[k], tx['data_points'][k][:len(frames)])
    return dict(ber=errs / nbits, errs=errs, nbits=nbits, evm_db=evm, frames=len(frames))


def snr_at_ber(snrs, bers, target=1e-3):
    """Interpolate the SNR (dB) where BER crosses `target` (log-linear); None if not bracketed."""
    pts = [(s, b) for s, b in zip(snrs, bers) if b is not None]
    for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
        if b0 >= target > b1:
            if b1 <= 0:
                b1 = 0.5 / 1e6
            f = (np.log10(b0) - np.log10(target)) / (np.log10(b0) - np.log10(b1))
            return s0 + f * (s1 - s0)
    return None


def snr_at_ber_sigma(snrs, bers, nbits, target=1e-3):
    """1-sigma uncertainty (dB) of snr_at_ber from the Poisson error counts at the two bracketing points:
    sigma(log10 BER) = 1 / (ln 10 * sqrt(n_err)) (n_err >= 1), propagated through the log-linear
    interpolation. None if not bracketed."""
    pts = [(s, b, n) for s, b, n in zip(snrs, bers, nbits) if b is not None]
    for (s0, b0, n0), (s1, b1, n1) in zip(pts, pts[1:]):
        if b0 >= target > b1:
            l0, l1 = np.log10(b0), np.log10(b1 if b1 > 0 else 0.5 / 1e6)
            f = (l0 - np.log10(target)) / (l0 - l1)
            sig = [1 / (np.log(10) * np.sqrt(max(1.0, b * n))) for b, n in ((b0, n0), (b1, n1))]
            return float((s1 - s0) / (l0 - l1) * np.hypot((1 - f) * sig[0], f * sig[1]))
    return None


def recorded_curves(N):
    """The characterisation rows written by `python3 ofdm_loop.py` for this N (N = 32: the earlier design point)."""
    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    return json.load(open(os.path.join(here, 'study_results', f'ber_curves{"" if N == 32 else f"_n{N}"}.json')))


def theory_ber(cfg, snr_db):
    """Gray 16-QAM (or QPSK / 64-QAM) BER at the bin SNR implied by the time-domain SNR with an ideal
    1/N FFT: noise per bin = sigma^2 / N (complex), data-bin power = (a/2)^2 E|c|^2."""
    from scipy.special import erfc
    sigma2 = cfg.signal_power() / 10 ** (snr_db / 10)
    es = (cfg.a / 2) ** 2 * sg.qam_mean_power(cfg.modulation)
    n0 = sigma2 / cfg.N
    M = 1 << (2 * sg.MODS[cfg.modulation])
    k = np.log2(M)
    x = np.sqrt(3 * es / ((M - 1) * n0))
    return float((4 / k) * (1 - 1 / np.sqrt(M)) * 0.5 * erfc(x / np.sqrt(2)))


MODES = {
    # name: (config, demod kwargs)
    'v1':      (sg.V1, dict(dd=False, avg_window=dm.AVG_WINDOW)),     # acceptance (two pilots, averaging)
    'v1_raw':  (sg.V1, dict(dd=False, avg_window=1)),                 # no pilot averaging
    'v1_dd':   (sg.V1, dict(dd=True, avg_window=dm.AVG_WINDOW)),      # optional refinement
    'legacy':  (sg.LEGACY_1P, dict(dd=False, avg_window=1)),          # single-pilot legacy signal, design ramp only
    'legacy_dd': (sg.LEGACY_1P, dict(dd=True, avg_window=1)),         # best single-pilot variant (decision feedback)
}


def _curve_point(args):
    cfg_cp, snr, seed, nsym, modes = args
    out = dict(cfg_cp=cfg_cp, snr=snr)
    by_cfg = {}
    for name in modes:
        base, kw = MODES[name]
        cfg = sg.OFDMConfig(**{**base.__dict__, 'cfg_cp': cfg_cp})
        key = (cfg.pilot_bins, cfg.data_bins)
        if key not in by_cfg:                               # same TX bits and noise for every mode of a signal
            by_cfg[key] = sg.generate(cfg, nsym, seed, snr_db=snr, ppm=0.0, phase0=0.0)
            by_cfg[key] = by_cfg[key] + (run_chip(cfg, by_cfg[key][0]),)
        adc, tx, bd, pins = by_cfg[key]
        out[f'theory_{name}'] = theory_ber(cfg, snr)
        a = reference_run(cfg, tx, bd, adc, 'float', **kw)
        b = reference_run(cfg, tx, bd, adc, 'quant', **kw)
        c = score_chip(cfg, dm.demodulate(pins, cfg, **kw), tx, bd)
        k = cfg.data_bins[0]
        out[f'A_{name}'], out[f'B_{name}'], out[f'C_{name}'] = a['ber'], b['ber'], c['ber']
        out[f'A_{name}_bits'], out[f'B_{name}_bits'], out[f'C_{name}_bits'] = a['nbits'], b['nbits'], c['nbits']
        out[f'evmA_{name}'], out[f'evmB_{name}'], out[f'evmC_{name}'] = a['evm_db'][k], b['evm_db'][k], c['evm_db'][k]
    return out


def ber_curves(snrs=tuple(range(8, 25, 2)), nsym=6000, seed=4242, modes=('v1', 'v1_raw', 'legacy_dd')):
    from multiprocessing import Pool
    jobs = [(c, s, seed + 17 * s + c, nsym, modes) for c in sg.cfg_cps(sg.V1.N) for s in snrs]
    with Pool() as p:
        return p.map(_curve_point, jobs)


if __name__ == '__main__':
    import json, os, sys, time
    t0 = time.time()
    modes = ('v1', 'v1_raw', 'v1_dd', 'legacy', 'legacy_dd')
    rows = ber_curves(snrs=tuple(range(6, 27, 2)), nsym=6000, seed=4242, modes=modes)
    os.makedirs('study_results', exist_ok=True)
    N = sg.V1.N
    json.dump(rows, open(f'study_results/ber_curves{"" if N == 32 else f"_n{N}"}.json', 'w'), indent=1)
    f = lambda v: '—' if v is None else ('0' if v == 0 else f'{v:.1e}')
    ev = lambda v: '—' if v is None else f'{v:.1f}'
    for cp in sg.cfg_cps(N):
        rs = [r for r in rows if r['cfg_cp'] == cp]
        print(f"\nN = {N}, CP = {sg.OFDMConfig(N=N, cfg_cp=cp).CP}: v1 = two pilots (4, 6) + 16-QAM at 5, a = 36.95 LSB; legacy = one pilot (4) + data 5, a = 45.25 LSB")
        print("| SNR (dB) | theory v1 | A v1 | B v1 | C v1 (accept.) | C v1 no avg | C v1 + DD | theory legacy | C legacy (pilot only) | C legacy + DD | EVM A / B / C v1 (dB) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            print(f"| {r['snr']} | {f(r['theory_v1'])} | {f(r['A_v1'])} | {f(r['B_v1'])} | {f(r['C_v1'])} | {f(r['C_v1_raw'])} | {f(r['C_v1_dd'])} | "
                  f"{f(r['theory_legacy'])} | {f(r['C_legacy'])} | {f(r['C_legacy_dd'])} | {ev(r['evmA_v1'])} / {ev(r['evmB_v1'])} / {ev(r['evmC_v1'])} |")
        sn = [r['snr'] for r in rs]
        keys = ('theory_v1', 'A_v1', 'B_v1', 'C_v1', 'A_v1_raw', 'B_v1_raw', 'C_v1_raw', 'C_v1_dd', 'theory_legacy', 'A_legacy', 'B_legacy', 'C_legacy', 'C_legacy_dd')
        g = {k: snr_at_ber(sn, [r[k] for r in rs]) for k in keys}
        print("SNR @ BER 1e-3: " + ", ".join(f"{k} {v:.2f}" for k, v in g.items() if v is not None))
    print(f"[{time.time() - t0:.0f} s]")
