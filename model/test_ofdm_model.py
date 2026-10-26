# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generator -> system model -> host demodulator, closed loop (mini-OFDM v1 signal, bus framing per docs/info.md).

Acceptance: the v1 signal (BPSK pilots at k = 4 and 6 with their own PRBS phases,
16-QAM at k = 5, 37 LSB per carrier) and the two-pilot demodulator with pilot averaging (W = 16 frames
between slips) and NO decision feedback. The decision-directed refinement is exercised as an optional
mode; the single-pilot demodulator (LEGACY_1P signal, design ramp only) stays as a strict expected
failure. Numbers: model/README.md, model/study_results/ber_curves.json (N = 32) and
ber_curves_n16.json (N = 16).
Runs at the project N (signal_gen.PROJECT_N: 16) and only the CFG_CP settings that exist there
(N = 16: CP = 4 only, fixed at N/4).
"""
import os
import numpy as np
import pytest
from multiprocessing import Pool

import signal_gen as sg
import demod as dm
import ofdm_loop as ol

ACCEPT = dict(dd=False, avg_window=dm.AVG_WINDOW)     # acceptance demodulator (two pilots, averaging, no DD)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
CPS = sg.cfg_cps(sg.V1.N)                    # CFG_CP settings at the project N


def lock_bound_sym(cfg):
    return 2 * cfg.L * (4 * cfg.N // cfg.CP)  # two full hypothesis sweeps of K = 4N/CP (p95 561 / 1120 sym at N = 32)


# ---- generator ------------------------------------------------------------------------------------
def test_generator_basics(tmp_path):
    cfg = sg.V1
    assert cfg.pilot_bins == (4, 6) and cfg.data_bins == (5,) and cfg.pilot_prbs_offsets == (0, 63)
    adc, tx, bd = sg.generate(cfg, 300, seed=1, phase0=0.0)
    assert adc.min() >= 0 and adc.max() <= 255 and tx['clip_fraction'] == 0.0
    assert abs(cfg.a - 64 / np.sqrt(3)) < 1e-9                 # 64/sqrt(M) per carrier, M = 3 carriers
    y = bd['y_float']
    N, L, CP = cfg.N, cfg.L, cfg.CP
    assert np.abs(y).max() <= 127
    for j in (3, 50, 200):
        s_ = y[j * L:(j + 1) * L]
        assert np.allclose(s_[:CP], s_[N:N + CP])                 # CP = copy of the last CP samples
        X = np.fft.fft(s_[CP:]) / N
        for k in cfg.pilot_bins:
            assert abs(X[k] - cfg.a / 2 * tx['pilot_signs'][k][j]) < 1e-9
        assert abs(X[5] - cfg.a / 2 * tx['data_points'][5][j]) < 1e-9
        assert all(abs(X[k]) < 1e-9 for k in range(1, N // 2) if k not in (4, 5, 6))
    # PRBS7 x^7 + x^6 + 1, seed 0x7F, against an independent LFSR; pilot 6 = pilot 4 shifted by 63
    p = sg.prbs7(254)
    s, ind = 0x7F, []
    for _ in range(254):
        ind.append((s >> 6) & 1)
        s = ((s << 1) | (((s >> 6) ^ (s >> 5)) & 1)) & 0x7F
    assert list(p) == ind and np.array_equal(p[:127], p[127:])
    j = np.arange(300)
    assert np.array_equal(tx['pilot_signs'][4], 1 - 2 * p[j % 127])
    assert np.array_equal(tx['pilot_signs'][6], 1 - 2 * p[(j + 63) % 127])
    assert abs(np.mean(tx['pilot_signs'][4] * tx['pilot_signs'][6])) < 0.05   # not a fixed waveform
    assert np.array_equal(sg.to_signed(adc), np.clip(np.round(y), -128, 127).astype(int))   # offset binary
    _, _, b2 = sg.generate(cfg, 50, seed=2, ppm=100, phase0=1.5)
    assert np.allclose(np.diff(b2['t']), 1 + 100e-6) and b2['t'][0] == 1.5                  # exact resample
    pref = str(tmp_path / 'vec')
    sg.save(pref, adc, tx, bd)
    a2, t2, _ = sg.load(pref)
    assert np.array_equal(a2, adc) and np.array_equal(t2['pilot_signs'][6], tx['pilot_signs'][6])
    assert open(pref + '.csv').read().splitlines()[1] == str(int(adc[0]))
    assert sg.LEGACY_1P.pilot_bins == (4,) and abs(sg.LEGACY_1P.a - 64 / np.sqrt(2)) < 1e-9


# ---- acceptance: v1 configuration -----------------------------------------------------------------
@pytest.mark.parametrize("cfg_cp", CPS)
def test_v1_high_snr(cfg_cp):
    cfg = sg.OFDMConfig(cfg_cp=cfg_cp)
    r = ol.chip_run(cfg, lock_bound_sym(cfg) + 1100, seed=31, snr_db=30, **ACCEPT)
    print(f"CP={cfg.CP}: lock after {r['first_lock_sym']:.0f} symbols (bound {lock_bound_sym(cfg)}), "
          f"{r['frames']} frames demodulated, BER {r['ber']} ({r['nbits']} bits), EVM {r['evm_db'][5]:.1f} dB, "
          f"PRBS sync score {r['prbs'][0]['score']:.2f}")
    assert r['first_lock_sym'] <= lock_bound_sym(cfg)
    assert r['frames'] >= 1000 and r['intervals'] == 1
    assert r['ber'] == 0.0
    assert r['prbs'][0]['score'] >= 0.9


def _phase_job(a):
    cfg_cp, phase0, seed = a
    cfg = sg.OFDMConfig(cfg_cp=cfg_cp)
    r = ol.chip_run(cfg, lock_bound_sym(cfg) + 1100, seed=seed, snr_db=30, phase0=phase0, **ACCEPT)   # >= 1000 locked
    return dict(phase0=phase0, lock=r['first_lock_sym'], frames=r['frames'], ber=r['ber'], evm=r['evm_db'][5])


@pytest.mark.parametrize("cfg_cp", CPS)
def test_phase0_sweep(cfg_cp):
    cfg = sg.OFDMConfig(cfg_cp=cfg_cp)
    phases = np.linspace(0, cfg.L, 9)[:-1] + 0.37              # 8 phases, fractional on purpose
    with Pool() as p:
        res = p.map(_phase_job, [(cfg_cp, float(ph), 900 + i) for i, ph in enumerate(phases)])
    for r in res:
        print(f"CP={cfg.CP} phase0={r['phase0']:5.2f}: lock {r['lock']:.0f} sym, frames {r['frames']}, BER {r['ber']}, EVM {r['evm']:.1f} dB")
        assert r['lock'] <= lock_bound_sym(cfg) and r['frames'] >= 1000 and r['ber'] == 0.0


def _ppm_job(a):
    cfg_cp, ppm, mode = a
    base, kw = {'accept': (sg.V1, ACCEPT), 'dd': (sg.V1, dict(dd=True, avg_window=dm.AVG_WINDOW)),
                'legacy': (sg.LEGACY_1P, dict(dd=False, avg_window=1))}[mode]
    cfg = sg.OFDMConfig(**{**base.__dict__, 'cfg_cp': cfg_cp})
    r = ol.chip_run(cfg, 11_200 if cfg_cp == 0 else 11_800, seed=1200 + cfg_cp * 10 + (ppm > 0), snr_db=30, ppm=ppm, **kw)
    drift = r['locked_sym'] * cfg.L * ppm * 1e-6
    return dict(cfg_cp=cfg_cp, ppm=ppm, frames=r['frames'], ber=r['ber'], intervals=r['intervals'], slips=r['slips'],
                drift=drift, evm=r['evm_db'][5], lock=r['first_lock_sym'],
                net=sum(d for iv in r['res']['intervals'] for _, d in iv['slips']))


@pytest.mark.parametrize("cfg_cp", CPS)
def test_ppm_long(cfg_cp):
    with Pool() as p:
        res = p.map(_ppm_job, [(cfg_cp, ppm, 'accept') for ppm in (-100, 100)])
    for r in res:
        print(f"CP={sg.OFDMConfig(cfg_cp=cfg_cp).CP} ppm={r['ppm']:+d}: {r['frames']} frames, BER {r['ber']}, EVM {r['evm']:.1f} dB, "
              f"lock intervals {r['intervals']}, slips seen by the reader {r['slips']} (net {r['net']:+d}), drift {r['drift']:+.1f} samples")
        assert r['frames'] >= 10_000 and r['intervals'] == 1 and r['ber'] == 0.0
        assert abs(abs(r['net']) - abs(r['drift'])) <= 4 and np.sign(r['net']) == -np.sign(r['drift'])


@pytest.mark.xfail(strict=True, reason="with ONE pilot (the legacy signal) the design-ramp demodulator "
                                       "cannot see the chip's residual timing error (11.25 deg per sample on the adjacent bin)")
def test_ppm_long_single_pilot_legacy():
    with Pool() as p:
        res = p.map(_ppm_job, [(c, 100, 'legacy') for c in CPS])
    for r in res:
        print(f"legacy single pilot CP={sg.OFDMConfig(cfg_cp=r['cfg_cp']).CP} ppm=+100: BER {r['ber']:.2e}, EVM {r['evm']:.1f} dB")
    assert all(r['ber'] == 0.0 for r in res)


def test_ppm_long_optional_dd():
    """The optional decision-directed refinement on top of the two pilots (not acceptance) also holds."""
    with Pool() as p:
        res = p.map(_ppm_job, [(CPS[0], 100, 'dd'), (CPS[-1], -100, 'dd')])
    for r in res:
        print(f"two pilots + DD CP={sg.OFDMConfig(cfg_cp=r['cfg_cp']).CP} ppm={r['ppm']:+d}: BER {r['ber']}, EVM {r['evm']:.1f} dB")
        assert r['ber'] == 0.0


def test_dc_offset():
    for cfg_cp in CPS:
        cfg = sg.OFDMConfig(cfg_cp=cfg_cp)
        for dc in (-20.0, 20.0):
            r = ol.chip_run(cfg, lock_bound_sym(cfg) + 1100, seed=77, snr_db=30, dc_offset=dc, **ACCEPT)
            print(f"CP={cfg.CP} dc={dc:+.0f} LSB: lock {r['first_lock_sym']:.0f} sym, BER {r['ber']}, clip {r['clip']:.4f}")
            assert r['ber'] == 0.0 and r['frames'] >= 1000


# ---- BER vs SNR, reference paths, dB gaps --------------------------------------------------------------
def test_ber_curve_and_gaps():
    rows = ol.ber_curves(snrs=(8, 10, 12, 14, 16, 18, 22), nsym=4000, seed=777, modes=('v1', 'v1_raw'))
    ref_rows = ol.recorded_curves(sg.V1.N)
    for cp in CPS:
        rs = [r for r in rows if r['cfg_cp'] == cp]
        snrs = [r['snr'] for r in rs]
        s = {k: ol.snr_at_ber(snrs, [r[k] for r in rs]) for k in ('theory_v1', 'A_v1', 'B_v1', 'C_v1', 'C_v1_raw')}
        print(f"CP={sg.OFDMConfig(cfg_cp=cp).CP}: SNR @ BER 1e-3: theory {s['theory_v1']:.2f}, A float {s['A_v1']:.2f}, "
              f"B 8-bit {s['B_v1']:.2f}, C chip {s['C_v1']:.2f}; without pilot averaging C {s['C_v1_raw']:.2f}")
        for r in rs:
            fmt = lambda v: 'none' if v is None else f'{v:.1e}'
            print(f"   {r['snr']:2d} dB: theory {fmt(r['theory_v1'])}  A {fmt(r['A_v1'])}  B {fmt(r['B_v1'])}  C {fmt(r['C_v1'])}"
                  f"  C(no avg) {fmt(r['C_v1_raw'])}   EVM bin 5 A/B/C {r['evmA_v1']:.1f}/{r['evmB_v1']:.1f}/"
                  f"{'-' if r['evmC_v1'] is None else round(r['evmC_v1'], 1)} dB")
        assert abs(s['A_v1'] - s['theory_v1']) <= 0.5          # averaged two-pilot estimate ~ ideal knowledge
        # 8-bit ADC + fixed-point FFT, B - A. The band is derived, not hand-set:
        # centre = the gap the model measured for this N in its recorded characterisation (ofdm_loop.py ->
        # study_results/ber_curves[_nN].json: 1.4-1.8 dB at N = 32, 0.38 dB at N = 16); width = 3 sigma of the
        # difference between that gap and this run's, sigma from the Poisson error counts at the interpolation
        # points (A and B share the noise realisation, so treating them as independent is conservative).
        rr = [r for r in ref_rows if r['cfg_cp'] == cp]
        rsn = [r['snr'] for r in rr]
        gap, gap_ref = s['B_v1'] - s['A_v1'], ol.snr_at_ber(rsn, [r['B_v1'] for r in rr]) - ol.snr_at_ber(rsn, [r['A_v1'] for r in rr])
        sig = lambda R, sn, p: ol.snr_at_ber_sigma(sn, [r[f'{p}_v1'] for r in R], [r[f'{p}_v1_bits'] for r in R])
        tol = 3 * np.sqrt(sig(rs, snrs, 'A') ** 2 + sig(rs, snrs, 'B') ** 2 + sig(rr, rsn, 'A') ** 2 + sig(rr, rsn, 'B') ** 2)
        print(f"   quantisation gap B - A: {gap:.2f} dB; recorded for N = {sg.V1.N}: {gap_ref:.2f} dB; band +-{tol:.2f} dB (3 sigma)")
        assert abs(gap - gap_ref) <= tol
        assert -0.5 <= s['C_v1'] - s['B_v1'] <= 0.5             # chip timing recovery
        assert s['C_v1_raw'] - s['C_v1'] >= 0.7                 # pilot averaging gain
        assert all(r['C_v1'] == 0.0 for r in rs if r['snr'] >= 18)


# ---- stretch rows (documentation only, not gating) ---------------------------------------------------------
STRETCH = {
    '2 data (5, 7), 16-QAM': dict(data_bins=(5, 7)),         # bin 7 is extrapolated one bin beyond pilot 6
    '2 data (3, 5), 16-QAM': dict(data_bins=(3, 5)),         # bin 3 extrapolated one bin below pilot 4
    '64-QAM at 5': dict(modulation='64QAM'),
}


def _stretch_job(a):
    name, ppm, cfg_cp = a
    cfg = sg.OFDMConfig(**{**sg.V1.__dict__, **STRETCH[name], 'cfg_cp': cfg_cp})
    out = {}
    for tag, kw in (('accept', ACCEPT), ('dd', dict(dd=True, avg_window=dm.AVG_WINDOW))):
        r = ol.chip_run(cfg, 2400, seed=77, snr_db=30, ppm=ppm, **kw)
        out[tag] = dict(ber=r['ber'], evm=r['evm_db'], frames=r['frames'])
    return dict(name=name, ppm=ppm, cfg_cp=cfg_cp, a=cfg.a, lock=r['first_lock_sym'], **out)


def test_stretch_rows_documentation():
    """Report only (stretch rows): runs, locks, prints BER/EVM per bin; the only assertion is lock."""
    jobs = [(n, ppm, c) for n in STRETCH for ppm in (0, 100) for c in CPS]
    with Pool() as p:
        res = p.map(_stretch_job, jobs)
    for r in res:
        ev = lambda d: ", ".join(f"{k}: {v:.1f}" for k, v in d.items())
        print(f"{r['name']:22s} CP={sg.OFDMConfig(cfg_cp=r['cfg_cp']).CP} ppm={r['ppm']:+d} a={r['a']:.1f}: lock {r['lock']:.0f} sym | "
              f"two pilots: BER {r['accept']['ber']:.1e} EVM {ev(r['accept']['evm'])} | + DD: BER {r['dd']['ber']:.1e} EVM {ev(r['dd']['evm'])}")
        assert r['accept']['frames'] >= 1000


# ---- constellation pictures ----------------------------------------------------------------------------------
def test_constellation_png():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    ref = np.array([sg.qam_map(np.array([(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1]), '16QAM') for v in range(16)])
    for cfg_cp in CPS:
        panels = [('legacy signal, one pilot,\ndesign ramp only', sg.LEGACY_1P, dict(dd=False, avg_window=1)),
                  ('v1: two pilots,\nno averaging', sg.V1, dict(dd=False, avg_window=1)),
                  ('v1: two pilots, W = 16\n(acceptance)', sg.V1, ACCEPT)]
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
        for ax, (title, base, kw) in zip(axs, panels):
            cfg = sg.OFDMConfig(**{**base.__dict__, 'cfg_cp': cfg_cp})
            r = ol.chip_run(cfg, 2400, seed=55, snr_db=25, ppm=100, **kw)
            pts = r['points'][5]
            ax.plot(pts.real, pts.imag, '.', ms=1.5, alpha=0.4)
            ax.plot(ref.real, ref.imag, 'r+', ms=10)
            ax.set_aspect('equal'); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3)
            ax.set_title(f"{title}\nBER {r['ber']:.1e}, EVM {r['evm_db'][5]:.1f} dB", fontsize=9)
        fig.suptitle(f"bin 5, 16-QAM, CP = {cfg.CP}, 25 dB, +100 ppm, chip model -> pins -> host demodulator")
        fn = os.path.join(OUT, f"constellation_n{cfg.N}_cp{cfg.CP}.png")
        fig.tight_layout(); fig.savefig(fn, dpi=100); plt.close(fig)
        assert os.path.getsize(fn) < 512 * 1024
