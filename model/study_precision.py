# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision study of the FFT datapath (model only). Options, as fft_model.SDF parameters (the chip adopted b):
    baseline  W = 8 everywhere, /2 per stage (gain 1/N), Q1.7 twiddles        (the earlier RTL, bit-exact)
    b         last butterfly without /2, saturated to 8 bits (gain 2/N)
    c9        9-bit internal datapath, Q1.8 twiddles, convergent rounding to the 8-bit output
    c10       10-bit internal datapath, Q1.9 twiddles, convergent rounding to the 8-bit output
Signal: v1 (pilots 4/6, data 5, 37 LSB per carrier), two-pilot demodulator with 16-frame averaging (acceptance).

    python3 study_precision.py noise     # bin noise vs ideal FFT, EVM floor per bin
    python3 study_precision.py ber       # BER vs SNR, 16/64/256-QAM at k = 5, 0 ppm and +-100 ppm with DC +-20
    python3 study_precision.py sat       # option b: where bins start to saturate
    python3 study_precision.py two       # second data carrier (5 and 7, 4 carriers at 32 LSB)
Results: study_results/precision_<what>.json and markdown on stdout (`report` collects them on one page).
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool

import fft_model as fm
import signal_gen as sg
import demod as dm
import ofdm_loop as ol

N = 16
# explicit settings (the chip adopted b, so last_div2=False is now the SDF default; the study JSONs used these)
OPTS = {'baseline': dict(last_div2=True), 'b': dict(last_div2=False), 'c9': dict(Wi=9, last_div2=True),
        'c10': dict(Wi=10, last_div2=True), 'b+c9': dict(Wi=9, last_div2=False),
        'b+c10': dict(Wi=10, last_div2=False)}               # b+c*: not asked, information only
GAIN = {'baseline': 1, 'b': 2, 'c9': 1, 'c10': 1, 'b+c9': 2, 'b+c10': 2}   # output gain in units of 1/N
ACCEPT = dict(dd=False, avg_window=dm.AVG_WINDOW)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study_results')
MODS = ('16QAM', '64QAM', '256QAM')
SNRS = {'16QAM': range(8, 27, 2), '64QAM': range(14, 37, 2), '256QAM': range(20, 49, 2)}
CONDS = {'0 ppm': (0.0, 0.0), '+100 ppm, DC +20': (100.0, 20.0), '-100 ppm, DC -20': (-100.0, -20.0)}
NSYM = 6000


def md(rows, cols):
    f = lambda v: '—' if v is None else (f'{v:.2f}' if isinstance(v, float) else str(v))
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(f(r.get(c)) for c in cols) + ' |' for r in rows])


# ---- 1. bin noise vs the ideal FFT, EVM floor per bin --------------------------------------------------------
def noise():
    rng = np.random.default_rng(3)
    x = rng.integers(-128, 128, N * 4000)
    cfg = sg.V1
    adc, tx, bd = sg.generate(cfg, 2001, 11, snr_db=None, phase0=0.0)
    xs, yf = sg.to_signed(adc), bd['y_float']
    adc_d, _, _ = sg.generate(cfg, 2001, 12, snr_db=30, phase0=0.0)     # dithered: 30 dB AWGN
    xd = sg.to_signed(adc_d)
    start = cfg.CP - cfg.CP // 2                                 # ideal window, CP/2 into the CP
    rows = []
    for name, kw in OPTS.items():
        g = GAIN[name]
        m = fm.SDF(N, **kw)
        out = np.array([complex(*m.step(int(v))) for v in x])[N - 1:]
        E = np.concatenate([out[f * N:(f + 1) * N] - fm.reference(x[f * N:(f + 1) * N], N, gain=1) * g for f in range(3998)])
        r = dict(option=name, rand_max=float(np.abs(E).max()), rand_rms=float(np.sqrt(np.mean(np.abs(E) ** 2))),
                 rand_rms_rel=float(np.sqrt(np.mean(np.abs(E) ** 2))) / g, sat_rand=m.sat)
        # v1 signal, ideal windows: per-bin error vs float FFT of (a) the 8-bit samples, (b) the analog waveform
        eq, ea, ed = np.zeros(N // 2), np.zeros(N // 2), np.zeros(N // 2)
        nw = 0
        sat = 0

        def window_bins(w):
            mm = fm.SDF(N, **kw)
            o = [mm.step(int(v)) for v in list(w) + [0] * N][N - 1:2 * N - 1]
            return {fm.bitrev(t, 4): complex(*o[t]) for t in range(0, N, 2)}, mm.sat
        for j in range(2000):
            w0 = j * cfg.L + start
            w8, wa, wd = xs[w0:w0 + N], yf[w0:w0 + N], xd[w0:w0 + N]
            got, st = window_bins(w8)
            gotd, std = window_bins(wd)
            sat += st + std
            X8, Xa, Xd = np.fft.fft(w8) / N * g, np.fft.fft(wa) / N * g, np.fft.fft(wd) / N * g
            for k in range(N // 2):
                eq[k] += abs(got[k] - X8[k]) ** 2
                ea[k] += abs(got[k] - Xa[k]) ** 2
                ed[k] += abs(gotd[k] - Xd[k]) ** 2
            nw += 1
        ref = (g * cfg.a / 2) ** 2                                 # carrier power in a bin (|c| = 1)
        r.update({f'fft_k{k}': float(10 * np.log10(eq[k] / nw / ref)) for k in range(1, N // 2)})
        r.update({f'tot_k{k}': float(10 * np.log10(max(ea[k], 1e-30) / nw / ref)) for k in range(1, N // 2)})
        r.update({f'dith_k{k}': float(10 * np.log10(ed[k] / nw / ref)) for k in range(1, N // 2)})
        r['sat_v1'] = sat
        rows.append(r)
    return rows


# ---- 2. BER vs SNR through the chip model and the acceptance demodulator ---------------------------------------
def _ber_job(a):
    opt, mod, cond, snr, seed = a
    ppm, dc = CONDS[cond]
    cfg = sg.OFDMConfig(modulation=mod)
    r = ol.chip_run(cfg, NSYM, seed, snr_db=snr, ppm=ppm, phase0=3.3 + (seed % 7), dc_offset=dc, prec=OPTS[opt], **ACCEPT)
    return dict(opt=opt, mod=mod, cond=cond, snr=snr, ber=r['ber'], nbits=r['nbits'], frames=r['frames'],
                evm=r['evm_db'][5], intervals=r['intervals'], theory=ol.theory_ber(cfg, snr))


def ber():
    jobs = [(o, m, c, s, 9000 + 17 * s + 101 * MODS.index(m) + 7 * list(CONDS).index(c))
            for o in OPTS for m in MODS for c in CONDS for s in SNRS[m]]
    with Pool() as p:
        res = p.map(_ber_job, jobs, chunksize=2)
    return res


def ber_summary(res):
    rows = []
    for mod in MODS:
        th = [r for r in res if r['mod'] == mod and r['opt'] == 'baseline' and r['cond'] == '0 ppm']
        s_th = ol.snr_at_ber([r['snr'] for r in th], [r['theory'] for r in th])
        for opt in OPTS:
            row = dict(mod=mod, option=opt, theory=s_th)
            for cond in CONDS:
                rs = sorted((r for r in res if r['mod'] == mod and r['opt'] == opt and r['cond'] == cond), key=lambda r: r['snr'])
                s = ol.snr_at_ber([r['snr'] for r in rs], [r['ber'] for r in rs])
                sig = ol.snr_at_ber_sigma([r['snr'] for r in rs], [r['ber'] for r in rs], [r['nbits'] for r in rs])
                row[cond] = s
                row[cond + ' sigma'] = sig
                row[cond + ' floor'] = rs[-1]['ber']
                row[cond + ' evm_top'] = rs[-1]['evm']
                row[cond + ' unlocked'] = sum(r['intervals'] != 1 for r in rs)
            row['loss'] = None if (row['0 ppm'] is None or s_th is None) else row['0 ppm'] - s_th
            rows.append(row)
    return rows


# ---- 3. option b: saturation thresholds ----------------------------------------------------------------------
def sat():
    kw = OPTS['b']
    out = {}
    # DC
    first = None
    for v in range(0, 128):
        for sgn in (1, -1):
            m = fm.SDF(N, **kw)
            for _ in range(3 * N):
                m.step(sgn * v if sgn * v >= -128 else -128)
            if m.sat and first is None:
                first = sgn * v
    out['dc_first_saturating_level'] = first
    # single real tones (bin-centred and half-bin), amplitude sweep, 8 random phases each
    rng = np.random.default_rng(1)
    tones = {}
    for f in [k for k in range(1, N // 2)] + [k + 0.5 for k in range(0, N // 2)]:
        thr = None
        for A in range(40, 128):
            hit = False
            for ph in rng.uniform(0, 2 * np.pi, 8):
                n = np.arange(4 * N)
                x = np.clip(np.round(A * np.cos(2 * np.pi * f * n / N + ph)), -128, 127).astype(int)
                m = fm.SDF(N, **kw)
                for v in x:
                    m.step(int(v))
                hit |= m.sat > 0
            if hit:
                thr = A
                break
        tones[str(f)] = thr
    out['tone_first_saturating_amplitude'] = tones
    # the v1 signal scaled up to the ADC limit (3 carriers: a <= 42), with DC +-20
    v1 = {}
    for a in (36.95, 40.0, 42.0):
        for dc in (0.0, 20.0, -20.0):
            cfg = sg.OFDMConfig(amp=a)
            adc, tx, _ = sg.generate(cfg, 1500, 5, snr_db=None, phase0=1.0, dc_offset=dc)
            m = fm.SDF(N, **kw)
            for v in sg.to_signed(adc):
                m.step(int(v))
            v1[f'a={a:g} dc={dc:+g}'] = dict(sat=m.sat, clip=tx['clip_fraction'])
    out['v1_scaled'] = v1
    # full-scale random input: saturation rate per output component
    x = np.random.default_rng(2).integers(-128, 128, N * 4000)
    m = fm.SDF(N, **kw)
    for v in x:
        m.step(int(v))
    out['random_fullscale_sat_per_component'] = m.sat / (2 * len(x))
    return out


# ---- 4. second data carrier ---------------------------------------------------------------------------------
def _two_job(a):
    opt, mod, snr, seed = a
    cfg = sg.OFDMConfig(modulation=mod, data_bins=(5, 7))
    r = ol.chip_run(cfg, NSYM, seed, snr_db=snr, phase0=4.1, prec=OPTS[opt], **ACCEPT)
    return dict(opt=opt, mod=mod, snr=snr, ber=r['ber'], nbits=r['nbits'], evm5=r['evm_db'][5], evm7=r['evm_db'][7],
                theory=ol.theory_ber(cfg, snr))


def two():
    snrs = {'16QAM': range(10, 29, 2), '64QAM': range(16, 39, 2)}
    jobs = [(o, m, s, 7000 + s) for o in OPTS for m in snrs for s in snrs[m]]
    with Pool() as p:
        res = p.map(_two_job, jobs, chunksize=2)
    rows = []
    for mod in snrs:
        for opt in OPTS:
            rs = sorted((r for r in res if r['mod'] == mod and r['opt'] == opt), key=lambda r: r['snr'])
            rows.append(dict(mod=mod, option=opt, a=32.0,
                             snr_1e3=ol.snr_at_ber([r['snr'] for r in rs], [r['ber'] for r in rs]),
                             theory=ol.snr_at_ber([r['snr'] for r in rs], [r['theory'] for r in rs]),
                             floor=rs[-1]['ber'], evm5_top=rs[-1]['evm5'], evm7_top=rs[-1]['evm7']))
    return res, rows


if __name__ == '__main__' and sys.argv[1] != 'report':
    what = sys.argv[1]
    t0 = time.time()
    if what == 'noise':
        rows = noise()
        json.dump(rows, open(os.path.join(OUT, 'precision_noise.json'), 'w'), indent=1)
        print(md(rows, ['option', 'rand_max', 'rand_rms', 'rand_rms_rel', 'sat_rand', 'sat_v1']))
        print(md(rows, ['option'] + [f'fft_k{k}' for k in range(1, 8)]))
        print(md(rows, ['option'] + [f'tot_k{k}' for k in range(1, 8)]))
        print(md(rows, ['option'] + [f'dith_k{k}' for k in range(1, 8)]))
    elif what == 'ber':
        res = ber()
        rows = ber_summary(res)
        json.dump(dict(points=res, summary=rows), open(os.path.join(OUT, 'precision_ber.json'), 'w'), indent=1)
        cols = ['mod', 'option', 'theory'] + [c for cond in CONDS for c in (cond, cond + ' floor')] + ['0 ppm evm_top', 'loss']
        print(md(rows, cols))
    elif what == 'sat':
        r = sat()
        json.dump(r, open(os.path.join(OUT, 'precision_sat.json'), 'w'), indent=1)
        print(json.dumps(r, indent=1))
    elif what == 'two':
        res, rows = two()
        json.dump(dict(points=res, summary=rows), open(os.path.join(OUT, 'precision_two.json'), 'w'), indent=1)
        print(md(rows, ['mod', 'option', 'a', 'theory', 'snr_1e3', 'floor', 'evm5_top', 'evm7_top']))
    print(f"[{what}: {time.time() - t0:.0f} s]")


# ---- 5. one-page summary (reads the JSONs above + area_timing.json: Yosys area, flops, ss slack per option) -----
POST_BASE, SYN_RATIO, CORE_AREA, CEIL, SS_BASE = 23299.0, 23299.0 / 19778.0, 34255.0, 0.80, 5.58


def report():
    J = lambda f: json.load(open(os.path.join(OUT, f)))
    nz, bs, tw, st = J('precision_noise.json'), J('precision_ber.json'), J('precision_two.json'), J('precision_sat.json')
    at = {r['option']: r for r in json.load(open(os.path.join(OUT, 'area_timing.json')))}
    base_at = at['baseline_src']
    pts = bs['points']
    fl = lambda v: '0' if v == 0 else f'{v:.1e}'
    def s1e3(opt, mod, cond):
        rs = sorted((r for r in pts if r['opt'] == opt and r['mod'] == mod and r['cond'] == cond), key=lambda r: r['snr'])
        return ol.snr_at_ber([r['snr'] for r in rs], [r['ber'] for r in rs]), rs
    lines = ['# Precision study — model only; area/timing from Yosys synthesis of each option', '']
    hdr = ['option', 'output gain', 'EVM floor 64-QAM (dB, demod., 36 dB)', 'FFT floor bin 5 (dB, dithered)',
           '16-QAM SNR@1e-3 (0 / worst ±100 ppm+DC)', '64-QAM SNR@1e-3', '256-QAM SNR@1e-3 (BER at 48 dB)',
           '64-QAM, 2 data carriers', 'synth Δ µm² (flops)', 'est. post-layout µm² (% core)', 'est. ss slack ns']
    lines += ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    th = {m: ol.snr_at_ber([r['snr'] for r in s1e3('baseline', m, '0 ppm')[1]], [r['theory'] for r in s1e3('baseline', m, '0 ppm')[1]]) for m in MODS}
    two_rows = {(r['mod'], r['option']): r for r in tw['summary']}
    for opt in OPTS:
        cells = [opt, f'{GAIN[opt]}/N']
        _, r64 = s1e3(opt, '64QAM', '0 ppm')
        cells.append(f"{r64[-1]['evm']:.1f}")
        cells.append(f"{next(r for r in nz if r['option'] == opt)['dith_k5']:.1f}")
        s16 = [s1e3(opt, '16QAM', c)[0] for c in CONDS]
        cells.append(f"{s16[0]:.2f} / {max(s16[1:]):.2f}")
        s64 = [s1e3(opt, '64QAM', c)[0] for c in CONDS]
        cells.append(f"{s64[0]:.2f} / {max(s64[1:]):.2f}")
        s256, r256 = s1e3(opt, '256QAM', '0 ppm')
        cells.append(('floor' if s256 is None else f'{s256:.1f}') + f" ({fl(r256[-1]['ber'])})")
        t2 = two_rows[('64QAM', opt)]['snr_1e3']
        cells.append('—' if t2 is None else f'{t2:.2f}')
        a = at[opt if opt != 'baseline' else 'baseline_src']
        d = a['area'] - base_at['area']
        post = POST_BASE + d * SYN_RATIO
        cells.append(f"{d:+.0f} ({a['flops'] - base_at['flops']:+d})")
        cells.append(f"{post:,.0f} ({100 * post / CORE_AREA:.1f} %)")
        cells.append(f"{SS_BASE + a['slack_ss'] - base_at['slack_ss']:+.1f}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f"Theory (v1, 37 LSB per carrier, ideal 1/N FFT): 16-QAM {th['16QAM']:.2f} dB, 64-QAM {th['64QAM']:.2f} dB, "
              f"256-QAM {th['256QAM']:.2f} dB; 2 data carriers (32 LSB): 64-QAM {two_rows[('64QAM', 'baseline')]['theory']:.2f} dB.",
              f"80 % of the core = {CEIL * CORE_AREA:,.0f} µm². Post-layout = the first N = 16 layout's "
              f"{POST_BASE:,.0f} µm² + synth Δ × {SYN_RATIO:.3f} (its post-layout / synthesis); ss slack = its "
              f"post-layout +{SS_BASE} ns + Δ(pre-layout core slack at ss_100C_1v60).", '']
    lines += ['Option b saturation (model): ' + f"DC |level| >= {abs(st['dc_first_saturating_level'])} LSB saturates bin 0; "
              'single real tones up to full scale: ' + ', '.join(f"f={k}: {v if v else 'none'}" for k, v in st['tone_first_saturating_amplitude'].items())
              + f"; v1 at the ADC limit (a <= 42) with DC ±20: {sum(v['sat'] for v in st['v1_scaled'].values())} saturations; "
              f"full-scale random input: {st['random_fullscale_sat_per_component']:.1e} per output component.", '']
    nzc = ['option'] + [f'dith_k{k}' for k in range(1, 8)]
    lines += ['FFT fixed-point error per bin vs the float FFT of the same 8-bit samples (v1 + 30 dB AWGN, dB re. carrier corner power):',
              md(nz, nzc), '', 'Same, noise-free v1 (deterministic patterns; worst case for bins 1-3):',
              md(nz, ['option'] + [f'fft_k{k}' for k in range(1, 8)]), '',
              'Random full-scale input, |error| at the 8-bit output (LSB; rel = divided by the output gain):',
              md(nz, ['option', 'rand_max', 'rand_rms', 'rand_rms_rel'])]
    txt = '\n'.join(lines) + '\n'
    open(os.path.join(OUT, 'precision.md'), 'w').write(txt)
    return txt


if __name__ == '__main__' and sys.argv[1] == 'report':
    print(report())
