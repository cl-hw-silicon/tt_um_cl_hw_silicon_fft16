# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CP timing block at N = 16 (CP fixed at 4, L = 20, the N = 32 constants rescaled to K*CP = 64: K = 16,
T_lock = 48, T_unlock = 24). Cycle-exact CPTiming on the v1 signal (signal_gen, two pilots + 16-QAM).

    python3 study_n16.py acq      # acquisition: 200 random phases x 10/20/30 dB, false locks (noise, no CP)
    python3 study_n16.py hm       # tracking at +-100 ppm: hysteresis H x check interval M (re-validates M = 16, H = 8)
    python3 study_n16.py edges    # the datasheet's edge cases: LOCK fall after the signal stops (signal -> noise),
                                  # lock time on a constant input, tracking range at +-300 / +-400 ppm
Results: study_results/n16_*.json and markdown tables on stdout.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
import signal_gen as sg
from symgen import MiniOFDM, noise_stream
from timing import CPTiming
from study_timing import window_error, pct, md

N = 16
CFG = sg.OFDMConfig(N=N)
L, CP = CFG.L, CFG.CP
K = 4 * N // CP
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study_results')


def acq_trial(args):
    snr, seed = args
    rng = np.random.default_rng(seed)
    adc, _, bd = sg.generate(CFG, 3 * L * K + 8, seed, snr_db=snr, phase0=float(rng.uniform(0, L)))
    tim = CPTiming(N=N)
    for i, v in enumerate(sg.to_signed(adc)):
        if tim.step(int(v))['lock_rise']:
            e, clean = window_error(bd['t'], i + 2, L, CP, tim.delta, N=N)   # first window starts at ph = 0, 2 cycles on
            return dict(locked=1, t=(i + 1) / L, e=e, clean=clean)
    return dict(locked=0)


def false_trial(args):
    kind, seed = args
    if kind == 'noise':
        x = noise_stream(500_000, sigma=40.0, seed=seed)
    else:                                              # same carriers, no cyclic prefix
        x, _ = MiniOFDM(N=N, CP=0, seed=seed).stream(nsym=30_000, offset=0.0, snr_db=30)
    tim = CPTiming(N=N)
    return sum(tim.step(int(v))['lock_rise'] for v in x), len(x)


def track_trial(args):
    H, M, ppm, snr, seed, nsym = args
    rng = np.random.default_rng(seed)
    adc, _, bd = sg.generate(CFG, nsym, seed, snr_db=snr, ppm=ppm, phase0=float(rng.uniform(0, L)))
    x, t = sg.to_signed(adc), bd['t']
    tim = CPTiming(N=N, M=M, hyst=H)
    late = early = unlocks = windows = unclean = 0
    emin = emax = 0.0
    for i, v in enumerate(x):
        r = tim.step(int(v))
        if r['slip']:
            late += r['slip_dir'] > 0
            early += r['slip_dir'] < 0
        unlocks += r['unlock']
        if r['ph'] == 0 and r['lock'] and i + N <= len(x):
            e, clean = window_error(t, i, L, CP, tim.delta, N=N)
            windows += 1
            unclean += not clean
            emin, emax = min(emin, e), max(emax, e)
    return dict(H=H, M=M, ppm=ppm, snr=snr, late=late, early=early, unlocks=unlocks, windows=windows,
                unclean=unclean, emin=emin, emax=emax)


def run_acq(seeds=200):
    jobs = [(snr, 7000 + 31 * s + snr) for snr in (10, 20, 30) for s in range(seeds)]
    fl = [(k, 8000 + s) for k in ('noise', 'nocp') for s in range(9)]
    with Pool() as p:
        ra = p.map(acq_trial, jobs, chunksize=4)
        rf = p.map(false_trial, fl, chunksize=1)
    row = dict(N=N, CP=CP, L=L, K=K, T_lock=CPTiming(N=N).T_lock, T_unlock=CPTiming(N=N).T_unlock, H=8, M=16)
    for snr in (10, 20, 30):
        rs = [r for (s, _), r in zip(jobs, ra) if s == snr]
        lk = [r for r in rs if r['locked']]
        tt = [r['t'] for r in lk]
        row[f'lock%@{snr}'] = round(100 * len(lk) / len(rs), 1)
        row[f't_med@{snr}'] = float(np.median(tt)) if tt else float('nan')
        row[f't_p95@{snr}'] = pct(tt, 95)
        row[f'unclean@{snr}'] = sum(1 for r in lk if not r['clean'])
    for kind in ('noise', 'nocp'):
        rr = [r for (k, _), r in zip(fl, rf) if k == kind]
        row[f'false_{kind}'] = sum(r[0] for r in rr)
        row[f'ms_{kind}'] = round(sum(r[1] for r in rr) / 32e3)
    return [row]


def run_hm(seeds=3, nsym=10_000):
    jobs = [(H, M, ppm, snr, 9000 + s, nsym) for H in (2, 4, 8, 12) for M in (8, 16, 32) for ppm in (-100, 100)
            for snr in (10, 20) for s in range(seeds)]
    with Pool() as p:
        res = p.map(track_trial, jobs, chunksize=1)
    rows = []
    for H in (2, 4, 8, 12):
        for M in (8, 16, 32):
            for snr in (10, 20):
                rs = [r for r in res if (r['H'], r['M'], r['snr']) == (H, M, snr)]
                wrong = sum(r['early'] if r['ppm'] < 0 else r['late'] for r in rs)   # direction check below
                right = sum(r['late'] if r['ppm'] < 0 else r['early'] for r in rs)
                rows.append(dict(H=H, M=M, snr=snr, runs=len(rs), unlocks=sum(r['unlocks'] for r in rs),
                                 unclean=sum(r['unclean'] for r in rs), slips=right + wrong,
                                 wrong_way=min(right, wrong), e_range=f"[{min(r['emin'] for r in rs):+.2f}, {max(r['emax'] for r in rs):+.2f}]"))
    return rows


def unlock_trial(seed):
    """v1 at 30 dB until locked + a random 40..120 further symbols, then noise only: symbols until LOCK falls."""
    rng = np.random.default_rng(seed)
    adc, _, _ = sg.generate(CFG, 3 * L * K + 8, seed, snr_db=30, phase0=float(rng.uniform(0, L)))
    tim = CPTiming(N=N)
    x = sg.to_signed(adc)
    for i, v in enumerate(x):
        if tim.step(int(v))['lock_rise']:
            break
    stop = i + 1 + L * int(rng.integers(40, 121))
    for v in x[i + 1:stop]:
        tim.step(int(v))
    assert tim.lock, 'lost lock on the clean signal'
    for j, v in enumerate(noise_stream(L * 2000, sigma=40.0, seed=seed + 1)):
        if tim.step(int(v))['unlock']:
            return (j + 1) / L
    return float('nan')


def const_trial(code):
    """Constant offset-binary input `code`: symbols until LOCK rises (every phase matches the CP test)."""
    tim = CPTiming(N=N)
    for i in range(L * 200):
        if tim.step(code - 128)['lock_rise']:
            return (i + 1) / L
    return float('nan')


def run_edges(runs=20, nsym=10_000):
    with Pool() as p:
        un = p.map(unlock_trial, [9500 + s for s in range(runs)], chunksize=1)
        cs = p.map(const_trial, (0x00, 0x40, 0x80, 0xA5, 0xFF))
        tr = p.map(track_trial, [(8, 16, ppm, 20, 9600 + s, nsym) for ppm in (-400, -300, 300, 400) for s in range(3)],
                   chunksize=1)
    row = dict(unlock_runs=runs, unlock_sym_min=min(un), unlock_sym_max=max(un), unlock_sym_med=float(np.median(un)),
               const_codes='0x00 0x40 0x80 0xA5 0xFF', const_lock_sym_min=min(cs), const_lock_sym_max=max(cs))
    for ppm in (-400, -300, 300, 400):
        rs = [r for r in tr if r['ppm'] == ppm]
        row[f'unlocks@{ppm:+d}ppm'] = sum(r['unlocks'] for r in rs)
        row[f'runs@{ppm:+d}ppm'] = f"{len(rs)} x {nsym} symbols, 20 dB"
    return [row]


if __name__ == '__main__':
    what = sys.argv[1]
    t0 = time.time()
    if what == 'acq':
        tab = run_acq()
        cols = ['N', 'CP', 'L', 'K', 'T_lock', 'T_unlock', 'lock%@10', 't_med@10', 't_p95@10', 'unclean@10', 'lock%@20',
                't_med@20', 't_p95@20', 'unclean@20', 'lock%@30', 't_med@30', 't_p95@30', 'unclean@30',
                'false_noise', 'ms_noise', 'false_nocp', 'ms_nocp']
    elif what == 'edges':
        tab = run_edges()
        cols = list(tab[0])
    else:
        tab = run_hm()
        cols = ['H', 'M', 'snr', 'runs', 'unlocks', 'unclean', 'slips', 'wrong_way', 'e_range']
    json.dump(tab, open(os.path.join(OUT, f'n16_{what}.json'), 'w'), indent=1)
    print(md(tab, cols))
    print(f"[n16 {what}: {time.time() - t0:.0f} s]")
