# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Parameter study for the CP timing block: acquisition time, lock error, false lock,
tracking under sample-rate offset. Uses the cycle-exact CPTiming model (no shortcuts).

    python3 study_timing.py acq      # K x lock_frac x SNR x CP, random initial phase
    python3 study_timing.py noise    # false lock on pure noise
    python3 study_timing.py track    # +-100 ppm, 10^4 symbols, M x hysteresis x unlock_frac
Results are written to study_results/*.json and printed as markdown tables.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from symgen import MiniOFDM, noise_stream
from timing import CPTiming

N = 32
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study_results')


def window_error(t, i0, L, CP, delta, N=N):
    """Timing error of the FFT window whose first sample is stream index i0: e = u0 - (CP - delta)
    (u0 = position of that sample in its TX symbol), and whether all N samples are in one symbol."""
    tt = t[i0:i0 + N]
    j = np.floor(tt / L)
    u0 = tt[0] - j[0] * L
    e = float(u0 - (CP - delta))
    e = (e + L / 2) % L - L / 2                      # wrap to [-L/2, L/2): signed error
    return e, bool(len(tt) == N and np.all(j == j[0]))


def acq_trial(args):
    cfg_cp, K, lock_frac, snr, seed, max_sym = args
    CP = N // 4 if cfg_cp == 0 else N // 8
    L = N + CP
    rng = np.random.default_rng(seed)
    offset = float(rng.uniform(0, L))
    g = MiniOFDM(N=N, CP=CP, seed=seed)
    x, info = g.stream(nsym=max_sym + 4, offset=offset, snr_db=snr)
    tim = CPTiming(N=N, cfg_cp=cfg_cp, K=K, M=16, lock_frac=lock_frac, unlock_frac=0.5)
    hyps = 0
    for i, v in enumerate(x):
        r = tim.step(int(v))
        hyps += r['hyp_adv']
        if r['lock_rise']:
            # first FFT window starts at the next ph == 0 cycle: i + 2 (ph L-2 -> L-1 -> 0)
            e, clean = window_error(info['t'], i + 2, L, CP, tim.delta)
            return dict(locked=1, t_sym=(i + 1) / L, e=e, clean=clean, hyps=hyps, offset=offset)
    return dict(locked=0, t_sym=len(x) / L, e=None, clean=None, hyps=hyps, offset=offset)


def noise_trial(args):
    cfg_cp, K, lock_frac, seed, nsamp = args
    tim = CPTiming(N=N, cfg_cp=cfg_cp, K=K, M=16, lock_frac=lock_frac, unlock_frac=0.5)
    x = noise_stream(nsamp, sigma=40.0, seed=seed)
    locks = 0
    for v in x:
        locks += tim.step(int(v))['lock_rise']
    return dict(locks=locks, nsamp=nsamp)


def track_trial(args):
    cfg_cp, K, lock_frac, unlock_frac, M, H, ppm, snr, seed, nsym = args
    CP = N // 4 if cfg_cp == 0 else N // 8
    L = N + CP
    rng = np.random.default_rng(seed)
    g = MiniOFDM(N=N, CP=CP, seed=seed)
    x, info = g.stream(nsym=nsym, offset=float(rng.uniform(0, L)), ppm=ppm, snr_db=snr)
    tim = CPTiming(N=N, cfg_cp=cfg_cp, K=K, M=M, lock_frac=lock_frac, unlock_frac=unlock_frac, hyst=H)
    t = info['t']
    lock_t = None
    slips = {+1: 0, -1: 0}
    unlocks = locked_windows = unclean = 0
    emax = emin = 0.0
    for i, v in enumerate(x):
        r = tim.step(int(v))
        if r['lock_rise'] and lock_t is None:
            lock_t = i
        if r['slip']:
            slips[r['slip_dir']] += 1
        unlocks += r['unlock']
        # an FFT window starts at every cycle with ph == 0 while locked (run region ph 0..N-1)
        if r['ph'] == 0 and r['lock'] and i + N <= len(x):
            e, clean = window_error(t, i, L, CP, tim.delta)
            locked_windows += 1
            unclean += (not clean)
            emax, emin = max(emax, e), min(emin, e)
    drift = (len(x) * ppm * 1e-6)                       # samples of accumulated drift over the run
    return dict(lock_sym=None if lock_t is None else lock_t / L, slips_late=slips[+1], slips_early=slips[-1],
                unlocks=unlocks, windows=locked_windows, unclean=unclean, emax=emax, emin=emin, drift=drift)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float('nan')


def run_acq(seeds=40, Ks=(4, 8, 16), lfs=(0.5, 0.625, 0.75, 0.875), snrs=(10, 20, 30, 40)):
    jobs = []
    for cfg_cp in (0, 1):
        for K in Ks:
            for lf in lfs:
                for snr in snrs:
                    max_sym = 3 * (N + (N // 4 if cfg_cp == 0 else N // 8)) * K
                    jobs += [(cfg_cp, K, lf, snr, 1000 + s, max_sym) for s in range(seeds)]
    with Pool() as p:
        res = p.map(acq_trial, jobs, chunksize=4)
    rows = {}
    for j, r in zip(jobs, res):
        rows.setdefault(j[:4], []).append(r)
    table = []
    for (cfg_cp, K, lf, snr), rs in rows.items():
        lk = [r for r in rs if r['locked']]
        ts = [r['t_sym'] for r in lk]
        es = [r['e'] for r in lk]
        table.append(dict(cp=N // 4 if cfg_cp == 0 else N // 8, K=K, lock_frac=lf, snr=snr, n=len(rs),
                          lock_rate=len(lk) / len(rs), t_med=pct(ts, 50), t_p95=pct(ts, 95), t_max=max(ts) if ts else None,
                          e_min=min(es) if es else None, e_max=max(es) if es else None,
                          e_p01=pct(es, 1), e_p99=pct(es, 99), unclean=sum(1 for r in lk if not r['clean'])))
    return table


def run_noise(seeds=18, nsamp=200_000):
    jobs = [(cfg_cp, K, lf, 5000 + s, nsamp) for cfg_cp in (0, 1) for K in (4, 8, 16)
            for lf in (0.5, 0.625, 0.75, 0.875) for s in range(seeds)]
    with Pool() as p:
        res = p.map(noise_trial, jobs, chunksize=1)
    rows = {}
    for j, r in zip(jobs, res):
        rows.setdefault(j[:3], []).append(r)
    return [dict(cp=N // 4 if k[0] == 0 else N // 8, K=k[1], lock_frac=k[2],
                 locks=sum(r['locks'] for r in v), samples=sum(r['nsamp'] for r in v)) for k, v in rows.items()]


def run_track(grid, seeds=3, nsym=10_000):
    jobs = [g + (s,) + (nsym,) for g in grid for s in range(seeds)]
    with Pool() as p:
        res = p.map(track_trial, [(j[0], j[1], j[2], j[3], j[4], j[5], j[6], j[7], 7000 + j[8], j[9]) for j in jobs], chunksize=1)
    rows = {}
    for j, r in zip(jobs, res):
        rows.setdefault(j[:8], []).append(r)
    table = []
    for k, rs in rows.items():
        table.append(dict(cp=N // 4 if k[0] == 0 else N // 8, K=k[1], lock_frac=k[2], unlock_frac=k[3], M=k[4], H=k[5],
                          ppm=k[6], snr=k[7], runs=len(rs), unlocks=sum(r['unlocks'] for r in rs),
                          unclean=sum(r['unclean'] for r in rs), windows=sum(r['windows'] for r in rs),
                          slips_late=sum(r['slips_late'] for r in rs), slips_early=sum(r['slips_early'] for r in rs),
                          drift=sum(r['drift'] for r in rs), emin=min(r['emin'] for r in rs), emax=max(r['emax'] for r in rs)))
    return table


def md(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(("%.3g" % r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return s


if __name__ == '__main__' and sys.argv[1] != 'final':
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1]
    t0 = time.time()
    acq_cols = ['cp', 'K', 'lock_frac', 'snr', 'n', 'lock_rate', 't_med', 't_p95', 't_max', 'e_min', 'e_p01', 'e_p99', 'e_max', 'unclean']
    if what == 'acq':
        tab, cols = run_acq(), acq_cols
    elif what == 'acq2':
        tab, cols = run_acq(seeds=300, Ks=(8, 16), lfs=(0.75, 0.8125, 0.875), snrs=(10, 15, 20, 30)), acq_cols
    elif what == 'noise':
        tab = run_noise()
        cols = ['cp', 'K', 'lock_frac', 'locks', 'samples']
    elif what == 'track':
        K = None if sys.argv[2] == "auto" else int(sys.argv[2]); lf = float(sys.argv[3])
        grid = []
        for cfg_cp in (0, 1):
            for M in (8, 16, 32):
                for H in ((None,) if K is None else (max(1, K // 4), max(1, K // 2))):
                    for uf in (0.375, 0.5):
                        for ppm in (-100, 100):
                            grid.append((cfg_cp, K, lf, uf, M, H, ppm, 20))
        tab = run_track(grid)
        cols = ['cp', 'K', 'lock_frac', 'unlock_frac', 'M', 'H', 'ppm', 'snr', 'runs', 'unlocks', 'unclean', 'windows',
                'slips_late', 'slips_early', 'drift', 'emin', 'emax']
    json.dump(tab, open(os.path.join(OUT, f'{what}.json'), 'w'), indent=1)
    print(md(tab, cols))
    print(f"[{what}: {time.time() - t0:.0f} s]")


# ---- final trade-off table around the chosen point --------------------------------------------------------
def _acq_cfg(args):
    cfg_cp, K, lf, snr, seed = args
    CP = N // 4 if cfg_cp == 0 else N // 8
    L = N + CP
    Kv = K if K else 4 * N // CP
    rng = np.random.default_rng(seed)
    offset = float(rng.uniform(0, L))
    x, info = MiniOFDM(N=N, CP=CP, seed=seed).stream(nsym=3 * L * Kv + 4, offset=offset, snr_db=snr)
    tim = CPTiming(N=N, cfg_cp=cfg_cp, K=K, lock_frac=lf)
    for i, v in enumerate(x):
        if tim.step(int(v))['lock_rise']:
            e, clean = window_error(info['t'], i + 2, L, CP, tim.delta)
            return dict(locked=1, t=(i + 1) / L, clean=clean)
    return dict(locked=0)


def _fl_cfg(args):
    kind, cfg_cp, K, lf, seed = args
    if kind == 'noise':
        x = noise_stream(500_000, sigma=40.0, seed=seed)
    else:
        x, _ = MiniOFDM(N=N, CP=0, seed=seed).stream(nsym=15000, offset=0.0, snr_db=30)
    tim = CPTiming(N=N, cfg_cp=cfg_cp, K=K, lock_frac=lf)
    return sum(tim.step(int(v))['lock_rise'] for v in x), len(x)


def run_final(seeds=200):
    cfgs = [(8, 0.75), (None, 0.625), (None, 0.75), (None, 0.8125), (None, 0.875)]
    acq = [(c, K, lf, snr, 2000 + s) for c in (0, 1) for K, lf in cfgs for snr in (10, 20, 30) for s in range(seeds)]
    fl = [(k, c, K, lf, 3000 + s) for c in (0, 1) for K, lf in cfgs for k in ('noise', 'nocp') for s in range(9)]
    with Pool() as p:
        ra = p.map(_acq_cfg, acq, chunksize=8)
        rf = p.map(_fl_cfg, fl, chunksize=1)
    rows = []
    for c in (0, 1):
        CP = N // 4 if c == 0 else N // 8
        for K, lf in cfgs:
            row = dict(cp=CP, K=K if K else 4 * N // CP, lock_frac=lf)
            for snr in (10, 20, 30):
                rs = [r for a, r in zip(acq, ra) if a[:4] == (c, K, lf, snr)]
                lk = [r for r in rs if r['locked']]
                t = [r['t'] for r in lk]
                row[f'lock%@{snr}'] = round(100 * len(lk) / len(rs), 1)
                row[f't_med@{snr}'] = float(np.median(t)) if t else float('nan')
                row[f't_p95@{snr}'] = pct(t, 95)
                row.setdefault('unclean', 0)
                row['unclean'] += sum(1 for r in lk if not r['clean'])
            for kind in ('noise', 'nocp'):
                rr = [r for a, r in zip(fl, rf) if a[:4] == (kind, c, K, lf)]
                row[f'false_{kind}'] = sum(r[0] for r in rr)
                row[f'ms_{kind}'] = round(sum(r[1] for r in rr) / 32e3)
            rows.append(row)
    return rows


if __name__ == '__main__' and sys.argv[1] == 'final':
    t0 = time.time()
    tab = run_final()
    json.dump(tab, open(os.path.join(OUT, 'final.json'), 'w'), indent=1)
    print(md(tab, ['cp', 'K', 'lock_frac', 'lock%@10', 't_med@10', 't_p95@10', 'lock%@20', 't_med@20', 't_p95@20',
                   'lock%@30', 't_p95@30', 'unclean', 'false_noise', 'ms_noise', 'false_nocp', 'ms_nocp']))
    print(f"[final: {time.time() - t0:.0f} s]")
