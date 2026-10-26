# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest for the CP timing block (timing.py) and the system model (system_model.py), with the
parameters chosen by the parameter study (K = 4N/CP, M = 16, lock 0.75, unlock 0.375, hysteresis 8).
Statistical tests run in parallel (multiprocessing). See model/README.md for the
full parameter study (study_timing.py)."""
import numpy as np
import pytest
from multiprocessing import Pool

from symgen import MiniOFDM, noise_stream
from timing import CPTiming
from system_model import FFTChipModel
from fft_model import SDF

N = 32
PARAMS = dict(K=None, M=16, lock_frac=0.75, unlock_frac=0.375, hyst=8)    # K = 4N/CP: 16 @ CP=8, 32 @ CP=4
KSYM = {0: 16, 1: 32}
CPS = {0: 8, 1: 4}


def tolerance(CP):
    """A window starting e samples after its nominal start (CP - delta into the symbol) holds one
    symbol iff -(CP - delta) <= e < delta + 1."""
    d = CP // 2
    return -(CP - d), d + 1


def window_error(t, i0, L, CP):
    tt = t[i0:i0 + N]
    j = np.floor(tt / L)
    e = float(tt[0] - j[0] * L - (CP - CP // 2))
    e = (e + L / 2) % L - L / 2
    return e, bool(np.all(j == j[0]))


# ---- (a) correlator usable with the v1 carrier set ------------------------------------------------------
@pytest.mark.parametrize("cfg_cp", [0, 1])
def test_metric_on_off_cp(cfg_cp):
    CP = CPS[cfg_cp]
    L = N + CP
    for snr in (None, 20):
        g = MiniOFDM(N=N, CP=CP, seed=21)
        x, _ = g.stream(nsym=2000, offset=0.0, snr_db=snr)
        sg = (x < 0).astype(int)
        m = np.where(sg[N:] == sg[:-N], 1, -1)
        per = lambda d: np.array([m[j * L + d: j * L + d + CP].sum() for j in range(2, 1990)])
        on = per(0)                     # window on the last CP samples of the symbol (the CP source)
        off = np.concatenate([per(d) for d in range(CP, L - CP + 1)])
        print(f"CP={CP} snr={snr}: on-CP mean {on.mean():.3f}, off-CP mean {off.mean():+.3f} (per symbol, max {CP})")
        if snr is None:
            assert np.all(on == CP)
        else:
            assert on.mean() >= 0.85 * CP
        assert abs(off.mean()) <= 0.03 * CP


# ---- (b) acquisition vs SNR and initial phase -------------------------------------------------------------
def _acq(args):
    cfg_cp, snr, seed = args
    CP = CPS[cfg_cp]
    L = N + CP
    rng = np.random.default_rng(seed)
    offset = float(rng.uniform(0, L))
    g = MiniOFDM(N=N, CP=CP, seed=seed)
    max_sym = 3 * L * KSYM[cfg_cp]
    x, info = g.stream(nsym=max_sym + 4, offset=offset, snr_db=snr)
    tim = CPTiming(N=N, cfg_cp=cfg_cp, **PARAMS)
    for i, v in enumerate(x):
        if tim.step(int(v))['lock_rise']:
            e, clean = window_error(info['t'], i + 2, L, CP)
            return dict(locked=True, t_sym=(i + 1) / L, e=e, clean=clean, offset=offset)
    return dict(locked=False, offset=offset)


@pytest.mark.parametrize("cfg_cp", [0, 1])
def test_acquisition(cfg_cp):
    CP = CPS[cfg_cp]
    L = N + CP
    lo, hi = tolerance(CP)
    snrs = (10, 20, 30, 40)
    jobs = [(cfg_cp, snr, 100 + s) for snr in snrs for s in range(36)]
    with Pool() as p:
        res = p.map(_acq, jobs)
    for snr in snrs:
        rs = [r for j, r in zip(jobs, res) if j[1] == snr]
        lk = [r for r in rs if r['locked']]
        t = np.array([r['t_sym'] for r in lk])
        e = np.array([r['e'] for r in lk])
        print(f"CP={CP} snr={snr}: locked {len(lk)}/{len(rs)}, t_med {np.median(t):.0f} t_p95 {np.percentile(t, 95):.0f}"
              f" t_max {t.max():.0f} symbols, lock error [{e.min():+.2f}, {e.max():+.2f}] (clean if in [{lo}, {hi}))")
        assert all(r['clean'] for r in lk)
        if snr >= 20:
            assert len(lk) == len(rs)                                   # every initial phase locks
            assert t.max() <= 2 * L * KSYM[cfg_cp]                      # within two hypothesis sweeps
        else:
            assert len(lk) >= 0.85 * len(rs)


# ---- (c) false lock --------------------------------------------------------------------------------------
def _false(args):
    kind, cfg_cp, seed = args
    if kind == 'noise':
        x = noise_stream(500_000, sigma=40.0, seed=seed)
    else:                                                                # OFDM-like, but no cyclic prefix
        x, _ = MiniOFDM(N=N, CP=0, seed=seed).stream(nsym=15000, offset=0.0, snr_db=30)
    tim = CPTiming(N=N, cfg_cp=cfg_cp, **PARAMS)
    return sum(tim.step(int(v))['lock_rise'] for v in x), len(x)


@pytest.mark.parametrize("cfg_cp", [0, 1])
def test_false_lock(cfg_cp):
    jobs = [(k, cfg_cp, 300 + s) for k in ('noise', 'nocp') for s in range(9)]
    with Pool() as p:
        res = p.map(_false, jobs)
    for kind in ('noise', 'nocp'):
        locks = sum(r[0] for j, r in zip(jobs, res) if j[0] == kind)
        n = sum(r[1] for j, r in zip(jobs, res) if j[0] == kind)
        print(f"CP={CPS[cfg_cp]} {kind}: {locks} false locks in {n} samples ({n / 32e3:.0f} ms at 32 MS/s)")
        assert locks == 0


# ---- (d) tracking under +-100 ppm ------------------------------------------------------------------------
def _track(args):
    cfg_cp, ppm, snr, seed, nsym = args
    CP = CPS[cfg_cp]
    L = N + CP
    rng = np.random.default_rng(seed)
    x, info = MiniOFDM(N=N, CP=CP, seed=seed).stream(nsym=nsym, offset=float(rng.uniform(0, L)), ppm=ppm, snr_db=snr)
    tim = CPTiming(N=N, cfg_cp=cfg_cp, **PARAMS)
    lock_i = None
    s = dict(late=0, early=0, unlocks=0, windows=0, unclean=0, emin=0.0, emax=0.0)
    for i, v in enumerate(x):
        r = tim.step(int(v))
        if r['lock_rise'] and lock_i is None:
            lock_i = i
        if r['slip']:
            s['late' if r['slip_dir'] > 0 else 'early'] += 1
        s['unlocks'] += r['unlock']
        if r['ph'] == 0 and r['lock'] and i + N <= len(x):
            e, clean = window_error(info['t'], i, L, CP)
            s['windows'] += 1
            s['unclean'] += (not clean)
            s['emin'], s['emax'] = min(s['emin'], e), max(s['emax'], e)
    s['drift'] = (len(x) - (lock_i or 0)) * ppm * 1e-6
    s['locked_sym'] = None if lock_i is None else (len(x) - lock_i) / L
    return s


@pytest.mark.parametrize("cfg_cp", [0, 1])
def test_tracking_ppm(cfg_cp):
    nsym = 12_000 if cfg_cp == 0 else 14_000              # >= 10^4 symbols *after* a slow 10 dB acquisition
    jobs = [(cfg_cp, ppm, snr, 400 + k, nsym) for k, (ppm, snr) in enumerate([(p, s) for p in (-100, 100) for s in (10, 20, 30)])]
    with Pool() as p:
        res = p.map(_track, jobs)
    for j, s in zip(jobs, res):
        ppm, snr = j[1], j[2]
        print(f"CP={CPS[cfg_cp]} ppm={ppm:+d} snr={snr}: locked for {s['locked_sym']:.0f} symbols, {s['windows']} windows,"
              f" unclean {s['unclean']}, unlocks {s['unlocks']}, slips late/early {s['late']}/{s['early']},"
              f" drift {s['drift']:+.1f} samples, e in [{s['emin']:+.2f}, {s['emax']:+.2f}]")
        assert s['locked_sym'] >= 10_000
        assert s['unlocks'] == 0 and s['unclean'] == 0
        net = s['early'] - s['late']                     # RX faster (ppm > 0) -> symbols arrive early
        assert abs(net - s['drift']) <= 4
        assert (s['early'] if ppm < 0 else s['late']) <= 0.1 * abs(s['drift']) + 3   # few wrong-way slips


# ---- (e) system model: bytes, framing, VALID bookkeeping --------------------------------------------------
def frame_bytes(samples):
    m = SDF(N)
    out = [m.step(int(v)) for v in list(samples) + [0] * N][N - 1: 2 * N - 1]
    return [(out[c][0] if c % 2 == 0 else out[c - 1][1]) & 0xFF for c in range(N)]


def run_system(cfg_cp, x, rst_cycles=3):
    chip = FFTChipModel(N=N, cfg_cp=cfg_cp, **PARAMS)
    return [chip.step(int(v), rst_n=0 if i < rst_cycles else 1) for i, v in enumerate(x)]


RST_SYNC = 2          # FFTChipModel's reset synchroniser stages


def check_system(outs, x, info, CP, rst_cycles=3):
    """Independent re-derivation of the bus framing (docs/info.md) from the timing records, compared
    cycle by cycle with the model's pins; plus bit-exact bytes and clean windows for every VALID frame.
    Indexing: outs[i] = pins after edge i = what the reader sees during cycle i+1 (vis(k) = outs[k-1]).
    Checked from k0 = the first visible cycle after the (synchronised) reset: rst_cycles + RST_SYNC + 1."""
    L = N + CP
    k0 = rst_cycles + RST_SYNC + 1
    lo, hi = tolerance(CP)
    # capture regions: N run cycles with ph = 0..N-1 (pause cycles may not interrupt them)
    caps, cur = [], None
    for i, o in enumerate(outs):
        r = o['tim']
        if not r['run']:
            continue
        if r['ph'] == 0:
            cur = [i]
        elif cur is not None and r['ph'] == len(cur):
            cur.append(i)
        else:
            cur = None                                 # free-running cycle outside ph 0..N-1
        if cur is not None and len(cur) == N:
            caps.append(cur)
            cur = None
    # expected VALID per visible cycle: frame captured in `a` is shown during the run cycles of `b`
    exp_valid = np.zeros(len(outs) + 1, dtype=int)
    frame_of_vis = {}
    nslipinv = 0
    for a, b in zip(caps, caps[1:]):
        ok = outs[a[0]]['tim']['lock'] == 1
        slip = any(outs[k]['tim']['slip'] for k in range(a[-1] + 1, b[0]))
        nslipinv += ok and slip
        ok = ok and not slip and all(outs[k]['tim']['run'] == 0 for k in range(a[-1] + 1, b[0]))
        for c, k in enumerate(b):
            exp_valid[k] = 1 if (ok and outs[k]['tim']['lock']) else 0
            frame_of_vis[k] = (tuple(a), c)
    got_valid = np.array([0] + [o['valid'] for o in outs])          # visible during cycle k
    assert np.array_equal(got_valid[k0:], exp_valid[k0:len(outs) + 1]), np.flatnonzero(got_valid != exp_valid)[:10]
    # SOF visible with byte 0 of every frame; DBG = slip pulse; LOCK pin = lock register
    for k in range(k0, len(outs)):
        r = outs[k]['tim']
        assert outs[k - 1]['sof'] == (1 if (r['run'] and (r['ph'] % N) == 0) else 0)
    for i, o in enumerate(outs[k0 - 1:], start=k0 - 1):
        assert o['dbg'] == o['tim']['slip'] and o['lock'] == o['tim']['lock_next']
    # bytes and windows of every VALID frame
    frames = {}
    for k in np.flatnonzero(got_valid):
        a, c = frame_of_vis[k]
        frames.setdefault(a, {})[c] = outs[k - 1]['uo'] & 0xFF
    for a, bts in frames.items():
        assert sorted(bts) == list(range(N))                        # all N bytes valid, none partial
        assert [bts[c] for c in range(N)] == frame_bytes(x[list(a)])
        e, clean = window_error(info['t'], a[0], L, CP)
        assert clean and lo <= e < hi
    # VALID runs are exactly N bytes starting with SOF (bus during the pause is don't-care)
    v = got_valid
    starts = np.flatnonzero((v[1:] == 1) & (v[:-1] == 0)) + 1
    for k in starts:
        assert outs[k - 1]['sof'] == 1 and np.all(v[k:k + N] == 1) and v[k + N] == 0
    return dict(valid_frames=len(frames), slip_invalidated=nslipinv, captures=len(caps))


@pytest.mark.parametrize("cfg_cp,ppm", [(0, 0), (0, 100), (0, -100), (1, 100), (1, -100)])
def test_system_bytes_and_bookkeeping(cfg_cp, ppm):
    CP = CPS[cfg_cp]
    g = MiniOFDM(N=N, CP=CP, seed=500 + cfg_cp)
    x, info = g.stream(nsym=2600 if cfg_cp == 0 else 3400, offset=13.37, ppm=ppm, snr_db=25)   # CP=4 acquires slower (K=32)
    outs = run_system(cfg_cp, x)
    st = check_system(outs, x, info, CP)
    slips = sum(o['tim']['slip'] for o in outs)
    print(f"CP={CP} ppm={ppm:+d}: {st['valid_frames']} VALID frames bit-exact, {slips} slips, "
          f"{st['slip_invalidated']} frames invalidated by slips")
    assert st['valid_frames'] >= 2000
    assert st['slip_invalidated'] == slips                           # exactly one frame per slip
    first_lock = next(i for i, o in enumerate(outs) if o['lock'])
    assert not any(o['valid'] for o in outs[:first_lock + 1])        # nothing valid before lock


def test_unlock_and_relock():
    """Signal -> noise -> signal: LOCK drops within two tracking cycles of the signal loss, VALID
    stays 0 while unlocked, and the chip re-acquires."""
    CP, cfg_cp = 8, 0
    L = N + CP
    g = MiniOFDM(N=N, CP=CP, seed=600)
    x1, i1 = g.stream(nsym=1200, offset=5.0, snr_db=30)
    xn = noise_stream(400 * L, sigma=45.0, seed=601)
    x2, i2 = MiniOFDM(N=N, CP=CP, seed=602).stream(nsym=1200, offset=21.0, snr_db=30)
    x = np.concatenate([x1, xn, x2])
    outs = run_system(cfg_cp, x)
    lk = np.array([o['lock'] for o in outs])
    t_loss = len(x1)
    assert lk[t_loss - 1] == 1
    drop = t_loss + int(np.argmax(lk[t_loss:] == 0))
    cycle = 3 * KSYM[cfg_cp] + PARAMS['M']
    print(f"LOCK dropped {(drop - t_loss) / L:.0f} symbols after signal loss (bound {2 * cycle + KSYM[cfg_cp]})")
    assert (drop - t_loss) / L <= 2 * cycle + KSYM[cfg_cp]
    relock = len(x1) + len(xn) + int(np.argmax(lk[len(x1) + len(xn):] == 1))
    assert lk[relock] == 1
    print(f"re-locked {(relock - len(x1) - len(xn)) / L:.0f} symbols after the signal returned")
    v = np.array([o['valid'] for o in outs])
    assert not np.any(v & (lk == 0))                                 # VALID implies LOCK


def test_reset_mid_stream():
    CP, cfg_cp = 8, 0
    g = MiniOFDM(N=N, CP=CP, seed=700)
    x, info = g.stream(nsym=2400, offset=3.0, snr_db=30)
    chip = FFTChipModel(N=N, cfg_cp=cfg_cp, **PARAMS)
    outs = []
    for i, v in enumerate(x):
        outs.append(chip.step(int(v), rst_n=0 if (i < 3 or 40000 <= i < 40005) else 1))
    lk = np.array([o['lock'] for o in outs])
    assert lk[39999] == 1 and lk[40005] == 0 and lk[-1] == 1
    assert lk[40000 + RST_SYNC - 1] == 1 and lk[40000 + RST_SYNC] == 0       # reset acts RST_SYNC edges later
    assert not any(o['valid'] for o in outs[40000 + RST_SYNC:40100])


def test_invariants_locked():
    """While locked: exactly N run cycles per symbol, run == (ph < N), slips only in the pause."""
    CP, cfg_cp = 4, 1
    g = MiniOFDM(N=N, CP=CP, seed=800)
    x, _ = g.stream(nsym=3000, offset=9.0, ppm=100, snr_db=20)
    tim = CPTiming(N=N, cfg_cp=cfg_cp, **PARAMS)
    recs = [tim.step(int(v)) for v in x]
    runs_in_sym = 0
    for r in recs:
        if r['lock'] and r['lock_next']:
            assert r['run'] == (1 if r['ph'] < N else 0)
            if r['slip'] or r['hyp_adv']:
                assert r['ph'] >= N
        if r['lock']:
            runs_in_sym += r['run']
            if r['sym_end']:
                assert runs_in_sym in (0, N)
                runs_in_sym = 0
