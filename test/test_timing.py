# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CP timing block, register for register against model/timing.CPTiming (same correlator constants):
every cycle compares ph, run, lock, state, A, A_on, symcnt, stall, primed, weak and the combinational
slip / ph_next / run_next / lock_next. Stimuli are v1 mini-OFDM streams (model/signal_gen.py): acquisition
and tracking, both CFG_CP values, heavy sample-rate offsets (forced slips both ways), signal -> noise ->
signal (unlock / relock), reset mid-stream. Alignment searched once over the ph sequence, then exact.
N from the environment (units.mk, default 16). At N = 16 CFG_CP has no function (CP fixed at 4): the stimulus is
always CP = N/4 and the CFG_CP = 1 cases drive the pin high to show it is ignored; the sample-rate offsets
follow the N = 16 tracking range measured with the model (no unlock at +-300 ppm, unlocks at
+-400 ppm); N = 32 keeps its own, smaller values (see test_forced_slips)."""
import os
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer
import signal_gen as sg
from timing import CPTiming

N = int(os.environ.get("N", "16"))
L0 = N + N // 4                                    # symbol length at CP = N/4
CLK_PS = int(os.environ.get("CLK_PS", "31250"))      # also run at 16 MHz (nothing is frequency-dependent)


def sv(v):
    """signed value of a LogicArray (None if X/Z)"""
    return v.to_signed() if v.is_resolvable else None


def iv(v):
    return int(v) if v.is_resolvable else None


async def run_compare(dut, x, cfg_cp, rst=None, what="", force_state=None):
    """x: two's-complement samples; rst: per-cycle rst_n (default: 3 reset cycles); force_state: {cycle: code}
    deposits that code into the state register of the RTL and of the model during that cycle."""
    cocotb.start_soon(Clock(dut.clk, CLK_PS, unit="ps").start())
    dut.cfg_cp.value = cfg_cp
    model = CPTiming(N=N, cfg_cp=cfg_cp)
    rst = rst if rst is not None else [0, 0, 0] + [1] * (len(x) - 3)
    u = dut.u_tim
    got, exp = [], []
    await FallingEdge(dut.clk)
    for k, v in enumerate(x):
        dut.x_sgn.value = 1 if v < 0 else 0
        dut.rst_n.value = rst[k]
        if force_state and k in force_state:
            u.state.value = force_state[k]
            model.state = force_state[k]
        await Timer(5, unit="ns")
        pre = dict(primed=model.primed, weak=model.weak)
        r = model.step(int(v), rst[k])
        g = (iv(dut.ph.value), iv(dut.run.value), iv(dut.lock.value), iv(u.state.value), sv(u.acc.value),
             sv(u.acc_on.value), iv(u.symcnt.value), iv(u.stall.value), iv(u.primed.value), iv(u.weak_q.value),
             iv(dut.slip.value), iv(dut.ph_nx.value), iv(dut.run_nx.value), iv(dut.lock_nx.value))
        e = (r['ph'], r['run'], r['lock'], r['state'], r['A'], r['A_on'], r['symcnt'], r['stall'], pre['primed'],
             pre['weak'], r['slip'], r['ph_next'], r['run_next'], r['lock_next'])
        got.append(g)
        exp.append(e)
        await FallingEdge(dut.clk)
    # alignment search once on the ph sequence, then exact on everything
    lags = {lag: sum(1 for a, b in zip(got[10:2000], exp[10 + lag:2000 + lag]) if a[0] == b[0]) for lag in (-1, 0, 1)}
    lag = max(lags, key=lags.get)
    dut._log.info(f"{what}: alignment search -> lag {lag} cycles ({lags[lag]}/1990 ph match)")
    assert lag == 0
    # cycle 0 precedes the first (reset) clock edge: the DUT registers still hold their power-up / previous
    # state there; the synchronous reset defines them from cycle 1 on, and from then on all must match
    bad = [(k, a, b) for k, (a, b) in enumerate(zip(got, exp)) if k >= 1 and a != b]
    names = "ph run lock state A A_on symcnt stall primed weak slip ph_nx run_nx lock_nx".split()
    if bad:
        k, a, b = bad[0]
        diff = {n: (p, q) for n, p, q in zip(names, a, b) if p != q}
        assert False, f"{what}: {len(bad)} mismatching cycles, first at {k}: {diff}"
    locks = sum(1 for a, b in zip(exp, exp[1:]) if b[2] and not a[2])
    unlocks = sum(1 for a, b in zip(exp, exp[1:]) if a[2] and not b[2])
    slips = sum(e[10] for e in exp)
    dut._log.info(f"{what}: {len(x) - 1} cycles bit-exact (14 signals/cycle); {locks} lock(s), {unlocks} unlock(s), {slips} slip(s)")
    return dict(locks=locks, unlocks=unlocks, slips=slips, lock=[e[2] for e in exp], state=[e[3] for e in exp])


def v1(cfg_cp, nsym, seed, snr=30, ppm=0.0, phase0=None):
    cfg = sg.OFDMConfig(N=N, cfg_cp=0 if N == 16 else cfg_cp)        # N = 16: CP fixed at 4
    adc, _, _ = sg.generate(cfg, nsym, seed, snr_db=snr, ppm=ppm, phase0=phase0)
    return sg.to_signed(adc)


@cocotb.test()
async def test_acquire_track_cp8(dut):
    s = await run_compare(dut, v1(0, 1500, 11), 0, what=f"CFG_CP=0 (CP {N // 4}) acquisition + tracking")
    assert s['locks'] >= 1


@cocotb.test()
async def test_acquire_track_cp4(dut):
    s = await run_compare(dut, v1(1, 2400, 12), 1, what=f"CFG_CP=1 (CP {N // 4 if N == 16 else N // 8}) acquisition + tracking")
    assert s['locks'] >= 1


@cocotb.test()
async def test_forced_slips(dut):
    """Sample-rate offsets inside the tracking range (measured with the model: at N = 32 no unlock up to ~+-200 ppm
    at CP 8 and ~+-150 ppm at CP 4; at N = 16 none at +-300 ppm, study_results/n16_edges.json) force frequent slips,
    both directions, both CP settings."""
    cases = ((0, -150), (0, 150), (1, -120), (1, 120)) if N == 32 else ((0, -200), (0, 200), (1, -250), (1, 250))
    for cfg_cp, ppm in cases:
        nsym = (2600 if cfg_cp == 0 else 3400) if N == 32 else 3600
        s = await run_compare(dut, v1(cfg_cp, nsym, 13 + cfg_cp + (ppm > 0), snr=30, ppm=ppm), cfg_cp,
                              what=f"CFG_CP={cfg_cp} {ppm:+d} ppm")
        assert s['slips'] >= 10 and s['unlocks'] == 0


@cocotb.test()
async def test_drift_beyond_tracking(dut):
    """+-600 ppm (N = 32) / +-1000 ppm (N = 16) exceeds the tracking range: the block must unlock and
    re-acquire, bit-exact throughout."""
    for cfg_cp, ppm in (((0, -600), (1, 600)) if N == 32 else ((0, -1000), (1, 1000))):
        s = await run_compare(dut, v1(cfg_cp, 2600 if N == 32 else 4000, 13 + cfg_cp, snr=30, ppm=ppm), cfg_cp,
                              what=f"CFG_CP={cfg_cp} {ppm:+d} ppm (beyond tracking)")
        assert s['unlocks'] >= 1 and s['locks'] >= 2


@cocotb.test()
async def test_lock_unlock_sequence(dut):
    """signal -> noise -> signal (other phase): lock, unlock, relock."""
    L = L0
    rng = np.random.default_rng(5)
    x = np.concatenate([v1(0, 900, 21), np.clip(np.round(rng.normal(0, 40, 300 * L)), -128, 127).astype(int),
                        v1(0, 1200, 22, phase0=17.5)])
    s = await run_compare(dut, x, 0, what="signal -> noise -> signal")
    assert s['locks'] >= 2 and s['unlocks'] >= 1


@cocotb.test()
async def test_reset_mid_stream(dut):
    x = v1(0, 1400 if N == 32 else 2800, 31)                         # reset at cycle 30000 in both cases
    rst = [0, 0, 0] + [1] * (len(x) - 3)
    for k in range(30000, 30007):
        rst[k] = 0
    s = await run_compare(dut, x, 0, rst=rst, what="reset mid-stream (after lock)")
    assert s['locks'] >= 2


@cocotb.test()
async def test_unused_state_codes(dut):
    """The state register forced to each unused encoding (5, 6, 7) while locked goes back
    to ACQ at the next edge with LOCK = 0 and run = 1, register for register with the model; then re-acquires."""
    x = v1(0, 3600, 41)
    force = {15000: 5, 35000: 6, 55000: 7}
    s = await run_compare(dut, x, 0, force_state=force, what="unused state codes 5/6/7 while locked")
    for k, code in force.items():
        assert s["lock"][k - 1] == 1 and s["state"][k] == code, f"cycle {k}: not locked before forcing {code}"
        assert s["state"][k + 1] == 0 and s["lock"][k + 1] == 0, f"cycle {k + 1}: state {s['state'][k + 1]}"
    assert s["unlocks"] == 3 and s["locks"] == 4
