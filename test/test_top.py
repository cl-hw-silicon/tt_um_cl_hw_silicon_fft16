# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt_um_cl_hw_silicon_fft16 through its pins (pinout, bus protocol and frame structure: docs/info.md).

Every test drives only ui_in / uio_in / rst_n / ena and reads only uo_out / uio_out / uio_oe (except the
white-box ADC code sweep, which also peeks at the capture register). The oracle is
model/system_model.FFTChipModel (via chip_io.model_pins), compared cycle by cycle and bit-exact.

  test_pins_bit_exact_*      v1 mini-OFDM with a sample-rate offset large enough to force several slips:
                             uo_out, SOF, VALID, LOCK, DBG, uio_oe, uio_out[6:5] every cycle; frame
                             bookkeeping from the pins (N VALID bytes per frame, CP pause, SOF period L,
                             L +- 1 at a slip, the frame in flight at a slip invalidated).
  test_bin_and_byte_order    a tone that walks k = 0..7 one symbol at a time: bus pair m carries bin
                             BIN_ORDER[m], I before Q (the DC symbol has Q = 0); sample-to-byte latency.
  test_adc_conversion        all 256 codes through the capture register (both edges); DC corner codes
                             through the FFT to the pins.
  test_reset                 pins during and after reset (two-flop synchroniser: rst_n acts 2 edges later), reset
                             while locked, re-acquisition.
  test_reset_release_phases  rst_n released at 8 phases of the clock period (at the rising edge, at the falling edge,
                             between them) on a locked stream: pins bit-exact, quiet in reset, SCLK / ADC_CLK clean,
                             re-acquisition and BER 0 after every release.
  test_bus_timing            pins read 4x per cycle: SCLK high after the rising edge, low after the falling
                             edge; data / SOF / VALID / LOCK / DBG change only at the rising edge.
  test_adc_tco_sweep         ADC data valid t_co after the clock edge (0..30 ns, +-2.5 ns uncertainty driven
                             as X on icarus) for both CFG_EDGE settings: clean capture exactly where the
                             edge is outside the uncertainty window, with the predicted latency.
  test_clock_gate_pauses     (white-box) the ICG: clk_dp has an edge exactly when `run` was 1, never in a CP
                             pause; at gate level also every clk_dp leaf clock net, and none of the 192 datapath
                             flops changes in a pause while the datapath moves when running.
CLK_PS sets the clock (default 31250 ps = 32 MHz; a 16 MHz run checks nothing depends on the clock frequency).
"""
import os
import re

import numpy as np
import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from cocotb.types import LogicArray

import signal_gen as sg
import chip_io as io
from chip_io import N, CP, L, CLK_PS


def ofdm(nsym, seed, snr_db=30, ppm=0.0, phase0=None, dc=0.0):
    adc, tx, bd = sg.generate(sg.OFDMConfig(N=N), nsym, seed, snr_db=snr_db, ppm=ppm, phase0=phase0, dc_offset=dc)
    return adc, tx, bd


def check_bookkeeping(dut, pins, what):
    """Frame structure from the pins alone (docs/info.md)."""
    fr = io.frames_from_pins(pins)
    # every VALID run is exactly one frame: N bytes, starting with SOF
    for s, ln in fr["runs"]:
        assert ln == N and pins["sof"][s] == 1, f"{what}: VALID run at {s} of length {ln} (SOF {pins['sof'][s]})"
    # SOF period while locked: L, or L +- 1 right after a slip (one-sample re-phase inside the pause)
    per = fr["periods"]
    assert set(per) <= {L - 1, L, L + 1}, f"{what}: SOF periods {sorted(set(per))}"
    n_slip_periods = sum(p != L for p in per)
    # DBG pulses (one per slip) and the frame in flight: the first SOF at or after a DBG pulse is invalid (an
    # early slip re-phases at the same edge, so its SOF coincides with DBG; a late slip repeats a pause cycle)
    frames = fr["frames"]
    sofs = np.array([c for c, _ in frames])
    inval_after_slip = 0
    for d in fr["dbg"]:
        nxt = np.flatnonzero(sofs >= d)
        assert len(nxt), f"{what}: no SOF after the DBG pulse at {d}"
        c, ok = frames[nxt[0]]
        assert not ok, f"{what}: the frame in flight at the slip (DBG at {d}, SOF at {c}) is VALID"
        inval_after_slip += 1
    # VALID = 0 during every CP pause and outside lock: VALID implies LOCK, and no VALID byte in the L - N
    # cycles after a VALID frame
    assert np.all(pins["lock"][pins["valid"] == 1] == 1), f"{what}: VALID without LOCK"
    for s, ln in fr["runs"]:
        assert np.all(pins["valid"][s + ln: s + ln + CP - 1] == 0), f"{what}: VALID during the CP pause after {s}"
    # while locked, the only invalid frames are the one captured before LOCK rose (one per lock interval) and
    # the ones in flight at a slip
    lk = pins["lock"]
    lock_rises = int(np.sum((lk[:-1] == 0) & (lk[1:] == 1)))      # 0 -> 1 only (gate level: X before the first edge)
    inval_locked = sum(1 for c, ok in frames if not ok and pins["lock"][c] == 1)
    assert inval_locked == lock_rises + inval_after_slip, \
        f"{what}: {inval_locked} invalid locked frames vs {lock_rises} lock rises + {inval_after_slip} slips"
    nvalid = sum(ok for _, ok in frames)
    dut._log.info(f"{what}: {len(frames)} frames ({nvalid} VALID), {len(fr['dbg'])} slips (DBG), "
                  f"{n_slip_periods} SOF periods of L +- 1, {inval_after_slip} frames invalidated at a slip")
    return fr


async def run_bit_exact(dut, cfg_edge, uio6, nsym, seed, ppm, phase0, what, latency_search=True):
    await io.start(dut, cfg_edge=cfg_edge, uio6=uio6)
    adc, _, _ = ofdm(nsym, seed, ppm=ppm, phase0=phase0)
    rst = np.ones(len(adc), dtype=int)
    rst[:4] = 0
    pins = await io.stream(dut, adc, rst)
    if latency_search:
        d, score = io.find_latency(adc, rst, pins, 1, 3000)
        dut._log.info(f"{what}: alignment search -> model latency {d} edges ({score})")
        assert d == 2
    exp = io.model_pins(adc, rst, latency=2)
    bad = io.compare(pins, exp)
    assert not bad, f"{what}: {len(bad)} pin mismatches, first {bad[:6]}"
    # SCLK / ADC_CLK read a quarter period after the rising edge: high from the first edge after the (synchronised)
    # reset releases
    k1 = max(reset_edges(rst, len(adc))) + 1
    assert np.all(pins["sclk"][k1:] == 1) and np.all(pins["aclk"][k1:] == 1)
    fr = check_bookkeeping(dut, pins, what)
    ndef = int(np.sum(exp["uo"] != -2))
    dut._log.info(f"{what}: {len(adc)} cycles bit-exact (uo_out on {ndef} cycles), LOCK at cycle "
                  f"{int(np.argmax(pins['lock'] == 1))}, clock {CLK_PS} ps")
    assert len(fr["dbg"]) >= 3 and sum(ok for _, ok in fr["frames"]) >= 500
    return pins, exp, fr


@cocotb.test()
async def test_pins_bit_exact_posedge_early_slips(dut):
    """CFG_EDGE = 0, +250 ppm (RX clock slower than TX: early slips, a pause cycle skipped, SOF period L - 1)."""
    await run_bit_exact(dut, cfg_edge=0, uio6=0, nsym=1300, seed=101, ppm=250.0, phase0=7.3,
                        what="CFG_EDGE=0, +250 ppm")


@cocotb.test()
async def test_pins_bit_exact_negedge_late_slips(dut):
    """CFG_EDGE = 1, -250 ppm (late slips, a pause cycle repeated, SOF period L + 1); uio[6] (unused input)
    driven high."""
    await run_bit_exact(dut, cfg_edge=1, uio6=1, nsym=1300, seed=102, ppm=-250.0, phase0=13.6,
                        what="CFG_EDGE=1, -250 ppm, uio[6]=1")


# ---- helpers for the structured tests ------------------------------------------------------------------
def s8(b):
    b = int(b) & 0xFF
    return b - 256 if b >= 128 else b


def frame_windows(recs, sof_cycles, latency=2):
    """For each SOF cycle c of a locked frame: (c, first, last) ADC sample indices of the window it carries:
    the datapath edges cnt = 0 .. N-1 of the last complete window before c (model step k consumes adc[k - latency])."""
    dp = [(k, r["dp"]["cnt"]) for k, r in enumerate(recs) if r is not None and r["dp"] is not None]
    ends = [i for i, (k, cnt) in enumerate(dp) if cnt == N - 1 and i >= N - 1 and
            [c for _, c in dp[i - N + 1: i + 1]] == list(range(N)) and dp[i][0] - dp[i - N + 1][0] == N - 1]
    out = []
    for c in sof_cycles:
        e = [dp[i][0] for i in ends if dp[i][0] < c]
        if e:
            e1 = e[-1]
            out.append((int(c), e1 - (N - 1) - latency, e1 - latency))
    return out


def tone_walk(nsym, seed, a=60.0, theta=0.3):
    """Symbol j carries one real tone at bin k = j mod N/2 with a random sign (k = 0: DC +-a), CP = copy."""
    rng = np.random.default_rng(seed)
    u = np.arange(L)
    x = [a * (1 if rng.random() < 0.5 else -1) * np.cos(2 * np.pi * (j % (N // 2)) * (u - CP) / N + (theta if j % (N // 2) else 0.0))
         for j in range(nsym)]
    return np.clip(np.round(np.concatenate(x)) + 128, 0, 255).astype(int)


@cocotb.test()
async def test_bin_and_byte_order(dut):
    """Bus pair m carries bin BIN_ORDER[m] (docs/info.md), I before Q; sample-to-first-byte latency measured."""
    await io.start(dut)
    adc = tone_walk(700, seed=5)
    rst = np.ones(len(adc), dtype=int)
    rst[:4] = 0
    pins = await io.stream(dut, adc, rst)
    exp, recs = io.model_pins(adc, rst, keep=True)
    bad = io.compare(pins, exp)
    assert not bad, f"{len(bad)} pin mismatches, first {bad[:6]}"
    fr = io.frames_from_pins(pins)
    # symbol index of a frame = its SOF count inside the lock interval (one SOF per symbol while locked, also at a
    # slip; the frame in flight at a slip is invalid and skipped, the count still advances)
    lk = pins["lock"]
    assert int(np.sum((lk[:-1] == 0) & (lk[1:] == 1))) == 1 and np.all(lk[np.argmax(lk == 1):] == 1), "expected one lock interval"
    locked = [(c, ok) for c, ok in fr["frames"] if pins["lock"][c] == 1]
    vf = [c for c, ok in locked if ok]
    idx = np.array([i for i, (c, ok) in enumerate(locked) if ok])
    assert len(vf) >= 300
    B = np.array([[s8(pins["uo"][c + t]) for t in range(N)] for c in vf])
    I, Q = B[:, 0::2], B[:, 1::2]                         # pair m = bytes (2m, 2m+1)
    E = I.astype(float) ** 2 + Q.astype(float) ** 2
    pair = E.argmax(1)
    second = np.sort(E, axis=1)[:, -2]
    assert np.all(E.max(1) >= 50 * np.maximum(second, 1.0)), "each frame must carry one dominant bin"
    mag = np.sqrt(E.max(1))
    G = io.GAIN                                          # last stage without /2: bus = FFT * G / N
    dc = np.flatnonzero(mag > 45 * G)                    # k = 0: |X0| = G * a = 120; k > 0: G * a/2 = 60
    k_of = (idx - idx[dc[0]]) % (N // 2)                 # the walk advances one bin per symbol
    assert set(dc) == set(np.flatnonzero(k_of == 0))
    seen = {k: sorted(set(pair[k_of == k].tolist())) for k in range(N // 2)}
    assert all(len(v) == 1 for v in seen.values()), seen
    order = [next(k for k, v in seen.items() if v[0] == m) for m in range(N // 2)]
    dut._log.info(f"bin order on the bus (pair m = bytes 2m, 2m+1): {order}; expected: {list(io.BIN_ORDER)}")
    assert tuple(order) == io.BIN_ORDER
    # byte order: the DC frames carry +-a in the I byte (byte 0) and exactly 0 in the Q byte (byte 1)
    assert np.all(np.abs(B[dc, 0]) >= 58 * G) and np.all(B[dc, 1] == 0), (B[dc, :2][:5])
    assert np.all(np.abs(mag[k_of != 0] - 30 * G) <= 4)
    # latency: last / first sample of a window (captured at edge i + 1) -> its SOF + byte 0 (launched at edge c)
    w = frame_windows(recs, vf)
    lat_last = sorted({c - (il + 1) for c, i0, il in w})
    lat_first = sorted({c - (i0 + 1) for c, i0, il in w})
    dut._log.info(f"latency: window's last sample captured at edge E -> SOF/byte 0 launched at edge E + {lat_last}; "
                  f"first sample -> E + {lat_first} ({len(w)} frames, CP = {CP}, L = {L}; {len(fr['dbg'])} slips)")
    assert len(lat_last) == 1 and len(lat_first) == 1 and lat_first[0] - lat_last[0] == N - 1


@cocotb.test()
async def test_adc_conversion(dut):
    """Offset binary -> two's complement (0x80 = mid-scale): all 256 codes at the capture register for both edges
    (white box), then DC corner codes through the FFT to the pins (bin 0 I = sat(2 (code - 128)) with the gain 2/N
    (last stage without /2), every other byte 0)."""
    await io.start(dut)
    T = CLK_PS
    for cfg_edge in ((0, 1) if not io.GL else ()):           # gate level: no `cap` net name; the pin part covers it
        dut.uio_in.value = cfg_edge << 5
        codes = list(range(256)) + [0x00, 0xFF, 0x80, 0x7F, 0x81, 0x01]
        got = []
        for k in range(len(codes) + 3):
            v = dut.user_project.cap.value                   # after edge k: the code driven in cycle k - 1
            got.append(int(v) if v.is_resolvable else None)
            dut.ui_in.value = codes[k] if k < len(codes) else 0x80
            await Timer(T, unit="ps")
        conv = [None if g is None else s8(g) for g in got[1:len(codes) + 1]]
        bad = [(c, g) for c, g in zip(codes, conv) if g != c - 128]
        assert not bad, f"CFG_EDGE={cfg_edge}: capture register wrong for {bad[:5]}"
        dut._log.info(f"CFG_EDGE={cfg_edge}: 256 codes -> capture register = code - 128 (0x00 -> -128, 0x7F -> -1, "
                      "0x80 -> 0, 0xFF -> +127)")
    # DC through the FFT: lock on a constant (every sign matches: hypothesis 0 locks), then the corner codes
    dut.uio_in.value = 0
    corners = (0x00, 0x01, 0x7F, 0x80, 0x81, 0xFE, 0xFF)
    adc = np.array([0x90] * (40 * L) + sum(([c] * (4 * L) for c in corners), []) + [0x90] * (2 * L))
    rst = np.ones(len(adc), dtype=int)
    rst[:4] = 0
    pins = await io.stream(dut, adc, rst)
    exp, recs = io.model_pins(adc, rst, keep=True)
    bad = io.compare(pins, exp)
    assert not bad, f"{len(bad)} pin mismatches, first {bad[:6]}"
    fr = io.frames_from_pins(pins)
    seen = {}
    for c, i0, il in frame_windows(recs, [c for c, ok in fr["frames"] if ok]):
        win = adc[i0:il + 1]
        if np.all(win == win[0]) and win[0] in corners:
            by = [s8(pins["uo"][c + t]) for t in range(N)]
            assert by == [max(-128, min(127, io.GAIN * (int(win[0]) - 128)))] + [0] * (N - 1), f"DC code {win[0]:#04x}: bytes {by}"
            seen[int(win[0])] = by[0]
    assert sorted(seen) == sorted(corners), seen
    dut._log.info("DC corners through the FFT, bin 0 I byte: " + ", ".join(f"{c:#04x} -> {v:+d}" for c, v in sorted(seen.items())))


def reset_edges(rst, n):
    """Edges k at which the control registers see the synchronised reset: rst[j] is driven after edge j, taken by the
    synchroniser at edges j + 1 and j + 2, so it acts at edge j + 3. Before the first drive rst_n is low (and
    chip_io.start() has already flushed the synchroniser, so edges 0..2 are reset edges)."""
    return [k for k in range(0, n) if (rst[k - 3] if k >= 3 else 0) == 0]


def check_reset_pins(pins, lo_ph, in_rst, what):
    """While the synchronised reset acts: LOCK / VALID / SOF / DBG = 0; SCLK / ADC_CLK idle low, except that the
    first reset cycle of an episode may complete one full-width pulse (the toggle flop is reset at that edge while
    the negedge follower still holds the old level; it is low again after the falling edge)."""
    first = {k for k in in_rst if k - 1 not in in_rst}
    for k in in_rst:
        for f in ("sof", "valid", "lock", "dbg"):
            assert pins[f][k] == 0, f"{what}: {f} = {pins[f][k]} during reset (cycle {k})"
        for f in ("sclk", "aclk"):
            if k in first:
                assert lo_ph[f][k] == 0 or (k == min(in_rst) and lo_ph[f][k] < 0), f"{what}: {f} in reset, {k}"
            else:
                assert (pins[f][k], lo_ph[f][k]) == (0, 0), f"{what}: {f} not idle low in reset, cycle {k}"


@cocotb.test()
async def test_reset(dut):
    """uio_oe constant from t = 0; during reset SOF = VALID = LOCK = DBG = 0 and SCLK = ADC_CLK = 0; SCLK runs
    from the first edge after release; a reset while locked drops LOCK at once and the chip re-acquires."""
    dut.ena.value = 1
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await Timer(1, unit="ns")
    assert int(dut.uio_oe.value) == io.OE, "uio_oe must be constant, also before the first clock edge"
    await io.start(dut)
    adc, _, _ = ofdm(1300, seed=31, phase0=4.0)
    n, r0, r1 = len(adc), 10, 13000
    rst = np.ones(n, dtype=int)
    rst[:r0] = 0
    rst[r1:r1 + 3] = 0
    pins = await io.stream(dut, adc, rst, extra_phases=(0.75,))
    lo_ph = pins["extra"][0.75]
    exp = io.model_pins(adc, rst)
    bad = io.compare(pins, exp)
    assert not bad, f"{len(bad)} pin mismatches, first {bad[:6]}"
    in_rst = reset_edges(rst, n)                                   # edges where the synchronised reset acts
    check_reset_pins(pins, lo_ph, in_rst, "reset test")
    run = [k for k in range(r0 + 3, n) if rst[k - 3] == 1]
    assert all(pins["sclk"][k] == 1 and lo_ph["sclk"][k] == 0 and pins["aclk"][k] == 1 and lo_ph["aclk"][k] == 0 for k in run)
    assert pins["lock"][r1 + 2] == 1 and pins["lock"][r1 + 3] == 0, "LOCK must drop 3 edges after rst_n is driven low"
    relock = r1 + 6 + int(np.argmax(pins["lock"][r1 + 6:] == 1))
    assert pins["lock"][relock] == 1 and np.sum(pins["valid"][relock:]) >= 100 * N
    dut._log.info(f"reset: pins quiet in {len(in_rst)} reset cycles, SCLK from the first edge after release; first lock "
                  f"at {int(np.argmax(pins['lock'] == 1))}; reset at {r1} while locked; re-lock at {relock} "
                  f"({(relock - r1) / L:.0f} symbols); bit-exact throughout")


@cocotb.test()
async def test_bus_timing(dut):
    """Cycle-by-cycle bus timing (docs/info.md): pins read at T/4, 3T/8, 5T/8, 7T/8 and every value change
    time-stamped. SCLK / ADC_CLK rise at the rising clock edge and fall at the falling edge; uo_out, SOF, VALID,
    LOCK, DBG change only at the rising edge, so a reader sampling on the falling SCLK edge has T/2 on each side
    (RTL; the post-layout skew comes from static timing, docs/info.md)."""
    from cocotb.simtime import get_sim_time
    await io.start(dut)
    T = CLK_PS
    t_edge0 = get_sim_time(unit="ps") - T // 4
    log = []

    async def watch(sig, name):
        prev = str(sig.value)
        while True:
            await sig.value_change
            cur = str(sig.value)
            log.append((get_sim_time(unit="ps") - t_edge0, name, prev, cur))
            prev = cur
    w1 = cocotb.start_soon(watch(dut.uo_out, "uo_out"))
    w2 = cocotb.start_soon(watch(dut.uio_out, "uio_out"))
    adc, _, _ = ofdm(600, seed=41, phase0=0.0)
    rst = np.ones(len(adc), dtype=int)
    rst[:4] = 0
    ph = (0.375, 0.625, 0.875)
    pins = await io.stream(dut, adc, rst, extra_phases=ph)
    w1.cancel()
    w2.cancel()
    exp = io.model_pins(adc, rst)
    assert not io.compare(pins, exp)
    ks = np.arange(3 * N + 8, len(adc))
    for f in ("uo", "sof", "valid", "lock", "dbg"):
        for p in ph:
            d = np.flatnonzero(pins["extra"][p][f][ks] != pins[f][ks])
            assert not len(d), f"{f} changed inside the cycle (read at {p} T vs T/4) at cycles {ks[d][:5]}"
    for f in ("sclk", "aclk"):
        assert np.all(pins[f][ks] == 1) and np.all(pins["extra"][0.375][f][ks] == 1)
        assert np.all(pins["extra"][0.625][f][ks] == 0) and np.all(pins["extra"][0.875][f][ks] == 0)
    # value-change time stamps, per pin, as an offset into the clock period
    pin_bits = {"SCLK": 0, "SOF": 1, "VALID": 2, "ADC_CLK": 3, "LOCK": 4, "DBG": 7}
    when = {k: set() for k in list(pin_bits) + ["uo_out"]}
    t_min = (3 * N + 8) * T
    for t, name, a, b in log:
        if t < t_min:
            continue
        if name == "uo_out":
            when["uo_out"].add(t % T)
            continue
        for pn, i in pin_bits.items():
            if a[7 - i] != b[7 - i]:
                when[pn].add(t % T)
    dut._log.info("value-change offsets into the period (ps): " + ", ".join(f"{k} {sorted(v)}" for k, v in when.items()))
    if not io.GL:                                       # RTL: exactly at the edges
        assert when["SCLK"] == {0, T // 2} and when["ADC_CLK"] == {0, T // 2}
        for pn in ("SOF", "VALID", "LOCK", "uo_out"):
            assert when[pn] == {0}, f"{pn} changes at {sorted(when[pn])} ps into the period"
        assert when["DBG"] <= {0}
    else:                                               # gate level (unit cell delays): within T/4 after the edge
        rise = lambda v: all(0 <= t < T // 4 for t in v)
        assert all(rise({t for t in when[p] if t < T // 2}) and rise({t - T // 2 for t in when[p] if t >= T // 2})
                   for p in ("SCLK", "ADC_CLK"))
        for pn in ("SOF", "VALID", "LOCK", "DBG", "uo_out"):
            assert rise(when[pn]), f"{pn} changes at {sorted(when[pn])} ps into the period"
    fr = io.frames_from_pins(pins)
    assert all(pins["sof"][s] == 1 and ln == N for s, ln in fr["runs"]) and len(fr["runs"]) >= 200
    dut._log.info(f"SOF = first VALID byte in all {len(fr['runs'])} frames; VALID high {N} cycles, low {CP} cycles "
                  f"(SOF period {sorted(set(fr['periods']))}); data valid for the whole period around the SCLK fall")


@cocotb.test()
async def test_adc_tco_sweep(dut):
    """ADC clocked by the shared external clock: sample k is valid t_co after clock edge k (the clock and
    the data see the same pad + mux insertion), with +-U of uncertainty (pin-to-pin skew < 2 ns
    plus flop setup/hold) driven as X on icarus. Swept t_co = 0..30 ns for both CFG_EDGE settings.
    Prediction: posedge capture is clean unless an edge k or k+1 lies in [t_co - U, t_co + U]; negedge capture
    unless T/2 does; latency 2 edges, 3 for negedge capture with t_co - U > T/2. Verilator (2-state) runs the
    predicted-clean points only, without the X window."""
    from cocotb.simtime import get_sim_time
    from cocotb.triggers import RisingEdge
    T, U = CLK_PS, 2500
    x_inj = SIM_IS_ICARUS
    await io.start(dut)
    rng = np.random.default_rng(9)
    table = {}
    for cfg_edge in (0, 1):
        for tco_ns in range(0, 31):
            tco = tco_ns * 1000
            lo_w, hi_w = tco - U, tco + U
            if cfg_edge == 0:
                safe, lat = not (lo_w <= 0 or hi_w >= T), 2
            else:
                safe = not (lo_w <= T // 2 <= hi_w or lo_w <= 3 * T // 2 <= hi_w)
                lat = 2 if hi_w < T // 2 else 3
            if not safe and not x_inj:
                table[(cfg_edge, tco_ns)] = None
                continue
            n, nrst = 150, 4
            adc = rng.integers(0, 256, n)
            rst = np.ones(n, dtype=int)
            rst[:nrst] = 0
            dut.uio_in.value = cfg_edge << 5
            await RisingEdge(dut.clk)                        # edge 0 of this point
            t0 = get_sim_time(unit="ps")
            ev = []
            for k in range(n):
                if x_inj and k * T + lo_w > 0:
                    ev.append((k * T + lo_w, 0, k))         # data uncertain
                ev.append((k * T + hi_w, 1, k))             # data valid (value of sample k)
                ev.append((k * T + T // 4, 2, k))           # read pins after edge k, drive rst_n for edge k + 1
            ev.sort()
            pins = {f: np.zeros(n, dtype=np.int64) for f in io.FIELDS}
            now = 0
            for t, kind, k in ev:
                if t > now:
                    await Timer(t - now, unit="ps")
                    now = t
                if kind == 0:
                    dut.ui_in.value = LogicArray("X" * 8)
                elif kind == 1:
                    dut.ui_in.value = int(adc[k])
                else:
                    p = io.read_pins(dut)
                    for f in io.FIELDS:
                        pins[f][k] = p[f]
                    dut.rst_n.value = int(rst[k])
            assert get_sim_time(unit="ps") - t0 == now
            lo, hi = nrst + 3 * N, n
            ok_x = bool(np.all(pins["uo"][lo:hi] >= 0))
            d, score = io.find_latency(adc, rst, pins, lo, hi)
            exp = io.model_pins(adc, rst, latency=d)
            clean = ok_x and not io.compare(pins, exp, lo, hi)
            table[(cfg_edge, tco_ns)] = (clean, d if clean else None)
            assert clean == safe, f"CFG_EDGE={cfg_edge} t_co={tco_ns} ns: clean {clean}, predicted {safe} ({score})"
            if clean:
                assert d == lat, f"CFG_EDGE={cfg_edge} t_co={tco_ns} ns: latency {d}, predicted {lat}"
    row = lambda e: " ".join("." if table[(e, t)] is None else (str(table[(e, t)][1]) if table[(e, t)][0] else "x")
                             for t in range(31))
    dut._log.info(f"t_co 0..30 ns, U = +-{U / 1000} ns, T = {T / 1000} ns (digit = latency in edges, x = corrupted, "
                  f". = not run):\n  posedge: {row(0)}\n  negedge: {row(1)}")
    for t in range(31):
        assert any(table[(e, t)] and table[(e, t)][0] for e in (0, 1)) or not x_inj, f"t_co = {t} ns: no clean setting"



@cocotb.test()
async def test_reset_release_phases(dut):
    """One reset pulse per release phase on a locked 30 dB stream. rst_n is driven low a
    quarter period after edge a and released at phase phi after edge a + 4, phi in {0 (exactly at the rising edge),
    T/8, ..., T/2 (exactly at the falling edge), ..., 7T/8}. The synchroniser takes a release at phi > 0 at edge
    a + 5; at phi = 0 either edge is legal (simulator ordering, metastability in silicon) and the alignment is
    searched. Checks: all pins bit-exact with the model; LOCK / VALID / SOF / DBG = 0 while the synchronised reset
    acts; SCLK / ADC_CLK change only at the clock edges (no runt pulse), idle low in reset, running from the first
    edge after release; every lock interval after a release demodulates with BER 0 from the pins."""
    import demod as dm
    import ofdm_loop as ol
    from cocotb.simtime import get_sim_time
    T = CLK_PS
    phases = [int(round(i * T / 8)) for i in range(8)]
    ep = 800 * L                                          # cycles per episode (re-acquisition + >= 400 frames)
    n = ep * (len(phases) + 1)
    cfg = sg.OFDMConfig(N=N)
    adc, tx, bd = sg.generate(cfg, n // L + 2, 777, snr_db=30, phase0=6.4)
    adc = adc[:n]
    assert_at = [ep * (i + 1) for i in range(len(phases))]
    await io.start(dut)
    t0 = get_sim_time(unit="ps") - T // 4                 # edge 0
    log = []

    async def watch():
        prev = str(dut.uio_out.value)
        while True:
            await dut.uio_out.value_change
            cur = str(dut.uio_out.value)
            log.append((get_sim_time(unit="ps") - t0, prev, cur))
            prev = cur
    w = cocotb.start_soon(watch())
    ev = []
    for k in range(n):
        ev.append((k * T + T // 4, 1, k))                 # read pins, drive ui_in (and rst_n low at an assert cycle)
        ev.append((k * T + 3 * T // 4, 2, k))             # read pins in the second half (SCLK low)
    for a, ph in zip(assert_at, phases):
        ev.append(((a + 4) * T + ph, 0, a))               # release (kind 0 sorts before a read at the same time)
    ev.sort()
    pins = {f: np.zeros(n, dtype=np.int64) for f in io.FIELDS}
    lo_ph = {f: np.zeros(n, dtype=np.int64) for f in io.FIELDS}
    now = T // 4
    for t, kind, k in ev:
        if t < now:
            continue                                      # (edge 0 events before the start point)
        if t > now:
            await Timer(t - now, unit="ps")
            now = t
        if kind == 0:
            dut.rst_n.value = 1
        elif kind == 1:
            p = io.read_pins(dut)
            for f in io.FIELDS:
                pins[f][k] = p[f]
            dut.ui_in.value = int(adc[k])
            if k in assert_at:
                dut.rst_n.value = 0
            elif k == 3:
                dut.rst_n.value = 1                        # end of the power-up reset
        else:
            p = io.read_pins(dut)
            for f in io.FIELDS:
                lo_ph[f][k] = p[f]
    w.cancel()
    # per-cycle rst (value taken at edge k + 1); phase 0: the release may be taken at edge a + 4 or a + 5
    base = np.ones(n, dtype=int)
    base[:3] = 0
    for a in assert_at:
        base[a:a + 4] = 0                                 # rst[k] = level taken at edge k + 1: low at a + 1 .. a + 4,
    alt = base.copy()                                     # high again at a + 5 (release after edge a + 4, phi > 0)
    alt[assert_at[0] + 3] = 1                             # phi = 0 episode: release taken at edge a + 4 itself
    best = None
    for name, r in (("edge a+5", base), ("edge a+4", alt)):
        exp = io.model_pins(adc, r)
        bad = io.compare(pins, exp)
        if best is None or len(bad) < len(best[2]):
            best = (name, r, bad, exp)
    name, rst, bad, exp = best
    assert not bad, f"{len(bad)} pin mismatches, first {bad[:6]}"
    dut._log.info(f"release exactly at the rising edge was taken at {name}; all {n} cycles bit-exact")
    in_rst = reset_edges(rst, n)
    check_reset_pins(pins, lo_ph, in_rst, "release phases")
    run = [k for k in range(6, n) if k not in in_rst]
    assert all(pins[f][k] == 1 and lo_ph[f][k] == 0 for k in run for f in ("sclk", "aclk")), "SCLK not running"
    offs = set()
    for t, a_, b_ in log:
        for i in (0, 3):
            if t >= 4 * T and a_[7 - i] != b_[7 - i]:
                offs.add(t % T)
    dut._log.info(f"SCLK / ADC_CLK transition offsets into the period over {len(phases)} resets: {sorted(offs)} ps")
    assert offs <= ({0, T // 2} if not io.GL else set(range(0, T // 4)) | set(range(T // 2, T // 2 + T // 4)))
    # re-acquisition and BER 0 from the pins only
    p = {f: np.where(pins[f][2:] < 0, 0, pins[f][2:]) for f in ("sof", "valid", "lock")}
    p["uo"] = np.where(pins["uo"][2:] < 0, 0, pins["uo"][2:])
    res = dm.demodulate(p, cfg, dd=False, avg_window=dm.AVG_WINDOW)
    sc = ol.score_chip(cfg, res, tx, bd)
    ivs = res["intervals"]
    relock = []
    for a, ph in zip(assert_at, phases):
        nxt = [iv for iv in ivs if iv["start_cycle"] is not None and iv["start_cycle"] + 2 > a]
        assert nxt and nxt[0]["prbs"] is not None and nxt[0]["n_valid"] >= 300, f"no re-lock after the release at {ph} ps"
        relock.append((ph, (int(np.argmax(pins["lock"][a + 5:] == 1))) / L))
    dut._log.info("re-lock after release (phase ps -> symbols): " + ", ".join(f"{ph}: {s:.0f}" for ph, s in relock) +
                  f"; {len(ivs)} lock intervals, {sc['frames']} frames, BER {sc['ber']:.1e} ({sc['errs']}/{sc['nbits']})")
    assert len(ivs) == len(phases) + 1 and sc["ber"] == 0.0


def half_bin_tone_symbols(nsym, seed, amp=127, f0=0.5):
    """CP-structured symbols carrying one full-scale real tone half-way between bins (f0 bins, next to DC): useful part
    amp * cos(2 pi f0 u / N + phi_j) with a random phase per symbol, CP = a copy of its last CP samples (so the
    correlator locks). The window leaks energy into bins 0 / 1 well beyond full scale at gain 2/N."""
    rng = np.random.default_rng(seed)
    u = np.arange(N)
    out = []
    for _ in range(nsym):
        x = np.clip(np.round(amp * np.cos(2 * np.pi * f0 * u / N + rng.uniform(0, 2 * np.pi))), -128, 127)
        out.append(np.concatenate([x[N - CP:], x]))
    return (np.concatenate(out) + 128).astype(int)


@cocotb.test()
async def test_saturation_directed(dut):
    """Saturation, directed: (A) DC = +64 / -65 / +63 LSB (locked on the constant): bin 0 reads 127 / -128 (saturated)
    / 126; (B) after a reset, CP-structured full-scale half-bin tones next to DC: bins 0 / 1 saturate. Checks: all pins
    bit-exact; every VALID byte equals the ideal FFT * 2/N of its window clipped to 8 bits within the model bound,
    and components the ideal puts beyond the rail by more than the bound sit exactly on the rail with the ideal's sign
    (no wrap); LOCK / VALID are unaffected (one lock interval per section, frame bookkeeping, the host parser sees
    every frame)."""
    import demod as dm
    await io.start(dut)
    sec_a = sum(([0x80 + v] * (12 * L) for v in (64, -65, 63)), [0x80 + 64] * (40 * L))
    sec_b = list(half_bin_tone_symbols(500, seed=8))
    adc = np.array(sec_a + [0x80] * 8 + sec_b)
    rst = np.ones(len(adc), dtype=int)
    rst[:4] = 0
    rb = len(sec_a)
    rst[rb:rb + 4] = 0                                      # reset between the sections
    pins = await io.stream(dut, adc, rst)
    exp, recs = io.model_pins(adc, rst, keep=True)
    bad = io.compare(pins, exp)
    assert not bad, f"{len(bad)} pin mismatches, first {bad[:6]}"
    fr = io.frames_from_pins(pins)
    vf = [c for c, ok in fr["frames"] if ok]
    bound = 4.5                                             # model error bound at N = 16 (model/test_model.py BOUNDS)
    seen_dc, n_sat, n_frames_b = {}, 0, 0
    for c, i0, il in frame_windows(recs, vf):
        w = adc[i0:il + 1].astype(int) - 128
        by = np.array([s8(pins["uo"][c + t]) for t in range(N)])
        ideal = np.fft.fft(w) * io.GAIN / N
        for m in range(N // 2):
            k = io.BIN_ORDER[m]
            comps = [(by[2 * m], ideal[k].real)] + ([] if k == 0 else [(by[2 * m + 1], ideal[k].imag)])
            for part, val in comps:
                assert abs(part - max(-128.0, min(127.0, val))) <= bound, f"frame {c} bin {k}: {part} vs ideal {val:.1f}"
                if val > 127 + bound or val < -128 - bound:     # surely saturated: exactly on the rail, same sign
                    assert part == (127 if val > 0 else -128), f"frame {c} bin {k}: {part} vs ideal {val:.1f} (wrap?)"
                    n_sat += 1
        if np.all(w == w[0]):
            seen_dc[int(w[0])] = int(by[0])
        if c > rb:
            n_frames_b += 1
    assert seen_dc == {64: 127, -65: -128, 63: 126}, seen_dc
    lk = pins["lock"]
    rises = np.flatnonzero((lk[:-1] == 0) & (lk[1:] == 1))
    assert len(rises) == 2 and rises[0] < rb < rises[1], f"lock rises {rises}"
    assert np.all(lk[rises[1] + 1:] == 1), "LOCK dropped while the bins saturate"
    check_bookkeeping(dut, pins, "saturation directed")
    iv = dm.parse_pins(pins["uo"] & 0xFF, pins["sof"], pins["valid"], np.where(lk < 0, 0, lk), N, L)
    sym_b = (len(adc) - rises[1]) // L
    dut._log.info(f"DC: +64 -> {seen_dc[64]}, -65 -> {seen_dc[-65]}, +63 -> {seen_dc[63]}; half-bin tones: {n_frames_b} VALID "
                  f"frames, {n_sat} components beyond the rail land exactly on it with the ideal's sign (no wrap); "
                  f"model saturation events {exp['sat']}; host parser: {len(iv)} lock intervals, {len(iv[-1])} frames in "
                  f"section B for {sym_b} locked symbols")
    assert n_sat >= 100 and exp["sat"] >= 100 and len(iv) == 2 and n_frames_b >= 400
    assert len(iv[-1]) >= sym_b - 1


SIM_IS_ICARUS = io.SIM.startswith("icarus")

# ---- the clock gate seen from inside (RTL and gate level) ----------------------------------------------
def clk_dp_registers():
    """Gate level: the datapath flops (clocked from the clk_dp tree) and the clk_dp leaf clock nets, read from the
    netlist this simulation was built with (test/gate_level_netlist.v)."""
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_level_netlist.v")).read()
    flops, nets = [], set()
    for m in re.finditer(r"sky130_fd_sc_hd__dfxtp_\d+\s+(\S+)\s*\((.*?)\);", txt, re.S):
        ck = dict(re.findall(r"\.(\w+)\(([^)]*)\)", m.group(2)))["CLK"].strip()
        if "clk_dp" in ck:
            flops.append(m.group(1))
            nets.add(ck)
    return flops, sorted(nets)


@cocotb.test()
async def test_clock_gate_pauses(dut):
    """clk_dp stops in every CP pause and no datapath flop changes there; it runs, and the datapath moves,
    otherwise. Per cycle the ICG enable `run` is read in the low phase of clk (latch transparent); the next rising
    edge of clk must give one clk_dp edge (gate level: one edge on every clk_dp leaf net) exactly when it was 1, and
    no datapath register may change when it was 0. The pins stay bit-exact against the model meanwhile."""
    top = dut.user_project
    run, gclk = top.run, top.clk_dp
    if io.GL:
        names, leaf_names = clk_dp_registers()
        qs = [top._id(n, extended=False).Q for n in names]
        leaves = [top._id(n, extended=False) for n in leaf_names]
    else:
        names, qs, leaves = [], [], []
    edges = {"dp": 0, "leaf": 0}

    async def count(sig, key):
        while True:
            await RisingEdge(sig)
            edges[key] += 1

    adc, _, _ = ofdm(320, seed=303)
    rst = np.ones(len(adc), dtype=int)
    rst[:4] = 0
    await io.start(dut)
    counters = [cocotb.start_soon(count(gclk, "dp"))] + [cocotb.start_soon(count(s, "leaf")) for s in leaves]
    log = []

    async def monitor(n):
        for _ in range(n):
            await FallingEdge(dut.clk)
            v = run.value
            log.append((int(v) if v.is_resolvable else -1, edges["dp"], edges["leaf"],
                        "".join(str(q.value) for q in qs)))

    mon = cocotb.start_soon(monitor(len(adc) - 1))
    pins = await io.stream(dut, adc, rst)
    await mon
    for c in counters:
        c.cancel()
    bad = io.compare(pins, io.model_pins(adc, rst))
    assert not bad, f"pins differ from the bit-exact model: {len(bad)}, first {bad[:5]}"
    en = np.array([r[0] for r in log])
    assert np.all(en >= 0), f"ICG enable `run` not resolvable at cycles {np.flatnonzero(en < 0)[:5]}"
    dp = np.diff([r[1] for r in log])
    lf = np.diff([r[2] for r in log])
    e = en[:-1]
    assert np.array_equal(dp, e), f"clk_dp edges != run at cycles {np.flatnonzero(dp != e)[:10]}"
    if io.GL:
        assert np.array_equal(lf, e * len(leaves)), f"leaf clock edges != run at {np.flatnonzero(lf != e * len(leaves))[:10]}"
        moved = np.array([log[i + 1][3] != log[i][3] for i in range(len(log) - 1)])
        assert "x" not in log[-1][3].lower(), "datapath flops still X at the end"
        assert not np.any(moved & (e == 0)), f"datapath flop changed in a pause at cycles {np.flatnonzero(moved & (e == 0))[:10]}"
    first_lock = int(np.argmax(pins["lock"] == 1))
    assert pins["lock"][first_lock] == 1, "no lock"
    after = np.arange(len(e)) >= first_lock
    paused = int(np.sum((e == 0) & after))
    running = int(np.sum((e == 1) & after))
    assert paused > 0 and np.all(e[~after] == 1), "pauses only once locked (run = ~lock | ph < N)"
    msg = (f"clock gate: LOCK at cycle {first_lock}; {paused} paused / {running} running cycles after lock; clk_dp "
           f"edges = run every cycle ({len(e)} cycles)")
    if io.GL:
        mv = int(np.sum(moved & (e == 1) & after))
        msg += (f"; {len(leaves)} leaf nets edge-exact; {len(names)} datapath flops: 0 changes in pauses, changes in "
                f"{mv}/{running} running cycles")
        assert mv >= 0.95 * running
    dut._log.info(msg)
