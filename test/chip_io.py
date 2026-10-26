# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin-level driver, monitor and oracle for tt_um_cl_hw_silicon_fft16, shared by test_top.py and test_ofdm.py.

Clock convention. `start()` starts `clk` and waits for its first rising edge (edge 0). Every later step
happens a fixed phase after an edge: by default at T/4, the drivers write ui_in / rst_n for the next
edge and the monitor reads the pins, which then show the state after that edge. pins[k] is what a
reader sees between edge k and edge k+1 (it samples on the SCLK falling edge, T/2 after edge k).
adc[k] is driven after edge k, so it is captured at edge k+1 (either CFG_EDGE setting with this
drive phase) and enters the timing block and the datapath at edge k+2: the model consumes it two edges
later (`latency` = 2; found by search once, then required to be exact).

Oracle: model/system_model.FFTChipModel, register for register, including the two-flop reset synchroniser:
a level on rst_n driven after edge k reaches the control registers at edge k + 3. The datapath has no
reset (by design), so uo_out is compared only after 2N datapath cycles since reset (the model starts its delay
lines at 0). start() flushes the synchroniser with three reset edges before edge 0, so the control registers are
in reset from edge 0 in the RTL and in the model alike (the synchroniser flops power up X).
"""
import os

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from system_model import FFTChipModel

N = 16
CP = N // 4                       # CP fixed at N/4 = 4 (no CFG_CP at N = 16)
L = N + CP
OE = 0x9F                         # uio_oe = 8'b1001_1111 (uio[5], uio[6] are inputs)
BIN_ORDER = (0, 4, 2, 6, 1, 5, 3, 7)   # bus pair m (bytes 2m, 2m+1) carries bin bitrev4(2m), see docs/info.md
GAIN = 2                          # bus value = FFT * GAIN / N, saturated to 8 bits (last stage without /2)
CLK_PS = int(os.environ.get("CLK_PS", "31250"))
SIM = cocotb.SIM_NAME.lower() if cocotb.SIM_NAME else ""
GL = os.environ.get("GATES", "") == "yes"     # gate-level run (test/Makefile GATES=yes): no internal names, cell delays
FIELDS = ("uo", "sof", "valid", "lock", "dbg", "oe", "sclk", "aclk", "uio56")


def _logic(v):
    """LogicArray -> (int with X/Z bits as 0, mask of X/Z bits)."""
    if v.is_resolvable:
        return int(v), 0
    s = str(v).lower()
    val = mask = 0
    for ch in s:
        val, mask = val << 1, mask << 1
        if ch == "1":
            val |= 1
        elif ch != "0":
            mask |= 1
    return val, mask


def read_pins(dut):
    """One snapshot: dict of pin values; -1 for a field with any X/Z bit."""
    uo, uom = _logic(dut.uo_out.value)
    uio, uiom = _logic(dut.uio_out.value)
    oe, oem = _logic(dut.uio_oe.value)

    def b(i):
        return -1 if (uiom >> i) & 1 else (uio >> i) & 1
    return dict(uo=-1 if uom else uo, sclk=b(0), sof=b(1), valid=b(2), aclk=b(3), lock=b(4), dbg=b(7),
                uio56=-1 if (uiom >> 5) & 3 else (uio >> 5) & 3, oe=-1 if oem else oe)


async def start(dut, cfg_edge=0, uio6=0, clk_ps=CLK_PS):
    """Clock on, rst_n low, config pins set; returns at edge 0 + T/4. Three reset edges precede edge 0 so that the
    two-flop reset synchroniser holds reset and the control registers are already reset at edge 0, as the
    model assumes, whatever the previous test left in the simulator."""
    dut.ena.value = 1
    dut.uio_in.value = (cfg_edge << 5) | (uio6 << 6)
    dut.ui_in.value = 0x80
    dut.rst_n.value = 0
    cocotb.start_soon(Clock(dut.clk, clk_ps, unit="ps").start())
    for _ in range(4):                                  # 3 flush edges + edge 0
        await RisingEdge(dut.clk)
    await Timer(clk_ps // 4, unit="ps")


async def stream(dut, adc, rst, clk_ps=CLK_PS, extra_phases=()):
    """From edge 0 + T/4 (after start()): per cycle k read the pins, then drive ui_in = adc[k], rst_n = rst[k].
    extra_phases: further read points inside the cycle, as fractions of T after the edge (> 1/4, < 1).
    Returns {field: np.array} (and {phase: {field: array}} under 'extra')."""
    n = len(adc)
    out = {f: np.zeros(n, dtype=np.int64) for f in FIELDS}
    extra = {ph: {f: np.zeros(n, dtype=np.int64) for f in FIELDS} for ph in extra_phases}
    q = clk_ps // 4
    marks = [q] + [int(round(ph * clk_ps)) for ph in extra_phases] + [clk_ps + q]
    steps = [b - a for a, b in zip(marks, marks[1:])]
    for k in range(n):
        p = read_pins(dut)
        for f in FIELDS:
            out[f][k] = p[f]
        dut.ui_in.value = int(adc[k])
        dut.rst_n.value = int(rst[k])
        for i, ph in enumerate(extra_phases):
            await Timer(steps[i], unit="ps")
            p = read_pins(dut)
            for f in FIELDS:
                extra[ph][f][k] = p[f]
        await Timer(steps[-1], unit="ps")
    if extra_phases:
        out["extra"] = extra
    return out


def model_pins(adc, rst, latency=2, keep=False):
    """Expected pins per cycle k (state after edge k) from FFTChipModel; uo = -2 where the model leaves the
    bus undefined (datapath not yet refilled since reset). With keep=True also returns the model records."""
    n = len(adc)
    x = [int(v) - 128 for v in adc]                     # offset binary (0x80 = mid-scale) -> two's complement
    m = FFTChipModel(N=N)
    exp = {f: np.full(n, -2, dtype=np.int64) for f in ("uo", "sof", "valid", "lock", "dbg")}
    recs = [None] * n
    ndp = 0
    for k in range(1, n):
        r = m.step(x[k - latency] if k - latency >= 0 else 0, int(rst[k - 1]))
        if not rst[k - 1]:
            ndp = 0
        elif r["dp"] is not None:
            ndp += 1
        if r["uo"] is not None and ndp > 2 * N:
            exp["uo"][k] = r["uo"] & 0xFF
        exp["sof"][k], exp["valid"][k], exp["lock"][k], exp["dbg"][k] = r["sof"], r["valid"], r["lock"], r["dbg"]
        if keep:
            recs[k] = r
    exp["sat"] = m.sdf.sat                              # last-stage saturation events in the model
    return (exp, recs) if keep else exp


def compare(pins, exp, lo=1, hi=None):
    """List of (cycle, field, pin, expected) mismatches; uo only where the model defines it."""
    hi = hi or len(pins["uo"])
    bad = []
    for f in ("uo", "sof", "valid", "lock", "dbg"):
        p, e = pins[f][lo:hi], exp[f][lo:hi]
        sel = e != -2
        for i in np.flatnonzero(sel & (p != e)):
            bad.append((lo + int(i), f, int(p[i]), int(e[i])))
    for i in np.flatnonzero(pins["oe"][lo:hi] != OE):
        bad.append((lo + int(i), "uio_oe", int(pins["oe"][lo + i]), OE))
    for i in np.flatnonzero(pins["uio56"][lo:hi] != 0):
        bad.append((lo + int(i), "uio_out[6:5]", int(pins["uio56"][lo + i]), 0))
    return sorted(bad)


def find_latency(adc, rst, pins, lo, hi, cands=(1, 2, 3, 4)):
    """Alignment search (once per setting): the model latency with the most matching cycles in [lo, hi)."""
    score = {}
    for d in cands:
        e = model_pins(adc, rst, latency=d)
        score[d] = sum(1 for k in range(lo, hi) if e["uo"][k] != -2 and pins["uo"][k] == e["uo"][k])
    best = max(score, key=score.get)
    return best, score


def frames_from_pins(pins):
    """Frame bookkeeping seen from the pins only (what a reader can check). Returns a dict:
    sof (cycles), periods between SOFs inside one lock interval, valid runs [(start, length)],
    frames [(sof cycle, valid flag of its N bytes)], dbg pulses."""
    sof = np.flatnonzero(pins["sof"] == 1)
    valid = pins["valid"]
    lock = pins["lock"]
    runs, k, n = [], 0, len(valid)
    while k < n:
        if valid[k] == 1:
            j = k
            while j < n and valid[j] == 1:
                j += 1
            runs.append((k, j - k))
            k = j
        else:
            k += 1
    periods = [int(b - a) for a, b in zip(sof, sof[1:]) if lock[a] == 1 and np.all(lock[a:b + 1] == 1)]
    frames = [(int(c), bool(c + N <= n and np.all(valid[c:c + N] == 1))) for c in sof]
    return dict(sof=sof, periods=periods, runs=runs, frames=frames, dbg=np.flatnonzero(pins["dbg"] == 1))
