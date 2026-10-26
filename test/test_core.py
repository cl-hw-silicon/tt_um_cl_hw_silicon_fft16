# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FFT core + Hermitian tail behind the real ICG cell, bit-exact against model/fft_model.SDF and the
system model's tail. Inputs (run, cnt, x) are driven at the falling edge; the combinational core output
is sampled 5 ns later (before the next rising edge) and compared with SDF.step(x, cnt) on run cycles;
the tail register o is compared with the model tail state. Alignment (output latency) is searched
once, then required to be exact."""
import os
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer
import fft_model as fm

N = int(os.environ.get("N", "16"))
NB = N.bit_length() - 1
CLK_PS = int(os.environ.get("CLK_PS", "31250"))      # also run at 16 MHz (nothing is frequency-dependent)


def s8(v):
    v = int(v) & 0xFF
    return v - 256 if v > 127 else v


LAST_REF = None


class Ref:
    """Model of the datapath as wired in the chip: SDF + tail, advancing only on run cycles."""
    def __init__(self):
        self.sdf = fm.SDF(N)
        self.q_hold = self.o = None

    def cycle(self, run, cnt, x):
        if not run:
            return None
        I, Q = self.sdf.step(int(x), cnt)
        slot = (cnt + 1) % N
        o_new = I if slot % 2 == 0 else self.q_hold
        self.o, self.q_hold = o_new, Q
        return I, Q, slot


async def drive(dut, stim, flush=2 * N):
    """stim: list of (run, cnt, x). Returns per-cycle DUT samples and reference values."""
    global LAST_REF
    cocotb.start_soon(Clock(dut.clk, CLK_PS, unit="ps").start())
    ref = LAST_REF = Ref()
    got, exp, o_got, o_exp = [], [], [], []
    await FallingEdge(dut.clk)
    nrun = 0
    for k, (run, cnt, x) in enumerate(stim):
        dut.run.value, dut.cnt.value, dut.x.value = run, cnt, int(x) & 0xFF
        await Timer(5, unit="ns")
        o_exp.append(ref.o)                      # tail state before this cycle's edge
        r = ref.cycle(run, cnt, x)
        if run:
            nrun += 1
            if nrun > flush:
                yi, yq, sl = dut.yi.value, dut.yq.value, dut.slot.value
                assert yi.is_resolvable and yq.is_resolvable, f"cycle {k}: X on the core output after flush"
                got.append((k, s8(yi), s8(yq), int(sl)))
                exp.append((k,) + r)
        if nrun > flush + 2:
            o_got.append((k, s8(dut.o.value)))
        else:
            o_got.append((k, None))
        await FallingEdge(dut.clk)
    return got, exp, o_got, o_exp


def check(got, exp, o_got, o_exp, dut, what):
    # alignment search once (latency of the comparison in run cycles), then exact
    best = None
    for lag in (-1, 0, 1, 2):
        n = sum(1 for g, e in zip(got[2:300], exp[2 + lag:300 + lag]) if g[1:] == e[1:])
        if best is None or n > best[1]:
            best = (lag, n)
    dut._log.info(f"{what}: alignment search -> lag {best[0]} run cycles ({best[1]}/298 match)")
    assert best[0] == 0
    bad = [(g, e) for g, e in zip(got, exp) if g[1:] != e[1:]]
    assert not bad, f"{what}: {len(bad)} core mismatches, first {bad[:3]}"
    ob = [(g, e) for g, e in zip(o_got, o_exp) if g[1] is not None and e is not None and g[1] != e]
    n_o = sum(1 for g, e in zip(o_got, o_exp) if g[1] is not None and e is not None)
    assert not ob, f"{what}: {len(ob)} tail mismatches, first {ob[:3]}"
    dut._log.info(f"{what}: {len(got)} core outputs and {n_o} tail bytes bit-exact")


@cocotb.test()
async def test_random_free_running(dut):
    rng = np.random.default_rng(1)
    xs = rng.integers(-128, 128, 12_000)
    stim = [(1, k % N, x) for k, x in enumerate(xs)]
    check(*await drive(dut, stim), dut, f"N={N} random, free-running counter")


@cocotb.test()
async def test_every_counter_phase(dut):
    rng = np.random.default_rng(2)
    stim = []
    for c0 in range(N):                          # restart the frame counter at every phase
        for k in range(4 * N):
            stim.append((1, (c0 + k) % N, int(rng.integers(-128, 128))))
    got, exp, og, oe = await drive(dut, stim)
    check(got, exp, og, oe, dut, f"N={N} every counter phase ({N} restarts)")


@cocotb.test()
async def test_pauses_and_rephase(dut):
    """Gated clock: random pauses, CP-like pauses (N run + CP pause), counter re-phase jumps."""
    rng = np.random.default_rng(3)
    stim, cnt = [], 0
    for blk in range(160):
        mode = blk % 3
        for _ in range(N + 8):
            if mode == 0:
                run = int(rng.random() < 0.7)
            elif mode == 1:
                run = 1 if (_ < N) else 0          # chip-like: N samples then a CP pause
            else:
                run = 1
            stim.append((run, cnt, int(rng.integers(-128, 128))))
            if run:
                cnt = (cnt + 1) % N
        if blk % 10 == 9:
            cnt = int(rng.integers(0, N))          # re-phase (lock / slip / reset of the phase counter)
    check(*await drive(dut, stim), dut, f"N={N} pauses + re-phase ({len(stim)} cycles)")


@cocotb.test()
async def test_structured(dut):
    nidx = np.arange(N)
    frames = [np.zeros(N, int), np.full(N, 127), np.full(N, -128), np.array([127, -128] * (N // 2)),
              np.eye(1, N, 0, dtype=int)[0] * -128, np.eye(1, N, N - 1, dtype=int)[0] * 127,
              np.round(np.linspace(-128, 127, N)).astype(int)]
    for amp in (1, 64, 127):
        for k in range(N):
            frames.append(np.clip(np.round(amp * np.cos(2 * np.pi * k * nidx / N + 0.3)), -128, 127).astype(int))
    xs = np.concatenate([np.tile(f, 2) for f in frames])
    stim = [(1, k % N, x) for k, x in enumerate(xs)]
    check(*await drive(dut, stim), dut, f"N={N} structured ({len(frames)} frames: DC, -128, Nyquist, impulses, ramp, tones x3 amplitudes)")


@cocotb.test()
async def test_last_stage_saturation(dut):
    """Frames that drive the last stage (no /2, saturating) into saturation, bit-exact with the model:
    DC levels around +-64 LSB and at full scale, full-scale half-bin tones (next to DC and to Nyquist) at 16 phases,
    random +-extremes. The model counts the saturation events; there must be some, and they must all match."""
    rng = np.random.default_rng(4)
    nidx = np.arange(N)
    frames = [np.full(N, v) for v in (63, 64, 65, 127, -64, -65, -128)]
    for f0 in (0.5, N / 2 - 0.5):
        for ph in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            frames.append(np.clip(np.round(127 * np.cos(2 * np.pi * f0 * nidx / N + ph)), -128, 127).astype(int))
    frames += [rng.choice([-128, 127], size=N) for _ in range(64)]
    xs = np.concatenate([np.tile(f, 2) for f in frames])
    stim = [(1, k % N, x) for k, x in enumerate(xs)]
    check(*await drive(dut, stim), dut, f"N={N} last-stage saturation ({len(frames)} frames)")
    dut._log.info(f"N={N}: {LAST_REF.sdf.sat} saturation events in the model, all matched bit-exactly by the RTL")
    assert LAST_REF.sdf.sat >= 20
