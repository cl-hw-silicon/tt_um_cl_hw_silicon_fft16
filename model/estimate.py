# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bottom-up area estimate for the whole chip at a given N (default 32), sky130_fd_sc_hd size-1 areas
from the liberty, logic calibrated against Yosys synthesis (shift-add / adder logic
synthesises ~25 % smaller than "1 FA per bit"; flops are exact), plus the post-layout overheads
measured on hold-fix toy designs and the hold-only clock uncertainty of 0.1 ns.

    python3 estimate.py [N] [--twiddle 4mult|gauss]

Structure mirrors the models: fft_model.SDF (datapath on clk_dp), timing.CPTiming (clk), the
system_model tail/bookkeeping, and the I/O wrapper (pinout: docs/info.md). Not a synthesis result: the RTL
synthesis (`yosys -flatten ... stat -liberty` per module) supersedes every line.
"""
import sys

# liberty areas (um^2), sky130_fd_sc_hd tt corner (verified against the liberty)
A = dict(dff=20.02, fa=20.02, ha=12.51, xor2=8.76, xnor2=8.76, mux2=11.26, mux4=26.28, nand2=3.75,
         nor2=3.75, inv=3.75, and2=6.26, a21o=7.51, a21oi=5.00, o21ai=5.00, dlclkp=17.52, conb=3.75)
LOGIC_FACTOR = 0.75            # calibration: synthesised / estimated for combinational logic
W, TW = 8, 7


def adder(bits):  return bits * A['fa']
def subtr(bits):  return bits * (A['fa'] + A['inv'])
def negate_sat(bits): return bits * (A['inv'] + A['ha']) + 15 + bits * A['mux2']
def mux(bits):    return bits * A['mux2']
def mult(a, b):   return a * b * A['and2'] + (a - 1) * b * A['fa']           # signed array multiplier
def cmp_const(bits): return bits * 1.5 * A['nand2']
def sat(bits_in, bits_out): return 2 * cmp_const(bits_in) + bits_out * A['mux2']


def estimate(N=32, twiddle='4mult'):
    n = N.bit_length() - 1
    blocks = []   # (name, flops, logic_um2_raw, domain)

    # ---- SDF datapath (clk_dp) -----------------------------------------------------------------
    for s in range(1, n + 1):
        D = N >> s
        paths = 1 if s == 1 else 2
        flops = paths * W * D
        logic = paths * (adder(W) + subtr(W) + 2 * mux(W))
        kind = 'BF2I' if s % 2 else 'BF2II'
        if kind == 'BF2II':
            logic += negate_sat(W) + 2 * mux(W)                     # -j: negate with saturation, swap
        blocks.append((f"stage {s} {kind} D={D}{' real' if s == 1 else ''}", flops, logic, 'clk_dp'))
        if s % 2 == 0 and s < n:
            M = 1 << (n - s + 2)
            if M == 8:
                # W8: e in {0,1,2,3}: x91 CSD on (I+Q), (Q-I), -j swap, 4-way select, saturation
                logic = 2 * adder(W + 1) + 2 * 3 * adder(15) + 2 * W * A['mux4'] + 2 * sat(10, W) + 60
                blocks.append(("twiddle W8 (CSD x91 + -j)", 0, logic, 'clk_dp'))
            else:
                if twiddle == 'gauss':                               # 3 real mults: (a+b)c, b(c+s), a(s-c)
                    m = mult(W + 1, W + 1) + 2 * mult(W, W + 1) + 3 * adder(W + 1)
                else:
                    m = 4 * mult(W, W)
                logic = m + 2 * adder(2 * W) + 2 * sat(2 * W - TW + 2, W) + 2 * mux(W)
                rom = 2 * (W) * (M // 4) * 1.5 * A['nand2']           # coefficient LUT on the exponent
                expo = 3 * A['ha'] * (n - s) + (n - s) * A['and2'] * 2 + 40
                blocks.append((f"twiddle W{M} ({twiddle})", 0, logic + rom + expo, 'clk_dp'))
    blocks.append(("stage control decode (from ph)", 0, 12 * A['xor2'] + 10 * A['and2'], 'clk_dp'))
    blocks.append(("Hermitian tail: q_hold, o (mux)  [2 registers]", 2 * W, mux(W), 'clk_dp'))

    # ---- CP timing block (clk) -------------------------------------------------------------------
    CPmax = N // 4
    Kmax = 4 * N // (N // 8)
    A_bits = (4 * N).bit_length() + 1                                   # |A| <= K*CP = 4N
    ph_bits = (N + CPmax - 1).bit_length()
    sym_bits = max(Kmax, 16).bit_length()
    tim_flops = N + ph_bits + 2 * A_bits + sym_bits + 3 + 6             # sign line, ph, A, A_on, symcnt, state, weak/lock/run/stall/primed
    tim_logic = (A_bits * (A['fa'] + A['xor2'])                           # +-1 accumulate
                 + 2 * A_bits * A['fa']                                  # A_on + H, compare with A_next
                 + 3 * cmp_const(A_bits)                                 # T_lock, T_unlock, sign
                 + 6 * cmp_const(ph_bits) + ph_bits * (A['ha'] + A['mux2'])   # windows, run, ph next / stall / skip
                 + sym_bits * (A['ha'] + A['mux2']) + 120 + A['xnor2'])  # symbol counter (K or M terminal), FSM
    blocks.append((f"CP timing: sign line {N}b, ph, A/A_on {A_bits}b, FSM", tim_flops, tim_logic, 'clk'))

    # ---- I/O wrapper (clk) -----------------------------------------------------------------------------
    io_flops = 2 * W + 3 + 2 + 4 + 2           # capture pos/neg, SOF/VALID/DBG, cap_ok/emit_ok, SCLK/ADC_CLK ODDR pairs, CFG regs
    io_logic = mux(W) + A['inv'] + 2 * A['xor2'] + 10 * A['and2']
    blocks.append(("I/O: capture (CFG_EDGE), SOF/VALID/DBG, ODDR SCLK/ADC_CLK, cfg", io_flops, io_logic, 'clk'))
    blocks.append(("ICG dlclkp_1", 0, A['dlclkp'] / LOGIC_FACTOR, 'clk'))
    return blocks


def post_layout(blocks, N, hold_unc_D16=True, pl_margin=0.05):
    """Post-layout std-cell area projection from the hold-fix toy-design measurements."""
    n = N.bit_length() - 1
    flops = sum(b[1] for b in blocks)
    hops = sum((1 if s == 1 else 2) * W * ((N >> s) - 1) for s in range(1, n + 1)) + (N - 1)       # delay lines, sign line
    crossings = 2 * W                                                     # clk->clk_dp entry, clk_dp->clk exits (x2 buffers)
    short = flops - hops
    if hold_unc_D16:      # hold-only uncertainty 0.1 ns, margin 0.05: only the tree-offset crossings (toy counterfactual)
        hold = crossings * 10.01
    else:                 # default 0.25 ns: one dlygate per zero-logic hop + 25 % (0.05) / 67 % (0.10) of short endpoints
        hold = (hops + crossings) * 10.01 + short * (0.25 if pl_margin <= 0.05 else 0.67) * 10.01
    clock = 375 + 1.06 * flops                                            # toys: 479 @ 99 ff, 682 @ 291 ff
    taps = 570                                                            # 1x2 measured
    misc = 100                                                            # input buffers, ties, diodes
    return dict(flops=flops, hops=hops, hold=hold, clock=clock, taps=taps, misc=misc)


def report(N=32, twiddle='4mult'):
    blocks = estimate(N, twiddle)
    print(f"{'block':56s} {'flops':>5s} {'flop um2':>9s} {'logic raw':>9s} {'logic cal':>9s} {'total':>8s}")
    tot_ff = tot_raw = tot_cal = 0
    for name, ff, lg, dom in blocks:
        cal = lg * LOGIC_FACTOR
        tot_ff += ff; tot_raw += lg; tot_cal += cal
        print(f"{name:56s} {ff:5d} {ff * A['dff']:9.0f} {lg:9.0f} {cal:9.0f} {ff * A['dff'] + cal:8.0f}  [{dom}]")
    syn = tot_ff * A['dff'] + tot_cal
    print(f"{'TOTAL (Yosys-level, calibrated logic)':56s} {tot_ff:5d} {tot_ff * A['dff']:9.0f} {tot_raw:9.0f} {tot_cal:9.0f} {syn:8.0f}")
    core = 34255.4          # 1x2 core area (measured)
    for d16 in (True, False):
        p = post_layout(blocks, N, hold_unc_D16=d16)
        ll = syn * 1.025                                       # LibreLane synth vs Yosys (clock-gate toy design)
        total = ll + p['hold'] + p['clock'] + p['taps'] + p['misc']
        tag = "hold-only uncertainty 0.1 ns, margin 0.05" if d16 else "LibreLane default (0.25 ns), margin 0.05"
        print(f"post-layout projection, {tag}:")
        print(f"   synth {ll:7.0f} + hold-fix {p['hold']:6.0f} ({p['hops']} zero-logic hops) + clock tree {p['clock']:5.0f}"
              f" + taps {p['taps']} + misc {p['misc']} = {total:7.0f} um^2 = {100 * total / core:4.1f} % of the 1x2 core"
              f" (go/no-go limit 27,404 = 80 %)")
    return syn


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 32
    tw = 'gauss' if '--twiddle' in sys.argv and sys.argv[-1] == 'gauss' else '4mult'
    report(N, tw)
