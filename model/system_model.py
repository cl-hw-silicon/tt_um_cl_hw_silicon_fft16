# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""System model: captured ADC samples -> CP timing block -> (paused) SDF FFT -> Hermitian byte tail ->
bus byte + SOF / VALID / LOCK / DBG, register for register (pins and bus framing: docs/info.md).

Per clock (see timing.py for the step() convention):
  * timing block (clk): ph, run, LOCK, slip events.
  * datapath (clk_dp = clk gated by `run`): SDF.step(x, cnt = ph mod N) and the tail registers
        q_hold <= Q(slot)                        (slot = cnt + 1)
        o      <= slot even ? I(slot) : q_hold   (= uo_out)
    o after the datapath edge with cnt = c-1 holds byte c, so during the run cycle with cnt = c
    (ph = c when locked) the bus shows byte c of the frame captured in the previous symbol. A locked
    frame is visible in the N run cycles ph = 0..N-1 of the next symbol; during the CP pause the bus
    is don't-care (it already shows the next frame's byte 0) and VALID = 0.
    Byte c: c even -> I of slot c (bin bitrev(c)), c odd -> Q of slot c-1.
  * bookkeeping (clk), registered from the *next* state so the pins line up with the visible byte:
        SOF   <= run_next & (ph_next mod N == 0)          byte 0 of every frame, valid or not
        VALID <= run_next & emit_ok_next & LOCK_next
        DBG   <= slip                                      one-clock pulse per tracking slip
        cap_ok  : frame being captured is clean = LOCK at its start, cleared by a slip decided after
                  its capture (it is the frame in flight);
        emit_ok : cap_ok of the frame now being emitted.
        At the edge into ph = 0: emit_ok <= cap_ok (after slip clear), cap_ok <= LOCK(next).
step() returns the outputs *after* the edge, i.e. what the pins show during the next cycle, plus
the intermediates (timing record, datapath record).
Reset: rst_n passes a two-flop synchroniser (rst_sync stages, default 2) before it
reaches the timing block and the bookkeeping, so a level on rst_n sampled at edge k acts at edge k + 2. The
synchroniser powers up as if reset were held (the RTL flops are X until rst_n has been low for two clocks).
"""
from fft_model import SDF, log2i
from timing import CPTiming


class FFTChipModel:
    def __init__(self, N=32, W=8, TW=7, cfg_cp=0, rst_sync=2, prec=None, **timing_kw):
        """prec: SDF precision options (fft_model.SDF Wi / last_div2 / TWi; precision study); None = the RTL
        (last stage without /2, saturated, gain 2/N)."""
        self.N, self.W, self.n = N, W, log2i(N)
        self.rst_pipe = [0] * rst_sync                  # reset synchroniser (clk, no reset), [-1] = rst_sync_n
        self.tim = CPTiming(N=N, cfg_cp=cfg_cp, **timing_kw)
        self.sdf = SDF(N, W, TW, **(prec or {}))
        self.q_hold = self.o = None                    # clk_dp, no reset (2-register tail)
        self.sof = self.valid = self.dbg = 0            # clk, reset
        self.cap_ok = self.emit_ok = 0

    def step(self, x, rst_n=1):
        N = self.N
        if self.rst_pipe:                               # the control logic sees the synchronised reset
            rst_n, self.rst_pipe = self.rst_pipe[-1], [rst_n] + self.rst_pipe[:-1]
        r = self.tim.step(x, rst_n)
        dp = None
        if r['run']:                                    # datapath edge (ICG passes this clock)
            cnt = r['ph'] & (N - 1)
            I, Q = self.sdf.step(x, cnt)
            slot = (cnt + 1) % N
            o_new = I if slot % 2 == 0 else self.q_hold
            dp = dict(cnt=cnt, slot=slot, I=I, Q=Q, x=x, byte_index=slot)
            self.o, self.q_hold = o_new, Q
        cap_ok_n = 0 if r['slip'] else self.cap_ok
        emit_ok_n = self.emit_ok
        if r['ph_next'] == 0 and rst_n:                 # edge into a new symbol / new capture
            emit_ok_n = cap_ok_n
            cap_ok_n = r['lock_next']
        sof_n = 1 if (r['run_next'] and (r['ph_next'] & (N - 1)) == 0) else 0
        valid_n = 1 if (r['run_next'] and emit_ok_n and r['lock_next']) else 0
        if not rst_n:
            self.sof = self.valid = self.dbg = self.cap_ok = self.emit_ok = 0
        else:
            self.sof, self.valid, self.dbg = sof_n, valid_n, r['slip']
            self.cap_ok, self.emit_ok = cap_ok_n, emit_ok_n
        return dict(uo=self.o, sof=self.sof, valid=self.valid, lock=self.tim.lock, dbg=self.dbg,
                    tim=r, dp=dp)


def to_byte(v):
    """Signed W=8 value -> pin byte."""
    return None if v is None else v & 0xFF
