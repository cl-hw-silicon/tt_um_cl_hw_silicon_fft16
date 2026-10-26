# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cycle-exact model of the CP timing-recovery block (src/cp_timing.v mirrors it).

Algorithm and constants as in the chip: sign-only CP correlator (N x 1-bit sign delay line, XNOR, +-1 up/down count),
sequential acquisition with one accumulator over K symbols per hypothesis, early/late tracking with
hysteresis, unlock after two consecutive weak on-time measurements, auto-pause of the datapath
during the CP once locked. Parameters (K, M, thresholds, hysteresis) are constructor arguments.

Implementation choices, all in the ungated clock domain:
* One symbol-phase counter ph = 0..L-1 (L = N + CP) is the only notion of time. It is laid out so
  that the FFT window (the run region) is exactly ph in [0, N): the datapath frame counter is ph's
  low n bits (no separate counter, no re-phasing logic). The pause is ph in [N, L).
* The FFT window starts delta = CP/2 samples inside the CP (it holds one symbol, cyclically
  rotated by delta), so a timing error e in [-delta, CP - delta] still gives a clean window.
* Correlation windows (CP samples each, compared with the sign N samples earlier):
      ON    ph in [a, a+CP)          a = N - CP + delta   (aligned: the last CP samples of the symbol)
      EARLY ph in [a-1, a+CP-1)      (hypothesis p-1)
      LATE  ph in [a+1, a+CP+1)      (hypothesis p+1)
* Once per symbol, at the edge that ends the cycle ph == L-2 (inside the pause), the FSM decides.
  A hypothesis advance / late slip repeats the ph = L-2 cycle (stall); an early slip skips the
  ph = L-1 cycle. Both cycles are pause cycles, so the datapath always sees exactly N contiguous
  samples per symbol and a slip never cuts a frame.
* Decisions use A_next (the accumulator including this cycle's sample) because for CP = N/8 the
  LATE window ends exactly at ph = L-2.

Clock convention (identical for every model in this directory): step(x) is one rising clock edge.
`x` is the captured sample present during the cycle that ends at this edge. The returned record
holds the register values *during* that cycle (ph, run, lock, state, ...) plus the events decided
at this edge (slip, lock_rise, unlock, hyp_adv) and the values the registers take after the edge
(*_next). `run` during a cycle = the datapath advances at the edge that ends it (it drives the ICG
GATE directly from a flop).
"""

ACQ, T_ON, T_EARLY, T_LATE, T_WAIT = range(5)
STATE_NAMES = ['ACQ', 'T_ON', 'T_EARLY', 'T_LATE', 'T_WAIT']


def frac_threshold(frac, K, CP):
    """Threshold in accumulator units for a fraction of the noise-free on-time maximum K*CP.
    RTL constant (one per CFG_CP setting): ceil(frac * K * CP)."""
    import math
    return int(math.ceil(frac * K * CP - 1e-9))


class CPTiming:
    def __init__(self, N=32, cfg_cp=0, K=None, M=16, lock_frac=0.75, unlock_frac=0.375, hyst=8):
        """K: symbols per measurement; int, or dict {cfg_cp: K}, or None = 4N/CP (K*CP = 4N, i.e.
        16 for CP = N/4 and 32 for CP = N/8 at N = 32: equal statistics and equal thresholds in both
        CFG_CP settings). Defaults = the chip's parameters (from study_timing.py): K = 4N/CP,
        M = 16, T_lock = 0.75*K*CP, T_unlock = 0.375*K*CP, hysteresis 8 (absolute; K*CP is constant,
        so the metric noise and all thresholds are the same in both CFG_CP settings).
        hyst: None = K // 2 (study only)."""
        self.N = N
        if N == 16:
            cfg_cp = 0                  # at N = 16 CFG_CP is removed, CP fixed at N/4 (as cp_timing.v)
        self.cfg_cp = cfg_cp
        self.CP = N // 4 if cfg_cp == 0 else N // 8
        self.L = N + self.CP
        self.delta = self.CP // 2
        assert 2 <= self.delta <= self.CP - 2, "need ph = L-2 and L-1 inside the pause and the windows before L-2"
        self.a = N - self.CP + self.delta
        if K is None:
            K = 4 * N // self.CP
        elif isinstance(K, dict):
            K = K[cfg_cp]
        self.K, self.M = K, M
        self.T_lock = frac_threshold(lock_frac, K, self.CP)
        self.T_unlock = frac_threshold(unlock_frac, K, self.CP)
        self.H = max(1, K // 2) if hyst is None else hyst
        self.params = dict(N=N, CP=self.CP, L=self.L, delta=self.delta, K=K, M=M,
                           T_lock=self.T_lock, T_unlock=self.T_unlock, H=self.H,
                           lock_frac=lock_frac, unlock_frac=unlock_frac)
        # accumulator range: |A| <= K*CP  -> width for the RTL
        self.A_bits = (K * self.CP).bit_length() + 1
        self.sline = [None] * N            # sign delay line, no reset (X in RTL); [0] newest, [-1] = N ago
        self.reset()

    # -- reset values of the control registers (rst_n, synchronous) ------------------------------
    def reset(self):
        self.ph = 0
        self.stall = 0       # 1 during the repeated ph = L-2 cycle after a hypothesis advance / late slip
        self.primed = 0      # sign delay line has been filled (set at the first wrap of ph)
        self.A = 0
        self.A_on = 0
        self.symcnt = 0
        self.state = ACQ
        self.weak = 0
        self.lock = 0
        self.run = 1

    def window(self, state, ph):
        a, CP = self.a, self.CP
        if state in (ACQ, T_ON):
            return a <= ph < a + CP
        if state == T_EARLY:
            return a - 1 <= ph < a + CP - 1
        if state == T_LATE:
            return a + 1 <= ph < a + CP + 1
        return False

    def step(self, x, rst_n=1):
        N, L = self.N, self.L
        sgn = 1 if x < 0 else 0                     # bit W-1 of the two's-complement sample
        tail = self.sline[-1]
        rec = dict(ph=self.ph, run=self.run, lock=self.lock, state=self.state, stall=self.stall,
                   A=self.A, A_on=self.A_on, symcnt=self.symcnt, sgn=sgn,
                   slip=0, slip_dir=0, lock_rise=0, unlock=0, hyp_adv=0, decide=0, sym_end=0)
        # ---- correlator ------------------------------------------------------------------------
        inwin = self.primed and not self.stall and self.window(self.state, self.ph)
        if inwin:
            assert tail is not None
            m = 1 if sgn == tail else -1
        else:
            m = 0
        A_next = self.A + m
        rec['m'], rec['inwin'] = m, int(inwin)
        # ---- defaults for the next state -----------------------------------------------------------
        ph_next = 0 if self.ph >= L - 1 else self.ph + 1
        stall_next = 0
        state_next, symcnt_next, lock_next = self.state, self.symcnt, self.lock
        A_on_next, weak_next = self.A_on, self.weak
        # ---- unused state encodings (5..7, unreachable; as the RTL's default branch): back to acquisition at once,
        # lock 0 (so run 1), fresh measurement; ph keeps counting (any ph >= L-1 wraps to 0 by itself)
        if self.state not in (ACQ, T_ON, T_EARLY, T_LATE, T_WAIT):
            state_next, lock_next, weak_next, symcnt_next, A_next = ACQ, 0, 0, 0, 0
            rec['illegal'] = 1
        # ---- once-per-symbol decision at the end of the ph = L-2 cycle ----------------------------
        elif self.ph == L - 2 and not self.stall:
            rec['decide'] = 1
            n_sym = self.M if self.state == T_WAIT else self.K
            if self.symcnt < n_sym - 1:
                symcnt_next = self.symcnt + 1
            else:
                symcnt_next = 0
                A_next_after = 0
                if self.state == ACQ:
                    if self.primed and A_next >= self.T_lock:
                        lock_next, state_next = 1, T_ON
                        rec['lock_rise'] = 1
                    else:                                       # next hypothesis: window 1 sample later
                        ph_next, stall_next = L - 2, 1
                        rec['hyp_adv'] = 1
                elif self.state == T_ON:
                    A_on_next = A_next
                    w = 1 if A_next < self.T_unlock else 0
                    if w and self.weak:
                        lock_next, state_next, weak_next = 0, ACQ, 0
                        rec['unlock'] = 1
                    else:
                        weak_next, state_next = w, T_EARLY
                elif self.state == T_EARLY:
                    if A_next > self.A_on + self.H:            # p-1 is better: skip one pause cycle
                        ph_next = 0
                        rec['slip'], rec['slip_dir'] = 1, -1
                        state_next = T_WAIT if self.M > 0 else T_ON
                    else:
                        state_next = T_LATE
                elif self.state == T_LATE:
                    if A_next > self.A_on + self.H:            # p+1 is better: repeat one pause cycle
                        ph_next, stall_next = L - 2, 1
                        rec['slip'], rec['slip_dir'] = 1, +1
                    state_next = T_WAIT if self.M > 0 else T_ON
                elif self.state == T_WAIT:                      # T_WAIT done
                    state_next = T_ON
                A_next = A_next_after
        # ---- register update ------------------------------------------------------------------------
        if ph_next == 0:
            rec['sym_end'] = 1
        primed_next = 1 if (self.primed or self.ph >= L - 1) else 0
        run_next = 1 if (not lock_next or ph_next < N) else 0
        self.sline = [sgn] + self.sline[:-1]                   # sign delay line: every clock, no reset
        if not rst_n:
            self.reset()
            self.primed = 0
        else:
            self.ph, self.stall, self.primed = ph_next, stall_next, primed_next
            self.A, self.A_on, self.symcnt, self.state = A_next, A_on_next, symcnt_next, state_next
            self.weak, self.lock, self.run = weak_next, lock_next, run_next
        rec.update(ph_next=self.ph, run_next=self.run, lock_next=self.lock)
        return rec
