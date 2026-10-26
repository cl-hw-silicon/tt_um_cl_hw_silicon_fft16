/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
// CP timing recovery (model: model/timing.py, register for register).
// Sign-only CP correlator (N x 1-bit sign delay line, XNOR, +-1 accumulation), sequential acquisition
// with one accumulator over K symbols per hypothesis, early/late tracking with hysteresis, unlock after
// two consecutive weak_q on-time measurements, auto-pause of the datapath (`run`) once locked.
//   * ph = symbol-phase counter 0..L-1 (L = N + CP); the FFT window is ph in [0, N) (its low bits are the
//     FFT frame counter), the pause is ph in [N, L). The window starts CP/2 samples into the CP.
//   * correlation windows: ON [N - CP/2, N + CP/2), EARLY = ON - 1, LATE = ON + 1.
//   * one decision per symbol at the edge that ends ph == L-2 (inside the pause), using A including this
//     cycle's sample; a hypothesis advance / late slip repeats the L-2 cycle (stall), an early slip skips
//     the L-1 cycle: the datapath always gets N contiguous samples per symbol.
//   * `run` is a register and drives the ICG GATE directly (clk_gate.v).
//   * reset: rst_sync_n is the synchronised chip reset (two-flop synchroniser in the top level, docs/info.md
//     Reset). Unused state encodings (5..7) return to ACQ at the next edge with lock = 0 (so run = 1); ph and
//     symcnt are self-correcting (ph >= L-1 wraps to 0, symcnt >= K-1 ends the measurement).
// Constants: K = 4N/CP = 16, M = 16, T_lock = 3N = 48, T_unlock = (3N+1)/2 = 24 at N = 16 (as built),
// hysteresis 8. CFG_CP: 0 = CP N/4, 1 = CP N/8; ignored at N = 16 (CP fixed at N/4).
module cp_timing #(
    parameter int N = 32
) (
    input  logic                           clk,
    input  logic                           rst_sync_n,   // synchronised reset (top level), active low
    input  logic                           x_sgn,     // sign bit of the captured sample (1 = negative)
    input  logic                           cfg_cp,
    output logic [$clog2(N + N / 4)-1:0]   ph,        // symbol phase (register)
    output logic                           run,       // datapath advances at the edge ending this cycle
    output logic                           lock,      // TRACK state
    output logic                           slip,      // tracking slip decided at the edge ending this cycle
    output logic [$clog2(N + N / 4)-1:0]   ph_nx,     // next-state values (for the pin bookkeeping)
    output logic                           run_nx,
    output logic                           lock_nx
);
  localparam int PHB = $clog2(N + N / 4);
  localparam int AB = $clog2(4 * N) + 2;            // |A| <= K*CP = 4N
  localparam int SB = 5;                            // symbol counter: max(K) - 1 = 31
  localparam int CP0 = N / 4;
  localparam int CP1 = (N == 16) ? N / 4 : N / 8;  // N = 16: CP fixed at N/4
  localparam int K0 = 4 * N / CP0;
  localparam int K1 = 4 * N / CP1;
  localparam int M = 16;
  localparam int T_LOCK = 3 * N;                    // ceil(0.75   * K * CP)
  localparam int T_UNLOCK = (3 * N + 1) / 2;        // ceil(0.375  * K * CP)
  localparam int H = 8;

  localparam logic [2:0] ACQ = 3'd0, T_ON = 3'd1, T_EARLY = 3'd2, T_LATE = 3'd3, T_WAIT = 3'd4;

  // ---- configuration (static pin) ---------------------------------------------------------
  logic                 sel;
  logic [PHB-1:0]       lm1, lm2, on_lo, on_hi;
  logic [SB-1:0]        k_last;
  assign sel    = (N == 16) ? 1'b0 : cfg_cp;
  assign lm1    = sel ? PHB'(N + CP1 - 1) : PHB'(N + CP0 - 1);        // L - 1
  assign lm2    = sel ? PHB'(N + CP1 - 2) : PHB'(N + CP0 - 2);        // L - 2
  assign on_lo  = sel ? PHB'(N - CP1 / 2) : PHB'(N - CP0 / 2);
  assign on_hi  = sel ? PHB'(N + CP1 / 2) : PHB'(N + CP0 / 2);        // exclusive
  assign k_last = sel ? SB'(K1 - 1) : SB'(K0 - 1);

  // ---- registers ------------------------------------------------------------------------------
  logic [N-1:0]          sline;                  // sign delay line, no reset (gated by `primed`)
  logic                  stall, primed, weak_q;
  logic signed [AB-1:0]  acc, acc_on;
  logic [SB-1:0]         symcnt;
  logic [2:0]            state;

  // ---- correlator --------------------------------------------------------------------------------
  logic in_on, in_early, in_late, inwin, match;
  logic signed [AB-1:0] acc_nx0;
  assign in_on    = (ph >= on_lo) && (ph < on_hi);
  assign in_early = (ph >= on_lo - 1'b1) && (ph < on_hi - 1'b1);
  assign in_late  = (ph >= on_lo + 1'b1) && (ph < on_hi + 1'b1);
  always_comb begin
    case (state)
      ACQ, T_ON: inwin = in_on;
      T_EARLY:   inwin = in_early;
      T_LATE:    inwin = in_late;
      default:   inwin = 1'b0;
    endcase
    inwin = inwin & primed & ~stall;
  end
  assign match   = (x_sgn == sline[N-1]);
  assign acc_nx0 = inwin ? (match ? acc + AB'(1) : acc - AB'(1)) : acc;

  // ---- once-per-symbol decision and next state -------------------------------------------------------
  logic                  decide, last, gt_side, ge_lock, lt_unlock;
  logic [PHB-1:0]        ph_n;
  logic                  stall_n, weak_n, lock_n, slip_n;
  logic signed [AB-1:0]  acc_n, acc_on_n;
  logic [SB-1:0]         symcnt_n;
  logic [2:0]            state_n;
  logic signed [AB:0]    on_plus_h;
  assign decide    = (ph == lm2) && !stall;
  assign last      = !(symcnt < ((state == T_WAIT) ? SB'(M - 1) : k_last));
  assign on_plus_h = (AB + 1)'(acc_on) + (AB + 1)'(H);
  assign gt_side   = (AB + 1)'(acc_nx0) > on_plus_h;
  assign ge_lock   = acc_nx0 >= AB'(T_LOCK);
  assign lt_unlock = acc_nx0 < AB'(T_UNLOCK);

  always_comb begin
    ph_n     = (ph >= lm1) ? '0 : ph + 1'b1;
    stall_n  = 1'b0;
    state_n  = state;
    symcnt_n = symcnt;
    lock_n   = lock;
    acc_n    = acc_nx0;
    acc_on_n = acc_on;
    weak_n   = weak_q;
    slip_n   = 1'b0;
    if (state > T_WAIT) begin                    // unused encoding: back to acquisition at once
      state_n  = ACQ;
      lock_n   = 1'b0;
      weak_n   = 1'b0;
      symcnt_n = '0;
      acc_n    = '0;
    end else if (decide) begin
      if (!last) begin
        symcnt_n = symcnt + 1'b1;
      end else begin
        symcnt_n = '0;
        acc_n    = '0;
        case (state)
          ACQ: begin
            if (primed && ge_lock) begin
              lock_n  = 1'b1;
              state_n = T_ON;
            end else begin                       // next hypothesis: window one sample later
              ph_n    = lm2;
              stall_n = 1'b1;
            end
          end
          T_ON: begin
            acc_on_n = acc_nx0;
            if (lt_unlock && weak_q) begin
              lock_n  = 1'b0;
              state_n = ACQ;
              weak_n  = 1'b0;
            end else begin
              weak_n  = lt_unlock;
              state_n = T_EARLY;
            end
          end
          T_EARLY: begin
            if (gt_side) begin                   // p-1 better: skip one pause cycle
              ph_n    = '0;
              slip_n  = 1'b1;
              state_n = T_WAIT;
            end else begin
              state_n = T_LATE;
            end
          end
          T_LATE: begin
            if (gt_side) begin                   // p+1 better: repeat one pause cycle
              ph_n    = lm2;
              stall_n = 1'b1;
              slip_n  = 1'b1;
            end
            state_n = T_WAIT;
          end
          T_WAIT:  state_n = T_ON;               // T_WAIT done
          default: begin                         // unreachable (caught above); kept: default -> ACQ
            state_n = ACQ;
            lock_n  = 1'b0;
            weak_n  = 1'b0;
          end
        endcase
      end
    end
  end

  always_ff @(posedge clk) sline <= {sline[N-2:0], x_sgn};

  always_ff @(posedge clk) begin
    if (!rst_sync_n) begin
      ph     <= '0;
      stall  <= 1'b0;
      primed <= 1'b0;
      acc    <= '0;
      acc_on <= '0;
      symcnt <= '0;
      state  <= ACQ;
      weak_q   <= 1'b0;
      lock   <= 1'b0;
      run    <= 1'b1;
    end else begin
      ph     <= ph_n;
      stall  <= stall_n;
      primed <= primed | (ph >= lm1);
      acc    <= acc_n;
      acc_on <= acc_on_n;
      symcnt <= symcnt_n;
      state  <= state_n;
      weak_q   <= weak_n;
      lock   <= lock_n;
      run    <= ~lock_n | (ph_n < PHB'(N));
    end
  end

  assign slip    = slip_n & rst_sync_n;
  assign ph_nx   = rst_sync_n ? ph_n : '0;
  assign lock_nx = rst_sync_n ? lock_n : 1'b0;
  assign run_nx  = rst_sync_n ? (~lock_n | (ph_n < PHB'(N))) : 1'b1;
endmodule

`default_nettype wire
