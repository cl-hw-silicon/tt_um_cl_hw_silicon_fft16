/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
// One radix-2^2 single-path delay-feedback stage, DIF (model: fft_model.SDF._stage).
//   s = 0: out = tail,                 push x
//   s = 1: SCALE = 1: out = (tail + x) >>> 1,   push (tail - x) >>> 1   (floor, W+1-bit sum, never overflows)
//          SCALE = 0: out = sat(tail + x),      push sat(tail - x)      (no shift, saturated to W bits; the
//                     last stage: pipeline gain 2/N; model fft_model.bf_add_sat / bf_sub_sat)
//   NEGJ = 1 (BF2II): the incoming element is first multiplied by -j when tj: (I + jQ) -> (Q - jI),
//                     -(-2^(W-1)) saturates to 2^(W-1)-1
//   REAL = 1 (stage 1 of the real-input FFT): no Q path (xq ignored, yq = 0)
// Delay line: D-word concatenation shift register, no reset, no enable (the datapath clock clk_dp is
// gated by the CP pause; this module does not know).
module sdf_stage #(
    parameter int W    = 8,
    parameter int D    = 16,
    parameter int NEGJ = 0,
    parameter int REAL = 0,
    parameter int SCALE = 1
) (
    input  logic                clk_dp,
    input  logic                s,
    input  logic                tj,
    input  logic signed [W-1:0] xi,
    input  logic signed [W-1:0] xq,
    output logic signed [W-1:0] yi,
    output logic signed [W-1:0] yq
);
  localparam logic signed [W-1:0] MINV = {1'b1, {(W - 1) {1'b0}}};
  localparam logic signed [W-1:0] MAXV = {1'b0, {(W - 1) {1'b1}}};

  // -j on the incoming element (BF2II only)
  logic signed [W-1:0] ni, ai, aq;
  assign ni = (xi == MINV) ? MAXV : -xi;
  assign ai = (NEGJ != 0 && tj) ? xq : xi;
  assign aq = (NEGJ != 0 && tj) ? ni : xq;

  // butterfly output from the W+1-bit sum / difference: SCALE = 1 -> >>> 1 (floor) = the top W bits;
  // SCALE = 0 -> saturate to W bits: on overflow (the two top bits differ) the rail with the sum's sign,
  // MINV or MAXV, built from the sign bit (this form maps ~0.7 k um^2 smaller under AREA 3 than a MINV/MAXV mux)
  function automatic logic signed [W-1:0] bf_out(input logic signed [W:0] v);
    logic ovf;
    ovf = v[W] ^ v[W-1];
    if (SCALE != 0) bf_out = v[W:1];
    else bf_out = ovf ? {v[W], {(W - 1) {~v[W]}}} : v[W-1:0];
  endfunction

  // ---- I path ----------------------------------------------------------------------------
  logic        [W*D-1:0] di;
  logic signed [W-1:0]   ti, pi;
  logic signed [W:0]     sum_i, dif_i;
  assign ti    = di[W*D-1-:W];
  assign sum_i = {ti[W-1], ti} + {ai[W-1], ai};
  assign dif_i = {ti[W-1], ti} - {ai[W-1], ai};
  assign yi    = s ? bf_out(sum_i) : ti;
  assign pi    = s ? bf_out(dif_i) : ai;

  if (D == 1) begin : g_di1
    always_ff @(posedge clk_dp) di <= pi;
  end else begin : g_din
    always_ff @(posedge clk_dp) di <= {di[W*(D-1)-1:0], pi};
  end

  // ---- Q path (absent in the real-only stage) -------------------------------------------------
  if (REAL != 0) begin : g_real
    assign yq = '0;
    logic _unused_q;
    assign _unused_q = &{1'b0, aq};
  end else begin : g_cplx
    logic        [W*D-1:0] dq;
    logic signed [W-1:0]   tq, pq;
    logic signed [W:0]     sum_q, dif_q;
    assign tq    = dq[W*D-1-:W];
    assign sum_q = {tq[W-1], tq} + {aq[W-1], aq};
    assign dif_q = {tq[W-1], tq} - {aq[W-1], aq};
    assign yq    = s ? bf_out(sum_q) : tq;
    assign pq    = s ? bf_out(dif_q) : aq;
    if (D == 1) begin : g_dq1
      always_ff @(posedge clk_dp) dq <= pq;
    end else begin : g_dqn
      always_ff @(posedge clk_dp) dq <= {dq[W*(D-1)-1:0], pq};
    end
  end
endmodule

`default_nettype wire
