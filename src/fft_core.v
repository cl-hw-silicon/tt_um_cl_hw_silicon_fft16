/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
// N-point real-input radix-2^2 SDF FFT, DIF, W-bit I/Q, Q1.7 twiddles, /2 per stage except the last, whose sum and
// difference saturate instead (SCALE = 0: net gain 2/N).
// Mirrors model/fft_model.SDF step(x, cnt): one real sample in, the combinational output of slot
// (cnt + 1) mod N out, every clk_dp edge. Output slot t carries bin bitrev(t); latency N-1.
//   stage s (1..n), D = N >> s: odd s BF2I, even s BF2II (-j when ~cnt[n-s+1] & cnt[n-s]);
//   after even s < n: twiddle W_M^e, M = 2^(n-s+2), e = (k1 + 2 k2) * cnt[n-s-1:0],
//   k1 = cnt[n-s+1] ^ cnt[n-s], k2 = ~cnt[n-s]          (schedule proven in fft_model.schedule_proof)
// N = 32: BF2I(16) BF2II(8) xW32 BF2I(4) BF2II(2) xW8 BF2I(1)
// N = 16: BF2I(8)  BF2II(4) xW16 BF2I(2) BF2II(1)         (as built)
// The frame counter is the low bits of the symbol-phase counter: `cnt` is an input.
// The clock port is clk_dp; the core does not know whether it is gated. Supported N: 16 and 32.
// (Explicit per-stage wiring rather than a generate loop over an unpacked array: Verilator 5.052's
// cycle breaker crashes on the chained array; the explicit form is also easier to read.)
module fft_core #(
    parameter int N  = 32,
    parameter int W  = 8,
    /* verilator lint_off UNUSEDPARAM */
    parameter int TW = 7          // fixed by the generated twiddle tables (Q1.7)
    /* verilator lint_on UNUSEDPARAM */
) (
    input  logic                     clk_dp,
    input  logic [$clog2(N)-1:0]     cnt,
    input  logic signed [W-1:0]      x,
    output logic signed [W-1:0]      yi,
    output logic signed [W-1:0]      yq,
    output logic [$clog2(N)-1:0]     slot
);
  logic signed [W-1:0] zero;
  assign zero = '0;

  // twiddle exponent (k1 + 2 k2) * m as shift-add (no `*`): kk = {k2, k1}
  function automatic logic [4:0] tw_exp(input logic [1:0] kk, input logic [4:0] m);
    tw_exp = (kk[0] ? m : 5'd0) + (kk[1] ? (m << 1) : 5'd0);
  endfunction

  if (N == 32) begin : g_n32
    logic signed [W-1:0] o1i, o1q, o2i, o2q, t2i, t2q, o3i, o3q, o4i, o4q, t4i, t4q;
    logic [4:0] e2, e4;
    // cnt bits b4..b0
    sdf_stage #(.W(W), .D(16), .NEGJ(0), .REAL(1)) u_s1 (
        .clk_dp(clk_dp), .s(cnt[4]), .tj(1'b0), .xi(x), .xq(zero), .yi(o1i), .yq(o1q));
    sdf_stage #(.W(W), .D(8), .NEGJ(1), .REAL(0)) u_s2 (
        .clk_dp(clk_dp), .s(cnt[3]), .tj(~cnt[4] & cnt[3]), .xi(o1i), .xq(o1q), .yi(o2i), .yq(o2q));
    assign e2 = tw_exp({~cnt[3], cnt[4] ^ cnt[3]}, {2'b00, cnt[2:0]});         // W32^e2, e2 <= 21
    twiddle_w32 u_tw2 (.e(e2), .xi(o2i), .xq(o2q), .yi(t2i), .yq(t2q));
    sdf_stage #(.W(W), .D(4), .NEGJ(0), .REAL(0)) u_s3 (
        .clk_dp(clk_dp), .s(cnt[2]), .tj(1'b0), .xi(t2i), .xq(t2q), .yi(o3i), .yq(o3q));
    sdf_stage #(.W(W), .D(2), .NEGJ(1), .REAL(0)) u_s4 (
        .clk_dp(clk_dp), .s(cnt[1]), .tj(~cnt[2] & cnt[1]), .xi(o3i), .xq(o3q), .yi(o4i), .yq(o4q));
    assign e4 = tw_exp({~cnt[1], cnt[2] ^ cnt[1]}, {4'b0000, cnt[0]});         // W8^e4, e4 <= 3
    twiddle_w8 u_tw4 (.e(e4[1:0]), .xi(o4i), .xq(o4q), .yi(t4i), .yq(t4q));
    sdf_stage #(.W(W), .D(1), .NEGJ(0), .REAL(0), .SCALE(0)) u_s5 (
        .clk_dp(clk_dp), .s(cnt[0]), .tj(1'b0), .xi(t4i), .xq(t4q), .yi(yi), .yq(yq));
    logic _unused_e;
    assign _unused_e = &{1'b0, e4[4:2]};
  end else begin : g_n16
    logic signed [W-1:0] o1i, o1q, o2i, o2q, t2i, t2q, o3i, o3q;
    logic [4:0] e2;
    // cnt bits b3..b0
    sdf_stage #(.W(W), .D(8), .NEGJ(0), .REAL(1)) u_s1 (
        .clk_dp(clk_dp), .s(cnt[3]), .tj(1'b0), .xi(x), .xq(zero), .yi(o1i), .yq(o1q));
    sdf_stage #(.W(W), .D(4), .NEGJ(1), .REAL(0)) u_s2 (
        .clk_dp(clk_dp), .s(cnt[2]), .tj(~cnt[3] & cnt[2]), .xi(o1i), .xq(o1q), .yi(o2i), .yq(o2q));
    assign e2 = tw_exp({~cnt[2], cnt[3] ^ cnt[2]}, {3'b000, cnt[1:0]});        // W16^e2, e2 <= 9
    twiddle_w16 u_tw2 (.e(e2[3:0]), .xi(o2i), .xq(o2q), .yi(t2i), .yq(t2q));
    sdf_stage #(.W(W), .D(2), .NEGJ(0), .REAL(0)) u_s3 (
        .clk_dp(clk_dp), .s(cnt[1]), .tj(1'b0), .xi(t2i), .xq(t2q), .yi(o3i), .yq(o3q));
    sdf_stage #(.W(W), .D(1), .NEGJ(1), .REAL(0), .SCALE(0)) u_s4 (
        .clk_dp(clk_dp), .s(cnt[0]), .tj(~cnt[1] & cnt[0]), .xi(o3i), .xq(o3q), .yi(yi), .yq(yq));
    logic _unused_e;
    assign _unused_e = &{1'b0, e2[4]};
  end

  assign slot = cnt + 1'b1;
endmodule

`default_nettype wire
