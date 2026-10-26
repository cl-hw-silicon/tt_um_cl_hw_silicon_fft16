/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
// tt_um_cl_hw_silicon_fft16 — streaming real-input FFT with on-chip CP timing recovery (pins: docs/info.md).
// N = 16 (CP fixed at N/4 = 4, uio[6] unused). Register for register with
// model/system_model.FFTChipModel (verified through the pins: test/test_top.py, test/test_ofdm.py):
//   ui_in -> ADC capture (CFG_EDGE: posedge, or negedge then posedge; offset binary -> two's complement)
//         -> cp_timing (clk) -> ICG (`run` -> clk_dp) -> fft_core -> iq_tail (2 registers) -> uo_out;
//   SOF / VALID / DBG registered from the timing block's next state, LOCK = the timing register;
//   SCLK / ADC_CLK = ODDR-style toggle pairs: high after every rising edge, low after every
//   falling edge, so data launched at the rising edge is sampled by the reader on the SCLK falling edge.
// Pipeline (posedge capture): the last sample of an FFT window captured at edge E -> its frame's SOF and byte 0
// launched at edge E + CP + 1 = E + 5; the first sample -> E + L = E + 20 (measured, test_bin_and_byte_order).
// Reset: rst_n passes a two-flop synchroniser (rst_sync_n) that drives every reset in the
// design, so a level on rst_n sampled at rising edge k acts on the control registers at edge k + 2 (two clocks of
// assertion and release latency). While reset: correlator in ACQ, LOCK/VALID/SOF/DBG = 0, SCLK/ADC_CLK idle low,
// datapath clock running (run = 1). Not reset: delay lines, sign line, capture and CFG_EDGE registers, the
// negedge flops (the SCLK/ADC_CLK followers copy the reset toggle flops within half a cycle).
module tt_um_cl_hw_silicon_fft16 (
    input  wire [7:0] ui_in,     // ADC D[7:0], offset binary
    output wire [7:0] uo_out,    // bus IO[7:0]: I on even slots, Q on odd slots
    input  wire [7:0] uio_in,    // [5] CFG_EDGE; [6] unused (no CP-length setting at N = 16)
    output wire [7:0] uio_out,   // [0] SCLK, [1] SOF, [2] VALID, [3] ADC_CLK, [4] LOCK, [7] DBG
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,       // 32 MHz = f_s
    input  wire       rst_n      // active low, control registers only (synchronous, via rst_sync_n)
);
  localparam int N = 16;
  localparam int NB = $clog2(N);
  localparam int PHB = $clog2(N + N / 4);

  // ---- reset synchroniser: two plain flops on clk; rst_sync_n drives every reset below ----------------
  logic rst_meta_n, rst_sync_n;
  always_ff @(posedge clk) begin
    rst_meta_n <= rst_n;
    rst_sync_n <= rst_meta_n;
  end

  // ---- static configuration pin, registered (set it before / during reset; a change takes one clock) ----
  logic cfg_edge;
  always_ff @(posedge clk) cfg_edge <= uio_in[5];

  // ---- ADC capture: posedge (default) or negedge then posedge; offset binary -> two's complement
  logic        [7:0] d_neg;
  logic signed [7:0] cap;
  always_ff @(negedge clk) d_neg <= ui_in;
  always_ff @(posedge clk) cap <= cfg_edge ? {~d_neg[7], d_neg[6:0]} : {~ui_in[7], ui_in[6:0]};

  // ---- CP timing recovery (clk) ---------------------------------------------------------
  logic [PHB-1:0] ph, ph_nx;
  logic           run, lock, slip, run_nx, lock_nx;
  cp_timing #(.N(N)) u_tim (
      .clk(clk), .rst_sync_n(rst_sync_n), .x_sgn(cap[7]), .cfg_cp(1'b0), .ph(ph), .run(run), .lock(lock),
      .slip(slip), .ph_nx(ph_nx), .run_nx(run_nx), .lock_nx(lock_nx)
  );

  // ---- datapath on the gated clock (the only clock gate) ---------------------------------
  logic clk_dp;
  clk_gate u_cg (.clk(clk), .en(run), .gclk(clk_dp));

  logic signed [7:0] yi, yq;
  logic [NB-1:0]     slot;
  fft_core #(.N(N)) u_core (.clk_dp(clk_dp), .cnt(ph[NB-1:0]), .x(cap), .yi(yi), .yq(yq), .slot(slot));

  logic [7:0] o;
  iq_tail u_tail (.clk_dp(clk_dp), .slot_lsb(slot[0]), .yi(yi), .yq(yq), .o(o));

  // ---- frame bookkeeping (clk): pins registered from the next state ----------------------
  // cap_ok: the frame being captured started locked and no slip has hit it; emit_ok: the same for the
  // frame now on the bus. At the edge into ph = 0 the captured frame becomes the emitted one.
  logic sof, valid, dbg, cap_ok, emit_ok;
  logic cap_ok_s, cap_ok_n, emit_ok_n;
  always_comb begin
    cap_ok_s  = cap_ok & ~slip;
    cap_ok_n  = cap_ok_s;
    emit_ok_n = emit_ok;
    if (ph_nx == '0) begin
      emit_ok_n = cap_ok_s;
      cap_ok_n  = lock_nx;
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_sync_n) begin
      sof     <= 1'b0;
      valid   <= 1'b0;
      dbg     <= 1'b0;
      cap_ok  <= 1'b0;
      emit_ok <= 1'b0;
    end else begin
      sof     <= run_nx & (ph_nx[NB-1:0] == '0);
      valid   <= run_nx & emit_ok_n & lock_nx;
      dbg     <= slip;                              // one pulse per slip
      cap_ok  <= cap_ok_n;
      emit_ok <= emit_ok_n;
    end
  end

  // ---- SCLK / ADC_CLK: toggle on posedge, follow on negedge, XOR (rises after posedge, falls after
  // negedge, like the data launch). `b <= a` instead of `b <= ~b` keeps the phase right whatever
  // half-cycle reset is released in.
  logic sclk_a, sclk_b, aclk_a, aclk_b;
  always_ff @(posedge clk) begin
    if (!rst_sync_n) begin
      sclk_a <= 1'b0;
      aclk_a <= 1'b0;
    end else begin
      sclk_a <= ~sclk_a;
      aclk_a <= ~aclk_a;
    end
  end
  always_ff @(negedge clk) begin
    sclk_b <= sclk_a;
    aclk_b <= aclk_a;
  end

  assign uo_out  = o;
  assign uio_out = {dbg, 2'b00, lock, aclk_a ^ aclk_b, valid, sof, sclk_a ^ sclk_b};
  assign uio_oe  = 8'b1001_1111;

  wire _unused = &{ena, uio_in[4:0], uio_in[7:6], ph[PHB-1:NB], slot[NB-1:1], 1'b0};
endmodule
`default_nettype wire
