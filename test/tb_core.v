/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
`timescale 1ns / 1ps
// Unit bench: clock gate (real PDK cell model) + FFT core + Hermitian tail, as wired in the chip:
// clk -> clk_gate(en = run) -> clk_dp. The testbench drives run, cnt and x at the falling edge.
module tb_core #(
    parameter int N = 32
) (
    input  wire                  clk,
    input  wire                  run,
    input  wire [$clog2(N)-1:0]  cnt,
    input  wire signed [7:0]     x,
    output wire signed [7:0]     yi,
    output wire signed [7:0]     yq,
    output wire [$clog2(N)-1:0]  slot,
    output wire [7:0]            o
);
  wire clk_dp;
  clk_gate u_cg (.clk(clk), .en(run), .gclk(clk_dp));
  fft_core #(.N(N)) u_core (.clk_dp(clk_dp), .cnt(cnt), .x(x), .yi(yi), .yq(yq), .slot(slot));
  iq_tail u_tail (.clk_dp(clk_dp), .slot_lsb(slot[0]), .yi(yi), .yq(yq), .o(o));
endmodule
`default_nettype wire
