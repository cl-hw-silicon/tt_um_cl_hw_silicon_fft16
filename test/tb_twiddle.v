/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
`timescale 1ns / 1ps
// Exhaustive twiddle bench: 256 parallel instances of each twiddle module, instance q has xq = q - 128;
// the test drives xi and e and reads all 256 results at once (packed), so 65,536 x e vectors take
// 256 steps per exponent.
module tb_twiddle (
    input  wire signed [7:0] xi,
    input  wire [4:0]        e32,
    input  wire [3:0]        e16,
    input  wire [1:0]        e8,
    output wire [256*16-1:0] y32,
    output wire [256*16-1:0] y16,
    output wire [256*16-1:0] y8
);
  for (genvar q = 0; q < 256; q++) begin : g_q
    localparam logic signed [7:0] XQ = 8'(q - 128);
    twiddle_w32 u32 (.e(e32), .xi(xi), .xq(XQ), .yi(y32[q*16+8+:8]), .yq(y32[q*16+:8]));
    twiddle_w16 u16 (.e(e16), .xi(xi), .xq(XQ), .yi(y16[q*16+8+:8]), .yq(y16[q*16+:8]));
    twiddle_w8  u8  (.e(e8),  .xi(xi), .xq(XQ), .yi(y8[q*16+8+:8]),  .yq(y8[q*16+:8]));
  end
endmodule
`default_nettype wire
