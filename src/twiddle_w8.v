/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
// W8 twiddle for the second radix-2^2 boundary at N = 32 (model: fft_model.twiddle with W32 exponents
// E = 4e). W8^e, e = 0..3, Q1.7: (127/bypass), (91, -91), (0, -128), (-91, -91):
//   e = 0: bypass                          e = 1: re = 91(I+Q) >>> 7,  im = 91(Q-I) >>> 7
//   e = 2: re = Q, im = -I (sat)  (= 128-scaled exact -j)
//   e = 3: re = 91(Q-I) >>> 7,  im = -(91(I+Q)) >>> 7
// x91 as CSD shift-add (64 + 32 - 4 - 1): exact products, so floor(>>> 7) equals the model bit for bit.
module twiddle_w8 (
    input  logic        [1:0] e,
    input  logic signed [7:0] xi,
    input  logic signed [7:0] xq,
    output logic signed [7:0] yi,
    output logic signed [7:0] yq
);
  logic signed [8:0]  spq, dqi;
  logic signed [15:0] spq16, dqi16, s91, d91, ns91;
  assign spq   = {xi[7], xi} + {xq[7], xq};          // I + Q
  assign dqi   = {xq[7], xq} - {xi[7], xi};          // Q - I
  assign spq16 = 16'(spq);
  assign dqi16 = 16'(dqi);
  assign s91   = (spq16 <<< 6) + (spq16 <<< 5) - (spq16 <<< 2) - spq16;
  assign d91   = (dqi16 <<< 6) + (dqi16 <<< 5) - (dqi16 <<< 2) - dqi16;
  assign ns91  = 16'sd0 - s91;                         // -(91(I+Q)), negated before the floor shift

  function automatic logic signed [7:0] sat8(input logic signed [15:0] v);
    logic signed [15:0] sh;
    sh = v >>> 7;
    if (sh > 16'sd127) sat8 = 8'sd127;
    else if (sh < -16'sd128) sat8 = -8'sd128;
    else sat8 = sh[7:0];
  endfunction

  logic signed [7:0] negi;
  assign negi = (xi == -8'sd128) ? 8'sd127 : -xi;

  always_comb begin
    case (e)
      2'd1:    begin yi = sat8(s91); yq = sat8(d91);  end
      2'd2:    begin yi = xq;        yq = negi;       end
      2'd3:    begin yi = sat8(d91); yq = sat8(ns91); end
      default: begin yi = xi;        yq = xq;         end
    endcase
  end
endmodule

`default_nettype wire
