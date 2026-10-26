/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
// Generated (variant mult4) from model/fft_model.tw_table: keep it bit-exact with the model.
// W16 twiddle: (I + jQ) * W16^e, Q1.7 coefficients, exact products, floor(>>> 7), saturation to 8 bits,
// e == 0 exact bypass. Bit-exact with fft_model.twiddle (exhaustively tested in test/test_twiddle.py).
module twiddle_w16 (
    input  logic        [3:0] e,
    input  logic signed [7:0] xi,
    input  logic signed [7:0] xq,
    output logic signed [7:0] yi,
    output logic signed [7:0] yq
);
  // coefficient ROM (generated from model/fft_model.tw_table)
  logic signed [7:0] c, s;
  always_comb begin
    case (e)
      4'd0 : begin c =  8'sd127; s =  8'sd0; end
      4'd1 : begin c =  8'sd118; s = -8'sd49; end
      4'd2 : begin c =  8'sd91; s = -8'sd91; end
      4'd3 : begin c =  8'sd49; s = -8'sd118; end
      4'd4 : begin c =  8'sd0; s = -8'sd128; end
      4'd5 : begin c = -8'sd49; s = -8'sd118; end
      4'd6 : begin c = -8'sd91; s = -8'sd91; end
      4'd7 : begin c = -8'sd118; s = -8'sd49; end
      4'd8 : begin c = -8'sd128; s =  8'sd0; end
      4'd9 : begin c = -8'sd118; s =  8'sd49; end
      4'd10: begin c = -8'sd91; s =  8'sd91; end
      4'd11: begin c = -8'sd49; s =  8'sd118; end
      4'd12: begin c =  8'sd0; s =  8'sd127; end
      4'd13: begin c =  8'sd49; s =  8'sd118; end
      4'd14: begin c =  8'sd91; s =  8'sd91; end
      4'd15: begin c =  8'sd118; s =  8'sd49; end
      default: begin c = 8'sd0; s = 8'sd0; end
    endcase
  end

  // four real products, exact: re = c*I - s*Q, im = s*I + c*Q
  logic signed [15:0] p_ic, p_qs, p_is, p_qc;
  logic signed [16:0] re_f0, im_f0;
  logic signed [17:0] re_f, im_f;
  assign p_ic  = 16'(xi) * 16'(c);
  assign p_qs  = 16'(xq) * 16'(s);
  assign p_is  = 16'(xi) * 16'(s);
  assign p_qc  = 16'(xq) * 16'(c);
  assign re_f0 = 17'(p_ic) - 17'(p_qs);
  assign im_f0 = 17'(p_is) + 17'(p_qc);
  assign re_f  = 18'(re_f0);
  assign im_f  = 18'(im_f0);

  // >>> 7 then saturate: |product sum| <= 2*128*128 -> 10 significant bits after the shift
  logic signed [10:0] re_s, im_s;
  logic signed [7:0]  re_8, im_8;
  assign re_s = 11'(re_f >>> 7);
  assign im_s = 11'(im_f >>> 7);
  assign re_8 = (re_s > 11'sd127) ? 8'sd127 : (re_s < -11'sd128) ? -8'sd128 : re_s[7:0];
  assign im_8 = (im_s > 11'sd127) ? 8'sd127 : (im_s < -11'sd128) ? -8'sd128 : im_s[7:0];

  // exponent 0: exact bypass (no 127/128 gain)
  assign yi = (e == 4'd0) ? xi : re_8;
  assign yq = (e == 4'd0) ? xq : im_8;
endmodule

`default_nettype wire
