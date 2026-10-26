/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
// Hermitian byte tail (model: system_model.FFTChipModel): two plain registers on clk_dp.
//   q_hold <= Q(slot)                        every datapath clock
//   o      <= slot even ? I(slot) : q_hold   -> during the clock with cnt = c the bus shows byte c:
//                                               c even = I of slot c (bin bitrev(c)), c odd = Q of slot c-1
// Even slots carry exactly the bins k < N/2, so N bytes per frame carry the whole real-input spectrum.
module iq_tail #(
    parameter int W = 8
) (
    input  logic                clk_dp,
    input  logic                slot_lsb,
    input  logic signed [W-1:0] yi,
    input  logic signed [W-1:0] yq,
    output logic        [W-1:0] o
);
  logic [W-1:0] q_hold;
  always_ff @(posedge clk_dp) begin
    q_hold <= yq;
    o      <= slot_lsb ? q_hold : yi;
  end
endmodule

`default_nettype wire
