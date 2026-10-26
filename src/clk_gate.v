/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
// The design's only clock gate: sky130 latch-based integrated clock gate on the datapath
// clock. GATE must come straight from a clk-domain flop (cp_timing.run). Simulation and synthesis use
// the PDK cell (functional model in sim, liberty cell in synthesis); Verilator lint needs a stub (UDPs).
module clk_gate (
    input  logic clk,
    input  logic en,
    output logic gclk
);
  sky130_fd_sc_hd__dlclkp_1 u_icg (
      .CLK (clk),
      .GATE(en),
      .GCLK(gclk)
  );
endmodule

`default_nettype wire
