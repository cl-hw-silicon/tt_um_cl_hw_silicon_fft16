/*
 * Copyright (c) 2026 Cable Television Laboratories, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none
`timescale 1ns / 1ps
module tb_timing #(
    parameter int N = 32
) (
    input  wire clk,
    input  wire rst_n,
    input  wire x_sgn,
    input  wire cfg_cp,
    output wire [$clog2(N + N / 4)-1:0] ph,
    output wire run,
    output wire lock,
    output wire slip,
    output wire [$clog2(N + N / 4)-1:0] ph_nx,
    output wire run_nx,
    output wire lock_nx
);
  cp_timing #(.N(N)) u_tim (.clk(clk), .rst_sync_n(rst_n), .x_sgn(x_sgn), .cfg_cp(cfg_cp), .ph(ph), .run(run),
                            .lock(lock), .slip(slip), .ph_nx(ph_nx), .run_nx(run_nx), .lock_nx(lock_nx));
endmodule
`default_nettype wire
