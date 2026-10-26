# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive twiddle test: every exponent value of W32, W16 and W8 x all 65,536 (I, Q) inputs,
bit-exact against fft_model.twiddle (W8 = W32 table at E = 4e, as in the model's control)."""
import cocotb
from cocotb.triggers import Timer
import fft_model as fm

T32, T16 = fm.tw_table(32), fm.tw_table(16)


def unpack(v):
    v = int(v)
    out = []
    for q in range(256):
        w = (v >> (16 * q)) & 0xFFFF
        yi, yq = (w >> 8) & 0xFF, w & 0xFF
        out.append((yi - 256 if yi > 127 else yi, yq - 256 if yq > 127 else yq))
    return out


@cocotb.test()
async def test_twiddle_exhaustive(dut):
    checked = 0
    for e in range(32):
        dut.e32.value = e
        dut.e16.value = e % 16
        dut.e8.value = e % 4
        for xi in range(-128, 128):
            dut.xi.value = xi & 0xFF
            await Timer(1, unit="ns")
            g32, g16, g8 = unpack(dut.y32.value), unpack(dut.y16.value), unpack(dut.y8.value)
            for q in range(256):
                a = (xi, q - 128)
                assert g32[q] == fm.twiddle(a, e, T32, 8, 7), ("W32", e, a, g32[q])
                assert g16[q] == fm.twiddle(a, e % 16, T16, 8, 7), ("W16", e % 16, a, g16[q])
                assert g8[q] == fm.twiddle(a, 4 * (e % 4), T32, 8, 7), ("W8", e % 4, a, g8[q])
            checked += 3 * 256
    dut._log.info(f"{checked} twiddle results bit-exact (W32: 32 e, W16: 16 e, W8: 4 e, x 65,536 inputs)")
