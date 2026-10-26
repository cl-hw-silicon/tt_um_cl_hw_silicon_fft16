# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mini-OFDM acceptance from the chip's output pins only.

model/signal_gen.py (v1: BPSK pilots at k = 4 and 6 with their own PRBS phases, 16-QAM at k = 5, 37 LSB per
carrier) -> ui_in of the RTL chip -> uo_out / SOF / VALID / LOCK -> model/demod.py (acceptance
demodulator: two pilots, W = 16 pilot averaging, no decision feedback) -> bits scored against the TX bits.
The demodulator sees nothing but those four pins, sampled once per clock (what a reader sampling on the SCLK
falling edge gets). Every run is also compared pin-for-pin with the bit-exact model (chip_io.model_pins), so
the pin-level BER is the model's BER on the same samples.

  test_accept_*        30 dB, random start phase, >= 1000 locked symbols: BER 0, both CFG_EDGE settings.
                       There is no CFG_CP at N = 16 (CP fixed at 4); the negedge run drives the
                       unused uio[6] high to show it has no effect.
  test_ber_vs_snr      BER per SNR from the pins (reported; equal to the model loop's BER on the same stream).
  test_phase0_sweep    8 start phases across the 20-sample symbol: lock, BER 0.
  test_ppm_tracking    +-100 ppm with a +-20 LSB DC offset: one lock interval, slips seen in the SOF period,
                       net slips = sample drift, BER 0.
  test_64qam           64-QAM at k = 5: BER at 30 dB, BER vs SNR from the pins (= model loop), SNR at
                       BER 1e-3 against the model's measurement within 3 sigma of the counting statistics.
Pin index convention: pins[k] is the state after clock edge k; adc[k] enters the datapath at edge k + 2, so the
scorer (which maps frames to TX symbols by sample index) reads pins shifted by 2 (ofdm_loop.run_chip convention).
"""
import numpy as np
import cocotb

import signal_gen as sg
import demod as dm
import ofdm_loop as ol
import chip_io as io
from chip_io import N, L, CLK_PS

ACCEPT = dict(dd=False, avg_window=dm.AVG_WINDOW)
CFG = sg.OFDMConfig(N=N)                    # v1 signal (pilots at k = 4 and 6, 16-QAM at k = 5), CP = 4


async def chip_ofdm(dut, nsym, seed, snr_db, cfg_edge=0, uio6=0, ppm=0.0, phase0=None, dc=0.0, cfg=CFG):
    """One stream through the RTL; returns the demodulator result + score from the pins, and the pins."""
    await io.start(dut, cfg_edge=cfg_edge, uio6=uio6)
    adc, tx, bd = sg.generate(cfg, nsym, seed, snr_db=snr_db, ppm=ppm, phase0=phase0, dc_offset=dc)
    rst = np.ones(len(adc), dtype=int)
    rst[:4] = 0
    pins = await io.stream(dut, adc, rst)
    bad = io.compare(pins, io.model_pins(adc, rst))
    assert not bad, f"pins differ from the bit-exact model: {len(bad)}, first {bad[:5]}"
    p = {f: np.where(pins[f][2:] < 0, 0, pins[f][2:]) for f in ("sof", "valid", "lock")}   # X (gate level, t = 0) -> 0
    p["uo"] = np.where(pins["uo"][2:] < 0, 0, pins["uo"][2:])      # undefined bus (before the first flush) -> 0
    res = dm.demodulate(p, cfg, **ACCEPT)
    sc = ol.score_chip(cfg, res, tx, bd)
    fr = io.frames_from_pins(pins)
    sc.update(first_lock_sym=None if res["first_lock_cycle"] is None else res["first_lock_cycle"] / L,
              intervals=len(res["intervals"]), slips=[s for iv in res["intervals"] for s in iv["slips"]],
              prbs=[iv["prbs"] for iv in res["intervals"]], adc=adc, pins=pins, fr=fr)
    return sc


def summary(sc):
    evm = sc["evm_db"][5]
    return (f"lock after {sc['first_lock_sym']:.0f} symbols, {sc['intervals']} lock interval(s), {sc['frames']} frames "
            f"demodulated, BER {sc['ber']:.2e} ({sc['errs']}/{sc['nbits']}), EVM {evm:.1f} dB, "
            f"{len(sc['slips'])} slips, PRBS score {sc['prbs'][0]['score']:.2f}")


async def accept(dut, cfg_edge, uio6, seed, phase0, what):
    sc = await chip_ofdm(dut, 1500, seed, 30, cfg_edge=cfg_edge, uio6=uio6, phase0=phase0)
    dut._log.info(f"{what}: {summary(sc)}")
    assert sc["intervals"] == 1 and sc["frames"] >= 1000 and sc["ber"] == 0.0


@cocotb.test()
async def test_accept_posedge(dut):
    await accept(dut, 0, 0, seed=201, phase0=11.7, what="30 dB, CFG_EDGE=0")


@cocotb.test()
async def test_accept_negedge_uio6_high(dut):
    await accept(dut, 1, 1, seed=202, phase0=3.2, what="30 dB, CFG_EDGE=1, uio[6]=1 (unused input)")


@cocotb.test()
async def test_ber_vs_snr(dut):
    """BER vs SNR from the pins (reported, not gated). The model loop on the same ADC stream must give the same BER
    (the pins are bit-exact); 18 dB must be error-free."""
    snrs, rows = (10, 12, 14, 16, 18), []
    for snr in snrs:
        sc = await chip_ofdm(dut, 1500, 300 + snr, snr, phase0=5.5)
        m = ol.score_chip(CFG, dm.demodulate(ol.run_chip(CFG, sc["adc"]), CFG, **ACCEPT), *sg.generate(
            CFG, 1500, 300 + snr, snr_db=snr, phase0=5.5)[1:])
        rows.append((snr, sc["ber"], sc["nbits"], m["ber"], ol.theory_ber(CFG, snr), sc["evm_db"][5]))
        assert sc["ber"] == m["ber"], f"{snr} dB: pins BER {sc['ber']} vs model loop {m['ber']}"
    tab = "\n".join(f"  {s:2d} dB: BER {b:.2e} ({int(round(b * n))}/{n} bits)  model loop {mb:.2e}  theory {t:.1e}  EVM {e:.1f} dB"
                    for s, b, n, mb, t, e in rows)
    s_chip = ol.snr_at_ber([r[0] for r in rows], [r[1] for r in rows])
    s_th = ol.snr_at_ber([r[0] for r in rows], [r[4] for r in rows])
    import json, os                                # the model's figure from its study file (no hard-coded copy)
    here = os.path.dirname(os.path.abspath(__file__))
    ref = next(r for r in json.load(open(os.path.join(here, '..', 'model', 'study_results', 'precision_ber.json')))['summary']
               if r['option'] == 'b' and r['mod'] == '16QAM')
    dut._log.info(f"BER vs SNR from the pins (1500 symbols per point):\n{tab}\n  SNR at BER 1e-3: pins {s_chip:.2f} dB, "
                  f"theory {s_th:.2f} dB (model, 6000 symbols, 0 ppm: {ref['0 ppm']:.2f} / {ref['theory']:.2f})")
    assert rows[-1][1] == 0.0
    assert all(a[1] >= b[1] for a, b in zip(rows, rows[1:]))


@cocotb.test()
async def test_phase0_sweep(dut):
    """Random-phase acquisition: 8 start phases spread over one 20-sample symbol (25 dB)."""
    out = []
    for j in range(8):
        ph = j * L / 8 + 0.37
        sc = await chip_ofdm(dut, 800, 400 + j, 25, phase0=ph)
        out.append((ph, sc))
        assert sc["intervals"] == 1 and sc["ber"] == 0.0 and sc["frames"] >= 300, f"phase0 {ph}: {summary(sc)}"
    dut._log.info("phase0 sweep (25 dB):\n" + "\n".join(f"  phase0 {ph:5.2f}: {summary(sc)}" for ph, sc in out))


@cocotb.test()
async def test_ppm_tracking(dut):
    """+-100 ppm sample-rate offset (and +-20 LSB DC): LOCK holds, the SOF period shows the slips, net slips =
    drift, BER 0."""
    for ppm, dc in ((100.0, 20.0), (-100.0, -20.0)):
        sc = await chip_ofdm(dut, 3000, 500, 30, ppm=ppm, phase0=8.1, dc=dc)
        lk = sc["pins"]["lock"]
        locked_samples = int(np.sum(lk == 1))
        net = sum(d for _, d in sc["slips"])
        drift = -ppm * 1e-6 * locked_samples              # +ppm: fewer RX samples per symbol -> early slips (L - 1)
        dut._log.info(f"{ppm:+.0f} ppm, DC {dc:+.0f} LSB: {summary(sc)}; net slip {net:+d} samples vs drift "
                      f"{drift:+.1f} over {locked_samples} locked samples")
        assert sc["intervals"] == 1 and sc["ber"] == 0.0 and sc["frames"] >= 2000
        assert len(sc["slips"]) >= 3 and abs(net - drift) <= 3


@cocotb.test()
async def test_64qam(dut):
    """64-QAM at k = 5 through the pins. The BER at each SNR equals the model loop's on the same stream;
    the SNR at BER 1e-3 agrees with the model measurement for the gain-2/N pipeline (model/study_results/
    precision_ber.json, option b, 0 ppm: 22.24 dB) within 3 sigma of the counting statistics of both."""
    import json
    import os
    cfg = sg.OFDMConfig(N=N, modulation='64QAM')
    rows = []
    for snr in (18, 20, 22, 24, 26, 30):
        nsym = 3000
        sc = await chip_ofdm(dut, nsym, 600 + snr, snr, phase0=2.7, cfg=cfg)
        m = ol.score_chip(cfg, dm.demodulate(ol.run_chip(cfg, sc["adc"]), cfg, **ACCEPT),
                          *sg.generate(cfg, nsym, 600 + snr, snr_db=snr, phase0=2.7)[1:])
        assert sc["ber"] == m["ber"], f"{snr} dB: pins BER {sc['ber']} vs model loop {m['ber']}"
        rows.append((snr, sc["ber"], sc["nbits"], ol.theory_ber(cfg, snr), sc["evm_db"][5]))
    curve = rows[:-1]
    s = ol.snr_at_ber([r[0] for r in curve], [r[1] for r in curve])
    sig = ol.snr_at_ber_sigma([r[0] for r in curve], [r[1] for r in curve], [r[2] for r in curve])
    here = os.path.dirname(os.path.abspath(__file__))
    rec = json.load(open(os.path.join(here, '..', 'model', 'study_results', 'precision_ber.json')))['points']
    rr = sorted((p for p in rec if p['opt'] == 'b' and p['mod'] == '64QAM' and p['cond'] == '0 ppm'), key=lambda p: p['snr'])
    s_ref = ol.snr_at_ber([p['snr'] for p in rr], [p['ber'] for p in rr])
    sig_ref = ol.snr_at_ber_sigma([p['snr'] for p in rr], [p['ber'] for p in rr], [p['nbits'] for p in rr])
    s_th = ol.snr_at_ber([r[0] for r in curve], [r[3] for r in curve])
    tab = "\n".join(f"  {a:2d} dB: BER {b:.2e} ({int(round(b * n))}/{n} bits)  theory {t:.1e}  EVM {e:.1f} dB" for a, b, n, t, e in rows)
    dut._log.info(f"64-QAM from the pins (3000 symbols per point, all equal to the model loop):\n{tab}\n  SNR at BER 1e-3: "
                  f"pins {s:.2f} +- {sig:.2f} dB, model {s_ref:.2f} +- {sig_ref:.2f} dB, theory {s_th:.2f} dB")
    assert abs(s - s_ref) <= 3 * np.hypot(sig, sig_ref)
