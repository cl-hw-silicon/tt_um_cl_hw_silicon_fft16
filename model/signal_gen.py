# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mini-OFDM test-signal generator (the chip's acceptance signal; docs/info.md "How to test").

Real-valued mini-OFDM with cyclic prefix: symbols of CP + N samples (CP = N/4 or N/8 per CFG_CP),
BPSK pilots whose signs follow a PRBS per symbol (v1: two pilots at k = 4 and k = 6, each
with its own PRBS phase), one or more QAM data carriers with random Gray-coded bits (v1: 16-QAM at
k = 5), random initial symbol phase, optional AWGN, DC offset and TX/RX sample-rate offset, 8-bit
offset-binary ADC output. The earlier single-pilot configuration (LEGACY_1P) stays selectable.

Signal model. A carrier of time-domain amplitude `a` (ADC LSB) at bin k with complex symbol c (|c| <= 1
for the outer constellation corner) contributes, inside symbol j at position u in [0, L) (u = 0 is the
first CP sample):
        a * Re( c * exp(j*2*pi*k*(u - CP)/N) )
The CP is therefore an exact copy of the last CP useful samples. An ideal 1/N FFT of the useful part
shows a*c/2 in bin k.

PRBS (the demodulator regenerates it): PRBS7, polynomial x^7 + x^6 + 1, Fibonacci LFSR, 7-bit state,
seed 0x7F. Output bit b_i (i = 0, 1, ...) = state bit 6, then state <- ((state << 1) | (b6 ^ b5)) & 0x7F.
Period 127. Pilot p (p = 0, 1, ... in cfg.pilot_bins order) uses offset o_p (cfg.pilot_prbs_offsets,
v1: 0 for k = 4, 63 for k = 6): sign of pilot p in symbol j = +1 if b_{(j + o_p) mod 127} == 0 else -1.
Offset 63 is half the period: the two sign sequences are different shifts of the m-sequence, their
product is another shift, so the two pilot tones never form a fixed waveform.

Sample-rate offset (ppm). The RX samples the ideal continuous-time waveform at TX times
t_n = phase0 + n * (1 + ppm * 1e-6) (in TX sample periods): an exact resample of the ideal stream (the
waveform inside every symbol is a finite sum of complex exponentials, evaluated directly), not a
carrier-frequency offset.

Amplitude: per-carrier amplitude a = 64 / sqrt(M) for M carriers (pilots included) by
default — 36.95 LSB for the v1 two pilots + one data carrier (45.25 LSB for the legacy 2-carrier
signal) — so the peak of the sum stays inside the 8-bit range (clipping is counted and reported).

SNR: per-sample SNR = analytic mean signal power / AWGN power at the ADC input (before quantisation).
"""
import csv
import json
from dataclasses import dataclass, field, asdict

import numpy as np


# ------------------------------------------------------------------------------------------------------
# PRBS and Gray QAM (shared with symgen.py, the minimal generator of the first timing study)
def prbs7(nbits, seed=0x7F):
    """PRBS7 x^7 + x^6 + 1, seed 0x7F; returns nbits bits (b_0 = MSB of the seed state)."""
    s, out = seed, []
    for _ in range(nbits):
        b = ((s >> 6) ^ (s >> 5)) & 1
        out.append((s >> 6) & 1)
        s = ((s << 1) | b) & 0x7F
    return np.array(out)


def gray_levels(bits_per_axis):
    """Gray-coded PAM levels per axis, index = bit pattern (MSB first), values in {-(2^m-1) ... 2^m-1}."""
    m = bits_per_axis
    lv = np.zeros(1 << m)
    for v in range(1 << m):                       # natural order level index v -> Gray code g
        g = v ^ (v >> 1)
        lv[g] = 2 * v - ((1 << m) - 1)
    return lv


MODS = {'BPSK': 0, 'QPSK': 1, '16QAM': 2, '64QAM': 3, '256QAM': 4}   # bits per axis (BPSK separate; 256QAM: precision study)
QAM16_LEVELS = gray_levels(2) / 3.0                           # 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3 (/3)


def qam_map(bits, mod):
    """bits: (..., 2m) -> complex points normalised so that the outer corner has |c| = 1."""
    m = MODS[mod]
    lv = gray_levels(m) / ((1 << m) - 1)
    w = 1 << np.arange(m - 1, -1, -1)
    bi = (bits[..., :m] * w).sum(-1)
    bq = (bits[..., m:2 * m] * w).sum(-1)
    return (lv[bi] + 1j * lv[bq]) / np.sqrt(2)


def qam16(bits4):
    """Single 16-QAM point (kept for symgen.py)."""
    return complex(qam_map(np.asarray(bits4)[None, :], '16QAM')[0])


def qam_mean_power(mod):
    m = MODS[mod]
    lv = gray_levels(m) / ((1 << m) - 1)
    return 2 * np.mean(lv ** 2) / 2


# ------------------------------------------------------------------------------------------------------
PROJECT_N = 16                           # the chip's N; N = 32 (the earlier design point) kept for history


def cfg_cps(N):
    """CFG_CP settings that exist at this N: both at N = 32, only CP = N/4 at N = 16 (N/8 = 2 is unusable)."""
    return (0,) if N == 16 else (0, 1)


@dataclass
class OFDMConfig:
    N: int = PROJECT_N
    cfg_cp: int = 0                      # 0 -> CP = N/4 (default), 1 -> CP = N/8; ignored at N = 16 (CP fixed at N/4)
    fs: float = 32e6
    pilot_bins: tuple = (4, 6)           # two pilots bracketing the data bin
    data_bins: tuple = (5,)
    modulation: str = '16QAM'
    amp: float = None                    # per-carrier time-domain amplitude [LSB]; None -> 64/sqrt(M)
    prbs_seed: int = 0x7F
    pilot_prbs_offsets: tuple = (0, 63)  # PRBS7 phase offset per pilot (symbols)

    @property
    def pilot_bin(self):                 # first (reference) pilot
        return self.pilot_bins[0]

    @property
    def CP(self):
        return self.N // 4 if (self.cfg_cp == 0 or self.N == 16) else self.N // 8

    @property
    def L(self):
        return self.N + self.CP

    @property
    def n_carriers(self):
        return len(self.pilot_bins) + len(self.data_bins)

    @property
    def a(self):
        return self.amp if self.amp is not None else 64.0 / np.sqrt(self.n_carriers)

    @property
    def bits_per_symbol(self):
        return 2 * MODS[self.modulation]

    def signal_power(self):
        """Analytic mean power of the noise-free waveform: sum over carriers of a^2 E|c|^2 / 2."""
        return self.a ** 2 / 2 * (len(self.pilot_bins) + len(self.data_bins) * qam_mean_power(self.modulation))

    def pilot_signs(self, n_symbols):
        """{k: +-1 per symbol} for every pilot (regenerable by the demodulator)."""
        b = prbs7(127, self.prbs_seed)
        j = np.arange(n_symbols)
        return {k: 1 - 2 * b[(j + o) % 127] for k, o in zip(self.pilot_bins, self.pilot_prbs_offsets)}


V1 = OFDMConfig()                                                      # v1 acceptance signal
LEGACY_1P = OFDMConfig(pilot_bins=(4,), pilot_prbs_offsets=(0,))       # single-pilot legacy signal (expected failure)


def tx_symbols(cfg, n_symbols, rng):
    """Per-symbol carrier matrix (symbol x bin) and the TX record."""
    ps = cfg.pilot_signs(n_symbols)
    first = cfg.pilot_bins[0]
    rec = dict(pilot_signs=ps, pilot_sign=ps[first],
               pilot_prbs=(1 - ps[first]) // 2, data_bits={}, data_points={})
    for k in cfg.data_bins:
        bits = rng.integers(0, 2, size=(n_symbols, cfg.bits_per_symbol))
        rec['data_bits'][k] = bits
        rec['data_points'][k] = qam_map(bits, cfg.modulation)
    carriers = np.zeros((n_symbols, cfg.N), dtype=complex)      # c per (symbol, bin), 0 for unused bins
    for k in cfg.pilot_bins:
        carriers[:, k] = ps[k]
    for k in cfg.data_bins:
        carriers[:, k] = rec['data_points'][k]
    return carriers, rec


def waveform(cfg, carriers, t):
    """Ideal real waveform at TX times t (fractional, in samples)."""
    t = np.asarray(t, dtype=float)
    j = np.floor(t / cfg.L).astype(int)
    u = t - j * cfg.L
    bins = list(cfg.pilot_bins) + list(cfg.data_bins)
    y = np.zeros_like(t)
    for k in bins:
        y += cfg.a * np.real(carriers[j, k] * np.exp(2j * np.pi * k * (u - cfg.CP) / cfg.N))
    return y


def generate(cfg, n_symbols, seed, snr_db=None, dc_offset=0.0, phase0=None, ppm=0.0):
    """Returns (adc, tx, bounds):
    adc     np.ndarray of offset-binary ADC codes 0..255 (one per RX sample, 32 MS/s)
    tx      dict: pilot_prbs / pilot_sign per symbol, data_bits / data_points per data bin, config,
            amplitude, signal power, clipping fraction
    bounds  dict: t (TX time of every RX sample), sym (TX symbol index of every RX sample),
            sym_start_rx (fractional RX sample index of the start (first CP sample) of every symbol),
            phase0, ppm, y_float (the same noisy samples before rounding/clipping)
    phase0: TX time (samples, in [0, L)) of the first RX sample; None -> uniform random from `seed`."""
    rng = np.random.default_rng(seed)
    if phase0 is None:
        phase0 = float(rng.uniform(0, cfg.L))
    carriers, rec = tx_symbols(cfg, n_symbols, rng)
    step = 1.0 + ppm * 1e-6
    nsamp = int(np.floor(((n_symbols * cfg.L) - phase0) / step - 1e-9))
    t = phase0 + np.arange(nsamp) * step
    y = waveform(cfg, carriers, t) + dc_offset
    if snr_db is not None:
        y = y + rng.normal(0.0, np.sqrt(cfg.signal_power() / 10 ** (snr_db / 10)), size=y.shape)
    q = np.round(y)
    clip = float(np.mean((q < -128) | (q > 127)))
    x = np.clip(q, -128, 127).astype(int)
    adc = (x + 128).astype(np.int64)                         # offset binary: 0x80 = mid-scale
    tx = dict(rec, config=asdict(cfg), CP=cfg.CP, L=cfg.L, a=cfg.a, signal_power=cfg.signal_power(),
              clip_fraction=clip, snr_db=snr_db, dc_offset=dc_offset, seed=seed)
    bounds = dict(t=t, sym=np.floor(t / cfg.L).astype(int),
                  sym_start_rx=(np.arange(n_symbols) * cfg.L - phase0) / step, phase0=phase0, ppm=ppm,
                  y_float=y)                                  # noisy analog samples before the ADC (reference path)
    return adc, tx, bounds


def analog(cfg, tx, bounds, snr_db=None, seed=0):
    """Float (unquantised) version of the same RX stream: noise-free ideal waveform at the same
    sampling instants plus (optionally) fresh AWGN at snr_db. (generate() also returns the noisy float
    samples it quantised, bounds['y_float'], which is what the reference path uses.)"""
    n = len(tx['pilot_sign'])
    carriers = np.zeros((n, cfg.N), dtype=complex)
    for k in cfg.pilot_bins:
        carriers[:, k] = tx['pilot_signs'][k]
    for k in cfg.data_bins:
        carriers[:, k] = tx['data_points'][k]
    y = waveform(cfg, carriers, bounds['t']) + tx['dc_offset']
    if snr_db is not None:
        y = y + np.random.default_rng(seed).normal(0, np.sqrt(cfg.signal_power() / 10 ** (snr_db / 10)), size=y.shape)
    return y


def to_signed(adc):
    """Offset binary -> two's-complement sample value (what the chip's capture register produces)."""
    return np.asarray(adc, dtype=int) - 128


# ------------------------------------------------------------------------------------------------------
def save(path_prefix, adc, tx, bounds):
    """Bench export: <prefix>.npy (ADC codes), <prefix>.csv (one code per line), <prefix>_tx.json
    (TX record incl. bits, PRBS and config) and <prefix>_bounds.npz (true timing)."""
    np.save(path_prefix + '.npy', np.asarray(adc, dtype=np.uint8))
    with open(path_prefix + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['adc_offset_binary'])
        for v in adc:
            w.writerow([int(v)])
    js = {k: v for k, v in tx.items() if k not in ('data_bits', 'data_points', 'pilot_prbs', 'pilot_sign', 'pilot_signs')}
    js['pilot_prbs'] = np.asarray(tx['pilot_prbs']).tolist()
    js['pilot_signs'] = {str(k): np.asarray(v).tolist() for k, v in tx['pilot_signs'].items()}
    js['data_bits'] = {str(k): np.asarray(v).tolist() for k, v in tx['data_bits'].items()}
    js['data_points'] = {str(k): [[float(c.real), float(c.imag)] for c in v] for k, v in tx['data_points'].items()}
    with open(path_prefix + '_tx.json', 'w') as f:
        json.dump(js, f)
    np.savez(path_prefix + '_bounds.npz', t=bounds['t'], sym=bounds['sym'], sym_start_rx=bounds['sym_start_rx'],
             phase0=bounds['phase0'], ppm=bounds['ppm'])


def load(path_prefix):
    adc = np.load(path_prefix + '.npy').astype(np.int64)
    with open(path_prefix + '_tx.json') as f:
        js = json.load(f)
    tx = dict(js)
    tx['pilot_prbs'] = np.array(js['pilot_prbs'])
    tx['pilot_signs'] = {int(k): np.array(v) for k, v in js['pilot_signs'].items()}
    tx['pilot_sign'] = 1 - 2 * tx['pilot_prbs']
    tx['data_bits'] = {int(k): np.array(v) for k, v in js['data_bits'].items()}
    tx['data_points'] = {int(k): np.array([complex(a, b) for a, b in v]) for k, v in js['data_points'].items()}
    z = np.load(path_prefix + '_bounds.npz')
    bounds = dict(t=z['t'], sym=z['sym'], sym_start_rx=z['sym_start_rx'], phase0=float(z['phase0']), ppm=float(z['ppm']))
    return adc, tx, bounds


def config_from_dict(d):
    d = dict(d)
    for k in ('data_bins', 'pilot_bins', 'pilot_prbs_offsets'):
        d[k] = tuple(d[k])
    return OFDMConfig(**d)


# ------------------------------------------------------------------------------------------------------
def bench_stimulus(n_symbols, seed, snr_db=None, dc_offset=0.0, ppm=0.0, modulation='16QAM'):
    """Stimulus for playback into the chip on the bench (docs/info.md "How to test"). With ppm = 0 the
    stream starts at a symbol boundary (phase0 = 0) and holds exactly n_symbols * L samples, so it can be played
    in a loop; with n_symbols a multiple of 127 the pilot PRBS also continues across the loop seam. One extra
    symbol is generated and cut (generate() stops a fraction of a sample short of the last symbol's end)."""
    cfg = OFDMConfig(modulation=modulation)
    adc, tx, bd = generate(cfg, n_symbols + 1, seed, snr_db=snr_db, dc_offset=dc_offset, phase0=0.0, ppm=ppm)
    n = n_symbols * cfg.L
    adc = adc[:n]
    for k in ('pilot_sign', 'pilot_prbs'):
        tx[k] = tx[k][:n_symbols]
    for k in ('pilot_signs', 'data_bits', 'data_points'):
        tx[k] = {b: v[:n_symbols] for b, v in tx[k].items()}
    bd = dict(bd, t=bd['t'][:n], sym=bd['sym'][:n], sym_start_rx=bd['sym_start_rx'][:n_symbols])
    return cfg, adc, tx, bd


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Write a mini-OFDM ADC stimulus (v1 signal) for the bench: "
                                 "PREFIX.csv (one offset-binary code per line, 32 MS/s or any rate the chip is clocked "
                                 "at), PREFIX.npy, PREFIX_tx.json (bits, pilots, config: model/demod.py --tx) and "
                                 "PREFIX_bounds.npz.")
    ap.add_argument("prefix")
    ap.add_argument("--symbols", type=int, default=1270, help="symbols of 20 samples (default 1270 = 10 x 127: loopable)")
    ap.add_argument("--snr", type=float, default=None, help="AWGN SNR in dB at the ADC input (default: no noise)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dc", type=float, default=0.0, help="DC offset in LSB")
    ap.add_argument("--ppm", type=float, default=0.0, help="TX/RX sample-rate offset (the file is then not loopable)")
    ap.add_argument("--modulation", default="16QAM", choices=[m for m in MODS if m != "BPSK"])
    a = ap.parse_args(argv)
    cfg, adc, tx, bd = bench_stimulus(a.symbols, a.seed, a.snr, a.dc, a.ppm, a.modulation)
    save(a.prefix, adc, tx, bd)
    loop = "loopable" if a.ppm == 0 and a.symbols % 127 == 0 else \
           "not loopable (ppm != 0)" if a.ppm else "loop seam breaks the pilot PRBS (symbols not a multiple of 127)"
    print(f"{a.prefix}.csv: {len(adc)} samples = {a.symbols} symbols of {cfg.L}, {a.modulation} at k = "
          f"{list(cfg.data_bins)}, pilots k = {list(cfg.pilot_bins)}, {cfg.a:.1f} LSB per carrier, SNR "
          f"{'none' if a.snr is None else a.snr}, clipping {tx['clip_fraction']:.1e}; {loop}")


if __name__ == "__main__":
    main()
