# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal mini-OFDM symbol generator for the first CP-timing study (signal_gen.py supersedes it).

Real signal, symbols = CP + N samples. Carriers: BPSK pilot whose sign follows a PRBS per symbol
(PRBS7, x^7 + x^6 + 1, seed 0x7F, first output used for symbol 0) and 16-QAM data with random Gray
symbols. A carrier of amplitude `a` (ADC LSB, time-domain cosine amplitude) at bin k contributes
a * Re(c * exp(j*2*pi*k*(u - CP)/N)) at symbol position u (u = 0 is the first CP sample), |c| <= 1,
so the chip's 1/N FFT shows a*c/2 in bin k.

Continuous-time evaluation inside each symbol makes TX/RX sample-rate offsets exact: the receiver
samples at TX times t_n = t0 + n * (1 + ppm*1e-6) (in TX sample periods). AWGN is added at a given
per-sample SNR (signal power / noise power), then rounded and clipped to the 8-bit ADC range.
"""
import numpy as np

from signal_gen import prbs7, qam16, QAM16_LEVELS     # shared primitives (identical definitions)


class MiniOFDM:
    def __init__(self, N=32, CP=8, pilot_bin=4, data_bins=(5,), a_pilot=48.0, a_data=48.0, seed=1):
        self.N, self.CP, self.L = N, CP, N + CP
        self.pilot_bin, self.data_bins = pilot_bin, tuple(data_bins)
        self.a_pilot, self.a_data = a_pilot, a_data
        self.rng = np.random.default_rng(seed)

    def symbols(self, nsym):
        """Per-symbol carrier lists [(bin, amplitude, complex c)] and the TX record."""
        pil = prbs7(nsym)
        rec = {'pilot_sign': 1 - 2 * pil, 'data_bits': {}, 'data_sym': {}}
        carriers = []
        for k in self.data_bins:
            bits = self.rng.integers(0, 2, size=(nsym, 4))
            rec['data_bits'][k] = bits
            rec['data_sym'][k] = np.array([qam16(b) for b in bits])
        for j in range(nsym):
            cj = [(self.pilot_bin, self.a_pilot, complex(rec['pilot_sign'][j]))]
            cj += [(k, self.a_data, rec['data_sym'][k][j]) for k in self.data_bins]
            carriers.append(cj)
        return carriers, rec

    def waveform(self, carriers, t):
        """Ideal real waveform at (fractional) TX times t >= 0 (in samples)."""
        t = np.asarray(t, dtype=float)
        j = np.floor(t / self.L).astype(int)
        u = t - j * self.L
        y = np.zeros_like(t)
        for jj in np.unique(j):
            sel = j == jj
            for (k, a, c) in carriers[jj]:
                y[sel] += a * np.real(c * np.exp(2j * np.pi * k * (u[sel] - self.CP) / self.N))
        return y

    def stream(self, nsym, offset=0.0, ppm=0.0, snr_db=None, nsamp=None):
        """ADC samples (two's-complement ints in [-128, 127]) starting `offset` TX samples into
        symbol 0 (random initial phase), with sample-rate offset `ppm` and AWGN at `snr_db`.
        Returns (x, info) with info['t'] = TX time of every RX sample (true timing for scoring)."""
        carriers, rec = self.symbols(nsym)
        if nsamp is None:
            nsamp = int((nsym * self.L - offset - 2 * self.L) / (1 + ppm * 1e-6))
        t = offset + np.arange(nsamp) * (1 + ppm * 1e-6)
        assert t[-1] < nsym * self.L
        y = self.waveform(carriers, t)
        if snr_db is not None:
            p = np.mean(y ** 2)
            y = y + self.rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), size=y.shape)
        x = np.clip(np.round(y), -128, 127).astype(int)
        return x, {'t': t, 'carriers': carriers, 'rec': rec, 'L': self.L, 'CP': self.CP}


def noise_stream(nsamp, sigma=40.0, seed=2):
    """Pure noise input (false-lock tests)."""
    rng = np.random.default_rng(seed)
    return np.clip(np.round(rng.normal(0, sigma, nsamp)), -128, 127).astype(int)
