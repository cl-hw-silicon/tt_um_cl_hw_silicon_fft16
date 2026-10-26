# model/ — bit-exact models and the mini-OFDM loop

Integer, cycle-exact Python models that the RTL mirrors 1:1, plus the test-signal generator and the
host demodulator. `python3 -m pytest model` (from the repository root) runs everything; one strict
xfail documents the single-pilot limit.

| File | What |
| --- | --- |
| `fft_model.py` | `SDF(N, W=8, TW=7)`: N-point real-input R2²SDF FFT. `step(x, cnt=None) -> (I, Q)`; `tw_table`, `bitrev`, `reference`, `run_frames`, `control` (the schedule), `schedule_proof` (proof by linearity). |
| `timing.py` | `CPTiming(N, cfg_cp, K, M, lock_frac, unlock_frac, hyst)`: sign-only CP correlator, sequential acquisition, early/late tracking, pause. `step(x) -> record` (ph, run, lock, slip, …). |
| `system_model.py` | `FFTChipModel`: timing block + paused SDF + Hermitian tail + SOF/VALID/LOCK/DBG, register for register. |
| `symgen.py` | The minimal generator of the first timing study (kept for its timing tests; shares PRBS/QAM with `signal_gen.py`). |
| `signal_gen.py` | Mini-OFDM generator: `generate(cfg, n_symbols, seed, snr_db, dc_offset, phase0, ppm) -> (adc offset-binary, tx record, true bounds)`, `save`/`load` (.npy / CSV / JSON). |
| `demod.py` | Host demodulator from the pins (uo_out, SOF, VALID, LOCK): frames, joint PRBS sync, two-pilot interpolation with pilot averaging (the acceptance demodulator), optional decision-directed refinement, Gray slicer, EVM. |
| `ofdm_loop.py` | Generator → chip model → demod, the reference paths A (float, ideal timing) / B (8-bit + bit-exact FFT, ideal timing), BER curves (`python3 ofdm_loop.py`). |
| `test_ofdm_model.py` | Closed-loop pytest: generator, v1 acceptance (30 dB, 8 phases, DC ±20, ±100 ppm × 10⁴), legacy strict xfail, optional DD, BER curve + loss breakdown, stretch rows (report only), constellation PNGs (`out/`). |
| `estimate.py` | Area estimate (an early estimate: cell areas from the liberty, logic calibrated against Yosys, measured post-layout overheads). |
| `study_timing.py` | CP-timing parameter study at N = 32 (`acq`, `acq2`, `noise`, `track`, `final`); results in `study_results/`. |
| `study_n16.py` | The CP timing block at N = 16 as built (`acq`, `hm`, `edges`: LOCK fall, constant input, ±300/±400 ppm); `study_results/n16_*.json`. |
| `study_precision.py` | The precision study that chose the 2/N last stage; `study_results/precision_*.json`. |
| `test_model.py`, `test_timing.py` | pytest suites. |

## FFT (`fft_model.py`)

* **Pipeline, N = 2ⁿ:** stage s = 1..n, delay N/2ˢ. Odd s are BF2I, even s are BF2II (−j on the
  incoming element when `~b[n−s+1] & b[n−s]`). After each even s < n comes a twiddle W_M^e with
  M = 2^(n−s+2), e = (k1 + 2k2)·(cnt mod 2^(n−s)), k1 = b[n−s+1] ^ b[n−s], k2 = ~b[n−s]. It is
  stored as W_N^(e·2^(s−2)) in one table.
  N = 16 (as built): BF2I(8) BF2II(4) ×W16 BF2I(2) BF2II(1). N = 32 (still supported by the parameterised
  model and RTL): BF2I(16) BF2II(8) ×W32 BF2I(4) BF2II(2) ×W8 BF2I(1).
* **Schedule proven, not sampled:** `schedule_proof(N)` measures the linear map from frames
  (f−1, f, f+1) to frame f's outputs with 3N impulses. It equals [0 | DFT_bitrev | 0] to ≤ 1.4e−13
  for N = 4…128, so outputs depend on frame f only and the latency is N − 1. A wrong k1 fails it.
* **Output order:** slot t = (cnt + 1) mod N carries bin k = bitrev(t). Even slots carry exactly
  k < N/2. Order on the bus at N = 16 (as built): 0, 4, 2, 6, 1, 5, 3, 7 (at N = 32 it would be 0, 8, 4, 12, 2, 10,
  6, 14, 1, 9, 5, 13, 3, 11, 7, 15).
* **Scaling and arithmetic:**
  - Every butterfly except the last computes (a ± b) >> 1 with floor. **As built:** the last stage computes
    a ± b without the shift, saturated to 8 bits (`bf_add_sat` / `bf_sub_sat`, exhaustively tested with corners),
    for a net gain of **2/N** (fitted gain 2.00 ± 0.02, phase < 0.5°). `SDF(..., last_div2=True)` keeps the earlier
    1/N pipeline for reference only.
  - Other options for studies only (the precision study): `Wi` (internal width with Q1.(Wi−1) twiddles and convergent
    rounding to 8 bits). Results are in `study_results/precision_*.json`.
  - `fm.reference()` is the ideal FFT × 2/N, each component clipped to 8 bits. Saturation is 1-Lipschitz, so the
    error against it is the fixed-point error alone.
  - Twiddle coefficients are Q1.7, rounded half-up, and clipped at +127. Exponent 0 bypasses the multiplier exactly.
  - No used exponent needs the +128 → +127 clip. For N = 32 the used W32 exponents are
    {0…10, 12, 14, 15, 18, 21} and the W8 exponents {0, 4, 8, 12}.
  - Inside the pipeline, saturation happens only at the −128 corner: −j of −128 → +127, and a twiddle on (−128, 0).
  - At the last stage it happens for |DC| ≥ 64 LSB, for near-full-scale half-bin tones next to DC/Nyquist,
    and at 1.6e−5 per component for full-scale random input.
* **Noise vs the clipped `np.fft.fft · 2/N`** (`test_model.py::test_random_frames`, seed N, 2000 random
  full-scale frames for N ≤ 32, 600 for N = 64). The earlier 1/N pipeline's values
  are in brackets:

  | N | max \|err\| | rms | rms / 2 (relative to the 1/N scale) | bound in test |
  | --- | --- | --- | --- | --- |
  | 16 (as built) | 3.78 LSB (2.59) | 1.194 LSB (0.875) | 0.60 | 4.5 / 1.35 |
  | 32 | 4.47 LSB (2.88) | 1.396 LSB (0.966) | 0.70 | 5.2 / 1.55 |
  | 64 | 5.25 LSB (3.29) | 1.505 LSB (1.017) | 0.75 | 5.7 / 1.7 |

  Odd (conjugate) slots differ from the exact conjugate of their even partner by up to 3.6 LSB
  (fixed-point asymmetry). They are never emitted.
* **Bit-exactness in the RTL:** any exact product structure (4 multipliers, Gauss 3-multiplier with
  full-width intermediates, CSD) gives the same bits, because the model forms the exact sum before
  the `>> 7`.

## CP timing (`timing.py`)

* **Correlator:** sign bit of the captured sample, an N × 1 sign delay line, XNOR → ±1, accumulated in A
  (signed, |A| ≤ K·CP = 4N → 9 bits) over the window samples of K symbols.
* **Symbol-phase counter `ph` = 0…L−1** (L = N + CP) is the only time base. The **run region
  (FFT window) is ph ∈ [0, N)**, so the datapath counter is `ph`'s low n bits (no second counter).
  The pause is ph ∈ [N, L).
* **Window placement:** the FFT window starts δ = CP/2 samples inside the CP. It holds exactly one
  symbol, cyclically rotated by δ, for any timing error e ∈ [−(CP−δ), δ+1): [−2, 3) at CP = 4 (as built),
  [−4, 5) at CP = 8 (N = 32).
* **Correlation windows** (a = N − CP + δ, where the last CP samples of the symbol fall when aligned):
  - ON: ph ∈ [a, a+CP)
  - EARLY: ph ∈ [a−1, a+CP−1)
  - LATE: ph ∈ [a+1, a+CP+1)
* **Decisions** are made once per symbol, at the edge ending the ph = L−2 cycle, using A including
  that cycle's sample.
  - A hypothesis advance or late slip repeats the L−2 cycle. An early slip skips the L−1 cycle.
    Both are pause cycles, so every frame gets exactly N contiguous samples and a slip never cuts a frame.
  - ACQ: after K symbols, A ≥ T_lock → LOCK (TRACK); else advance the hypothesis by one sample.
  - TRACK cycle: ON(K) → EARLY(K) → LATE(K) → WAIT(M).
    - Slip if the neighbour's A > A_on + H.
    - After an early slip, LATE is skipped.
    - Unlock after two consecutive ON measurements with A_on < T_unlock.
* **Before lock** the datapath free-runs (`run` = 1) with VALID = 0. **Once locked**, `run` = (ph < N).
  `run` is a register driving the ICG GATE directly.
* **Reset and priming:** synchronous reset of the control registers. The sign line has no reset;
  accumulation starts after the first ph wrap (`primed`), so no X reaches A in GL sim.
* **Parameters** (chosen by the CP-timing parameter study; rescaled with N by the formulas, K·CP = 4N). As built
  (N = 16, CP = 4) and at the earlier N = 32 design point (both CFG_CP settings):

  | parameter | formula | **N = 16, CP = 4 (as built)** | N = 32, CP = 8 | N = 32, CP = 4 |
  | --- | --- | --- | --- | --- |
  | K (symbols per measurement) | 4N/CP | **16** | 16 | 32 |
  | M (wait symbols per tracking cycle) | — | **16** | 16 | 16 |
  | T_lock = 0.75·K·CP | 3N | **48** | 96 | 96 |
  | T_unlock = 0.375·K·CP | ⌈1.5N⌉ | **24** | 48 | 48 |
  | H (hysteresis, A units) | — | **8** | 8 | 8 |

* **Lock behaviour, as built** (N = 16, v1 signal, 200 random start phases per SNR; `study_results/n16_acq.json`):
  median / p95 acquisition 160 / 305 symbols (0.10 / 0.19 ms) at 20 dB, 176 / 305 symbols at 30 dB; 93.5 % lock
  within 968 symbols at 10 dB; 0 unclean locks; 0 false locks in 141 ms of noise and 135 ms of CP-less OFDM.
* **Lock behaviour of the first timing study** (N = 32, the single-pilot `symgen` signal: PRBS pilot at k = 4,
  16-QAM at k = 5, ≈ 48 LSB each; 200 random start phases per SNR; `study_results/final.json`):

  | CP | 10 dB: locked / median / p95 | 20 dB: median / p95 | 30 dB: p95 | unclean locks | false locks (noise / no-CP) |
  | --- | --- | --- | --- | --- | --- |
  | 8 | 93.5 % / 513 / 1700 sym | 288 / 561 sym (0.36 / 0.70 ms) | 545 sym | 0 / 600 | 0 in 141 ms / 0 in 135 ms |
  | 4 | 91 % / 897 / 2790 sym | 657 / 1120 sym (0.74 / 1.26 ms) | 1120 sym | 0 / 600 | 0 in 141 ms / 0 in 135 ms |

  ("locked" = within three full hypothesis sweeps; one symbol = 40 or 36 samples at 32 MS/s.)
* **Tracking** at ±100 ppm, 10–30 dB, ≥ 10⁴ locked symbols: 0 unlocks, 0 unclean windows, net
  slips = accumulated drift ± 4 (N = 32: |e| ≤ 2.8 samples at CP = 8 and ≤ 1.6 at CP = 4; N = 16: H = 8, M = 16
  re-validated in `study_results/n16_hm.json`). As built, ±300 ppm at 20 dB gives 0 unlocks; −400 ppm unlocks
  (`study_results/n16_edges.json`).
* **Signal loss:** LOCK drops ≤ 2 tracking cycles + K symbols after the signal disappears (as built: 74–130
  symbols over 20 runs, `n16_edges.json`; N = 32: 67 symbols), and the chip re-acquires when the signal returns.

## System (`system_model.py`): the pins of docs/info.md

* **Tail on `clk_dp`, two plain registers:** `q_hold <= Q`, `o <= slot even ? I : q_hold`,
  and `uo_out = o`.
  - During the run cycle with cnt = c (ph = c when locked), the bus shows byte c of the frame
    captured in the previous symbol. A locked frame occupies the N run cycles ph = 0 … N−1, with SOF
    on byte 0.
  - During the CP pause the bus is **don't-care** (it already shows the next frame's byte 0) and
    VALID = 0.
  - Byte c: c even → I of slot c (bin bitrev(c)); c odd → Q of slot c − 1.
* **Pin registers** are computed from the *next* state (ph_next, run_next, LOCK_next), so they line
  up with the byte on the bus:
  - SOF = run & (ph mod N == 0).
  - VALID = run & emit_ok & LOCK.
  - DBG = slip pulse. LOCK = the TRACK state.
* **Latency:** from a frame's first sample to its byte 0 on the bus is N datapath clocks plus the CP
  pause in between, plus the I/O registers: 20 clocks as built (measured through the pins, `test_top`).
* **VALID bookkeeping:**
  - cap_ok ← LOCK at every symbol start, cleared by a slip decided after that frame's capture. The
    frame in flight at a slip is the one captured just before it, so exactly one frame is invalidated
    per slip.
  - emit_ok ← cap_ok when its emission starts.
* **SOF before lock** pulses at ph = 0 and ph = N (free-running counter).
* **N = 16, as built:** CFG_CP has no function there (CP = N/4 = 4 fixed). `CPTiming`
  refuses N = 16 with cfg_cp = 1 (CP = 2 is unsupported by the window/slip scheme).

## Mini-OFDM loop: `signal_gen.py`, `demod.py`, `ofdm_loop.py`

* **v1 signal (the acceptance signal):**
  - As built N = 16, CP = 4, symbol = 20 samples (`OFDMConfig(N=32, cfg_cp=…)` still gives the N = 32 signal with
    CP = N/4 or N/8).
  - Pilots: BPSK at **k = 4 and k = 6**. Signs from PRBS7 (x⁷ + x⁶ + 1, seed 0x7F, period 127) at
    offsets 0 and 63: sign of pilot p in symbol j = 1 − 2·b[(j + o_p) mod 127].
  - Data: Gray 16-QAM at **k = 5**.
  - Per-carrier amplitude a = 64/√M = **36.95 LSB** (peak 100 LSB). A carrier at bin k contributes
    a·Re(c·e^{j2πk(u−CP)/N}); the chip's 2/N FFT shows a·c (a·c/2 with the 1/N reference pipeline).
  - `LEGACY_1P` is the earlier single-pilot signal (pilot 4, data 5, 45.25 LSB).
  - ppm: the RX samples the ideal continuous waveform at t_n = phase0 + n(1 + ppm·1e-6).
* **Demodulator inputs:** only the pins and the bus protocol (docs/info.md).
  - Frames are SOF-led runs of N VALID bytes.
  - Symbol index = SOF count inside a LOCK interval. A slip shows up as an SOF-to-SOF period ≠ L.
  - The PRBS phase comes from a joint sign correlation over both pilots and their known offsets.
* **Acceptance equaliser (no decision feedback):**
  1. h_p = P_p / s_p per pilot, averaged over a centred W = 16 frame window inside a slip segment.
  2. Slope per bin = θ_d + wrap(arg h₆ − arg h₄ − 2θ_d)/2, with θ_d = −2π(CP/2)/N (unambiguous for |e| < 8).
  3. Bin k: phase arg h₄ + (k − 4)·slope; gain interpolated from |h₄|, |h₆|.
  - Optional (`dd=True`): decision-directed residual rotation, 32-frame window, 3 iterations
    (ramp-aware in single-pilot mode). It is slightly worse than plain two-pilot at moderate SNR, so it
    stays off.
* **Results as built** (N = 16, gain 2/N; `study_results/precision_ber.json`, option b, and the pins in
  `test/test_ofdm.py`): 16-QAM BER 1e−3 at 14.34 dB (model, 0 ppm; pins 14.63 dB; theory
  14.11 dB), 64-QAM at 22.24 dB (pins 22.38 dB; theory 20.95 dB), 64-QAM EVM floor −28.2 dB.
* **Results of the first loop** (N = 32, gain 1/N; `study_results/ber_curves.json`):
  - Losses at BER 1e-3: theory → float ≈ 0 dB; quantisation 1.4–1.8 dB; timing 0.0–0.2 dB.
  - Pilot averaging +1.3–1.4 dB.
  - Third carrier −2.15 dB SNR, net +0.4 dB vs the best single-pilot demodulator.
  - BER 0 at 30 dB over phases / DC / ±100 ppm; EVM floor −20.5 dB (chip fixed point).
  - 64-QAM is quantisation-limited.
