"""SignalChain — signal synthesis and ADC modelling.

Deterministic waveform generators (sine, square, saw, chirp, multi-tone,
AM/FM), noise sources (white, pink) and a quantiser that models a real
analogue-to-digital converter: full-scale range, bit depth, offset, gain
error, clock jitter and dithering.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

PI = math.pi
TAU = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Waveform generators
# --------------------------------------------------------------------------- #
def time_axis(n: int, fs: float) -> List[float]:
    return [i / fs for i in range(n)]


def sine(freq: float, fs: float, n: int, amp: float = 1.0,
         phase: float = 0.0, dc: float = 0.0) -> List[float]:
    w = TAU * freq / fs
    return [dc + amp * math.sin(w * i + phase) for i in range(n)]


def cosine(freq: float, fs: float, n: int, amp: float = 1.0,
           phase: float = 0.0, dc: float = 0.0) -> List[float]:
    w = TAU * freq / fs
    return [dc + amp * math.cos(w * i + phase) for i in range(n)]


def square(freq: float, fs: float, n: int, amp: float = 1.0,
           duty: float = 0.5, dc: float = 0.0) -> List[float]:
    period = fs / freq
    out = []
    for i in range(n):
        ph = (i % period) / period
        out.append(dc + (amp if ph < duty else -amp))
    return out


def sawtooth(freq: float, fs: float, n: int, amp: float = 1.0,
             dc: float = 0.0) -> List[float]:
    period = fs / freq
    return [dc + amp * (2.0 * ((i % period) / period) - 1.0) for i in range(n)]


def triangle(freq: float, fs: float, n: int, amp: float = 1.0,
             dc: float = 0.0) -> List[float]:
    period = fs / freq
    out = []
    for i in range(n):
        ph = (i % period) / period
        out.append(dc + amp * (4.0 * abs(ph - 0.5) - 1.0))
    return out


def chirp(f0: float, f1: float, fs: float, n: int, amp: float = 1.0,
          method: str = "linear") -> List[float]:
    """Sweep from f0 to f1 over n samples (linear or logarithmic)."""
    t = time_axis(n, fs)
    T = t[-1] if t else 0.0
    out = []
    for i, ti in enumerate(t):
        if method == "log" and f0 > 0 and f1 > 0:
            k = math.log(f1 / f0) / T if T > 0 else 0.0
            f = f0 * math.exp(k * ti)
            phase = TAU * f0 * (math.exp(k * ti) - 1.0) / k if k else TAU * f0 * ti
        else:
            k = (f1 - f0) / T if T > 0 else 0.0
            phase = TAU * (f0 * ti + 0.5 * k * ti * ti)
        out.append(amp * math.sin(phase))
    return out


def multi_tone(freqs: Sequence[float], fs: float, n: int,
               amp: float = 1.0) -> List[float]:
    """Sum of equal-amplitude sinusoids (normalised to ``amp``)."""
    a = amp / max(1, len(freqs))
    return [a * sum(math.sin(TAU * f * i / fs) for f in freqs) for i in range(n)]


def am_modulate(carrier: float, mod: float, fs: float, n: int,
                depth: float = 0.8, amp: float = 1.0) -> List[float]:
    wc = TAU * carrier / fs
    wm = TAU * mod / fs
    return [amp * (1.0 + depth * math.sin(wm * i)) * math.sin(wc * i)
            for i in range(n)]


def fm_modulate(carrier: float, mod: float, fs: float, n: int,
                deviation: float, amp: float = 1.0) -> List[float]:
    out = []
    phase = 0.0
    wm = TAU * mod / fs
    wc = TAU * carrier / fs
    kf = TAU * deviation / fs
    for i in range(n):
        out.append(amp * math.sin(phase))
        phase += wc + kf * math.sin(wm * i)
    return out


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def white_noise(n: int, rms: float = 0.01, seed: Optional[int] = None) -> List[float]:
    """Gaussian white noise of a given RMS level."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, rms) for _ in range(n)]


def pink_noise(n: int, rms: float = 0.01, seed: Optional[int] = None) -> List[float]:
    """Approximate 1/f (pink) noise via the Voss–McCartney algorithm."""
    rng = random.Random(seed)
    rows = 12
    counters = [0] * rows
    prev = 0.0
    out = []
    for _ in range(n):
        # update the row counters (each row flips at 2^row intervals)
        for r in range(rows):
            counters[r] += 1
            if counters[r] >= (1 << r):
                counters[r] = 0
        # white + running average of low-frequency rows gives 1/f
        white = rng.gauss(0.0, 1.0)
        pink = sum(rng.gauss(0.0, 1.0) if counters[r] == 0 else 0.0
                   for r in range(rows))
        out.append(white + pink)
    # normalise to requested RMS
    r = (sum(x * x for x in out) / n) ** 0.5
    scale = rms / r if r else 0.0
    return [x * scale for x in out]


# --------------------------------------------------------------------------- #
# ADC modelling
# --------------------------------------------------------------------------- #
def quantize(x: Sequence[float], bits: int = 12, vref: float = 3.3,
             bipolar: bool = False, offset: float = 0.0,
             gain_error: float = 0.0, dither: bool = False,
             seed: Optional[int] = None) -> Tuple[List[int], List[float]]:
    """Model an ADC.

    Returns ``(codes, voltages)`` where voltages is the reconstructed
    (quantised) analogue signal.  ``bipolar`` selects a signed mid-tread
    quantiser spanning [-vref, +vref]; otherwise unsigned [0, vref].
    """
    levels = 1 << bits
    if bipolar:
        lo, hi = -vref, vref
    else:
        lo, hi = 0.0, vref
    step = (hi - lo) / levels
    rng = random.Random(seed) if seed is not None else random.Random()

    codes = []
    voltages = []
    for v in x:
        # gain error (LSB-referred multiplicative error)
        vg = v * (1.0 + gain_error)
        # offset (in volts)
        vg = vg + offset
        if dither:
            vg = vg + rng.uniform(-0.5 * step, 0.5 * step)
        # clamp and quantise
        code = int(round((vg - lo) / step))
        code = max(0, min(levels - 1, code))
        codes.append(code)
        voltages.append(lo + code * step + 0.5 * step)
    return codes, voltages


def adc_snr_theoretical(bits: int) -> float:
    """Theoretical SNR of an ideal N-bit ADC (6.02N + 1.76 dB)."""
    return 6.02 * bits + 1.76


def add_jitter(signal: Sequence[float], fs: float, jitter_ps: float,
               seed: Optional[int] = None) -> List[float]:
    """Model sample-clock jitter as additive slope-scaled noise.

    Jitter error = slope(t) * dt, with dt a Gaussian of the given RMS (ps).
    This reproduces the classic result that jitter SNR degrades fastest for
    high-frequency, high-slew-rate signals.
    """
    rng = random.Random(seed)
    dt_rms = jitter_ps * 1e-12
    out = []
    prev = signal[0]
    for i, v in enumerate(signal):
        slope = (v - prev) * fs  # volts per second (finite difference)
        out.append(v + slope * rng.gauss(0.0, dt_rms))
        prev = v
    return out
