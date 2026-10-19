"""SignalChain — spectral figures of merit.

SINAD, SNR, ENOB, THD, SFDR and intermodulation measurement on a real
signal, built on the hand-written FFT in ``dsp``.  These are the quantities
that turn "the spectrum looks noisy" into a number you can put in a spec.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import dsp


def _find_peak(freqs: Sequence[float], mags: Sequence[float],
               f_lo: float, f_hi: float) -> Tuple[float, float]:
    """Largest magnitude within [f_lo, f_hi]; returns (freq, mag)."""
    best_f, best_m = 0.0, -1.0
    for f, m in zip(freqs, mags):
        if f_lo <= f <= f_hi and m > best_m:
            best_f, best_m = f, m
    return best_f, best_m


def _sum_power(freqs: Sequence[float], mags: Sequence[float],
               f_lo: float, f_hi: float, exclude: Sequence[float] = ()) -> float:
    """Sum of squared magnitudes (power) over a band, excluding narrow bins."""
    total = 0.0
    for f, m in zip(freqs, mags):
        if f_lo <= f <= f_hi:
            if any(abs(f - ex) < 1e-9 for ex in exclude):
                continue
            total += m * m
    return total


def measure_spectrum(x: Sequence[float], fs: float,
                     window_name: str = "blackman_harris",
                     f_signal: Optional[float] = None,
                     num_tones: int = 1) -> Dict:
    """Compute SINAD / SNR / ENOB / THD / SFDR from a real signal.

    The fundamental(s) are located automatically unless ``f_signal`` is
    given.  Harmonics are excluded from the noise floor up to the 10th
    (or fewer if they fall past Nyquist).  A windowed, coherent-gain-corrected
    FFT is used so that amplitude measurements are accurate.
    """
    n = len(x)
    if n < 16:
        raise ValueError("need at least 16 samples")

    w = dsp.window(window_name, n)
    cg = dsp.window_coherent_gain(w)
    xw = [a * b for a, b in zip(x, w)]
    freqs, mags = dsp.fft_magnitude(xw, fs)
    mags = [m / cg for m in mags]  # correct for window coherent gain

    nyquist = fs / 2.0
    freqs = freqs[: n // 2 + 1]
    mags = mags[: n // 2 + 1]

    # locate the fundamental tone(s)
    if f_signal is not None:
        fundamentals = []
        for k in range(num_tones):
            f_k = f_signal * (k + 1)
            if f_k >= nyquist:
                break
            ff, mm = _find_peak(freqs, mags, f_k - fs / n * 2, f_k + fs / n * 2)
            fundamentals.append((ff, mm))
    else:
        # assume the largest peak above a small fraction of the spectrum is the signal
        peak_idx = max(range(len(mags)), key=lambda i: mags[i])
        fundamentals = [(freqs[peak_idx], mags[peak_idx])]

    if not fundamentals:
        return {"sinadDb": None, "snrDb": None, "enob": None,
                "thdDb": None, "sfdrDb": None, "fundamental": None}

    f0, a0 = fundamentals[0]
    signal_power = sum(m * m for _, m in fundamentals)

    # bin width in Hz, and the exclusion half-width in bins.  Blackman-Harris
    # has an 8-bin main lobe, so the fundamental + harmonics must be blanked
    # out over ~10 bins each or their skirt energy is counted as noise.
    bin_hz = fs / n
    blank_bins = 10.0

    # harmonics
    harmonic_freqs = []
    harmonic_power = 0.0
    for k in range(2, 11):
        fh = f0 * k
        if fh >= nyquist:
            break
        ff, mm = _find_peak(freqs, mags, fh - bin_hz * 2, fh + bin_hz * 2)
        harmonic_freqs.append(ff)
        harmonic_power += mm * mm

    # total power excluding DC, fundamental, and harmonics
    exclude = [f for f, _ in fundamentals] + harmonic_freqs
    noise_power = 0.0
    for f, m in zip(freqs, mags):
        if f < bin_hz * 0.5:  # skip DC bin
            continue
        if any(abs(f - ex) < blank_bins * bin_hz for ex in exclude):
            continue
        noise_power += m * m

    total_distortion = harmonic_power
    noise_plus_dist = noise_power + harmonic_power

    sinad = 10.0 * math.log10(signal_power / noise_plus_dist) if noise_plus_dist > 0 else float("inf")
    snr = 10.0 * math.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
    enob = (sinad - 1.76) / 6.02 if math.isfinite(sinad) else float("inf")
    thd = 10.0 * math.log10(harmonic_power / signal_power) if signal_power > 0 and harmonic_power > 0 else float("-inf")

    # SFDR: ratio of fundamental to largest spurious (non-harmonic, non-DC)
    spur_mag = 0.0
    for f, m in zip(freqs, mags):
        if f < fs / n * 0.5:
            continue
        if any(abs(f - ex) < blank_bins * bin_hz for ex in exclude):
            continue
        spur_mag = max(spur_mag, m)
    sfdr = 20.0 * math.log10(a0 / spur_mag) if spur_mag > 0 else float("inf")

    return {
        "sinadDb": round(sinad, 2),
        "snrDb": round(snr, 2),
        "enob": round(enob, 2),
        "thdDb": round(thd, 2),
        "sfdrDb": round(sfdr, 2),
        "fundamentalHz": round(f0, 3),
        "fundamentalMag": round(a0, 5),
        "window": window_name,
        "numHarmonics": len(harmonic_freqs),
    }


def measure_snr_simple(signal: Sequence[float], noise: Sequence[float]) -> float:
    """Time-domain SNR = 10 log10(var(signal)/var(noise))."""
    def var(v):
        m = sum(v) / len(v)
        return sum((x - m) ** 2 for x in v) / len(v)
    sv, nv = var(signal), var(noise)
    if nv <= 0:
        return float("inf")
    return 10.0 * math.log10(sv / nv)


def total_harmonic_distortion(x: Sequence[float], fs: float,
                              f_signal: float,
                              window_name: str = "blackman_harris") -> Dict:
    """Convenience wrapper returning THD in dBc and percent."""
    r = measure_spectrum(x, fs, window_name, f_signal)
    thd_db = r["thdDb"]
    thd_pct = 100.0 * 10 ** (thd_db / 20.0) if math.isfinite(thd_db) else 0.0
    return {"thdDb": thd_db, "thdPct": round(thd_pct, 4), **r}


def intermodulation(f1: float, f2: float, fs: float, n: int,
                    amp: float = 1.0, window_name: str = "blackman_harris") -> Dict:
    """Two-tone intermodulation: IMD3 in dBc relative to either tone."""
    from . import signals
    x = [amp * (math.sin(2 * math.pi * f1 * i / fs)
                + math.sin(2 * math.pi * f2 * i / fs)) for i in range(n)]
    w = dsp.window(window_name, n)
    cg = dsp.window_coherent_gain(w)
    xw = [a * b for a, b in zip(x, w)]
    freqs, mags = dsp.fft_magnitude(xw, fs)
    mags = [m / cg for m in mags]

    def peak(f):
        return _find_peak(freqs, mags, f - fs / n * 2, f + fs / n * 2)

    tone1 = peak(f1)[1]
    tone2 = peak(f2)[1]
    # third-order products: 2f1-f2 and 2f2-f1
    imd3_f = [abs(2 * f1 - f2), abs(2 * f2 - f1)]
    imd3 = max(peak(f)[1] for f in imd3_f if f < fs / 2.0)
    ref = max(tone1, tone2)
    imd3_db = 20.0 * math.log10(imd3 / ref) if ref > 0 and imd3 > 0 else float("-inf")
    return {
        "imd3Db": round(imd3_db, 2),
        "f1": f1, "f2": f2,
        "products": imd3_f,
        "toneMag": round(ref, 5),
    }
