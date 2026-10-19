"""SignalChain — core DSP primitives implemented from scratch.

Everything here runs on pure Python (no NumPy): a radix-2 iterative FFT,
window functions, FIR/IIR filter design, convolution, resampling and
frequency-domain measurements.

The convention throughout is ``exp(-i 2 pi k n / N)`` (the standard DFT
sign convention), so that a real cosine ``cos(2 pi f0 t)`` shows up as a
pair of positive-frequency and negative-frequency peaks.
"""

from __future__ import annotations

import cmath
import math
from typing import List, Optional, Sequence, Tuple

PI = math.pi
TAU = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# FFT
# --------------------------------------------------------------------------- #
def next_pow2(n: int) -> int:
    """Smallest power of two >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(x: Sequence[float]) -> List[complex]:
    """Radix-2 iterative Cooley–Tukey FFT of a real or complex sequence.

    The input is zero-padded to the next power of two.  Returns the complex
    spectrum of length ``N = next_pow2(len(x))``.
    """
    n = len(x)
    N = next_pow2(n)
    a = [complex(v, 0.0) if not isinstance(v, complex) else v for v in x]
    a += [0j] * (N - n)

    # bit-reversal permutation
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # butterflies
    length = 2
    while length <= N:
        ang = -TAU / length
        wlen = complex(math.cos(ang), math.sin(ang))
        half = length >> 1
        for start in range(0, N, length):
            w = 1.0 + 0j
            for k in range(start, start + half):
                u = a[k]
                v = a[k + half] * w
                a[k] = u + v
                a[k + half] = u - v
                w *= wlen
        length <<= 1
    return a


def fft_complex(x: Sequence[complex]) -> List[complex]:
    """Radix-2 iterative FFT that accepts a complex input directly."""
    n = len(x)
    N = next_pow2(n)
    a = list(x)
    a += [0j] * (N - n)

    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= N:
        ang = -TAU / length
        wlen = complex(math.cos(ang), math.sin(ang))
        half = length >> 1
        for start in range(0, N, length):
            w = 1.0 + 0j
            for k in range(start, start + half):
                u = a[k]
                v = a[k + half] * w
                a[k] = u + v
                a[k + half] = u - v
                w *= wlen
        length <<= 1
    return a


def ifft(x: Sequence[complex]) -> List[complex]:
    """Inverse FFT (scaled by 1/N)."""
    n = len(x)
    conj = [complex(c.real, -c.imag) for c in x]
    out = [complex(c.real, -c.imag) for c in fft_complex(conj)]
    return [c / n for c in out]


def rfft_freqs(n: int, fs: float) -> List[float]:
    """Positive-frequency bin centres for a real FFT of length n."""
    N = next_pow2(n)
    return [k * fs / N for k in range(N // 2 + 1)]


def fft_magnitude(x: Sequence[float], fs: float) -> Tuple[List[float], List[float]]:
    """Return (frequencies, single-sided magnitude spectrum) of a real signal."""
    N = next_pow2(len(x))
    X = fft(x)
    mags = []
    for k in range(N // 2 + 1):
        m = abs(X[k])
        if k == 0 or (k == N // 2 and N % 2 == 0):
            mags.append(m / N)
        else:
            mags.append(2.0 * m / N)
    return rfft_freqs(len(x), fs), mags


def power_spectrum(x: Sequence[float], fs: float) -> Tuple[List[float], List[float]]:
    """Single-sided power spectral density (linear power, not dB)."""
    N = next_pow2(len(x))
    X = fft(x)
    psd = []
    for k in range(N // 2 + 1):
        p = abs(X[k]) ** 2 / (N * N)
        if 0 < k < N // 2:
            p *= 2.0
        psd.append(p)
    return rfft_freqs(len(x), fs), psd


def db(x: float, floor: float = -300.0) -> float:
    """20*log10(|x|), clamped to a floor to avoid -inf."""
    if x <= 0.0:
        return floor
    return max(floor, 20.0 * math.log10(x))


# --------------------------------------------------------------------------- #
# Window functions
# --------------------------------------------------------------------------- #
def window(name: str, n: int, *args: float) -> List[float]:
    """Return a length-n window.  Supported: hann, hamming, blackman,
    blackman_harris, kaiser (beta), rectangular, flattop, bartlett."""
    name = name.lower().replace("-", "_").replace(" ", "_")
    if n <= 0:
        return []
    if n == 1:
        return [1.0]

    if name in ("rect", "rectangular", "boxcar", "none"):
        return [1.0] * n
    if name == "bartlett":
        return [1.0 - abs(2.0 * i / (n - 1) - 1.0) for i in range(n)]
    if name in ("hann", "hanning"):
        return [0.5 - 0.5 * math.cos(TAU * i / (n - 1)) for i in range(n)]
    if name == "hamming":
        return [0.54 - 0.46 * math.cos(TAU * i / (n - 1)) for i in range(n)]
    if name == "blackman":
        return [0.42 - 0.5 * math.cos(TAU * i / (n - 1))
                + 0.08 * math.cos(2 * TAU * i / (n - 1)) for i in range(n)]
    if name == "blackman_harris":
        a = (0.35875, 0.48829, 0.14128, 0.01168)
        return [a[0] - a[1] * math.cos(TAU * i / (n - 1))
                + a[2] * math.cos(2 * TAU * i / (n - 1))
                - a[3] * math.cos(3 * TAU * i / (n - 1)) for i in range(n)]
    if name == "flattop":
        a = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
        return [a[0] - a[1] * math.cos(TAU * i / (n - 1))
                + a[2] * math.cos(2 * TAU * i / (n - 1))
                - a[3] * math.cos(3 * TAU * i / (n - 1))
                + a[4] * math.cos(4 * TAU * i / (n - 1)) for i in range(n)]
    if name in ("kaiser", "kaiser_bessel"):
        beta = args[0] if args else 8.0
        return [kaiser_window_value(i, n, beta) for i in range(n)]
    raise ValueError("unknown window: %r" % name)


def _bessel_i0(x: float) -> float:
    """Modified Bessel function of the first kind, order 0 (series approx)."""
    total = 1.0
    term = 1.0
    k = 1
    while True:
        term *= (x * x) / (4.0 * k * k)
        total += term
        if term < 1e-15 * total:
            break
        k += 1
    return total


def kaiser_window_value(i: int, n: int, beta: float) -> float:
    """The i-th sample of a Kaiser window of length n with shape beta."""
    denom = _bessel_i0(beta)
    x = beta * math.sqrt(1.0 - ((2.0 * i / (n - 1)) - 1.0) ** 2)
    return _bessel_i0(x) / denom


def window_coherent_gain(w: Sequence[float]) -> float:
    """Mean of the window (amplitude correction factor for a coherent signal)."""
    return sum(w) / len(w)


def window_noise_bw(w: Sequence[float]) -> float:
    """Equivalent noise bandwidth in bins (ENBW = sum(w^2)/sum(w)^2)."""
    return sum(x * x for x in w) / (sum(w) ** 2)


# --------------------------------------------------------------------------- #
# FIR filters
# --------------------------------------------------------------------------- #
def sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    return math.sin(PI * x) / (PI * x)


def fir_lowpass(fc: float, fs: float, numtaps: int,
                window_name: str = "hamming") -> List[float]:
    """Windowed-sinc low-pass FIR.  fc is the -6 dB corner in Hz."""
    if numtaps % 2 == 0:
        numtaps += 1
    ft = fc / fs
    M = numtaps - 1
    h = []
    for n in range(numtaps):
        h.append(2.0 * ft * sinc(2.0 * ft * (n - M / 2.0)))
    w = window(window_name, numtaps)
    return [a * b for a, b in zip(h, w)]


def fir_highpass(fc: float, fs: float, numtaps: int,
                 window_name: str = "hamming") -> List[float]:
    lp = fir_lowpass(fc, fs, numtaps, window_name)
    lp[len(lp) // 2] -= 1.0  # spectral inversion
    return [-c for c in lp]


def fir_bandpass(fl: float, fh: float, fs: float, numtaps: int,
                 window_name: str = "hamming") -> List[float]:
    lp_h = fir_lowpass(fh, fs, numtaps, window_name)
    lp_l = fir_lowpass(fl, fs, numtaps, window_name)
    return [a - b for a, b in zip(lp_h, lp_l)]


def fir_bandstop(fl: float, fh: float, fs: float, numtaps: int,
                 window_name: str = "hamming") -> List[float]:
    lp_h = fir_lowpass(fh, fs, numtaps, window_name)
    lp_l = fir_lowpass(fl, fs, numtaps, window_name)
    bp = [a - b for a, b in zip(lp_h, lp_l)]
    bp[len(bp) // 2] += 1.0
    return bp


def convolve(x: Sequence[float], h: Sequence[float]) -> List[float]:
    """Linear convolution of two real sequences."""
    nx, nh = len(x), len(h)
    if nh > nx:
        x, h = h, x
        nx, nh = nh, nx
    out = [0.0] * (nx + nh - 1)
    for i, hi in enumerate(h):
        if hi == 0.0:
            continue
        for j in range(nx):
            out[i + j] += hi * x[j]
    return out


def lfilter(b: Sequence[float], a: Sequence[float],
            x: Sequence[float]) -> List[float]:
    """Direct-form II transposed IIR filter (scipy.lfilter compatible)."""
    a0 = a[0]
    b = [v / a0 for v in b]
    a = [v / a0 for v in a]
    nb = len(b)
    na = len(a)
    n = max(nb, na)
    b = b + [0.0] * (n - nb)
    a = a + [0.0] * (n - na)

    out = [0.0] * len(x)
    z = [0.0] * (n - 1)  # state vector of length n-1
    for i, xi in enumerate(x):
        y = b[0] * xi + z[0] if n > 1 else b[0] * xi
        out[i] = y
        for k in range(n - 2):
            z[k] = z[k + 1] + b[k + 1] * xi - a[k + 1] * y
        if n > 1:
            z[n - 2] = b[n - 1] * xi - a[n - 1] * y
    return out


# --------------------------------------------------------------------------- #
# IIR filters (Butterworth via bilinear transform)
# --------------------------------------------------------------------------- #
def _butter_poles_analog(order: int) -> List[complex]:
    """Analog Butterworth poles on the unit circle (left half plane)."""
    poles = []
    for m in range(order):
        ang = PI * (0.5 + (2 * m + 1) / (2 * order))
        poles.append(complex(math.cos(ang), math.sin(ang)))
    return poles


def butter_lowpass(cutoff: float, fs: float, order: int = 4) -> Tuple[List[float], List[float]]:
    """Butterworth low-pass IIR (b, a) via the bilinear transform.

    Uses the cascade-of-biquads method: design a normalised analog low-pass
    (cutoff = 1 rad/s), prewarp the digital cutoff to an analog frequency,
    scale the poles, transform each pole to the z-plane and assemble
    second-order sections.  The ``n`` analog zeros at infinity all map to
    ``z = -1``, giving the characteristic ``(1 + z^-1)^2`` numerator per
    biquad.  DC gain is normalised to exactly 1.0.
    """
    wd = 2.0 * PI * cutoff / fs
    wa = 2.0 * fs * math.tan(wd / 2.0)  # prewarped analog cutoff
    c = 2.0 * fs

    poles = [pp * wa for pp in _butter_poles_analog(order)]
    # sort so conjugate pairs are adjacent (descending imaginary part), with
    # the single real pole (odd order) at the end
    poles.sort(key=lambda p: (-abs(p.imag), p.real))
    b, a = [1.0], [1.0]
    i = 0
    while i < len(poles):
        if i + 1 < len(poles) and abs(poles[i].imag) > 1e-12:
            p1 = poles[i]
            p2 = poles[i + 1]  # conjugate
            z1 = (c + p1) / (c - p1)
            z2 = (c + p2) / (c - p2)
            nb = [1.0, 2.0, 1.0]
            na = [1.0, -(z1 + z2).real, (z1 * z2).real]
            b = convolve(b, nb)
            a = convolve(a, na)
            i += 2
        else:
            p1 = poles[i]
            z1 = (c + p1) / (c - p1)
            nb = [1.0, 1.0]
            na = [1.0, -z1.real]
            b = convolve(b, nb)
            a = convolve(a, na)
            i += 1

    # normalise DC gain to 1.0
    gain = sum(b) / sum(a)
    if gain != 0.0:
        b = [v / gain for v in b]
    return b, a


def butter_highpass(cutoff: float, fs: float, order: int = 4) -> Tuple[List[float], List[float]]:
    # lowpass -> highpass via z -> -z shifts the cutoff to fs/2 - cutoff, so
    # design the lowpass at the reflected frequency first
    reflected = fs / 2.0 - cutoff
    if reflected <= 0.0:
        reflected = cutoff  # degenerate guard
    b_lp, a_lp = butter_lowpass(reflected, fs, order)
    b = [c if i % 2 == 0 else -c for i, c in enumerate(b_lp)]
    a = [c if i % 2 == 0 else -c for i, c in enumerate(a_lp)]
    return b, a


def butter_bandpass(fl: float, fh: float, fs: float, order: int = 4) -> Tuple[List[float], List[float]]:
    lo = butter_lowpass(fh, fs, order)
    hi = butter_highpass(fl, fs, order)
    # cascade: convolve numerators, convolve denominators
    b = convolve(lo[0], hi[0])
    a = convolve(lo[1], hi[1])
    return b, a


# --------------------------------------------------------------------------- #
# Frequency response
# --------------------------------------------------------------------------- #
def freqz(b: Sequence[float], a: Sequence[float],
          worN: int = 512, fs: float = 1.0) -> Tuple[List[float], List[complex]]:
    """Frequency response H(e^{jw}) of an IIR/FIR filter."""
    freqs = []
    H = []
    for i in range(worN):
        w = PI * i / (worN - 1)
        z = complex(math.cos(w), -math.sin(w))
        num = sum(bj * (z ** -j) for j, bj in enumerate(b))
        den = sum(aj * (z ** -j) for j, aj in enumerate(a))
        H.append(num / den)
        freqs.append(fs * i / (2 * (worN - 1)))
    return freqs, H


def magnitude_db(H: Sequence[complex]) -> List[float]:
    return [db(abs(h)) for h in H]


def phase_deg(H: Sequence[complex], unwrap: bool = True) -> List[float]:
    ph = [math.degrees(cmath.phase(h)) for h in H]
    if not unwrap:
        return ph
    out = [ph[0]]
    for i in range(1, len(ph)):
        d = ph[i] - ph[i - 1]
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        out.append(out[-1] + d)
    return out


def group_delay(b: Sequence[float], a: Sequence[float],
                worN: int = 512, fs: float = 1.0) -> List[float]:
    """Group delay in samples (=-d(phase)/dw) approximated by finite difference."""
    _, H = freqz(b, a, worN, fs)
    ph = [cmath.phase(h) for h in H]
    gd = []
    for i in range(1, len(ph) - 1):
        dw = 2 * PI / worN
        dphi = ph[i + 1] - ph[i - 1]
        gd.append(-dphi / (2 * dw))
    gd = [gd[0]] + gd + [gd[-1]]
    return gd


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def upsample(x: Sequence[float], factor: int) -> List[float]:
    """Insert factor-1 zeros between samples."""
    out = [0.0] * (len(x) * factor)
    for i, v in enumerate(x):
        out[i * factor] = v
    return out


def decimate(x: Sequence[float], factor: int) -> List[float]:
    """Keep every factor-th sample (no anti-alias filtering)."""
    return [x[i] for i in range(0, len(x), factor)]


def resample_poly(x: Sequence[float], up: int, down: int) -> List[float]:
    """Rational resampling via zero-insert + FIR interpolation + decimation."""
    y = upsample(x, up)
    # simple low-pass interpolation filter
    h = fir_lowpass(0.5 / max(up, down), 1.0, 32, "kaiser")
    y = convolve(y, h)
    delay = len(h) // 2
    y = y[delay:]
    return decimate(y, down)
