"""SignalChain — sampling, aliasing and decimation demonstrations.

These helpers make the classic sampling-theorem mistakes visible: what
happens when you under-sample a tone, and why you need an anti-alias filter
before decimation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from . import dsp, signals


def sample_continuous(freq: float, fs: float, n: int, amp: float = 1.0,
                      phase: float = 0.0) -> Tuple[List[float], List[float]]:
    """Sample a continuous tone at ``fs``.

    Returns (time, samples).  If ``freq > fs/2`` the result is aliased — the
    returned samples are identical to those of the alias frequency
    ``|freq - fs * round(freq/fs)|``.
    """
    t = signals.time_axis(n, fs)
    w = 2.0 * math.pi * freq / fs
    return t, [amp * math.sin(w * i + phase) for i in range(n)]


def alias_frequency(freq: float, fs: float) -> float:
    """The apparent frequency a tone at ``freq`` appears at after sampling."""
    if fs <= 0:
        return 0.0
    f = freq
    while f >= fs / 2.0:
        f = abs(f - fs)
    return f


def downsample_no_filter(x: List[float], factor: int) -> List[float]:
    """Decimate without an anti-alias filter (the textbook mistake)."""
    return dsp.decimate(x, factor)


def downsample_filtered(x: List[float], factor: int,
                        fs: float) -> List[float]:
    """Decimate by ``factor`` with a proper anti-alias FIR first."""
    # low-pass at the new Nyquist (with a little margin)
    new_nyq = fs / (2.0 * factor)
    h = dsp.fir_lowpass(new_nyq * 0.9, fs, 65, "hamming")
    y = dsp.convolve(x, h)
    delay = len(h) // 2
    y = y[delay:delay + len(x)]
    return dsp.decimate(y, factor)


def oversample_and_filter(x: List[float], factor: int,
                          fs: float) -> List[float]:
    """Interpolate by ``factor`` with a reconstruction low-pass."""
    y = dsp.upsample(x, factor)
    h = dsp.fir_lowpass(fs / 2.0 * 0.9, fs * factor, 65, "hamming")
    z = dsp.convolve(y, h)
    delay = len(h) // 2
    return z[delay:delay + len(x) * factor]


def snr_vs_bits(bits_range: Tuple[int, int] = (4, 16)) -> List[Tuple[int, float]]:
    """Theoretical SNR (6.02N + 1.76) across bit depths."""
    return [(n, 6.02 * n + 1.76) for n in range(bits_range[0], bits_range[1] + 1)]
