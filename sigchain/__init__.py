"""SignalChain — a local DSP workbench for embedded / optoelectronic systems.

Design a signal, push it through an ADC model, filter it, and measure its
spectral figures of merit (SINAD, ENOB, THD, SFDR, IMD) — all in pure Python
with no third-party dependencies.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "SignalChain contributors"

from . import dsp, signals, sampling, filters, analysis, demos
from .dsp import fft, ifft, fft_magnitude, power_spectrum, window, freqz, db
from .signals import (sine, cosine, square, sawtooth, triangle, chirp,
                      white_noise, pink_noise, quantize)
from .sampling import sample_continuous, alias_frequency
from .filters import FilterStage, SignalChain, make_stage
from .analysis import (measure_spectrum, total_harmonic_distortion,
                       intermodulation, measure_snr_simple)

__all__ = [
    "__version__",
    "dsp", "signals", "sampling", "filters", "analysis", "demos",
    "fft", "ifft", "fft_magnitude", "power_spectrum", "window", "freqz", "db",
    "sine", "cosine", "square", "sawtooth", "triangle", "chirp",
    "white_noise", "pink_noise", "quantize",
    "sample_continuous", "alias_frequency",
    "FilterStage", "SignalChain", "make_stage",
    "measure_spectrum", "total_harmonic_distortion", "intermodulation",
    "measure_snr_simple",
]
