"""SignalChain — a composable filter chain.

A ``FilterStage`` wraps a single FIR or IIR filter; a ``SignalChain`` applies
a sequence of stages in order and can report the composite frequency
response.  This is the "design a chain" entry point that the CLI and web UI
both drive.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import dsp


class FilterStage:
    """One filter: FIR (a=[1]) or IIR, with a human-readable label."""

    def __init__(self, b: Sequence[float], a: Sequence[float],
                 kind: str, name: str = "", fs: float = 1.0,
                 params: Optional[dict] = None):
        self.b = list(b)
        self.a = list(a)
        self.kind = kind
        self.name = name or kind
        self.fs = fs
        self.params = params or {}

    def apply(self, x: Sequence[float]) -> List[float]:
        return dsp.lfilter(self.b, self.a, x)

    def response(self, worN: int = 512) -> Dict:
        freqs, H = dsp.freqz(self.b, self.a, worN, self.fs)
        return {
            "frequencies": [round(f, 4) for f in freqs],
            "magnitudeDb": [round(dsp.db(abs(h)), 4) for h in H],
            "phaseDeg": [round(p, 4) for p in dsp.phase_deg(H)],
        }

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "b": [round(v, 8) for v in self.b],
            "a": [round(v, 8) for v in self.a],
            "params": self.params,
        }


def make_stage(kind: str, fs: float, **kwargs) -> FilterStage:
    """Construct a filter stage from a kind string and parameters.

    Kinds: ``fir_lowpass``, ``fir_highpass``, ``fir_bandpass``,
    ``fir_bandstop``, ``iir_lowpass``, ``iir_highpass``, ``iir_bandpass``.
    """
    if kind == "fir_lowpass":
        b = dsp.fir_lowpass(kwargs["cutoff"], fs, kwargs.get("taps", 65),
                            kwargs.get("window", "hamming"))
        return FilterStage(b, [1.0], kind, kwargs.get("name", "FIR low-pass"),
                           fs, {"cutoff": kwargs["cutoff"], "taps": len(b)})
    if kind == "fir_highpass":
        b = dsp.fir_highpass(kwargs["cutoff"], fs, kwargs.get("taps", 65),
                             kwargs.get("window", "hamming"))
        return FilterStage(b, [1.0], kind, kwargs.get("name", "FIR high-pass"),
                           fs, {"cutoff": kwargs["cutoff"], "taps": len(b)})
    if kind == "fir_bandpass":
        b = dsp.fir_bandpass(kwargs["fl"], kwargs["fh"], fs,
                             kwargs.get("taps", 65), kwargs.get("window", "hamming"))
        return FilterStage(b, [1.0], kind, kwargs.get("name", "FIR band-pass"),
                           fs, {"fl": kwargs["fl"], "fh": kwargs["fh"], "taps": len(b)})
    if kind == "fir_bandstop":
        b = dsp.fir_bandstop(kwargs["fl"], kwargs["fh"], fs,
                             kwargs.get("taps", 65), kwargs.get("window", "hamming"))
        return FilterStage(b, [1.0], kind, kwargs.get("name", "FIR band-stop"),
                           fs, {"fl": kwargs["fl"], "fh": kwargs["fh"], "taps": len(b)})
    if kind == "iir_lowpass":
        b, a = dsp.butter_lowpass(kwargs["cutoff"], fs, kwargs.get("order", 4))
        return FilterStage(b, a, kind, kwargs.get("name", "IIR low-pass"),
                           fs, {"cutoff": kwargs["cutoff"], "order": kwargs.get("order", 4)})
    if kind == "iir_highpass":
        b, a = dsp.butter_highpass(kwargs["cutoff"], fs, kwargs.get("order", 4))
        return FilterStage(b, a, kind, kwargs.get("name", "IIR high-pass"),
                           fs, {"cutoff": kwargs["cutoff"], "order": kwargs.get("order", 4)})
    if kind == "iir_bandpass":
        b, a = dsp.butter_bandpass(kwargs["fl"], kwargs["fh"], fs, kwargs.get("order", 4))
        return FilterStage(b, a, kind, kwargs.get("name", "IIR band-pass"),
                           fs, {"fl": kwargs["fl"], "fh": kwargs["fh"], "order": kwargs.get("order", 4)})
    raise ValueError("unknown filter kind: %r" % kind)


class SignalChain:
    """An ordered list of filter stages applied in sequence."""

    def __init__(self, stages: Optional[Sequence[FilterStage]] = None,
                 fs: float = 1.0):
        self.stages: List[FilterStage] = list(stages or [])
        self.fs = fs

    def add(self, stage: FilterStage) -> "SignalChain":
        self.stages.append(stage)
        return self

    def apply(self, x: Sequence[float]) -> List[float]:
        y = list(x)
        for stage in self.stages:
            y = stage.apply(y)
        return y

    def response(self, worN: int = 512) -> Dict:
        """Composite frequency response (cascade of all stages)."""
        freqs = None
        H = None
        for stage in self.stages:
            f, h = dsp.freqz(stage.b, stage.a, worN, self.fs)
            if H is None:
                freqs, H = f, h
            else:
                H = [a * b for a, b in zip(H, h)]
        if freqs is None:
            freqs = [self.fs * i / (2 * (worN - 1)) for i in range(worN)]
            H = [complex(1.0, 0.0)] * worN
        return {
            "frequencies": [round(v, 4) for v in freqs],
            "magnitudeDb": [round(dsp.db(abs(h)), 4) for h in H],
            "phaseDeg": [round(p, 4) for p in dsp.phase_deg(H)],
        }

    def to_dict(self) -> dict:
        return {"fs": self.fs, "stages": [s.to_dict() for s in self.stages]}

    def __len__(self) -> int:
        return len(self.stages)
