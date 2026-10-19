"""SignalChain — ready-made demonstrations.

Each demo returns a self-contained dictionary of the signal, its processing
chain and the numbers that prove the point, so the CLI and web UI can both
drive the same scenarios.
"""

from __future__ import annotations

from typing import Dict, List

from . import analysis, dsp, filters, sampling, signals


def _common(fs: float, n: int, f_signal: float, x: List[float]) -> Dict:
    spec = analysis.measure_spectrum(x, fs, f_signal=f_signal)
    freqs, psd = dsp.power_spectrum(x, fs)
    return {
        "fs": fs,
        "n": n,
        "fSignal": f_signal,
        "signal": [round(v, 6) for v in x],
        "spectrum": {"frequencies": [round(f, 2) for f in freqs],
                     "psdDb": [round(dsp.db(p), 2) for p in psd]},
        "metrics": spec,
    }


def demo_adc_resolution() -> Dict:
    """Show how ENOB tracks ADC bit depth."""
    fs, n, f0 = 100_000.0, 8192, 1000.0
    x = signals.sine(f0, fs, n, amp=1.0)
    results = []
    for bits in (6, 8, 10, 12, 14):
        _, v = signals.quantize(x, bits=bits, vref=3.3, bipolar=True)
        spec = analysis.measure_spectrum(v, fs, f_signal=f0)
        results.append({"bits": bits, "enob": spec["enob"],
                        "sinadDb": spec["sinadDb"], "snrDb": spec["snrDb"]})
    base = _common(fs, n, f0, signals.quantize(x, bits=12, vref=3.3, bipolar=True)[1])
    base["bitSweep"] = results
    return base


def demo_aliasing() -> Dict:
    """A 60 kHz tone sampled at 100 kHz reappears at 40 kHz."""
    fs, n = 100_000.0, 2048
    true_f = 60_000.0
    x = signals.sine(true_f, fs, n)
    alias = sampling.alias_frequency(true_f, fs)
    base = _common(fs, n, true_f, x)
    base["trueFrequency"] = true_f
    base["aliasFrequency"] = alias
    base["note"] = ("%.1f kHz sampled at %.1f kHz aliases to %.1f kHz"
                    % (true_f / 1000, fs / 1000, alias / 1000))
    return base


def _math_sin(freq: float, fs: float, i: int) -> float:
    import math
    return math.sin(2.0 * math.pi * freq * i / fs)


def demo_filter_design() -> Dict:
    """A noisy multi-tone cleaned by a Butterworth low-pass."""
    fs, n = 20_000.0, 8192
    clean = [_math_sin(500, fs, i) + 0.5 * _math_sin(1200, fs, i) for i in range(n)]
    noisy = [c + nz for c, nz in zip(clean, signals.white_noise(n, rms=0.3, seed=7))]
    chain = filters.SignalChain([filters.make_stage("iir_lowpass", fs, cutoff=700, order=4)], fs)
    filtered = chain.apply(noisy)
    base = _common(fs, n, 500.0, noisy)
    base["clean"] = [round(v, 6) for v in clean]
    base["filtered"] = [round(v, 6) for v in filtered]
    base["response"] = chain.response(512)
    base["before"] = analysis.measure_spectrum(noisy, fs, f_signal=500.0)
    base["after"] = analysis.measure_spectrum(filtered, fs, f_signal=500.0)
    return base


def demo_jitter() -> Dict:
    """Sample-clock jitter hurts high-frequency signals most."""
    fs, n = 100_000.0, 8192
    jitter_ps = 50.0
    f_low, f_high = 1000.0, 40_000.0
    s_low = signals.add_jitter(signals.sine(f_low, fs, n), fs, jitter_ps, seed=3)
    s_high = signals.add_jitter(signals.sine(f_high, fs, n), fs, jitter_ps, seed=3)
    base = _common(fs, n, f_low, s_low)
    base["lowFreq"] = {"freq": f_low, "snrDb": analysis.measure_spectrum(s_low, fs, f_signal=f_low)["snrDb"]}
    base["highFreq"] = {"freq": f_high, "snrDb": analysis.measure_spectrum(s_high, fs, f_signal=f_high)["snrDb"]}
    base["jitterPs"] = jitter_ps
    return base


def demo_thd() -> Dict:
    """A clipped/quantised sine shows rising harmonics and THD."""
    fs, n, f0 = 100_000.0, 8192, 1000.0
    x = signals.sine(f0, fs, n, amp=1.5)
    clipped = [max(-1.0, min(1.0, v)) for v in x]  # hard clip at +/-1
    base = _common(fs, n, f0, clipped)
    base["thd"] = analysis.total_harmonic_distortion(clipped, fs, f0)
    return base


def demo_intermodulation() -> Dict:
    """Two tones produce IMD3 products in a nonlinearity."""
    fs, n = 100_000.0, 8192
    f1, f2 = 10_000.0, 12_000.0
    imd = analysis.intermodulation(f1, f2, fs, n)
    base = {"fs": fs, "n": n, "f1": f1, "f2": f2, "imd": imd,
            "products": imd["products"]}
    return base


DEMOS = [
    ("adc", "ADC 分辨率与 ENOB", demo_adc_resolution),
    ("alias", "混叠演示", demo_aliasing),
    ("filter", "滤波设计", demo_filter_design),
    ("jitter", "采样时钟抖动", demo_jitter),
    ("thd", "谐波失真 THD", demo_thd),
    ("imd", "互调失真 IMD", demo_intermodulation),
]


def demo_list() -> List[Dict]:
    return [{"id": i, "name": n} for i, n, _ in DEMOS]


def build_demo(demo_id: str) -> Dict:
    for i, n, fn in DEMOS:
        if i == demo_id:
            data = fn()
            data["id"] = i
            data["name"] = n
            return data
    raise ValueError("unknown demo: %r" % demo_id)
