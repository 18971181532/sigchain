"""Signals, ADC model, sampling theory and spectral measurements."""

import math
import unittest

from sigchain import analysis, dsp, sampling, signals


class TestGenerators(unittest.TestCase):
    def test_sine_amplitude_and_dc(self):
        # 625 Hz at 10 kS/s = 16 samples per cycle: a sample lands on the peak
        x = signals.sine(625, 10000, 4096, amp=2.0, dc=0.5)
        self.assertAlmostEqual(max(x), 2.5, places=3)
        self.assertAlmostEqual(min(x), -1.5, places=3)
        self.assertAlmostEqual(sum(x) / len(x), 0.5, places=2)

    def test_square_symmetry(self):
        x = signals.square(500, 10000, 2000, amp=1.0, duty=0.5)
        pos = sum(1 for v in x if v > 0)
        self.assertAlmostEqual(pos / len(x), 0.5, delta=0.05)

    def test_chirp_covers_range(self):
        x = signals.chirp(100, 2000, 20000, 4096)
        self.assertGreater(max(x), 0.99)
        self.assertLess(min(x), -0.99)

    def test_noise_rms(self):
        x = signals.white_noise(8192, rms=0.25, seed=42)
        rms = math.sqrt(sum(v * v for v in x) / len(x))
        self.assertAlmostEqual(rms, 0.25, delta=0.03)

    def test_pink_noise_rms(self):
        x = signals.pink_noise(8192, rms=0.1, seed=1)
        rms = math.sqrt(sum(v * v for v in x) / len(x))
        self.assertAlmostEqual(rms, 0.1, places=3)


class TestAdc(unittest.TestCase):
    def test_codes_within_range(self):
        x = signals.sine(1000, 100000, 1024, amp=1.0, dc=1.65)
        codes, v = signals.quantize(x, bits=10, vref=3.3)
        self.assertTrue(all(0 <= c < 1024 for c in codes))
        self.assertEqual(len(codes), len(v))

    def test_bipolar_codes_offset_binary(self):
        # bipolar mode uses offset-binary codes 0..levels-1 spanning [-vref, +vref]
        codes, _ = signals.quantize(signals.sine(500, 10000, 1024, amp=3.29),
                                    bits=8, vref=3.3, bipolar=True)
        self.assertTrue(all(0 <= c <= 255 for c in codes))
        self.assertGreater(max(codes), 200)   # near +FS
        self.assertLess(min(codes), 55)       # near -FS

    def test_more_bits_less_error(self):
        x = signals.sine(1000, 100000, 2048, amp=1.0)
        err = {}
        for bits in (6, 10, 14):
            _, v = signals.quantize(x, bits=bits, vref=3.3, bipolar=True)
            err[bits] = sum(abs(a - b) for a, b in zip(x, v)) / len(x)
        self.assertLess(err[14], err[10])
        self.assertLess(err[10], err[6])

    def test_theoretical_snr(self):
        self.assertAlmostEqual(signals.adc_snr_theoretical(12), 74.0, places=1)

    def test_jitter_hurts_high_frequency_more(self):
        fs = 100000.0
        x_low = signals.sine(1000, fs, 8192, amp=1.0)
        x_high = signals.sine(40000, fs, 8192, amp=1.0)
        snr_low = analysis.measure_spectrum(
            signals.add_jitter(x_low, fs, 50), fs, f_signal=1000)["snrDb"]
        snr_high = analysis.measure_spectrum(
            signals.add_jitter(x_high, fs, 50), fs, f_signal=40000)["snrDb"]
        self.assertGreater(snr_low, snr_high)


class TestSampling(unittest.TestCase):
    def test_alias_frequency(self):
        self.assertAlmostEqual(sampling.alias_frequency(60000, 100000), 40000, places=6)
        self.assertAlmostEqual(sampling.alias_frequency(1000, 100000), 1000, places=6)
        self.assertAlmostEqual(sampling.alias_frequency(90000, 100000), 10000, places=6)
        self.assertAlmostEqual(sampling.alias_frequency(120000, 100000), 20000, places=6)

    def test_decimate_length(self):
        x = list(range(100))
        self.assertEqual(len(sampling.downsample_no_filter(x, 4)), 25)

    def test_filtered_decimation_suppresses_alias(self):
        fs = 8000.0
        # 2.9 kHz tone, decimate by 4 (new Nyquist 1 kHz) -> aliases to 900 Hz
        x = signals.sine(2900, fs, 4096, amp=1.0)
        raw = sampling.downsample_no_filter(x, 4)
        filt = sampling.downsample_filtered(x, 4, fs)
        # the unfiltered version keeps most of the aliased energy
        p_raw = sum(v * v for v in raw) / len(raw)
        p_filt = sum(v * v for v in filt) / len(filt)
        self.assertLess(p_filt, p_raw * 0.5)


class TestAnalysis(unittest.TestCase):
    FS = 100000.0
    F0 = 1000.0

    def _spectrum(self, x):
        return analysis.measure_spectrum(x, self.FS, f_signal=self.F0)

    def test_clean_sine_low_thd(self):
        x = signals.sine(self.F0, self.FS, 8192, amp=1.0)
        r = self._spectrum(x)
        self.assertLess(r["thdDb"], -90.0)
        self.assertAlmostEqual(r["fundamentalHz"], self.F0, delta=2.0)

    def test_enob_rises_with_bits(self):
        x = signals.sine(self.F0, self.FS, 8192, amp=3.29)
        enobs = []
        for bits in (6, 8, 10, 12):
            _, v = signals.quantize(x, bits=bits, vref=3.3, bipolar=True)
            enobs.append(analysis.measure_spectrum(v, self.FS, f_signal=self.F0)["enob"])
        for a, b in zip(enobs, enobs[1:]):
            self.assertLess(a, b)
            # each added bit should add roughly 6 dB -> ~1 ENOB bit
            self.assertGreater(b - a, 0.5)

    def test_square_wave_thd(self):
        sq = signals.square(self.F0, self.FS, 8192, amp=1.0)
        r = self._spectrum(sq)
        # fundamental vs 3rd harmonic: -9.5 dB, plus higher harmonics
        self.assertLess(r["thdDb"], -5.0)
        self.assertGreater(r["thdDb"], -12.0)

    def test_clipping_raises_thd(self):
        clean = signals.sine(self.F0, self.FS, 8192, amp=1.0)
        clipped = [max(-0.5, min(0.5, v)) for v in clean]
        self.assertGreater(self._spectrum(clipped)["thdDb"],
                           self._spectrum(clean)["thdDb"])

    def test_sfdr_positive(self):
        x = signals.sine(self.F0, self.FS, 8192, amp=1.0)
        _, v = signals.quantize(x, bits=10, vref=3.3, bipolar=True)
        r = analysis.measure_spectrum(v, self.FS, f_signal=self.F0)
        self.assertGreater(r["sfdrDb"], 30.0)

    def test_short_signal_raises(self):
        with self.assertRaises(ValueError):
            analysis.measure_spectrum([0.0] * 8, 1000.0)


if __name__ == "__main__":
    unittest.main()
