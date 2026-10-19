"""FFT correctness: round-trip, known bins, window properties."""

import cmath
import math
import random
import unittest

from sigchain import dsp


class TestFft(unittest.TestCase):
    def test_roundtrip(self):
        n = 64
        x = [math.sin(2 * math.pi * 5 * i / n) + 0.5 * math.cos(2 * math.pi * 11 * i / n)
             for i in range(n)]
        y = dsp.ifft(dsp.fft(x))
        self.assertAlmostEqual(max(abs(a - b.real) for a, b in zip(x, y)), 0.0, places=12)

    def test_matches_naive_dft(self):
        rng = random.Random(0)
        n = 32
        x = [rng.uniform(-1, 1) for _ in range(n)]
        X = dsp.fft(x)
        for k in range(n):
            expected = sum(x[j] * cmath.exp(-2j * math.pi * k * j / n) for j in range(n))
            self.assertAlmostEqual(abs(X[k] - expected), 0.0, places=9)

    def test_tone_lands_on_bin(self):
        n = 128
        x = [math.cos(2 * math.pi * 8 * i / n) for i in range(n)]
        freqs, mags = dsp.fft_magnitude(x, fs=128.0)
        peak = max(range(len(mags)), key=lambda i: mags[i])
        self.assertEqual(freqs[peak], 8.0)
        self.assertAlmostEqual(mags[peak], 1.0, places=6)

    def test_zero_padding(self):
        X = dsp.fft([1.0, 2.0, 3.0])  # 3 samples -> padded to 4
        self.assertEqual(len(X), 4)
        # X[0] = 6 (the DC sum is preserved)
        self.assertAlmostEqual(abs(X[0]), 6.0, places=9)


class TestWindows(unittest.TestCase):
    def test_coherent_gain_values(self):
        # theoretical mean values
        for name, expected in (("hann", 0.5), ("hamming", 0.54), ("rect", 1.0)):
            cg = dsp.window_coherent_gain(dsp.window(name, 4096))
            self.assertAlmostEqual(cg, expected, places=2, msg=name)

    def test_enbw_ordering(self):
        # ENBW: hamming (1.36 bins) < hann (1.50) < blackman (1.73)
        enbw = {name: dsp.window_noise_bw(dsp.window(name, 8192))
                for name in ("hann", "hamming", "blackman")}
        self.assertLess(enbw["hamming"], enbw["hann"])
        self.assertLess(enbw["hann"], enbw["blackman"])

    def test_kaiser_extremes(self):
        w = dsp.window("kaiser", 64, 0.0)  # beta=0 -> rectangular
        self.assertAlmostEqual(min(w), 1.0, places=9)

    def test_unknown_window_raises(self):
        with self.assertRaises(ValueError):
            dsp.window("nope", 16)


class TestFir(unittest.TestCase):
    def test_lowpass_attenuates(self):
        fs = 1000.0
        h = dsp.fir_lowpass(100.0, fs, 51)
        _, H = dsp.freqz(h, [1.0], 512, fs)
        self.assertLess(dsp.db(abs(H[128])), -25.0)   # 200 Hz
        self.assertGreater(dsp.db(abs(H[0])), -0.5)   # DC

    def test_highpass_is_complement(self):
        fs = 1000.0
        h = dsp.fir_highpass(100.0, fs, 51)
        _, H = dsp.freqz(h, [1.0], 512, fs)
        self.assertGreater(dsp.db(abs(H[400])), -1.0)  # ~780 Hz passes
        self.assertLess(dsp.db(abs(H[10])), -20.0)     # DC blocked

    def test_convolve_basic(self):
        self.assertEqual(dsp.convolve([1, 2, 1], [1, 1]), [1, 3, 3, 1])


class TestIir(unittest.TestCase):
    def test_lowpass_3db_point(self):
        fs = 1000.0
        b, a = dsp.butter_lowpass(100.0, fs, 4)
        freqs, H = dsp.freqz(b, a, 4096, fs)
        for i in range(1, len(H)):
            if dsp.db(abs(H[i])) < -3.0:
                self.assertAlmostEqual(freqs[i], 100.0, delta=2.0)
                break

    def test_stability(self):
        fs = 1000.0
        for order in (1, 2, 3, 4, 6):
            b, a = dsp.butter_lowpass(200.0, fs, order)
            imp = dsp.lfilter(b, a, [1.0] + [0.0] * 4000)
            tail = max(abs(v) for v in imp[1000:])
            self.assertLess(tail, 1e-6, "order %d unstable" % order)

    def test_step_settles_to_one(self):
        fs = 1000.0
        b, a = dsp.butter_lowpass(100.0, fs, 4)
        y = dsp.lfilter(b, a, [1.0] * 500)
        self.assertAlmostEqual(y[-1], 1.0, places=6)

    def test_highpass_3db_point(self):
        fs = 1000.0
        b, a = dsp.butter_highpass(100.0, fs, 4)
        freqs, H = dsp.freqz(b, a, 4096, fs)
        for i in range(len(H) - 1, 0, -1):
            if dsp.db(abs(H[i])) < -3.0:
                self.assertAlmostEqual(freqs[i], 100.0, delta=2.0)
                break
        self.assertGreater(dsp.db(abs(H[-1])), -0.5)  # Nyquist passes

    def test_bandpass_edges(self):
        fs = 1000.0
        b, a = dsp.butter_bandpass(50.0, 200.0, fs, 4)
        freqs, H = dsp.freqz(b, a, 4096, fs)
        peak = max(range(len(H)), key=lambda i: abs(H[i]))
        self.assertGreater(dsp.db(abs(H[peak])), -1.0)

    def test_lfilter_fir_and_iir(self):
        self.assertEqual(dsp.lfilter([1, 1], [1], [1, 0, 0, 0]), [1.0, 1.0, 0.0, 0.0])
        out = dsp.lfilter([1], [1, -0.5], [1, 0, 0, 0, 0])
        self.assertEqual([round(v, 4) for v in out], [1.0, 0.5, 0.25, 0.125, 0.0625])


if __name__ == "__main__":
    unittest.main()
