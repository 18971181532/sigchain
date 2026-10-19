"""Filter chains, demos and the HTTP API."""

import io
import json
import math
import unittest
from contextlib import redirect_stdout
from urllib.request import Request, urlopen

from sigchain import demos, dsp, filters, signals


class TestFilterStage(unittest.TestCase):
    def test_fir_stage_response(self):
        st = filters.make_stage("fir_lowpass", 1000.0, cutoff=100.0, taps=33)
        resp = st.response(256)
        self.assertEqual(len(resp["frequencies"]), 256)
        self.assertGreater(resp["magnitudeDb"][0], -0.5)      # DC passes
        self.assertLess(resp["magnitudeDb"][-1], -20.0)       # Nyquist blocked

    def test_iir_stage_stable(self):
        st = filters.make_stage("iir_lowpass", 1000.0, cutoff=100.0, order=4)
        y = st.apply([1.0] * 500)
        self.assertAlmostEqual(y[-1], 1.0, places=5)

    def test_chain_composite_response(self):
        chain = filters.SignalChain([
            filters.make_stage("iir_lowpass", 10000.0, cutoff=1000, order=4),
            filters.make_stage("iir_highpass", 10000.0, cutoff=100, order=4),
        ], 10000.0)
        resp = chain.response(256)
        # band 100-1000 Hz should pass roughly flat-ish; extremes blocked
        self.assertLess(resp["magnitudeDb"][0], -10.0)   # DC blocked by HP
        self.assertLess(resp["magnitudeDb"][-1], -10.0)  # Nyquist blocked by LP

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            filters.make_stage("nope", 1000.0, cutoff=100)

    def test_serialisation_roundtrip(self):
        st = filters.make_stage("iir_lowpass", 1000.0, cutoff=200.0, order=2)
        d = st.to_dict()
        st2 = filters.FilterStage(d["b"], d["a"], d["kind"], d["name"],
                                  1000.0, d["params"])
        for a, b in zip(st.b, st2.b):
            self.assertAlmostEqual(a, b, places=6)
        for a, b in zip(st.a, st2.a):
            self.assertAlmostEqual(a, b, places=6)


class TestDemos(unittest.TestCase):
    def test_every_demo_builds(self):
        for item in demos.demo_list():
            d = demos.build_demo(item["id"])
            self.assertEqual(d["id"], item["id"])
            self.assertTrue(d["name"])
            self.assertIn("fs", d)

    def test_unknown_demo_raises(self):
        with self.assertRaises(ValueError):
            demos.build_demo("nope")

    def test_adc_demo_enob_monotonic(self):
        d = demos.build_demo("adc")
        enobs = [row["enob"] for row in d["bitSweep"]]
        self.assertEqual(enobs, sorted(enobs))

    def test_alias_demo_numbers(self):
        d = demos.build_demo("alias")
        self.assertEqual(d["trueFrequency"], 60000.0)
        self.assertEqual(d["aliasFrequency"], 40000.0)

    def test_filter_demo_improves_snr(self):
        d = demos.build_demo("filter")
        self.assertGreater(d["after"]["snrDb"], d["before"]["snrDb"])


class TestServer(unittest.TestCase):
    """End-to-end API tests against a live server thread."""

    PORT = 8791
    base = "http://127.0.0.1:%d" % PORT

    @classmethod
    def setUpClass(cls):
        from sigchain import server as srv
        import threading
        srv.ThreadingHTTPServer.allow_reuse_address = True
        cls.httpd = srv.ThreadingHTTPServer(("127.0.0.1", cls.PORT), srv.Handler)
        t = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        t.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _get(self, path):
        with urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def _post(self, path, body):
        req = Request(self.base + path, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def test_health(self):
        d = self._get("/api/health")
        self.assertTrue(d["ok"])

    def test_index_served(self):
        with urlopen(self.base + "/", timeout=10) as r:
            html = r.read().decode("utf-8")
        self.assertIn("SignalChain", html)

    def test_run_demo(self):
        d = self._post("/api/run", {"demo": "alias"})
        self.assertEqual(d["aliasFrequency"], 40000.0)

    def test_signal_api(self):
        d = self._post("/api/signal", {"kind": "sine", "freq": 1000.0,
                                       "fs": 100000.0, "n": 4096, "bits": 12})
        self.assertIn("metrics", d)
        self.assertIsNotNone(d["metrics"]["sinadDb"])

    def test_filter_api(self):
        d = self._post("/api/filter", {"kind": "lowpass", "cutoff": 1000.0,
                                       "fs": 100000.0, "order": 4})
        self.assertGreater(d["response"]["magnitudeDb"][0], -0.5)

    def test_analyze_api(self):
        x = [math.sin(2 * math.pi * 1000 * i / 100000) for i in range(4096)]
        d = self._post("/api/analyze", {"signal": x, "fs": 100000.0,
                                        "f_signal": 1000.0})
        self.assertLess(d["metrics"]["thdDb"], -90.0)

    def test_unknown_demo_400(self):
        with self.assertRaises(Exception):
            self._post("/api/run", {"demo": "nope"})


if __name__ == "__main__":
    unittest.main()
