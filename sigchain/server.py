"""SignalChain — HTTP server serving the web workbench and a JSON API."""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from . import __version__, demos, analysis, filters, signals, sampling
from . import dsp


WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(BaseHTTPRequestHandler):
    server_version = "SignalChain/" + __version__

    # ------------------------------------------------------------------ utils
    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------ routes
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/health":
            return self._json({"ok": True, "version": __version__,
                               "demos": len(demos.demo_list())})
        if path == "/api/demos":
            return self._json({"demos": demos.demo_list()})
        if path == "/api/version":
            return self._json({"version": __version__})
        if path == "/" or path == "":
            return self._serve_file("index.html")
        if path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        return self._serve_static(path)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_body()
        if path == "/api/run":
            return self._run_demo(body)
        if path == "/api/signal":
            return self._gen_signal(body)
        if path == "/api/filter":
            return self._design_filter(body)
        if path == "/api/analyze":
            return self._analyze(body)
        return self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------------ actions
    def _run_demo(self, body: dict):
        demo_id = body.get("demo", "adc")
        try:
            data = demos.build_demo(demo_id)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        return self._json(data)

    def _gen_signal(self, body: dict):
        kind = body.get("kind", "sine")
        fs = float(body.get("fs", 100000.0))
        n = int(body.get("n", 4096))
        amp = float(body.get("amp", 1.0))
        dc = float(body.get("dc", 0.0))
        freq = float(body.get("freq", 1000.0))
        freq2 = float(body.get("freq2", 20000.0))
        seed = body.get("seed", 1)

        if kind == "sine":
            x = signals.sine(freq, fs, n, amp, dc=dc)
        elif kind == "square":
            x = signals.square(freq, fs, n, amp)
        elif kind == "saw":
            x = signals.sawtooth(freq, fs, n, amp)
        elif kind == "triangle":
            x = signals.triangle(freq, fs, n, amp)
        elif kind == "chirp":
            x = signals.chirp(freq, freq2, fs, n)
        elif kind == "noise":
            x = signals.white_noise(n, rms=amp, seed=seed)
        elif kind == "pink":
            x = signals.pink_noise(n, rms=amp, seed=seed)
        else:
            return self._json({"error": "unknown kind %r" % kind}, 400)

        bits = body.get("bits")
        if bits is not None:
            _, x = signals.quantize(x, bits=int(bits), vref=float(body.get("vref", 3.3)),
                                    bipolar=True)

        f_signal = freq if kind not in ("noise", "pink") else None
        spec = analysis.measure_spectrum(x, fs, f_signal=f_signal)
        freqs, psd = dsp.power_spectrum(x, fs)
        return self._json({
            "kind": kind, "fs": fs, "n": n,
            "signal": [round(v, 6) for v in x],
            "spectrum": {"frequencies": [round(f, 2) for f in freqs],
                         "psdDb": [round(dsp.db(p), 2) for p in psd]},
            "metrics": spec,
        })

    def _design_filter(self, body: dict):
        fs = float(body.get("fs", 100000.0))
        kind = body.get("kind", "lowpass")
        try:
            if kind == "lowpass":
                stage = filters.make_stage("iir_lowpass", fs, cutoff=float(body.get("cutoff", 1000)),
                                           order=int(body.get("order", 4)))
            elif kind == "highpass":
                stage = filters.make_stage("iir_highpass", fs, cutoff=float(body.get("cutoff", 1000)),
                                           order=int(body.get("order", 4)))
            elif kind == "bandpass":
                stage = filters.make_stage("iir_bandpass", fs, fl=float(body.get("fl", 100)),
                                           fh=float(body.get("fh", 1000)), order=int(body.get("order", 4)))
            elif kind == "fir":
                stage = filters.make_stage("fir_lowpass", fs, cutoff=float(body.get("cutoff", 1000)),
                                           taps=int(body.get("taps", 65)))
            else:
                return self._json({"error": "unknown kind"}, 400)
        except Exception as e:
            return self._json({"error": str(e)}, 400)

        # apply to a test signal if provided
        result = {"kind": kind, "stage": stage.to_dict(), "response": stage.response(512)}
        if "signal" in body:
            x = body["signal"]
            result["filtered"] = [round(v, 6) for v in stage.apply(x)]
        return self._json(result)

    def _analyze(self, body: dict):
        x = body.get("signal", [])
        fs = float(body.get("fs", 100000.0))
        f_signal = body.get("f_signal")
        if not x:
            return self._json({"error": "no signal"}, 400)
        spec = analysis.measure_spectrum([float(v) for v in x], fs, f_signal=f_signal)
        freqs, psd = dsp.power_spectrum(x, fs)
        return self._json({
            "metrics": spec,
            "spectrum": {"frequencies": [round(f, 2) for f in freqs],
                         "psdDb": [round(dsp.db(p), 2) for p in psd]},
        })

    # ------------------------------------------------------------------ static
    def _serve_file(self, name: str):
        path = os.path.join(WEB_DIR, name)
        if not os.path.isfile(path):
            return self._send(404, b"not found", "text/plain")
        ctype = "text/html"
        if name.endswith(".css"):
            ctype = "text/css"
        elif name.endswith(".js"):
            ctype = "application/javascript"
        with open(path, "rb") as f:
            return self._send(200, f.read(), ctype)

    def _serve_static(self, path: str):
        name = path.lstrip("/") or "index.html"
        return self._serve_file(name)

    def log_message(self, *args):
        pass  # keep the console clean


def run(port: int = 8779, no_browser: bool = False) -> int:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://127.0.0.1:%d/" % port
    print("SignalChain 工作台已启动：%s" % url)
    print("按 Ctrl+C 停止。")
    if not no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0
