"""SignalChain — command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__, demos, analysis, dsp, filters, signals, sampling


def _cmd_demos(args) -> int:
    print("可用演示：")
    for item in demos.demo_list():
        print("  %-8s %s" % (item["id"], item["name"]))
    return 0


def _cmd_run(args) -> int:
    demo_id = args.demo
    try:
        data = demos.build_demo(demo_id)
    except ValueError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        print("=== %s · %s ===" % (data["id"], data["name"]))
        if "metrics" in data and data["metrics"].get("sinadDb") is not None:
            m = data["metrics"]
            print("SINAD %s dB   ENOB %s bit   THD %s dBc   SFDR %s dBc"
                  % (m["sinadDb"], m["enob"], m["thdDb"], m["sfdrDb"]))
            print("fundamental %.1f Hz (magnitude %.4f)" % (m["fundamentalHz"], m["fundamentalMag"]))
        if "note" in data:
            print(data["note"])
        if "before" in data:
            print("SNR 滤波前 %.1f dB -> 滤波后 %.1f dB" % (data["before"]["snrDb"], data["after"]["snrDb"]))
        if "bitSweep" in data:
            print("ENOB 随位数：")
            for b in data["bitSweep"]:
                print("  %2d bit -> ENOB %.2f" % (b["bits"], b["enob"]))
        if "lowFreq" in data:
            print("抖动 %.0f ps：%.0f Hz -> SNR %.1f dB；%.0f Hz -> SNR %.1f dB"
                  % (data["jitterPs"], data["lowFreq"]["freq"], data["lowFreq"]["snrDb"],
                     data["highFreq"]["freq"], data["highFreq"]["snrDb"]))
        if "imd" in data:
            print("IMD3 %.1f dBc（产物在 %s Hz）" % (data["imd"]["imd3Db"], data["imd"]["products"]))
        if "aliasFrequency" in data:
            print("混叠：%.0f Hz 采样到 %.0f Hz -> %.0f Hz"
                  % (data["trueFrequency"], data["fs"], data["aliasFrequency"]))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print("已保存到 %s" % args.save)
    return 0


def _cmd_signal(args) -> int:
    fs, n = args.fs, args.n
    kind = args.kind
    if kind == "sine":
        x = signals.sine(args.freq, fs, n, amp=args.amp, dc=args.dc)
    elif kind == "square":
        x = signals.square(args.freq, fs, n, amp=args.amp)
    elif kind == "saw":
        x = signals.sawtooth(args.freq, fs, n, amp=args.amp)
    elif kind == "chirp":
        x = signals.chirp(args.freq, args.freq2, fs, n)
    elif kind == "noise":
        x = signals.white_noise(n, rms=args.amp, seed=args.seed)
    else:
        print("error: unknown kind %r" % kind, file=sys.stderr)
        return 1

    if args.quantize:
        _, v = signals.quantize(x, bits=args.bits, vref=args.vref, bipolar=True)
        x = v

    spec = analysis.measure_spectrum(x, fs, f_signal=args.freq if kind != "noise" else None)
    out = {"kind": kind, "fs": fs, "n": n, "signal": [round(v, 6) for v in x],
           "metrics": spec}
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print("SINAD %.2f dB  ENOB %.2f bit  THD %.2f dBc" % (spec["sinadDb"], spec["enob"], spec["thdDb"]))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print("已保存到 %s" % args.save)
    return 0


def _cmd_filter(args) -> int:
    fs = args.fs
    kind = args.kind
    try:
        if kind == "lowpass":
            stage = filters.make_stage("iir_lowpass", fs, cutoff=args.cutoff, order=args.order)
        elif kind == "highpass":
            stage = filters.make_stage("iir_highpass", fs, cutoff=args.cutoff, order=args.order)
        elif kind == "bandpass":
            stage = filters.make_stage("iir_bandpass", fs, fl=args.fl, fh=args.fh, order=args.order)
        elif kind == "fir":
            stage = filters.make_stage("fir_lowpass", fs, cutoff=args.cutoff, taps=args.taps)
        else:
            print("error: unknown kind %r" % kind, file=sys.stderr)
            return 1
    except Exception as e:
        print("error: %s" % e, file=sys.stderr)
        return 1

    resp = stage.response(512)
    if args.json:
        print(json.dumps({"kind": kind, "stage": stage.to_dict(), "response": resp}, ensure_ascii=False))
    else:
        print("%s: %s" % (stage.name, json.dumps(stage.params, ensure_ascii=False)))
        # print a few response points
        fs_arr = resp["frequencies"]
        mags = resp["magnitudeDb"]
        for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
            idx = int(frac * (len(fs_arr) - 1))
            print("  %8.1f Hz -> %8.2f dB" % (fs_arr[idx], mags[idx]))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"kind": kind, "stage": stage.to_dict(), "response": resp}, f, ensure_ascii=False)
        print("已保存到 %s" % args.save)
    return 0


def _cmd_serve(args) -> int:
    from . import server
    return server.run(port=args.port, no_browser=args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sigchain",
        description="SignalChain — 嵌入式/光电系统的 DSP 信号链工作台（零依赖）")
    p.add_argument("--version", action="version", version="sigchain " + __version__)
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("demos", help="列出内置演示")
    sp.set_defaults(func=_cmd_demos)

    sp = sub.add_parser("run", help="运行一个演示")
    sp.add_argument("demo", help="演示 id")
    sp.add_argument("--json", action="store_true", help="输出 JSON")
    sp.add_argument("--save", help="保存结果到文件")
    sp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("signal", help="生成并分析一个信号")
    sp.add_argument("--kind", default="sine", help="sine/square/saw/chirp/noise")
    sp.add_argument("--freq", type=float, default=1000.0, help="频率 Hz")
    sp.add_argument("--freq2", type=float, default=20000.0, help="chirp 终止频率")
    sp.add_argument("--fs", type=float, default=100000.0, help="采样率 Hz")
    sp.add_argument("--n", type=int, default=8192, help="样本数")
    sp.add_argument("--amp", type=float, default=1.0, help="幅度")
    sp.add_argument("--dc", type=float, default=0.0, help="直流偏置")
    sp.add_argument("--quantize", action="store_true", help="经过 ADC 量化")
    sp.add_argument("--bits", type=int, default=12, help="ADC 位宽")
    sp.add_argument("--vref", type=float, default=3.3, help="ADC 参考电压")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--save")
    sp.set_defaults(func=_cmd_signal)

    sp = sub.add_parser("filter", help="设计一个滤波器并查看频响")
    sp.add_argument("--kind", default="lowpass", help="lowpass/highpass/bandpass/fir")
    sp.add_argument("--cutoff", type=float, default=1000.0, help="截止频率 Hz")
    sp.add_argument("--fl", type=float, default=100.0)
    sp.add_argument("--fh", type=float, default=1000.0)
    sp.add_argument("--fs", type=float, default=100000.0)
    sp.add_argument("--order", type=int, default=4)
    sp.add_argument("--taps", type=int, default=65)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--save")
    sp.set_defaults(func=_cmd_filter)

    sp = sub.add_parser("serve", help="启动本地 Web 工作台")
    sp.add_argument("--port", type=int, default=8779)
    sp.add_argument("--no-browser", action="store_true")
    sp.set_defaults(func=_cmd_serve)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)
