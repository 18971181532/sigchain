"use strict";

// ---------------------------------------------------------------- utilities
const $ = (sel) => document.querySelector(sel);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt = (v, d) => (v === null || v === undefined || !isFinite(v) ? "—" : Number(v).toFixed(d));

const state = {
  tab: "demo",
  demos: [],
  currentDemo: null,
  signal: null,
  filter: null,
};

// ---------------------------------------------------------------- canvas setup
function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(300, rect.width);
  const h = Math.max(200, rect.height || 220);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function makePlot(ctx, w, h, opts) {
  const pad = opts.pad || { l: 54, r: 16, t: 16, b: 30 };
  const box = { x: pad.l, y: pad.t, w: w - pad.l - pad.r, h: h - pad.t - pad.b };
  const sx = (v) => box.x + ((v - opts.xd[0]) / (opts.xd[1] - opts.xd[0])) * box.w;
  const sy = (v) => box.y + box.h - ((v - opts.yd[0]) / (opts.yd[1] - opts.yd[0])) * box.h;

  ctx.save();
  ctx.strokeStyle = cssVar("--grid");
  ctx.fillStyle = cssVar("--muted");
  ctx.lineWidth = 1;
  ctx.font = "10px ui-monospace, monospace";

  const xticks = opts.xticks || 6;
  for (let i = 0; i <= xticks; i++) {
    const v = opts.xd[0] + (i / xticks) * (opts.xd[1] - opts.xd[0]);
    const x = sx(v);
    ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(opts.xfmt(v), x, box.y + box.h + 14);
  }
  const yticks = opts.yticks || 5;
  for (let i = 0; i <= yticks; i++) {
    const v = opts.yd[0] + (i / yticks) * (opts.yd[1] - opts.yd[0]);
    const y = sy(v);
    ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(opts.yfmt(v), box.x - 6, y + 3);
  }

  if (opts.xlabel) {
    ctx.textAlign = "center";
    ctx.fillText(opts.xlabel, box.x + box.w / 2, h - 4);
  }
  ctx.restore();
  return { box, sx, sy };
}

function drawLine(ctx, pts, color, width, dash) {
  if (!pts || pts.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width || 1.4;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  ctx.stroke();
  ctx.restore();
}

// ---------------------------------------------------------------- time plot
function drawTime(canvas, data, opts) {
  opts = opts || {};
  const { ctx, w, h } = setup(canvas);
  if (!data || !data.length) {
    ctx.fillStyle = cssVar("--muted");
    ctx.textAlign = "center";
    ctx.fillText("无数据", w / 2, h / 2);
    return;
  }
  const n = Math.min(data.length, opts.maxN || 4096);
  const xs = data.slice(0, n);
  const lo = Math.min.apply(null, xs);
  const hi = Math.max.apply(null, xs);
  const span = (hi - lo) || 1;
  const fs = opts.fs || 100000;
  const t1 = n / fs;

  const plot = makePlot(ctx, w, h, {
    xd: [0, t1 * 1000],
    yd: [lo - span * 0.1, hi + span * 0.1],
    xfmt: (v) => (v >= 1 ? v.toFixed(1) + "ms" : (v * 1000).toFixed(0) + "µs"),
    yfmt: (v) => v.toFixed(2),
    xlabel: "时间",
  });

  const pts = xs.map((v, i) => [plot.sx((i / fs) * 1000), plot.sy(v)]);
  drawLine(ctx, pts, opts.color || cssVar("--accent"), 1.2);
}

// ---------------------------------------------------------------- spectrum plot
function drawSpectrum(canvas, spec, opts) {
  opts = opts || {};
  const { ctx, w, h } = setup(canvas);
  if (!spec || !spec.frequencies) {
    ctx.fillStyle = cssVar("--muted");
    ctx.textAlign = "center";
    ctx.fillText("无频谱数据", w / 2, h / 2);
    return;
  }
  const freqs = spec.frequencies;
  const psd = spec.psdDb;
  const xmax = freqs[freqs.length - 1];
  let ymax = Math.max.apply(null, psd);
  let ymin = Math.min(ymax - 120, Math.min.apply(null, psd.filter((v) => isFinite(v))));

  const plot = makePlot(ctx, w, h, {
    xd: [0, xmax],
    yd: [ymin, ymax + 5],
    xfmt: (v) => (v >= 1000 ? (v / 1000).toFixed(0) + "k" : v.toFixed(0)),
    yfmt: (v) => v.toFixed(0) + "dB",
    xlabel: "频率 (Hz)",
  });

  const pts = freqs.map((f, i) => [plot.sx(f), plot.sy(clamp(psd[i], ymin, ymax + 5))]);
  drawLine(ctx, pts, opts.color || cssVar("--accent-2"), 1.3);

  // highlight the fundamental if given
  if (opts.fundamental) {
    const fx = plot.sx(opts.fundamental);
    ctx.save();
    ctx.strokeStyle = cssVar("--r");
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(fx, plot.box.y); ctx.lineTo(fx, plot.box.y + plot.box.h); ctx.stroke();
    ctx.fillStyle = cssVar("--r");
    ctx.textAlign = "left";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(fmt(opts.fundamental, 0) + " Hz", fx + 4, plot.box.y + 10);
    ctx.restore();
  }
}

// ---------------------------------------------------------------- response plot
function drawResponse(canvas, resp) {
  const { ctx, w, h } = setup(canvas);
  if (!resp || !resp.frequencies) {
    ctx.fillStyle = cssVar("--muted");
    ctx.textAlign = "center";
    ctx.fillText("无响应数据", w / 2, h / 2);
    return;
  }
  const freqs = resp.frequencies;
  const mags = resp.magnitudeDb;
  const xmax = freqs[freqs.length - 1];
  const ymax = Math.max(0, Math.max.apply(null, mags));
  const ymin = Math.min(-100, ymax - 100);

  const plot = makePlot(ctx, w, h, {
    xd: [0, xmax],
    yd: [ymin, ymax],
    xfmt: (v) => (v >= 1000 ? (v / 1000).toFixed(0) + "k" : v.toFixed(0)),
    yfmt: (v) => v.toFixed(0) + "dB",
    xlabel: "频率 (Hz)",
  });

  const pts = freqs.map((f, i) => [plot.sx(f), plot.sy(clamp(mags[i], ymin, ymax))]);
  drawLine(ctx, pts, cssVar("--accent"), 1.6);

  // -3dB line
  const y3 = plot.sy(-3);
  if (y3 > plot.box.y) {
    ctx.save();
    ctx.strokeStyle = cssVar("--r");
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(plot.box.x, y3); ctx.lineTo(plot.box.x + plot.box.w, y3); ctx.stroke();
    ctx.fillStyle = cssVar("--r");
    ctx.textAlign = "left";
    ctx.fillText("-3 dB", plot.box.x + 4, y3 - 3);
    ctx.restore();
  }
}

// ---------------------------------------------------------------- metrics display
function renderMetrics(host, m) {
  if (!m || m.sinadDb === null || m.sinadDb === undefined) {
    host.innerHTML = '<div class="empty">无指标</div>';
    return;
  }
  const items = [
    ["SINAD", m.sinadDb + " dB"],
    ["ENOB", m.enob + " bit"],
    ["THD", m.thdDb + " dBc"],
    ["SFDR", m.sfdrDb + " dBc"],
    ["SNR", m.snrDb + " dB"],
    ["f₀", fmt(m.fundamentalHz, 0) + " Hz"],
  ];
  host.innerHTML = items
    .map(([k, v]) => '<div class="metric"><div class="k">' + k + '</div><div class="v">' + v + "</div></div>")
    .join("");
}

// ---------------------------------------------------------------- demos
function loadDemos() {
  fetch("/api/demos")
    .then((r) => r.json())
    .then((d) => {
      state.demos = d.demos;
      const host = $("#demoList");
      host.innerHTML = state.demos
        .map((it, i) => '<div class="demo-item" data-id="' + it.id + '">' +
          '<div class="t">' + it.name + '</div><div class="d">' + it.id + "</div></div>")
        .join("");
      host.querySelectorAll(".demo-item").forEach((el) => {
        el.addEventListener("click", () => loadDemo(el.dataset.id, el));
      });
    });
}

function loadDemo(id, el) {
  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ demo: id }),
  })
    .then((r) => r.json())
    .then((d) => {
      state.currentDemo = d;
      $("#demoName").textContent = d.name;
      document.querySelectorAll(".demo-item").forEach((e) => e.classList.remove("active"));
      if (el) el.classList.add("active");
      renderDemo(d);
    });
}

function renderDemo(d) {
  const verdict = $("#demoVerdict");
  let html = "";
  if (d.metrics) {
    const m = d.metrics;
    const rows = [
      ["SINAD", m.sinadDb + " dB"],
      ["ENOB", m.enob + " bit"],
      ["THD", m.thdDb + " dBc"],
      ["SFDR", m.sfdrDb + " dBc"],
    ];
    html += '<div class="metric-row">' +
      rows.map(([k, v]) => metricHtml(k, v)).join("") + "</div>";
  }
  if (d.note) html += '<div class="note">' + d.note + "</div>";
  else if (d.bitSweep) {
    html += '<div class="note">ENOB 随 ADC 位宽单调上升：' +
      d.bitSweep.map((b) => b.bits + "bit→" + fmt(b.enob, 1)).join("，") + "</div>";
  } else if (d.before) {
    html += '<div class="note">滤波后 SNR 由 ' + fmt(d.before.snrDb, 1) +
      " dB 提升到 " + fmt(d.after.snrDb, 1) + " dB（干扰被抑制）</div>";
  } else if (d.lowFreq) {
    html += '<div class="note">同样的 ' + d.jitterPs + " ps 抖动：低频 " +
      fmt(d.lowFreq.freq, 0) + " Hz → " + fmt(d.lowFreq.snrDb, 1) +
      " dB，高频 " + fmt(d.highFreq.freq, 0) + " Hz → " + fmt(d.highFreq.snrDb, 1) + " dB</div>";
  } else if (d.imd) {
    html += '<div class="note">IMD3 = ' + d.imd.imd3Db + " dBc，三阶互调产物在 " +
      d.imd.products.join(" / ") + " Hz</div>";
  }
  verdict.innerHTML = html || '<div class="empty">无额外结论</div>';

  drawTime($("#timeCanvas"), d.signal, { fs: d.fs });
  drawSpectrum($("#specCanvas"), d.spectrum, { fundamental: d.metrics && d.metrics.fundamentalHz });

  if (d.response) {
    $("#filterRespCard").style.display = "";
    drawResponse($("#respCanvas"), d.response);
  } else {
    $("#filterRespCard").style.display = "none";
  }
}

function metricHtml(k, v) {
  return '<div class="metric"><div class="k">' + k + '</div><div class="v">' + v + "</div></div>";
}

// ---------------------------------------------------------------- signal generation
function genSignal() {
  const body = {
    kind: $("#sigKind").value,
    freq: parseFloat($("#sigFreq").value) || 1000,
    fs: parseFloat($("#sigFs").value) || 100000,
    n: parseInt($("#sigN").value) || 4096,
    amp: parseFloat($("#sigAmp").value) || 1,
    freq2: 20000,
  };
  const bits = $("#sigBits").value;
  if (bits) body.bits = parseInt(bits);

  fetch("/api/signal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then((r) => r.json())
    .then((d) => {
      state.signal = d;
      renderMetrics($("#sigMetrics"), d.metrics);
      drawTime($("#sigTimeCanvas"), d.signal, { fs: d.fs });
      drawSpectrum($("#sigSpecCanvas"), d.spectrum, { fundamental: d.metrics.fundamentalHz });
    });
}

// ---------------------------------------------------------------- filter design
function designFilter() {
  const body = {
    kind: $("#fltKind").value,
    cutoff: parseFloat($("#fltCutoff").value) || 1000,
    fl: parseFloat($("#fltFl").value) || 100,
    fh: parseFloat($("#fltFh").value) || 2000,
    fs: parseFloat($("#fltFs").value) || 100000,
    order: parseInt($("#fltOrder").value) || 4,
    taps: 65,
  };
  fetch("/api/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then((r) => r.json())
    .then((d) => {
      state.filter = d;
      const p = d.stage.params;
      $("#fltInfo").textContent = (p.cutoff ? "cutoff " + p.cutoff + " Hz" : "") +
        (p.fl ? " [" + p.fl + "–" + p.fh + " Hz]" : "") +
        (p.order ? " · 阶数 " + p.order : "");
      drawResponse($("#fltRespCanvas"), d.response);
      runFilterDemo(d);
    });
}

function runFilterDemo(filter) {
  const fs = 100000;
  const n = 4096;
  const t = Array.from({ length: n }, (_, i) => i / fs);
  const clean = t.map((ti) => Math.sin(2 * Math.PI * 500 * ti));
  const noisy = t.map((ti) => Math.sin(2 * Math.PI * 500 * ti) + 0.6 * Math.sin(2 * Math.PI * 4000 * ti));

  fetch("/api/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...filter.stage, kind: filter.kind, signal: noisy, fs: fs }),
  })
    .then((r) => r.json())
    .then((d) => {
      const { ctx, w, h } = setup($("#fltDemoCanvas"));
      const show = noisy.slice(0, 800);
      const out = d.filtered.slice(0, 800);
      const lo = -2, hi = 2;
      const plot = makePlot(ctx, w, h, {
        xd: [0, 800 / fs * 1000],
        yd: [lo, hi],
        xfmt: (v) => v.toFixed(0) + "ms",
        yfmt: (v) => v.toFixed(1),
        xlabel: "时间",
      });
      const ptsIn = show.map((v, i) => [plot.sx(i / fs * 1000), plot.sy(v)]);
      const ptsOut = out.map((v, i) => [plot.sx(i / fs * 1000), plot.sy(v)]);
      drawLine(ctx, ptsIn, cssVar("--muted"), 1.1);
      drawLine(ctx, ptsOut, cssVar("--accent"), 1.8);
      ctx.fillStyle = cssVar("--muted");
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText("— 输入（500Hz + 4kHz 干扰）   — 输出（滤波后）", plot.box.x + 4, plot.box.y + 10);
    });
}

// ---------------------------------------------------------------- tabs
function switchTab(name) {
  state.tab = name;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("#view-" + name).classList.add("active");
  if (name === "signal" && !state.signal) genSignal();
  if (name === "filter" && !state.filter) designFilter();
}

// ---------------------------------------------------------------- theme
function applyTheme(dark) {
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}
function toggleTheme() {
  const dark = document.documentElement.dataset.theme !== "dark";
  applyTheme(dark);
  try { localStorage.setItem("sigchain-theme", dark ? "dark" : "light"); } catch (e) {}
  redrawAll();
}

function redrawAll() {
  if (state.currentDemo) renderDemo(state.currentDemo);
  if (state.signal) {
    drawTime($("#sigTimeCanvas"), state.signal.signal, { fs: state.signal.fs });
    drawSpectrum($("#sigSpecCanvas"), state.signal.spectrum, { fundamental: state.signal.metrics.fundamentalHz });
  }
  if (state.filter) drawResponse($("#fltRespCanvas"), state.filter.response);
}

// ---------------------------------------------------------------- init
function init() {
  const saved = (() => { try { return localStorage.getItem("sigchain-theme"); } catch (e) { return null; } })();
  applyTheme(saved === "dark");

  document.querySelectorAll(".tabs button").forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));
  $("#themeBtn").addEventListener("click", toggleTheme);
  $("#sigGenBtn").addEventListener("click", genSignal);
  $("#fltDesignBtn").addEventListener("click", designFilter);

  window.addEventListener("resize", redrawAll);

  fetch("/api/health")
    .then((r) => r.json())
    .then((d) => { $("#serverStatus").textContent = "本地服务 · v" + d.version; })
    .catch(() => { $("#serverStatus").textContent = "连接失败"; });

  loadDemos();
  loadDemo("adc");
}

init();
