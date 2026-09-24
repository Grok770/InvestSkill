"""Interactive line charts: how the business did vs. how the stock did.

``performance_html`` writes one self-contained HTML page (no external
scripts; works offline) with, per company:

1. **Business vs. stock price, indexed to 100.** Stock price, revenue and
   EPS (plus SPY as a grey reference) all start at 100 on the same date, so a
   single axis shows whether the share price ran ahead of, or behind, the
   business. (Two different units on two y-axes would mislead; indexing to a
   common base is the honest way to compare them.)
2. **P/E over time** — price ÷ the latest annual EPS that had been
   *published* by that date. A rising line means the price outran earnings.
3. **News sentiment by day** (when news is supplied) — diverging bars.

Every chart has a hover crosshair + tooltip, direct end labels, a legend,
light/dark themes, and a data table underneath.
"""

from __future__ import annotations

import html
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .financials import derive

REPORTING_LAG_DAYS = 60  # when no filing date is known, assume a 10-K lands ~60 days after FY end


def _ts(d) -> int:
    return int(pd.Timestamp(d).timestamp() * 1000)


def _clean(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)


def company_chart_data(ticker: str, prices: pd.Series, fin: pd.DataFrame | None,
                       bench: pd.Series | None = None, news=None, name: str | None = None) -> dict:
    """Everything the page needs for one company, as plain JSON-able data."""
    monthly = prices.dropna().resample("ME").last().dropna()
    data: dict = {"ticker": ticker, "name": name or ticker, "notes": [], "indexed": [], "pe": [],
                  "news": [], "table": []}
    if fin is None or len(fin) < 2:
        base = monthly.index[0]
        data["notes"].append("No multi-year financials available, so only the stock price is shown. "
                             "Use --provider edgar for SEC figures.")
        fin_d = None
    else:
        fin_d = derive(fin).sort_index()
        in_range = fin_d[fin_d.index >= monthly.index[0]]
        base = in_range.index[0] if len(in_range) else monthly.index[0]
        fin_d = fin_d[fin_d.index >= base]

    def price_at(series: pd.Series, when) -> float | None:
        s = series[series.index <= when]
        return float(s.iloc[-1]) if len(s) else (float(series.iloc[0]) if len(series) else None)

    p0 = price_at(monthly, base)
    series = [{"key": "price", "name": f"{ticker} share price", "short": "Price", "color": "--series-1",
               "points": [[_ts(d), _clean(v / p0 * 100)] for d, v in monthly[monthly.index >= base].items()]}]
    if fin_d is not None:
        for key, col, color, label in (("revenue", "revenue", "--series-2", "Revenue"),
                                       ("eps", "eps_diluted", "--series-3", "EPS (diluted)")):
            vals = fin_d[col].dropna()
            if len(vals) < 2:
                continue
            if vals.iloc[0] <= 0:
                data["notes"].append(f"{label} was zero or negative at the start, so it can't be "
                                     "indexed to 100 and is left off the chart.")
                continue
            series.append({"key": key, "name": label, "color": color, "markers": True,
                           "points": [[_ts(d), _clean(v / vals.iloc[0] * 100)] for d, v in vals.items()]})
    if bench is not None and len(bench.dropna()):
        bm = bench.dropna().resample("ME").last()
        b0 = price_at(bm, base)
        series.append({"key": "bench", "name": "SPY (reference)", "color": "--reference", "dashed": True,
                       "points": [[_ts(d), _clean(v / b0 * 100)] for d, v in bm[bm.index >= base].items()]})
    data["indexed"] = series
    data["base"] = str(pd.Timestamp(base).date())

    # P/E using only EPS that was public at each month-end (no look-ahead).
    if fin_d is not None and fin_d["eps_diluted"].notna().sum():
        eps = fin_d["eps_diluted"].dropna()
        known = pd.Series(eps.values, index=[
            pd.Timestamp(fin_d.loc[d, "filed"]) if "filed" in fin_d and pd.notna(fin_d.loc[d, "filed"])
            else d + pd.Timedelta(days=REPORTING_LAG_DAYS) for d in eps.index]).sort_index()
        pts = []
        for d, px in monthly.items():
            k = known[known.index <= d]
            if len(k) and k.iloc[-1] > 0:
                pts.append([_ts(d), _clean(px / k.iloc[-1])])
        if pts:
            data["pe"] = [{"key": "pe", "name": "P/E (price ÷ latest published annual EPS)",
                           "short": "P/E", "color": "--series-1", "points": pts}]
        if any(v <= 0 for v in eps):
            data["notes"].append("P/E is left blank for months when the latest annual EPS was negative.")
        for d, r in fin_d.iterrows():
            data["table"].append({
                "fiscal_year_end": str(d.date()),
                "revenue": _clean(r.get("revenue")), "eps": _clean(r.get("eps_diluted")),
                "fcf": _clean(r.get("free_cash_flow")), "op_margin": _clean(r.get("operating_margin")),
                "price": _clean(price_at(monthly, d)),
            })
    if news is not None and getattr(news, "daily", None):
        data["news"] = news.daily
        data["news_summary"] = f"{news.outlook} ({news.score:.1f}/10, {news.items_used} items)"
    return data


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --reference: #898781;
  --pos: #2a78d6; --neg: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --reference: #898781;
    --pos: #3987e5; --neg: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --reference: #898781;
  --pos: #3987e5; --neg: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 36px 0 4px; }
h3 { font-size: 15px; margin: 0 0 2px; }
.sub, .note { color: var(--ink-2); margin: 0 0 12px; }
.note { font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px;
  padding: 16px; margin: 12px 0; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 6px 0 8px; font-size: 13px;
  color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { width: 16px; height: 0; border-top: 2px solid; display: inline-block; }
.key.dashed { border-top-style: dashed; }
.plot { position: relative; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.end-label { font-size: 12px; fill: var(--ink-2); }
.tip { position: absolute; pointer-events: none; background: var(--surface); color: var(--ink);
  border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); min-width: 170px; display: none; z-index: 2; }
.tip .row { display: flex; align-items: center; gap: 8px; }
.tip b { font-variant-numeric: tabular-nums; }
.tip .when { color: var(--muted); margin-bottom: 4px; }
details { margin-top: 10px; font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
.table-wrap { overflow-x: auto; }
footer { color: var(--muted); font-size: 12px; margin-top: 32px; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="sub">How each company's business performed compared with its share price.
Lines start at 100 on the same date, so you can see whether the price ran ahead of, or behind, the business.</p>
<div id="root"></div>
<footer>__DISCLAIMER__ Generated __GENERATED__ by investskill-analyst.</footer>
</main>
<script id="chart-data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("chart-data").textContent);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const NS = "http://www.w3.org/2000/svg";
  const fmtDate = (t) => new Date(t).toISOString().slice(0, 7);
  const el = (tag, attrs, parent, ns) => {
    const n = ns ? document.createElementNS(NS, tag) : document.createElement(tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const text = (tag, s, parent, cls) => { const n = el(tag, cls ? {class: cls} : {}, parent); n.textContent = s; return n; };

  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }

  function legend(parent, series) {
    const lg = el("div", {class: "legend"}, parent);
    series.forEach((s) => {
      const item = el("span", {}, lg);
      el("i", {class: "key" + (s.dashed ? " dashed" : ""), style: "border-color: var(" + s.color + ")"}, item);
      text("span", s.name, item);
    });
  }

  // Last known value at or before t (annual series hold their value between reports).
  function valueAt(s, t) {
    let v = null, when = null;
    for (const [x, y] of s.points) { if (x <= t + 16 * 864e5) { v = y; when = x; } else break; }
    return v === null ? null : {v: v, when: when};
  }

  function lineChart(parent, series, opts) {
    series = series.filter((s) => s.points.length);
    if (!series.length) return;
    if (series.length > 1) legend(parent, series);   // one series: the title names it
    const plot = el("div", {class: "plot"}, parent);
    // Draw at the real container width so text stays 11-12px on every screen.
    const W = Math.max(300, Math.round(parent.clientWidth - 32));
    const narrow = W < 600;
    const H = narrow ? 240 : 340, m = {t: 12, r: narrow ? 12 : 132, b: 28, l: 44};
    const svg = el("svg", {viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.label}, plot, true);
    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => v !== null);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(...ys, opts.includeZero ? 0 : Infinity), y1 = Math.max(...ys);
    const pad = (y1 - y0) * 0.06; y0 = opts.includeZero ? Math.min(0, y0) : y0 - pad; y1 += pad;
    const X = (t) => m.l + (t - x0) / (x1 - x0 || 1) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
    niceTicks(y0, y1, 5).forEach((v) => {
      el("line", {x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css("--grid"), "stroke-width": 1}, svg, true);
      const t = el("text", {x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "tick"}, svg, true);
      t.textContent = opts.fmt(v);
    });
    if (opts.refLine !== undefined && opts.refLine >= y0 && opts.refLine <= y1) {
      el("line", {x1: m.l, x2: W - m.r, y1: Y(opts.refLine), y2: Y(opts.refLine), stroke: css("--axis"),
        "stroke-width": 1}, svg, true);
    }
    const y0d = new Date(x0).getUTCFullYear(), y1d = new Date(x1).getUTCFullYear();
    const every = Math.max(1, Math.ceil((y1d - y0d + 1) / (narrow ? 5 : 10)));
    for (let y = y0d; y <= y1d; y += every) {
      const t = Date.UTC(y, 0, 1);
      if (t < x0 || t > x1) continue;
      const lbl = el("text", {x: X(t), y: H - 8, "text-anchor": "middle", class: "tick"}, svg, true);
      lbl.textContent = String(y);
    }
    const ends = [];
    series.forEach((s) => {
      const pts = s.points.filter((p) => p[1] !== null);
      const d = pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(" ");
      el("path", {d: d, fill: "none", stroke: "var(" + s.color + ")", "stroke-width": s.dashed ? 1.5 : 2,
        "stroke-dasharray": s.dashed ? "4 4" : "", "stroke-linejoin": "round", "stroke-linecap": "round"}, svg, true);
      if (s.markers) pts.forEach((p) => el("circle", {cx: X(p[0]), cy: Y(p[1]), r: 4, fill: "var(" + s.color + ")",
        stroke: css("--surface"), "stroke-width": 2}, svg, true));
      const last = pts[pts.length - 1];
      ends.push({y: Y(last[1]), label: (s.short || s.name.replace(" (reference)", "")) + " " + opts.fmt(last[1])});
    });
    if (!narrow) {   // direct end labels; on phones the legend + tooltip carry identity
      ends.sort((a, b) => a.y - b.y);   // nudge labels apart so they never collide
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
      ends.forEach((e) => { const t = el("text", {x: W - m.r + 8, y: e.y + 4, class: "end-label"}, svg, true); t.textContent = e.label; });
    }

    const cross = el("line", {y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden"}, svg, true);
    const tip = el("div", {class: "tip"}, plot);
    const hit = el("rect", {x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent"}, svg, true);
    const grid = [...new Set(xs)].sort((a, b) => a - b);
    function show(evt) {
      const r = svg.getBoundingClientRect();
      const px = (evt.clientX - r.left) / r.width * W;
      const t = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
      const snap = grid.reduce((a, b) => Math.abs(b - t) < Math.abs(a - t) ? b : a);
      cross.setAttribute("x1", X(snap)); cross.setAttribute("x2", X(snap)); cross.setAttribute("visibility", "visible");
      tip.replaceChildren();
      text("div", fmtDate(snap), tip, "when");
      series.forEach((s) => {
        const hitv = valueAt(s, snap);
        if (!hitv) return;
        const row = el("div", {class: "row"}, tip);
        el("i", {class: "key" + (s.dashed ? " dashed" : ""), style: "border-color: var(" + s.color + ")"}, row);
        text("b", opts.fmt(hitv.v), row);
        text("span", s.name + (s.markers ? " (FY " + fmtDate(hitv.when) + ")" : ""), row);
      });
      tip.style.display = "block";
      const left = (X(snap) / W) * r.width;
      tip.style.left = Math.min(left + 12, r.width - tip.offsetWidth - 4) + "px";
      tip.style.top = "8px";
    }
    hit.addEventListener("pointermove", show);
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
  }

  function barChart(parent, days) {
    const plot = el("div", {class: "plot"}, parent);
    const W = Math.max(300, Math.round(parent.clientWidth - 32)), H = 160, m = {t: 8, r: 12, b: 24, l: 44};
    const svg = el("svg", {viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Daily news sentiment"}, plot, true);
    const Y = (v) => m.t + (1 - (v + 1) / 2) * (H - m.t - m.b);
    [-1, -0.5, 0, 0.5, 1].forEach((v) => {
      el("line", {x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css(v === 0 ? "--axis" : "--grid")}, svg, true);
      const t = el("text", {x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "tick"}, svg, true);
      t.textContent = (v > 0 ? "+" : "") + v;
    });
    const bw = Math.max(4, Math.min(24, (W - m.l - m.r) / days.length - 2));
    const tip = el("div", {class: "tip"}, plot);
    days.forEach((d, i) => {
      const cx = m.l + (i + 0.5) * (W - m.l - m.r) / days.length;
      const y = Y(d.sentiment), y0 = Y(0);
      el("rect", {x: cx - bw / 2, y: Math.min(y, y0), width: bw, height: Math.max(1, Math.abs(y - y0)), rx: 2,
        fill: "var(" + (d.sentiment >= 0 ? "--pos" : "--neg") + ")"}, svg, true);
      if (i % Math.ceil(days.length / (W < 600 ? 4 : 8)) === 0) {
        const t = el("text", {x: cx, y: H - 6, "text-anchor": "middle", class: "tick"}, svg, true);
        t.textContent = d.date.slice(5);
      }
      const hit = el("rect", {x: cx - Math.max(bw, 24) / 2, y: m.t, width: Math.max(bw, 24), height: H - m.t - m.b,
        fill: "transparent", tabindex: 0}, svg, true);
      const on = () => {
        tip.replaceChildren();
        text("div", d.date, tip, "when");
        const row = el("div", {class: "row"}, tip);
        text("b", (d.sentiment >= 0 ? "+" : "") + d.sentiment.toFixed(2), row);
        text("span", d.count + " item" + (d.count === 1 ? "" : "s"), row);
        tip.style.display = "block";
        const r = svg.getBoundingClientRect();
        tip.style.left = Math.min(cx / W * r.width + 12, r.width - tip.offsetWidth - 4) + "px";
        tip.style.top = "8px";
      };
      hit.addEventListener("pointermove", on); hit.addEventListener("focus", on);
      const off = () => { tip.style.display = "none"; };
      hit.addEventListener("pointerleave", off); hit.addEventListener("blur", off);
    });
  }

  function table(parent, rows) {
    if (!rows.length) return;
    const det = el("details", {}, parent);
    text("summary", "Data table", det);
    const wrap = el("div", {class: "table-wrap"}, det);
    const t = el("table", {}, wrap);
    const head = el("tr", {}, el("thead", {}, t));
    ["Fiscal year end", "Revenue", "EPS", "Free cash flow", "Op. margin", "Share price"].forEach((h) => text("th", h, head));
    const body = el("tbody", {}, t);
    const money = (v) => v === null ? "n/a" : (Math.abs(v) >= 1e9 ? "$" + (v / 1e9).toFixed(1) + "B" : "$" + (v / 1e6).toFixed(0) + "M");
    rows.forEach((r) => {
      const tr = el("tr", {}, body);
      [r.fiscal_year_end, money(r.revenue), r.eps === null ? "n/a" : r.eps.toFixed(2), money(r.fcf),
       r.op_margin === null ? "n/a" : (r.op_margin * 100).toFixed(1) + "%", r.price === null ? "n/a" : "$" + r.price.toFixed(2)]
        .forEach((c) => text("td", c, tr));
    });
  }

  function render() {
    const root = document.getElementById("root");
    root.replaceChildren();
    DATA.companies.forEach((c) => {
      text("h2", c.ticker + (c.name && c.name !== c.ticker ? " — " + c.name : ""), root);
      const a = el("div", {class: "card"}, root);
      text("h3", "Business vs. share price (indexed, " + c.base + " = 100)", a);
      text("p", "Above 100 = grown since " + c.base + ". If the price line rises faster than revenue and EPS, the stock got more expensive; slower, it got cheaper.", a, "note");
      lineChart(a, c.indexed, {label: c.ticker + " indexed performance", fmt: (v) => v.toFixed(0), refLine: 100});
      table(a, c.table);
      if (c.pe.length) {
        const b = el("div", {class: "card"}, root);
        text("h3", "Price-to-earnings ratio", b);
        text("p", "Share price ÷ the latest annual EPS that had been published by that month.", b, "note");
        lineChart(b, c.pe, {label: c.ticker + " P/E over time", fmt: (v) => v >= 100 ? v.toFixed(0) : v.toFixed(1)});
      }
      if (c.news.length) {
        const n = el("div", {class: "card"}, root);
        text("h3", "News & social sentiment by day — " + c.news_summary, n);
        text("p", "Average sentiment of that day's items (−1 very negative … +1 very positive). Short-term signal only.", n, "note");
        barChart(n, c.news);
      }
      c.notes.forEach((s) => text("p", s, root, "note"));
    });
  }
  render();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  let lastW = window.innerWidth, timer = null;
  window.addEventListener("resize", () => {
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth; clearTimeout(timer); timer = setTimeout(render, 150);
  });
})();
</script>
</body>
</html>
"""


def performance_html(companies: list[dict], title: str | None = None, disclaimer: str = "") -> str:
    tickers = ", ".join(c["ticker"] for c in companies)
    title = title or f"{tickers}: business vs. share price"
    payload = json.dumps({"companies": companies}).replace("</", "<\\/")  # safe inside <script>
    return (PAGE.replace("__TITLE__", html.escape(title))
            .replace("__DISCLAIMER__", html.escape(disclaimer))
            .replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M"))
            .replace("__DATA__", payload))
