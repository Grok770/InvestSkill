"""Markdown rendering in the InvestSkill house format (tables + Signal block)."""

from __future__ import annotations

import pandas as pd

from . import DISCLAIMER
from .backtest import BacktestResult
from .data import Fundamentals
from .factors import FACTORS, Signal, to_signal
from .signals import TradePlan

SCORE_GUIDE = (
    "Score Guide: 8.0–10.0 Strongly Bullish | 6.0–7.9 Moderately Bullish | "
    "4.0–5.9 Neutral | 2.0–3.9 Moderately Bearish | 0.0–1.9 Strongly Bearish"
)


def _pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def _num(x, digits: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


def signal_block(sig: Signal) -> str:
    """The standard box-drawn InvestSkill Investment Signal block."""
    width = 46
    rows = [
        ("Signal:", sig.signal),
        ("Confidence:", sig.confidence),
        ("Horizon:", sig.horizon),
        ("Score:", f"{sig.score:.1f} / 10"),
    ]
    rows2 = [("Action:", sig.action), ("Conviction:", sig.conviction)]

    def line(label: str, value: str) -> str:
        return "║ " + f"{label:<13}{value}".ljust(width - 1) + "║"

    return "\n".join(
        ["╔" + "═" * width + "╗",
         "║" + "INVESTMENT SIGNAL".center(width) + "║",
         "╠" + "═" * width + "╣",
         *[line(a, b) for a, b in rows],
         "╠" + "═" * width + "╣",
         *[line(a, b) for a, b in rows2],
         "╚" + "═" * width + "╝"]
    )


def screen_markdown(scored: pd.DataFrame, techs: dict, funds: dict, top: int = 15,
                    style: str = "balanced") -> str:
    lines = [
        f"# Stock Screen — {len(scored)} names ranked ({style} weights)",
        "",
        "| # | Ticker | Score | Signal | Value | Quality | Growth | Momentum | Low risk | 12-1 mo | Price | Sector |",
        "|---|--------|------:|--------|------:|--------:|-------:|---------:|---------:|--------:|------:|--------|",
    ]
    for i, (t, row) in enumerate(scored.head(top).iterrows(), 1):
        sig = to_signal(row, techs.get(t))
        fz = [f"{row[f]:+.2f}" if pd.notna(row[f]) else "n/a" for f in FACTORS]
        tech = techs.get(t, {})
        lines.append(
            f"| {i} | **{t}** | {row['score']:.1f} | {sig.action} ({sig.conviction.lower()}) | "
            + " | ".join(fz)
            + f" | {_pct(tech.get('ret_12_1'))} | {_num(tech.get('price'))} | "
            + f"{funds[t].sector or 'n/a'} |"
        )
    bottom = scored.tail(min(5, max(len(scored) - top, 0)))
    if len(bottom):
        lines += ["", "**Avoid list (bottom of the ranking):** "
                  + ", ".join(f"{t} ({r['score']:.1f})" for t, r in bottom.iterrows())]
    lines += [
        "",
        "Factor columns are cross-sectional z-scores (0 = universe average, +1 = one "
        "standard deviation better). Scores rank names *within this universe only*.",
        "",
        "**Next steps:** run `investskill-analyst analyze <TICKER>` for a trade plan, or "
        "`investskill-analyst research <TICKER>` for a full Claude-written research note.",
        "",
        f"**Disclaimer:** {DISCLAIMER}",
    ]
    return "\n".join(lines)


def analysis_markdown(ticker: str, row: pd.Series, sig: Signal, tech: dict,
                      fund: Fundamentals, plan: TradePlan, universe_size: int) -> str:
    f = fund
    lines = [
        f"# {ticker} — {f.name or ticker}" + (f" · {f.sector}" if f.sector else ""),
        "",
        f"Data as of {tech['as_of']} · ranked #{int(row['rank'])} of {universe_size} "
        f"in the peer universe · input coverage {row['coverage']:.0%}",
        "",
        "## Factor profile",
        "",
        "| Factor | z-score | Read |",
        "|--------|--------:|------|",
    ]
    for fac in FACTORS:
        z = row.get(fac)
        read = ("n/a" if pd.isna(z) else "strong" if z > 0.5 else "above avg" if z > 0
                else "below avg" if z > -0.5 else "weak")
        lines.append(f"| {fac.replace('_', ' ').title()} | {_num(z)} | {read} |")

    lines += [
        "",
        "## Key metrics",
        "",
        "| Metric | Value | Metric | Value |",
        "|--------|------:|--------|------:|",
        f"| P/E (ttm) | {_num(f.trailing_pe, 1)} | ROE | {_pct(f.roe)} |",
        f"| Forward P/E | {_num(f.forward_pe, 1)} | Operating margin | {_pct(f.operating_margin)} |",
        f"| EV/EBITDA | {_num(f.ev_to_ebitda, 1)} | Revenue growth | {_pct(f.revenue_growth)} |",
        f"| FCF yield | {_pct(f.fcf_yield)} | Earnings growth | {_pct(f.earnings_growth)} |",
        f"| Debt/Equity | {_num(f.debt_to_equity)} | Beta | {_num(f.beta)} |",
        "",
        "## Technicals",
        "",
        "| Price | SMA50 | SMA200 | RSI(14) | MACD hist | ATR(14) | 52w high | 1y vol | 1y max DD |",
        "|------:|------:|-------:|--------:|----------:|--------:|---------:|-------:|----------:|",
        f"| {_num(tech['price'])} | {_num(tech['sma50'])} | {_num(tech['sma200'])} | "
        f"{_num(tech['rsi14'], 0)} | {_num(tech['macd_hist'])} | {_num(tech['atr14'])} | "
        f"{_num(tech['high_52w'])} | {_pct(tech['vol_1y'])} | {_pct(tech['max_drawdown_1y'])} |",
        "",
        f"Returns: 1m {_pct(tech['ret_1m'])} · 3m {_pct(tech['ret_3m'])} · "
        f"6m {_pct(tech['ret_6m'])} · 12m {_pct(tech['ret_12m'])}",
        "",
        "## Trade plan",
        "",
        f"**Setup:** {plan.setup}",
        "",
    ]
    if plan.entry is not None:
        lines += [
            "| Entry | Entry zone | Stop | Target 1 (2R) | Target 2 (3R) | Shares | Position | At risk |",
            "|------:|-----------|-----:|--------------:|--------------:|-------:|---------:|--------:|",
            f"| {_num(plan.entry)} | {_num(plan.entry_zone[0])}–{_num(plan.entry_zone[1])} | "
            f"{_num(plan.stop)} | {_num(plan.target_1)} | {_num(plan.target_2)} | {plan.shares} | "
            f"${_num(plan.position_value, 0)} | ${_num(plan.capital_at_risk, 0)} |",
        ]
        if plan.tranches:
            lines += ["", "Staged entry: " + " · ".join(
                f"{tr['pct']}% ({tr['shares']} sh) @ {_num(tr['price'])} {tr['when']}"
                for tr in plan.tranches)]
    elif plan.stop is not None:
        lines.append(f"Confirmation level: a close below **{_num(plan.stop)}** (1×ATR under price).")
    if plan.notes:
        lines += [""] + [f"- {n}" for n in plan.notes]

    lines += [
        "",
        "## Thesis invalidation",
        "",
        f"- A close below the stop ({_num(plan.stop)}) — exit, no averaging down." if plan.stop
        else "- Score falls below 4.0 on the next re-rank.",
        "- Composite score drops two bands (e.g. from ≥ 8 to < 6) at the next monthly re-rank.",
        "- Re-run after the next earnings release, a ±15% price move, or 30 days — whichever is first.",
        "",
        signal_block(sig),
        "",
        SCORE_GUIDE,
        "Confidence: HIGH (≥ 80% inputs, factors agree) | MEDIUM (mixed) | LOW (sparse or conflicting)",
        "Horizon: SHORT-TERM (1 week–3 months) | MEDIUM-TERM (3 months–1 year) | LONG-TERM (1+ years)",
        "",
        f"**Disclaimer:** {DISCLAIMER}",
    ]
    return "\n".join(lines)


def backtest_markdown(r: BacktestResult, cost_bps: float) -> str:
    return "\n".join([
        f"# Backtest — top {r.top_n} of {r.universe_size}, monthly rebalance",
        "",
        f"{r.start} → {r.end} · {r.months} months · {cost_bps:.0f} bps per unit of turnover",
        "",
        "| | Strategy | Equal-weight universe |",
        "|---|---:|---:|",
        f"| CAGR | {_pct(r.cagr)} | {_pct(r.benchmark_cagr)} |",
        f"| Sharpe (rf = 0) | {r.sharpe:.2f} | {r.benchmark_sharpe:.2f} |",
        f"| Max drawdown | {_pct(r.max_drawdown)} | {_pct(r.benchmark_max_drawdown)} |",
        f"| Annual volatility | {_pct(r.ann_vol)} | |",
        "",
        f"Excess CAGR {_pct(r.excess_cagr)} · beat the benchmark in {r.monthly_hit_rate:.0%} of months · "
        f"average turnover {r.avg_turnover:.0%} per rebalance",
        "",
        "Current holdings: " + ", ".join(r.last_holdings),
        "",
        "**Read this before trusting it:** price factors only (fundamentals are not "
        "point-in-time in free data); today's universe (survivorship bias); no slippage "
        "beyond the flat cost; past performance does not predict future returns.",
        "",
        f"**Disclaimer:** {DISCLAIMER}",
    ])
