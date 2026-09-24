"""High-level operations shared by the CLI and the Claude agent's tools."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, load_closes, run_backtest
from .data import DataProvider
from .factors import STYLE_WEIGHTS, Signal, build_universe_table, score_table, to_signal
from .signals import RiskSettings, TradePlan, build_trade_plan
from .universe import US_LARGE_CAP


def _warn(ticker: str, exc: Exception) -> None:
    print(f"warning: skipped {ticker}: {exc}", file=sys.stderr)


@dataclass
class ScreenResult:
    scored: pd.DataFrame
    techs: dict
    funds: dict
    style: str

    def leaderboard(self, top: int = 15) -> list[dict]:
        out = []
        for t, row in self.scored.head(top).iterrows():
            sig = to_signal(row, self.techs.get(t))
            out.append({
                "ticker": t,
                "score": row["score"],
                "action": sig.action,
                "conviction": sig.conviction,
                "confidence": sig.confidence,
                "sector": self.funds[t].sector,
                "price": self.techs[t]["price"],
                "factors": {k: (None if pd.isna(row[k]) else round(float(row[k]), 2))
                            for k in STYLE_WEIGHTS["balanced"]},
                "ret_12_1": self.techs[t].get("ret_12_1"),
            })
        return out


def screen(provider: DataProvider, tickers: list[str], style: str = "balanced",
           on_error=_warn) -> ScreenResult:
    if style not in STYLE_WEIGHTS:
        raise ValueError(f"unknown style {style!r}; choose from {sorted(STYLE_WEIGHTS)}")
    metrics, techs, funds = build_universe_table(provider, tickers, on_error=on_error)
    if metrics.empty:
        raise RuntimeError("no tickers could be loaded — check the symbols and your data provider")
    return ScreenResult(score_table(metrics, STYLE_WEIGHTS[style]), techs, funds, style)


@dataclass
class Analysis:
    ticker: str
    row: pd.Series
    signal: Signal
    tech: dict
    fund: object
    plan: TradePlan
    universe_size: int


def analyze(provider: DataProvider, ticker: str, peers: list[str] | None = None,
            style: str = "balanced", risk: RiskSettings | None = None,
            on_error=_warn) -> Analysis:
    """Score one ticker *relative to a peer universe* and build its trade plan."""
    ticker = ticker.upper()
    universe = list(dict.fromkeys([ticker, *(peers or US_LARGE_CAP)]))
    res = screen(provider, universe, style=style, on_error=on_error)
    if ticker not in res.scored.index:
        raise LookupError(f"could not load data for {ticker}")
    row = res.scored.loc[ticker]
    tech = res.techs[ticker]
    sig = to_signal(row, tech)
    plan = build_trade_plan(ticker, sig, tech, risk)
    return Analysis(ticker, row, sig, tech, res.funds[ticker], plan, len(res.scored))


def backtest(provider: DataProvider, tickers: list[str], years: float = 5.0,
             top_n: int = 10, cost_bps: float = 10.0) -> tuple[BacktestResult, pd.DataFrame]:
    closes = load_closes(provider, tickers, years)
    if closes.shape[1] < top_n + 1:
        raise ValueError(f"only {closes.shape[1]} tickers loaded; need more than top_n={top_n}")
    return run_backtest(closes, top_n=top_n, cost_bps=cost_bps)
