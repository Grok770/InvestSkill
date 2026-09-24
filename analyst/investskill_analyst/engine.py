"""High-level operations shared by the CLI and the Claude agent's tools."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .backtest import BacktestResult, load_closes, run_backtest
from .data import DataProvider
from .factors import STYLE_WEIGHTS, Signal, build_universe_table, score_table, to_signal
from .financials import business_summary, derive, price_performance
from .news import NewsSignal, news_signal
from .quality import check_fundamentals, check_prices, cross_check, financials_age
from .signals import RiskSettings, TradePlan, build_trade_plan
from .universe import US_LARGE_CAP
from .verdict import TrackRecord, Verdict, make_verdict, track_record

BENCHMARK = "SPY"


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


def load_financials(provider: DataProvider, ticker: str) -> pd.DataFrame | None:
    """Multi-year financials if the provider supports them, else None."""
    getter = getattr(provider, "annual_financials", None)
    if getter is None:
        return None
    try:
        fin = getter(ticker)
    except Exception:  # noqa: BLE001 - history is an enhancement, not a requirement
        return None
    return fin if fin is not None and len(fin) >= 2 else None


def signal_from_verdict(v: Verdict, coverage: float) -> Signal:
    """Express the combined verdict as an InvestSkill signal block."""
    signal = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(v.action, "NEUTRAL")
    conviction = "STRONG" if v.label.startswith("STRONG") else "WEAK" if v.rails and \
        v.action == "HOLD" and v.score >= 6 else "MODERATE"
    vals = [x for x in v.pillars.values() if x is not None and not pd.isna(x)]
    spread = max(vals) - min(vals) if vals else 10
    if len(vals) == 3 and spread <= 3.5 and coverage >= 0.7:
        confidence = "HIGH"
    elif len(vals) >= 2 and spread <= 5.5 and coverage >= 0.5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    return Signal(signal, v.action, conviction, confidence, "MEDIUM-TERM", v.score, list(v.rails))


@dataclass
class NewsOptions:
    """Where to get news and how to score it. ``sources=[]`` turns news off."""
    sources: list[str] = field(default_factory=list)
    scorer: object = None          # news.LexiconScorer / news.ClaudeScorer
    news_file: str | None = None
    days: int = 14


def get_news(provider: DataProvider, ticker: str, opts: NewsOptions | None,
             company: str | None = None, on_error=_warn):
    if not opts or not opts.sources:
        return None, []
    client = getattr(provider, "client", None)  # reuse the EDGAR client (cache, user agent)
    return news_signal(ticker, opts.sources, opts.scorer, company, opts.news_file,
                       sec_client=client, days=opts.days,
                       on_error=(lambda s, e: on_error(f"{ticker} news/{s}", e)) if on_error else None)


@dataclass
class Analysis:
    ticker: str
    row: pd.Series
    signal: Signal
    tech: dict
    fund: object
    plan: TradePlan
    universe_size: int
    verdict: Verdict | None = None
    business: dict | None = None
    data_quality: list[str] = field(default_factory=list)
    news: NewsSignal | None = None


def analyze(provider: DataProvider, ticker: str, peers: list[str] | None = None,
            style: str = "balanced", risk: RiskSettings | None = None,
            on_error=_warn, today: date | None = None,
            news: NewsOptions | None = None) -> Analysis:
    """Score one ticker vs. peers, add its business trend, and give the verdict + trade plan."""
    ticker = ticker.upper()
    universe = list(dict.fromkeys([ticker, *(peers or US_LARGE_CAP)]))
    res = screen(provider, universe, style=style, on_error=on_error)
    if ticker not in res.scored.index:
        raise LookupError(f"could not load data for {ticker}")
    row = res.scored.loc[ticker]
    tech = res.techs[ticker]
    fund = res.funds[ticker]

    fin = load_financials(provider, ticker)
    business = business_summary(fin) if fin is not None else None
    signal, _ = get_news(provider, ticker, news, fund.name, on_error)
    verdict = make_verdict(ticker, float(row["score"]), tech, business, signal)
    sig = signal_from_verdict(verdict, float(row.get("coverage", 0) or 0))
    plan = build_trade_plan(ticker, sig, tech, risk)

    quality = check_prices(provider.history(ticker, 1), today=today or date.today())
    quality += check_fundamentals(fund)
    if fin is not None:
        quality += financials_age(fin, today)
    secondary = getattr(provider, "secondary", None)
    if secondary is not None:
        try:
            quality += cross_check(fund, secondary.fundamentals(ticker))
        except Exception:  # noqa: BLE001 - cross-check is best effort
            pass
    return Analysis(ticker, row, sig, tech, fund, plan, len(res.scored),
                    verdict, business, quality, signal)


@dataclass
class CompanyHistory:
    ticker: str
    stock: dict
    financials: pd.DataFrame | None
    business: dict | None
    track: TrackRecord | None
    benchmark: str


def company_history(provider: DataProvider, ticker: str, years: int = 10,
                    benchmark: str = BENCHMARK) -> CompanyHistory:
    """The company's past performance: the stock, the business, and the indicator."""
    ticker = ticker.upper()
    df = provider.history(ticker, years=years)
    bench = None
    try:
        bench = provider.history(benchmark, years=years)["close"]
    except Exception:  # noqa: BLE001 - benchmark is optional
        benchmark = ""
    fin = load_financials(provider, ticker)
    return CompanyHistory(
        ticker=ticker,
        stock=price_performance(df["close"], bench),
        financials=derive(fin) if fin is not None else None,
        business=business_summary(fin) if fin is not None else None,
        track=track_record(df),
        benchmark=benchmark,
    )


def watchlist(provider: DataProvider, tickers: list[str], peers: list[str] | None = None,
              style: str = "balanced", on_error=_warn,
              news: NewsOptions | None = None) -> list[Verdict]:
    """Buy/sell verdicts for several tickers, ranked against one peer universe."""
    tickers = [t.upper() for t in tickers]
    res = screen(provider, list(dict.fromkeys([*tickers, *(peers or US_LARGE_CAP)])),
                 style=style, on_error=on_error)
    out = []
    for t in tickers:
        if t not in res.scored.index:
            continue
        fin = load_financials(provider, t)
        signal, _ = get_news(provider, t, news, res.funds[t].name, on_error)
        out.append(make_verdict(t, float(res.scored.loc[t, "score"]), res.techs[t],
                                business_summary(fin) if fin is not None else None, signal))
    return sorted(out, key=lambda v: v.score, reverse=True)


def backtest(provider: DataProvider, tickers: list[str], years: float = 5.0,
             top_n: int = 10, cost_bps: float = 10.0) -> tuple[BacktestResult, pd.DataFrame]:
    closes = load_closes(provider, tickers, years)
    if closes.shape[1] < top_n + 1:
        raise ValueError(f"only {closes.shape[1]} tickers loaded; need more than top_n={top_n}")
    return run_backtest(closes, top_n=top_n, cost_bps=cost_bps)


def chart_data(provider: DataProvider, ticker: str, years: int = 10,
               benchmark: str = BENCHMARK, news: NewsOptions | None = None,
               on_error=_warn) -> dict:
    """Data for the business-vs-price charts (see charts.py)."""
    from .charts import company_chart_data

    ticker = ticker.upper()
    prices = provider.history(ticker, years=years)["close"]
    try:
        bench = provider.history(benchmark, years=years)["close"]
    except Exception:  # noqa: BLE001 - benchmark is optional
        bench = None
    try:
        name = provider.fundamentals(ticker).name
    except Exception:  # noqa: BLE001
        name = None
    signal, _ = get_news(provider, ticker, news, name, on_error)
    return company_chart_data(ticker, prices, load_financials(provider, ticker), bench, signal, name)
