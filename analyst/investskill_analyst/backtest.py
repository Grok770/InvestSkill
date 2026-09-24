"""Walk-forward backtest of the ranking rule.

At every month-end the model ranks the universe using **only data up to that
date**, buys the top ``top_n`` names equal-weight, and holds them until the
next month-end. Weights take effect from the *next* trading day, and
turnover pays ``cost_bps`` each way.

Only price-derived factors are used (12-1 momentum, 6-month momentum, trend
vs. the 200-day SMA, low volatility). Free data sources don't provide
point-in-time fundamentals, and scoring history with *today's* P/E or ROE
would be look-ahead bias. Treat fundamental factors as a live overlay that
this test does not validate.

Known bias to keep in mind: the universe is a list of names that exist
*today* (survivorship bias), which flatters every long-only result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import DataProvider
from .indicators import TRADING_DAYS

PRICE_FACTORS = {"ret_12_1": 0.4, "ret_6m": 0.2, "dist_sma200": 0.2, "low_vol": 0.2}


@dataclass
class BacktestResult:
    start: str
    end: str
    months: int
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    benchmark_cagr: float
    benchmark_sharpe: float
    benchmark_max_drawdown: float
    excess_cagr: float
    monthly_hit_rate: float
    avg_turnover: float
    top_n: int
    universe_size: int
    last_holdings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _price_scores(closes: pd.DataFrame) -> pd.Series:
    """Composite price-factor z-score for each column, using all rows given."""
    c = closes.dropna(axis=1, thresh=TRADING_DAYS + 1)
    if c.shape[1] == 0:
        return pd.Series(dtype=float)
    last = c.iloc[-1]
    f = pd.DataFrame(
        {
            "ret_12_1": c.iloc[-22] / c.iloc[-TRADING_DAYS - 1] - 1,
            "ret_6m": last / c.iloc[-127] - 1,
            "dist_sma200": last / c.iloc[-200:].mean() - 1,
            "low_vol": -c.pct_change().iloc[-TRADING_DAYS:].std(),
        }
    )
    z = (f - f.mean()) / f.std().replace(0, np.nan)
    return (z.fillna(0) * pd.Series(PRICE_FACTORS)).sum(axis=1)


def _stats(daily: pd.Series) -> tuple[float, float, float, float]:
    equity = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily.mean() * TRADING_DAYS) / vol if vol else 0.0
    mdd = (equity / equity.cummax() - 1).min()
    return float(cagr), float(vol), float(sharpe), float(mdd)


def load_closes(provider: DataProvider, tickers: list[str], years: float) -> pd.DataFrame:
    series = {}
    for t in dict.fromkeys(tk.upper() for tk in tickers):
        try:
            series[t] = provider.history(t, years=years)["close"]
        except Exception:  # noqa: BLE001 - skip tickers without data
            continue
    return pd.DataFrame(series).sort_index()


def run_backtest(
    closes: pd.DataFrame,
    top_n: int = 10,
    cost_bps: float = 10.0,
    warmup_days: int = TRADING_DAYS + 1,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Backtest on a close-price panel (index=date, columns=tickers).

    Returns summary stats and a daily frame with ``strategy``/``benchmark``
    returns and equity curves. The benchmark is the equal-weight universe.
    """
    closes = closes.sort_index().ffill()
    rets = closes.pct_change().fillna(0.0)
    month_ends = closes.groupby(closes.index.to_period("M")).tail(1).index
    rebal_dates = [d for d in month_ends if closes.index.get_loc(d) >= warmup_days]
    if len(rebal_dates) < 3:
        raise ValueError(
            f"need at least {warmup_days} trading days plus 3 month-ends of history; "
            f"got {len(closes)} days"
        )

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    turnovers, prev_w = [], pd.Series(0.0, index=closes.columns)
    cost = pd.Series(0.0, index=closes.index)
    holdings: list[str] = []
    for i, d in enumerate(rebal_dates):
        scores = _price_scores(closes.loc[:d])
        holdings = list(scores.nlargest(min(top_n, len(scores))).index)
        w = pd.Series(0.0, index=closes.columns)
        w[holdings] = 1.0 / len(holdings)
        pos = closes.index.get_loc(d) + 1  # trade takes effect next session
        nxt = closes.index.get_loc(rebal_dates[i + 1]) + 1 if i + 1 < len(rebal_dates) else len(closes)
        weights.iloc[pos:nxt] = w.values
        turnover = float((w - prev_w).abs().sum())
        turnovers.append(turnover)
        if pos < len(closes):
            cost.iloc[pos] = turnover * cost_bps / 10_000
        prev_w = w

    start_pos = closes.index.get_loc(rebal_dates[0]) + 1
    strat = (weights * rets).sum(axis=1).iloc[start_pos:] - cost.iloc[start_pos:]
    valid = closes.notna().astype(float)
    bench = ((rets * valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)).fillna(0.0)
    bench = bench.iloc[start_pos:]

    s_cagr, s_vol, s_sharpe, s_mdd = _stats(strat)
    b_cagr, _, b_sharpe, b_mdd = _stats(bench)
    monthly = pd.DataFrame({"s": strat, "b": bench}).groupby(strat.index.to_period("M")).apply(
        lambda g: (1 + g).prod() - 1
    )
    daily = pd.DataFrame({"strategy": strat, "benchmark": bench})
    daily["strategy_equity"] = (1 + strat).cumprod()
    daily["benchmark_equity"] = (1 + bench).cumprod()

    result = BacktestResult(
        start=str(strat.index[0].date()),
        end=str(strat.index[-1].date()),
        months=len(monthly),
        cagr=round(s_cagr, 4),
        ann_vol=round(s_vol, 4),
        sharpe=round(s_sharpe, 2),
        max_drawdown=round(s_mdd, 4),
        benchmark_cagr=round(b_cagr, 4),
        benchmark_sharpe=round(b_sharpe, 2),
        benchmark_max_drawdown=round(b_mdd, 4),
        excess_cagr=round(s_cagr - b_cagr, 4),
        monthly_hit_rate=round(float((monthly["s"] > monthly["b"]).mean()), 3),
        avg_turnover=round(float(np.mean(turnovers)), 3),
        top_n=top_n,
        universe_size=closes.shape[1],
        last_holdings=holdings,
    )
    return result, daily
