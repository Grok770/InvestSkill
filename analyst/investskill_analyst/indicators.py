"""Technical and risk indicators on a price DataFrame (see ``data.PRICE_COLUMNS``).

Conventions follow the InvestSkill ``technical-analysis`` and
``risk-stress-test`` frameworks: Wilder RSI(14), MACD(12, 26, 9), ATR(14),
252 trading days per year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def total_return(close: pd.Series, days: int, skip: int = 0) -> float | None:
    """Return from ``days`` ago to ``skip`` days ago (12-1 momentum = 252, 21)."""
    if len(close) <= days:
        return None
    end = close.iloc[-1 - skip]
    start = close.iloc[-1 - days]
    return float(end / start - 1)


def annualized_vol(close: pd.Series, days: int = TRADING_DAYS) -> float | None:
    rets = close.pct_change().dropna().iloc[-days:]
    if len(rets) < 20:
        return None
    return float(rets.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series) -> float:
    """Most negative peak-to-trough decline, as a negative decimal."""
    peak = close.cummax()
    return float((close / peak - 1).min())


def beta(close: pd.Series, bench: pd.Series, days: int = TRADING_DAYS) -> float | None:
    joined = pd.concat([close.pct_change(), bench.pct_change()], axis=1, join="inner").dropna()
    joined = joined.iloc[-days:]
    if len(joined) < 60:
        return None
    var = joined.iloc[:, 1].var()
    return float(joined.cov().iloc[0, 1] / var) if var else None


def technical_snapshot(df: pd.DataFrame) -> dict:
    """The per-ticker technical readout used by the trade planner and the agent."""
    close = df["close"]
    last = float(close.iloc[-1])
    ma50, ma200 = sma(close, 50).iloc[-1], sma(close, 200).iloc[-1]
    m = macd(close).iloc[-1]
    window = close.iloc[-TRADING_DAYS:]

    def _f(x):
        return None if x is None or pd.isna(x) else round(float(x), 4)

    return {
        "price": round(last, 2),
        "as_of": str(close.index[-1].date()),
        "sma50": _f(ma50),
        "sma200": _f(ma200),
        "above_sma50": bool(last > ma50) if not pd.isna(ma50) else None,
        "above_sma200": bool(last > ma200) if not pd.isna(ma200) else None,
        "golden_cross": bool(ma50 > ma200) if not (pd.isna(ma50) or pd.isna(ma200)) else None,
        "rsi14": _f(rsi(close).iloc[-1]),
        "macd_hist": _f(m["hist"]),
        "atr14": _f(atr(df).iloc[-1]),
        "high_52w": round(float(window.max()), 2),
        "low_52w": round(float(window.min()), 2),
        "pct_from_52w_high": round(last / float(window.max()) - 1, 4),
        "ret_1m": _f(total_return(close, 21)),
        "ret_3m": _f(total_return(close, 63)),
        "ret_6m": _f(total_return(close, 126)),
        "ret_12m": _f(total_return(close, 252)),
        "ret_12_1": _f(total_return(close, 252, skip=21)),
        "vol_1y": _f(annualized_vol(close)),
        "max_drawdown_1y": _f(max_drawdown(window)),
    }
