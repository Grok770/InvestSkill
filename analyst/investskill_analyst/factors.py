"""Cross-sectional multi-factor model.

Five factor families, each an average of winsorized z-scores of its metrics
(sign-adjusted so that *higher is always better*):

    value     earnings yield, forward earnings yield, FCF yield, EBITDA/EV, book/price
    quality   ROE, ROA, gross margin, operating margin, low leverage, liquidity
    growth    revenue growth, earnings growth
    momentum  12-1 month return, 6-month return, distance above the 200-day SMA
    low_risk  low volatility, shallow drawdown, low beta

The composite is a weighted blend of the families (weights renormalize over
whatever data exists for a ticker). The composite is then ranked within the
universe and mapped onto InvestSkill's 0–10 Score Guide, so the output drops
straight into the standard Investment Signal block.

Scores are *relative*: a 9/10 means "top decile of this universe today", not
"this stock will go up".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DataProvider, Fundamentals
from .indicators import technical_snapshot

DEFAULT_WEIGHTS: dict[str, float] = {
    "value": 0.20,
    "quality": 0.25,
    "growth": 0.15,
    "momentum": 0.25,
    "low_risk": 0.15,
}

# Investor-style presets, selectable with --style on the CLI.
STYLE_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": DEFAULT_WEIGHTS,
    "value": {"value": 0.40, "quality": 0.30, "growth": 0.05, "momentum": 0.10, "low_risk": 0.15},
    "growth": {"value": 0.05, "quality": 0.20, "growth": 0.40, "momentum": 0.30, "low_risk": 0.05},
    "momentum": {"value": 0.05, "quality": 0.15, "growth": 0.10, "momentum": 0.60, "low_risk": 0.10},
    "defensive": {"value": 0.20, "quality": 0.35, "growth": 0.05, "momentum": 0.10, "low_risk": 0.30},
}

# metric name -> (factor family, sign). sign=-1 means "lower is better".
METRICS: dict[str, tuple[str, int]] = {
    "earnings_yield": ("value", 1),
    "fwd_earnings_yield": ("value", 1),
    "fcf_yield": ("value", 1),
    "ebitda_ev": ("value", 1),
    "book_to_price": ("value", 1),
    "roe": ("quality", 1),
    "roa": ("quality", 1),
    "gross_margin": ("quality", 1),
    "operating_margin": ("quality", 1),
    "debt_to_equity": ("quality", -1),
    "current_ratio": ("quality", 1),
    "revenue_growth": ("growth", 1),
    "earnings_growth": ("growth", 1),
    "ret_12_1": ("momentum", 1),
    "ret_6m": ("momentum", 1),
    "dist_sma200": ("momentum", 1),
    "vol_1y": ("low_risk", -1),
    "max_drawdown_1y": ("low_risk", 1),  # already negative; closer to 0 is better
    "beta": ("low_risk", -1),
}

FACTORS = list(DEFAULT_WEIGHTS)


def _inv(x: float | None) -> float | None:
    """Inverse of a multiple; a negative multiple (losses) maps to a negative yield."""
    if x is None or x == 0 or not np.isfinite(x):
        return None
    return 1.0 / x


def raw_metrics(f: Fundamentals, tech: dict) -> dict[str, float | None]:
    """Flatten fundamentals + technicals into the metric set the model scores."""
    dist = None
    if tech.get("sma200"):
        dist = tech["price"] / tech["sma200"] - 1
    return {
        "earnings_yield": _inv(f.trailing_pe),
        "fwd_earnings_yield": _inv(f.forward_pe),
        "fcf_yield": f.fcf_yield,
        "ebitda_ev": _inv(f.ev_to_ebitda),
        "book_to_price": _inv(f.price_to_book),
        "roe": f.roe,
        "roa": f.roa,
        "gross_margin": f.gross_margin,
        "operating_margin": f.operating_margin,
        "debt_to_equity": f.debt_to_equity,
        "current_ratio": f.current_ratio,
        "revenue_growth": f.revenue_growth,
        "earnings_growth": f.earnings_growth,
        "ret_12_1": tech.get("ret_12_1"),
        "ret_6m": tech.get("ret_6m"),
        "dist_sma200": dist,
        "vol_1y": tech.get("vol_1y"),
        "max_drawdown_1y": tech.get("max_drawdown_1y"),
        "beta": f.beta,
    }


def _zscore(col: pd.Series, clip: float = 0.05) -> pd.Series:
    valid = col.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=col.index)  # too few peers to compare
    if valid.std() == 0:
        return col * 0.0
    lo, hi = valid.quantile(clip), valid.quantile(1 - clip)
    w = col.clip(lo, hi)
    return (w - w.mean()) / w.std()


def score_table(
    metrics: pd.DataFrame, weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """Score a metrics table (index=ticker, columns ⊆ METRICS).

    Returns factor z-scores, ``composite``, ``score`` (0–10, rank-based),
    ``coverage`` (share of metrics present) and ``rank`` (1 = best).
    """
    weights = weights or DEFAULT_WEIGHTS
    z = pd.DataFrame(index=metrics.index)
    for name, (_, sign) in METRICS.items():
        if name in metrics:
            z[name] = sign * _zscore(pd.to_numeric(metrics[name], errors="coerce"))

    out = pd.DataFrame(index=metrics.index)
    for fam in FACTORS:
        cols = [m for m, (f, _) in METRICS.items() if f == fam and m in z]
        out[fam] = z[cols].mean(axis=1, skipna=True) if cols else np.nan

    w = pd.Series(weights, dtype=float).reindex(FACTORS).fillna(0.0)
    present = out[FACTORS].notna()
    wsum = present.mul(w, axis=1).sum(axis=1).replace(0, np.nan)
    out["composite"] = out[FACTORS].fillna(0.0).mul(w, axis=1).sum(axis=1) / wsum

    n = out["composite"].notna().sum()
    pct = out["composite"].rank(pct=True, method="average")
    # Spread ranks across 0.5–9.5 so a universe's best isn't pinned to exactly 10.
    out["score"] = ((pct - 0.5 / max(n, 1)) * 10).clip(0, 10).round(1) if n else np.nan
    out["coverage"] = metrics.reindex(columns=list(METRICS)).notna().mean(axis=1).round(2)
    out["rank"] = out["composite"].rank(ascending=False, method="min")
    return out.sort_values("composite", ascending=False)


@dataclass
class Signal:
    """InvestSkill Investment Signal block fields."""

    signal: str       # BULLISH / NEUTRAL / BEARISH
    action: str       # BUY / HOLD / SELL
    conviction: str   # STRONG / MODERATE / WEAK
    confidence: str   # HIGH / MEDIUM / LOW
    horizon: str      # SHORT / MEDIUM / LONG-TERM
    score: float
    notes: list[str] = field(default_factory=list)


def to_signal(row: pd.Series, tech: dict | None = None) -> Signal:
    """Map a scored row onto the InvestSkill Score Guide, with a trend gate.

    Score Guide: 8.0–10 Strongly Bullish | 6.0–7.9 Moderately Bullish |
    4.0–5.9 Neutral | 2.0–3.9 Moderately Bearish | 0.0–1.9 Strongly Bearish
    """
    s = float(row["score"]) if pd.notna(row.get("score")) else 5.0
    notes: list[str] = []
    if s >= 8:
        sig, act, conv = "BULLISH", "BUY", "STRONG"
    elif s >= 6:
        sig, act, conv = "BULLISH", "BUY", "MODERATE"
    elif s >= 4:
        sig, act, conv = "NEUTRAL", "HOLD", "MODERATE"
    elif s >= 2:
        sig, act, conv = "BEARISH", "SELL", "MODERATE"
    else:
        sig, act, conv = "BEARISH", "SELL", "STRONG"

    # Trend gate: don't buy a stock in a primary downtrend on factor rank alone.
    if tech and act == "BUY" and tech.get("above_sma200") is False:
        conv = "WEAK"
        notes.append("Price is below its 200-day SMA — wait for a trend reclaim before buying.")
    if tech and tech.get("rsi14") is not None:
        if tech["rsi14"] >= 75 and act == "BUY":
            notes.append(f"RSI {tech['rsi14']:.0f} is overbought — scale in or wait for a pullback.")
        elif tech["rsi14"] <= 25 and act == "SELL":
            notes.append(f"RSI {tech['rsi14']:.0f} is oversold — avoid chasing the exit.")

    fz = [row.get(f) for f in FACTORS if pd.notna(row.get(f))]
    agree = (np.sign(fz) == np.sign(np.mean(fz))).mean() if fz else 0.0
    cov = float(row.get("coverage", 0) or 0)
    if cov >= 0.8 and agree >= 0.8:
        conf = "HIGH"
    elif cov >= 0.5 and agree >= 0.6:
        conf = "MEDIUM"
    else:
        conf = "LOW"
    if cov < 0.5:
        notes.append(f"Only {cov:.0%} of model inputs available — treat as low-information.")

    return Signal(sig, act, conv, conf, "MEDIUM-TERM", round(s, 1), notes)


def build_universe_table(
    provider: DataProvider,
    tickers: list[str],
    years: float = 2.0,
    on_error=None,
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, Fundamentals]]:
    """Fetch data for every ticker and return (metrics, technicals, fundamentals).

    Tickers that fail to load are skipped (and reported through ``on_error``)
    rather than aborting the whole screen.
    """
    rows, techs, funds = {}, {}, {}
    for t in dict.fromkeys(tk.upper() for tk in tickers):
        try:
            hist = provider.history(t, years=years)
            if len(hist) < 60:
                raise LookupError(f"only {len(hist)} bars of history")
            tech = technical_snapshot(hist)
            fund = provider.fundamentals(t)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not sink a screen
            if on_error:
                on_error(t, exc)
            continue
        techs[t], funds[t] = tech, fund
        rows[t] = raw_metrics(fund, tech)
    return pd.DataFrame.from_dict(rows, orient="index"), techs, funds
