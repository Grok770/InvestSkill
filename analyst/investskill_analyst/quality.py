"""Data-quality checks: catch bad inputs before they become bad trades.

Each check returns a list of human-readable warnings (empty = clean). They are
shown in every ``analyze`` report and passed to the research agent, so a
stale price, an unadjusted split, or two sources that disagree is visible
rather than silently scored.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .data import Fundamentals

# Only fields both sources define the same way. Revenue growth (Yahoo: latest
# quarter YoY; SEC: fiscal year) and ROE (different equity averaging) legitimately
# differ, so comparing them would only produce noise.
CROSS_CHECK_FIELDS = ("trailing_pe", "price_to_book", "gross_margin",
                      "operating_margin", "market_cap")


def check_prices(df: pd.DataFrame, today: date | None = None, max_stale_days: int = 5) -> list[str]:
    warnings = []
    close = df["close"].dropna()
    if close.empty:
        return ["No price data."]
    last = close.index[-1]
    if today is not None:
        stale = len(pd.bdate_range(last + pd.Timedelta(days=1), pd.Timestamp(today)))
        if stale > max_stale_days:
            warnings.append(f"Prices are stale: last bar {last.date()} is {stale} trading days old.")
    if (close <= 0).any():
        warnings.append("Non-positive prices in history — the feed is corrupt for this ticker.")
    gaps = close.index.to_series().diff().dt.days
    if (gaps > 7).any():
        worst = gaps.idxmax()
        warnings.append(f"Missing data: a {int(gaps.max())}-day gap in prices ending {worst.date()}.")
    jumps = close.pct_change().abs()
    big = jumps[jumps > 0.35]
    if len(big):
        d = big.index[-1]
        warnings.append(
            f"{len(big)} one-day move(s) over 35% (latest {d.date()}, {jumps[d]:.0%}). "
            "Confirm it's real news and not an unadjusted split."
        )
    if "volume" in df and (df["volume"].iloc[-20:] == 0).sum() >= 5:
        warnings.append("Zero volume on 5+ of the last 20 days — illiquid or bad feed.")
    return warnings


def check_fundamentals(f: Fundamentals) -> list[str]:
    warnings = []
    if f.trailing_pe is not None and f.trailing_pe < 0:
        warnings.append("Trailing P/E is negative: the company lost money over the last year.")
    elif f.trailing_pe is not None and f.trailing_pe > 200:
        warnings.append(f"Trailing P/E {f.trailing_pe:.0f} — earnings are near zero, so value metrics are unreliable.")
    if f.roe is not None and f.roe > 1.0:
        warnings.append(
            f"ROE {f.roe:.0%} — usually small or negative equity from buybacks, not true "
            "profitability. Compare ROA and ROIC instead."
        )
    for name in ("gross_margin", "operating_margin"):
        v = getattr(f, name)
        if v is not None and not -2 <= v <= 1:
            warnings.append(f"{name.replace('_', ' ')} {v:.0%} is out of range — likely a data error.")
    missing = [k for k in ("trailing_pe", "roe", "revenue_growth", "fcf_yield")
               if getattr(f, k) is None]
    if missing:
        warnings.append("Missing inputs: " + ", ".join(missing) + ".")
    return warnings


def cross_check(primary: Fundamentals, secondary: Fundamentals, tolerance: float = 0.15,
                fields: tuple[str, ...] = CROSS_CHECK_FIELDS) -> list[str]:
    """Flag fields where two independent sources disagree by more than ``tolerance``."""
    warnings = []
    for name in fields:
        a, b = getattr(primary, name), getattr(secondary, name)
        if a is None or b is None:
            continue
        scale = max(abs(a), abs(b))
        if scale == 0:
            continue
        diff = abs(a - b) / scale
        if diff > tolerance:
            warnings.append(
                f"{name}: sources disagree by {diff:.0%} "
                f"({primary.sources.get(name, 'primary')}: {a:,.4g} vs "
                f"{secondary.sources.get(name, 'secondary')}: {b:,.4g})."
            )
    return warnings


def financials_age(fin: pd.DataFrame, today: date | None = None) -> list[str]:
    age = (pd.Timestamp(today or date.today()) - fin.index[-1]).days
    if age > 15 * 30:
        return [f"Latest annual report is for a fiscal year that ended {age // 30} months ago."]
    return []
