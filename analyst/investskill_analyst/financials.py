"""Company track record: multi-year financials and stock performance.

Two questions, answered separately because they often disagree:

* **How has the business done?** ``derive`` + ``business_summary`` turn an
  annual-financials table (from SEC EDGAR, Yahoo, or your CSV) into margins,
  returns on capital, growth rates, consistency counts, and a 0–10
  *business trend score* with the reasons behind it.
* **How has the stock done?** ``price_performance`` gives annualized returns
  over 1/3/5/10 years, calendar-year returns vs. a benchmark, volatility and
  drawdowns.

Annual-financials table contract: index = fiscal-year-end dates (ascending),
columns ⊆ ``ANNUAL_COLUMNS``, values in USD (EPS in USD/share). An optional
``filed`` column holds the date the numbers first became public.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import TRADING_DAYS, max_drawdown

ANNUAL_COLUMNS = [
    "revenue", "gross_profit", "operating_income", "net_income", "eps_diluted",
    "operating_cash_flow", "capex", "free_cash_flow", "dna", "dividends_paid",
    "total_assets", "total_liabilities", "current_assets", "current_liabilities",
    "equity", "cash", "long_term_debt", "shares_diluted",
]


def _div(a, b):
    return a / b.where(b != 0) if isinstance(b, pd.Series) else (a / b if b else np.nan)


def derive(fin: pd.DataFrame) -> pd.DataFrame:
    """Add margins, returns, leverage and growth columns to an annual table."""
    df = fin.sort_index().copy()
    for col in ANNUAL_COLUMNS:
        if col not in df:
            df[col] = np.nan
    # capex is reported as a positive outflow; FCF = OCF − capex when not given.
    df["free_cash_flow"] = df["free_cash_flow"].fillna(
        df["operating_cash_flow"] - df["capex"].abs()
    )
    df["gross_margin"] = _div(df["gross_profit"], df["revenue"])
    df["operating_margin"] = _div(df["operating_income"], df["revenue"])
    df["net_margin"] = _div(df["net_income"], df["revenue"])
    df["fcf_margin"] = _div(df["free_cash_flow"], df["revenue"])
    # Returns on *average* equity/assets — standard, and less noisy than year-end.
    avg_eq = df["equity"].rolling(2, min_periods=1).mean()
    avg_assets = df["total_assets"].rolling(2, min_periods=1).mean()
    df["roe"] = _div(df["net_income"], avg_eq.where(avg_eq > 0))
    df["roa"] = _div(df["net_income"], avg_assets)
    df["debt_to_equity"] = _div(df["long_term_debt"], df["equity"].where(df["equity"] > 0))
    df["current_ratio"] = _div(df["current_assets"], df["current_liabilities"])
    for col, src in (("revenue_growth", "revenue"), ("eps_growth", "eps_diluted"),
                     ("fcf_growth", "free_cash_flow"), ("net_income_growth", "net_income")):
        prev = df[src].shift()
        df[col] = ((df[src] - prev) / prev.abs()).where(prev.abs() > 0)
    return df


def cagr(series: pd.Series, years: int) -> float | None:
    """Compound annual growth over the last ``years`` fiscal years (needs positive ends)."""
    s = series.dropna()
    if len(s) < years + 1:
        return None
    start, end = s.iloc[-1 - years], s.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return float((end / start) ** (1 / years) - 1)


def _band(value, bands: list[tuple[float, float]]) -> float:
    """First band whose threshold ``value`` exceeds → points."""
    for threshold, points in bands:
        if value > threshold:
            return points
    return 0.0


def business_summary(fin: pd.DataFrame) -> dict:
    """Growth, consistency and the 0–10 business trend score (with reasons).

    Score components (max points; missing data drops a component and the
    total is rescaled to 10):

    ============================  ====  ==========================================
    revenue 3-yr CAGR             2     > 15% → 2 · > 5% → 1.5 · > 0% → 1
    EPS 3-yr CAGR                 2     > 15% → 2 · > 5% → 1.5 · > 0% → 1
    operating-margin change, 3y   2     > +2 pp → 2 · within ±2 pp → 1
    positive-FCF years, last 5    2     share of years × 2
    ROE, latest year              2     > 20% → 2 · > 10% → 1.5 · > 0% → 0.5
    ============================  ====  ==========================================
    """
    df = derive(fin)
    last = df.iloc[-1]
    n = len(df)
    rev_cagr = {y: cagr(df["revenue"], y) for y in (3, 5, 10)}
    eps_cagr = {y: cagr(df["eps_diluted"], y) for y in (3, 5, 10)}
    fcf_cagr = {y: cagr(df["free_cash_flow"], y) for y in (3, 5, 10)}
    growth_years = int((df["revenue_growth"].dropna() > 0).sum())
    growth_obs = int(df["revenue_growth"].notna().sum())
    fcf_last5 = df["free_cash_flow"].dropna().iloc[-5:]
    margin_delta = None
    if df["operating_margin"].notna().sum() >= 4:
        om = df["operating_margin"].dropna()
        margin_delta = float(om.iloc[-1] - om.iloc[-4])

    points, possible, pros, cons = 0.0, 0.0, [], []

    def add(pts: float, max_pts: float, good: str, bad: str):
        nonlocal points, possible
        points += pts
        possible += max_pts
        (pros if pts >= max_pts * 0.6 else cons).append(good if pts >= max_pts * 0.6 else bad)

    if rev_cagr[3] is not None:
        add(_band(rev_cagr[3], [(0.15, 2), (0.05, 1.5), (0.0, 1)]), 2,
            f"Revenue grew {rev_cagr[3]:.1%}/yr over 3 years",
            f"Revenue growth is weak ({rev_cagr[3]:.1%}/yr over 3 years)")
    if eps_cagr[3] is not None:
        add(_band(eps_cagr[3], [(0.15, 2), (0.05, 1.5), (0.0, 1)]), 2,
            f"EPS grew {eps_cagr[3]:.1%}/yr over 3 years",
            f"EPS growth is weak ({eps_cagr[3]:.1%}/yr over 3 years)")
    elif df["eps_diluted"].notna().sum() >= 4:
        add(0, 2, "", "EPS was negative at the start or end of the last 3 years")
    if margin_delta is not None:
        pp = f"{margin_delta * 100:+.1f} pp over 3 years"
        if margin_delta > 0.02:
            add(2, 2, f"Operating margin expanded ({pp})", "")
        elif margin_delta > -0.02:
            points += 1
            possible += 2
            pros.append(f"Operating margin held steady ({pp})")
        else:
            add(0, 2, "", f"Operating margin contracted ({pp})")
    if len(fcf_last5):
        share = float((fcf_last5 > 0).mean())
        add(round(2 * share, 2), 2,
            f"Free cash flow positive in {int((fcf_last5 > 0).sum())} of the last {len(fcf_last5)} years",
            f"Free cash flow negative in {int((fcf_last5 <= 0).sum())} of the last {len(fcf_last5)} years")
    if pd.notna(last.get("roe")):
        add(_band(last["roe"], [(0.20, 2), (0.10, 1.5), (0.0, 0.5)]), 2,
            f"ROE {last['roe']:.1%} in the latest year",
            f"ROE is low ({last['roe']:.1%})")

    score = round(10 * points / possible, 1) if possible else None
    trend = (None if score is None else "IMPROVING" if score >= 7
             else "STABLE" if score >= 4 else "DETERIORATING")
    return {
        "years_of_data": n,
        "first_year": str(df.index[0].date()),
        "latest_year": str(df.index[-1].date()),
        "revenue_cagr": rev_cagr,
        "eps_cagr": eps_cagr,
        "fcf_cagr": fcf_cagr,
        "revenue_growth_years": f"{growth_years}/{growth_obs}",
        "operating_margin_change_3y": margin_delta,
        "latest": {k: (None if pd.isna(last.get(k)) else float(last[k])) for k in (
            "revenue", "net_income", "eps_diluted", "free_cash_flow", "gross_margin",
            "operating_margin", "net_margin", "roe", "debt_to_equity")},
        "score": score,
        "trend": trend,
        "pros": pros,
        "cons": cons,
    }


def price_performance(close: pd.Series, bench: pd.Series | None = None) -> dict:
    """Annualized and calendar-year stock returns, optionally vs. a benchmark."""
    close = close.dropna()
    out: dict = {"start": str(close.index[0].date()), "end": str(close.index[-1].date())}

    def ann(series: pd.Series, years: int) -> float | None:
        days = years * TRADING_DAYS
        if len(series) <= days - 5:
            return None
        start = series.iloc[max(0, len(series) - 1 - days)]
        return float((series.iloc[-1] / start) ** (1 / years) - 1)

    out["annualized"] = {f"{y}y": ann(close, y) for y in (1, 3, 5, 10)}
    yearly = close.groupby(close.index.year).last()
    first = close.groupby(close.index.year).first()
    cal = yearly.pct_change()
    cal.iloc[0] = yearly.iloc[0] / first.iloc[0] - 1  # partial first year, from first bar
    table = pd.DataFrame({"stock": cal})
    if bench is not None and len(bench.dropna()):
        b = bench.dropna()
        out["benchmark_annualized"] = {f"{y}y": ann(b, y) for y in (1, 3, 5, 10)}
        by = b.groupby(b.index.year).last()
        table["benchmark"] = by.pct_change().reindex(table.index)
        table["excess"] = table["stock"] - table["benchmark"]
    out["calendar_years"] = {int(k): {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
                             for k, row in table.iterrows()}
    full = cal.iloc[1:] if len(cal) > 1 else cal
    out["best_year"] = (int(full.idxmax()), float(full.max())) if len(full) else None
    out["worst_year"] = (int(full.idxmin()), float(full.min())) if len(full) else None
    out["positive_years"] = f"{int((full > 0).sum())}/{len(full)}"
    rets = close.pct_change().dropna()
    out["volatility"] = float(rets.std() * np.sqrt(TRADING_DAYS))
    out["max_drawdown"] = max_drawdown(close)
    out["from_all_time_high"] = float(close.iloc[-1] / close.max() - 1)
    return out
