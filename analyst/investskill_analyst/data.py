"""Market-data providers.

Every provider returns the same two shapes, so the model never cares where the
numbers came from:

* ``history(ticker, years)`` → DataFrame indexed by date with columns
  ``open, high, low, close, volume`` (close is split/dividend adjusted).
* ``fundamentals(ticker)``   → :class:`Fundamentals` (any field may be ``None``).

Providers:

* :class:`YFinanceProvider` — free Yahoo Finance data, no API key.
* :class:`CSVProvider`      — your own files (``<dir>/<TICKER>.csv`` +
  ``<dir>/fundamentals.json``), for vendor data or offline work.
* :class:`SyntheticProvider` — deterministic fake markets for demos and tests.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class Fundamentals:
    """Point-in-time snapshot of the metrics the factor model uses."""

    ticker: str
    name: str | None = None
    sector: str | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    ev_to_ebitda: float | None = None
    fcf_yield: float | None = None          # free cash flow / market cap
    dividend_yield: float | None = None
    roe: float | None = None                # return on equity (decimal)
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    debt_to_equity: float | None = None     # ratio, e.g. 0.8 (not 80)
    current_ratio: float | None = None
    revenue_growth: float | None = None     # YoY, decimal
    earnings_growth: float | None = None    # YoY, decimal
    beta: float | None = None
    # field name -> where the value came from (filled by providers that know)
    sources: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Fundamentals":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class DataProvider(Protocol):
    name: str

    def history(self, ticker: str, years: float = 3.0) -> pd.DataFrame: ...

    def fundamentals(self, ticker: str) -> Fundamentals: ...

    # Optional: providers that can supply multi-year financial statements also
    # implement ``annual_financials(ticker) -> DataFrame`` (see financials.py).


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price history is missing columns: {missing}")
    df = df[PRICE_COLUMNS].astype(float)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index().dropna(subset=["close"])


def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class YFinanceProvider:
    """Yahoo Finance via the ``yfinance`` package (free, no key, rate-limited)."""

    name = "yfinance"

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "yfinance is not installed: pip install 'investskill-analyst[yahoo]'"
            ) from exc
        self._cache: dict[tuple, object] = {}

    def history(self, ticker: str, years: float = 3.0) -> pd.DataFrame:
        import yfinance as yf

        key = ("h", ticker, years)
        if key not in self._cache:
            period_days = int(years * 365) + 10
            raw = yf.Ticker(ticker).history(
                period=f"{period_days}d", auto_adjust=True, actions=False
            )
            if raw.empty:
                raise LookupError(f"no price history for {ticker}")
            self._cache[key] = _normalize_history(raw)
        return self._cache[key]  # type: ignore[return-value]

    def fundamentals(self, ticker: str) -> Fundamentals:
        import yfinance as yf

        key = ("f", ticker)
        if key not in self._cache:
            info = yf.Ticker(ticker).info or {}
            mcap = _num(info.get("marketCap"))
            fcf = _num(info.get("freeCashflow"))
            de = _num(info.get("debtToEquity"))
            dy = _num(info.get("dividendYield"))
            self._cache[key] = Fundamentals(
                ticker=ticker,
                name=info.get("shortName") or info.get("longName"),
                sector=info.get("sector"),
                market_cap=mcap,
                trailing_pe=_num(info.get("trailingPE")),
                forward_pe=_num(info.get("forwardPE")),
                price_to_book=_num(info.get("priceToBook")),
                ev_to_ebitda=_num(info.get("enterpriseToEbitda")),
                fcf_yield=(fcf / mcap) if fcf is not None and mcap else None,
                # Yahoo reports debtToEquity in percent (80 = 0.8x).
                debt_to_equity=(de / 100.0) if de is not None else None,
                # Newer yfinance returns dividendYield in percent; older in decimal.
                dividend_yield=(dy / 100.0 if dy is not None and dy > 1 else dy),
                roe=_num(info.get("returnOnEquity")),
                roa=_num(info.get("returnOnAssets")),
                gross_margin=_num(info.get("grossMargins")),
                operating_margin=_num(info.get("operatingMargins")),
                current_ratio=_num(info.get("currentRatio")),
                revenue_growth=_num(info.get("revenueGrowth")),
                earnings_growth=_num(info.get("earningsGrowth")),
                beta=_num(info.get("beta")),
            )
            f = self._cache[key]
            f.sources = {k: "Yahoo Finance (unofficial)" for k, v in f.to_dict().items()
                         if v is not None and k not in ("ticker", "name", "sources")}
        return self._cache[key]  # type: ignore[return-value]

    # Yahoo statement rows -> financials.ANNUAL_COLUMNS (about 4 fiscal years).
    _ROWS = {
        "income_stmt": {"Total Revenue": "revenue", "Gross Profit": "gross_profit",
                        "Operating Income": "operating_income", "Net Income": "net_income",
                        "Diluted EPS": "eps_diluted", "Diluted Average Shares": "shares_diluted"},
        "cashflow": {"Operating Cash Flow": "operating_cash_flow",
                     "Capital Expenditure": "capex", "Free Cash Flow": "free_cash_flow",
                     "Cash Dividends Paid": "dividends_paid",
                     "Depreciation And Amortization": "dna"},
        "balance_sheet": {"Total Assets": "total_assets",
                          "Total Liabilities Net Minority Interest": "total_liabilities",
                          "Current Assets": "current_assets",
                          "Current Liabilities": "current_liabilities",
                          "Stockholders Equity": "equity",
                          "Cash And Cash Equivalents": "cash", "Long Term Debt": "long_term_debt"},
    }

    def annual_financials(self, ticker: str) -> pd.DataFrame:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        cols = {}
        for attr, mapping in self._ROWS.items():
            stmt = getattr(tk, attr)
            if stmt is None or stmt.empty:
                continue
            for row, col in mapping.items():
                if row in stmt.index:
                    cols[col] = stmt.loc[row]
        if not cols:
            raise LookupError(f"no annual financial statements for {ticker}")
        df = pd.DataFrame(cols)
        df.index = pd.to_datetime(df.index)
        if "capex" in df:
            df["capex"] = df["capex"].abs()
        if "dividends_paid" in df:
            df["dividends_paid"] = df["dividends_paid"].abs()
        return df.sort_index().astype(float)


class CSVProvider:
    """Reads ``<root>/<TICKER>.csv`` (date + OHLCV columns) and an optional
    ``<root>/fundamentals.json`` mapping ticker → Fundamentals fields."""

    name = "csv"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        fpath = self.root / "fundamentals.json"
        self._fund = json.loads(fpath.read_text()) if fpath.exists() else {}

    def history(self, ticker: str, years: float = 3.0) -> pd.DataFrame:
        path = self.root / f"{ticker}.csv"
        if not path.exists():
            raise LookupError(f"no CSV for {ticker} at {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _normalize_history(df)
        cutoff = df.index.max() - pd.Timedelta(days=int(years * 365))
        return df[df.index >= cutoff]

    def fundamentals(self, ticker: str) -> Fundamentals:
        return Fundamentals.from_dict({"ticker": ticker, **self._fund.get(ticker, {})})

    def annual_financials(self, ticker: str) -> pd.DataFrame:
        """``<root>/<TICKER>_financials.csv``: fiscal-year-end date + ANNUAL_COLUMNS."""
        path = self.root / f"{ticker}_financials.csv"
        if not path.exists():
            raise LookupError(f"no financials CSV for {ticker} at {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


class SyntheticProvider:
    """Deterministic geometric-Brownian-motion market with a common factor.

    Each ticker gets its own drift, volatility and fundamentals derived from a
    hash of its symbol, so results are reproducible across runs and machines.
    Use it to learn the tool, run the demo offline, and in tests. The numbers
    are fake — never trade on them.
    """

    name = "synthetic"

    def __init__(self, seed: int = 7, end: str = "2026-06-30") -> None:
        self.seed = seed
        self.end = pd.Timestamp(end)
        self._market: pd.Series | None = None

    def _rng(self, ticker: str) -> np.random.Generator:
        return np.random.default_rng(zlib.crc32(ticker.encode()) ^ self.seed)

    def _dates(self, years: float) -> pd.DatetimeIndex:
        start = self.end - pd.Timedelta(days=int(years * 365))
        return pd.bdate_range(start, self.end)

    def _market_returns(self, dates: pd.DatetimeIndex) -> pd.Series:
        if self._market is None or not dates.isin(self._market.index).all():
            rng = np.random.default_rng(self.seed)
            full = pd.bdate_range(self.end - pd.Timedelta(days=int(15 * 365)), self.end)
            self._market = pd.Series(rng.normal(0.0003, 0.010, len(full)), index=full)
        return self._market.reindex(dates).fillna(0.0)

    def history(self, ticker: str, years: float = 3.0) -> pd.DataFrame:
        rng = self._rng(ticker)
        dates = self._dates(years)
        beta = rng.uniform(0.6, 1.6)
        drift = rng.normal(0.0002, 0.0004)
        idio_vol = rng.uniform(0.008, 0.022)
        rets = beta * self._market_returns(dates).to_numpy() + rng.normal(
            drift, idio_vol, len(dates)
        )
        close = rng.uniform(20, 400) * np.exp(np.cumsum(rets))
        spread = np.abs(rng.normal(0, idio_vol, len(dates))) * close
        open_ = close * (1 + rng.normal(0, idio_vol / 3, len(dates)))
        high = np.maximum(open_, close) + spread / 2
        low = np.minimum(open_, close) - spread / 2
        volume = rng.lognormal(15, 0.4, len(dates))
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

    def fundamentals(self, ticker: str) -> Fundamentals:
        rng = self._rng(ticker + ":f")
        sectors = ["Technology", "Healthcare", "Financials", "Industrials",
                   "Consumer Discretionary", "Energy", "Communication Services"]
        return Fundamentals(
            ticker=ticker,
            name=f"{ticker} (synthetic)",
            sector=sectors[int(rng.integers(len(sectors)))],
            market_cap=float(rng.lognormal(25, 1.2)),
            trailing_pe=float(rng.uniform(8, 60)),
            forward_pe=float(rng.uniform(7, 45)),
            price_to_book=float(rng.uniform(1, 15)),
            ev_to_ebitda=float(rng.uniform(5, 40)),
            fcf_yield=float(rng.normal(0.04, 0.03)),
            dividend_yield=float(max(0.0, rng.normal(0.012, 0.012))),
            roe=float(rng.normal(0.16, 0.10)),
            roa=float(rng.normal(0.07, 0.05)),
            gross_margin=float(rng.uniform(0.2, 0.8)),
            operating_margin=float(rng.normal(0.18, 0.10)),
            debt_to_equity=float(rng.uniform(0.0, 2.5)),
            current_ratio=float(rng.uniform(0.7, 3.0)),
            revenue_growth=float(rng.normal(0.08, 0.12)),
            earnings_growth=float(rng.normal(0.10, 0.20)),
            beta=float(rng.uniform(0.6, 1.6)),
        )

    def annual_financials(self, ticker: str, years: int = 10) -> pd.DataFrame:
        rng = self._rng(ticker + ":fin")
        f = self.fundamentals(ticker)
        ends = pd.date_range(end=self.end - pd.offsets.YearEnd(1), periods=years, freq="YE")
        growth = rng.normal(f.revenue_growth or 0.06, 0.06, years)
        revenue = rng.uniform(5e9, 2e11) * np.cumprod(1 + growth)
        op_margin = np.clip((f.operating_margin or 0.15) + np.cumsum(rng.normal(0, 0.01, years)), -0.2, 0.6)
        net_income = revenue * op_margin * 0.8
        shares = rng.uniform(5e8, 5e9) * np.cumprod(1 - rng.uniform(0, 0.02, years))
        equity = revenue * rng.uniform(0.3, 1.0)
        ocf = net_income * rng.uniform(1.0, 1.4, years)
        capex = revenue * rng.uniform(0.02, 0.08, years)
        return pd.DataFrame({
            "revenue": revenue, "gross_profit": revenue * (f.gross_margin or 0.4),
            "operating_income": revenue * op_margin, "net_income": net_income,
            "eps_diluted": net_income / shares, "operating_cash_flow": ocf, "capex": capex,
            "total_assets": equity * 2.2, "total_liabilities": equity * 1.2, "equity": equity,
            "cash": revenue * 0.1, "long_term_debt": equity * (f.debt_to_equity or 0.5),
            "shares_diluted": shares,
        }, index=ends)


def get_provider(name: str = "yfinance", **kwargs) -> DataProvider:
    """Factory used by the CLI and the agent: ``edgar`` | ``yfinance`` | ``csv`` | ``synthetic``."""
    name = name.lower()
    if name in ("yfinance", "yahoo"):
        return YFinanceProvider()
    if name == "csv":
        if not kwargs.get("root"):
            raise ValueError("the csv provider needs a data directory (--data-dir)")
        return CSVProvider(kwargs["root"])
    if name in ("synthetic", "demo"):
        return SyntheticProvider()
    if name in ("edgar", "sec"):
        # Audited SEC fundamentals, Yahoo prices; Yahoo also fills forward P/E + beta
        # and acts as the second source for data cross-checks.
        from .edgar import EdgarClient, EdgarProvider

        yahoo = YFinanceProvider()
        return EdgarProvider(yahoo, EdgarClient(kwargs.get("user_agent")), fill=yahoo)
    raise ValueError(f"unknown data provider: {name!r}")
