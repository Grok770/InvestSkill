"""SEC EDGAR fundamentals — the primary source, free, no API key.

Every US-listed company files its financial statements with the SEC in
machine-readable XBRL. The SEC publishes all of a company's reported values in
one JSON document ("company facts"):

    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

This module turns that into:

* an **annual financials table** (up to ~15 fiscal years, 10-K values, latest
  restatement shown, with the date each year was *first filed* so the data can
  be used point-in-time);
* **trailing-twelve-month (TTM)** flow values — last 10-K + current YTD −
  prior-year YTD from the 10-Qs — and the latest balance sheet;
* a :class:`EdgarProvider` that pairs these audited fundamentals with prices
  from any other provider (Yahoo by default) and records the source of every
  field.

SEC fair-access rules: identify yourself with a User-Agent that contains a
contact email (set ``SEC_USER_AGENT="Your Name you@example.com"``), and stay
under 10 requests/second. Responses are cached on disk for a day.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from .data import DataProvider, Fundamentals
from .financials import derive

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
CACHE_DIR = Path(os.environ.get("INVESTSKILL_CACHE", Path.home() / ".cache" / "investskill-analyst"))
CACHE_TTL = 24 * 3600
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "20-F", "20-F/A", "40-F")

# Our column → XBRL concepts in priority order. Companies switch concepts over
# time (e.g. SalesRevenueNet → RevenueFromContractWithCustomer… after ASC 606),
# so for each fiscal year the first concept with a value wins.
FLOW_CONCEPTS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "SalesRevenueGoodsNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "dna": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}
INSTANT_CONCEPTS: dict[str, list[str]] = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt",
                       "LongTermDebtAndCapitalLeaseObligations"],
}
UNITS = {"eps_diluted": "USD/shares", "shares_diluted": "shares"}


class EdgarClient:
    """Minimal, cached, rate-limited client for the two EDGAR endpoints we need."""

    def __init__(self, user_agent: str | None = None, cache_dir: Path = CACHE_DIR,
                 fetch=None) -> None:
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT")
        self.cache_dir = Path(cache_dir)
        self._fetch = fetch  # injectable for tests: fetch(url) -> dict
        self._last = 0.0
        self._tickers: dict[str, int] | None = None
        self._facts: dict[int, dict] = {}

    def _get(self, url: str, cache_name: str) -> dict:
        path = self.cache_dir / cache_name
        if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
            return json.loads(path.read_text())
        if self._fetch is not None:
            data = self._fetch(url)
        else:
            if not self.user_agent or "@" not in self.user_agent:
                raise RuntimeError(
                    "SEC EDGAR requires a User-Agent with a contact email: "
                    "export SEC_USER_AGENT='Your Name you@example.com'"
                )
            wait = 0.12 - (time.time() - self._last)  # ≤ ~8 requests/second
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(url, headers={
                "User-Agent": self.user_agent, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            self._last = time.time()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data

    def cik(self, ticker: str) -> int:
        if self._tickers is None:
            raw = self._get(TICKERS_URL, "company_tickers.json")
            self._tickers = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
        key = ticker.upper()
        for candidate in (key, key.replace("-", "."), key.replace(".", "-")):
            if candidate in self._tickers:
                return self._tickers[candidate]
        raise LookupError(f"{ticker} not found in the SEC ticker list (ETFs and funds aren't covered)")

    def company_facts(self, ticker: str) -> dict:
        cik = self.cik(ticker)
        if cik not in self._facts:
            self._facts[cik] = self._get(FACTS_URL.format(cik=cik), f"facts_{cik}.json")
        return self._facts[cik]


# ------------------------------------------------------------- parsing ----

def _entries(facts: dict, concept: str, unit: str = "USD") -> pd.DataFrame:
    node = facts.get("facts", {}).get("us-gaap", {}).get(concept) \
        or facts.get("facts", {}).get("ifrs-full", {}).get(concept)
    rows = (node or {}).get("units", {}).get(unit, [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if "start" in df:
        df["start"] = pd.to_datetime(df["start"])
        df["days"] = (df["end"] - df["start"]).dt.days
    return df


def _annual(facts: dict, concepts: list[str], unit: str, flow: bool) -> pd.DataFrame:
    """One value per fiscal-year end: latest filed (restated) value + first-filed date."""
    parts = []
    for rank, concept in enumerate(concepts):
        df = _entries(facts, concept, unit)
        if df.empty:
            continue
        df = df[df["form"].isin(ANNUAL_FORMS)]
        if flow:
            df = df[df["days"].between(340, 390)]
        if df.empty:
            continue
        df = df.assign(rank=rank)
        parts.append(df)
    if not parts:  # concept never reported: empty, but date-indexed so joins still work
        return pd.DataFrame({"val": pd.Series(dtype=float), "filed": pd.Series(dtype="datetime64[ns]")},
                            index=pd.DatetimeIndex([], name="end"))
    df = pd.concat(parts)
    # Best-ranked concept per year, then the most recent filing (restatements win).
    df = df.sort_values(["end", "rank", "filed"], ascending=[True, True, False])
    first_filed = df.groupby("end")["filed"].min()
    best = df.groupby("end").first()
    return pd.DataFrame({"val": best["val"].astype(float), "filed": first_filed})


def annual_financials(facts: dict, years: int = 15) -> pd.DataFrame:
    """Annual financials table (see ``financials`` for the column contract)."""
    anchor = _annual(facts, FLOW_CONCEPTS["net_income"], "USD", flow=True)
    rev = _annual(facts, FLOW_CONCEPTS["revenue"], "USD", flow=True)
    fy_ends = anchor.index.union(rev.index)
    if not len(fy_ends):
        raise LookupError("no annual (10-K) income data in EDGAR for this company")
    out = pd.DataFrame(index=fy_ends)
    for col, concepts in FLOW_CONCEPTS.items():
        out[col] = _annual(facts, concepts, UNITS.get(col, "USD"), flow=True)["val"]
    for col, concepts in INSTANT_CONCEPTS.items():
        vals = _annual(facts, concepts, "USD", flow=False)["val"]
        # Balance-sheet dates can differ from the fiscal-year end by a few days.
        out[col] = vals.reindex(fy_ends, method="nearest", tolerance=pd.Timedelta(days=7))
    out["filed"] = anchor["filed"].combine_first(rev["filed"]).reindex(fy_ends)
    out.index.name = "fiscal_year_end"
    # Drop stub years with neither revenue nor net income (e.g. transition periods).
    out = out[out[["revenue", "net_income"]].notna().any(axis=1)]
    return out.iloc[-years:]


def ttm_value(facts: dict, concepts: list[str], unit: str = "USD") -> tuple[float, str] | None:
    """Trailing twelve months = last FY + current YTD − prior-year YTD.

    Returns (value, period description) or None. Falls back to the last FY
    when no later 10-Q exists.
    """
    for concept in concepts:
        df = _entries(facts, concept, unit)
        if df.empty or "days" not in df:
            continue
        fy = df[df["form"].isin(ANNUAL_FORMS) & df["days"].between(340, 390)]
        if fy.empty:
            continue
        fy = fy.sort_values(["end", "filed"]).iloc[-1]
        q = df[df["form"].str.startswith("10-Q")]
        ytd = q[(q["start"] - fy["end"]).dt.days.between(0, 10) & q["days"].between(80, 290)]
        if ytd.empty:
            return float(fy["val"]), f"FY ended {fy['end'].date()} (10-K)"
        cur = ytd.sort_values(["end", "filed"]).iloc[-1]
        prior = q[((q["end"] - (cur["end"] - pd.DateOffset(years=1))).dt.days.abs() <= 10)
                  & ((q["days"] - cur["days"]).abs() <= 15)]
        if prior.empty:
            return float(fy["val"]), f"FY ended {fy['end'].date()} (10-K)"
        prior = prior.sort_values("filed").iloc[-1]
        val = float(fy["val"] + cur["val"] - prior["val"])
        return val, f"TTM to {cur['end'].date()} (10-K + 10-Q)"
    return None


def latest_instant(facts: dict, concepts: list[str], unit: str = "USD") -> tuple[float, str] | None:
    for concept in concepts:
        df = _entries(facts, concept, unit)
        df = df[df["form"].isin(ANNUAL_FORMS + ("10-Q", "10-Q/A"))] if not df.empty else df
        if not df.empty:
            row = df.sort_values(["end", "filed"]).iloc[-1]
            return float(row["val"]), f"as of {row['end'].date()} ({row['form']})"
    return None


def shares_outstanding(facts: dict) -> tuple[float, str] | None:
    node = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding")
    rows = (node or {}).get("units", {}).get("shares", [])
    if rows:
        # Multi-class companies report one row per class for the same date; sum them.
        df = pd.DataFrame(rows)
        df["end"] = pd.to_datetime(df["end"])
        latest = df[df["end"] == df["end"].max()]
        latest = latest[latest["filed"] == latest["filed"].max()]  # one filing's rows
        return float(latest["val"].sum()), f"cover page as of {df['end'].max().date()}"
    return None


# ------------------------------------------------------------ provider ----

class EdgarProvider:
    """Audited SEC fundamentals + prices from another provider.

    ``fill`` (optional, e.g. a YFinanceProvider) supplies fields the SEC can't
    — forward P/E (an analyst estimate) and beta — and serves as the second
    source for ``quality.cross_check``.
    """

    name = "edgar"

    def __init__(self, prices: DataProvider, client: EdgarClient | None = None,
                 fill: DataProvider | None = None) -> None:
        self.prices = prices
        self.client = client or EdgarClient()
        self.secondary = fill

    def history(self, ticker: str, years: float = 3.0) -> pd.DataFrame:
        return self.prices.history(ticker, years)

    def annual_financials(self, ticker: str, years: int = 15) -> pd.DataFrame:
        return annual_financials(self.client.company_facts(ticker), years)

    def fundamentals(self, ticker: str) -> Fundamentals:
        facts = self.client.company_facts(ticker)
        src: dict[str, str] = {}

        def ttm(col):
            got = ttm_value(facts, FLOW_CONCEPTS[col], UNITS.get(col, "USD"))
            if got:
                src[col] = f"SEC EDGAR, {got[1]}"
                return got[0]
            return None

        def inst(col):
            got = latest_instant(facts, INSTANT_CONCEPTS[col])
            if got:
                src[col] = f"SEC EDGAR, {got[1]}"
                return got[0]
            return None

        revenue, ni, ocf = ttm("revenue"), ttm("net_income"), ttm("operating_cash_flow")
        capex, op_inc, gp, dna = ttm("capex"), ttm("operating_income"), ttm("gross_profit"), ttm("dna")
        divs = ttm("dividends_paid")
        equity, assets, cash, ltd = inst("equity"), inst("total_assets"), inst("cash"), inst("long_term_debt")
        ca, cl = inst("current_assets"), inst("current_liabilities")

        price = float(self.prices.history(ticker, 1)["close"].iloc[-1])
        shares = shares_outstanding(facts)
        mcap = price * shares[0] if shares else None
        if shares:
            src["market_cap"] = f"price × shares ({shares[1]})"

        fin = derive(annual_financials(facts, years=3))
        last = fin.iloc[-1]

        def ratio(a, b):
            return a / b if a is not None and b not in (None, 0) else None

        fcf = ocf - abs(capex) if ocf is not None and capex is not None else ocf
        ev = mcap + (ltd or 0) - (cash or 0) if mcap else None
        ebitda = op_inc + (dna or 0) if op_inc is not None else None
        f = Fundamentals(
            ticker=ticker.upper(),
            name=facts.get("entityName"),
            market_cap=mcap,
            trailing_pe=ratio(mcap, ni),
            price_to_book=ratio(mcap, equity) if equity and equity > 0 else None,
            ev_to_ebitda=ratio(ev, ebitda) if ebitda and ebitda > 0 else None,
            fcf_yield=ratio(fcf, mcap),
            dividend_yield=ratio(abs(divs), mcap) if divs is not None else None,
            roe=ratio(ni, equity) if equity and equity > 0 else None,
            roa=ratio(ni, assets),
            gross_margin=ratio(gp, revenue),
            operating_margin=ratio(op_inc, revenue),
            debt_to_equity=ratio(ltd or 0.0, equity) if equity and equity > 0 else None,
            current_ratio=ratio(ca, cl),
            revenue_growth=None if pd.isna(last.get("revenue_growth")) else float(last["revenue_growth"]),
            earnings_growth=None if pd.isna(last.get("net_income_growth")) else float(last["net_income_growth"]),
            sources=src,
        )
        for k in ("revenue_growth", "earnings_growth"):
            if getattr(f, k) is not None:
                src[k] = f"SEC EDGAR, 10-K FY {fin.index[-1].date()} vs prior year"
        if self.secondary is not None:
            try:
                other = self.secondary.fundamentals(ticker)
            except Exception:  # noqa: BLE001 - the fill source is optional
                other = None
            if other is not None:
                for k in ("forward_pe", "beta", "sector"):
                    if getattr(f, k) is None and getattr(other, k) is not None:
                        setattr(f, k, getattr(other, k))
                        f.sources[k] = f"{self.secondary.name} (not in SEC filings)"
                f.name = f.name or other.name
        return f


def fiscal_age_days(fin: pd.DataFrame, today: date | None = None) -> int:
    """Days since the latest fiscal year in ``fin`` ended — staleness check."""
    today = today or date.today()
    return int((pd.Timestamp(today) - fin.index[-1]).days)


__all__ = ["EdgarClient", "EdgarProvider", "annual_financials", "ttm_value",
           "latest_instant", "shares_outstanding", "fiscal_age_days"]
