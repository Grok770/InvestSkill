"""EDGAR parsing, data-quality checks, company history, and the buy/sell verdict."""

import json

import numpy as np
import pandas as pd
import pytest

from investskill_analyst.cli import main
from investskill_analyst.data import Fundamentals, SyntheticProvider
from investskill_analyst.edgar import (EdgarClient, EdgarProvider, annual_financials,
                                       shares_outstanding, ttm_value)
from investskill_analyst.engine import analyze, company_history, watchlist
from investskill_analyst.financials import business_summary, cagr, derive, price_performance
from investskill_analyst.quality import check_fundamentals, check_prices, cross_check
from investskill_analyst.verdict import make_verdict, timing_score, track_record

P = SyntheticProvider()


# ------------------------------------------------- SEC company-facts fixture ---

def _fy(start, end, val, filed, form="10-K", fp="FY"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fp": fp,
            "fy": int(filed[:4]), "accn": "0000000000-00-000000"}


def _inst(end, val, filed, form="10-K"):
    return {"end": end, "val": val, "filed": filed, "form": form, "fp": "FY", "fy": int(filed[:4])}


FACTS = {
    "cik": 1234,
    "entityName": "Example Corp",
    "facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2024-10-20", "val": 1_000, "filed": "2024-11-01", "form": "10-K"},
            {"end": "2025-04-20", "val": 950, "filed": "2025-05-01", "form": "10-Q"},
        ]}}},
        "us-gaap": {
            # Older years under the pre-ASC 606 concept; newer under the new one.
            "SalesRevenueNet": {"units": {"USD": [
                _fy("2020-10-01", "2021-09-30", 800.0, "2021-11-01"),
            ]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                _fy("2021-10-01", "2022-09-30", 900.0, "2022-11-01"),
                _fy("2021-10-01", "2022-09-30", 900.0, "2023-11-01"),   # comparative, same value
                _fy("2022-10-01", "2023-09-30", 1000.0, "2023-11-01"),
                _fy("2022-10-01", "2023-09-30", 990.0, "2024-11-01"),   # restated later
                _fy("2023-10-01", "2024-09-30", 1200.0, "2024-11-01"),
                # quarterly noise that must not be treated as annual
                _fy("2024-07-01", "2024-09-30", 320.0, "2024-11-01"),
                # YTD 10-Qs for TTM: 6 months of FY25 and of FY24
                _fy("2024-10-01", "2025-03-31", 700.0, "2025-05-01", form="10-Q", fp="Q2"),
                _fy("2023-10-01", "2024-03-31", 560.0, "2024-05-01", form="10-Q", fp="Q2"),
            ]}},
            "NetIncomeLoss": {"units": {"USD": [
                _fy("2020-10-01", "2021-09-30", 80.0, "2021-11-01"),
                _fy("2021-10-01", "2022-09-30", 95.0, "2022-11-01"),
                _fy("2022-10-01", "2023-09-30", 110.0, "2023-11-01"),
                _fy("2023-10-01", "2024-09-30", 150.0, "2024-11-01"),
                _fy("2024-10-01", "2025-03-31", 90.0, "2025-05-01", form="10-Q", fp="Q2"),
                _fy("2023-10-01", "2024-03-31", 70.0, "2024-05-01", form="10-Q", fp="Q2"),
            ]}},
            "EarningsPerShareDiluted": {"units": {"USD/shares": [
                _fy("2022-10-01", "2023-09-30", 0.11, "2023-11-01"),
                _fy("2023-10-01", "2024-09-30", 0.15, "2024-11-01"),
            ]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                _fy("2023-10-01", "2024-09-30", 200.0, "2024-11-01"),
            ]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
                _fy("2023-10-01", "2024-09-30", 50.0, "2024-11-01"),
            ]}},
            "OperatingIncomeLoss": {"units": {"USD": [
                _fy("2023-10-01", "2024-09-30", 240.0, "2024-11-01"),
            ]}},
            "StockholdersEquity": {"units": {"USD": [
                _inst("2023-09-30", 500.0, "2023-11-01"),
                _inst("2024-09-28", 600.0, "2024-11-01"),        # 2 days off the FY end
                _inst("2025-03-31", 640.0, "2025-05-01", form="10-Q"),
            ]}},
            "Assets": {"units": {"USD": [_inst("2024-09-30", 1500.0, "2024-11-01")]}},
            "LongTermDebtNoncurrent": {"units": {"USD": [_inst("2024-09-30", 300.0, "2024-11-01")]}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
                _inst("2024-09-30", 100.0, "2024-11-01")]}},
        },
    },
}
TICKERS = {"0": {"cik_str": 1234, "ticker": "EXMP", "title": "Example Corp"},
           "1": {"cik_str": 1067983, "ticker": "BRK.B", "title": "Berkshire"}}


def _client(tmp_path):
    def fetch(url):
        return TICKERS if url.endswith("company_tickers.json") else FACTS
    return EdgarClient(cache_dir=tmp_path, fetch=fetch)


def test_edgar_annual_table_merges_concepts_and_takes_restatements():
    fin = annual_financials(FACTS)
    assert [d.year for d in fin.index] == [2021, 2022, 2023, 2024]
    assert fin["revenue"].tolist() == [800.0, 900.0, 990.0, 1200.0]  # 990 = restated FY23
    assert fin.loc["2023-09-30", "filed"] == pd.Timestamp("2023-11-01")  # first filed, for PIT use
    assert fin.loc["2024-09-30", "equity"] == 600.0  # balance sheet 2 days off still matched
    assert 320.0 not in fin["revenue"].tolist()     # quarter excluded


def test_edgar_ttm_uses_10q_ytd():
    val, desc = ttm_value(FACTS, ["RevenueFromContractWithCustomerExcludingAssessedTax"])
    assert val == pytest.approx(1200 + 700 - 560)
    assert "TTM to 2025-03-31" in desc
    ni, _ = ttm_value(FACTS, ["NetIncomeLoss"])
    assert ni == pytest.approx(150 + 90 - 70)
    ocf, desc = ttm_value(FACTS, ["NetCashProvidedByUsedInOperatingActivities"])
    assert ocf == 200 and "10-K" in desc  # no 10-Q → falls back to last FY


def test_edgar_shares_and_ticker_lookup(tmp_path):
    assert shares_outstanding(FACTS)[0] == 950
    c = _client(tmp_path)
    assert c.cik("exmp") == 1234 and c.cik("BRK-B") == 1067983
    with pytest.raises(LookupError):
        c.cik("NOPE")


def test_edgar_requires_contact_user_agent(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="User-Agent"):
        EdgarClient(cache_dir=tmp_path).cik("AAPL")


def test_edgar_provider_fundamentals_are_sourced(tmp_path):
    class Prices:
        name = "stub"

        def history(self, t, years=1):
            return pd.DataFrame({"open": [10.0], "high": [10.0], "low": [10.0],
                                 "close": [10.0], "volume": [1.0]},
                                index=pd.to_datetime(["2025-06-30"]))

        def fundamentals(self, t):
            return Fundamentals(ticker=t, forward_pe=40.0, beta=1.2, sector="Tech",
                                sources={"forward_pe": "stub", "beta": "stub"})

    p = EdgarProvider(Prices(), _client(tmp_path), fill=Prices())
    f = p.fundamentals("EXMP")
    assert f.market_cap == 9_500                       # 10 × 950 shares
    assert f.trailing_pe == pytest.approx(9_500 / 170)  # TTM net income
    assert f.roe == pytest.approx(170 / 640)            # latest equity (10-Q)
    assert f.forward_pe == 40.0 and "not in SEC filings" in f.sources["forward_pe"]
    assert f.sources["net_income"].startswith("SEC EDGAR, TTM")
    assert len(p.annual_financials("EXMP")) == 4


# -------------------------------------------------------------- financials ---

def test_derive_and_cagr():
    fin = derive(annual_financials(FACTS))
    assert fin.loc["2024-09-30", "free_cash_flow"] == 150.0  # 200 − 50
    assert fin.loc["2024-09-30", "revenue_growth"] == pytest.approx(1200 / 990 - 1)
    assert cagr(fin["revenue"], 3) == pytest.approx((1200 / 800) ** (1 / 3) - 1)
    assert cagr(pd.Series([-1.0, 2.0]), 1) is None  # undefined from a loss


def test_business_summary_scores_improving_vs_deteriorating():
    good = P.annual_financials("MSFT")
    bad = good.copy()
    bad["revenue"] = bad["revenue"].iloc[::-1].values          # shrinking
    bad["net_income"] = -bad["net_income"].abs()               # losses
    bad["eps_diluted"] = -bad["eps_diluted"].abs()
    bad["operating_cash_flow"] = -bad["operating_cash_flow"].abs()
    assert business_summary(good)["score"] > business_summary(bad)["score"]
    assert business_summary(bad)["trend"] == "DETERIORATING"


def test_price_performance_calendar_and_benchmark():
    idx = pd.bdate_range("2020-01-01", "2023-12-29")
    close = pd.Series(100 * 1.0004 ** np.arange(len(idx)), index=idx)
    perf = price_performance(close, close * 1.0)
    assert set(perf["calendar_years"]) == {2020, 2021, 2022, 2023}
    assert perf["calendar_years"][2022]["excess"] == pytest.approx(0.0)
    assert perf["max_drawdown"] == 0 and perf["annualized"]["3y"] > 0


# ----------------------------------------------------------------- quality ---

def test_check_prices_flags_split_gap_and_staleness():
    idx = pd.bdate_range("2024-01-01", periods=300)
    df = pd.DataFrame({"close": 100.0, "open": 100.0, "high": 101.0, "low": 99.0,
                       "volume": 1e6}, index=idx)
    df.iloc[200:, df.columns.get_loc("close")] = 25.0   # unadjusted 4:1 split
    df = df.drop(idx[100:110])                          # two-week hole
    w = " ".join(check_prices(df, today=idx[-1] + pd.Timedelta(days=30)))
    assert "35%" in w and "gap" in w and "stale" in w
    assert check_prices(P.history("AAPL", 2)) == []


def test_fundamental_sanity_and_cross_check():
    f = Fundamentals("X", trailing_pe=-5, roe=3.0, sources={"roe": "SEC"})
    w = " ".join(check_fundamentals(f))
    assert "negative" in w and "buybacks" in w
    a = Fundamentals("X", trailing_pe=20.0, roe=0.30, sources={"trailing_pe": "SEC"})
    b = Fundamentals("X", trailing_pe=30.0, roe=0.31, sources={"trailing_pe": "Yahoo"})
    diffs = cross_check(a, b)
    assert len(diffs) == 1 and "trailing_pe" in diffs[0] and "SEC" in diffs[0]


# ----------------------------------------------------------------- verdict ---

UP = dict(above_sma200=True, above_sma50=True, golden_cross=True, macd_hist=0.5, rsi14=55)
DOWN = dict(above_sma200=False, above_sma50=False, golden_cross=False, macd_hist=-0.5, rsi14=28)


def test_timing_score_extremes():
    assert timing_score(UP)[0] == 10.0
    assert timing_score(DOWN)[0] == 0.0
    assert timing_score({})[0] is None


def test_verdict_labels_and_rails():
    good_biz = {"score": 8.0, "pros": ["growing"], "cons": []}
    assert make_verdict("X", 9.0, UP, good_biz).label == "STRONG BUY"
    assert make_verdict("X", 1.0, DOWN, {"score": 1.0}).label == "STRONG SELL"
    # Great rank and business, but a downtrend → capped at HOLD with an explanation.
    v = make_verdict("X", 9.5, DOWN, {"score": 9.5})
    assert v.score >= 6 and v.label == "HOLD" and "Downtrend" in v.rails[0]
    # Missing business pillar → weights renormalize and the gap is disclosed.
    v = make_verdict("X", 8.0, UP, None)
    assert v.pillars["business"] is None and any("business-trend" in r for r in v.rails)
    assert "●" in v.gauge() and v.gauge().startswith("SELL") and v.gauge().endswith("BUY")


def test_track_record_is_point_in_time():
    tr = track_record(P.history("NVDA", 6))
    assert tr is not None and tr.months > 40
    assert tr.buy_signals + tr.sell_signals <= tr.months


def test_analyze_and_watchlist_carry_verdict():
    peers = ["AAPL", "MSFT", "AMD", "GOOGL", "META"]
    a = analyze(P, "NVDA", peers=peers, on_error=None)
    assert a.verdict is not None and a.business is not None
    assert a.signal.action == a.verdict.action          # signal block mirrors the verdict
    if a.verdict.action != "BUY":
        assert a.plan.shares == 0
    vs = watchlist(P, ["AAPL", "XOM"], peers=peers, on_error=None)
    assert [v.score for v in vs] == sorted([v.score for v in vs], reverse=True)


def test_company_history_and_cli(capsys):
    h = company_history(P, "AAPL", years=8)
    assert h.financials is not None and h.track is not None and h.benchmark == "SPY"
    assert main(["history", "AAPL", "--provider", "synthetic", "--years", "6"]) == 0
    out = capsys.readouterr().out
    assert "past performance" in out and "## The business" in out and "timing signal" in out
    assert main(["signal", "AAPL,MSFT", "--provider", "synthetic", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["ticker"] for r in rows} == {"AAPL", "MSFT"} and "gauge" in rows[0]
    assert main(["analyze", "AAPL", "--provider", "synthetic", "--universe", "mega-tech"]) == 0
    out = capsys.readouterr().out
    assert "## Bottom line" in out and "## Data quality & sources" in out
