import numpy as np
import pandas as pd
import pytest

from investskill_analyst.backtest import run_backtest
from investskill_analyst.data import CSVProvider, Fundamentals, SyntheticProvider
from investskill_analyst.engine import analyze, screen
from investskill_analyst.factors import score_table, to_signal
from investskill_analyst.indicators import atr, max_drawdown, rsi, technical_snapshot
from investskill_analyst.report import signal_block
from investskill_analyst.signals import RiskSettings, build_trade_plan
from investskill_analyst.universe import resolve_universe

P = SyntheticProvider()


# ------------------------------------------------------------ indicators ---

def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    noisy = P.history("AAPL", 2)["close"]
    r = rsi(noisy).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_max_drawdown_known_value():
    s = pd.Series([100, 120, 60, 90, 130], dtype=float)
    assert max_drawdown(s) == pytest.approx(-0.5)


def test_atr_positive_and_snapshot_fields():
    df = P.history("MSFT", 2)
    assert (atr(df).dropna() > 0).all()
    snap = technical_snapshot(df)
    for key in ("price", "sma50", "sma200", "rsi14", "atr14", "ret_12_1", "vol_1y"):
        assert snap[key] is not None


def test_synthetic_provider_is_deterministic():
    a = SyntheticProvider().history("NVDA", 1)
    b = SyntheticProvider().history("NVDA", 1)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------- factors ---

def _metrics(n=10):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["roe", "ret_12_1", "vol_1y", "fcf_yield"],
                      index=[f"T{i}" for i in range(n)])
    df.loc["BEST"] = [5.0, 5.0, -5.0, 5.0]    # best on every metric (low vol is good)
    df.loc["WORST"] = [-5.0, -5.0, 5.0, -5.0]
    return df


def test_score_table_orders_best_first_and_scores_in_range():
    out = score_table(_metrics())
    assert out.index[0] == "BEST" and out.index[-1] == "WORST"
    assert out["score"].between(0, 10).all()
    assert out.loc["BEST", "rank"] == 1


def test_score_table_renormalizes_missing_factors():
    m = _metrics()
    m.loc["BEST", ["roe", "fcf_yield"]] = np.nan  # quality/value missing
    out = score_table(m)
    assert out.loc["BEST", "coverage"] < out.loc["T0", "coverage"]
    assert out.loc["BEST", "rank"] <= 2  # still ranks on the factors it has


@pytest.mark.parametrize("score,action,conviction", [
    (9.0, "BUY", "STRONG"), (6.5, "BUY", "MODERATE"), (5.0, "HOLD", "MODERATE"),
    (3.0, "SELL", "MODERATE"), (1.0, "SELL", "STRONG"),
])
def test_signal_bands_follow_investskill_score_guide(score, action, conviction):
    sig = to_signal(pd.Series({"score": score, "coverage": 1.0}))
    assert (sig.action, sig.conviction) == (action, conviction)


def test_trend_gate_downgrades_buy_below_sma200():
    sig = to_signal(pd.Series({"score": 9.0, "coverage": 1.0}), {"above_sma200": False})
    assert sig.action == "BUY" and sig.conviction == "WEAK"
    assert any("200-day" in n for n in sig.notes)


# --------------------------------------------------------------- signals ---

def _tech(**kw):
    base = dict(price=100.0, atr14=2.0, sma50=97.0, sma200=90.0, above_sma200=True,
                high_52w=130.0, rsi14=55.0)
    return {**base, **kw}


def test_trade_plan_respects_risk_budget_and_cap():
    risk = RiskSettings(account_size=50_000, risk_per_trade=0.01, max_position_pct=0.2)
    plan = build_trade_plan("X", to_signal(pd.Series({"score": 9.0, "coverage": 1})), _tech(), risk)
    assert plan.stop < plan.entry < plan.target_1 < plan.target_2
    assert plan.capital_at_risk <= 50_000 * 0.01 + 1e-6
    assert plan.position_value <= 50_000 * 0.2 + 1e-6
    assert sum(t["shares"] for t in plan.tranches) == plan.shares > 0


def test_sell_and_hold_plans_never_buy():
    sell = build_trade_plan("X", to_signal(pd.Series({"score": 1.0, "coverage": 1})), _tech())
    hold = build_trade_plan("X", to_signal(pd.Series({"score": 5.0, "coverage": 1})), _tech())
    assert sell.shares == 0 and sell.action == "SELL"
    assert hold.shares == 0 and hold.entry is not None


def test_downtrend_uses_breakout_trigger():
    plan = build_trade_plan("X", to_signal(pd.Series({"score": 7.0, "coverage": 1})),
                            _tech(price=95.0, sma50=99.0, above_sma200=False))
    assert "Breakout" in plan.setup and plan.entry > 99.0


# -------------------------------------------------------------- backtest ---

def test_backtest_trades_next_session_no_lookahead():
    dates = pd.bdate_range("2020-01-01", "2022-12-30")
    rng = np.random.default_rng(1)
    n = len(dates)
    closes = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(np.full(n, 0.001) + rng.normal(0, 0.001, n))),  # steady winner
        "C": 100 * np.exp(np.cumsum(np.full(n, -0.0005) + rng.normal(0, 0.001, n))),
        "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n))),
    }, index=dates)
    month_ends = closes.groupby(closes.index.to_period("M")).tail(1).index
    d = month_ends[-6]
    closes.loc[d:, "B"] *= 1.5  # B jumps 50% ON the rebalance day

    result, daily = run_backtest(closes, top_n=1, cost_bps=0)
    rets = closes.pct_change()
    # On day d the portfolio still holds the name chosen a month earlier (A), not B.
    assert daily.loc[d, "strategy"] == pytest.approx(rets.loc[d, "A"])
    assert daily.loc[d, "strategy"] < 0.4
    assert result.months > 12 and -1 < result.max_drawdown <= 0


def test_backtest_requires_enough_history():
    closes = pd.DataFrame({"A": np.linspace(1, 2, 100), "B": np.linspace(2, 1, 100)},
                          index=pd.bdate_range("2024-01-01", periods=100))
    with pytest.raises(ValueError):
        run_backtest(closes, top_n=1)


# ---------------------------------------------------------- engine / data ---

def test_screen_and_analyze_end_to_end():
    res = screen(P, resolve_universe("mega-tech"), on_error=None)
    assert len(res.scored) == 10 and res.leaderboard(3)[0]["score"] >= res.leaderboard(3)[2]["score"]
    a = analyze(P, "aapl", peers=resolve_universe("mega-tech"), on_error=None)
    assert a.ticker == "AAPL" and a.universe_size == 10
    block = signal_block(a.signal)
    assert "INVESTMENT SIGNAL" in block and block.startswith("╔") and block.endswith("╝")
    assert len({len(line) for line in block.splitlines()}) == 1  # box edges line up


def test_screen_skips_bad_tickers(tmp_path):
    for t in ("AAA", "BBB", "CCC"):
        P.history(t, 2).to_csv(tmp_path / f"{t}.csv")
    (tmp_path / "fundamentals.json").write_text('{"AAA": {"roe": 0.3, "trailing_pe": 12}}')
    errors = []
    res = screen(CSVProvider(tmp_path), ["AAA", "BBB", "CCC", "MISSING"],
                 on_error=lambda t, e: errors.append(t))
    assert errors == ["MISSING"] and set(res.scored.index) == {"AAA", "BBB", "CCC"}
    assert CSVProvider(tmp_path).fundamentals("AAA").roe == 0.3


def test_fundamentals_from_dict_ignores_unknown_keys():
    f = Fundamentals.from_dict({"ticker": "X", "roe": 0.2, "bogus": 1})
    assert f.roe == 0.2
