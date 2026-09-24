"""The buy/sell indicator.

One number and one word per stock, built from three separate, explainable
parts:

    ┌──────────────┬────────┬──────────────────────────────────────────────┐
    │ pillar       │ weight │ question                                     │
    ├──────────────┼────────┼──────────────────────────────────────────────┤
    │ valuation &  │  40%   │ How does it rank against peers on value,     │
    │ quality rank │        │ quality, growth, momentum and risk right now?│
    │ business     │  35%   │ Is the company itself getting better or      │
    │ trend        │        │ worse? (revenue, EPS, margins, FCF, ROE)     │
    │ timing       │  25%   │ Is the price trend confirming it?            │
    │              │        │ (200/50-day averages, MACD, RSI)             │
    └──────────────┴────────┴──────────────────────────────────────────────┘

The weighted score (0–10) maps onto InvestSkill's Score Guide:
≥ 8 STRONG BUY · 6–7.9 BUY · 4–5.9 HOLD · 2–3.9 SELL · < 2 STRONG SELL.

Two safety rails can only *lower* a buy, never raise it:

* timing < 3 (a clear downtrend) → at most HOLD: "good company, wait".
* business trend < 3 (a deteriorating company) → at most HOLD: "cheap for a reason?".

``track_record`` answers "has this indicator worked on this stock?" for the
timing pillar, the only one that can be re-computed honestly from price
history alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .indicators import technical_snapshot

WEIGHTS = {"rank": 0.40, "business": 0.35, "timing": 0.25}
LABELS = [(8.0, "STRONG BUY"), (6.0, "BUY"), (4.0, "HOLD"), (2.0, "SELL"), (-1.0, "STRONG SELL")]


def timing_score(tech: dict) -> tuple[float | None, list[str], list[str]]:
    """0–10 trend/timing score from a technical snapshot, with pros and cons."""
    pts, possible, pros, cons = 0.0, 0.0, [], []

    def add(cond, max_pts, good, bad, partial: float | None = None):
        nonlocal pts, possible
        if cond is None:
            return
        possible += max_pts
        if cond is True:
            pts += max_pts
            pros.append(good)
        elif partial is not None:
            pts += partial
        else:
            cons.append(bad)

    add(tech.get("above_sma200"), 3, "Price above its 200-day average (long-term uptrend)",
        "Price below its 200-day average (long-term downtrend)")
    add(tech.get("above_sma50"), 2, "Price above its 50-day average",
        "Price below its 50-day average (short-term weakness)")
    add(tech.get("golden_cross"), 1.5, "50-day average above 200-day (golden cross)",
        "50-day average below 200-day (death cross)")
    if tech.get("macd_hist") is not None:
        add(tech["macd_hist"] > 0, 1.5, "MACD momentum positive", "MACD momentum negative")
    r = tech.get("rsi14")
    if r is not None:
        possible += 2
        if 40 <= r <= 70:
            pts += 2
            pros.append(f"RSI {r:.0f} — healthy, not stretched")
        elif 30 <= r < 40 or 70 < r <= 80:
            pts += 1
            cons.append(f"RSI {r:.0f} — {'overbought, wait for a pullback' if r > 70 else 'weak momentum'}")
        else:
            cons.append(f"RSI {r:.0f} — {'extremely overbought' if r > 80 else 'oversold, falling knife risk'}")
    if not possible:
        return None, pros, cons
    return round(10 * pts / possible, 1), pros, cons


@dataclass
class Verdict:
    ticker: str
    score: float
    label: str
    pillars: dict                 # pillar -> 0–10 score or None
    reasons_for: list[str] = field(default_factory=list)
    reasons_against: list[str] = field(default_factory=list)
    rails: list[str] = field(default_factory=list)

    @property
    def action(self) -> str:
        return "BUY" if "BUY" in self.label else "SELL" if "SELL" in self.label else "HOLD"

    def gauge(self, width: int = 21) -> str:
        """Text gauge, e.g.  SELL ━━━━━━━━━━━━━●━━━━━━━ BUY"""
        pos = int(round(self.score / 10 * (width - 1)))
        return "SELL " + "━" * pos + "●" + "━" * (width - 1 - pos) + " BUY"

    def to_dict(self) -> dict:
        return {**asdict(self), "action": self.action, "gauge": self.gauge()}


def _label(score: float) -> str:
    return next(lbl for threshold, lbl in LABELS if score >= threshold)


def make_verdict(ticker: str, rank_score: float | None, tech: dict,
                 business: dict | None = None) -> Verdict:
    """Combine the three pillars into the buy/sell indicator."""
    t_score, t_pros, t_cons = timing_score(tech)
    b_score = business.get("score") if business else None
    pillars = {"rank": rank_score, "business": b_score, "timing": t_score}
    avail = {k: v for k, v in pillars.items() if v is not None and not pd.isna(v)}
    if not avail:
        raise ValueError("no pillar has data — cannot form a verdict")
    wsum = sum(WEIGHTS[k] for k in avail)
    score = round(sum(WEIGHTS[k] * v for k, v in avail.items()) / wsum, 1)
    label = _label(score)

    pros, cons = list(t_pros), list(t_cons)
    if rank_score is not None:
        note = f"Ranks {rank_score:.1f}/10 vs. peers on value, quality, growth, momentum and risk"
        if rank_score >= 6:
            pros.append(note)
        elif rank_score < 4:
            cons.append(note)
    if business:
        pros += business.get("pros", [])
        cons += business.get("cons", [])

    rails = []
    if "BUY" in label and t_score is not None and t_score < 3:
        label = "HOLD"
        rails.append("Downtrend rail: the stock scores well but the price trend is down — "
                     "wait for it to reclaim the 50/200-day averages before buying.")
    if "BUY" in label and b_score is not None and b_score < 3:
        label = "HOLD"
        rails.append("Business rail: fundamentals are deteriorating — a low price may be "
                     "cheap for a reason.")
    if "business" not in avail:
        rails.append("No multi-year financials available — the business-trend pillar is missing, "
                     "so the verdict rests on rank and timing only.")
    return Verdict(ticker, score, label, pillars, pros, cons, rails)


@dataclass
class TrackRecord:
    months: int
    horizon_days: int
    buy_signals: int
    buy_avg_return: float | None
    buy_hit_rate: float | None
    sell_signals: int
    sell_avg_return: float | None
    sell_hit_rate: float | None      # share of SELL signals followed by a decline
    all_avg_return: float | None     # unconditional average — the bar to beat

    def to_dict(self) -> dict:
        return asdict(self)


def track_record(df: pd.DataFrame, horizon_days: int = 63, warmup: int = 260) -> TrackRecord | None:
    """How the *timing* pillar's BUY (≥ 6) and SELL (< 4) readings did on this stock.

    At each month-end, compute the timing score from data up to that day only,
    then measure the return over the next ``horizon_days`` trading days.
    """
    df = df.dropna(subset=["close"])
    close = df["close"]
    ends = close.groupby(close.index.to_period("M")).tail(1).index
    rows = []
    for d in ends:
        i = close.index.get_loc(d)
        if i < warmup or i + horizon_days >= len(close):
            continue
        score, _, _ = timing_score(technical_snapshot(df.iloc[: i + 1]))
        fwd = close.iloc[i + horizon_days] / close.iloc[i] - 1
        rows.append((score, fwd))
    if len(rows) < 6:
        return None
    r = pd.DataFrame(rows, columns=["score", "fwd"])
    buy, sell = r[r["score"] >= 6], r[r["score"] < 4]

    def avg(s):
        return None if s.empty else round(float(s.mean()), 4)

    return TrackRecord(
        months=len(r), horizon_days=horizon_days,
        buy_signals=len(buy), buy_avg_return=avg(buy["fwd"]),
        buy_hit_rate=None if buy.empty else round(float((buy["fwd"] > 0).mean()), 3),
        sell_signals=len(sell), sell_avg_return=avg(sell["fwd"]),
        sell_hit_rate=None if sell.empty else round(float((sell["fwd"] < 0).mean()), 3),
        all_avg_return=avg(r["fwd"]),
    )


__all__ = ["Verdict", "TrackRecord", "make_verdict", "timing_score", "track_record"]
