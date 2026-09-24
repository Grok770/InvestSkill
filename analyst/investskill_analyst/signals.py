"""Turn a signal + technicals into a concrete, risk-first trade plan.

Mirrors the InvestSkill ``technical-analysis`` (entry / target / stop) and
``position-ladder`` (staged entry, size from risk) frameworks:

* Stop  = entry − ``atr_mult`` × ATR(14)  — volatility-scaled, not a fixed %.
* Size  = (account × risk-per-trade) ÷ (entry − stop), capped at
  ``max_position_pct`` of the account. You decide how much you are willing
  to *lose*, and the share count falls out of that.
* Targets at 2R and 3R (R = entry − stop), with the 52-week high noted as
  overhead resistance.

Long-only by design: a SELL signal produces an exit/avoid plan, never a short.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .factors import Signal


@dataclass
class RiskSettings:
    account_size: float = 100_000.0
    risk_per_trade: float = 0.01      # lose at most 1% of the account if stopped
    max_position_pct: float = 0.10    # never more than 10% of the account in one name
    atr_mult: float = 2.5


@dataclass
class TradePlan:
    ticker: str
    action: str
    setup: str
    entry: float | None = None
    entry_zone: tuple[float, float] | None = None
    stop: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    shares: int = 0
    position_value: float = 0.0
    capital_at_risk: float = 0.0
    tranches: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_trade_plan(
    ticker: str, signal: Signal, tech: dict, risk: RiskSettings | None = None
) -> TradePlan:
    risk = risk or RiskSettings()
    price, atr = tech["price"], tech.get("atr14")
    notes = list(signal.notes)

    if signal.action == "SELL":
        stop = round(price - (atr or 0) * 1.0, 2) if atr else None
        return TradePlan(
            ticker, "SELL", "Exit / avoid",
            stop=stop,
            notes=notes + [
                "Model ranks this name in the bottom of the universe. If held, "
                "reduce or exit; a close below the listed level confirms weakness.",
                "Check tax consequences (short- vs long-term, wash-sale) before selling.",
            ],
        )

    if not atr:
        return TradePlan(ticker, signal.action, "Insufficient data",
                         notes=notes + ["ATR unavailable — not enough history to size a trade."])

    sma50 = tech.get("sma50")
    trend_ok = tech.get("above_sma200") is not False
    if price > (sma50 or 0) and trend_ok:
        setup = "Pullback buy in an uptrend"
        zone_low = round(max(sma50 or price - atr, price - atr), 2)
        entry = round((zone_low + price) / 2, 2)
        zone = (zone_low, round(price, 2))
    elif sma50:
        setup = "Breakout trigger — buy only on a close back above the 50-day SMA"
        entry = round(max(sma50, price) + 0.1 * atr, 2)
        zone = (entry, round(entry + 0.5 * atr, 2))
    else:
        setup = "Market entry (short history)"
        entry, zone = round(price, 2), (round(price, 2), round(price, 2))

    stop = round(entry - risk.atr_mult * atr, 2)
    per_share_risk = entry - stop
    shares_by_risk = int(risk.account_size * risk.risk_per_trade / per_share_risk)
    shares_by_cap = int(risk.account_size * risk.max_position_pct / entry)
    shares = max(0, min(shares_by_risk, shares_by_cap))
    if shares_by_cap < shares_by_risk:
        notes.append(f"Size capped at {risk.max_position_pct:.0%} of the account.")

    t1 = round(entry + 2 * per_share_risk, 2)
    t2 = round(entry + 3 * per_share_risk, 2)
    if tech.get("high_52w") and tech["high_52w"] < t1:
        notes.append(f"52-week high ${tech['high_52w']:.2f} sits below target 1 — expect resistance.")

    if signal.action == "HOLD":
        notes.append("HOLD: no new money. Levels below are a watch-list plan if the score improves to ≥ 6.")
        shares = 0

    # Staged entry (position-ladder): 50% now, 30% on a 1×ATR dip, 20% on confirmation.
    tranches = []
    if shares:
        a, b = int(shares * 0.5), int(shares * 0.3)
        tranches = [
            {"pct": 50, "shares": a, "price": entry, "when": "initial entry"},
            {"pct": 30, "shares": b, "price": round(entry - atr, 2), "when": "on a 1×ATR pullback"},
            {"pct": 20, "shares": shares - a - b, "price": round(entry + atr, 2),
             "when": "add on strength once +1×ATR confirms"},
        ]

    return TradePlan(
        ticker=ticker,
        action=signal.action,
        setup=setup,
        entry=entry,
        entry_zone=zone,
        stop=stop,
        target_1=t1,
        target_2=t2,
        shares=shares,
        position_value=round(shares * entry, 2),
        capital_at_risk=round(shares * per_share_risk, 2),
        tranches=tranches,
        notes=notes,
    )
