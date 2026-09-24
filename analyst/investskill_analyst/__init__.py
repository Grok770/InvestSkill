"""InvestSkill Analyst — a quantitative + LLM research engine for US equities.

Layers (each usable on its own):

    data        → price history + fundamentals (yfinance, CSV, or synthetic)
    indicators  → SMA/EMA/RSI/MACD/ATR/volatility/drawdown/beta
    factors     → cross-sectional value · quality · growth · momentum · risk scoring
    signals     → per-ticker trade plan (entry / stop / target / size)
    backtest    → walk-forward monthly-rebalance test of the ranking rule
    agent       → Claude research agent that runs InvestSkill frameworks on top

Educational research tooling only. Not financial advice.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Quantitative scores are "
    "mechanical and backward-looking; verify every figure against primary "
    "sources and consult a licensed advisor before trading."
)
