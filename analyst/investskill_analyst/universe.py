"""Built-in stock universes. Pass your own with ``--tickers`` or ``--universe-file``."""

from __future__ import annotations

from pathlib import Path

# Liquid US large caps across all 11 GICS sectors — a sensible default peer set.
US_LARGE_CAP = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "QCOM",
    # Communication services
    "GOOGL", "META", "NFLX", "TMUS", "DIS",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "BKNG",
    # Consumer staples
    "WMT", "PG", "COST", "KO", "PEP",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ISRG",
    # Financials
    "JPM", "V", "MA", "BAC", "GS", "BRK-B", "SPGI",
    # Industrials
    "GE", "CAT", "RTX", "HON", "UNP", "DE",
    # Energy
    "XOM", "CVX", "COP",
    # Materials, utilities, real estate
    "LIN", "SHW", "NEE", "SO", "PLD", "AMT",
]

UNIVERSES = {
    "us-large-cap": US_LARGE_CAP,
    "mega-tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "ORCL", "AMD"],
    "dividend": ["JNJ", "PG", "KO", "PEP", "XOM", "CVX", "ABBV", "MRK", "HD", "MCD",
                 "SO", "NEE", "O", "VZ", "T", "MMM", "IBM", "TXN", "LMT", "UPS"],
}


def resolve_universe(
    name: str | None = None, tickers: str | list[str] | None = None, path: str | None = None
) -> list[str]:
    """Priority: explicit tickers > file (one ticker per line / comma list) > named set."""
    if tickers:
        items = tickers.split(",") if isinstance(tickers, str) else tickers
        return [t.strip().upper() for t in items if t.strip()]
    if path:
        text = Path(path).read_text()
        return [t.strip().upper() for t in text.replace(",", "\n").splitlines()
                if t.strip() and not t.strip().startswith("#")]
    key = name or "us-large-cap"
    if key not in UNIVERSES:
        raise ValueError(f"unknown universe {key!r}; choose from {sorted(UNIVERSES)}")
    return list(UNIVERSES[key])
