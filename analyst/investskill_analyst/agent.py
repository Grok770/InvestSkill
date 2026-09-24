"""Claude research agent: InvestSkill frameworks + the quant engine as tools.

The division of labour:

* The **quant engine** (screen / analyze / backtest tools) produces the numbers
  — deterministic, reproducible, never hallucinated.
* An **InvestSkill framework** (``prompts/<skill>.md``) is loaded into the
  system prompt and tells Claude *how* to structure the analysis.
* Claude **reasons** over both, searches the web for news, filings and
  catalysts the numbers can't see, red-teams its own thesis, and writes the
  report ending in the standard Investment Signal block.

Requires ``pip install 'investskill-analyst[llm]'`` and Anthropic credentials
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

from . import DISCLAIMER
from .data import DataProvider
from .engine import analyze, backtest, company_history, screen
from .factors import STYLE_WEIGHTS
from .signals import RiskSettings
from .universe import UNIVERSES, resolve_universe

DEFAULT_MODEL = "claude-opus-5"
MAX_TURNS = 30
# InvestSkill's universal prompts; override with INVESTSKILL_PROMPTS when the
# package is installed outside the repo checkout.
PROMPTS_DIR = Path(os.environ.get(
    "INVESTSKILL_PROMPTS", Path(__file__).resolve().parents[2] / "prompts"))
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

SYSTEM_PROMPT = """\
You are an equity research analyst for US-listed stocks, working for one
individual investor. You combine a quantitative engine (exposed as tools) with
the InvestSkill analysis framework included below.

How you work:
- Get numbers from the tools, never from memory. Price, factor scores, the
  buy/sell verdict, trade levels, past performance and backtest statistics come
  from `analyze_stock`, `company_history`, `screen_stocks` and
  `backtest_strategy`. If a tool fails, say what is missing rather than
  estimating it.
- Report any `data_quality_warnings` to the investor and lower your confidence
  accordingly. Say which figures are audited (SEC EDGAR) and which are not.
- Use web search for what the numbers can't see: recent earnings and
  guidance, filings, news, analyst revisions, upcoming catalysts. Cite the
  source and date of every fact you bring in from the web.
- The factor score ranks a stock *within its peer universe*; it is not a
  forecast. Treat it as one input, and say where your judgement departs from it
  and why.
- Before any BUY view, argue the bear case (you can load the `bear-case`
  framework) and state what would invalidate the thesis.
- Respect the trade plan's risk sizing. Never suggest leverage, options, or
  concentration beyond the plan's position cap unless the investor asks.
- You are not a licensed financial advisor. Be direct about the analysis, and
  end with the disclaimer line.

Output: follow the framework's structure. End with the InvestSkill Investment
Signal block (box-drawn), the Score Guide / Confidence / Horizon lines, and
`**Disclaimer:** {disclaimer}`.

Available InvestSkill frameworks (load more with `load_framework`):
{framework_list}

=== ACTIVE FRAMEWORK: {framework} ===
{framework_body}
"""


def list_frameworks(prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    return sorted(p.stem for p in prompts_dir.glob("*.md"))


def load_framework_text(name: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    if not re.fullmatch(r"[a-z0-9\-]+", name or ""):
        raise ValueError(f"invalid framework name {name!r}")
    path = prompts_dir / f"{name}.md"
    if not path.exists():
        raise LookupError(f"unknown framework {name!r}; available: {', '.join(list_frameworks(prompts_dir))}")
    return path.read_text()


def build_system_prompt(framework: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    return SYSTEM_PROMPT.format(
        disclaimer=DISCLAIMER,
        framework_list=", ".join(list_frameworks(prompts_dir)),
        framework=framework,
        framework_body=load_framework_text(framework, prompts_dir),
    )


# ---------------------------------------------------------------- tools ----

def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


_TICKERS = {"type": "array", "items": {"type": "string"},
            "description": "US ticker symbols, e.g. [\"AAPL\", \"MSFT\"]. Use BRK-B style for share classes."}
_UNIVERSE = {"type": "string", "enum": sorted(UNIVERSES),
             "description": "Named built-in universe, used when `tickers` is empty."}

TOOLS = [
    _tool(
        "screen_stocks",
        "Rank a universe of US stocks with the multi-factor model (value, quality, growth, "
        "momentum, low risk). Returns the leaderboard with 0-10 scores, actions and factor "
        "z-scores. Use it to find candidates.",
        {"tickers": _TICKERS, "universe": _UNIVERSE,
         "style": {"type": "string", "enum": sorted(STYLE_WEIGHTS)},
         "top": {"type": "integer", "description": "How many rows to return (default 15)."}},
        [],
    ),
    _tool(
        "analyze_stock",
        "Full quantitative workup of one stock against a peer universe: the buy/sell verdict "
        "(rank + business trend + timing, with reasons), factor z-scores, fundamentals with "
        "their sources, data-quality warnings, technicals (SMA/RSI/MACD/ATR, returns, "
        "volatility, drawdown), and a risk-sized trade plan (entry, stop, targets, shares).",
        {"ticker": {"type": "string"}, "peers": _TICKERS,
         "style": {"type": "string", "enum": sorted(STYLE_WEIGHTS)}},
        ["ticker"],
    ),
    _tool(
        "company_history",
        "A company's past performance: annualized and calendar-year stock returns vs. SPY, "
        "drawdowns, up to 15 years of annual financials (revenue, EPS, FCF, margins, ROE), "
        "growth rates, a 0-10 business trend score, and how the timing signal has done on "
        "this stock historically.",
        {"ticker": {"type": "string"},
         "years": {"type": "integer", "description": "Years of history, 1-20 (default 10)."}},
        ["ticker"],
    ),
    _tool(
        "backtest_strategy",
        "Walk-forward monthly backtest of the price-factor ranking rule on a universe (top-N "
        "equal weight vs. equal-weight universe). Use it to judge how much to trust the "
        "model's momentum/risk signals. Fundamentals are not backtested.",
        {"tickers": _TICKERS, "universe": _UNIVERSE,
         "years": {"type": "number", "description": "History length, 2-15 (default 5)."},
         "top_n": {"type": "integer", "description": "Holdings per month (default 10)."}},
        [],
    ),
    _tool(
        "load_framework",
        "Load another InvestSkill framework's full instructions (e.g. bear-case, "
        "stock-valuation, earnings-preview, risk-stress-test) to apply in this analysis.",
        {"name": {"type": "string"}},
        ["name"],
    ),
]

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 8}


def _clean_tickers(raw) -> list[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("tickers must be a list of strings")
    out = [str(t).strip().upper() for t in raw if str(t).strip()]
    bad = [t for t in out if not TICKER_RE.match(t)]
    if bad:
        raise ValueError(f"invalid ticker symbols: {bad}")
    return out[:120] or None


def _style(inp: dict) -> str:
    style = inp.get("style") or "balanced"
    if style not in STYLE_WEIGHTS:
        raise ValueError(f"style must be one of {sorted(STYLE_WEIGHTS)}")
    return style


class ToolExecutor:
    """Validates model-supplied tool input, then calls the quant engine."""

    def __init__(self, provider: DataProvider, risk: RiskSettings | None = None,
                 prompts_dir: Path = PROMPTS_DIR) -> None:
        self.provider = provider
        self.risk = risk or RiskSettings()
        self.prompts_dir = prompts_dir

    def __call__(self, name: str, inp: dict) -> str:
        if not isinstance(inp, dict):
            raise ValueError("tool input must be an object")
        handler: Callable[[dict], object] | None = getattr(self, f"_t_{name}", None)
        if handler is None:
            raise ValueError(f"unknown tool {name!r}")
        result = handler(inp)
        return result if isinstance(result, str) else json.dumps(result, default=str)

    def _t_screen_stocks(self, inp: dict):
        tickers = resolve_universe(inp.get("universe"), _clean_tickers(inp.get("tickers")))
        top = int(inp.get("top") or 15)
        res = screen(self.provider, tickers, style=_style(inp), on_error=None)
        return {"universe_size": len(res.scored), "style": res.style,
                "leaderboard": res.leaderboard(max(1, min(top, 60)))}

    def _t_analyze_stock(self, inp: dict):
        tickers = _clean_tickers([inp.get("ticker", "")])
        if not tickers:
            raise ValueError("ticker is required")
        ticker = tickers[0]
        a = analyze(self.provider, ticker, peers=_clean_tickers(inp.get("peers")),
                    style=_style(inp), risk=self.risk, on_error=None)
        row = {k: (None if v != v else v) for k, v in a.row.to_dict().items()}  # NaN → None
        return {
            "ticker": a.ticker,
            "peer_universe_size": a.universe_size,
            "model_row": row,
            "verdict": a.verdict.to_dict() if a.verdict else None,
            "signal": a.signal.__dict__,
            "business_trend": a.business,
            "technicals": a.tech,
            "fundamentals": a.fund.to_dict(),
            "data_quality_warnings": a.data_quality,
            "trade_plan": a.plan.to_dict(),
            "risk_settings": self.risk.__dict__,
        }

    def _t_company_history(self, inp: dict):
        tickers = _clean_tickers([inp.get("ticker", "")])
        if not tickers:
            raise ValueError("ticker is required")
        years = int(inp.get("years") or 10)
        if not 1 <= years <= 20:
            raise ValueError("years must be 1-20")
        h = company_history(self.provider, tickers[0], years=years)
        fin = None
        if h.financials is not None:
            cols = ["revenue", "net_income", "eps_diluted", "free_cash_flow",
                    "gross_margin", "operating_margin", "roe", "debt_to_equity", "revenue_growth"]
            fin = json.loads(h.financials[[c for c in cols if c in h.financials]]
                             .to_json(orient="index", date_format="iso"))
        return {"ticker": h.ticker, "stock_performance": h.stock, "benchmark": h.benchmark,
                "business_trend": h.business, "annual_financials": fin,
                "timing_signal_track_record": h.track.to_dict() if h.track else None}

    def _t_backtest_strategy(self, inp: dict):
        tickers = resolve_universe(inp.get("universe"), _clean_tickers(inp.get("tickers")))
        years = float(inp.get("years") or 5)
        top_n = int(inp.get("top_n") or 10)
        if not 2 <= years <= 15 or not 1 <= top_n <= 50:
            raise ValueError("years must be 2-15 and top_n 1-50")
        result, _ = backtest(self.provider, tickers, years=years, top_n=top_n)
        return result.to_dict()

    def _t_load_framework(self, inp: dict):
        return load_framework_text(str(inp.get("name", "")), self.prompts_dir)


# ---------------------------------------------------------------- loop -----

def run_research(
    request: str,
    provider: DataProvider,
    framework: str = "stock-eval",
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    web_search: bool = True,
    risk: RiskSettings | None = None,
    out=sys.stdout,
    client=None,
) -> str:
    """Run the agent loop for one research request and return the final report."""
    import anthropic

    client = client or anthropic.Anthropic()
    execute = ToolExecutor(provider, risk)
    tools = TOOLS + ([WEB_SEARCH_TOOL] if web_search else [])
    system = [{"type": "text", "text": build_system_prompt(framework),
               "cache_control": {"type": "ephemeral"}}]
    messages: list[dict] = [{"role": "user", "content": request}]
    report = ""
    json_retries = 0

    for _turn in range(MAX_TURNS):
        try:
            with client.beta.messages.stream(
                model=model,
                max_tokens=64000,
                system=system,
                tools=tools,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                # Server-side refusal fallback: a declined request is re-run on
                # Anthropic's recommended fallback model inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                for text in stream.text_stream:
                    out.write(text)
                    out.flush()
                response = stream.get_final_message()
            json_retries = 0
        except ValueError:
            # Tool-input JSON the SDK couldn't parse (eager input streaming).
            json_retries += 1
            if json_retries > 2:
                raise
            continue

        messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text")
        if text.strip():
            report = text  # the last turn with prose is the report

        if response.stop_reason == "refusal":
            out.write("\n[the model declined this request]\n")
            break
        if response.stop_reason == "pause_turn":
            continue  # a long server-side web search paused; resend to resume
        if response.stop_reason == "max_tokens":
            out.write("\n[output hit max_tokens — report truncated]\n")
            break
        if response.stop_reason != "tool_use":
            break

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            out.write(f"\n  ↳ {block.name}({json.dumps(block.input)[:120]})\n")
            try:
                content, is_error = execute(block.name, block.input), False
            except Exception as exc:  # noqa: BLE001 - report every failure back to the model
                content, is_error = f"Error: {type(exc).__name__}: {exc}", True
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        out.write(f"\n[stopped after {MAX_TURNS} turns]\n")

    out.write("\n")
    return report
