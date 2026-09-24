# InvestSkill Analyst

**A quantitative stock model and a Claude research agent, built on the InvestSkill frameworks.**

InvestSkill's 30 frameworks tell an AI *how* to analyze a stock. This package adds
the parts a prompt can't provide: real numbers computed by code, a ranking model
you can backtest, and an agent that combines the two.

```
                 ┌──────────────────────────────────────────────┐
  you ──────────▶│ research agent (Claude)                      │
 "find me 3      │  system prompt = InvestSkill framework       │
  quality names" │  (stock-eval, bear-case, stock-screener, …)  │
                 └───────┬───────────────┬──────────────┬───────┘
                         │ tools         │ web search   │ load_framework
             ┌───────────▼─────────┐     ▼              ▼
             │ quant engine        │   news, filings,   prompts/*.md
             │  screen_stocks      │   catalysts
             │  analyze_stock      │
             │  backtest_strategy  │
             └───────────┬─────────┘
                         │
   data: yfinance (free) · your CSVs · synthetic (offline demo)
```

| Layer | Module | What it does |
|-------|--------|--------------|
| Data | `data.py` | Price history + fundamentals from Yahoo Finance (no key), your own CSVs, or a deterministic synthetic market |
| Indicators | `indicators.py` | SMA/EMA, RSI(14), MACD(12,26,9), ATR(14), returns, volatility, drawdown, beta |
| Factor model | `factors.py` | Ranks a universe on **value · quality · growth · momentum · low risk** (winsorized z-scores), maps each stock onto InvestSkill's 0–10 Score Guide, applies a 200-day trend gate |
| Trade plan | `signals.py` | Entry zone, ATR stop, 2R/3R targets, **position size set by how much you're willing to lose**, staged entry (the `position-ladder` method) |
| Backtest | `backtest.py` | Monthly walk-forward test of the ranking rule; trades the next session, charges costs, compares against the equal-weight universe |
| Agent | `agent.py` | Claude loop: InvestSkill framework as the system prompt, quant engine + web search as tools, output ends in the standard Investment Signal block |

## Install

```bash
cd analyst
pip install -e '.[all]'          # numpy + pandas + yfinance + anthropic
# or: pip install -e .           # quant model only (bring your own data)
```

Python ≥ 3.10. The `research` command needs Anthropic credentials, either
`ANTHROPIC_API_KEY` or `ant auth login`. The other commands need no keys.

## Use it

```bash
# 1. Find candidates: rank 56 US large caps (or your own list)
investskill-analyst screen
investskill-analyst screen --style value --top 10
investskill-analyst screen --tickers AAPL,MSFT,NVDA,AMD,INTC,AVGO,QCOM,TXN

# 2. Work up one name against its peers, with a trade plan sized to YOUR account
investskill-analyst analyze NVDA --account 50000 --risk 0.01 --max-position 0.08

# 3. Check whether the ranking rule has actually worked
investskill-analyst backtest --years 8 --top-n 10 --equity-csv equity.csv

# 4. Have Claude write a full research note (framework + numbers + news)
investskill-analyst research NVDA
investskill-analyst research "Find 3 high-quality stocks near a pullback entry" --framework stock-screener
investskill-analyst research AAPL --framework bear-case --out aapl-bear.md

# Offline demo: any command with --provider synthetic (fake data, no network)
investskill-analyst screen --provider synthetic
```

`--json` on `screen`/`analyze`/`backtest` gives machine-readable output for
spreadsheets or your own scripts. `investskill-analyst frameworks` lists every
InvestSkill framework the agent can load.

### Style presets (`--style`)

| Style | Value | Quality | Growth | Momentum | Low risk |
|-------|------:|--------:|-------:|---------:|---------:|
| balanced (default) | 20% | 25% | 15% | 25% | 15% |
| value | 40% | 30% | 5% | 10% | 15% |
| growth | 5% | 20% | 40% | 30% | 5% |
| momentum | 5% | 15% | 10% | 60% | 10% |
| defensive | 20% | 35% | 5% | 10% | 30% |

### Your own data

`--provider csv --data-dir ./mydata` reads `mydata/<TICKER>.csv` (a date column
plus `open,high,low,close,volume`) and an optional `mydata/fundamentals.json`
(`{"AAPL": {"trailing_pe": 31.2, "roe": 1.47, ...}}`, using the field names in
`data.Fundamentals`). Use this for a paid data vendor or a broker export.

## How to read the output

- **Score (0–10) is relative.** 9.0 means "top decile of *this universe* today",
  not "will go up". Screen a different universe and the scores change.
- **Signal bands follow InvestSkill's Score Guide:** ≥ 8 strong BUY, 6–7.9 BUY,
  4–5.9 HOLD, 2–3.9 SELL, < 2 strong SELL. A BUY on a stock below its 200-day
  average is downgraded to WEAK conviction, with a note to wait for the trend to
  turn back up.
- **Confidence** comes from data coverage and whether the five factors agree.
- **The trade plan starts from risk.** The share count is whatever keeps a stop-out
  at or under `--risk` of the account, capped at `--max-position`. HOLD and SELL
  signals never produce a buy.

## Honest limitations

1. **The backtest only tests price factors.** Free data has no point-in-time
   fundamentals, and scoring history with today's P/E would be look-ahead bias.
   The value, quality and growth factors are an overlay the backtest does not validate.
2. **Survivorship bias.** Built-in universes are today's large caps, and every
   long-only backtest on them looks better than it should.
3. **Yahoo data is unofficial.** It is rate-limited, sometimes stale, and fields
   occasionally go missing. Missing inputs lower the Confidence rating. Verify
   anything you trade on against SEC filings (the `fact-check` framework does this).
4. **The agent can be wrong.** The numbers it quotes come from tools, but its
   interpretation and anything it finds on the web can still be mistaken. Read the
   cited sources.
5. **No order execution.** This is a research tool. See *Extending* below if you
   want paper trading.

## Extending it with *Awesome AI in Finance*

The design borrows from projects on the
[awesome-ai-in-finance](https://github.com/georgezouq/awesome-ai-in-finance) list,
and several of them are natural next steps:

| Want to… | Look at | How it fits |
|----------|---------|-------------|
| Multi-agent debate (bull vs. bear vs. risk manager) | TradingAgents, FinRobot, AI Hedge Fund | Run `research` twice (`stock-eval`, then `bear-case`) and have a third call adjudicate between them; or port their agent roles onto these tools |
| Primary-source fundamentals and filings | edgartools, sec-edgar-mcp (MCP servers) | Add a `DataProvider` backed by EDGAR XBRL; it also enables point-in-time fundamentals for the backtest |
| Macro regime overlay | fred-mcp-server | Feed yield curve / unemployment into `economics-analysis` or a regime switch on the style weights |
| Proper factor evaluation | alphalens, empyrical, pyfolio | `backtest --equity-csv` output plugs straight into their tear sheets |
| Portfolio construction beyond equal weight | skfolio, DeepDow | Replace the top-N equal weight in `backtest.py` with mean-variance / risk-parity weights |
| Paper trading | alpaca-mcp-server, pylivetrader | Turn a `TradePlan` into bracket orders on a **paper** account first |
| Stricter backtest validation | Quant Research skill, backtrader, LEAN | Parameter-stability and walk-forward checks before trusting any rule |

## Tests

```bash
cd analyst && python -m pytest -q
```

The tests run offline. They use the synthetic market and a scripted fake Claude
client, and cover indicator math, factor ranking, score bands, risk sizing,
no-look-ahead in the backtest, tool-input validation and the agent loop.

---

*Educational and research use only. Not financial advice. Nothing here is a
recommendation to buy or sell any security. Consult a licensed financial
advisor before making investment decisions.*
