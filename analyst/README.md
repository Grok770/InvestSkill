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
| SEC data | `edgar.py` | Audited 10-K/10-Q figures from SEC EDGAR (free, no key), trailing-12-month values, 15-year history, source recorded for each field |
| Data checks | `quality.py` | Stale prices, unadjusted splits, gaps, impossible values, and disagreement between two sources |
| Track record | `financials.py` | 10-year business history (revenue, EPS, FCF, margins, ROE, CAGRs) and stock history (annual and calendar-year returns vs. SPY) |
| Buy/sell indicator | `verdict.py` | One verdict from rank + business trend + price timing, with its reasons, safety rails, and a per-stock track record |
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
# 0. Use audited SEC data (recommended for real decisions; free, no key)
export SEC_USER_AGENT="Your Name you@example.com"   # SEC requires a contact email
# then add --provider edgar to any command below


# 1. Find candidates: rank 56 US large caps (or your own list)
investskill-analyst screen
investskill-analyst screen --style value --top 10
investskill-analyst screen --tickers AAPL,MSFT,NVDA,AMD,INTC,AVGO,QCOM,TXN

# 2. Work up one name against its peers: verdict, reasons, data checks, trade plan sized to YOUR account
investskill-analyst analyze NVDA --provider edgar --account 50000 --risk 0.01 --max-position 0.08

# 2b. The buy/sell indicator for a whole watchlist
investskill-analyst signal AAPL,MSFT,NVDA,JPM,XOM --provider edgar

# 2c. The company's past performance: stock vs. SPY, 10+ years of financials, and the indicator's track record
investskill-analyst history AAPL --provider edgar --years 15

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

## Accurate data

Numbers come from three places. Each field is labelled with its source in the
report's **Data quality & sources** section.

| Provider | Fundamentals | Prices | Accuracy |
|----------|--------------|--------|----------|
| `--provider edgar` (**recommended**) | **SEC EDGAR**: the company's own audited 10-K and 10-Q filings, up to ~15 years; trailing-12-month values = last 10-K + this year's YTD − last year's YTD | Yahoo | Primary source. Only forward P/E and beta, which aren't in filings, come from Yahoo, and they're labelled |
| `--provider yfinance` | Yahoo Finance (about 4 years) | Yahoo | Unofficial. Usually right, sometimes stale or missing |
| `--provider csv` | Your files (a paid vendor, a broker export) | Your files | As good as your source |

With `edgar`, every analysis also **cross-checks** SEC figures against Yahoo and
flags any field where the two differ by more than 15%. Every run also checks:

- stale prices (last bar more than 5 trading days old)
- one-day moves over 35% (often a split that wasn't adjusted)
- gaps in the price history
- negative P/E
- ROE over 100% (usually buyback-shrunk equity rather than real profitability)
- missing inputs

A warning lowers how much you should trust that report. It doesn't block it.

For fully paid-grade data, point `--provider csv` at an export from a vendor
(Polygon, Financial Modeling Prep, Alpha Vantage, etc., all listed in
awesome-ai-in-finance), or add a small provider class in `data.py`.

## Past performance

`investskill-analyst history TICKER` answers three questions:

1. **How has the stock done?** 1/3/5/10-year annualized returns and calendar-year
   returns next to SPY, plus best and worst year, volatility, maximum drawdown,
   and distance from the all-time high.
2. **How has the business done?** A year-by-year table of revenue, net income,
   EPS, free cash flow, operating margin and ROE. Also 3/5/10-year CAGRs, how many
   years revenue grew, and a **business trend score** (0–10, IMPROVING / STABLE /
   DETERIORATING) with its reasons.
3. **Has the timing signal worked on this stock?** At every past month-end, the
   timing signal is recomputed from data available on that day. The table shows
   what happened over the next 3 months after BUY-zone and SELL-zone readings,
   compared with the stock's average.

## The buy/sell indicator

Every `analyze` report opens with a verdict. `signal` gives the same verdict for
a whole watchlist:

```
## Bottom line: **BUY** (7.1 / 10)
`SELL ━━━━━━━━━━━━━━●━━━━━━ BUY`
```

| Pillar | Weight | Measures |
|--------|-------:|----------|
| Valuation & quality rank | 40% | Rank vs. peers on value, quality, growth, momentum and low risk |
| Business trend | 35% | Revenue and EPS growth, margin direction, FCF consistency, ROE (from filings) |
| Price timing | 25% | 200/50-day averages, golden/death cross, MACD, RSI |

**Score → verdict:** ≥ 8 STRONG BUY · 6–7.9 BUY · 4–5.9 HOLD · 2–3.9 SELL ·
< 2 STRONG SELL, which is InvestSkill's Score Guide. Each verdict lists its
reasons **in favour** and **against**.

**Safety rails can only turn a BUY into a HOLD. They never create one:**

- a clear price downtrend (timing < 3) means "good company, wait for the trend";
- a deteriorating business (trend < 3) means "cheap for a reason?".

HOLD and SELL verdicts never produce a share count in the trade plan.

**What it is not:** a prediction. It summarizes evidence and says which way that
evidence leans. The `history` track record shows how the timing part has done on
each stock. `backtest` tests the price-based ranking across a universe. Neither
guarantees the future.

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
3. **Yahoo data is unofficial.** Use `--provider edgar` for audited fundamentals.
   Prices still come from Yahoo, and the automatic checks flag stale or broken
   price series. SEC data is annual or quarterly, so the business picture can
   lag events by up to a quarter.
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
| Richer filings (segments, 8-K events, insider Form 4) | edgartools, sec-edgar-mcp (MCP servers) | `edgar.py` already reads XBRL company facts. Those tools add filing text and insider data. The `filed` dates in the EDGAR table also make a point-in-time fundamental backtest possible |
| Macro regime overlay | fred-mcp-server | Feed yield curve / unemployment into `economics-analysis` or a regime switch on the style weights |
| Proper factor evaluation | alphalens, empyrical, pyfolio | `backtest --equity-csv` output plugs straight into their tear sheets |
| Portfolio construction beyond equal weight | skfolio, DeepDow | Replace the top-N equal weight in `backtest.py` with mean-variance / risk-parity weights |
| Paper trading | alpaca-mcp-server, pylivetrader | Turn a `TradePlan` into bracket orders on a **paper** account first |
| Stricter backtest validation | Quant Research skill, backtrader, LEAN | Parameter-stability and walk-forward checks before trusting any rule |

## Tests

```bash
cd analyst && python -m pytest -q
```

The tests run offline. They use the synthetic market, an SEC company-facts
fixture in the real EDGAR format, and a scripted fake Claude client. Coverage:

- indicator math, factor ranking, score bands and risk sizing
- EDGAR parsing: concept changes, restatements, trailing-12-month values and share counts
- data-quality checks
- business trend scoring, verdict labels and safety rails
- no look-ahead in the backtest and the track record
- tool-input validation and the agent loop

---

*Educational and research use only. Not financial advice. Nothing here is a
recommendation to buy or sell any security. Consult a licensed financial
advisor before making investment decisions.*
