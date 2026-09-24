"""Command-line interface: ``investskill-analyst <command> ...``

    screen     rank a universe and print the leaderboard
    analyze    score one ticker vs. peers and print a risk-sized trade plan
    signal     buy/sell indicator for one or more tickers (watchlist)
    history    a company's past performance: stock, business, and indicator track record
    chart      interactive HTML line charts: business vs. share price (+ news sentiment)
    news       news, SEC 8-K and X posts scored into a short-term outlook
    backtest   walk-forward test of the ranking rule
    research   Claude writes a full research note using an InvestSkill framework
    frameworks list the InvestSkill frameworks the agent can use
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import DISCLAIMER, __version__
from .data import get_provider
from .engine import (NewsOptions, analyze, backtest, chart_data, company_history, get_news,
                     screen, watchlist)
from .factors import STYLE_WEIGHTS
from .report import (analysis_markdown, backtest_markdown, history_markdown,
                     news_report_markdown, screen_markdown, watchlist_markdown)
from .signals import RiskSettings
from .universe import UNIVERSES, resolve_universe


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="yfinance",
                   choices=["edgar", "yfinance", "csv", "synthetic"],
                   help="edgar = audited SEC fundamentals + Yahoo prices (most accurate); "
                        "synthetic = offline fake data for demos")
    p.add_argument("--data-dir", help="directory for the csv provider")
    p.add_argument("--sec-user-agent", help="'Name email' for SEC EDGAR (or set SEC_USER_AGENT)")


def _add_universe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--universe", choices=sorted(UNIVERSES), help="built-in universe")
    p.add_argument("--tickers", help="comma-separated tickers (overrides --universe)")
    p.add_argument("--universe-file", help="file with one ticker per line")


def _add_risk_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--account", type=float, default=100_000, help="account size in USD")
    p.add_argument("--risk", type=float, default=0.01, help="max loss per trade as a fraction (0.01 = 1%%)")
    p.add_argument("--max-position", type=float, default=0.10, help="max position as a fraction of the account")


def _add_news_args(p: argparse.ArgumentParser, default_on: bool = True) -> None:
    p.add_argument("--news-sources",
                   help="comma list of yahoo,google,sec,x,file,synthetic (default: yahoo,google "
                        "+ sec if SEC_USER_AGENT is set + x if X_BEARER_TOKEN is set)")
    p.add_argument("--news-file", help="JSON list of your own news items / posts (source 'file')")
    p.add_argument("--scorer", default="auto", choices=["auto", "claude", "lexicon"],
                   help="how to score news: claude (best, needs API key), lexicon (offline), "
                        "auto = claude when ANTHROPIC_API_KEY is set")
    p.add_argument("--news-days", type=int, default=14, help="look-back window in days")
    if default_on:
        p.add_argument("--no-news", action="store_true", help="skip news (faster, offline)")


def _news_opts(a) -> NewsOptions | None:
    from .news import default_sources, get_scorer

    if getattr(a, "no_news", False):
        return None
    if a.news_sources:
        sources = [s.strip() for s in a.news_sources.split(",") if s.strip()]
    else:
        sources = default_sources(a.provider)
    if a.news_file and "file" not in sources:
        sources.append("file")
    return NewsOptions(sources, get_scorer(a.scorer), a.news_file, a.news_days)


def _risk(a) -> RiskSettings:
    return RiskSettings(account_size=a.account, risk_per_trade=a.risk, max_position_pct=a.max_position)


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text + "\n")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="investskill-analyst", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("screen", help="rank a universe")
    _add_data_args(p)
    _add_universe_args(p)
    p.add_argument("--style", default="balanced", choices=sorted(STYLE_WEIGHTS))
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out")

    p = sub.add_parser("analyze", help="one ticker vs. peers, with a trade plan")
    p.add_argument("ticker")
    _add_data_args(p)
    _add_universe_args(p)
    _add_risk_args(p)
    _add_news_args(p)
    p.add_argument("--style", default="balanced", choices=sorted(STYLE_WEIGHTS))
    p.add_argument("--json", action="store_true")
    p.add_argument("--out")

    p = sub.add_parser("signal", help="buy/sell indicator for a watchlist")
    p.add_argument("tickers", help="comma-separated, e.g. AAPL,MSFT,NVDA")
    _add_data_args(p)
    _add_news_args(p)
    p.add_argument("--universe", choices=sorted(UNIVERSES), help="peer universe for ranking")
    p.add_argument("--style", default="balanced", choices=sorted(STYLE_WEIGHTS))
    p.add_argument("--json", action="store_true")
    p.add_argument("--out")

    p = sub.add_parser("history", help="a company's past performance")
    p.add_argument("ticker")
    _add_data_args(p)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out")

    p = sub.add_parser("chart", help="HTML line charts: business vs. share price")
    p.add_argument("tickers", help="comma-separated, e.g. AAPL,MSFT")
    _add_data_args(p)
    _add_news_args(p)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--out", default="performance.html", help="output HTML file")

    p = sub.add_parser("news", help="news / 8-K / X outlook for one ticker")
    p.add_argument("ticker")
    _add_data_args(p)
    _add_news_args(p, default_on=False)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out")

    p = sub.add_parser("backtest", help="walk-forward test of the ranking rule")
    _add_data_args(p)
    _add_universe_args(p)
    p.add_argument("--years", type=float, default=5)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=10)
    p.add_argument("--json", action="store_true")
    p.add_argument("--equity-csv", help="write the daily equity curves to this CSV")
    p.add_argument("--out")

    p = sub.add_parser("research", help="Claude-written research note (needs Anthropic credentials)")
    p.add_argument("request", nargs="+",
                   help='a ticker ("NVDA") or a question ("find 3 undervalued quality stocks")')
    _add_data_args(p)
    _add_risk_args(p)
    p.add_argument("--framework", default="stock-eval",
                   help="InvestSkill framework to apply (see `frameworks`)")
    p.add_argument("--model", default=None, help="Claude model id (default: claude-opus-5)")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--no-web", action="store_true", help="disable web search")
    p.add_argument("--out", help="also save the final report to this file")

    sub.add_parser("frameworks", help="list InvestSkill frameworks")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (LookupError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _research(args, provider) -> str:
    import anthropic

    from .agent import DEFAULT_MODEL, run_research

    request = " ".join(args.request)
    if len(args.request) == 1 and len(request) <= 6 and \
            request.replace("-", "").replace(".", "").isalnum():
        request = (f"Research {request.upper()} for me. Start with analyze_stock and "
                   f"company_history, then check recent news and filings, and give me a "
                   f"clear buy/hold/sell view and trade plan.")
    try:
        return run_research(request, provider, framework=args.framework,
                            model=args.model or DEFAULT_MODEL, effort=args.effort,
                            web_search=not args.no_web, risk=_risk(args))
    except anthropic.AuthenticationError as exc:
        raise RuntimeError("Anthropic credentials rejected — set ANTHROPIC_API_KEY "
                           "or run `ant auth login`") from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError("rate limited by the Anthropic API — retry shortly") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError("could not reach the Anthropic API — check your network") from exc


def _run(args) -> int:

    if args.cmd == "frameworks":
        from .agent import list_frameworks
        print("\n".join(list_frameworks()))
        return 0

    provider = get_provider(args.provider, root=args.data_dir, user_agent=args.sec_user_agent)

    if args.cmd == "screen":
        tickers = resolve_universe(args.universe, args.tickers, args.universe_file)
        res = screen(provider, tickers, style=args.style)
        if args.json:
            _emit(json.dumps(res.leaderboard(args.top), indent=2, default=str), args.out)
        else:
            _emit(screen_markdown(res.scored, res.techs, res.funds, args.top, args.style), args.out)

    elif args.cmd == "analyze":
        peers = resolve_universe(args.universe, args.tickers, args.universe_file)
        a = analyze(provider, args.ticker, peers=peers, style=args.style, risk=_risk(args),
                    news=_news_opts(args))
        if args.json:
            _emit(json.dumps({"verdict": a.verdict.to_dict() if a.verdict else None,
                              "signal": a.signal.__dict__, "business": a.business,
                              "news": a.news.to_dict() if a.news else None,
                              "technicals": a.tech, "fundamentals": a.fund.to_dict(),
                              "trade_plan": a.plan.to_dict(), "data_quality": a.data_quality},
                             indent=2, default=str), args.out)
        else:
            _emit(analysis_markdown(a.ticker, a.row, a.signal, a.tech, a.fund, a.plan,
                                    a.universe_size, a.verdict, a.business, a.data_quality,
                                    a.news),
                  args.out)

    elif args.cmd == "signal":
        tickers = resolve_universe(tickers=args.tickers)
        peers = resolve_universe(args.universe) if args.universe else None
        verdicts = watchlist(provider, tickers, peers=peers, style=args.style,
                             news=_news_opts(args))
        _emit(json.dumps([v.to_dict() for v in verdicts], indent=2, default=str) if args.json
              else watchlist_markdown(verdicts), args.out)

    elif args.cmd == "history":
        h = company_history(provider, args.ticker, years=args.years, benchmark=args.benchmark)
        if args.json:
            _emit(json.dumps({"stock": h.stock, "business": h.business,
                              "track_record": h.track.to_dict() if h.track else None,
                              "financials": None if h.financials is None else
                              json.loads(h.financials.to_json(orient="index", date_format="iso"))},
                             indent=2, default=str), args.out)
        else:
            _emit(history_markdown(h), args.out)

    elif args.cmd == "chart":
        from .charts import performance_html

        opts = _news_opts(args)
        companies = [chart_data(provider, t, years=args.years, benchmark=args.benchmark, news=opts)
                     for t in resolve_universe(tickers=args.tickers)]
        Path(args.out).write_text(performance_html(companies, disclaimer=DISCLAIMER))
        print(f"wrote {args.out} — open it in a browser", file=sys.stderr)

    elif args.cmd == "news":
        try:
            company = provider.fundamentals(args.ticker.upper()).name
        except Exception:  # noqa: BLE001 - a name only sharpens the search
            company = None
        signal, items = get_news(provider, args.ticker.upper(), _news_opts(args), company)
        _emit(json.dumps({"signal": signal.to_dict(), "items": [i.to_dict() for i in items]},
                         indent=2, default=str) if args.json
              else news_report_markdown(signal, items), args.out)

    elif args.cmd == "backtest":
        tickers = resolve_universe(args.universe, args.tickers, args.universe_file)
        result, daily = backtest(provider, tickers, years=args.years, top_n=args.top_n,
                                 cost_bps=args.cost_bps)
        if args.equity_csv:
            daily.to_csv(args.equity_csv)
        _emit(json.dumps(result.to_dict(), indent=2) if args.json
              else backtest_markdown(result, args.cost_bps), args.out)

    elif args.cmd == "research":
        report = _research(args, provider)
        if args.out:
            Path(args.out).write_text(report + "\n")
            print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
