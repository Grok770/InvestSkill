"""Command-line interface: ``investskill-analyst <command> ...``

    screen     rank a universe and print the leaderboard
    analyze    score one ticker vs. peers and print a risk-sized trade plan
    backtest   walk-forward test of the ranking rule
    research   Claude writes a full research note using an InvestSkill framework
    frameworks list the InvestSkill frameworks the agent can use
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .data import get_provider
from .engine import analyze, backtest, screen
from .factors import STYLE_WEIGHTS
from .report import analysis_markdown, backtest_markdown, screen_markdown
from .signals import RiskSettings
from .universe import UNIVERSES, resolve_universe


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="yfinance", choices=["yfinance", "csv", "synthetic"],
                   help="data source (synthetic = offline fake data for demos)")
    p.add_argument("--data-dir", help="directory for the csv provider")


def _add_universe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--universe", choices=sorted(UNIVERSES), help="built-in universe")
    p.add_argument("--tickers", help="comma-separated tickers (overrides --universe)")
    p.add_argument("--universe-file", help="file with one ticker per line")


def _add_risk_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--account", type=float, default=100_000, help="account size in USD")
    p.add_argument("--risk", type=float, default=0.01, help="max loss per trade as a fraction (0.01 = 1%%)")
    p.add_argument("--max-position", type=float, default=0.10, help="max position as a fraction of the account")


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
    p.add_argument("--style", default="balanced", choices=sorted(STYLE_WEIGHTS))
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
        request = (f"Research {request.upper()} for me. Start with analyze_stock, then "
                   f"check recent news and filings, and give me a clear view and trade plan.")
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

    provider = get_provider(args.provider, root=args.data_dir) if args.provider == "csv" \
        else get_provider(args.provider)

    if args.cmd == "screen":
        tickers = resolve_universe(args.universe, args.tickers, args.universe_file)
        res = screen(provider, tickers, style=args.style)
        if args.json:
            _emit(json.dumps(res.leaderboard(args.top), indent=2, default=str), args.out)
        else:
            _emit(screen_markdown(res.scored, res.techs, res.funds, args.top, args.style), args.out)

    elif args.cmd == "analyze":
        peers = resolve_universe(args.universe, args.tickers, args.universe_file)
        a = analyze(provider, args.ticker, peers=peers, style=args.style, risk=_risk(args))
        if args.json:
            _emit(json.dumps({"signal": a.signal.__dict__, "technicals": a.tech,
                              "fundamentals": a.fund.to_dict(), "trade_plan": a.plan.to_dict()},
                             indent=2, default=str), args.out)
        else:
            _emit(analysis_markdown(a.ticker, a.row, a.signal, a.tech, a.fund, a.plan,
                                    a.universe_size), args.out)

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
