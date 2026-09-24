"""News / 8-K / X outlook, its effect on the verdict, and the performance charts."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from investskill_analyst import news
from investskill_analyst.charts import company_chart_data, performance_html
from investskill_analyst.cli import main
from investskill_analyst.data import SyntheticProvider
from investskill_analyst.edgar import EdgarClient
from investskill_analyst.news import (ClaudeScorer, LexiconScorer, NewsItem, aggregate,
                                      fetch_google, fetch_sec_8k, fetch_x, fetch_yahoo, load_file)
from investskill_analyst.verdict import make_verdict

UTC = timezone.utc
NOW = datetime(2026, 6, 30, tzinfo=UTC)
P = SyntheticProvider()


def _item(title, days_ago=0, kind="news", **kw):
    return NewsItem(title, NOW - timedelta(days=days_ago), kind, "test", **kw)


# ------------------------------------------------------------------- scoring ---

def test_lexicon_scores_direction_and_event():
    items = LexiconScorer().score("ACME", [
        _item("ACME beats estimates and raises full-year guidance"),
        _item("ACME misses estimates, cuts outlook"),
        _item("ACME announces $5B buyback"),
        _item("Weather is nice today"),
    ], company="Acme Corp")
    good, bad, buyback, noise = items
    assert good.sentiment > 0 and bad.sentiment < 0
    assert good.event == "guidance" and buyback.event == "capital"
    assert noise.relevance < good.relevance


def test_aggregate_weights_recent_items_and_shrinks_thin_evidence():
    old_good = _item("x", 20, sentiment=1.0, relevance=1.0, event="other", impact="medium")
    new_bad = _item("y", 0, sentiment=-1.0, relevance=1.0, event="other", impact="medium")
    sig = aggregate("ACME", [old_good, new_bad], now=NOW)
    assert sig.score < 5 and sig.outlook in ("NEGATIVE", "NEUTRAL")
    lone = aggregate("ACME", [_item("z", 0, sentiment=1.0, relevance=1.0, event="other",
                                    impact="low")], now=NOW)
    assert 5 < lone.score < 7.5 and lone.confidence == "LOW"   # one item can't swing it far
    assert aggregate("ACME", []).outlook == "NO DATA"


def test_social_posts_count_less_than_filings():
    post = _item("p", 0, "social", sentiment=-1.0, relevance=1.0, event="other", impact="high")
    filing = _item("f", 0, "filing", sentiment=1.0, relevance=1.0, event="other", impact="high")
    assert aggregate("ACME", [post, filing], now=NOW).score > 5


def test_red_flags_only_for_serious_events():
    restatement = _item("8-K: prior financial statements can no longer be relied on", 0, "filing",
                        sentiment=-1.0, relevance=1.0, event="accounting", impact="high")
    routine_suit = _item("ACME faces class-action lawsuit", 0, sentiment=-0.8, relevance=1.0,
                         event="legal", impact="high")
    probe = _item("DOJ opens fraud investigation into ACME", 0, sentiment=-0.9, relevance=1.0,
                  event="regulatory", impact="high")
    rumor = _item("ACME is a fraud!!!", 0, "social", sentiment=-1.0, relevance=1.0,
                  event="accounting", impact="high")
    flags = aggregate("ACME", [restatement, routine_suit, probe, rumor], now=NOW).red_flags
    assert len(flags) == 2
    assert any("relied" in f for f in flags) and any("fraud investigation" in f for f in flags)


def test_claude_scorer_uses_structured_output(monkeypatch):
    items = [_item("ACME beats but cuts guidance"), _item("Unrelated ACME Hardware news")]
    calls = []

    def create(**kw):
        calls.append(kw)
        out = {"items": [
            {"id": items[0].id, "sentiment": -0.6, "relevance": 1.0, "event": "guidance",
             "impact": "high", "rationale": "guidance cut outweighs the beat"},
            {"id": items[1].id, "sentiment": 3.0, "relevance": 0.05, "event": "other",
             "impact": "low", "rationale": "different company"},
        ]}
        return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(out))])

    client = NS(beta=NS(messages=NS(create=create)))
    ClaudeScorer(client=client).score("ACME", items, "Acme Corp")
    assert items[0].sentiment == -0.6 and items[0].event == "guidance"
    assert items[1].sentiment == 1.0 and items[1].relevance == 0.05   # clamped to range
    kw = calls[0]
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]


def test_claude_scorer_failure_falls_back_to_lexicon():
    def create(**kw):
        return NS(stop_reason="refusal", content=[])

    bad = ClaudeScorer(client=NS(beta=NS(messages=NS(create=create))))
    errors = []
    sig, items = news.news_signal("ACME", ["synthetic"], scorer=bad,
                                  on_error=lambda s, e: errors.append(s))
    assert errors == ["scorer:claude"] and sig.scorer == "lexicon" and sig.items_used > 0


# ------------------------------------------------------------------ fetchers ---

def test_fetch_yahoo_handles_old_and_new_formats(monkeypatch):
    import yfinance

    raw = [
        {"content": {"title": "New-format headline", "pubDate": "2026-06-29T12:00:00Z",
                     "summary": "s", "canonicalUrl": {"url": "https://a"},
                     "provider": {"displayName": "Reuters"}}},
        {"title": "Old-format headline", "providerPublishTime": 1782000000,
         "link": "https://b", "publisher": "AP"},
        {"content": {"title": None}},
    ]
    monkeypatch.setattr(yfinance, "Ticker", lambda t: NS(news=raw))
    items = fetch_yahoo("ACME")
    assert [i.source for i in items] == ["Reuters", "AP"] and items[0].url == "https://a"


def test_fetch_google_rss(monkeypatch):
    rss = b"""<?xml version="1.0"?><rss><channel>
      <item><title>ACME soars on record quarter</title><link>https://n/1</link>
        <pubDate>Mon, 29 Jun 2026 10:00:00 GMT</pubDate><source url="x">Bloomberg</source></item>
      <item><title></title><pubDate>Mon, 29 Jun 2026 10:00:00 GMT</pubDate></item>
    </channel></rss>"""
    seen = {}
    monkeypatch.setattr(news, "_http_get", lambda url, headers=None: seen.setdefault("u", url) and rss)
    items = fetch_google("ACME", "Acme Corp")
    assert len(items) == 1 and items[0].source == "Bloomberg"
    assert "Acme+Corp" in seen["u"] or "Acme%20Corp" in seen["u"]


def test_fetch_sec_8k_classifies_items(tmp_path):
    today = datetime.now(UTC).date()
    subs = {"filings": {"recent": {
        "form": ["8-K", "10-Q", "8-K", "8-K"],
        "filingDate": [str(today - timedelta(days=2)), str(today), str(today - timedelta(days=5)),
                       str(today - timedelta(days=400))],
        "items": ["4.02,9.01", "", "2.02,9.01", "1.01"],
        "accessionNumber": ["0001-26-000001", "0001-26-000002", "0001-26-000003", "0001-25-000004"],
        "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"],
    }}}

    def fetch(url):
        if url.endswith("company_tickers.json"):
            return {"0": {"cik_str": 42, "ticker": "ACME", "title": "Acme"}}
        return subs

    items = fetch_sec_8k("ACME", EdgarClient(cache_dir=tmp_path, fetch=fetch))
    assert len(items) == 2                                  # 10-Q and the old 8-K skipped
    restated = items[0]
    assert restated.sentiment == -1.0 and restated.event == "accounting" and restated.impact == "high"
    assert restated.url.endswith("/42/000126000001/a.htm")
    assert items[1].sentiment is None                       # results 8-K: left for the scorer
    assert news.aggregate("ACME", LexiconScorer().score("ACME", items)).red_flags


def test_fetch_x_parses_and_requires_token(monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="X_BEARER_TOKEN"):
        fetch_x("ACME")
    payload = {"data": [{"id": "1", "text": "$ACME crushing it", "created_at": "2026-06-29T10:00:00.000Z",
                         "public_metrics": {"like_count": 10, "retweet_count": 5, "quote_count": 1}}]}
    got = {}

    def fake(url, headers=None):
        got.update(url=url, headers=headers)
        return json.dumps(payload).encode()

    monkeypatch.setattr(news, "_http_get", fake)
    (post,) = fetch_x("ACME", bearer="tok")
    assert post.kind == "social" and post.engagement == 21 and post.url.endswith("/1")
    assert got["headers"]["Authorization"] == "Bearer tok" and "%24ACME" in got["url"]


def test_load_file_and_collect_dedupes(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps([
        {"title": "ACME wins contract", "published": "2026-06-29T00:00:00Z", "kind": "social"},
        {"title": "ACME wins contract!", "published": "2026-06-28T00:00:00Z"},
        {"title": "Old story", "published": "2026-01-01T00:00:00Z"},
    ]))
    assert len(load_file(path)) == 3
    kept = news.collect("ACME", ["file"], news_file=str(path), days=14)
    assert [i.title for i in kept] == ["ACME wins contract"]   # duplicate + stale dropped


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        news.collect("ACME", ["myspace"])


# ------------------------------------------------------------------- verdict ---

UP = dict(above_sma200=True, above_sma50=True, golden_cross=True, macd_hist=0.5, rsi14=55)


def test_news_pillar_and_red_flag_rail():
    good = aggregate("X", [_item(f"g{i}", i, sentiment=0.8, relevance=1.0, event="product",
                                 impact="medium") for i in range(8)], now=NOW)
    base = make_verdict("X", 5.0, UP, {"score": 5.0})
    boosted = make_verdict("X", 5.0, UP, {"score": 5.0}, good)
    assert boosted.pillars["news"] > 5 and boosted.score > base.score
    flagged = aggregate("X", [_item("8-K: auditor change", 0, "filing", sentiment=-0.9,
                                    relevance=1.0, event="accounting", impact="high")], now=NOW)
    v = make_verdict("X", 9.5, UP, {"score": 9.5}, flagged)
    assert v.label == "HOLD" and any("red flag" in r for r in v.rails)


# -------------------------------------------------------------------- charts ---

def _fin():
    ends = pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"])
    return pd.DataFrame({"revenue": [100.0, 120, 150, 180], "eps_diluted": [1.0, 1.5, 2.0, 2.5],
                         "net_income": [10.0, 15, 20, 25],
                         "filed": pd.to_datetime(["2021-02-15", "2022-02-15", "2023-02-15", "2024-02-15"])},
                        index=ends)


def _prices():
    idx = pd.bdate_range("2020-06-01", "2024-06-28")
    return pd.Series(np.linspace(50, 100, len(idx)), index=idx)


def test_chart_data_indexes_to_common_base():
    d = company_chart_data("ACME", _prices(), _fin(), bench=_prices())
    assert d["base"] == "2020-12-31"
    by_key = {s["key"]: s for s in d["indexed"]}
    assert set(by_key) == {"price", "revenue", "eps", "bench"}
    assert by_key["revenue"]["points"][0][1] == 100 and by_key["revenue"]["points"][-1][1] == 180
    assert by_key["eps"]["points"][-1][1] == 250
    assert abs(by_key["price"]["points"][0][1] - 100) < 2   # month-end nearest the base


def test_chart_pe_uses_only_published_eps():
    d = company_chart_data("ACME", _prices(), _fin())
    pe = dict(d["pe"][0]["points"])
    jan_2022 = int(pd.Timestamp("2022-01-31").timestamp() * 1000)
    mar_2022 = int(pd.Timestamp("2022-03-31").timestamp() * 1000)
    price = _prices().resample("ME").last()
    # FY2021 EPS (1.5) wasn't filed until 2022-02-15, so January still uses FY2020's 1.0.
    assert pe[jan_2022] == pytest.approx(price["2022-01-31"] / 1.0, rel=1e-3)
    assert pe[mar_2022] == pytest.approx(price["2022-03-31"] / 1.5, rel=1e-3)


def test_chart_skips_series_that_cannot_be_indexed():
    fin = _fin()
    fin["eps_diluted"] = [-1.0, 0.5, 1.0, 2.0]
    d = company_chart_data("ACME", _prices(), fin)
    assert "eps" not in {s["key"] for s in d["indexed"]}
    assert any("negative" in n for n in d["notes"])
    assert company_chart_data("ACME", _prices(), None)["indexed"][0]["key"] == "price"


def test_performance_html_is_self_contained_and_escaped():
    d = company_chart_data("ACME", _prices(), _fin())
    d["name"] = "</script><script>alert(1)</script>"
    page = performance_html([d], disclaimer="Not advice.")
    assert "<\\/script><script>alert(1)" in page            # can't break out of the data block
    assert "src=" not in page and "cdn" not in page.lower()  # no external scripts
    assert "prefers-color-scheme: dark" in page and "Not advice." in page


def test_cli_chart_and_news(tmp_path, capsys):
    out = tmp_path / "perf.html"
    assert main(["chart", "AAPL,MSFT", "--provider", "synthetic", "--out", str(out)]) == 0
    page = out.read_text()
    assert '"ticker": "AAPL"' in page and '"ticker": "MSFT"' in page and '"news": [{' in page
    assert main(["news", "NVDA", "--provider", "synthetic", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["signal"]["items_used"] > 0 and data["items"]
    assert main(["analyze", "AAPL", "--provider", "synthetic", "--universe", "mega-tech"]) == 0
    assert "## News outlook" in capsys.readouterr().out
    assert main(["signal", "AAPL", "--provider", "synthetic", "--no-news", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["pillars"]["news"] is None


def test_lexicon_relevance_needs_a_real_mention():
    items = LexiconScorer().score("AAPL", [
        _item("Klarna falls 3% as selling persists"),
        _item("Apple beats estimates"),
        _item("$AAPL breaks out"),
    ], company="Apple Inc.")
    assert [i.relevance for i in items] == [0.15, 1.0, 1.0]
    # Short tickers must not match inside ordinary words.
    (visa,) = LexiconScorer().score("V", [_item("Very strong quarter for retailers")], company="Visa Inc.")
    assert visa.relevance == 0.15
