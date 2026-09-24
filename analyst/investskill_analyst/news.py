"""News, SEC 8-K filings and X (Twitter) posts → a scored "news outlook".

Pipeline::

    fetch (yahoo · google · sec · x · file · synthetic)
      → score each item (Claude, or an offline finance lexicon)
      → aggregate: recency-, source- and impact-weighted sentiment
      → NewsSignal: 0–10 outlook, key developments, red flags

What this is — and is not
-------------------------
Fresh news moves prices over days to weeks, and material events (an 8-K
restatement notice, a guidance cut, an investigation) often matter for
longer. The outlook summarizes *which way new developments lean*. It is not
a price forecast. There is no free historical news archive to backtest it
against, so it enters the verdict with a small weight (15%) and its
confidence depends on how many items it saw. Social posts are the noisiest
source and are down-weighted accordingly.

Sources
-------
* ``yahoo``  — headlines via yfinance (free, unofficial).
* ``google`` — Google News RSS search (free).
* ``sec``    — the company's recent 8-K filings from SEC EDGAR, classified
  by Item number (free; needs ``SEC_USER_AGENT``). The most reliable source.
* ``x``      — X API v2 recent search (``X_BEARER_TOKEN``; requires an X
  developer plan with search access — check developer.x.com for tiers).
* ``file``   — a JSON list of items you collected yourself (e.g. exported
  posts): ``[{"title": ..., "published": ISO date, "kind": "social", ...}]``.
* ``synthetic`` — deterministic demo items for offline use and tests.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from email.utils import parsedate_to_datetime
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

UTC = timezone.utc

SOURCE_WEIGHT = {"filing": 1.0, "news": 0.8, "social": 0.35}
IMPACT_WEIGHT = {"low": 0.5, "medium": 1.0, "high": 2.0}
HALF_LIFE_DAYS = 3.0
SHRINK_K = 2.0  # evidence needed before the outlook moves far from neutral

EVENTS = ["earnings", "guidance", "analyst", "product", "deal", "legal", "regulatory",
          "management", "capital", "accounting", "macro", "other"]

# Events that, when strongly negative, block a BUY regardless of the other scores.
RED_FLAG_PATTERNS = {
    "accounting": "possible accounting problem (restatement / non-reliance / auditor change)",
    "legal": "legal or regulatory action",
    "regulatory": "legal or regulatory action",
    "capital": "financing stress (bankruptcy, delisting, dilution)",
}

SEVERE_LEGAL = re.compile(r"fraud|investigat|subpoena|indict|charged|criminal|wells notice")

# SEC 8-K item → (event, sentiment prior, impact, description)
EIGHT_K_ITEMS = {
    "1.01": ("deal", 0.2, "medium", "entered a material agreement"),
    "1.02": ("deal", -0.3, "medium", "terminated a material agreement"),
    "1.03": ("capital", -1.0, "high", "bankruptcy or receivership"),
    "1.05": ("other", -0.5, "high", "material cybersecurity incident"),
    "2.01": ("deal", 0.1, "medium", "completed an acquisition or disposition"),
    "2.02": ("earnings", 0.0, "high", "reported results of operations"),
    "2.03": ("capital", -0.1, "low", "took on a material financial obligation"),
    "2.05": ("capital", -0.4, "medium", "exit or restructuring costs"),
    "2.06": ("accounting", -0.6, "high", "material impairment"),
    "3.01": ("capital", -0.8, "high", "delisting notice or listing-rule failure"),
    "3.02": ("capital", -0.2, "low", "unregistered sale of equity (dilution)"),
    "4.01": ("accounting", -0.5, "high", "changed auditor"),
    "4.02": ("accounting", -1.0, "high", "prior financial statements can no longer be relied on"),
    "5.02": ("management", -0.1, "medium", "director or officer departure/appointment"),
    "5.07": ("other", 0.0, "low", "shareholder vote results"),
    "7.01": ("other", 0.0, "low", "Regulation FD disclosure"),
    "8.01": ("other", 0.0, "low", "other events"),
}


@dataclass
class NewsItem:
    title: str
    published: datetime
    kind: str = "news"            # news | filing | social
    source: str = ""
    url: str = ""
    text: str = ""
    engagement: int = 0           # likes + reposts for social posts
    id: str = ""
    # filled by a scorer
    sentiment: float | None = None   # −1 … +1
    relevance: float | None = None   # 0 … 1
    event: str | None = None
    impact: str | None = None        # low | medium | high
    rationale: str = ""

    def __post_init__(self):
        if self.published.tzinfo is None:
            self.published = self.published.replace(tzinfo=UTC)
        if not self.id:
            self.id = f"n{zlib.crc32((self.title + self.url).encode()) & 0xffffff:06x}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat()
        return d


# ------------------------------------------------------------------ fetchers ---

def _http_get(url: str, headers: dict | None = None, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 investskill-analyst",
                                               **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_yahoo(ticker: str) -> list[NewsItem]:
    import yfinance as yf

    items = []
    for raw in yf.Ticker(ticker).news or []:
        c = raw.get("content", raw)  # newer yfinance nests fields under "content"
        title = c.get("title")
        when = c.get("pubDate") or c.get("displayTime") or raw.get("providerPublishTime")
        if not title or not when:
            continue
        published = (datetime.fromtimestamp(when, UTC) if isinstance(when, (int, float))
                     else datetime.fromisoformat(str(when).replace("Z", "+00:00")))
        url = (c.get("canonicalUrl") or {}).get("url") or raw.get("link", "")
        provider = (c.get("provider") or {}).get("displayName") or raw.get("publisher", "Yahoo")
        items.append(NewsItem(title, published, "news", provider, url, c.get("summary", "") or ""))
    return items


def fetch_google(ticker: str, company: str | None = None) -> list[NewsItem]:
    q = f'"{company}" OR {ticker} stock' if company else f"{ticker} stock"
    url = ("https://news.google.com/rss/search?" +
           urllib.parse.urlencode({"q": q + " when:14d", "hl": "en-US", "gl": "US", "ceid": "US:en"}))
    root = ET.fromstring(_http_get(url))
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        pub = it.findtext("pubDate")
        if not title or not pub:
            continue
        src = it.find("source")
        items.append(NewsItem(title, parsedate_to_datetime(pub), "news",
                              src.text if src is not None else "Google News",
                              it.findtext("link") or ""))
    return items


def fetch_sec_8k(ticker: str, client=None, days: int = 30) -> list[NewsItem]:
    """Recent 8-K filings, pre-scored from their Item numbers."""
    from .edgar import EdgarClient

    client = client or EdgarClient()
    cik = client.cik(ticker)
    sub = client._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"submissions_{cik}.json")
    recent = sub.get("filings", {}).get("recent", {})
    cutoff = datetime.now(UTC) - timedelta(days=days)
    items = []
    for i, form in enumerate(recent.get("form", [])):
        if form not in ("8-K", "8-K/A"):
            continue
        filed = datetime.fromisoformat(recent["filingDate"][i]).replace(tzinfo=UTC)
        if filed < cutoff:
            continue
        codes = [c.strip() for c in str(recent.get("items", [""] * (i + 1))[i]).split(",") if c.strip()]
        known = [EIGHT_K_ITEMS[c] for c in codes if c in EIGHT_K_ITEMS]
        acc = recent["accessionNumber"][i].replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
               f"{recent['primaryDocument'][i]}")
        desc = "; ".join(k[3] for k in known) or "current report"
        item = NewsItem(f"8-K: {desc} (Items {', '.join(codes) or 'n/a'})", filed, "filing",
                        "SEC EDGAR", url)
        if known:
            worst = min(known, key=lambda k: k[1])  # the most negative item dominates
            item.sentiment, item.event, item.impact = worst[1], worst[0], worst[2]
            item.relevance = 1.0
            item.rationale = "classified from the 8-K Item number"
            if all(k[1] == 0 for k in known):
                item.sentiment = None  # neutral items (e.g. 2.02 results): let the scorer read the title
        items.append(item)
    return items


def fetch_x(ticker: str, company: str | None = None, bearer: str | None = None,
            max_results: int = 50) -> list[NewsItem]:
    """X API v2 recent search (last 7 days). Needs X_BEARER_TOKEN."""
    bearer = bearer or os.environ.get("X_BEARER_TOKEN")
    if not bearer:
        raise RuntimeError("X_BEARER_TOKEN is not set (X developer account with search access required)")
    terms = f'(${ticker} OR "{company}")' if company else f'(${ticker} OR "{ticker} stock")'
    params = {"query": f"{terms} lang:en -is:retweet -is:reply",
              "max_results": str(max(10, min(max_results, 100))),
              "tweet.fields": "created_at,public_metrics,author_id"}
    raw = _http_get("https://api.x.com/2/tweets/search/recent?" + urllib.parse.urlencode(params),
                    {"Authorization": f"Bearer {bearer}"})
    items = []
    for t in json.loads(raw).get("data", []):
        m = t.get("public_metrics", {})
        engagement = int(m.get("like_count", 0) + 2 * m.get("retweet_count", 0) + m.get("quote_count", 0))
        items.append(NewsItem(t["text"][:280], datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")),
                              "social", "X", f"https://x.com/i/web/status/{t['id']}",
                              engagement=engagement, id=f"x{t['id']}"))
    return items


def load_file(path: str | Path) -> list[NewsItem]:
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        r = dict(r)
        r["published"] = datetime.fromisoformat(str(r["published"]).replace("Z", "+00:00"))
        out.append(NewsItem(**{k: v for k, v in r.items() if k in NewsItem.__dataclass_fields__}))
    return out


_SYN_POS = ["{t} beats quarterly estimates and raises full-year guidance",
            "Analysts upgrade {t} to buy on strong demand",
            "{t} announces record revenue from new product line",
            "{t} unveils $10B share buyback program",
            "{t} wins major multi-year contract"]
_SYN_NEG = ["{t} misses earnings estimates, cuts outlook",
            "Regulators open investigation into {t} business practices",
            "{t} downgraded to sell as growth slows",
            "{t} announces layoffs amid weak demand",
            "{t} faces class-action lawsuit over disclosures"]
_SYN_NEU = ["{t} to present at industry conference next week",
            "{t} shares move with the broader market",
            "What to watch in {t} earnings next month"]


def synthetic_news(ticker: str, now: datetime | None = None, n: int = 12) -> list[NewsItem]:
    """Deterministic demo headlines with a per-ticker tilt. Fake — never trade on them."""
    now = now or datetime(2026, 6, 30, tzinfo=UTC)
    rng = np.random.default_rng(zlib.crc32(ticker.encode()) ^ 99)
    tilt = rng.uniform(-0.6, 0.6)
    items = []
    for i in range(n):
        u = rng.uniform()
        pool = _SYN_POS if u < 0.4 + tilt / 2 else _SYN_NEG if u < 0.8 else _SYN_NEU
        kind = "social" if i % 4 == 3 else "news"
        items.append(NewsItem(pool[int(rng.integers(len(pool)))].format(t=ticker),
                              now - timedelta(days=float(rng.uniform(0, 14))), kind,
                              "X (synthetic)" if kind == "social" else "Newswire (synthetic)",
                              engagement=int(rng.integers(0, 5000)) if kind == "social" else 0))
    return items


def collect(ticker: str, sources: list[str], company: str | None = None,
            news_file: str | None = None, sec_client=None, on_error=None,
            days: int = 14) -> list[NewsItem]:
    """Fetch from every requested source; a failing source is reported, not fatal."""
    fetchers = {
        "yahoo": lambda: fetch_yahoo(ticker),
        "google": lambda: fetch_google(ticker, company),
        "sec": lambda: fetch_sec_8k(ticker, sec_client, days=max(days, 30)),
        "x": lambda: fetch_x(ticker, company),
        "file": lambda: load_file(news_file) if news_file else [],
        "synthetic": lambda: synthetic_news(ticker),
    }
    items: list[NewsItem] = []
    for s in sources:
        if s not in fetchers:
            raise ValueError(f"unknown news source {s!r}; choose from {sorted(fetchers)}")
        try:
            items += fetchers[s]()
        except Exception as exc:  # noqa: BLE001 - one dead source must not sink the rest
            if on_error:
                on_error(s, exc)
    newest = max((i.published for i in items), default=datetime.now(UTC))
    cutoff = newest - timedelta(days=days)
    seen, out = set(), []
    for it in sorted(items, key=lambda i: i.published, reverse=True):
        key = re.sub(r"\W+", " ", it.title.lower()).strip()[:90]
        if key in seen or (it.kind != "filing" and it.published < cutoff):
            continue
        seen.add(key)
        out.append(it)
    return out


# ------------------------------------------------------------------- scorers ---

_POS = {"beat", "beats", "raise", "raises", "raised", "upgrade", "upgrades", "upgraded", "record",
        "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "approval", "approved",
        "wins", "win", "partnership", "buyback", "repurchase", "outperform", "strong", "growth",
        "expands", "launch", "launches", "breakthrough", "profit", "exceeds", "tops", "bullish",
        "dividend", "accelerates", "boost", "boosts"}
_NEG = {"miss", "misses", "missed", "cut", "cuts", "downgrade", "downgrades", "downgraded",
        "lawsuit", "sued", "probe", "investigation", "recall", "layoffs", "layoff", "resigns",
        "fraud", "plunge", "plunges", "slump", "falls", "drop", "drops", "weak", "warning",
        "warns", "decline", "declines", "loss", "losses", "delay", "delays", "bearish", "fine",
        "fined", "penalty", "antitrust", "subpoena", "restatement", "bankruptcy", "halt",
        "slows", "underperform", "short", "dilution"}
_EVENT_RULES = [
    ("accounting", r"restat|non-reliance|auditor|accounting|impairment"),
    ("legal", r"lawsuit|sued|court|class.action|settle|fraud|subpoena"),
    ("regulatory", r"regulat|investigation|probe|antitrust|\bsec\b|\bftc\b|\bdoj\b|\bfda\b|approval"),
    ("guidance", r"guidance|outlook|forecast"),
    ("earnings", r"earnings|quarter|results|\beps\b|revenue"),
    ("analyst", r"upgrade|downgrade|price target|analyst|rating"),
    ("deal", r"acquir|acquisition|merger|deal|contract|partnership"),
    ("capital", r"buyback|repurchase|dividend|offering|debt|bankruptcy|dilution"),
    ("product", r"launch|unveil|product|release|chip|model"),
    ("management", r"\bceo\b|\bcfo\b|resign|appoint|executive|board"),
]


class LexiconScorer:
    """Offline, deterministic finance-word scorer. Crude, but transparent."""

    name = "lexicon"

    def score(self, ticker: str, items: list[NewsItem], company: str | None = None) -> list[NewsItem]:
        # Word-boundary match on the ticker/cashtag and the company's first name word, so
        # "V" or "MA" don't match inside other words and "Apple Inc." matches "Apple".
        terms = [re.escape(ticker.lower())]
        if company:
            first = re.sub(r"[^a-z0-9]", "", company.lower().split()[0])
            if len(first) >= 3:
                terms.append(first)
        mention = re.compile(r"(?<![a-z0-9])\$?(?:" + "|".join(terms) + r")(?![a-z0-9])")
        for it in items:
            if it.sentiment is not None and it.event:
                continue  # already classified (e.g. an 8-K by Item number)
            text = f"{it.title} {it.text}".lower()
            words = re.findall(r"[a-z\-]+", text)
            pos = sum(w in _POS for w in words)
            neg = sum(w in _NEG for w in words)
            it.sentiment = 0.0 if pos == neg else float(np.tanh((pos - neg) / 1.5))
            it.event = it.event or next((e for e, pat in _EVENT_RULES if re.search(pat, text)), "other")
            # Feeds return related-market stories too; an item that never names the company
            # counts for little.
            it.relevance = it.relevance or (1.0 if mention.search(text) else 0.15)
            it.impact = it.impact or ("high" if it.event in ("earnings", "guidance", "accounting",
                                                            "legal", "regulatory") else "medium")
            it.rationale = it.rationale or f"lexicon: {pos} positive / {neg} negative terms"
        return items


class ClaudeScorer:
    """Scores every item in one structured-output call to Claude.

    Reads each headline/post the way an analyst would ("beats estimates but
    cuts guidance" is negative), judges relevance to *this* company, and tags
    the event type and likely price impact.
    """

    name = "claude"
    MAX_ITEMS = 80

    def __init__(self, model: str | None = None, client=None) -> None:
        from .agent import DEFAULT_MODEL

        self.model = model or DEFAULT_MODEL
        self._client = client

    def score(self, ticker: str, items: list[NewsItem], company: str | None = None) -> list[NewsItem]:
        import anthropic

        todo = [it for it in items if it.sentiment is None or not it.event][: self.MAX_ITEMS]
        if not todo:
            return items
        client = self._client or anthropic.Anthropic()
        listing = "\n".join(
            f"[{it.id}] ({it.kind}, {it.source}, {it.published:%Y-%m-%d}"
            f"{f', engagement {it.engagement}' if it.engagement else ''}) {it.title}"
            + (f" — {it.text[:300]}" if it.text and it.text != it.title else "")
            for it in todo)
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "sentiment": {"type": "number", "description": "-1 very bad … +1 very good for the stock"},
                    "relevance": {"type": "number", "description": "0 unrelated … 1 directly about the company"},
                    "event": {"type": "string", "enum": EVENTS},
                    "impact": {"type": "string", "enum": ["low", "medium", "high"]},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "sentiment", "relevance", "event", "impact", "rationale"],
                "additionalProperties": False,
            }}},
            "required": ["items"],
            "additionalProperties": False,
        }
        prompt = (
            f"Company: {company or ticker} (ticker {ticker}).\n"
            "Score each item below for what it implies about this company's stock over the next "
            "few weeks. Judge the substance, not the tone: a beat with a guidance cut is negative; "
            "a rumor or promotional post from an anonymous account deserves low relevance; items "
            "about a different company that shares a word with this one get relevance near 0. "
            "sentiment is -1 to +1, relevance 0 to 1. rationale: one short sentence.\n\n" + listing
        )
        response = client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"news scoring stopped early ({response.stop_reason})")
        text = next(b.text for b in response.content if b.type == "text")
        by_id = {r["id"]: r for r in json.loads(text)["items"]}
        for it in todo:
            r = by_id.get(it.id)
            if not r:
                continue
            it.sentiment = float(np.clip(r["sentiment"], -1, 1))
            it.relevance = float(np.clip(r["relevance"], 0, 1))
            it.event, it.impact, it.rationale = r["event"], r["impact"], r["rationale"]
        return LexiconScorer().score(ticker, items, company)  # anything Claude skipped


def get_scorer(name: str = "auto", model: str | None = None):
    if name == "auto":
        name = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "lexicon"
    if name == "claude":
        return ClaudeScorer(model)
    if name == "lexicon":
        return LexiconScorer()
    raise ValueError(f"unknown scorer {name!r}; choose claude, lexicon or auto")


# ---------------------------------------------------------------- aggregate ---

@dataclass
class NewsSignal:
    ticker: str
    score: float | None             # 0–10, 5 = neutral
    outlook: str                    # POSITIVE / NEUTRAL / NEGATIVE / NO DATA
    confidence: str
    items_used: int
    by_kind: dict
    attention: float | None         # share of weight from the last 3 days
    key_positive: list[dict] = field(default_factory=list)
    key_negative: list[dict] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    daily: list[dict] = field(default_factory=list)   # [{date, sentiment, count}]
    scorer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def item_weight(it: NewsItem, now: datetime) -> float:
    age = max(0.0, (now - it.published).total_seconds() / 86400)
    w = SOURCE_WEIGHT.get(it.kind, 0.5) * (it.relevance if it.relevance is not None else 0.6)
    w *= IMPACT_WEIGHT.get(it.impact or "medium", 1.0) * 0.5 ** (age / HALF_LIFE_DAYS)
    if it.kind == "social":
        w *= 0.5 + 0.25 * math.log10(1 + it.engagement)  # a viral post counts more, a quiet one less
    return w


def aggregate(ticker: str, items: list[NewsItem], now: datetime | None = None,
              scorer: str = "") -> NewsSignal:
    scored = [it for it in items if it.sentiment is not None]
    if not scored:
        return NewsSignal(ticker, None, "NO DATA", "LOW", 0, {}, None, scorer=scorer)
    now = now or max(it.published for it in scored)
    ws = np.array([item_weight(it, now) for it in scored])
    ss = np.array([it.sentiment for it in scored])
    total = float(ws.sum())
    mean = float((ws * ss).sum() / total) if total else 0.0
    shrunk = mean * total / (total + SHRINK_K)
    score = round(5 + 5 * shrunk, 1)
    outlook = "POSITIVE" if score >= 6 else "NEGATIVE" if score <= 4 else "NEUTRAL"
    confidence = "HIGH" if total >= 6 and len(scored) >= 10 else "MEDIUM" if total >= 2 else "LOW"
    recent = sum(w for w, it in zip(ws, scored) if (now - it.published).days < 3)

    contrib = sorted(zip(ws * ss, scored), key=lambda x: x[0])

    def brief(it: NewsItem) -> dict:
        return {"title": it.title, "date": it.published.date().isoformat(), "kind": it.kind,
                "source": it.source, "url": it.url, "sentiment": round(it.sentiment, 2),
                "event": it.event, "impact": it.impact, "why": it.rationale}

    flags = []
    for it in scored:
        if it.event not in RED_FLAG_PATTERNS or it.sentiment > -0.5 or it.impact != "high" \
                or (it.relevance or 0) < 0.7 or it.kind == "social":
            continue
        # Routine lawsuits are common for large companies; only fraud / government
        # investigations count as red flags among legal & regulatory news.
        if it.event in ("legal", "regulatory") and not SEVERE_LEGAL.search(it.title.lower()):
            continue
        flags.append(f"{RED_FLAG_PATTERNS[it.event]}: {it.title} ({it.published.date()})")

    daily: dict[str, list[float]] = {}
    for it in scored:
        daily.setdefault(it.published.date().isoformat(), []).append(it.sentiment)
    return NewsSignal(
        ticker=ticker, score=score, outlook=outlook, confidence=confidence,
        items_used=len(scored),
        by_kind={k: sum(1 for it in scored if it.kind == k) for k in ("filing", "news", "social")},
        attention=round(recent / total, 2) if total else None,
        key_positive=[brief(it) for c, it in reversed(contrib) if c > 0][:5],
        key_negative=[brief(it) for c, it in contrib if c < 0][:5],
        red_flags=list(dict.fromkeys(flags))[:5],
        daily=[{"date": d, "sentiment": round(float(np.mean(v)), 3), "count": len(v)}
               for d, v in sorted(daily.items())],
        scorer=scorer,
    )


def news_signal(ticker: str, sources: list[str], scorer=None, company: str | None = None,
                news_file: str | None = None, sec_client=None, on_error=None,
                days: int = 14) -> tuple[NewsSignal, list[NewsItem]]:
    items = collect(ticker, sources, company, news_file, sec_client, on_error, days)
    scorer = scorer or LexiconScorer()
    if items:
        try:
            scorer.score(ticker, items, company)
        except Exception as exc:  # noqa: BLE001 - fall back to the offline scorer
            if on_error:
                on_error(f"scorer:{scorer.name}", exc)
            scorer = LexiconScorer()
            scorer.score(ticker, items, company)
    return aggregate(ticker, items, scorer=scorer.name), items


def default_sources(provider_name: str) -> list[str]:
    if provider_name == "synthetic":
        return ["synthetic"]
    sources = ["yahoo", "google"]
    if os.environ.get("SEC_USER_AGENT"):
        sources.append("sec")
    if os.environ.get("X_BEARER_TOKEN"):
        sources.append("x")
    return sources
