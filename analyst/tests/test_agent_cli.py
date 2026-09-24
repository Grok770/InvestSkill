import io
import json
from types import SimpleNamespace as NS

import pytest

from investskill_analyst import agent
from investskill_analyst.cli import main
from investskill_analyst.data import SyntheticProvider


# ------------------------------------------------------------------ agent ---

def test_framework_catalog_matches_investskill_prompts():
    names = agent.list_frameworks()
    assert {"stock-eval", "bear-case", "stock-screener"} <= set(names)
    prompt = agent.build_system_prompt("stock-eval")
    assert "INVESTMENT SIGNAL" in prompt and "analyze_stock" in prompt


@pytest.mark.parametrize("bad", ["../CLAUDE", "stock-eval.md", "", "Stock-Eval"])
def test_load_framework_rejects_path_tricks(bad):
    with pytest.raises((ValueError, LookupError)):
        agent.load_framework_text(bad)


def test_tool_executor_validates_input():
    ex = agent.ToolExecutor(SyntheticProvider())
    with pytest.raises(ValueError):
        ex("analyze_stock", {"ticker": "DROP TABLE"})
    with pytest.raises(ValueError):
        ex("screen_stocks", {"tickers": "AAPL"})  # must be a list
    with pytest.raises(ValueError):
        ex("no_such_tool", {})
    out = json.loads(ex("analyze_stock", {"ticker": "nvda", "peers": ["AAPL", "MSFT", "AMD", "INTC"]}))
    assert out["ticker"] == "NVDA" and "trade_plan" in out and "signal" in out
    board = json.loads(ex("screen_stocks", {"universe": "mega-tech", "top": 3}))
    assert len(board["leaderboard"]) == 3


class _FakeStream:
    def __init__(self, message):
        self._m = message
        self.text_stream = [b.text for b in message.content if b.type == "text"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._m


class _FakeClient:
    """Scripted stand-in for anthropic.Anthropic(): tool call, then a report."""

    def __init__(self):
        self.calls = []
        turns = [
            NS(stop_reason="tool_use", content=[
                NS(type="text", text="Pulling the numbers."),
                NS(type="tool_use", id="tu_1", name="analyze_stock",
                   input={"ticker": "AAPL", "peers": ["MSFT", "NVDA", "AMD"]}),
                NS(type="tool_use", id="tu_2", name="load_framework", input={"name": "nope"}),
            ]),
            NS(stop_reason="end_turn", content=[NS(type="text", text="# AAPL report\nBUY")]),
        ]
        self.beta = NS(messages=NS(stream=self._stream))
        self._turns = iter(turns)

    def _stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot
        return _FakeStream(next(self._turns))


def test_run_research_executes_tools_and_returns_report():
    client, out = _FakeClient(), io.StringIO()
    report = agent.run_research("Research AAPL", SyntheticProvider(), client=client, out=out)
    assert report == "# AAPL report\nBUY"
    first, second = client.calls
    assert first["model"] == agent.DEFAULT_MODEL
    assert first["thinking"] == {"type": "adaptive"}
    assert first["fallbacks"] == "default"
    assert any(t.get("type") == "web_search_20260209" for t in first["tools"])
    results = second["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]  # both results, one message
    assert results[0]["is_error"] is False and "trade_plan" in results[0]["content"]
    assert results[1]["is_error"] is True


def test_run_research_stops_on_refusal():
    client = _FakeClient()
    client._turns = iter([NS(stop_reason="refusal", content=[])])
    out = io.StringIO()
    agent.run_research("x", SyntheticProvider(), client=client, out=out, web_search=False)
    assert "declined" in out.getvalue()
    assert not any(t.get("type") == "web_search_20260209" for t in client.calls[0]["tools"])


# -------------------------------------------------------------------- cli ---

def test_cli_screen_analyze_backtest(capsys, tmp_path):
    assert main(["screen", "--provider", "synthetic", "--universe", "mega-tech", "--json"]) == 0
    board = json.loads(capsys.readouterr().out)
    assert board[0]["ticker"] and board[0]["score"] >= board[-1]["score"]

    assert main(["analyze", "AMD", "--provider", "synthetic", "--universe", "mega-tech",
                 "--account", "25000"]) == 0
    text = capsys.readouterr().out
    assert "# AMD" in text and "INVESTMENT SIGNAL" in text and "**Disclaimer:**" in text

    eq = tmp_path / "eq.csv"
    assert main(["backtest", "--provider", "synthetic", "--universe", "us-large-cap",
                 "--years", "4", "--json", "--equity-csv", str(eq)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["top_n"] == 10 and eq.exists()
