from __future__ import annotations

from pathlib import Path

import pytest

from analyst_agent.config import Settings
from analyst_agent.llm import (
    FakePlanner,
    GoldenEntry,
    LangChainPlanner,
    Turn,
    _format_history,
    _format_result,
    _is_transient,
    load_golden_map,
    normalize,
    planner_from_settings,
)
from analyst_agent.schemas import QueryResult, Usage


def test_normalize() -> None:
    assert normalize("  How many  stations are in Florida?! ") == "how many stations are in florida"


def test_golden_map_includes_aliases_and_follow_ups(golden: dict[str, GoldenEntry]) -> None:
    assert normalize("How many stations are in Florida?") in golden
    assert normalize("How many weather stations are there in Florida?") in golden  # alias
    follow = golden[normalize("And the lowest?")]
    assert follow.requires_history
    assert "asc" in follow.sql
    assert not golden[normalize("How many stations are in Florida?")].requires_history


def test_golden_map_handles_missing_dir(tmp_path: Path) -> None:
    assert load_golden_map(tmp_path) == {}


def test_fake_planner_routes(golden: dict[str, GoldenEntry]) -> None:
    p = FakePlanner(golden=golden)
    assert p.route("How many stations are in Florida?", [])[0].route == "analyst"
    r, usage = p.route("Delete all rows from fct_station_day", [])
    assert (r.route, r.status) == ("respond", "refused")
    assert usage.llm_calls == 1 and usage.cost_usd == 0
    assert p.route("Hello", [])[0].status == "answered"
    assert p.route("What is the meaning of weather?", [])[0].status == "clarify"


def test_fake_planner_follow_up_needs_history(golden: dict[str, GoldenEntry]) -> None:
    p = FakePlanner(golden=golden)
    assert p.plan("And the lowest?", "", [], None)[0].action == "clarify"
    plan, _ = p.plan("And the lowest?", "", [Turn("user", "earlier"), Turn("assistant", "x")], None)
    assert plan.action == "sql" and "asc" in plan.sql


def test_fake_planner_cannot_repair(golden: dict[str, GoldenEntry]) -> None:
    p = FakePlanner(golden=golden)
    plan, _ = p.plan("How many stations are in Florida?", "", [], feedback="guard rejected the query")
    assert plan.action == "refuse"


def test_fake_planner_synthesize_and_verify(golden: dict[str, GoldenEntry]) -> None:
    p = FakePlanner(golden=golden)
    res = QueryResult(columns=["stations"], rows=[[2338]], row_count=1, truncated=False, elapsed_ms=1.0)
    text = "".join(p.synthesize("q", "sql", res, [], []))
    assert "stations=2338" in text and "1 row returned" in text
    empty = QueryResult(columns=["x"], rows=[], row_count=0, truncated=False, elapsed_ms=0.5)
    assert "no rows" in "".join(p.synthesize("q", "sql", empty, [], []))
    v, usage = p.verify("q", text, res)
    assert v.supported and usage.llm_calls == 1


def test_fake_planner_simulated_delay(golden: dict[str, GoldenEntry]) -> None:
    import time

    p = FakePlanner(golden=golden, delay_ms=20)
    t0 = time.perf_counter()
    p.route("Hello", [])
    assert time.perf_counter() - t0 >= 0.015


def test_planner_factory(settings: Settings) -> None:
    assert isinstance(planner_from_settings(settings), FakePlanner)
    assert planner_from_settings(settings, golden={}).name == "fake"
    with pytest.raises(ValueError, match="cannot serve"):
        LangChainPlanner(settings)


def test_langchain_planner_cost_and_usage_parsing(settings: Settings) -> None:
    class Msg:
        usage_metadata = {"input_tokens": 1_000_000, "output_tokens": 200_000}

    planner = LangChainPlanner.__new__(LangChainPlanner)
    planner.settings = settings.model_copy(update={"price_input_per_mtok": 1.0, "price_output_per_mtok": 5.0})
    planner._usage = Usage()
    usage = planner._usage_from(Msg())
    assert usage.input_tokens == 1_000_000 and usage.output_tokens == 200_000
    assert usage.cost_usd == pytest.approx(2.0)
    assert planner.last_usage() == usage
    assert planner._usage_from(object()).llm_calls == 1  # no metadata: counted, zero tokens


def test_transient_detection() -> None:
    class RateLimitError(Exception):
        pass

    assert _is_transient(RateLimitError("slow down"))
    assert _is_transient(RuntimeError("HTTP 529 overloaded"))
    assert _is_transient(TimeoutError())
    assert not _is_transient(ValueError("bad schema"))


def test_formatters() -> None:
    assert _format_history([]) == "(no prior turns)"
    assert _format_history([Turn("user", "a" * 600)]).count("a") == 500
    res = QueryResult(columns=["a", "b"], rows=[[1, None]] * 60, row_count=60, truncated=True, elapsed_ms=1)
    text = _format_result(res, max_rows=50)
    assert "a | b" in text and "10 more rows" in text and "truncated" in text


def test_usage_add() -> None:
    total = Usage(input_tokens=1, output_tokens=2, llm_calls=1, cost_usd=0.1).add(
        Usage(input_tokens=3, output_tokens=4, llm_calls=1, cost_usd=0.2)
    )
    assert total == Usage(input_tokens=4, output_tokens=6, llm_calls=2, cost_usd=0.3)
