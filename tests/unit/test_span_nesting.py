"""One chat turn is one trace: every node span is a child of agent.turn, even though LangGraph runs
nodes on its own executor threads."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from analyst_agent import tracing
from analyst_agent.config import Settings
from analyst_agent.llm import FakePlanner, GoldenEntry
from analyst_agent.service import AgentService


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    saved = tracing._provider
    tracing._provider = None
    exp = InMemorySpanExporter()
    tracing.configure_tracing(exporter=exp, service_name="test-agent")
    yield exp
    tracing._provider = saved


def test_node_spans_nest_under_the_turn(
    exporter: InMemorySpanExporter, settings: Settings, golden: dict[str, GoldenEntry]
) -> None:
    svc = AgentService(settings, planner=FakePlanner(golden=golden))
    svc.chat("trace-1", "How many stations are in Florida?")
    tracing.flush_tracing()
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    turn = next(s for s in spans if s.name == "agent.turn")
    children = sorted(s.name for s in spans if s.parent and s.parent.span_id == turn.context.span_id)
    assert children == [
        "agent.analyst",
        "agent.execute",
        "agent.guard",
        "agent.supervisor",
        "agent.synthesize",
        "agent.verify",
    ]
    assert all(by_id[s.parent.span_id].context.trace_id == turn.context.trace_id for s in spans if s.parent)
    execute = next(s for s in spans if s.name == "agent.execute")
    assert execute.attributes is not None and execute.attributes["rows"] == 1
    assert turn.attributes is not None and turn.attributes["thread_id"] == "trace-1"
