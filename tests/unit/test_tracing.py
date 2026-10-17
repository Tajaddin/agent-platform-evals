from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from analyst_agent import tracing


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    saved = tracing._provider
    tracing._provider = None
    exp = InMemorySpanExporter()
    tracing.configure_tracing(exporter=exp, service_name="test-agent")
    yield exp
    tracing._provider = saved


def test_spans_are_exported_with_attributes(exporter: InMemorySpanExporter) -> None:
    with tracing.span("agent.turn", thread_id="t1") as s:
        s.set_attribute("rows", 3)
        with tracing.span("agent.execute"):
            pass
    tracing.flush_tracing()
    names = [s.name for s in exporter.get_finished_spans()]
    assert names == ["agent.execute", "agent.turn"]
    turn = exporter.get_finished_spans()[1]
    assert turn.attributes is not None
    assert turn.attributes["thread_id"] == "t1" and turn.attributes["rows"] == 3
    assert turn.resource.attributes["service.name"] == "test-agent"


def test_configure_is_idempotent(exporter: InMemorySpanExporter) -> None:
    assert tracing.configure_tracing(exporter=InMemorySpanExporter()) is tracing._provider


def test_no_endpoint_means_no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = tracing._provider
    tracing._provider = None
    try:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert tracing.configure_tracing() is None
        with tracing.span("noop", k="v"):  # still works as a no-op
            pass
        tracing.flush_tracing()
    finally:
        tracing._provider = saved
