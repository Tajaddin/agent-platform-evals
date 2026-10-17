"""OpenTelemetry traces. Enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise spans are no-ops.

Langfuse accepts OTLP too: point the endpoint at your Langfuse host's /api/public/otel and set the
basic-auth header in OTEL_EXPORTER_OTLP_HEADERS, and every agent span lands there as a trace.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

_provider: TracerProvider | None = None


def configure_tracing(
    exporter: SpanExporter | None = None, service_name: str | None = None
) -> TracerProvider | None:
    global _provider  # noqa: PLW0603  (one provider per process)
    if _provider is not None:
        return _provider
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if exporter is None:
        if not endpoint:
            return None
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: PLC0415

        exporter = OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces", timeout=10)
    name = service_name or os.environ.get("OTEL_SERVICE_NAME", "analyst-agent")
    _provider = TracerProvider(resource=Resource.create({SERVICE_NAME: name}))
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    return _provider


def flush_tracing(timeout_ms: int = 5000) -> None:
    if _provider is not None:
        _provider.force_flush(timeout_millis=timeout_ms)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    tracer = (_provider or trace.get_tracer_provider()).get_tracer("analyst_agent")
    with tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            s.set_attribute(k, v)
        yield s
