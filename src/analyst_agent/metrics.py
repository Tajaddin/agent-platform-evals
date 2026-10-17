"""Prometheus metrics for the API: requests, latency, tokens, cost, guard decisions, SQL result sizes."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter("agent_requests_total", "Chat requests by final status", ["status"])
LATENCY = Histogram(
    "agent_request_seconds",
    "End-to-end latency of a chat request",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60),
)
TOKENS = Counter("agent_tokens_total", "LLM tokens by direction", ["direction"])
COST = Counter("agent_cost_usd_total", "Estimated LLM cost in USD")
LLM_CALLS = Counter("agent_llm_calls_total", "LLM calls")
GUARD_DECISIONS = Counter("agent_guard_decisions_total", "Query guard outcomes", ["decision"])
SQL_ROWS = Histogram(
    "agent_sql_result_rows", "Rows returned by executed queries", buckets=(0, 1, 5, 20, 50, 100, 200)
)


def record_request(
    status: str, seconds: float, input_tokens: int, output_tokens: int, llm_calls: int, cost: float
) -> None:
    REQUESTS.labels(status).inc()
    LATENCY.observe(seconds)
    TOKENS.labels("input").inc(input_tokens)
    TOKENS.labels("output").inc(output_tokens)
    LLM_CALLS.inc(llm_calls)
    COST.inc(cost)
