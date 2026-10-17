"""Typed contracts between the graph nodes, the LLM (structured outputs) and the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SqlPlan(BaseModel):
    """What the analyst proposes to run. Produced by the LLM as a structured output."""

    action: Literal["sql", "clarify", "refuse"] = Field(
        description=(
            "sql: run the query below; clarify: ask the user a question; refuse: the request is out of scope"
        )
    )
    sql: str = Field(default="", description="A single read-only SELECT over the allowed tables, or empty")
    rationale: str = Field(
        default="", description="One or two sentences on how the query answers the question"
    )
    message: str = Field(default="", description="For clarify/refuse: what to say to the user")


class Verification(BaseModel):
    """The verifier's judgement of a draft answer against the query result."""

    supported: bool = Field(description="Every number and claim in the draft is present in the result rows")
    issues: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float


class GuardDecision(BaseModel):
    allowed: bool
    sql: str
    reason: str = ""
    estimated_rows: int = 0
    needs_approval: bool = False


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    message: str = Field(min_length=1, max_length=4000)


class ApprovalRequest(BaseModel):
    approved: bool


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["answered", "clarify", "refused", "approval_required", "error"]
    answer: str = ""
    sql: str = ""
    result: QueryResult | None = None
    verification: Verification | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    approval: GuardDecision | None = None
