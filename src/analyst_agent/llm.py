"""Planners: the model-facing side of the agent.

The graph never calls a model directly; it calls a Planner. `LangChainPlanner` wraps Anthropic or Bedrock
with structured outputs, retries and token accounting. `FakePlanner` is deterministic and free: it answers
questions from the golden set with their gold SQL, so tests and CI exercise the whole pipeline without a key.
Nothing produced by the fake planner is a measurement of model quality.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from analyst_agent.config import Settings
from analyst_agent.schemas import QueryResult, SqlPlan, Usage, Verification


class Route(BaseModel):
    """The supervisor's decision: hand the question to the SQL analyst, or respond directly."""

    route: Literal["analyst", "respond"] = Field(description="analyst when the warehouse must be queried")
    message: str = Field(
        default="", description="For respond: the reply (clarification, refusal, or a direct answer)"
    )
    status: Literal["clarify", "refused", "answered"] = Field(default="clarify")


@dataclass
class Turn:
    role: str
    content: str


class Planner(Protocol):
    name: str

    def route(self, question: str, history: list[Turn]) -> tuple[Route, Usage]: ...
    def plan(
        self, question: str, schema: str, history: list[Turn], feedback: str | None
    ) -> tuple[SqlPlan, Usage]: ...
    def synthesize(
        self, question: str, sql: str, result: QueryResult, issues: list[str], history: list[Turn]
    ) -> Iterator[str]: ...
    def verify(self, question: str, answer: str, result: QueryResult) -> tuple[Verification, Usage]: ...
    def last_usage(self) -> Usage: ...


# ----------------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------------

SUPERVISOR_SYSTEM = """You are the supervisor of an analytics assistant for a weather data warehouse.
Decide whether the user's message needs a database query (route=analyst) or can be answered without one
(route=respond: greetings, questions about what data exists, requests to modify data which you refuse,
or questions too ambiguous to query, which you turn into one clarifying question).
The warehouse holds daily US weather station observations (temperatures, precipitation, snow) for
2026-10-03 to 2026-10-14 and station metadata. If a prior turn already established the metric or the
state the user means, keep using it."""

ANALYST_SYSTEM = """You write one DuckDB SQL SELECT that answers the user's question from these tables:

{schema}

Rules:
- Only SELECT; never modify data; never read files.
- Qualify tables with their schema (main_marts.table).
- Temperatures are Celsius. For Florida questions in Fahrenheit use tmax_f/tmin_f from
  mart_florida_weather; otherwise convert with c*9/5+32.
- Always add LIMIT (at most 200). Prefer the smallest table that answers the question.
- For state-level questions use mart_daily_state_summary, or join fct_station_day to dim_station
  (is_current = true).
- Dates are DATE columns; the data covers 2026-10-03 to 2026-10-14. Filter by obs_date whenever the
  question names a day or period.
- If the question cannot be answered from these tables, return action=refuse with a short
  message.
- If it is ambiguous (which state? which day?), return action=clarify with one question.
{feedback}"""

SYNTH_SYSTEM = """You explain query results to a business user. Use only numbers that appear in the result
rows. State the answer first, then one line of context (how many rows, which dates). Round temperatures to
one decimal. If the result is empty, say so plainly. Do not mention SQL."""

VERIFY_SYSTEM = """You check an assistant's answer against the query result it was based on.
supported=true only if every number and factual claim in the answer appears in (or is directly computed from)
the result rows. List each unsupported claim in issues."""


def _format_history(history: list[Turn], limit: int = 6) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{t.role}: {t.content[:500]}" for t in history[-limit:])


def _format_result(result: QueryResult, max_rows: int = 50) -> str:
    rows = result.rows[:max_rows]
    header = " | ".join(result.columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows)
    more = f"\n... {result.row_count - len(rows)} more rows" if result.row_count > len(rows) else ""
    trunc = " (truncated at the row limit)" if result.truncated else ""
    return f"{header}\n{body}{more}\n[{result.row_count} rows{trunc}]"


# ----------------------------------------------------------------------------------------------
# LangChain-backed planner (Anthropic API or Bedrock)
# ----------------------------------------------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    transient = ("ratelimit", "rate_limit", "429", "overloaded", "timeout", "503", "502", "connection")
    return any(k in name or k in text for k in transient)


class LangChainPlanner:
    name = "langchain"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._usage = Usage()
        if settings.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

            self.llm: Any = ChatAnthropic(
                model_name=settings.model,
                max_tokens_to_sample=settings.max_output_tokens,
                temperature=0,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,  # tenacity handles retries so they are visible and bounded
                stop=None,
            )
            self.name = f"anthropic:{settings.model}"
        elif settings.provider == "bedrock":
            from langchain_aws import ChatBedrockConverse  # noqa: PLC0415

            self.llm = ChatBedrockConverse(
                model=settings.bedrock_model,
                region_name=settings.aws_region,
                temperature=0,
                max_tokens=settings.max_output_tokens,
            )
            self.name = f"bedrock:{settings.bedrock_model}"
        else:
            raise ValueError(f"LangChainPlanner cannot serve provider {settings.provider}")

    def last_usage(self) -> Usage:
        return self._usage

    def _cost(self, inp: int, out: int) -> float:
        s = self.settings
        return round(inp / 1e6 * s.price_input_per_mtok + out / 1e6 * s.price_output_per_mtok, 6)

    def _usage_from(self, msg: Any) -> Usage:
        meta = getattr(msg, "usage_metadata", None) or {}
        inp = int(meta.get("input_tokens", 0) or 0)
        out = int(meta.get("output_tokens", 0) or 0)
        u = Usage(input_tokens=inp, output_tokens=out, llm_calls=1, cost_usd=self._cost(inp, out))
        self._usage = u
        return u

    def structured(self, schema: type[BaseModel], system: str, user: str) -> tuple[BaseModel, Usage]:
        runnable = self.llm.with_structured_output(schema, include_raw=True)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.llm_max_retries),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_exception(_is_transient),
        )
        def call() -> Any:
            return runnable.invoke([("system", system), ("human", user)])

        out = call()
        parsed = out.get("parsed")
        if parsed is None:
            raise ValueError(f"model returned no structured output: {out.get('parsing_error')}")
        return parsed, self._usage_from(out.get("raw"))

    def route(self, question: str, history: list[Turn]) -> tuple[Route, Usage]:
        parsed, usage = self.structured(
            Route, SUPERVISOR_SYSTEM, f"Prior turns:\n{_format_history(history)}\n\nUser: {question}"
        )
        assert isinstance(parsed, Route)
        return parsed, usage

    def plan(
        self, question: str, schema: str, history: list[Turn], feedback: str | None
    ) -> tuple[SqlPlan, Usage]:
        fb = f"\nYour previous attempt was rejected: {feedback}\nFix the query." if feedback else ""
        parsed, usage = self.structured(
            SqlPlan,
            ANALYST_SYSTEM.format(schema=schema, feedback=fb),
            f"Prior turns:\n{_format_history(history)}\n\nQuestion: {question}",
        )
        assert isinstance(parsed, SqlPlan)
        return parsed, usage

    def synthesize(
        self, question: str, sql: str, result: QueryResult, issues: list[str], history: list[Turn]
    ) -> Iterator[str]:
        fix = (
            f"\nA reviewer found these problems with your previous draft, fix them: {issues}"
            if issues
            else ""
        )
        user = f"Question: {question}\n\nQuery result:\n{_format_result(result)}{fix}"
        inp = out = 0
        for chunk in self.llm.stream([("system", SYNTH_SYSTEM), ("human", user)]):
            text = (
                chunk.content
                if isinstance(chunk.content, str)
                else "".join(part.get("text", "") for part in chunk.content if isinstance(part, dict))
            )
            meta = getattr(chunk, "usage_metadata", None)
            if meta:
                inp = max(inp, int(meta.get("input_tokens", 0) or 0))
                out = max(out, int(meta.get("output_tokens", 0) or 0))
            if text:
                yield text
        self._usage = Usage(input_tokens=inp, output_tokens=out, llm_calls=1, cost_usd=self._cost(inp, out))

    def verify(self, question: str, answer: str, result: QueryResult) -> tuple[Verification, Usage]:
        parsed, usage = self.structured(
            Verification,
            VERIFY_SYSTEM,
            f"Question: {question}\n\nAnswer: {answer}\n\nQuery result:\n{_format_result(result)}",
        )
        assert isinstance(parsed, Verification)
        return parsed, usage


# ----------------------------------------------------------------------------------------------
# Deterministic planner for tests and CI
# ----------------------------------------------------------------------------------------------


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


@dataclass(frozen=True)
class GoldenEntry:
    sql: str
    requires_history: bool = False  # a follow-up: only answerable when earlier turns are in the thread


CLARIFY = "Which state and which dates do you mean?"
ABOUT = (
    "I answer questions about US daily weather observations (2026-10-03 to 2026-10-14) and station metadata."
)


@dataclass
class FakePlanner:
    """Answers golden questions with their gold SQL; everything else gets a clarification.

    Follow-up entries (requires_history) are only answered when the thread history holds earlier turns,
    which is how the tests prove that memory actually reaches the planner.
    """

    golden: dict[str, GoldenEntry]  # normalised question -> entry
    name: str = "fake"
    delay_ms: float = 0.0  # simulated model latency, for load tests
    _usage: Usage = field(default_factory=Usage)

    def last_usage(self) -> Usage:
        return self._usage

    def _tick(self) -> None:
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        self._usage = Usage(input_tokens=0, output_tokens=0, llm_calls=1, cost_usd=0.0)

    def _lookup(self, question: str, history: list[Turn]) -> str | None:
        entry = self.golden.get(normalize(question))
        if entry is None or (entry.requires_history and not history):
            return None
        return entry.sql

    def route(self, question: str, history: list[Turn]) -> tuple[Route, Usage]:
        self._tick()
        q = normalize(question)
        if self._lookup(question, history) is not None:
            return Route(route="analyst"), self._usage
        if any(w in q.split() for w in ("delete", "drop", "update", "insert", "truncate", "modify")):
            msg = "I can only read the warehouse, not change it."
            return Route(route="respond", status="refused", message=msg), self._usage
        if q in {"hello", "hi", "thanks", "thank you"} or q.startswith("what can you"):
            return Route(route="respond", status="answered", message=ABOUT), self._usage
        return Route(route="respond", status="clarify", message=CLARIFY), self._usage

    def plan(
        self, question: str, schema: str, history: list[Turn], feedback: str | None
    ) -> tuple[SqlPlan, Usage]:
        self._tick()
        sql = self._lookup(question, history)
        if sql is None:
            return SqlPlan(action="clarify", message=CLARIFY), self._usage
        if feedback:  # the fake planner cannot repair; surface the rejection honestly
            return SqlPlan(action="refuse", message=f"The query was rejected: {feedback}"), self._usage
        return SqlPlan(action="sql", sql=sql, rationale="gold SQL from the golden set"), self._usage

    def synthesize(
        self, question: str, sql: str, result: QueryResult, issues: list[str], history: list[Turn]
    ) -> Iterator[str]:
        self._tick()
        if result.row_count == 0:
            yield "The query returned no rows."
            return
        first = ", ".join(f"{c}={v}" for c, v in zip(result.columns, result.rows[0], strict=False))
        text = f"Result: {first}. ({result.row_count} row{'s' if result.row_count != 1 else ''} returned.)"
        for word in text.split(" "):
            yield word + " "

    def verify(self, question: str, answer: str, result: QueryResult) -> tuple[Verification, Usage]:
        self._tick()
        return Verification(supported=True, issues=[]), self._usage


def planner_from_settings(settings: Settings, golden: dict[str, GoldenEntry] | None = None) -> Planner:
    if settings.provider == "fake":
        return FakePlanner(golden=golden if golden is not None else load_golden_map())
    return LangChainPlanner(settings)


GOLDEN_DIR = Path(__file__).resolve().parents[2] / "evals" / "golden"


def load_golden_map(golden_dir: Path = GOLDEN_DIR) -> dict[str, GoldenEntry]:
    """Normalised question -> gold SQL from the golden set and the multi-turn conversations."""
    out: dict[str, GoldenEntry] = {}
    questions = golden_dir / "questions.yaml"
    if questions.exists():
        data = yaml.safe_load(questions.read_text(encoding="utf-8")) or {}
        for item in data.get("questions", []):
            if item.get("gold_sql"):
                entry = GoldenEntry(sql=item["gold_sql"].strip())
                out[normalize(item["question"])] = entry
                for alias in item.get("aliases") or []:
                    out[normalize(alias)] = entry
    conversations = golden_dir / "conversations.yaml"
    if conversations.exists():
        data = yaml.safe_load(conversations.read_text(encoding="utf-8")) or {}
        for conv in data.get("conversations", []):
            for i, turn in enumerate(conv.get("turns", [])):
                if turn.get("gold_sql"):
                    out[normalize(turn["message"])] = GoldenEntry(
                        sql=turn["gold_sql"].strip(),
                        requires_history=bool(turn.get("requires_history", i > 0)),
                    )
    return out
