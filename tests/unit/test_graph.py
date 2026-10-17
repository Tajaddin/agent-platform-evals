"""The graph end to end with the fake planner: every route, loop and the human-in-the-loop interrupt."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from prometheus_client import generate_latest

from analyst_agent.config import Settings
from analyst_agent.graph import MAX_REPAIRS
from analyst_agent.llm import FakePlanner, GoldenEntry, Turn
from analyst_agent.schemas import QueryResult, SqlPlan, Usage, Verification
from analyst_agent.service import AgentService

FL = "How many stations are in Florida?"
TOTAL = "How many station-day rows are there in total?"
FIRST = "Which state had the highest average maximum temperature on 2026-10-12?"
FOLLOW = "And the lowest?"


def test_answered_path(service: AgentService) -> None:
    resp = service.chat("t1", FL)
    assert resp.status == "answered"
    assert "dim_station" in resp.sql and resp.sql.lower().endswith("limit 200")
    assert resp.result is not None and resp.result.rows == [[2338]]
    assert resp.verification is not None and resp.verification.supported
    assert resp.usage.llm_calls == 4  # route, plan, synthesize, verify
    assert "2338" in resp.answer
    assert resp.latency_ms > 0
    assert service.thread_history("t1") == [
        {"role": "user", "content": FL},
        {"role": "assistant", "content": resp.answer},
    ]


def test_usage_is_per_turn_not_cumulative(service: AgentService) -> None:
    service.chat("t2", FL)
    second = service.chat("t2", FL)
    assert second.usage.llm_calls == 4
    assert len(service.thread_history("t2")) == 4


def test_clarify_and_refuse_paths(service: AgentService) -> None:
    clarify = service.chat("t3", "Show me the weather")
    assert clarify.status == "clarify" and clarify.sql == "" and clarify.result is None
    assert clarify.usage.llm_calls == 1  # the supervisor answered directly
    refused = service.chat("t4", "Delete all rows from fct_station_day")
    assert refused.status == "refused"
    direct = service.chat("t5", "Hello")
    assert direct.status == "answered" and direct.result is None


def test_approval_interrupt_then_approve(service: AgentService) -> None:
    first = service.chat("t6", TOTAL)
    assert first.status == "approval_required"
    assert first.approval is not None and first.approval.needs_approval
    assert first.approval.estimated_rows > 100_000
    assert "fct_station_day" in first.sql
    assert service.thread_history("t6") == []  # nothing committed to memory yet
    final = service.approve("t6", True)
    assert final.status == "answered"
    assert final.result is not None and final.result.rows == [[232116]]
    assert final.usage.llm_calls == 4
    with pytest.raises(LookupError):
        service.approve("t6", True)


def test_approval_declined(service: AgentService) -> None:
    assert service.chat("t7", TOTAL).status == "approval_required"
    final = service.approve("t7", False)
    assert final.status == "refused"
    assert "not approved" in final.answer
    assert len(service.thread_history("t7")) == 2


def test_new_message_while_approval_pending_starts_over(service: AgentService) -> None:
    assert service.chat("t8", TOTAL).status == "approval_required"
    resp = service.chat("t8", FL)
    assert resp.status == "answered"
    with pytest.raises(LookupError):
        service.approve("t8", True)


def test_memory_makes_follow_ups_answerable(service: AgentService) -> None:
    cold = service.chat("t9-cold", FOLLOW)
    assert cold.status == "clarify"
    assert service.chat("t9", FIRST).status == "answered"
    warm = service.chat("t9", FOLLOW)
    assert warm.status == "answered"
    assert warm.result is not None and warm.result.rows[0][0] == "MT"


def test_lower_threshold_setting_changes_approval(settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    strict = settings.model_copy(update={"approval_row_threshold": 1_000})
    svc = AgentService(strict, planner=FakePlanner(golden=golden))
    # dim_station has 78,568 rows and the query has no date filter -> now needs approval
    assert svc.chat("t10", FL).status == "approval_required"


@dataclass
class ScriptedPlanner(FakePlanner):
    """A fake planner whose plan() follows a script, to drive the repair and reflection loops."""

    plans: list[SqlPlan] = field(default_factory=list)
    verifications: list[Verification] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    feedbacks: list[str | None] = field(default_factory=list)

    def plan(
        self, question: str, schema: str, history: list[Turn], feedback: str | None
    ) -> tuple[SqlPlan, Usage]:
        self._tick()
        self.calls.append("plan")
        self.feedbacks.append(feedback)
        return self.plans.pop(0), self._usage

    def synthesize(
        self, question: str, sql: str, result: QueryResult, issues: list[str], history: list[Turn]
    ) -> Iterator[str]:
        self.calls.append(f"synthesize:{len(issues)}")
        yield from super().synthesize(question, sql, result, issues, history)

    def verify(self, question: str, answer: str, result: QueryResult) -> tuple[Verification, Usage]:
        self._tick()
        self.calls.append("verify")
        if self.verifications:
            return self.verifications.pop(0), self._usage
        return Verification(supported=True), self._usage


GOOD = "select count(*) as stations from main_marts.dim_station where is_current and state = 'FL'"


def scripted(
    settings: Settings, golden: dict[str, GoldenEntry], **kw: object
) -> tuple[AgentService, ScriptedPlanner]:
    planner = ScriptedPlanner(golden=golden, **kw)  # type: ignore[arg-type]
    return AgentService(settings, planner=planner), planner


def test_guard_rejection_is_repaired(settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    svc, planner = scripted(
        settings,
        golden,
        plans=[SqlPlan(action="sql", sql="select * from main_marts.secret"), SqlPlan(action="sql", sql=GOOD)],
    )
    resp = svc.chat("r1", FL)
    assert resp.status == "answered"
    assert planner.feedbacks[0] is None
    assert planner.feedbacks[1] is not None and "allowlist" in planner.feedbacks[1]
    assert resp.usage.llm_calls == 5


def test_runtime_error_is_repaired(settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    broken = "select cast(station_name as integer) as x from main_marts.dim_station where is_current limit 1"
    svc, planner = scripted(
        settings, golden, plans=[SqlPlan(action="sql", sql=broken), SqlPlan(action="sql", sql=GOOD)]
    )
    resp = svc.chat("r2", FL)
    assert resp.status == "answered"
    assert planner.feedbacks[1] is not None and planner.feedbacks[1].startswith("query failed")


def test_repairs_are_bounded(settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    bad = [SqlPlan(action="sql", sql="select * from main_marts.secret") for _ in range(MAX_REPAIRS + 2)]
    svc, planner = scripted(settings, golden, plans=bad)
    resp = svc.chat("r3", FL)
    assert resp.status == "error"
    assert "could not produce a safe query" in resp.answer
    assert planner.calls.count("plan") == MAX_REPAIRS
    assert len(svc.thread_history("r3")) == 2


def test_verifier_sends_synthesizer_back_once(settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    svc, planner = scripted(
        settings,
        golden,
        plans=[SqlPlan(action="sql", sql=GOOD)],
        verifications=[
            Verification(supported=False, issues=["made-up number"]),
            Verification(supported=True),
        ],
    )
    resp = svc.chat("r4", FL)
    assert resp.status == "answered"
    assert planner.calls == ["plan", "synthesize:0", "verify", "synthesize:1", "verify"]
    assert resp.verification is not None and resp.verification.supported
    assert resp.usage.llm_calls == 6


def test_persistent_verifier_failure_still_answers(
    settings: Settings, golden: dict[str, GoldenEntry]
) -> None:
    svc, planner = scripted(
        settings,
        golden,
        plans=[SqlPlan(action="sql", sql=GOOD)],
        verifications=[
            Verification(supported=False, issues=["x"]),
            Verification(supported=False, issues=["x"]),
        ],
    )
    resp = svc.chat("r5", FL)
    assert resp.status == "answered"
    assert resp.verification is not None and not resp.verification.supported  # surfaced, not hidden
    assert planner.calls.count("synthesize:1") == 1


def test_stream_emits_nodes_tokens_and_done(service: AgentService) -> None:
    events = list(service.stream("s1", FL))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "node" and kinds[-1] == "done"
    nodes = [e["data"]["node"] for e in events if e["event"] == "node"]
    assert nodes == ["supervisor", "analyst", "guard", "execute", "synthesize", "verify"]
    tokens = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert "2338" in tokens
    assert events[-1]["data"]["status"] == "answered"


def test_stream_surfaces_interrupt_as_done_with_approval(service: AgentService) -> None:
    events = list(service.stream("s2", TOTAL))
    assert events[-1]["data"]["status"] == "approval_required"


def test_metrics_are_recorded(service: AgentService) -> None:
    service.chat("m1", FL)
    text = generate_latest().decode()
    assert 'agent_requests_total{status="answered"}' in text
    assert "agent_request_seconds_bucket" in text
    assert 'agent_guard_decisions_total{decision="allowed"}' in text
    assert "agent_sql_result_rows_bucket" in text


def test_ready_checks(service: AgentService) -> None:
    assert service.ready() == {"warehouse": True, "memory": True}
