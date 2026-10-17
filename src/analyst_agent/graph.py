"""The LangGraph: supervisor -> analyst -> guard -> (approval) -> execute -> synthesize -> verify.

Patterns on purpose: a supervisor that routes to specialists, a repair loop (guard or execution errors go
back to the analyst once with the error text), a reflection loop (the verifier can send the synthesizer
back once with its issues), a human-in-the-loop interrupt for expensive queries, and per-thread memory
through the checkpointer so follow-up questions see earlier turns.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from analyst_agent import metrics
from analyst_agent.config import Settings
from analyst_agent.llm import Planner, Turn
from analyst_agent.schemas import GuardDecision, QueryResult, SqlPlan, Usage, Verification
from analyst_agent.tracing import span
from analyst_agent.warehouse import Warehouse

MAX_REPAIRS = 2
MAX_REFLECTIONS = 1


def _append(a: list[dict[str, str]] | None, b: list[dict[str, str]] | None) -> list[dict[str, str]]:
    return (a or []) + (b or [])


class AgentState(TypedDict, total=False):
    # conversation memory, persisted per thread by the checkpointer
    history: Annotated[list[dict[str, str]], _append]
    # current turn (reset by new_turn_input)
    question: str
    route: str
    plan: dict[str, Any]
    guard: dict[str, Any]
    result: dict[str, Any]
    answer: str
    verification: dict[str, Any]
    status: str
    error: str
    repairs: int
    reflections: int
    usage: dict[str, Any]


def _history(state: AgentState) -> list[Turn]:
    return [Turn(role=h["role"], content=h["content"]) for h in state.get("history", [])]


def _usage(state: AgentState, add: Usage) -> dict[str, Any]:
    """Nodes run one at a time, so per-turn usage is summed explicitly instead of through a reducer."""
    return Usage(**(state.get("usage") or {})).add(add).model_dump()


def _exchange(question: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def build_graph(
    planner: Planner,
    warehouse: Warehouse,
    settings: Settings,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def supervisor(state: AgentState) -> dict[str, Any]:
        with span("agent.supervisor", question_chars=len(state["question"])):
            route, usage = planner.route(state["question"], _history(state))
        if route.route == "analyst":
            return {"route": "analyst", "usage": _usage(state, usage), "repairs": 0, "reflections": 0}
        return {
            "route": "respond",
            "answer": route.message,
            "status": route.status,
            "usage": _usage(state, usage),
            "history": _exchange(state["question"], route.message),
        }

    def analyst(state: AgentState) -> dict[str, Any]:
        feedback = state.get("error") or None
        with span("agent.analyst", repair=state.get("repairs", 0)):
            plan, usage = planner.plan(
                state["question"], warehouse.schema_prompt(), _history(state), feedback
            )
        out: dict[str, Any] = {"plan": plan.model_dump(), "usage": _usage(state, usage), "error": ""}
        if plan.action != "sql":
            out["status"] = "clarify" if plan.action == "clarify" else "refused"
            out["answer"] = plan.message
            out["history"] = _exchange(state["question"], plan.message)
        return out

    def guard(state: AgentState) -> dict[str, Any]:
        plan = SqlPlan(**state["plan"])
        with span("agent.guard"):
            decision = warehouse.guard(plan.sql, settings.approval_row_threshold)
        metrics.GUARD_DECISIONS.labels(
            "approval" if decision.needs_approval else ("allowed" if decision.allowed else "rejected")
        ).inc()
        out: dict[str, Any] = {"guard": decision.model_dump()}
        if not decision.allowed:
            out["error"] = f"guard rejected the query: {decision.reason}"
            out["repairs"] = state.get("repairs", 0) + 1
        return out

    def approval(state: AgentState) -> dict[str, Any]:
        decision = GuardDecision(**state["guard"])
        approved = interrupt(
            {
                "type": "approval_required",
                "sql": decision.sql,
                "estimated_rows": decision.estimated_rows,
                "reason": decision.reason,
            }
        )
        if approved is True:
            return {"guard": {**state["guard"], "needs_approval": False, "reason": "approved by a human"}}
        msg = "The query was not approved, so it did not run."
        return {"status": "refused", "answer": msg, "history": _exchange(state["question"], msg)}

    def execute(state: AgentState) -> dict[str, Any]:
        sql = state["guard"]["sql"]
        try:
            with span("agent.execute") as s:
                result = warehouse.run(sql)
                s.set_attribute("rows", result.row_count)
                s.set_attribute("elapsed_ms", result.elapsed_ms)
            metrics.SQL_ROWS.observe(result.row_count)
            return {"result": result.model_dump(), "error": ""}
        except Exception as exc:  # the analyst gets one chance to repair a failing query
            return {"error": f"query failed: {str(exc)[:300]}", "repairs": state.get("repairs", 0) + 1}

    def synthesize(state: AgentState) -> dict[str, Any]:
        result = QueryResult(**state["result"])
        issues = list((state.get("verification") or {}).get("issues", []))
        writer = get_stream_writer()
        parts: list[str] = []
        with span("agent.synthesize", reflection=state.get("reflections", 0)):
            for chunk in planner.synthesize(
                state["question"], state["guard"]["sql"], result, issues, _history(state)
            ):
                parts.append(chunk)
                writer({"type": "token", "text": chunk})
        answer = "".join(parts).strip()
        return {"answer": answer, "usage": _usage(state, planner.last_usage())}

    def verify(state: AgentState) -> dict[str, Any]:
        result = QueryResult(**state["result"])
        with span("agent.verify"):
            verification, usage = planner.verify(state["question"], state["answer"], result)
        out: dict[str, Any] = {"verification": verification.model_dump(), "usage": _usage(state, usage)}
        if not verification.supported and state.get("reflections", 0) < MAX_REFLECTIONS:
            out["reflections"] = state.get("reflections", 0) + 1
        else:
            out["status"] = "answered"
            out["history"] = _exchange(state["question"], state["answer"])
        return out

    def fail(state: AgentState) -> dict[str, Any]:
        msg = f"I could not produce a safe query for that question ({state.get('error', 'unknown error')})."
        return {"status": "error", "answer": msg, "history": _exchange(state["question"], msg)}

    # routing functions
    def after_supervisor(state: AgentState) -> Literal["analyst", "end"]:
        return "analyst" if state.get("route") == "analyst" else "end"

    def after_analyst(state: AgentState) -> Literal["guard", "end"]:
        return "guard" if SqlPlan(**state["plan"]).action == "sql" else "end"

    def after_guard(state: AgentState) -> Literal["approval", "execute", "analyst", "fail"]:
        g = GuardDecision(**state["guard"])
        if g.allowed:
            return "approval" if g.needs_approval else "execute"
        return "analyst" if state.get("repairs", 0) < MAX_REPAIRS else "fail"

    def after_approval(state: AgentState) -> Literal["execute", "end"]:
        return "end" if state.get("status") == "refused" else "execute"

    def after_execute(state: AgentState) -> Literal["synthesize", "analyst", "fail"]:
        if state.get("result") and not state.get("error"):
            return "synthesize"
        return "analyst" if state.get("repairs", 0) < MAX_REPAIRS else "fail"

    def after_verify(state: AgentState) -> Literal["synthesize", "end"]:
        v = Verification(**state["verification"])
        return "synthesize" if (not v.supported and state.get("status") != "answered") else "end"

    g: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)
    for name, fn in [
        ("supervisor", supervisor),
        ("analyst", analyst),
        ("guard", guard),
        ("approval", approval),
        ("execute", execute),
        ("synthesize", synthesize),
        ("verify", verify),
        ("fail", fail),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", after_supervisor, {"analyst": "analyst", "end": END})
    g.add_conditional_edges("analyst", after_analyst, {"guard": "guard", "end": END})
    g.add_conditional_edges(
        "guard",
        after_guard,
        {"approval": "approval", "execute": "execute", "analyst": "analyst", "fail": "fail"},
    )
    g.add_conditional_edges("approval", after_approval, {"execute": "execute", "end": END})
    g.add_conditional_edges(
        "execute", after_execute, {"synthesize": "synthesize", "analyst": "analyst", "fail": "fail"}
    )
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", after_verify, {"synthesize": "synthesize", "end": END})
    g.add_edge("fail", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def new_turn_input(question: str) -> dict[str, Any]:
    """State reset for a new user turn; `history` and the checkpoint carry the memory."""
    return {
        "question": question,
        "route": "",
        "plan": {},
        "guard": {},
        "result": {},
        "answer": "",
        "verification": {},
        "status": "",
        "error": "",
        "repairs": 0,
        "reflections": 0,
        "usage": Usage().model_dump(),
    }
