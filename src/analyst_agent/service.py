"""The agent as a service object: owns the planner, warehouse, graph and checkpointer; runs turns."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, StreamMode

from analyst_agent import metrics

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool
from analyst_agent.config import Settings
from analyst_agent.graph import build_graph, new_turn_input
from analyst_agent.llm import Planner, planner_from_settings
from analyst_agent.logging import get_logger
from analyst_agent.schemas import ChatResponse, GuardDecision, QueryResult, Usage, Verification
from analyst_agent.tracing import configure_tracing, span
from analyst_agent.warehouse import Warehouse

log = get_logger("analyst_agent.service")


class AgentService:
    def __init__(self, settings: Settings, planner: Planner | None = None, checkpointer: Any = None) -> None:
        self.settings = settings
        configure_tracing(service_name=settings.service_name)
        self.warehouse = Warehouse(
            settings.warehouse_path,
            settings.allowed_schemas,
            settings.row_limit,
            settings.query_timeout_seconds,
        )
        self.planner = planner or planner_from_settings(settings)
        self._pool: ConnectionPool[Any] | None = None
        if checkpointer is None and settings.database_url:
            checkpointer = self._postgres_checkpointer(settings.database_url)
        self.graph = build_graph(self.planner, self.warehouse, settings, checkpointer)

    def _postgres_checkpointer(self, url: str) -> Any:
        from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415
        from psycopg_pool import ConnectionPool  # noqa: PLC0415

        self._pool = ConnectionPool(
            url, min_size=1, max_size=8, kwargs={"autocommit": True, "prepare_threshold": 0}
        )
        saver = PostgresSaver(self._pool)
        saver.setup()
        return saver

    def close(self) -> None:
        if self._pool is not None:
            with suppress(Exception):
                self._pool.close()

    def ready(self) -> dict[str, bool]:
        ok_wh = True
        try:
            self.warehouse.tables()
        except Exception:
            ok_wh = False
        ok_db = True
        if self._pool is not None:
            try:
                with self._pool.connection() as conn:
                    conn.execute("select 1")
            except Exception:
                ok_db = False
        return {"warehouse": ok_wh, "memory": ok_db}

    # ------------------------------------------------------------------ turns

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    def _response(
        self, thread_id: str, state: dict[str, Any], t0: float, interrupted: dict[str, Any] | None
    ) -> ChatResponse:
        usage = Usage(**(state.get("usage") or {}))
        latency_ms = round((time.time() - t0) * 1000, 1)
        if interrupted is not None:
            status = "approval_required"
            resp = ChatResponse(
                thread_id=thread_id,
                status="approval_required",
                sql=interrupted.get("sql", ""),
                approval=GuardDecision(**state["guard"]) if state.get("guard") else None,
                usage=usage,
                latency_ms=latency_ms,
            )
        else:
            status = state.get("status") or "error"
            result = state.get("result") or None
            verification = state.get("verification") or None
            resp = ChatResponse(
                thread_id=thread_id,
                status=cast(Any, status),
                answer=state.get("answer", ""),
                sql=(state.get("guard") or {}).get("sql", "") if status == "answered" else "",
                result=QueryResult(**result) if result and status == "answered" else None,
                verification=Verification(**verification) if verification else None,
                usage=usage,
                latency_ms=latency_ms,
            )
        metrics.record_request(
            status,
            latency_ms / 1000,
            usage.input_tokens,
            usage.output_tokens,
            usage.llm_calls,
            usage.cost_usd,
        )
        log.info(
            "turn.done",
            thread_id=thread_id,
            status=status,
            latency_ms=latency_ms,
            llm_calls=usage.llm_calls,
            cost_usd=usage.cost_usd,
            rows=(resp.result.row_count if resp.result else None),
        )
        return resp

    def _interrupt_payload(self, thread_id: str) -> dict[str, Any] | None:
        snapshot = self.graph.get_state(self._config(thread_id))
        for task in snapshot.tasks:
            for intr in getattr(task, "interrupts", ()):
                value = getattr(intr, "value", None)
                if isinstance(value, dict):
                    return value
        return None

    def chat(self, thread_id: str, message: str) -> ChatResponse:
        t0 = time.time()
        with span("agent.turn", thread_id=thread_id):
            self.graph.invoke(new_turn_input(message), self._config(thread_id))
            state = self.graph.get_state(self._config(thread_id)).values
            return self._response(thread_id, state, t0, self._interrupt_payload(thread_id))

    def approve(self, thread_id: str, approved: bool) -> ChatResponse:
        t0 = time.time()
        if self._interrupt_payload(thread_id) is None:
            raise LookupError("nothing is waiting for approval on this thread")
        with span("agent.approve", thread_id=thread_id, approved=approved):
            self.graph.invoke(Command(resume=approved), self._config(thread_id))
            state = self.graph.get_state(self._config(thread_id)).values
            return self._response(thread_id, state, t0, self._interrupt_payload(thread_id))

    def stream(self, thread_id: str, message: str) -> Iterator[dict[str, Any]]:
        """Yield events while the graph runs: node starts, synthesizer tokens, then the final response."""
        t0 = time.time()
        q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        modes: list[StreamMode] = ["custom", "updates"]

        def run() -> None:
            try:
                for mode, chunk in self.graph.stream(
                    new_turn_input(message), self._config(thread_id), stream_mode=modes
                ):
                    if mode == "custom" and isinstance(chunk, dict):
                        q.put({"event": "token", "data": chunk})
                    elif mode == "updates" and isinstance(chunk, dict):
                        for node in chunk:
                            if node != "__interrupt__":
                                q.put({"event": "node", "data": {"node": node}})
            except Exception as exc:  # surfaced to the client as an error event
                q.put({"event": "error", "data": {"message": str(exc)[:300]}})
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        state = self.graph.get_state(self._config(thread_id)).values
        final = self._response(thread_id, state, t0, self._interrupt_payload(thread_id))
        yield {"event": "done", "data": final.model_dump()}

    def thread_history(self, thread_id: str) -> list[dict[str, str]]:
        state = self.graph.get_state(self._config(thread_id)).values
        return list(state.get("history", [])) if state else []
