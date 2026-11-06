"""FastAPI surface: chat (JSON and SSE), approvals, thread history, health, readiness, metrics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from analyst_agent.config import Settings, get_settings
from analyst_agent.logging import configure_logging, get_logger
from analyst_agent.schemas import ApprovalRequest, ChatRequest, ChatResponse
from analyst_agent.service import AgentService

log = get_logger("analyst_agent.api")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _key_or_ip(request: Request) -> str:
    return request.headers.get("X-API-Key") or get_remote_address(request)


def thread_owner(api_key: str) -> str:
    """The namespace of one API key's threads: a hash, so the key itself is never stored in a checkpoint."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def create_app(settings: Settings | None = None, service: AgentService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    # Only the chat endpoints are limited; health and metrics must stay reachable for probes.
    limiter = Limiter(key_func=_key_or_ip)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or AgentService(settings)
        log.info("startup", provider=app.state.service.planner.name, warehouse=str(settings.warehouse_path))
        try:
            yield
        finally:
            app.state.service.close()
            log.info("shutdown")

    app = FastAPI(title="analyst-agent", version="0.1.0", lifespan=lifespan, docs_url="/docs")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
        )
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")

    def caller(key: str | None = Security(api_key_header)) -> str:
        """Authenticate and return the caller's thread namespace ("" in open mode, where all share one)."""
        if not settings.api_keys:
            return ""
        if key is None or key not in settings.api_keys:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
        return thread_owner(key)

    def svc(request: Request) -> AgentService:
        return request.app.state.service  # type: ignore[no-any-return]

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz(request: Request) -> JSONResponse:
        checks = svc(request).ready()
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ok" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat", response_model=ChatResponse)
    @limiter.limit(settings.rate_limit)
    async def chat(request: Request, body: ChatRequest, owner: str = Depends(caller)) -> ChatResponse:
        return await run_in_threadpool(svc(request).chat, body.thread_id, body.message, owner=owner)

    @app.post("/v1/chat/stream")
    @limiter.limit(settings.rate_limit)
    async def chat_stream(
        request: Request, body: ChatRequest, owner: str = Depends(caller)
    ) -> EventSourceResponse:
        service = svc(request)

        async def gen() -> AsyncIterator[dict[str, Any]]:
            it = service.stream(body.thread_id, body.message, owner=owner)
            while True:
                item = await run_in_threadpool(next, it, None)
                if item is None:
                    break
                yield {"event": item["event"], "data": json.dumps(item["data"], default=str)}

        return EventSourceResponse(gen())

    @app.post("/v1/threads/{thread_id}/approve", response_model=ChatResponse)
    @limiter.limit(settings.rate_limit)
    async def approve(
        request: Request, thread_id: str, body: ApprovalRequest, owner: str = Depends(caller)
    ) -> ChatResponse:
        try:
            return await run_in_threadpool(svc(request).approve, thread_id, body.approved, owner=owner)
        except LookupError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/threads/{thread_id}")
    def thread(request: Request, thread_id: str, owner: str = Depends(caller)) -> dict[str, Any]:
        return {"thread_id": thread_id, "history": svc(request).thread_history(thread_id, owner=owner)}

    return app
