"""The HTTP surface through FastAPI's TestClient with the fake provider."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from analyst_agent.api import create_app
from analyst_agent.config import Settings
from analyst_agent.llm import FakePlanner, GoldenEntry
from analyst_agent.service import AgentService

HEADERS = {"X-API-Key": "test-key"}
FL = "How many stations are in Florida?"
TOTAL = "How many station-day rows are there in total?"


@pytest.fixture
def client(settings: Settings, service: AgentService) -> Iterator[TestClient]:
    with TestClient(create_app(settings, service)) as c:
        yield c


def test_health_ready_metrics_are_open(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["checks"] == {"warehouse": True, "memory": True}
    assert "agent_requests_total" in client.get("/metrics").text


def test_auth_required(client: TestClient) -> None:
    body = {"thread_id": "a1", "message": FL}
    assert client.post("/v1/chat", json=body).status_code == 401
    assert client.post("/v1/chat", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/threads/a1").status_code == 401


def test_validation(client: TestClient) -> None:
    assert (
        client.post("/v1/chat", json={"thread_id": "has space", "message": FL}, headers=HEADERS).status_code
        == 422
    )
    assert (
        client.post("/v1/chat", json={"thread_id": "ok", "message": "x" * 4001}, headers=HEADERS).status_code
        == 422
    )
    assert client.post("/v1/chat", json={"thread_id": "ok"}, headers=HEADERS).status_code == 422


def test_chat_and_thread_history(client: TestClient) -> None:
    r = client.post("/v1/chat", json={"thread_id": "a2", "message": FL}, headers=HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered"
    assert body["result"]["rows"] == [[2338]]
    assert body["usage"]["llm_calls"] == 4
    hist = client.get("/v1/threads/a2", headers=HEADERS).json()
    assert [m["role"] for m in hist["history"]] == ["user", "assistant"]


def test_approval_flow_over_http(client: TestClient) -> None:
    assert client.post("/v1/threads/a3/approve", json={"approved": True}, headers=HEADERS).status_code == 409
    first = client.post("/v1/chat", json={"thread_id": "a3", "message": TOTAL}, headers=HEADERS).json()
    assert first["status"] == "approval_required"
    assert first["approval"]["estimated_rows"] > 100_000
    final = client.post("/v1/threads/a3/approve", json={"approved": True}, headers=HEADERS).json()
    assert final["status"] == "answered" and final["result"]["rows"] == [[232116]]


def _sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):  # sse-starlette emits CRLF
        event, data = "", ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if event:
            events.append((event, json.loads(data)))
    return events


def test_streaming_endpoint(client: TestClient) -> None:
    with client.stream(
        "POST", "/v1/chat/stream", json={"thread_id": "a4", "message": FL}, headers=HEADERS
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _sse("".join(r.iter_text()))
    kinds = [e for e, _ in events]
    assert "node" in kinds and "token" in kinds and kinds[-1] == "done"
    assert events[-1][1]["status"] == "answered"
    assert "2338" in "".join(d["text"] for e, d in events if e == "token")


def test_rate_limit(settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    tight = settings.model_copy(update={"rate_limit": "2/minute"})
    svc = AgentService(tight, planner=FakePlanner(golden=golden))
    with TestClient(create_app(tight, svc)) as c:
        codes = [
            c.post("/v1/chat", json={"thread_id": "rl", "message": "Hello"}, headers=HEADERS).status_code
            for _ in range(3)
        ]
    assert codes == [200, 200, 429]
    assert c.get("/healthz").status_code == 200  # probes are never limited


def test_open_mode_when_no_keys_configured(settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    open_settings = settings.model_copy(update={"api_keys": []})
    svc = AgentService(open_settings, planner=FakePlanner(golden=golden))
    with TestClient(create_app(open_settings, svc)) as c:
        assert c.post("/v1/chat", json={"thread_id": "o1", "message": "Hello"}).status_code == 200
