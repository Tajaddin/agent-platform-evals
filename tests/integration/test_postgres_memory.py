"""Conversation memory in Postgres: needs DATABASE_URL (a service container in CI, compose locally)."""

from __future__ import annotations

import os
import uuid

import pytest

from analyst_agent.config import Settings
from analyst_agent.llm import FakePlanner, GoldenEntry
from analyst_agent.service import AgentService

pytestmark = pytest.mark.postgres

FIRST = "Which state had the highest average maximum temperature on 2026-10-12?"
FOLLOW = "And the lowest?"


@pytest.fixture
def pg_settings(settings: Settings) -> Settings:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    return settings.model_copy(update={"database_url": url})


def test_memory_survives_a_process_restart(pg_settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    thread = f"pg-{uuid.uuid4().hex[:8]}"
    first = AgentService(pg_settings, planner=FakePlanner(golden=golden))
    try:
        assert first.chat(thread, FIRST).status == "answered"
        assert first.ready() == {"warehouse": True, "memory": True}
    finally:
        first.close()

    second = AgentService(pg_settings, planner=FakePlanner(golden=golden))  # a "new replica"
    try:
        assert len(second.thread_history(thread)) == 2
        follow = second.chat(thread, FOLLOW)
        assert follow.status == "answered"
        assert follow.result is not None and follow.result.rows[0][0] == "MT"
    finally:
        second.close()


def test_pending_approval_survives_a_restart(pg_settings: Settings, golden: dict[str, GoldenEntry]) -> None:
    thread = f"pg-{uuid.uuid4().hex[:8]}"
    first = AgentService(pg_settings, planner=FakePlanner(golden=golden))
    try:
        assert (
            first.chat(thread, "How many station-day rows are there in total?").status == "approval_required"
        )
    finally:
        first.close()
    second = AgentService(pg_settings, planner=FakePlanner(golden=golden))
    try:
        assert second.approve(thread, True).status == "answered"
    finally:
        second.close()
