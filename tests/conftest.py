from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyst_agent.config import Settings  # noqa: E402
from analyst_agent.llm import FakePlanner, GoldenEntry, load_golden_map  # noqa: E402
from analyst_agent.service import AgentService  # noqa: E402


@pytest.fixture(autouse=True)
def _no_otel_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's .env must not turn on span export during tests."""
    for key in list(os.environ):
        if key.startswith("OTEL_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def warehouse_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The committed demo warehouse, built from data/sample on the fly when the .duckdb file is absent."""
    path = ROOT / "data" / "demo_warehouse.duckdb"
    if path.exists():
        return path
    from build_demo_warehouse import build  # type: ignore[import-not-found]

    out = tmp_path_factory.mktemp("wh") / "demo_warehouse.duckdb"
    build(out)
    return out


@pytest.fixture
def settings(warehouse_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        provider="fake",
        warehouse_path=warehouse_path,
        api_keys=["test-key"],
        rate_limit="1000/minute",
        log_level="WARNING",
    )


@pytest.fixture(scope="session")
def golden() -> dict[str, GoldenEntry]:
    entries = load_golden_map()
    assert entries, "golden set missing"
    return entries


@pytest.fixture
def service(settings: Settings, golden: dict[str, GoldenEntry]) -> Iterator[AgentService]:
    svc = AgentService(settings, planner=FakePlanner(golden=golden))
    yield svc
    svc.close()
