from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from analyst_agent.warehouse import Warehouse, _jsonable


@pytest.fixture
def wh(warehouse_path: Path) -> Warehouse:
    return Warehouse(warehouse_path, ["main_marts", "marts"], row_limit=200, timeout_seconds=20.0)


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Warehouse(tmp_path / "nope.duckdb", ["main_marts"], 200, 1.0)


def test_tables_and_schema_prompt(wh: Warehouse) -> None:
    names = {t.name for t in wh.tables()}
    assert names == {
        "fct_station_day",
        "dim_station",
        "mart_daily_state_summary",
        "mart_florida_weather",
        "mart_station_coverage",
    }
    prompt = wh.schema_prompt()
    for n in names:
        assert f"main_marts.{n}" in prompt
    assert "tmax_c double" in prompt
    assert "rows" in prompt


def test_guard_requires_approval_for_a_full_scan_without_date_filter(wh: Warehouse) -> None:
    d = wh.guard("select count(*) from main_marts.fct_station_day", approval_row_threshold=100_000)
    assert d.allowed and d.needs_approval
    assert d.estimated_rows >= 232_116
    assert "approval" in d.reason


def test_guard_passes_filtered_and_small_queries(wh: Warehouse) -> None:
    d = wh.guard(
        "select count(*) from main_marts.fct_station_day where obs_date = date '2026-10-10'", 100_000
    )
    assert d.allowed and not d.needs_approval
    d2 = wh.guard("select count(*) from main_marts.mart_station_coverage", 100_000)
    assert d2.allowed and not d2.needs_approval


def test_guard_rejects_and_explains(wh: Warehouse) -> None:
    d = wh.guard("delete from main_marts.dim_station", 100_000)
    assert not d.allowed
    assert "SELECT" in d.reason


def test_run_caps_rows_and_marks_truncation(wh: Warehouse) -> None:
    r = wh.run("select station_id from main_marts.dim_station")
    assert r.row_count == 200
    assert r.truncated
    assert r.columns == ["station_id"]
    assert r.elapsed_ms >= 0


def test_run_returns_json_friendly_values(wh: Warehouse) -> None:
    r = wh.run("select obs_date, tmax_c from main_marts.fct_station_day where tmax_c is not null limit 1")
    assert isinstance(r.rows[0][0], str) and r.rows[0][0].startswith("2026-10-")
    assert isinstance(r.rows[0][1], float)


def test_run_times_out(warehouse_path: Path) -> None:
    slow = Warehouse(warehouse_path, ["main_marts"], row_limit=10, timeout_seconds=0.05)
    heavy = (
        "select count(*) from main_marts.fct_station_day a, main_marts.fct_station_day b "
        "where a.tmax_c + b.tmin_c > 1000"
    )
    with pytest.raises(duckdb.Error):
        slow.run(heavy)


def test_jsonable() -> None:
    import datetime as dt
    import decimal

    assert _jsonable(dt.date(2026, 10, 3)) == "2026-10-03"
    assert _jsonable(decimal.Decimal("1.5")) == "1.5"
    assert _jsonable(3) == 3
    assert _jsonable(None) is None
