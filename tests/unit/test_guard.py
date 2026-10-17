from __future__ import annotations

import pytest

from analyst_agent.guard import check_sql, estimated_rows_from_explain, has_date_filter

SCHEMAS = ["main_marts", "marts"]
TABLES = {
    "main_marts": {"fct_station_day", "dim_station", "mart_daily_state_summary"},
    "raw": {"ghcn_observations"},
}


def guard(sql: str, limit: int = 200):
    return check_sql(sql, SCHEMAS, TABLES, limit)


def test_select_gets_a_limit_and_resolved_tables() -> None:
    out = guard("select state, count(*) from main_marts.dim_station group by state")
    assert out.allowed
    assert out.sql.lower().endswith("limit 200")
    assert out.tables == ("main_marts.dim_station",)


def test_unqualified_table_resolves_against_allowed_schemas() -> None:
    out = guard("select * from dim_station")
    assert out.allowed
    assert out.tables == ("main_marts.dim_station",)


def test_limit_is_lowered_but_small_limit_is_kept() -> None:
    assert "LIMIT 200" in guard("select * from dim_station limit 5000").sql
    assert "LIMIT 10" in guard("select * from dim_station limit 10").sql


def test_cte_and_union_are_allowed() -> None:
    out = guard(
        "with fl as (select * from main_marts.dim_station where state = 'FL') "
        "select station_id from fl union all select station_id from main_marts.dim_station where state = 'GA'"
    )
    assert out.allowed, out.reason
    assert out.tables == ("main_marts.dim_station",)


@pytest.mark.parametrize(
    ("sql", "reason_part"),
    [
        ("", "empty"),
        ("insert into main_marts.dim_station values (1)", "only SELECT"),
        ("update main_marts.dim_station set state = 'X'", "only SELECT"),
        ("delete from main_marts.dim_station", "only SELECT"),
        ("drop table main_marts.dim_station", "only SELECT"),
        ("copy main_marts.dim_station to '/tmp/x.csv'", "only SELECT"),
        ("attach 'other.duckdb' as o", "only SELECT"),
        ("pragma database_list", "only SELECT"),
        ("select 1; select 2", "exactly one statement"),
        ("select * from read_parquet('s3://bucket/x.parquet')", "read_parquet"),
        ("select * from glob('*.csv')", "glob"),
        ("select * from main_marts.secret_table", "not in the allowlist"),
        ("select * from raw.ghcn_observations", "not in the allowlist"),
        ("select * from information_schema.tables", "not in the allowlist"),
        ("select 1", "reads no table"),
        (
            "select * from dim_station where station_id in (select x from raw.ghcn_observations)",
            "not in the allowlist",
        ),
        ("select * frm dim_station", "could not parse"),
    ],
)
def test_rejections(sql: str, reason_part: str) -> None:
    out = guard(sql)
    assert not out.allowed
    assert reason_part.lower() in out.reason.lower(), out.reason


def test_banned_word_inside_a_select_is_fine() -> None:
    # the word "delete" appears in a string literal, which is legal in a SELECT
    out = guard("select 'delete me' as note, station_id from dim_station")
    assert out.allowed


def test_explain_estimates_both_plan_formats() -> None:
    assert estimated_rows_from_explain("│ ~6,927 rows │ ... │ ~232,116 rows │") == 232116
    assert estimated_rows_from_explain("SEQ_SCAN EC: 1,000 PROJECTION EC: 42") == 1000
    assert estimated_rows_from_explain("~1 row") == 1
    assert estimated_rows_from_explain("no estimates here") == 0


def test_has_date_filter() -> None:
    assert has_date_filter("select * from t where obs_date = date '2026-10-10'")
    assert has_date_filter("select * from t where valid_from < now()")
    assert not has_date_filter("select * from t where state = 'FL'")
    assert not has_date_filter("select obs_date from t")
    assert not has_date_filter("not sql at all (")
