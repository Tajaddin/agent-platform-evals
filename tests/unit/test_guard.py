from __future__ import annotations

import pytest

from analyst_agent.guard import check_sql, date_bounded_references, is_date_column, plan_estimate

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


FCT, DIM = "main_marts.fct_station_day", "main_marts.dim_station"
DATES = {FCT: {"obs_date"}, DIM: {"valid_from", "valid_to"}}


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("select * from fct_station_day where obs_date = date '2026-10-10'", [(FCT, True)]),
        ("select * from fct_station_day where date '2026-10-10' <= obs_date and tmax_c > 0", [(FCT, True)]),
        (
            "select * from fct_station_day f where f.obs_date in (date '2026-10-10', date '2026-10-11')",
            [(FCT, True)],
        ),
        (
            "select * from fct_station_day where (obs_date between date '2026-10-10' and date '2026-10-11')",
            [(FCT, True)],
        ),
        ("select * from dim_station where valid_from < now()", [(DIM, True)]),
        ("select * from fct_station_day where state = 'FL'", [(FCT, False)]),
        ("select * from fct_station_day where obs_date is not null", [(FCT, False)]),
        ("select * from fct_station_day where obs_date = date '2026-10-10' or tmax_c > 0", [(FCT, False)]),
        ("select * from fct_station_day where obs_date >= last_loaded_at", [(FCT, False)]),
        (
            "select * from fct_station_day where obs_date = (select max(obs_date) from fct_station_day)",
            [(FCT, False), (FCT, False)],
        ),
        (
            "select * from fct_station_day a, fct_station_day b where a.obs_date = date '2026-10-10'",
            [(FCT, True), (FCT, False)],
        ),
        (
            "select * from fct_station_day f join dim_station d on d.station_id = f.station_id "
            "and f.obs_date = date '2026-10-10'",
            [(FCT, True), (DIM, False)],
        ),
        (
            "with x as (select * from fct_station_day where obs_date = date '2026-10-10') "
            "select * from x a, x b",
            [(FCT, True)],
        ),
    ],
)
def test_date_bounded_references(sql: str, expected: list[tuple[str, bool]]) -> None:
    assert sorted(date_bounded_references(sql, SCHEMAS, TABLES, DATES)) == sorted(expected)


def test_date_columns() -> None:
    assert all(map(is_date_column, ["obs_date", "first_obs_date", "valid_from", "valid_to"]))
    assert not any(map(is_date_column, ["last_loaded_at", "state", "tmax_c"]))


def _scan(table: str, rows: int, filters: str | list[str] | None = None) -> dict:
    info = {"Table": f"demo.main_marts.{table}", "Estimated Cardinality": str(rows)}
    if filters is not None:
        info["Filters"] = filters
    return {"name": "SEQ_SCAN", "children": [], "extra_info": info}


def test_plan_estimate_multiplies_pairwise_joins_and_lists_scan_filters() -> None:
    plan = [
        {
            "name": "UNGROUPED_AGGREGATE",
            "extra_info": {},
            "children": [
                {
                    "name": "CROSS_PRODUCT",
                    "extra_info": {},
                    "children": [
                        _scan("fct_station_day", 1_000),
                        _scan("fct_station_day", 200, ["obs_date='2026-10-10'::DATE", "tmax_c>0"]),
                    ],
                }
            ],
        }
    ]
    est = plan_estimate(plan)
    assert (est.pairs, est.rows) == (200_000, 200_000)
    assert est.filtered_scans(FCT, {"obs_date"}) == 1
    assert est.filtered_scans(FCT, {"date"}) == 0  # whole names only
    assert est.filtered_scans(DIM, {"obs_date"}) == 0


def test_plan_estimate_counts_a_one_row_aggregate_as_one_row() -> None:
    total = {"name": "UNGROUPED_AGGREGATE", "extra_info": {}, "children": [_scan("dim_station", 5_000)]}
    plan = [{"name": "CROSS_PRODUCT", "extra_info": {}, "children": [_scan("dim_station", 5_000), total]}]
    assert plan_estimate(plan).pairs == 5_000
    hash_join = [
        {
            "name": "HASH_JOIN",
            "extra_info": {"Estimated Cardinality": "40"},
            "children": [_scan("dim_station", 5_000), _scan("fct_station_day", 70)],
        }
    ]
    assert (plan_estimate(hash_join).pairs, plan_estimate(hash_join).rows) == (0, 5_000)
