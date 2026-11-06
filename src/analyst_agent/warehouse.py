"""Read-only access to the warehouse, schema description for the model, guarded execution."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from analyst_agent.guard import (
    GuardOutcome,
    PlanEstimate,
    check_sql,
    date_bounded_references,
    is_date_column,
    plan_estimate,
)
from analyst_agent.schemas import GuardDecision, QueryResult


@dataclass
class ColumnInfo:
    name: str
    type: str


@dataclass
class TableInfo:
    schema: str
    name: str
    columns: list[ColumnInfo]
    row_count: int
    description: str = ""


TABLE_DESCRIPTIONS = {
    "fct_station_day": (
        "One row per weather station per day: tmax_c, tmin_c, tavg_c (degrees C), prcp_mm, snow_mm, "
        "snwd_mm (mm), awnd_ms (m/s), diurnal_range_c."
    ),
    "dim_station": (
        "Station dimension with history (SCD2). Use is_current = true for the current row. "
        "state is the two-letter US state code; station_name is upper case."
    ),
    "mart_daily_state_summary": (
        "Per US state and day: reporting_stations, avg/max tmax_c, avg/min tmin_c, avg_prcp_mm, "
        "stations_with_rain, max_prcp_mm."
    ),
    "mart_florida_weather": (
        "Florida station-days with station attributes and Fahrenheit/inch conversions "
        "(tmax_f, tmin_f, prcp_in)."
    ),
    "mart_station_coverage": (
        "Per station: first_obs_date, last_obs_date, days_reported, days_with_tmax, days_with_prcp, "
        "state, network_code."
    ),
}


class Warehouse:
    """A DuckDB file opened read-only. One connection per call keeps threads and asyncio simple.

    `threads` and `memory_limit` cap what queries can take from the host; DuckDB's defaults are every core
    and 80% of RAM.
    """

    def __init__(
        self,
        path: Path,
        allowed_schemas: list[str],
        row_limit: int,
        timeout_seconds: float,
        *,
        threads: int = 2,
        memory_limit: str = "512MiB",
    ) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"warehouse file {path} does not exist; run scripts/build_demo_warehouse.py"
            )
        self.path = Path(path)
        self.allowed_schemas = allowed_schemas
        self.row_limit = row_limit
        self.timeout_seconds = timeout_seconds
        self.threads = threads
        self.memory_limit = memory_limit
        self._tables: list[TableInfo] | None = None

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(
            str(self.path),
            read_only=True,
            config={"threads": self.threads, "memory_limit": self.memory_limit},
        )

    def tables(self) -> list[TableInfo]:
        if self._tables is None:
            con = self._connect()
            try:
                rows = con.execute(
                    "select table_schema, table_name from information_schema.tables "
                    "where table_schema in (select unnest(?::varchar[])) order by 1, 2",
                    [self.allowed_schemas],
                ).fetchall()
                out: list[TableInfo] = []
                for schema, name in rows:
                    cols = con.execute(
                        "select column_name, data_type from information_schema.columns "
                        "where table_schema = ? and table_name = ? order by ordinal_position",
                        [schema, name],
                    ).fetchall()
                    count = con.execute(f'select count(*) from "{schema}"."{name}"').fetchone()
                    out.append(
                        TableInfo(
                            schema=schema,
                            name=name,
                            columns=[ColumnInfo(c, t) for c, t in cols],
                            row_count=int(count[0]) if count else 0,
                            description=TABLE_DESCRIPTIONS.get(name, ""),
                        )
                    )
                self._tables = out
            finally:
                con.close()
        return self._tables

    def allowed_tables(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for t in self.tables():
            out.setdefault(t.schema, set()).add(t.name)
        return out

    def schema_prompt(self) -> str:
        """Compact schema description for the model's context."""
        lines = []
        for t in self.tables():
            cols = ", ".join(f"{c.name} {c.type.lower()}" for c in t.columns)
            lines.append(f"- {t.schema}.{t.name} ({t.row_count:,} rows): {t.description}\n  columns: {cols}")
        return "\n".join(lines)

    def guard(self, sql: str, approval_row_threshold: int) -> GuardDecision:
        """Allow or reject the SQL, and decide whether it needs a human's approval.

        Approval is needed when a pairwise join (no equality condition) would compare more row pairs than
        the threshold, or when the estimated scan exceeds the threshold and some reference to a table larger
        than the threshold lacks a real date range: a constant bound on a date column that the optimiser
        kept as a filter on that table's scan.
        """
        outcome: GuardOutcome = check_sql(sql, self.allowed_schemas, self.allowed_tables(), self.row_limit)
        if not outcome.allowed:
            return GuardDecision(allowed=False, sql=outcome.sql, reason=outcome.reason)
        sizes = {f"{t.schema}.{t.name}": t.row_count for t in self.tables()}
        dates = {
            f"{t.schema}.{t.name}": {c.name.lower() for c in t.columns if is_date_column(c.name)}
            for t in self.tables()
        }
        refs = date_bounded_references(outcome.sql, self.allowed_schemas, self.allowed_tables(), dates)
        plan = self._plan(outcome.sql)
        # EXPLAIN alone under-reports (a bare count(*) becomes a statistics lookup), so the scan estimate also
        # counts every reference at its table's full size; a self-join counts twice.
        estimated = max(plan.rows, plan.pairs, sum(sizes.get(t, 0) for t, _ in refs))
        large = Counter(t for t, _ in refs if sizes.get(t, 0) > approval_row_threshold)
        unbounded = sorted(
            {t for t, bounded in refs if t in large and not bounded}
            | {t for t, n in large.items() if plan.filtered_scans(t, dates.get(t, set())) < n}
        )
        date_bounded = not unbounded and any(bounded for _, bounded in refs)
        if plan.pairs > approval_row_threshold:
            needs_approval = True
            reason = f"a join without an equality condition compares about {plan.pairs:,} row pairs"
        elif estimated > approval_row_threshold and not date_bounded:
            needs_approval = True
            on = ", ".join(unbounded) or "any table"
            reason = f"estimated scan of {estimated:,} rows with no date range on {on}"
        else:
            needs_approval = False
            reason = f"ok; estimated scan {estimated:,} rows"
        if needs_approval:
            reason += "; needs approval"
        return GuardDecision(
            allowed=True,
            sql=outcome.sql,
            reason=reason,
            estimated_rows=estimated,
            needs_approval=needs_approval,
        )

    def _plan(self, sql: str) -> PlanEstimate:
        """The optimised plan's estimate; empty when DuckDB cannot plan the query (it would fail to run)."""
        con = self._connect()
        try:
            rows = con.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchall()
            return plan_estimate([node for _, text in rows for node in json.loads(text)])
        except duckdb.Error:
            return PlanEstimate()
        finally:
            con.close()

    def run(self, sql: str) -> QueryResult:
        """Execute guarded SQL with a wall-clock timeout; the connection is interrupted when it expires."""
        con = self._connect()
        timer = threading.Timer(self.timeout_seconds, con.interrupt)
        timer.daemon = True
        t0 = time.perf_counter()
        try:
            timer.start()
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(self.row_limit + 1)
        finally:
            timer.cancel()
            con.close()
        truncated = len(rows) > self.row_limit
        rows = rows[: self.row_limit]
        return QueryResult(
            columns=columns,
            rows=[[_jsonable(v) for v in r] for r in rows],
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


def _jsonable(v: Any) -> Any:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, int | float | str | bool) or v is None:
        return v
    return str(v)
