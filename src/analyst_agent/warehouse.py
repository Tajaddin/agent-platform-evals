"""Read-only access to the warehouse, schema description for the model, guarded execution."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from analyst_agent.guard import GuardOutcome, check_sql, estimated_rows_from_explain, has_date_filter
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
    """A DuckDB file opened read-only. One connection per call keeps threads and asyncio simple."""

    def __init__(
        self, path: Path, allowed_schemas: list[str], row_limit: int, timeout_seconds: float
    ) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"warehouse file {path} does not exist; run scripts/build_demo_warehouse.py"
            )
        self.path = Path(path)
        self.allowed_schemas = allowed_schemas
        self.row_limit = row_limit
        self.timeout_seconds = timeout_seconds
        self._tables: list[TableInfo] | None = None

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path), read_only=True)

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
        outcome: GuardOutcome = check_sql(sql, self.allowed_schemas, self.allowed_tables(), self.row_limit)
        if not outcome.allowed:
            return GuardDecision(allowed=False, sql=outcome.sql, reason=outcome.reason)
        # EXPLAIN alone under-reports (a bare count(*) becomes a statistics lookup), so the scan estimate
        # is the larger of the plan's cardinality and the size of the tables the query touches.
        estimated = max(self._estimate_rows(outcome.sql), self._table_rows(outcome.tables))
        needs_approval = estimated > approval_row_threshold and not has_date_filter(outcome.sql)
        reason = (
            f"estimated scan of {estimated:,} rows with no date filter; needs approval"
            if needs_approval
            else f"ok; estimated scan {estimated:,} rows"
        )
        return GuardDecision(
            allowed=True,
            sql=outcome.sql,
            reason=reason,
            estimated_rows=estimated,
            needs_approval=needs_approval,
        )

    def _table_rows(self, qualified_names: tuple[str, ...]) -> int:
        sizes = {f"{t.schema}.{t.name}": t.row_count for t in self.tables()}
        return sum(sizes.get(name, 0) for name in qualified_names)

    def _estimate_rows(self, sql: str) -> int:
        con = self._connect()
        try:
            plan = con.execute(f"EXPLAIN {sql}").fetchall()
            text = "\n".join(str(part) for row in plan for part in row)
            return estimated_rows_from_explain(text)
        except duckdb.Error:
            return 0
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
