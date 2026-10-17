"""Query guard: the model proposes SQL, this module decides whether it may run.

Rules (all enforced by parsing, not by string matching):
- exactly one statement, and it is a SELECT (CTEs allowed);
- every table referenced is in an allowed schema (unqualified names are resolved against them);
- no table functions that read files or the network (read_parquet, read_csv, glob, ...);
- no ATTACH, COPY, PRAGMA or other commands;
- a LIMIT is enforced (added when missing, lowered when too high).
The scan size is estimated from the referenced tables' row counts and DuckDB's EXPLAIN cardinalities so
expensive queries can be routed to a human for approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

BANNED_FUNCTIONS = {
    "read_parquet",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "glob",
    "parquet_scan",
    "csv_scan",
    "read_text",
    "read_blob",
    "sniff_csv",
    "duckdb_secrets",
}
BANNED_STATEMENT_WORDS = re.compile(
    r"\b(attach|detach|copy|export|import|install|load|pragma|set|call|create|insert|update|delete|drop|alter|"
    r"truncate|grant|revoke|vacuum|checkpoint)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardOutcome:
    allowed: bool
    sql: str
    reason: str = ""
    tables: tuple[str, ...] = ()


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return str(node.sql_name()).lower()  # type: ignore[no-untyped-call]


def _table_names(tree: exp.Expression) -> list[tuple[str | None, str]]:
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    out: list[tuple[str | None, str]] = []
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if not name or name in cte_names:
            continue
        schema = t.db.lower() if t.db else None
        out.append((schema, name))
    return out


def check_sql(  # noqa: PLR0911
    sql: str,
    allowed_schemas: list[str],
    allowed_tables: dict[str, set[str]],
    row_limit: int,
) -> GuardOutcome:
    """Validate and normalise the SQL. `allowed_tables` maps schema -> table names that exist."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return GuardOutcome(False, sql, "empty query")
    if BANNED_STATEMENT_WORDS.search(sql.split("--")[0]) and not sql.lower().lstrip().startswith(
        ("select", "with")
    ):
        return GuardOutcome(False, sql, "only SELECT statements are allowed")
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        return GuardOutcome(False, sql, f"could not parse SQL: {str(exc)[:160]}")
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return GuardOutcome(False, sql, "exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.Union):
        return GuardOutcome(False, sql, f"only SELECT statements are allowed, got {type(tree).__name__}")
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Func):  # glob('*.csv'), read_parquet(...), range(...)
            return GuardOutcome(False, sql, f"table function {_func_name(table.this)} is not allowed")
    for node in tree.find_all(exp.Func):
        if _func_name(node) in BANNED_FUNCTIONS:
            return GuardOutcome(False, sql, f"function {_func_name(node)} is not allowed")
    if list(tree.find_all(exp.Command)):
        return GuardOutcome(False, sql, "commands are not allowed")

    tables = _table_names(tree)
    if not tables:
        return GuardOutcome(False, sql, "the query reads no table")
    resolved: list[str] = []
    for schema, name in tables:
        candidates = [schema] if schema else allowed_schemas
        hit = next((s for s in candidates if s in allowed_tables and name in allowed_tables[s]), None)
        if hit is None or hit not in allowed_schemas:
            return GuardOutcome(
                False, sql, f"table {schema + '.' if schema else ''}{name} is not in the allowlist"
            )
        resolved.append(f"{hit}.{name}")

    # Enforce the row limit on the outermost query.
    limit_node = tree.args.get("limit")
    if limit_node is None:
        tree = tree.limit(row_limit)
    else:
        try:
            current = int(limit_node.expression.this)
        except (TypeError, ValueError, AttributeError):
            current = row_limit + 1
        if current > row_limit:
            tree.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
    return GuardOutcome(True, tree.sql(dialect="duckdb"), "", tuple(sorted(set(resolved))))


_EC = re.compile(r"EC:\s*([0-9][0-9,]*)")  # older DuckDB plan format
_TILDE = re.compile(r"~\s*([0-9][0-9,]*)\s+rows?")  # DuckDB 1.x: "~6,927 rows"


def estimated_rows_from_explain(explain_text: str) -> int:
    """Largest estimated cardinality in a DuckDB EXPLAIN plan (0 when the plan has none)."""
    found = _EC.findall(explain_text) + _TILDE.findall(explain_text)
    values = [int(v.replace(",", "")) for v in found]
    return max(values) if values else 0


def has_date_filter(sql: str) -> bool:
    """True when the WHERE clause constrains a date column (obs_date, first/last_obs_date, valid_from/to)."""
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.ParseError:
        return False
    for where in tree.find_all(exp.Where):
        for col in where.find_all(exp.Column):
            if "date" in col.name.lower() or col.name.lower() in {"valid_from", "valid_to"}:
                return True
    return False
