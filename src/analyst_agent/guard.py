"""Query guard: the model proposes SQL, this module decides whether it may run.

Rules (all enforced by parsing, not by string matching):
- exactly one statement, and it is a SELECT (CTEs allowed);
- every table referenced is in an allowed schema (unqualified names are resolved against them);
- no table functions that read files or the network (read_parquet, read_csv, glob, ...);
- no ATTACH, COPY, PRAGMA or other commands;
- a LIMIT is enforced (added when missing, lowered when too high).
For the approval gate it also reports which table references a constant date range bounds, and reads
DuckDB's optimised plan for cardinalities, joins that compare every pair of rows, and the filters each
table scan kept.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

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


def _resolve(
    schema: str | None, name: str, allowed_schemas: list[str], allowed_tables: dict[str, set[str]]
) -> str | None:
    """schema.table for an allowlisted table, or None."""
    candidates = [schema] if schema else allowed_schemas
    hit = next((s for s in candidates if s in allowed_tables and name in allowed_tables[s]), None)
    return f"{hit}.{name}" if hit is not None and hit in allowed_schemas else None


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
        hit = _resolve(schema, name, allowed_schemas, allowed_tables)
        if hit is None:
            return GuardOutcome(
                False, sql, f"table {schema + '.' if schema else ''}{name} is not in the allowlist"
            )
        resolved.append(hit)

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


def is_date_column(name: str) -> bool:
    """obs_date, first/last_obs_date, valid_from/to: the columns a date range can bound."""
    return "date" in name.lower() or name.lower() in {"valid_from", "valid_to"}


_BOUNDS = (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _conjuncts(condition: exp.Expression) -> Iterator[exp.Expression]:
    condition = condition.unnest()  # type: ignore[no-untyped-call]
    if isinstance(condition, exp.And):
        yield from _conjuncts(condition.left)
        yield from _conjuncts(condition.right)
    else:
        yield condition


def _bounded_column(predicate: exp.Expression) -> exp.Column | None:
    """The column bounded by `col = c`, `col IN (c, ...)`, `col BETWEEN c AND c` or `col < c` (any order)."""
    values: list[exp.Expression]
    if isinstance(predicate, exp.Between):
        column, values = predicate.this, [predicate.args["low"], predicate.args["high"]]
    elif isinstance(predicate, exp.In):
        column, values = predicate.this, list(predicate.expressions)
    elif isinstance(predicate, _BOUNDS):
        column, values = predicate.this, [predicate.expression]
        if not isinstance(column, exp.Column):
            column, values = predicate.expression, [predicate.this]
    else:
        return None
    constant = all(v.find(exp.Column, exp.Query) is None for v in values)
    return column if isinstance(column, exp.Column) and values and constant else None


def date_bounded_references(
    sql: str,
    allowed_schemas: list[str],
    allowed_tables: dict[str, set[str]],
    date_columns: dict[str, set[str]],
) -> list[tuple[str, bool]]:
    """Every table reference as (schema.table, bounded); a self-join yields two entries.

    A reference is bounded when its own SELECT's WHERE or JOIN ... ON clause has, among its AND-ed terms, a
    constant bound on one of its table's `date_columns`. `IS NOT NULL`, OR-ed terms and comparisons with
    other columns or subqueries do not count.
    """
    tree = sqlglot.parse_one(sql, read="duckdb")
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    out: list[tuple[str, bool]] = []
    for select in tree.find_all(exp.Select):
        refs: list[tuple[str, str]] = []  # (alias, schema.table)
        for t in select.find_all(exp.Table):
            if t.find_ancestor(exp.Select) is not select or t.name.lower() in cte_names:
                continue
            name = _resolve(t.db.lower() or None, t.name.lower(), allowed_schemas, allowed_tables)
            if name is not None:
                refs.append((t.alias_or_name.lower(), name))
        where = select.args.get("where")
        conditions = [where.this] if where is not None else []
        conditions += [
            j.args["on"]
            for j in select.find_all(exp.Join)
            if j.find_ancestor(exp.Select) is select and j.args.get("on") is not None
        ]
        bounded: set[str] = set()
        for predicate in (p for c in conditions for p in _conjuncts(c)):
            col = _bounded_column(predicate)
            if col is None:
                continue
            owners = [
                alias
                for alias, table in refs
                if (not col.table or col.table.lower() == alias)
                and col.name.lower() in date_columns.get(table, set())
            ]
            if len(owners) == 1:
                bounded.add(owners[0])
        out += [(table, alias in bounded) for alias, table in refs]
    return out


# Joins with no equality condition: DuckDB compares every pair of input rows.
PAIRWISE_JOINS = {"CROSS_PRODUCT", "NESTED_LOOP_JOIN", "BLOCKWISE_NL_JOIN"}


@dataclass(frozen=True)
class PlanEstimate:
    """What the approval gate reads from DuckDB's optimised plan (`EXPLAIN (FORMAT JSON)`)."""

    rows: int = 0  # largest estimated cardinality of any operator
    pairs: int = 0  # largest input product of a pairwise join
    scans: tuple[tuple[str, str], ...] = ()  # (schema.table, filters) of every table scan

    def filtered_scans(self, table: str, columns: set[str]) -> int:
        """Scans of `table` that kept a filter on one of `columns`.

        The optimiser removes a filter it can prove always true from the table's statistics
        (`obs_date >= date '1900-01-01'`, a range covering all the data), so such a range leaves no filter.
        """
        return sum(
            1
            for name, filters in self.scans
            if name == table and any(re.search(rf"\b{re.escape(c)}\b", filters) for c in columns)
        )


def plan_estimate(plan: list[dict[str, Any]]) -> PlanEstimate:
    rows = pairs = 0
    scans: list[tuple[str, str]] = []

    def walk(node: dict[str, Any]) -> int:
        nonlocal rows, pairs
        inputs = [walk(child) for child in node.get("children", [])]
        info = node.get("extra_info") or {}
        name = node.get("name")
        product = math.prod(max(n, 1) for n in inputs) if inputs else 0
        if name in PAIRWISE_JOINS:
            pairs = max(pairs, product)
        if "Estimated Cardinality" in info:
            card = int(info["Estimated Cardinality"])
        elif name == "UNGROUPED_AGGREGATE":
            card = 1
        elif name in PAIRWISE_JOINS:
            card = product
        else:
            card = max(inputs, default=0)
        if name == "SEQ_SCAN" and "Table" in info:
            filters = info.get("Filters", "")
            text = " AND ".join(filters) if isinstance(filters, list) else str(filters)
            scans.append((".".join(info["Table"].lower().split(".")[-2:]), text))
        rows = max(rows, card)
        return card

    for root in plan:
        walk(root)
    return PlanEstimate(rows=rows, pairs=pairs, scans=tuple(scans))
