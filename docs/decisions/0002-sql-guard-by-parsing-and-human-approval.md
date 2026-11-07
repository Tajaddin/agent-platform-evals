# 0002. Generated SQL is parsed and allowlisted, executed read-only, and gated by a human when expensive

Date: 2026-10-17. Status: accepted.

## Context

Text-to-SQL agents are injection targets: a user (or a document the model read) can steer it into
`DROP`, `COPY ... TO`, `read_parquet('s3://...')`, or a cross join that pins the warehouse. Prompt rules
("only SELECT") are not a control; the model follows them most of the time, which is not the same thing.

## Decision

Three independent layers (`guard.py`, `warehouse.py`):

1. **Parse, do not pattern-match.** sqlglot parses the statement in the DuckDB dialect. It must be exactly
   one `SELECT`/`UNION`; every table must resolve to the allowlist (`information_schema`, `raw`, anything
   unknown is rejected); any table function in `FROM` (`glob`, `read_parquet`, `range`) and any banned
   scalar function is rejected; CTE names are recognised so they are not mistaken for tables. A `LIMIT` is
   added or lowered to the configured cap on the outermost query.
2. **The connection cannot do damage anyway.** DuckDB is opened `read_only=True`, one connection per
   query, with a wall-clock timeout enforced by `connection.interrupt()` from a timer thread and a fetch
   cap of `row_limit + 1` so truncation is detected and reported. Each connection is opened with
   `threads` and `memory_limit` set (`AGENT_QUERY_THREADS`, `AGENT_QUERY_MEMORY_LIMIT`: 2 and 512 MiB)
   instead of DuckDB's defaults of every core and 80% of RAM, so a query that slips through cannot starve
   the API process or the host while the timeout runs.
3. **Expensive is a human decision.** The guard estimates the scan from DuckDB's optimised plan
   (`EXPLAIN (FORMAT JSON)`) and the row counts of every table reference (a self-join counts twice). The
   graph interrupts when a join without an equality condition (a cross product, a nested loop, or a range
   join such as `a.obs_date < b.obs_date`) could return more than `approval_row_threshold` (100,000) row
   pairs, whatever its filters; when the optimiser estimates that any join returns more rows than the
   threshold (an equality join on a column with few distinct values); or when the scan estimate is above
   the threshold and some reference to a table larger than the threshold has no real date range: a
   constant bound (`=`, `IN`, `BETWEEN`, `<`, `>`) on a date column, AND-ed into that reference's own
   `WHERE` or `ON`, that is false at the column's earliest or latest value, and that the optimiser kept as
   a filter on the table's scan. The earliest and latest values are read once, with the row counts, and
   each bound is evaluated against them by DuckDB. A range that holds at both ends covers all the data
   (`obs_date >= date '1900-01-01'`, however many other predicates sit next to it), so it does not count,
   and neither do `IS NOT NULL`, functions of the column (`year(obs_date) > 0`), OR-ed terms or comparisons
   with other columns. The API returns `approval_required` with the SQL and estimate, and the thread
   resumes only on `POST /v1/threads/{id}/approve`. Approval state survives process restarts because it
   lives in the checkpointer (tested against Postgres).

## Consequences

- 17 rejection cases and the allow paths are unit tested without a model; the eval gate proves the gold
  queries all pass the guard (any rule that is too strict fails CI).
- A rejected query goes back to the analyst with the reason, so a real model gets one chance to fix a
  schema slip rather than failing the turn. A query DuckDB cannot plan (an unknown column) is rejected with
  DuckDB's error rather than sent for approval.
- The row estimate is deliberately pessimistic (table size, not post-filter cardinality); a date range
  on every large reference is what lets a big table through without a human. The threshold is a setting
  and the tests show changing it moves the approval boundary.
- A range that excludes only part of the data (`obs_date > date '2026-10-03'` over 12 days) counts as a
  date range: selectivity is not estimated. An `IN` list is judged by the range it spans, so one that
  names both the earliest and the latest date needs approval even if it skips the days between. The rule
  is conservative elsewhere too: a range that reaches a reference only through a join condition or an
  outer query's filter does not count for it.
- Not covered: query cost in CPU terms (a nasty regex over 90,000 rows is "cheap" by row count). The
  timeout is the backstop for that.
