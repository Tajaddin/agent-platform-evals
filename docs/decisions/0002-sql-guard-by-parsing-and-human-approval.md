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
   cap of `row_limit + 1` so truncation is detected and reported.
3. **Expensive is a human decision.** The guard estimates the scan from the referenced tables' row counts
   and DuckDB's `EXPLAIN` cardinalities. Above `approval_row_threshold` (100,000 rows) with no date filter,
   the graph interrupts; the API returns `approval_required` with the SQL and estimate, and the thread
   resumes only on `POST /v1/threads/{id}/approve`. Approval state survives process restarts because it
   lives in the checkpointer (tested against Postgres).

## Consequences

- 17 rejection cases and the allow paths are unit tested without a model; the eval gate proves the gold
  queries all pass the guard (any rule that is too strict fails CI).
- A rejected query goes back to the analyst with the reason, so a real model gets one chance to fix a
  schema slip rather than failing the turn.
- The row estimate is deliberately pessimistic (table size, not post-filter cardinality) for queries
  without a date filter; a filtered query is trusted to `EXPLAIN`. The threshold is a setting and the
  tests show changing it moves the approval boundary.
- Not covered: query cost in CPU terms (a nasty regex over 90,000 rows is "cheap" by row count). The
  timeout is the backstop for that.
