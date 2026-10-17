# 0004. Conversation memory and pending approvals live in the LangGraph checkpointer (Postgres in production)

Date: 2026-10-17. Status: accepted.

## Context

Follow-up questions ("and the lowest?", "same for Georgia") need earlier turns. A pending approval must
survive the replica that created it being replaced. Keeping either in process memory means a deploy or a
second replica loses them.

## Decision

Thread state is the graph state: `history` (append-only, reducer) plus the current turn's fields, keyed by
`thread_id` through the checkpointer. Locally and in tests that is `MemorySaver`; when
`AGENT_DATABASE_URL` is set it is `PostgresSaver` on a psycopg connection pool (`saver.setup()` creates its
tables on first start). Each turn starts with `new_turn_input()`, which resets the per-turn fields and
leaves `history` to the reducer; per-turn `usage` is summed explicitly in the nodes rather than through a
reducer so it does not accumulate across turns (a bug the test suite caught).

The approval interrupt is a checkpoint like any other: `POST /v1/threads/{id}/approve` resumes with
`Command(resume=bool)`. A new message on a thread with a pending approval starts a fresh run and discards
the pending one, which is documented and tested rather than left implicit.

## Consequences

- `tests/integration/test_postgres_memory.py` creates a service, asks, closes it, creates a new one and
  asks the follow-up; and parks an approval in one instance and approves it from another. CI runs these
  against a Postgres service container.
- Readiness (`/readyz`) checks both the warehouse file and a `select 1` on the pool, so a replica with a
  broken database connection is taken out of the load balancer.
- History is the full exchange text; the planner sees the last six turns. There is no summarisation and
  no retention policy, which would be the next step for a long-lived deployment.
