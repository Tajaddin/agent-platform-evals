# Plan: analyst agent on a SQL warehouse

Goal: a production-shaped LLM agent that answers analytics questions over the marts built by
[elt-modern-data-stack](https://github.com/Tajaddin/elt-modern-data-stack), with the controls a reviewer
expects: guarded SQL, human approval for expensive queries, per-thread memory, traces and metrics,
evaluations that gate CI, and infrastructure as code.

## Milestones

| # | Milestone | Status | Evidence |
|---|---|---|---|
| 1 | Warehouse access, SQL guard, schemas | done | `tests/unit/test_guard.py`, `test_warehouse.py` |
| 2 | Graph: supervisor, analyst, guard, approval, execute, synthesize, verify; planners | done | `tests/unit/test_graph.py` (19 tests), `test_llm.py` |
| 3 | API: chat, SSE stream, approve, threads, health, metrics, auth, rate limit | done | `tests/integration/test_api.py` |
| 4 | Memory in Postgres, approvals survive restarts | done | `tests/integration/test_postgres_memory.py` (CI service container) |
| 5 | Golden set (48 + 8), conversations (6), eval runner with CI gate | done | `evals/`, `evals/results/latest-fake.md` |
| 6 | Observability: structlog JSON, Prometheus metrics, OTel spans, Grafana dashboard, 5 alerts with promtool tests | done | `observability/` |
| 7 | Compose stack, Dockerfile (non-root, healthcheck), k6 load test | done | `docker-compose.yml`, `load/` |
| 8 | CI: lint, types, tests, eval gate, alerts, gitleaks, image smoke, terraform validate; Dependabot | done | `.github/` |
| 9 | AWS: ECS Fargate + ALB + RDS + Bedrock IAM + OIDC deploy, teardown | validated, not applied | `infra/aws/` |
| 10 | Live eval run against the Anthropic API, baseline thresholds from real numbers | blocked on a key | `make evals-live` |

## Out of scope for this iteration

- Summarising long histories; retention policy for threads.
- HTTPS on the ALB (needs a domain and certificate).
- Traces and metrics shipped from AWS (CloudWatch logs only there).
- A UI; the API is the product surface, `/docs` is the console.
