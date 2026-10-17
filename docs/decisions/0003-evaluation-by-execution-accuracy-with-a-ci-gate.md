# 0003. Quality is measured by execution accuracy against gold SQL, gated in CI, labelled by provider

Date: 2026-10-17. Status: accepted.

## Context

"The agent works" has to mean something a reviewer can re-run. Comparing generated SQL text to a
reference is brittle (many correct spellings); asking a model to grade a model is circular when it is
the only signal.

## Decision

`evals/golden/questions.yaml` holds 48 questions with gold SQL and 8 behaviour cases (clarify, refuse,
direct answer); `evals/golden/conversations.yaml` holds 6 multi-turn threads whose later turns depend on
memory. `evals/run_evals.py` scores:

- **execution accuracy**: the agent's result rows equal the gold query's rows (order-insensitive multiset,
  numeric tolerance 1e-6), both run against the same warehouse at eval time, so nothing is snapshotted;
- **behaviour accuracy**: status lands in the expected set;
- **approval accuracy**: the human-in-the-loop interrupt fires exactly when expected (and the runner
  answers it);
- **conversation pass rate**: every turn of a thread passes;
- **latency p50/p95, model calls, tokens and estimated cost per turn**;
- two faithfulness proxies recorded but not gated: "the answer mentions a number from the first result
  row" and the in-graph verifier's verdict; with `--judge`, an LLM judge with a separate prompt.

`--gate` compares the summary with `evals/baseline.json`. The **fake provider** replays gold SQL, so its
thresholds are 100%: any miss there is a pipeline bug (guard, executor, memory, API). Live providers get
their own thresholds, set from the first real run and only ratcheted up. Results are written to
`evals/results/latest-<provider>.{json,md}` and the README states which provider produced each number.

## Consequences

- CI runs the fake gate on every push with no key and no cost (67 turns in about 2 seconds).
- Live evals run only on manual dispatch and only when the `ANTHROPIC_API_KEY` secret exists, so a fork
  or a PR cannot spend money.
- Adding a question is adding gold SQL; the eval refuses to start if gold SQL fails the guard, so the
  golden set cannot drift away from what the agent is allowed to do.
- Limits: 48 questions over one 12-day dataset is a smoke-level set, not a benchmark; execution accuracy
  cannot see a correct number produced by a wrong query that happens to agree; and the fake provider's
  latency says nothing about model speed. All three are stated next to the numbers.
