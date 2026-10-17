# 0001. A LangGraph supervisor with explicit nodes, not a single tool-calling loop

Date: 2026-10-17. Status: accepted.

## Context

The agent answers analytics questions over a warehouse. The simplest build is one model call in a
tool-calling loop with a `run_sql` tool. That works in a demo and fails in review: there is no place to
put the SQL guard that cannot be skipped, no way to pause for a human before an expensive query, no
bounded retry, and the trace is one opaque span.

## Decision

The agent is a LangGraph `StateGraph` with named nodes and typed state (`src/analyst_agent/graph.py`):

```
supervisor -> analyst -> guard -> [approval] -> execute -> synthesize -> verify
                 ^          |                      |            ^           |
                 +-- repair -+----------------------+            +-- reflect -+
```

- The **supervisor** routes: query the warehouse, or respond directly (clarify, refuse, small talk).
- The **analyst** returns a structured `SqlPlan` (sql | clarify | refuse), never free text.
- The **guard** is code, not a prompt. It can send the analyst back with the rejection reason
  (at most `MAX_REPAIRS` = 2 times), or route to **approval**, a LangGraph `interrupt` that suspends the
  thread until a human answers.
- **execute** runs the guarded SQL read-only with a timeout and row cap; a runtime error also goes back
  to the analyst with the error text.
- **synthesize** streams the answer; **verify** checks it against the rows and can send the synthesizer
  back once (`MAX_REFLECTIONS` = 1) with its issues. A still-unsupported answer is returned with
  `verification.supported = false` rather than hidden.

The model sits behind a `Planner` protocol with two implementations: `LangChainPlanner` (Anthropic API
or Bedrock, structured outputs, tenacity retries, token accounting) and `FakePlanner` (deterministic,
replays gold SQL). Every node is one OpenTelemetry span with its own attributes.

## Consequences

- Each safety property lives in one node and has a unit test that drives that node's branch
  (`tests/unit/test_graph.py`: repair, bounded repair, runtime repair, reflection, approval, decline).
- Loops are bounded by state counters, so a bad model cannot spin: the worst case is
  1 + 2 + 1 + 2 + 2 = 8 model calls per turn.
- The graph is more code than a loop (about 200 lines) and adds one model call for the supervisor on
  every turn. The fake planner makes that overhead measurable: 3.6 model calls per turn on the golden set.
- Swapping providers is a settings change; swapping the orchestration library is not.
