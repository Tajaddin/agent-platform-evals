"""Run the golden set and the multi-turn conversations through the agent and score them.

    uv run python evals/run_evals.py                      # provider from AGENT_PROVIDER (default fake)
    uv run python evals/run_evals.py --provider anthropic --judge
    uv run python evals/run_evals.py --gate               # exit 1 when below evals/baseline.json thresholds

Scores:
- execution accuracy: the agent's result rows equal the gold query's rows (order-insensitive, numeric tolerance)
- behaviour accuracy: clarify / refuse / direct answers land in the expected status set
- approval accuracy: the human-in-the-loop interrupt fires exactly when expected
- conversation pass rate: every turn of a multi-turn thread passes (memory is exercised)
- latency p50 / p95, LLM calls, tokens and estimated cost per turn

With the fake provider the pipeline is measured, not a model: the fake planner answers with the gold SQL, so
execution accuracy below 100% there means a bug in the guard, executor, memory or API, never model quality.
Results are written to evals/results/ with the provider in the file name so they are never confused.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analyst_agent.config import Settings  # noqa: E402
from analyst_agent.logging import configure_logging  # noqa: E402
from analyst_agent.schemas import ChatResponse, QueryResult  # noqa: E402
from analyst_agent.service import AgentService  # noqa: E402
from evals.scoring import answer_mentions_result, execution_match, percentile  # noqa: E402

GOLDEN = ROOT / "evals" / "golden"
RESULTS = ROOT / "evals" / "results"


@dataclass
class TurnScore:
    id: str
    question: str
    status: str
    approval_required: bool
    expected_approval: bool | None
    approval_ok: bool | None
    exec_match: bool | None
    status_ok: bool | None
    mentions_result: bool | None
    verifier_supported: bool | None
    judge_faithful: bool | None
    judge_complete: bool | None
    latency_ms: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    sql: str
    answer: str
    passed: bool
    notes: list[str] = field(default_factory=list)


def gold_result(svc: AgentService, sql: str) -> QueryResult:
    decision = svc.warehouse.guard(sql, approval_row_threshold=10**12)
    if not decision.allowed:
        raise RuntimeError(f"gold SQL rejected by the guard: {decision.reason}\n{sql}")
    return svc.warehouse.run(decision.sql)


def run_turn(
    svc: AgentService,
    thread_id: str,
    item: dict[str, Any],
    item_id: str,
    judge: Any | None,
) -> TurnScore:
    message = item.get("question") or item["message"]
    expect_approval = item.get("expect_approval")
    approve = item.get("approve", True)
    t0 = time.perf_counter()
    resp: ChatResponse = svc.chat(thread_id, message)
    approval_required = resp.status == "approval_required"
    if approval_required:
        resp = svc.approve(thread_id, bool(approve))
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    notes: list[str] = []
    exec_match: bool | None = None
    mentions: bool | None = None
    judge_f: bool | None = None
    judge_c: bool | None = None
    if item.get("gold_sql") and approve is not False:
        expected = gold_result(svc, item["gold_sql"])
        exec_match = resp.status == "answered" and execution_match(expected, resp.result)
        if not exec_match:
            notes.append(
                f"status={resp.status}; got {resp.result.row_count if resp.result else 0} rows, expected {expected.row_count}"
            )
        mentions = answer_mentions_result(resp.answer, resp.result) if resp.status == "answered" else False
        if judge is not None and resp.status == "answered" and resp.result is not None:
            verdict, _ = judge.grade(message, resp.answer, resp.result)
            judge_f, judge_c = verdict.faithful, verdict.complete
            if not (verdict.faithful and verdict.complete):
                notes.append(f"judge: {verdict.notes}")
    status_ok: bool | None = None
    if item.get("expect_status"):
        status_ok = resp.status in item["expect_status"]
        if not status_ok:
            notes.append(f"status {resp.status} not in {item['expect_status']}")
    approval_ok: bool | None = None
    if expect_approval is not None:
        approval_ok = approval_required == bool(expect_approval)
        if not approval_ok:
            notes.append(f"approval_required={approval_required}, expected {expect_approval}")
    elif approval_required:
        approval_ok = False
        notes.append("unexpected approval interrupt")

    checks = [c for c in (exec_match, status_ok, approval_ok) if c is not None]
    passed = bool(checks) and all(checks)
    return TurnScore(
        id=item_id,
        question=message,
        status=resp.status,
        approval_required=approval_required,
        expected_approval=expect_approval,
        approval_ok=approval_ok,
        exec_match=exec_match,
        status_ok=status_ok,
        mentions_result=mentions,
        verifier_supported=resp.verification.supported if resp.verification else None,
        judge_faithful=judge_f,
        judge_complete=judge_c,
        latency_ms=latency_ms,
        llm_calls=resp.usage.llm_calls,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        cost_usd=resp.usage.cost_usd,
        sql=resp.sql,
        answer=resp.answer[:500],
        passed=passed,
        notes=notes,
    )


def rate(flags: list[bool | None]) -> float | None:
    vals = [f for f in flags if f is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(turns: list[TurnScore], conversations: dict[str, list[TurnScore]]) -> dict[str, Any]:
    lat = [t.latency_ms for t in turns]
    conv_pass = [all(t.passed for t in ts) for ts in conversations.values()]
    return {
        "turns": len(turns),
        "execution_accuracy": rate([t.exec_match for t in turns]),
        "execution_n": sum(t.exec_match is not None for t in turns),
        "behaviour_accuracy": rate([t.status_ok for t in turns]),
        "behaviour_n": sum(t.status_ok is not None for t in turns),
        "approval_accuracy": rate([t.approval_ok for t in turns]),
        "approval_n": sum(t.approval_ok is not None for t in turns),
        "answer_mentions_result_rate": rate([t.mentions_result for t in turns]),
        "verifier_supported_rate": rate([t.verifier_supported for t in turns]),
        "judge_faithful_rate": rate([t.judge_faithful for t in turns]),
        "judge_complete_rate": rate([t.judge_complete for t in turns]),
        "conversations": len(conv_pass),
        "conversation_pass_rate": round(sum(conv_pass) / len(conv_pass), 4) if conv_pass else None,
        "latency_p50_ms": round(percentile(lat, 0.5), 1),
        "latency_p95_ms": round(percentile(lat, 0.95), 1),
        "latency_mean_ms": round(statistics.fmean(lat), 1) if lat else 0.0,
        "llm_calls_per_turn": round(statistics.fmean(t.llm_calls for t in turns), 2) if turns else 0.0,
        "input_tokens_total": sum(t.input_tokens for t in turns),
        "output_tokens_total": sum(t.output_tokens for t in turns),
        "cost_usd_total": round(sum(t.cost_usd for t in turns), 4),
        "cost_usd_per_turn": round(sum(t.cost_usd for t in turns) / len(turns), 5) if turns else 0.0,
        "failed": [t.id for t in turns if not t.passed],
    }


def markdown(summary: dict[str, Any], provider: str, model: str, when: str) -> str:
    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 100:.1f}%"

    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Provider / model | `{provider}` / `{model}` |",
        f"| Run (UTC) | {when} |",
        f"| Turns scored | {summary['turns']} |",
        f"| Execution accuracy (n={summary['execution_n']}) | {pct(summary['execution_accuracy'])} |",
        f"| Behaviour accuracy (n={summary['behaviour_n']}) | {pct(summary['behaviour_accuracy'])} |",
        f"| Approval accuracy (n={summary['approval_n']}) | {pct(summary['approval_accuracy'])} |",
        f"| Conversation pass rate (n={summary['conversations']}) | {pct(summary['conversation_pass_rate'])} |",
        f"| Answer mentions a result value | {pct(summary['answer_mentions_result_rate'])} |",
        f"| Verifier: answer supported | {pct(summary['verifier_supported_rate'])} |",
        f"| Judge: faithful / complete | {pct(summary['judge_faithful_rate'])} / {pct(summary['judge_complete_rate'])} |",
        f"| Latency p50 / p95 | {summary['latency_p50_ms']} ms / {summary['latency_p95_ms']} ms |",
        f"| LLM calls per turn | {summary['llm_calls_per_turn']} |",
        f"| Tokens in / out | {summary['input_tokens_total']:,} / {summary['output_tokens_total']:,} |",
        f"| Estimated cost | ${summary['cost_usd_total']:.4f} total, ${summary['cost_usd_per_turn']:.5f} per turn |",
    ]
    if summary["failed"]:
        lines.append(f"| Failed | {', '.join(summary['failed'])} |")
    return "\n".join(lines)


def gate(summary: dict[str, Any], provider: str) -> list[str]:
    baseline_path = ROOT / "evals" / "baseline.json"
    thresholds = json.loads(baseline_path.read_text(encoding="utf-8")).get(provider)
    if thresholds is None:
        return [f"no baseline for provider {provider!r} in {baseline_path.name}"]
    failures = []
    for key, minimum in thresholds.get("min", {}).items():
        value = summary.get(key)
        if value is None or value < minimum:
            failures.append(f"{key}={value} < {minimum}")
    for key, maximum in thresholds.get("max", {}).items():
        value = summary.get(key)
        if value is None or value > maximum:
            failures.append(f"{key}={value} > {maximum}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["fake", "anthropic", "bedrock"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge", action="store_true", help="also grade answers with an LLM judge (costs money)")
    ap.add_argument(
        "--gate", action="store_true", help="exit 1 when below the thresholds in evals/baseline.json"
    )
    ap.add_argument("--limit", type=int, default=None, help="only the first N golden questions (smoke runs)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    settings = Settings(**overrides)
    configure_logging("WARNING")
    svc = AgentService(settings)
    judge = None
    if args.judge:
        from evals.judge import Judge

        judge = Judge(settings)

    run_id = uuid.uuid4().hex[:8]
    questions = yaml.safe_load((GOLDEN / "questions.yaml").read_text(encoding="utf-8"))["questions"]
    conversations = yaml.safe_load((GOLDEN / "conversations.yaml").read_text(encoding="utf-8"))[
        "conversations"
    ]
    if args.limit:
        questions = questions[: args.limit]
        conversations = conversations[: max(1, args.limit // 10)]

    turns: list[TurnScore] = []
    for q in questions:
        score = run_turn(svc, f"eval-{run_id}-{q['id']}", q, q["id"], judge)
        turns.append(score)
        print(
            f"{'PASS' if score.passed else 'FAIL'} {score.id:5} {score.status:18} {score.latency_ms:8.1f} ms  {score.question[:70]}"
        )
        for n in score.notes:
            print(f"       {n}")

    conv_scores: dict[str, list[TurnScore]] = {}
    for conv in conversations:
        thread = f"eval-{run_id}-{conv['id']}"
        conv_scores[conv["id"]] = []
        for i, turn in enumerate(conv["turns"]):
            score = run_turn(svc, thread, turn, f"{conv['id']}.{i + 1}", judge)
            conv_scores[conv["id"]].append(score)
            turns.append(score)
            print(
                f"{'PASS' if score.passed else 'FAIL'} {score.id:5} {score.status:18} {score.latency_ms:8.1f} ms  {score.question[:70]}"
            )
            for n in score.notes:
                print(f"       {n}")
    svc.close()

    summary = summarize(turns, conv_scores)
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    model = svc.planner.name
    provider = settings.provider
    report = {
        "provider": provider,
        "model": model,
        "run_id": run_id,
        "timestamp_utc": when,
        "golden_questions": len(questions),
        "conversations": len(conversations),
        "summary": summary,
        "turns": [asdict(t) for t in turns],
        "note": (
            "fake provider: measures the pipeline (guard, executor, memory, API), not a model"
            if provider == "fake"
            else "live provider: model quality and cost are real; latency includes provider round trips"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (args.out / f"{provider}-{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / f"latest-{provider}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = markdown(summary, provider, model, when)
    (args.out / f"latest-{provider}.md").write_text(md + "\n", encoding="utf-8")
    print()
    print(md)

    if args.gate:
        failures = gate(summary, provider)
        if failures:
            print("\nGATE FAILED: " + "; ".join(failures))
            return 1
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
