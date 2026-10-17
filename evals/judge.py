"""Optional LLM-as-judge for answer faithfulness. Costs money: only used with --judge and a real provider.

The judge sees the question, the query result and the answer, and must say whether every claim in the answer
is supported by the result. It is a second opinion on top of the in-graph verifier, run with the same model
provider but a different prompt, and its verdicts are recorded next to the deterministic scores, never mixed
into them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from analyst_agent.config import Settings
from analyst_agent.llm import LangChainPlanner, _format_result
from analyst_agent.schemas import QueryResult, Usage

JUDGE_SYSTEM = """You are grading an analytics assistant. You get the user's question, the rows the assistant's
query returned, and the assistant's written answer. Grade strictly:
- faithful: every number and factual statement in the answer is present in, or directly computable from, the rows.
- complete: the answer actually addresses what was asked (not a different question).
Explain any failure in one sentence."""


class Verdict(BaseModel):
    faithful: bool = Field(description="No claim in the answer goes beyond the rows")
    complete: bool = Field(description="The answer addresses the question that was asked")
    notes: str = ""


class Judge:
    def __init__(self, settings: Settings) -> None:
        if settings.provider == "fake":
            raise ValueError("the judge needs a real provider (anthropic or bedrock)")
        self.planner = LangChainPlanner(settings)

    def grade(self, question: str, answer: str, result: QueryResult) -> tuple[Verdict, Usage]:
        user = f"Question: {question}\n\nRows:\n{_format_result(result)}\n\nAnswer: {answer}"
        verdict, usage = self.planner.structured(Verdict, JUDGE_SYSTEM, user)
        assert isinstance(verdict, Verdict)
        return verdict, usage
