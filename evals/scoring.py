"""Scoring helpers shared by the eval runner and the tests."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from analyst_agent.schemas import QueryResult

REL_TOL = 1e-6
ABS_TOL = 1e-6


def _norm(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, int | float):
        if isinstance(v, float) and math.isnan(v):
            return "nan"
        return round(float(v), 6)
    if v is None:
        return None
    return str(v)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return bool(a == b)


def rows_match(expected: list[list[Any]], got: list[list[Any]]) -> bool:
    """Order-insensitive multiset comparison of result rows, numbers within tolerance."""
    if len(expected) != len(got):
        return False
    if not expected:
        return True
    if any(len(r) != len(expected[0]) for r in expected + got):
        return False
    exp_norm = [tuple(_norm(v) for v in r) for r in expected]
    got_norm = [tuple(_norm(v) for v in r) for r in got]
    # exact multiset first (fast path), then tolerant matching
    if Counter(exp_norm) == Counter(got_norm):
        return True
    remaining = list(got_norm)
    for row in exp_norm:
        hit = next(
            (i for i, g in enumerate(remaining) if all(_same(x, y) for x, y in zip(row, g, strict=True))),
            None,
        )
        if hit is None:
            return False
        remaining.pop(hit)
    return not remaining


def execution_match(expected: QueryResult, got: QueryResult | None) -> bool:
    if got is None:
        return False
    return rows_match(expected.rows, got.rows)


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def answer_mentions_result(answer: str, result: QueryResult | None) -> bool:
    """True when at least one number from the first result row appears in the answer (rounded forms allowed).

    A cheap faithfulness proxy: a model that answers with numbers absent from the data fails it.
    Rows with no numeric values pass when any string value from the row appears in the answer.
    """
    if result is None or not result.rows:
        return False
    nums_in_answer = {n.replace(",", "") for n in _NUM.findall(answer)}
    first = result.rows[0]
    numeric = [v for v in first if isinstance(v, int | float) and not isinstance(v, bool)]
    if numeric:
        for v in numeric:
            forms = {str(v), f"{v:.0f}", f"{v:.1f}", f"{v:.2f}", f"{float(v):g}"}
            if isinstance(v, float) and v.is_integer():
                forms.add(str(int(v)))
            if forms & nums_in_answer:
                return True
        return False
    strings = [str(v) for v in first if isinstance(v, str)]
    return any(s.lower() in answer.lower() for s in strings)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
