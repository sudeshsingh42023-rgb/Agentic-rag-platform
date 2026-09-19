"""LLM-as-judge with explicit rubrics and structured output.

Two design choices worth defending in an interview:

1. One judge per dimension. Faithfulness, correctness and relevance are scored
   by separate calls with separate rubrics. A single blended score hides the
   most interesting failure -- an answer that is perfectly grounded in the
   retrieved context but does not address the question.

2. Structured JSON out, with claim-level decomposition for faithfulness. Free
   scoring drifts between runs; a fixed schema makes the result aggregable and
   the variance measurable. `self_consistency` quantifies what is left.

Known limits, state them rather than hide them: judges favour verbose answers,
favour their own family's output, and are unreliable on domain questions the
judge model itself would get wrong. Spot-check 20% against your own reading.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from ..llm import LLMClient, LLMError
from ..prompts import (
    CORRECTNESS_PROMPT,
    CORRECTNESS_SYSTEM,
    FAITHFULNESS_PROMPT,
    FAITHFULNESS_SYSTEM,
    RELEVANCE_PROMPT,
    RELEVANCE_SYSTEM,
)


@dataclass
class JudgeScores:
    faithfulness: float | None = None
    correctness: float | None = None
    relevance: float | None = None
    unsupported_claims: list[str] | None = None
    missing_facts: list[str] | None = None
    judge_cost_usd: float = 0.0
    errors: list[str] | None = None

    def asdict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "correctness": self.correctness,
            "relevance": self.relevance,
            "unsupported_claims": self.unsupported_claims or [],
            "missing_facts": self.missing_facts or [],
            "judge_cost_usd": round(self.judge_cost_usd, 6),
            "errors": self.errors or [],
        }


class Judge:
    def __init__(self, llm: LLMClient, model: str):
        self.llm = llm
        self.model = model

    def score(
        self,
        *,
        question: str,
        answer: str,
        context_blocks: list[str],
        reference: str | None = None,
        refused: bool = False,
    ) -> JudgeScores:
        scores = JudgeScores(errors=[])
        cost = 0.0

        # A refusal is not a quality failure -- it is judged by the refusal
        # metrics, not by the rubric judges. Scoring it here would punish the
        # exact behaviour the critic exists to produce.
        if refused:
            scores.faithfulness = None
            scores.relevance = None
            if reference is not None:
                scores.correctness = 0.0
            return scores

        context = "\n\n".join(
            f"[{i}] {c}" for i, c in enumerate(context_blocks, start=1)
        )

        try:
            payload, resp = self.llm.complete_json(
                FAITHFULNESS_PROMPT.format(context=context, answer=answer),
                system=FAITHFULNESS_SYSTEM,
                model=self.model,
                max_tokens=1200,
            )
            cost += resp.cost_usd()
            total = payload.get("total_count") or len(payload.get("claims", [])) or 0
            supported = payload.get("supported_count")
            if supported is None:
                supported = sum(1 for c in payload.get("claims", []) if c.get("supported"))
            scores.faithfulness = round(supported / total, 4) if total else None
            scores.unsupported_claims = [
                c["claim"] for c in payload.get("claims", []) if not c.get("supported")
            ]
        except (LLMError, KeyError, TypeError, ZeroDivisionError) as exc:
            scores.errors.append(f"faithfulness: {exc}")

        try:
            payload, resp = self.llm.complete_json(
                RELEVANCE_PROMPT.format(question=question, answer=answer),
                system=RELEVANCE_SYSTEM,
                model=self.model,
                max_tokens=300,
            )
            cost += resp.cost_usd()
            scores.relevance = round(float(payload["score"]) / 5.0, 4)
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            scores.errors.append(f"relevance: {exc}")

        if reference:
            try:
                payload, resp = self.llm.complete_json(
                    CORRECTNESS_PROMPT.format(
                        question=question, reference=reference, generated=answer
                    ),
                    system=CORRECTNESS_SYSTEM,
                    model=self.model,
                    max_tokens=600,
                )
                cost += resp.cost_usd()
                scores.correctness = round(float(payload["score"]) / 5.0, 4)
                scores.missing_facts = payload.get("missing", [])
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                scores.errors.append(f"correctness: {exc}")

        scores.judge_cost_usd = cost
        return scores

    def self_consistency(
        self, *, question: str, answer: str, context_blocks: list[str], runs: int = 3
    ) -> dict[str, Any]:
        """Score the same item n times and report the spread.

        Run this once on ~20 items and put the number in your README. If the
        judge's standard deviation is comparable to the gap between two of your
        pipeline variants, that comparison proves nothing -- and knowing that is
        worth more than any single metric in the table.
        """
        values = []
        for _ in range(runs):
            score = self.score(
                question=question, answer=answer, context_blocks=context_blocks
            )
            if score.faithfulness is not None:
                values.append(score.faithfulness)
        if len(values) < 2:
            return {"runs": len(values), "stdev": None, "values": values}
        return {
            "runs": len(values),
            "mean": round(statistics.fmean(values), 4),
            "stdev": round(statistics.stdev(values), 4),
            "spread": round(max(values) - min(values), 4),
            "values": values,
        }
