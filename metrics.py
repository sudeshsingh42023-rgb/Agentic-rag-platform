"""Retrieval and generation metrics.

Scored separately and on purpose. A blended "accuracy" number cannot distinguish
"the retriever never found the right chunk" from "the retriever found it and the
model ignored it", and those two failures have opposite fixes. Nearly every RAG
project that stalls, stalls because it only measured the final answer.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant chunks that appear in the top k."""
    relevant = set(relevant)
    if not relevant:
        return float("nan")  # unanswerable items have no gold chunks; excluded upstream
    hits = len(relevant & set(retrieved[:k]))
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(relevant) & set(retrieved[:k])) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Did at least one relevant chunk make the cut? The metric that best
    predicts whether the generator has any chance."""
    return 1.0 if set(relevant) & set(retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    for i, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG. Rewards putting the right chunk first, not merely
    somewhere in the window -- which matters once a reranker is in play."""
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, cid in enumerate(retrieved[:k], start=1)
        if cid in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class RetrievalScores:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    hit_rate_at_5: float
    mrr: float
    ndcg_at_5: float
    n: int

    def asdict(self) -> dict[str, Any]:
        return {
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "hit_rate@5": round(self.hit_rate_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "n": self.n,
        }


def score_retrieval(rows: list[dict[str, Any]]) -> RetrievalScores:
    """`rows` = [{"retrieved": [chunk_id...], "relevant": [chunk_id...]}, ...]

    Rows with no gold chunks (the unanswerable set) are skipped here and scored
    by `refusal_metrics` instead.
    """
    usable = [r for r in rows if r.get("relevant")]
    if not usable:
        return RetrievalScores(*([0.0] * 6), n=0)

    def mean(fn) -> float:
        values = [fn(r) for r in usable]
        values = [v for v in values if not math.isnan(v)]
        return statistics.fmean(values) if values else 0.0

    return RetrievalScores(
        recall_at_1=mean(lambda r: recall_at_k(r["retrieved"], r["relevant"], 1)),
        recall_at_3=mean(lambda r: recall_at_k(r["retrieved"], r["relevant"], 3)),
        recall_at_5=mean(lambda r: recall_at_k(r["retrieved"], r["relevant"], 5)),
        hit_rate_at_5=mean(lambda r: hit_rate_at_k(r["retrieved"], r["relevant"], 5)),
        mrr=mean(lambda r: reciprocal_rank(r["retrieved"], r["relevant"])),
        ndcg_at_5=mean(lambda r: ndcg_at_k(r["retrieved"], r["relevant"], 5)),
        n=len(usable),
    )


# --------------------------------------------------------------------------
# Refusal behaviour
# --------------------------------------------------------------------------


def refusal_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The unanswerable slice.

    `correct_refusal_rate` is the headline: of the questions the corpus genuinely
    cannot answer, how many did the system decline rather than invent?

    `false_refusal_rate` is the honest counterweight. A system that refuses
    everything scores a perfect 1.0 above and is useless. Always report both.
    """
    unanswerable = [r for r in rows if r.get("unanswerable")]
    answerable = [r for r in rows if not r.get("unanswerable")]

    correct = sum(1 for r in unanswerable if r.get("refused"))
    false_ref = sum(1 for r in answerable if r.get("refused"))
    return {
        "correct_refusal_rate": round(correct / len(unanswerable), 4) if unanswerable else None,
        "false_refusal_rate": round(false_ref / len(answerable), 4) if answerable else None,
        "hallucination_on_unanswerable": (
            round((len(unanswerable) - correct) / len(unanswerable), 4)
            if unanswerable else None
        ),
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
    }


# --------------------------------------------------------------------------
# Cost / latency
# --------------------------------------------------------------------------


def operational_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    costs = [r.get("cost_usd", 0.0) for r in rows]
    calls = [r.get("llm_calls", 0) for r in rows]

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(int(q * len(ordered)), len(ordered) - 1)
        return round(ordered[idx], 1)

    return {
        "latency_p50_ms": pct(latencies, 0.50),
        "latency_p95_ms": pct(latencies, 0.95),
        "cost_per_query_usd": round(statistics.fmean(costs), 6) if costs else 0.0,
        "cost_per_1k_queries_usd": round(statistics.fmean(costs) * 1000, 3) if costs else 0.0,
        "avg_llm_calls": round(statistics.fmean(calls), 2) if calls else 0.0,
    }


def stage_latency_breakdown(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean milliseconds per stage.

    Run this before you optimise anything. The reranker is usually the surprise.
    """
    buckets: dict[str, list[float]] = {}
    for row in rows:
        for stage, ms in (row.get("timings") or {}).items():
            buckets.setdefault(stage, []).append(ms)
    return {k: round(statistics.fmean(v), 1) for k, v in sorted(buckets.items())}


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float], iterations: int = 2000, alpha: float = 0.05, seed: int = 7
) -> tuple[float, float]:
    """95% confidence interval by bootstrap resampling.

    On a 90-question golden set, a 3-point difference in recall is noise. Report
    the interval so nobody -- including you -- over-reads a small gap.
    """
    import random

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int((alpha / 2) * iterations)]
    hi = means[min(int((1 - alpha / 2) * iterations), iterations - 1)]
    return (round(lo, 4), round(hi, 4))
