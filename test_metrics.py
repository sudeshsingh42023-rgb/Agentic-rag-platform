"""Metrics must be correct before they're trusted in a README table."""
import math
from agentic_rag.eval.metrics import (
    recall_at_k, hit_rate_at_k, reciprocal_rank, ndcg_at_k,
    score_retrieval, refusal_metrics, bootstrap_ci,
)


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5


def test_recall_at_k_respects_k():
    assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_hit_rate():
    assert hit_rate_at_k(["a", "b"], ["z", "b"], 5) == 1.0
    assert hit_rate_at_k(["a", "b"], ["z"], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "y"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_perfect_order():
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_ndcg_reversed_order_is_lower():
    perfect = ndcg_at_k(["a", "b"], ["a", "b"], 2)
    reversed_ = ndcg_at_k(["b", "a"], ["a", "b"], 2)
    assert reversed_ <= perfect


def test_score_retrieval_skips_unanswerable_rows():
    rows = [
        {"retrieved": ["a"], "relevant": ["a"]},
        {"retrieved": ["z"], "relevant": []},  # unanswerable -> excluded
    ]
    result = score_retrieval(rows)
    assert result.n == 1
    assert result.recall_at_5 == 1.0


def test_refusal_metrics():
    rows = [
        {"unanswerable": True, "refused": True},
        {"unanswerable": True, "refused": False},
        {"unanswerable": False, "refused": False},
        {"unanswerable": False, "refused": True},
    ]
    m = refusal_metrics(rows)
    assert m["correct_refusal_rate"] == 0.5
    assert m["false_refusal_rate"] == 0.5


def test_bootstrap_ci_bounds_the_mean():
    values = [0.8, 0.82, 0.79, 0.81, 0.80]
    lo, hi = bootstrap_ci(values)
    assert lo <= sum(values) / len(values) <= hi
