"""Evaluation and ablation runner.

    python -m agentic_rag.eval.run_eval --variant hybrid_rerank
    python -m agentic_rag.eval.run_eval --ablation retrieval
    python -m agentic_rag.eval.run_eval --ablation all --out results/

Writes one JSON file per variant plus a markdown table you paste into the README.
Every artefact carries the config fingerprint and git SHA, so a number in your
README can always be traced back to the exact code and settings that produced it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..graph import Pipeline, answer_question
from ..llm import LLMClient
from .judge import Judge
from .metrics import (
    bootstrap_ci,
    operational_metrics,
    refusal_metrics,
    score_retrieval,
    stage_latency_breakdown,
)

DEFAULT_GOLDEN = "data/golden/golden_set.jsonl"


# --------------------------------------------------------------------------
# Ablation grids
# --------------------------------------------------------------------------

ABLATIONS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "retrieval": [
        ("dense_only", {"retrieval": {"mode": "dense"}}),
        ("bm25_only", {"retrieval": {"mode": "bm25"}}),
        ("hybrid_rrf", {"retrieval": {"mode": "hybrid"}}),
        ("hybrid_rerank", {"retrieval": {"mode": "hybrid_rerank"}}),
    ],
    "chunking": [
        ("chunk_256", {"chunking": {"chunk_size": 256, "chunk_overlap": 32}}),
        ("chunk_512", {"chunking": {"chunk_size": 512, "chunk_overlap": 64}}),
        ("chunk_1024", {"chunking": {"chunk_size": 1024, "chunk_overlap": 128}}),
        ("chunk_512_no_headings", {"chunking": {"chunk_size": 512, "prepend_headings": False}}),
    ],
    "graph": [
        ("no_planner_no_critic", {"graph": {"enable_planner": False, "enable_critic": False}}),
        ("planner_only", {"graph": {"enable_planner": True, "enable_critic": False}}),
        ("critic_only", {"graph": {"enable_planner": False, "enable_critic": True}}),
        ("full_graph", {"graph": {"enable_planner": True, "enable_critic": True}}),
    ],
    "model": [
        ("small_model", {"llm": {"model": "claude-haiku-4-5"}}),
        ("frontier_model", {"llm": {"model": "claude-sonnet-4-5"}}),
    ],
}

# Chunking ablations change the index itself, so they cannot be swapped at query
# time the way the others can.
NEEDS_REINGEST = {"chunking"}


# --------------------------------------------------------------------------
# Golden set
# --------------------------------------------------------------------------


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy data/golden/golden_set.example.jsonl and "
            "write your own -- the golden set is the project."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    _validate_golden(rows)
    return rows


def _validate_golden(rows: list[dict[str, Any]]) -> None:
    problems = []
    ids = set()
    for i, row in enumerate(rows):
        if "id" not in row or "question" not in row:
            problems.append(f"row {i}: needs 'id' and 'question'")
            continue
        if row["id"] in ids:
            problems.append(f"duplicate id: {row['id']}")
        ids.add(row["id"])
        if not row.get("unanswerable") and not row.get("relevant_chunk_ids"):
            problems.append(
                f"{row['id']}: answerable rows need relevant_chunk_ids "
                "(run `make label` to fill them semi-automatically)"
            )
    if problems:
        raise ValueError("golden set validation failed:\n  " + "\n  ".join(problems))

    unanswerable = sum(1 for r in rows if r.get("unanswerable"))
    if unanswerable < max(5, len(rows) // 10):
        print(
            f"  warning: only {unanswerable}/{len(rows)} unanswerable items. "
            "Aim for ~15%; without them you cannot measure hallucination."
        )


# --------------------------------------------------------------------------
# Single variant
# --------------------------------------------------------------------------


def run_variant(
    name: str,
    overrides: dict[str, Any],
    golden: list[dict[str, Any]],
    *,
    use_judge: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    settings = Settings.load(**overrides)
    pipe = Pipeline.build(settings)
    judge = Judge(LLMClient(settings.llm), settings.llm.judge_model) if use_judge else None

    rows: list[dict[str, Any]] = []
    items = golden[:limit] if limit else golden

    print(f"\n=== {name} (fingerprint {settings.fingerprint()}) ===")
    for i, item in enumerate(items, start=1):
        started = time.perf_counter()
        try:
            result = answer_question(item["question"], settings, pipe=pipe)
        except Exception as exc:  # one bad row must not kill a 90-item run
            print(f"  [{i}/{len(items)}] {item['id']}: ERROR {exc}")
            rows.append({"id": item["id"], "error": str(exc), "refused": False})
            continue
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        evidence = result.get("evidence", [])
        row: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "answer": result.get("answer", ""),
            "refused": bool(result.get("refused")),
            "unanswerable": bool(item.get("unanswerable")),
            "retrieved": [e["chunk_id"] for e in evidence],
            "relevant": item.get("relevant_chunk_ids", []),
            "citations": result.get("citations", []),
            "latency_ms": latency_ms,
            "timings": result.get("timings", {}),
            "cost_usd": result.get("usage", {}).get("cost_usd", 0.0),
            "llm_calls": result.get("usage", {}).get("llm_calls", 0),
            "critique": result.get("critique", {}),
            "revisions": result.get("revisions", 0),
        }

        if judge:
            scores = judge.score(
                question=item["question"],
                answer=row["answer"],
                context_blocks=[e["text"] for e in evidence],
                reference=item.get("reference_answer"),
                refused=row["refused"],
            )
            row["judge"] = scores.asdict()
            row["cost_usd"] += scores.judge_cost_usd

        flag = "REFUSED" if row["refused"] else f"{row['latency_ms']:.0f}ms"
        print(f"  [{i}/{len(items)}] {item['id']}: {flag}")
        rows.append(row)

    return {
        "variant": name,
        "overrides": overrides,
        "fingerprint": settings.fingerprint(),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_items": len(rows),
        "retrieval": score_retrieval(rows).asdict(),
        "refusal": refusal_metrics(rows),
        "operational": operational_metrics(rows),
        "stage_latency_ms": stage_latency_breakdown(rows),
        "generation": _aggregate_judge(rows),
        "rows": rows,
    }


def _aggregate_judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("faithfulness", "correctness", "relevance"):
        values = [
            r["judge"][key]
            for r in rows
            if r.get("judge") and r["judge"].get(key) is not None
        ]
        if not values:
            out[key] = None
            continue
        lo, hi = bootstrap_ci(values)
        out[key] = {
            "mean": round(sum(values) / len(values), 4),
            "ci95": [lo, hi],
            "n": len(values),
        }
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def to_markdown(results: list[dict[str, Any]]) -> str:
    header = (
        "| Variant | recall@5 | MRR | nDCG@5 | Faithfulness | Correct refusal | "
        "False refusal | p95 ms | $/1k |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in results:
        ret, ref, ops = r["retrieval"], r["refusal"], r["operational"]
        faith = r["generation"].get("faithfulness")
        faith_cell = f"{faith['mean']:.3f}" if faith else "--"
        lines.append(
            f"| `{r['variant']}` | {ret['recall@5']:.3f} | {ret['mrr']:.3f} | "
            f"{ret['ndcg@5']:.3f} | {faith_cell} | "
            f"{_fmt(ref['correct_refusal_rate'])} | {_fmt(ref['false_refusal_rate'])} | "
            f"{ops['latency_p95_ms']:.0f} | ${ops['cost_per_1k_queries_usd']:.2f} |"
        )
    return header + "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "--"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--variant", help="run one named variant from --ablation group")
    parser.add_argument(
        "--ablation",
        choices=[*ABLATIONS.keys(), "all"],
        help="run a whole ablation grid",
    )
    parser.add_argument("--out", default="results")
    parser.add_argument("--limit", type=int, help="first N golden items (smoke test)")
    parser.add_argument("--no-judge", action="store_true", help="retrieval metrics only, no API spend")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation:
        groups = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
        variants = [(g, name, ov) for g in groups for name, ov in ABLATIONS[g]]
    elif args.variant:
        found = [
            (g, n, ov) for g, vs in ABLATIONS.items() for n, ov in vs if n == args.variant
        ]
        if not found:
            raise SystemExit(f"unknown variant {args.variant}")
        variants = found
    else:
        variants = [("default", "baseline", {})]

    results = []
    for group, name, overrides in variants:
        if group in NEEDS_REINGEST:
            print(
                f"\n  NOTE: '{name}' changes chunking. Re-run "
                f"`make ingest CHUNK_SIZE=...` before trusting these numbers."
            )
        result = run_variant(
            name, overrides, golden, use_judge=not args.no_judge, limit=args.limit
        )
        results.append(result)
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))

    table = to_markdown(results)
    (out_dir / "summary.md").write_text(table)
    print("\n" + table)
    print(f"Wrote {len(results)} result file(s) to {out_dir}/")
    print("Paste summary.md into your README -- do not retype the numbers by hand.")


if __name__ == "__main__":
    main()
