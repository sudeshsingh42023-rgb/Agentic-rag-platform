"""Command line entry point: ingest, ask, label, serve."""

from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .graph import Pipeline, answer_question
from .ingest import build_index


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.load()
    if args.chunk_size:
        settings.chunking.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        settings.chunking.chunk_overlap = args.chunk_overlap
    stats = build_index(settings, args.corpus)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.load()
    if args.mode:
        settings.retrieval.mode = args.mode
    result = answer_question(args.question, settings)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print("\n" + result.get("answer", ""))
    if result.get("citations"):
        print("\nSources:")
        for c in dict.fromkeys(result["citations"]):
            print(f"  - {c}")
    critique = result.get("critique") or {}
    if critique.get("verdict") not in (None, "skipped"):
        print(f"\nCritic: {critique['verdict']} (revisions: {result.get('revisions', 0)})")
        for issue in critique.get("issues", []):
            print(f"  ! {issue}")
    usage = result.get("usage", {})
    timings = result.get("timings", {})
    print(
        f"\n{usage.get('llm_calls', 0)} LLM calls, "
        f"${usage.get('cost_usd', 0):.5f}, {timings.get('total_ms', 0):.0f}ms total"
    )
    if args.verbose and timings:
        for stage, ms in sorted(timings.items(), key=lambda x: -x[1]):
            print(f"    {stage:<16} {ms:>8.1f}ms")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    """Semi-automatic gold labelling.

    Retrieves top-k for each question and prints the candidates so you can pick
    the relevant chunk ids. Deliberately NOT automatic: if you label the golden
    set using the retriever you are evaluating, your recall number is circular
    and meaningless. A human picks; the tool only saves typing.
    """
    settings = Settings.load()
    pipe = Pipeline.build(settings)
    rows = [json.loads(l) for l in open(args.golden) if l.strip()]

    for row in rows:
        if row.get("relevant_chunk_ids") or row.get("unanswerable"):
            continue
        print("\n" + "=" * 70)
        print(f"{row['id']}: {row['question']}")
        for hit in pipe.retriever.search(row["question"], top_k=args.top_k):
            preview = hit.chunk.text[:220].replace("\n", " ")
            print(f"\n  [{hit.chunk.chunk_id}] score={hit.score:.3f} {hit.chunk.citation()}")
            print(f"      {preview}...")
        print("\n  Paste the relevant chunk ids into relevant_chunk_ids, or set")
        print("  \"unanswerable\": true if none of these actually answer it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk, embed and index the corpus")
    p_ingest.add_argument("--corpus", default="data/corpus")
    p_ingest.add_argument("--chunk-size", type=int)
    p_ingest.add_argument("--chunk-overlap", type=int)
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=["dense", "bm25", "hybrid", "hybrid_rerank"])
    p_ask.add_argument("--json", action="store_true")
    p_ask.add_argument("--verbose", "-v", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_label = sub.add_parser("label", help="help label gold chunk ids")
    p_label.add_argument("--golden", default="data/golden/golden_set.jsonl")
    p_label.add_argument("--top-k", type=int, default=8)
    p_label.set_defaults(func=cmd_label)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
