"""Runs every arm of the ablation and prints the markdown table you paste into
the README. This table is the project. Everything else is plumbing.

    python -m eval.ablation --no-judge          # retrieval arms only (free)
    python -m eval.ablation                     # full, costs LLM calls

Arms:
  retrieval strategy : dense | bm25 | hybrid | hybrid_rerank
  chunk size         : 256 | 512 | 1024   (requires re-ingest, see --chunks)
  critic             : on | off
  model tier         : strong | cheap
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import run
from src.rag.config import get_settings

STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def md_table(rows: list[dict]) -> str:
    head = ("| arm | recall@5 | MRR | nDCG@5 | faithfulness | correct refusal | "
            "hallucination | p95 ms |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        g = r.get("generation") or {}
        lines.append(
            f"| {r['run_name']} | {r['retrieval']['recall@5']} | {r['retrieval']['mrr']} | "
            f"{r['retrieval']['ndcg@5']} | {g.get('faithfulness', '-')} | "
            f"{r['refusal']['correct_refusal_rate']} | {r['refusal']['hallucination_rate']} | "
            f"{r['latency']['p95_ms']} |")
    return "\n".join(lines)


def reingest(chunk_size: int) -> None:
    env = dict(os.environ, CHUNK_SIZE=str(chunk_size))
    subprocess.run([sys.executable, "-m", "scripts.ingest"], env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--chunks", action="store_true",
                    help="also sweep chunk size (re-ingests between arms)")
    ap.add_argument("--critic", action="store_true", help="also sweep critic on/off")
    a = ap.parse_args()
    use_judge = not a.no_judge
    results = []

    for strat in STRATEGIES:
        print(f"\n### arm: strategy={strat}")
        results.append(run(strat, use_judge, None, None, f"strategy={strat}"))

    if a.chunks:
        for cs in (256, 512, 1024):
            print(f"\n### arm: chunk_size={cs}")
            reingest(cs)
            get_settings.cache_clear()
            results.append(run("hybrid_rerank", use_judge, None, None, f"chunk={cs}"))
        reingest(512)
        get_settings.cache_clear()

    if a.critic:
        for flag in ("true", "false"):
            print(f"\n### arm: critic={flag}")
            os.environ["USE_CRITIC"] = flag
            get_settings.cache_clear()
            results.append(run("hybrid_rerank", use_judge, None, None, f"critic={flag}"))

    print("\n\nPaste this into README.md:\n")
    print(md_table(results))
    Path("eval/results").mkdir(exist_ok=True)
    Path(f"eval/results/ablation_{int(time.time())}.md").write_text(md_table(results))


if __name__ == "__main__":
    main()
