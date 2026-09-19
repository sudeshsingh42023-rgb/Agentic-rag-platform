"""Hybrid retrieval: BM25 (lexical) + dense vectors, fused with Reciprocal Rank
Fusion, optionally reranked with a cross-encoder.

RRF is used instead of score normalisation because BM25 scores and cosine
similarities are not on a comparable scale; fusing on *rank* sidesteps that.
    RRF(d) = sum over retrievers of 1 / (k + rank_r(d))
"""
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .config import get_settings
from .embeddings import embed
from . import telemetry as tel


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str = "hybrid"


class HybridStore:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: list[Chunk] = chunks or []
        self.vectors: np.ndarray | None = None
        self._bm25 = None

    # ---------- build ----------
    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectors = embed([c.embed_text for c in chunks])
        self._build_bm25()

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            corpus = [c.embed_text.lower().split() for c in self.chunks]
            self._bm25 = BM25Okapi(corpus)
        except Exception as exc:  # noqa: BLE001
            print(f"[store] BM25 unavailable ({exc}); dense-only")
            self._bm25 = None

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "vectors.npy"), self.vectors)
        with open(os.path.join(path, "chunks.jsonl"), "w") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str) -> "HybridStore":
        chunks = []
        with open(os.path.join(path, "chunks.jsonl")) as fh:
            for line in fh:
                chunks.append(Chunk(**json.loads(line)))
        store = cls(chunks)
        store.vectors = np.load(os.path.join(path, "vectors.npy"))
        store._build_bm25()
        return store

    # ---------- retrieval ----------
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        qv = embed([query])[0]
        sims = self.vectors @ qv
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def search(self, query: str, strategy: str | None = None) -> list[Hit]:
        """strategy: dense | bm25 | hybrid | hybrid_rerank (default from settings)"""
        s = get_settings()
        strategy = strategy or ("hybrid_rerank" if s.use_reranker else "hybrid")

        with tel.span("rag.retrieve", **{tel.RAG_QUERY: query, tel.RAG_STRATEGY: strategy}) as sp:
            if strategy == "dense":
                ranked = self.dense_search(query, s.top_k_final)
            elif strategy == "bm25":
                ranked = self.bm25_search(query, s.top_k_final)
            else:
                dense = self.dense_search(query, s.top_k_dense)
                lex = self.bm25_search(query, s.top_k_bm25)
                ranked = _rrf([dense, lex], k=s.rrf_k)[: s.top_k_dense]

            hits = [Hit(self.chunks[i], sc, strategy) for i, sc in ranked]

            if strategy == "hybrid_rerank":
                from .rerank import rerank
                hits = rerank(query, hits)

            hits = hits[: s.top_k_final]
            sp.set_attribute(tel.RAG_CHUNKS_RETURNED, len(hits))
            sp.set_attribute(tel.RAG_TOP_SCORE, hits[0].score if hits else 0.0)
            sp.set_attribute(tel.RAG_CHUNK_IDS, ",".join(h.chunk.id for h in hits))
            return hits


def _rrf(rank_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for rl in rank_lists:
        for rank, (idx, _score) in enumerate(rl):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])
