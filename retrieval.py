"""Retrieval: BM25, dense vectors, reciprocal rank fusion, cross-encoder rerank.

All four modes share one interface so the ablation runner can swap between them
by name. This is what produces the table in the README:

    dense            -- vectors only, the tutorial baseline
    bm25             -- lexical only, surprisingly strong on acronyms and IDs
    hybrid           -- both, fused with RRF
    hybrid_rerank    -- both, fused, then reordered by a cross-encoder

Dense retrieval alone loses on exact identifiers ("SP 800-53r5", "Section 4.2b")
because embeddings smear rare tokens. BM25 alone loses on paraphrase. The fusion
is not a trick to look sophisticated -- it is covering two different failure modes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from . import tracing
from .config import RetrievalConfig, Settings
from .ingest import Chunk, Embedder, load_chunks


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int
    source: str  # which retriever produced it: bm25 | dense | rrf | rerank


class Retriever(Protocol):
    name: str

    def search(self, query: str, top_k: int) -> list[ScoredChunk]: ...


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on",
    "at", "by", "with", "as", "be", "it", "this", "that", "from", "was", "were",
}


def tokenize(text: str) -> list[str]:
    # Identifiers like "800-53r5" survive as one token; that is deliberate.
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Retriever:
    """Okapi BM25. Implemented directly -- it is 40 lines and removes a dependency."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.embedding_text()) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(len(self.docs), 1)

        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in self.docs:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.tf.append(counts)
            for token in counts:
                self.df[token] = self.df.get(token, 0) + 1

        self.N = len(self.docs)
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in self.df.items()
        }

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        for i, counts in enumerate(self.tf):
            score = 0.0
            for term in terms:
                f = counts.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-9)
                )
                score += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return [
            ScoredChunk(self.chunks[i], s, rank, "bm25")
            for rank, (i, s) in enumerate(scores[:top_k], start=1)
        ]


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------


class DenseRetriever:
    name = "dense"

    def __init__(self, settings: Settings, chunks: list[Chunk]):
        self.settings = settings
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embedder = Embedder(settings.embedding)
        self._collection = None

    def _col(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=self.settings.store.path)
            self._collection = client.get_collection(self.settings.store.collection)
        return self._collection

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        vector = self.embedder.embed_query(query)
        result = self._col().query(query_embeddings=[vector], n_results=top_k)
        ids = result["ids"][0]
        distances = result.get("distances", [[0.0] * len(ids)])[0]
        out = []
        for rank, (cid, dist) in enumerate(zip(ids, distances), start=1):
            chunk = self.by_id.get(cid)
            if chunk is None:
                continue  # index and chunks.jsonl are out of sync -> re-ingest
            out.append(ScoredChunk(chunk, 1.0 - float(dist), rank, "dense"))
        return out


# --------------------------------------------------------------------------
# Fusion + rerank
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    runs: list[list[ScoredChunk]], k: int = 60, top_k: int = 20
) -> list[ScoredChunk]:
    """RRF: score = sum over runs of 1 / (k + rank).

    Rank-based rather than score-based, so it needs no score normalisation
    between a BM25 score (unbounded) and a cosine similarity (0-1). That is the
    entire reason it is the default fusion method in hybrid search.
    """
    accum: dict[str, float] = {}
    keep: dict[str, Chunk] = {}
    for run in runs:
        for item in run:
            cid = item.chunk.chunk_id
            accum[cid] = accum.get(cid, 0.0) + 1.0 / (k + item.rank)
            keep[cid] = item.chunk
    ordered = sorted(accum.items(), key=lambda x: -x[1])[:top_k]
    return [
        ScoredChunk(keep[cid], score, rank, "rrf")
        for rank, (cid, score) in enumerate(ordered, start=1)
    ]


class CrossEncoderReranker:
    """Reorders candidates by scoring (query, chunk) pairs jointly.

    Far more accurate than bi-encoder similarity and far too slow to run over the
    whole corpus -- which is exactly why it goes last, over ~20 candidates.
    Expect it to dominate your p95 latency; measure it before you defend it.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int,
        min_score: float | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.chunk.embedding_text()) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
        out = []
        for rank, (item, score) in enumerate(ranked[:top_k], start=1):
            if min_score is not None and float(score) < min_score:
                break  # honest low-confidence cut -> lets the graph refuse
            out.append(ScoredChunk(item.chunk, float(score), rank, "rerank"))
        return out


# --------------------------------------------------------------------------
# Facade
# --------------------------------------------------------------------------


class HybridRetriever:
    """One object, four strategies, selected by `settings.retrieval.mode`."""

    def __init__(self, settings: Settings, chunks: list[Chunk] | None = None):
        self.settings = settings
        self.cfg: RetrievalConfig = settings.retrieval
        self.chunks = chunks if chunks is not None else load_chunks(settings.store)
        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(settings, self.chunks)
        self.reranker = (
            CrossEncoderReranker(self.cfg.reranker_model)
            if self.cfg.mode == "hybrid_rerank"
            else None
        )

    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or self.cfg.top_k_final
        mode = self.cfg.mode

        with tracing.span("rag.retrieve", **{"rag.retriever.strategy": mode}) as sp:
            if mode == "bm25":
                results = self.bm25.search(query, top_k)
            elif mode == "dense":
                results = self.dense.search(query, top_k)
            else:
                with tracing.span("rag.retrieve.bm25"):
                    lexical = self.bm25.search(query, self.cfg.top_k_bm25)
                with tracing.span("rag.retrieve.dense"):
                    vectors = self.dense.search(query, self.cfg.top_k_dense)
                fused = reciprocal_rank_fusion(
                    [lexical, vectors],
                    k=self.cfg.rrf_k,
                    top_k=max(self.cfg.top_k_dense, self.cfg.top_k_bm25),
                )
                if mode == "hybrid_rerank" and self.reranker:
                    with tracing.span("rag.rerank", **{"rag.rerank.candidates": len(fused)}):
                        results = self.reranker.rerank(
                            query, fused, top_k, self.cfg.min_rerank_score
                        )
                else:
                    results = fused[:top_k]

            tracing.record_retrieval(
                sp,
                strategy=mode,
                query=query,
                chunk_ids=[r.chunk.chunk_id for r in results],
                scores=[r.score for r in results],
                top_k=top_k,
            )
        return results
