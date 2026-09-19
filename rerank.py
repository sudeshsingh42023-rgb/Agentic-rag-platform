"""Cross-encoder reranking. Measured separately in the ablation because it is
usually the single largest contributor to p95 latency."""
from .config import get_settings
from . import telemetry as tel

_ce = None


def _get_cross_encoder():
    global _ce
    if _ce is not None:
        return _ce
    try:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder(get_settings().reranker_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[rerank] cross-encoder unavailable ({exc}); passthrough")
        _ce = False
    return _ce


def rerank(query: str, hits):
    ce = _get_cross_encoder()
    if not ce:
        return hits
    with tel.span("rag.rerank", **{tel.RAG_QUERY: query, "rag.rerank.candidates": len(hits)}):
        pairs = [(query, h.chunk.embed_text) for h in hits]
        scores = ce.predict(pairs)
        for h, sc in zip(hits, scores):
            h.score = float(sc)
            h.source = "hybrid_rerank"
        return sorted(hits, key=lambda h: -h.score)
