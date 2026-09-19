from agentic_rag.ingest import Chunk
from agentic_rag.retrieval import BM25Retriever, reciprocal_rank_fusion, ScoredChunk


def _chunk(cid, text):
    return Chunk(chunk_id=cid, doc_id="d1", text=text, source_path="x.txt")


def test_bm25_ranks_exact_term_match_first():
    chunks = [
        _chunk("c1", "The eligibility criteria require a minimum CGPA of 6.0"),
        _chunk("c2", "Interviews are conducted in person at the nodal college"),
    ]
    bm25 = BM25Retriever(chunks)
    results = bm25.search("minimum CGPA eligibility", top_k=2)
    assert results[0].chunk.chunk_id == "c1"


def test_bm25_returns_nothing_for_unrelated_query():
    chunks = [_chunk("c1", "The eligibility criteria require a minimum CGPA")]
    bm25 = BM25Retriever(chunks)
    assert bm25.search("quantum entanglement photosynthesis", top_k=5) == []


def test_rrf_favours_items_ranked_high_in_multiple_runs():
    c1, c2, c3 = _chunk("c1", "a"), _chunk("c2", "b"), _chunk("c3", "c")
    run1 = [ScoredChunk(c1, 1.0, 1, "bm25"), ScoredChunk(c2, 0.9, 2, "bm25")]
    run2 = [ScoredChunk(c1, 0.8, 1, "dense"), ScoredChunk(c3, 0.7, 2, "dense")]
    fused = reciprocal_rank_fusion([run1, run2], k=60, top_k=3)
    assert fused[0].chunk.chunk_id == "c1"  # ranked #1 in both runs
