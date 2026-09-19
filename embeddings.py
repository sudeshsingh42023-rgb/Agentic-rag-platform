"""Embedding backend. Falls back to a deterministic hashing embedder when
sentence-transformers is unavailable, so the pipeline and tests run offline."""
import hashlib
import numpy as np

from .config import get_settings

_model = None


class _HashingEmbedder:
    """Deterministic offline stand-in. Not semantically good - it exists so the
    graph, eval harness and tests are runnable without downloading weights."""
    dim = 384

    def encode(self, texts, normalize_embeddings=True, **_):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.clip(norms, 1e-9, None)
        return out


def get_embedder():
    global _model
    if _model is not None:
        return _model
    s = get_settings()
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(s.embedding_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[embeddings] falling back to hashing embedder ({exc})")
        _model = _HashingEmbedder()
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(get_embedder().encode(texts, normalize_embeddings=True), dtype=np.float32)
