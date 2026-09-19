"""FastAPI surface. /ask is the product; /healthz is what a load balancer hits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.config import get_settings
from src.rag.graph import RAGPipeline
from src.rag.store import HybridStore

app = FastAPI(title="Agentic RAG Platform", version="0.1.0")
_pipeline: RAGPipeline | None = None


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: str | None = Field(default=None,
                                 description="dense | bm25 | hybrid | hybrid_rerank")
    model: str | None = None


@app.on_event("startup")
def _startup():
    global _pipeline
    s = get_settings()
    try:
        _pipeline = RAGPipeline(HybridStore.load(s.index_dir))
    except FileNotFoundError:
        _pipeline = None
        print(f"[api] no index at {s.index_dir}; run `python -m scripts.ingest`")


@app.get("/healthz")
def healthz():
    return {"ok": True, "index_loaded": _pipeline is not None}


@app.post("/ask")
def ask(req: AskRequest):
    if _pipeline is None:
        raise HTTPException(503, "index not built - run `python -m scripts.ingest`")
    pipe = (RAGPipeline(_pipeline.store, strategy=req.strategy, model=req.model)
            if (req.strategy or req.model) else _pipeline)
    return pipe.run(req.query)
