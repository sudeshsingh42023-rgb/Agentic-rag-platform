"""FastAPI service.

Two endpoints that matter: /ask returns the answer *plus* the evidence, the
critique and the cost. Hiding that behind a clean response object is how RAG
systems become unfalsifiable. /healthz checks the index is actually loadable,
not just that the process is alive -- a served-but-empty index is the classic
silent failure.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .graph import Pipeline, answer_question

app = FastAPI(title="Agentic RAG", version="0.1.0")

_settings: Settings | None = None
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _settings, _pipeline
    if _pipeline is None:
        _settings = Settings.load()
        _pipeline = Pipeline.build(_settings)
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    include_evidence: bool = True


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[str]
    critique: dict[str, Any]
    usage: dict[str, Any]
    timings: dict[str, float]
    guardrail_findings: list[dict[str, Any]]
    evidence: list[dict[str, Any]] | None = None


@app.on_event("startup")
def _warm() -> None:
    if not os.getenv("RAG_LAZY_START"):
        get_pipeline()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    try:
        pipe = get_pipeline()
        return {
            "status": "ok",
            "chunks_indexed": len(pipe.retriever.chunks),
            "retrieval_mode": pipe.settings.retrieval.mode,
            "fingerprint": pipe.settings.fingerprint(),
        }
    except Exception as exc:
        raise HTTPException(503, f"index not ready: {exc}") from exc


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    pipe = get_pipeline()
    result = answer_question(req.question, pipe.settings, pipe=pipe)
    return AskResponse(
        answer=result.get("answer", ""),
        refused=bool(result.get("refused")),
        citations=result.get("citations", []),
        critique=result.get("critique", {}),
        usage=result.get("usage", {}),
        timings=result.get("timings", {}),
        guardrail_findings=result.get("guardrail_findings", []),
        evidence=result.get("evidence") if req.include_evidence else None,
    )
