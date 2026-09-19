"""OpenTelemetry instrumentation using the GenAI semantic conventions (gen_ai.*).

Why this file exists: a RAG pipeline fails silently. A stale index returns zero
chunks, the LLM answers from parametric memory, and the HTTP status is still 200.
Span-level retrieval attributes are the only way to tell a retrieval miss from a
generation failure after the fact.

Exports over OTLP/HTTP, so it works unchanged against Langfuse, Phoenix, Jaeger,
Grafana Tempo or Datadog. Do not bind the instrumentation layer to a vendor.
"""
from contextlib import contextmanager
from typing import Any

from .config import get_settings

_tracer = None


class _NoopSpan:
    def set_attribute(self, *_a, **_k): ...
    def set_attributes(self, *_a, **_k): ...
    def record_exception(self, *_a, **_k): ...


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, *_a, **_k):
        yield _NoopSpan()


def _get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    s = get_settings()
    if not s.otel_enabled:
        _tracer = _NoopTracer()
        return _tracer
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": s.otel_service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{s.otel_exporter_otlp_endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)
    return _tracer


@contextmanager
def span(name: str, **attributes: Any):
    with _get_tracer().start_as_current_span(name) as sp:
        for k, v in attributes.items():
            if v is not None:
                sp.set_attribute(k, v)
        yield sp


# --- GenAI semantic convention attribute names (OTel semconv v1.37+) ---
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# --- retrieval attributes. No standard rag.* namespace exists yet, so these are
# ours; keep the names stable so dashboards do not silently break. ---
RAG_QUERY = "rag.query"
RAG_STRATEGY = "rag.retriever.strategy"
RAG_CHUNKS_RETURNED = "rag.chunks.returned"
RAG_TOP_SCORE = "rag.top_score"
RAG_CHUNK_IDS = "rag.chunk_ids"
