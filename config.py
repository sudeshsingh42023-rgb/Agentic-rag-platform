"""Central config. Everything tunable lives here so ablations are one env var away."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "echo"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    model_strong: str = "claude-sonnet-4-5"
    model_cheap: str = "claude-haiku-4-5"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    vector_backend: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "corpus"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5
    rrf_k: int = 60

    use_reranker: bool = True
    use_critic: bool = True
    max_critic_revisions: int = 2

    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "agentic-rag"

    index_dir: str = "data/index"
    corpus_dir: str = "data/corpus"


@lru_cache
def get_settings() -> Settings:
    return Settings()
