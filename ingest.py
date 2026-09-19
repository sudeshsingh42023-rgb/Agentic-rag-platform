"""Corpus loading, chunking and indexing.

Chunking is the highest-leverage and least glamorous part of a RAG system. Two
decisions here move recall more than any model swap:

1. Chunks carry their heading path. On regulations, syllabi and specs, the
   heading is often the only thing that disambiguates otherwise identical text
   ("Section 4.2 Eligibility" vs "Section 9.1 Eligibility").
2. Chunk boundaries respect paragraph structure before falling back to
   character counts, so a definition is rarely split from the term it defines.

Run `make ingest` after dropping files into data/corpus/.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import ChunkingConfig, EmbeddingConfig, Settings, StoreConfig

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".html", ".htm"}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    heading_path: str = ""
    page: int | None = None
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def embedding_text(self) -> str:
        if self.heading_path:
            return f"{self.heading_path}\n\n{self.text}"
        return self.text

    def citation(self) -> str:
        name = Path(self.source_path).name
        if self.page is not None:
            return f"{name} p.{self.page}"
        if self.heading_path:
            return f"{name} > {self.heading_path}"
        return name


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class RawDoc:
    doc_id: str
    text: str
    source_path: str
    pages: list[tuple[int, str]] | None = None


def load_corpus(corpus_dir: str | Path) -> list[RawDoc]:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    docs: list[RawDoc] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        doc_id = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = _read_pdf(path)
            text = "\n\n".join(t for _, t in pages)
            docs.append(RawDoc(doc_id, text, str(path), pages))
        elif suffix in {".html", ".htm"}:
            docs.append(RawDoc(doc_id, _read_html(path), str(path)))
        else:
            docs.append(RawDoc(doc_id, path.read_text(errors="ignore"), str(path)))
    if not docs:
        raise ValueError(f"no supported documents under {corpus_dir}")
    return docs


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install pypdf") from exc
    reader = pypdf.PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append((i, text))
    return out


def _read_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install beautifulsoup4") from exc
    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

_HEADING = re.compile(
    r"^\s*(?:(#{1,6})\s+(.+)"                       # markdown
    r"|((?:\d+\.)+\d*)\s+([A-Z][^\n]{3,80})"        # 4.2.1 Numbered Heading
    r"|([A-Z][A-Z \-]{6,80}))\s*$",                 # ALL CAPS HEADING
    re.MULTILINE,
)


def chunk_document(doc: RawDoc, cfg: ChunkingConfig) -> list[Chunk]:
    if doc.pages:
        segments = [(text, page) for page, text in doc.pages]
    else:
        segments = [(doc.text, None)]

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []

    for text, page in segments:
        for block in _split_blocks(text, cfg):
            level, title = _heading_of(block)
            if level is not None:
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                # A bare heading with no body is not worth indexing on its own.
                if len(block.strip()) <= len(title) + 8:
                    continue

            heading_path = " > ".join(t for _, t in heading_stack[-3:])
            ordinal = len(chunks)
            chunk_id = f"{doc.doc_id}-{ordinal:05d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=block.strip(),
                    source_path=doc.source_path,
                    heading_path=heading_path if cfg.prepend_headings else "",
                    page=page,
                    ordinal=ordinal,
                )
            )
    return chunks


def _heading_of(block: str) -> tuple[int | None, str]:
    first_line = block.strip().split("\n", 1)[0]
    m = _HEADING.match(first_line)
    if not m:
        return None, ""
    if m.group(1):
        return len(m.group(1)), m.group(2).strip()
    if m.group(3):
        return m.group(3).count(".") + 1, m.group(4).strip()
    return 1, m.group(5).strip()


def _split_blocks(text: str, cfg: ChunkingConfig) -> Iterable[str]:
    """Pack paragraphs up to chunk_size, with a sliding overlap tail."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    buffer: list[str] = []
    size = 0

    for para in paragraphs:
        para = para.strip()
        # A paragraph longer than the budget gets hard-split on sentences.
        if len(para) > cfg.chunk_size * 1.5:
            if buffer:
                yield "\n\n".join(buffer)
                buffer, size = [], 0
            yield from _split_sentences(para, cfg.chunk_size)
            continue

        if size + len(para) > cfg.chunk_size and buffer:
            block = "\n\n".join(buffer)
            yield block
            tail = block[-cfg.chunk_overlap :] if cfg.chunk_overlap else ""
            buffer = [tail, para] if tail else [para]
            size = len(tail) + len(para)
        else:
            buffer.append(para)
            size += len(para)

    if buffer:
        yield "\n\n".join(buffer)


def _split_sentences(para: str, limit: int) -> Iterable[str]:
    sentences = re.split(r"(?<=[.!?])\s+", para)
    buffer: list[str] = []
    size = 0
    for sentence in sentences:
        if size + len(sentence) > limit and buffer:
            yield " ".join(buffer)
            buffer, size = [], 0
        buffer.append(sentence)
        size += len(sentence)
    if buffer:
        yield " ".join(buffer)


# --------------------------------------------------------------------------
# Embedding + storage
# --------------------------------------------------------------------------


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.cfg.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.model)
        elif self.cfg.provider == "openai":
            from openai import OpenAI

            self._model = OpenAI()
        else:
            raise ValueError(f"unknown embedding provider {self.cfg.provider}")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        if self.cfg.provider == "openai":
            out = []
            for i in range(0, len(texts), self.cfg.batch_size):
                batch = texts[i : i + self.cfg.batch_size]
                resp = model.embeddings.create(model=self.cfg.model, input=batch)
                out.extend(d.embedding for d in resp.data)
            return out
        return model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        if self.cfg.provider == "openai":
            resp = model.embeddings.create(model=self.cfg.model, input=[text])
            return resp.data[0].embedding
        return model.encode(
            [self.cfg.query_prefix + text], normalize_embeddings=True
        ).tolist()[0]


def build_index(settings: Settings, corpus_dir: str | Path) -> dict[str, Any]:
    """Chunk the corpus, embed it, and persist both the vector and BM25 indexes."""
    docs = load_corpus(corpus_dir)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, settings.chunking))

    embedder = Embedder(settings.embedding)
    vectors = embedder.embed_documents([c.embedding_text() for c in chunks])

    _persist_vectors(settings.store, chunks, vectors)
    _persist_chunks(settings.store, chunks)

    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks), 1),
        "fingerprint": settings.fingerprint(),
    }


def _persist_vectors(cfg: StoreConfig, chunks: list[Chunk], vectors) -> None:
    if cfg.backend == "chroma":
        import chromadb

        client = chromadb.PersistentClient(path=cfg.path)
        try:
            client.delete_collection(cfg.collection)
        except Exception:
            pass
        col = client.create_collection(cfg.collection, metadata={"hnsw:space": "cosine"})
        step = 500
        for i in range(0, len(chunks), step):
            window = chunks[i : i + step]
            col.add(
                ids=[c.chunk_id for c in window],
                embeddings=vectors[i : i + step],
                documents=[c.text for c in window],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "source_path": c.source_path,
                        "heading_path": c.heading_path,
                        "page": c.page if c.page is not None else -1,
                    }
                    for c in window
                ],
            )
    elif cfg.backend == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = QdrantClient(url=cfg.qdrant_url)
        client.recreate_collection(
            cfg.collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        client.upsert(
            cfg.collection,
            points=[
                PointStruct(id=i, vector=v, payload=asdict(c))
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ],
        )
    else:
        raise ValueError(f"unknown store backend {cfg.backend}")


def _persist_chunks(cfg: StoreConfig, chunks: list[Chunk]) -> None:
    """Chunk text is kept on disk too: BM25 needs the raw corpus, and the eval
    harness needs to resolve a chunk_id back to its source without the vector DB."""
    path = Path(cfg.path).parent / "chunks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk)) + "\n")


def load_chunks(cfg: StoreConfig) -> list[Chunk]:
    path = Path(cfg.path).parent / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make ingest` first")
    return [Chunk(**json.loads(line)) for line in path.read_text().splitlines() if line]
