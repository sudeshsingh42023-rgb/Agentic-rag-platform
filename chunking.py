"""Chunking with overlap that preserves the source heading of each chunk.

Heading preservation is deliberate context engineering: a chunk reading
"the limit is 30 days" is useless without "4.2 Claim Intimation" above it.
"""
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    heading: str = ""
    ordinal: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def embed_text(self) -> str:
        return f"{self.heading}\n\n{self.text}".strip() if self.heading else self.text


MD_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 .,&\-']{6,}$")
NUM_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][^.]*$")


def is_heading(line: str) -> bool:
    """A heading is short, title-like and does not read as a sentence.

    The subtle failure this guards against: clause numbering. A line like
    "2.2 Pre-existing diseases are covered after 36 months." looks like a
    numbered heading to a naive regex, gets swallowed as one, and its text
    vanishes from the chunk body - silently deleting the very fact the
    retriever needs. Hence the word cap and the no-trailing-period rule.
    """
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if MD_HEADING_RE.match(line):
        return True
    if line.endswith("."):
        return False
    if len(line.split()) > 10:
        return False
    return bool(CAPS_HEADING_RE.match(line) or NUM_HEADING_RE.match(line))


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def split_document(doc_id: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    buf: list[str] = []
    heading = ""
    ordinal = 0

    def flush():
        nonlocal buf, ordinal
        body = "\n".join(buf).strip()
        if body:
            chunks.append(
                Chunk(id=f"{doc_id}::{ordinal}", doc_id=doc_id, text=body,
                      heading=heading, ordinal=ordinal)
            )
            ordinal += 1
        buf = []

    for line in lines:
        stripped = line.strip()
        if is_heading(stripped):
            flush()
            heading = stripped.lstrip("#").strip()
            continue
        buf.append(line)
        if _approx_tokens("\n".join(buf)) >= chunk_size:
            tail = buf[-max(1, overlap // 40):]
            flush()
            buf = list(tail)
    flush()
    return chunks


def load_corpus(corpus_dir: str) -> Iterator[tuple[str, str]]:
    for p in sorted(Path(corpus_dir).rglob("*")):
        if p.suffix.lower() in {".txt", ".md"}:
            yield p.stem, p.read_text(encoding="utf-8", errors="ignore")
