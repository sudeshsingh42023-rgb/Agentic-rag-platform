from agentic_rag.config import ChunkingConfig
from agentic_rag.ingest import RawDoc, chunk_document


def test_chunk_document_respects_size_budget():
    text = ("This is a sentence about eligibility criteria. " * 40)
    doc = RawDoc(doc_id="d1", text=text, source_path="x.txt")
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=200, chunk_overlap=20))
    assert len(chunks) > 1
    assert all(len(c.text) < 400 for c in chunks)  # generous ceiling for overlap


def test_chunk_document_captures_heading_path():
    text = "## Eligibility\n\nStudents need a minimum CGPA of 6.0 to apply for this role."
    doc = RawDoc(doc_id="d1", text=text, source_path="x.md")
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=500, prepend_headings=True))
    assert any("Eligibility" in c.heading_path for c in chunks)


def test_empty_heading_only_block_is_dropped():
    text = "## Section 1\n\n## Section 2\n\nActual content goes here about the role."
    doc = RawDoc(doc_id="d1", text=text, source_path="x.md")
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=500))
    assert all(len(c.text) > 15 for c in chunks)
