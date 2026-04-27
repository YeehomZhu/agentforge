"""Tests for chunking — semantic boundaries preserved + fixed-size fallback works."""
from __future__ import annotations

from agentforge.rag.chunking import chunk_markdown


def test_chunks_split_on_headings() -> None:
    text = """# Title
intro paragraph.

## Section A
content of A.

## Section B
content of B.
"""
    chunks = chunk_markdown(doc_id="doc1", text=text, max_tokens=512, overlap=64)
    paths = [c.section_path for c in chunks]
    assert any("Section A" in p for p in paths)
    assert any("Section B" in p for p in paths)


def test_falls_back_to_size_pack_when_no_headings() -> None:
    text = "a" * 5000  # ~1250 tokens, no headings
    chunks = chunk_markdown(doc_id="doc2", text=text, max_tokens=200, overlap=20)
    assert len(chunks) > 1
    assert all(c.token_count <= 250 for c in chunks)


def test_metadata_carried() -> None:
    text = "# T\n\n## S\nbody"
    chunks = chunk_markdown(doc_id="doc3", text=text)
    for c in chunks:
        assert c.doc_id == "doc3"
        assert c.token_count > 0
