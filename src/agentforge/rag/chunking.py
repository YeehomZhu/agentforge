"""Semantic-boundary chunker.

Splits markdown by heading hierarchy (h1/h2/h3) and code-block boundaries first;
falls back to fixed-size + overlap when no structural signal is present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    doc_id: str
    section_path: str
    chunk_index: int
    text: str
    token_count: int


def _approx_token_count(text: str) -> int:
    # rough: 1 token ≈ 4 chars
    return max(1, len(text) // 4)


def chunk_markdown(
    doc_id: str,
    text: str,
    max_tokens: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Split a markdown document on heading boundaries, then size-pack."""
    sections: list[tuple[str, str]] = []
    last_idx = 0
    last_path = doc_id

    for m in HEADING_RE.finditer(text):
        if m.start() > last_idx:
            sections.append((last_path, text[last_idx : m.start()].strip()))
        last_path = f"{doc_id}#{m.group(2).strip()}"
        last_idx = m.end()

    if last_idx < len(text):
        sections.append((last_path, text[last_idx:].strip()))

    chunks: list[Chunk] = []
    idx = 0
    for path, body in sections:
        if not body:
            continue
        for part in _size_pack(body, max_tokens, overlap):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    section_path=path,
                    chunk_index=idx,
                    text=part,
                    token_count=_approx_token_count(part),
                )
            )
            idx += 1
    return chunks


def _size_pack(text: str, max_tokens: int, overlap: int) -> list[str]:
    if _approx_token_count(text) <= max_tokens:
        return [text]
    char_window = max_tokens * 4
    char_overlap = overlap * 4
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + char_window)
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - char_overlap
    return out
