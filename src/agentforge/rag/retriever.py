"""Retriever — top-k + rerank + similarity threshold.

Returns empty list when nothing crosses the threshold — KB agent then declines
rather than fabricating an answer.
"""
from __future__ import annotations

from typing import Any

from agentforge.config import CONFIG
from agentforge.graph.state import Citation
from agentforge.tracing import traced


@traced("rag.retrieve")
def retrieve(query: str) -> tuple[list[dict[str, Any]], list[Citation]]:
    """Retrieve relevant chunks with rerank + threshold gating.

    Returns ([], []) when no chunk crosses `retriever_min_similarity`.
    """
    # Implementation: vector store similarity search → cross-encoder rerank
    # Pseudocode for the demo skeleton:
    raw_hits = _vector_search(query, k=CONFIG.retriever_top_k)
    reranked = _cross_encoder_rerank(query, raw_hits)[: CONFIG.retriever_rerank_top_n]
    gated = [h for h in reranked if h["similarity"] >= CONFIG.retriever_min_similarity]

    citations: list[Citation] = [
        {
            "doc_id": h["doc_id"],
            "section_path": h["section_path"],
            "similarity": h["similarity"],
        }
        for h in gated
    ]
    return gated, citations


def _vector_search(query: str, k: int) -> list[dict[str, Any]]:
    # TODO: wire FAISS index built by build_index.py
    raise NotImplementedError("hook FAISS index here")


def _cross_encoder_rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # TODO: cross-encoder/ms-marco-MiniLM-L-6-v2 or similar
    raise NotImplementedError("hook cross-encoder reranker here")
