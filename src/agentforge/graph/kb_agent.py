"""KB Agent — RAG-backed Q&A with strict grounding.

Five-layer hallucination defense (chunking, retrieval, threshold, prompt, critic).
"""
from __future__ import annotations

from agentforge.graph.state import GraphState
from agentforge.llm.provider import generate_grounded_answer
from agentforge.rag.retriever import retrieve
from agentforge.tracing import traced


@traced("kb_agent.run")
def run(state: GraphState) -> GraphState:
    query = state["messages"][-1].content

    # Layer 2 + 3: top-k + rerank + similarity threshold
    chunks, citations = retrieve(query)

    if not chunks:
        return {
            "retrieved_chunks": [],
            "citations": [],
            "draft_answer": "I don't have enough information to answer that.",
            "answer_confidence": 0.0,
        }

    # Layer 4: strict prompt — answer only from context
    answer, confidence = generate_grounded_answer(query=query, chunks=chunks)

    return {
        "retrieved_chunks": chunks,
        "citations": citations,
        "draft_answer": answer,
        "answer_confidence": confidence,
    }
