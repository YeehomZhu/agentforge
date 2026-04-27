"""Deterministic fallback paths.

Two flavors:
- `fallback_template`: when critic scores below threshold — return a safe degraded answer.
- `clarify`: when supervisor intent confidence is low — ask a clarifying question.
"""
from __future__ import annotations

from agentforge.graph.state import GraphState


def fallback_template(state: GraphState) -> GraphState:
    """Below-threshold confidence → degrade to a deterministic hedged answer."""
    citations = state.get("citations", [])
    if citations:
        sources = ", ".join(c["section_path"] for c in citations[:3])
        msg = (
            "I'm not confident enough to give a direct answer. "
            f"Most relevant references I found: {sources}. "
            "Please review these or rephrase your question."
        )
    else:
        msg = "I don't have enough information to answer that confidently. Could you rephrase or add context?"

    return {
        "final_answer": msg,
        "fallback_triggered": True,
        "fallback_reason": "critic_below_threshold",
    }


def clarify(state: GraphState) -> GraphState:
    return {
        "final_answer": (
            "I want to make sure I understand. Are you asking about "
            "(a) information from our knowledge base, "
            "(b) running code or a query, or "
            "(c) performing an action / API call? "
            "A short clarification will help."
        ),
        "fallback_triggered": True,
        "fallback_reason": "low_intent_confidence",
    }
