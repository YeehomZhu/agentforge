"""Ops agent — side-effecting actions (HTTP POST / mutations).

Requires explicit `confirm=true` flag for write actions (HITL gate).
"""
from __future__ import annotations

from agentforge.graph.state import GraphState


def run(state: GraphState) -> GraphState:
    # TODO: HITL gate → action executor → result.
    return {
        "draft_answer": "[ops_agent stub — HITL gate would intercept here]",
        "answer_confidence": 0.5,
        "retrieved_chunks": [],
        "citations": [],
    }
