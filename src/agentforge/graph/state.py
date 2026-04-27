"""Shared graph state for LangGraph.

Keeping the state contract small and explicit prevents accidental coupling between
sub-agents — each agent reads/writes only the fields it owns.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

Intent = Literal["kb_query", "code_task", "ops_action", "unclear"]


class Citation(TypedDict):
    doc_id: str
    section_path: str
    similarity: float


class CriticScore(TypedDict):
    factuality: float
    groundedness: float
    coverage: float
    aggregate: float


class GraphState(TypedDict, total=False):
    # Conversation
    conversation_id: str
    messages: Annotated[list, add_messages]

    # Supervisor outputs
    intent: Intent
    intent_confidence: float

    # KB Agent outputs
    retrieved_chunks: list[dict]
    citations: list[Citation]
    draft_answer: str
    answer_confidence: float

    # Critic outputs
    critic_score: CriticScore

    # Final
    final_answer: str
    fallback_triggered: bool
    fallback_reason: str | None
