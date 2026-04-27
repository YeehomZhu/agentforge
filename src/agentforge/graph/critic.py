"""Critic — self-reflection scorer.

Scores draft answer against retrieved evidence on factuality, groundedness, coverage.
Triggers fallback when aggregate score is below threshold.
"""
from __future__ import annotations

from agentforge.config import CONFIG
from agentforge.graph.state import CriticScore, GraphState
from agentforge.llm.provider import score_answer
from agentforge.tracing import traced


@traced("critic.score")
def critique(state: GraphState) -> GraphState:
    score = score_answer(
        question=state["messages"][-1].content,
        draft_answer=state.get("draft_answer", ""),
        evidence=state.get("retrieved_chunks", []),
    )
    aggregate = (score["factuality"] + score["groundedness"] + score["coverage"]) / 3
    full_score: CriticScore = {**score, "aggregate": aggregate}
    return {"critic_score": full_score}


def route_after_critic(state: GraphState) -> str:
    aggregate = state["critic_score"]["aggregate"]
    if aggregate < CONFIG.critic_threshold:
        return "fallback"
    return "commit"
