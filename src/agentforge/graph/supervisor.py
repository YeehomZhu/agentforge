"""Supervisor agent — intent classification + hierarchical delegation.

ReAct-style: classify intent → delegate to one sub-agent → consolidate output.
Falls back to a clarifying question when intent confidence is below threshold.
"""
from __future__ import annotations

from agentforge.config import CONFIG
from agentforge.graph.state import GraphState
from agentforge.llm.provider import classify_intent
from agentforge.tracing import traced


@traced("supervisor.classify_intent")
def classify(state: GraphState) -> GraphState:
    user_message = state["messages"][-1].content
    intent, confidence = classify_intent(user_message)
    return {
        "intent": intent,
        "intent_confidence": confidence,
    }


def route(state: GraphState) -> str:
    """Conditional edge: which sub-agent gets the work."""
    if state["intent_confidence"] < CONFIG.supervisor_intent_threshold:
        return "fallback_clarify"

    return {
        "kb_query": "kb_agent",
        "code_task": "code_agent",
        "ops_action": "ops_agent",
        "unclear": "fallback_clarify",
    }[state["intent"]]
