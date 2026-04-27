"""Stub agents kept minimal — full implementation left as exercise.

The point of this skeleton is to demonstrate the *topology* and *contracts*
between Supervisor → sub-agents → Critic. Code/Ops agents follow the same
pattern as KB Agent (call provider, return state delta).
"""
from __future__ import annotations

from agentforge.graph.state import GraphState


def run(state: GraphState) -> GraphState:
    # TODO: tool registry call + ReAct loop + structured output.
    return {
        "draft_answer": "[code_agent stub]",
        "answer_confidence": 0.5,
        "retrieved_chunks": [],
        "citations": [],
    }
