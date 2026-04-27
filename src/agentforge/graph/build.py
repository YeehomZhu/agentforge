"""Wire the LangGraph: supervisor → sub-agents → critic → commit | fallback."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agentforge.fallback.template import clarify, fallback_template
from agentforge.graph import code_agent, critic, kb_agent, ops_agent, supervisor
from agentforge.graph.state import GraphState


def build_graph():
    g = StateGraph(GraphState)

    # Nodes
    g.add_node("supervisor", supervisor.classify)
    g.add_node("kb_agent", kb_agent.run)
    g.add_node("code_agent", code_agent.run)
    g.add_node("ops_agent", ops_agent.run)
    g.add_node("critic", critic.critique)
    g.add_node("commit", lambda s: {"final_answer": s["draft_answer"]})
    g.add_node("fallback", fallback_template)
    g.add_node("fallback_clarify", clarify)

    # Edges
    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        supervisor.route,
        {
            "kb_agent": "kb_agent",
            "code_agent": "code_agent",
            "ops_agent": "ops_agent",
            "fallback_clarify": "fallback_clarify",
        },
    )
    for sub in ("kb_agent", "code_agent", "ops_agent"):
        g.add_edge(sub, "critic")

    g.add_conditional_edges(
        "critic",
        critic.route_after_critic,
        {"commit": "commit", "fallback": "fallback"},
    )
    g.add_edge("commit", END)
    g.add_edge("fallback", END)
    g.add_edge("fallback_clarify", END)

    return g.compile()
