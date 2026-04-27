"""LLM provider — swap between OpenAI, Vertex AI (Gemini), or local via config.

Keeping provider behind an interface makes it trivial to swap to Vertex AI / Gemini
or to add Anthropic / Bedrock later without touching agent logic.
"""
from __future__ import annotations

from typing import Any

from agentforge.config import CONFIG
from agentforge.graph.state import Intent

# ---------------------------------------------------------------------------
# Public API used by the agents — agent code never imports the SDK directly.
# ---------------------------------------------------------------------------


def classify_intent(user_message: str) -> tuple[Intent, float]:
    """Return (intent_label, confidence)."""
    response = _structured_call(
        prompt_name="supervisor_classify",
        variables={"user_message": user_message},
        schema={
            "type": "object",
            "properties": {
                "intent": {"enum": ["kb_query", "code_task", "ops_action", "unclear"]},
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
        },
    )
    return response["intent"], float(response["confidence"])


def generate_grounded_answer(query: str, chunks: list[dict]) -> tuple[str, float]:
    response = _structured_call(
        prompt_name="kb_agent_grounded",
        variables={"query": query, "chunks": chunks},
        schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        },
    )
    return response["answer"], float(response["confidence"])


def score_answer(question: str, draft_answer: str, evidence: list[dict]) -> dict[str, float]:
    return _structured_call(
        prompt_name="critic_score",
        variables={"question": question, "answer": draft_answer, "evidence": evidence},
        schema={
            "type": "object",
            "properties": {
                "factuality": {"type": "number"},
                "groundedness": {"type": "number"},
                "coverage": {"type": "number"},
            },
            "required": ["factuality", "groundedness", "coverage"],
        },
    )


# ---------------------------------------------------------------------------
# Internal: provider-specific dispatch
# ---------------------------------------------------------------------------


def _structured_call(prompt_name: str, variables: dict, schema: dict) -> dict[str, Any]:
    if CONFIG.llm_provider == "openai":
        return _openai_call(prompt_name, variables, schema)
    if CONFIG.llm_provider == "vertex":
        return _vertex_call(prompt_name, variables, schema)
    raise ValueError(f"Unknown LLM provider: {CONFIG.llm_provider}")


def _openai_call(prompt_name: str, variables: dict, schema: dict) -> dict[str, Any]:
    # TODO: load prompt from prompts/<name>.md, render with variables,
    # call openai client with response_format=json_schema.
    raise NotImplementedError("wire OpenAI client here")


def _vertex_call(prompt_name: str, variables: dict, schema: dict) -> dict[str, Any]:
    # TODO: vertexai.generative_models.GenerativeModel("gemini-1.5-pro")
    # with response_schema=schema for structured output.
    raise NotImplementedError("wire Vertex AI / Gemini client here")
