"""Tests for LLM-native metrics: latency, tokens/sec, cost-per-request."""
from __future__ import annotations

import time

from agentforge.metrics import (
    PRICING_USD_PER_1K_TOKENS,
    LLMMetrics,
    measure_llm_call,
)


def test_measure_llm_call_records_latency_and_cost() -> None:
    with measure_llm_call(model="gpt-4o-mini") as m:
        time.sleep(0.01)  # simulate LLM call
        m.set_usage(prompt_tokens=1000, completion_tokens=500)

    assert m.prompt_tokens == 1000
    assert m.completion_tokens == 500
    assert m.latency_ms >= 10
    # gpt-4o-mini pricing: 0.00015 prompt + 0.0006 completion per 1k tokens
    expected_cost = 1.0 * 0.00015 + 0.5 * 0.0006
    assert abs(m.cost_usd - expected_cost) < 1e-9
    assert m.tokens_per_sec > 0


def test_measure_llm_call_unknown_model_returns_zero_cost() -> None:
    with measure_llm_call(model="some-future-model") as m:
        m.set_usage(prompt_tokens=10, completion_tokens=10)
    assert m.cost_usd == 0.0


def test_pricing_table_has_both_openai_and_gemini() -> None:
    """Smoke test — provider-agnostic pricing must cover both stacks."""
    assert "gpt-4o-mini" in PRICING_USD_PER_1K_TOKENS
    assert "gemini-2.0-flash" in PRICING_USD_PER_1K_TOKENS


def test_summary_emits_valid_json() -> None:
    import json

    m = LLMMetrics(model="gpt-4o-mini", conversation_id="abc")
    m.set_usage(100, 50)
    parsed = json.loads(m.summary())
    assert parsed["model"] == "gpt-4o-mini"
    assert parsed["prompt_tokens"] == 100
