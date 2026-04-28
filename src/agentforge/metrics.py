"""LLM-native metrics: tokens/sec, cost-per-request, latency budgeting.

Why this exists:
    Production agents are judged on three numbers customers actually feel —
    tokens/sec (perceived speed), cost-per-request (unit economics), and
    p95 latency (tail experience). This module captures all three around any
    LLM call without coupling to the SDK.

Usage:
    with measure_llm_call(model="gpt-4o-mini") as m:
        response = client.chat.completions.create(...)
        m.set_usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
    print(m.summary())
"""
from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from agentforge.tracing import get_conversation_id, tracer

# Pricing in USD per 1K tokens. Keep this table boring and updatable.
# Numbers are approximations — production code should pull from a config service.
PRICING_USD_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    # model: (prompt, completion)
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gemini-2.0-flash": (0.00010, 0.0004),
    "gemini-2.0-pro": (0.00125, 0.005),
    "gemini-1.5-pro": (0.00125, 0.005),
}


@dataclass
class LLMMetrics:
    model: str
    conversation_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens_per_sec: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)

    def set_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def _finalize(self) -> None:
        self.cost_usd = _estimate_cost(
            self.model, self.prompt_tokens, self.completion_tokens
        )
        if self.latency_ms > 0 and self.completion_tokens > 0:
            self.tokens_per_sec = self.completion_tokens / (self.latency_ms / 1000.0)

    def summary(self) -> str:
        return json.dumps(asdict(self), indent=2)


@contextlib.contextmanager
def measure_llm_call(model: str) -> Iterator[LLMMetrics]:
    """Context manager: time an LLM call and emit a metrics record.

    The metrics are also attached as attributes on the active OTel span,
    so a `conversation_id`-keyed trace carries token/cost/latency end-to-end.
    """
    metrics = LLMMetrics(model=model, conversation_id=get_conversation_id())
    start = time.perf_counter()
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("conversation_id", metrics.conversation_id)
        try:
            yield metrics
        finally:
            metrics.latency_ms = (time.perf_counter() - start) * 1000.0
            metrics._finalize()
            span.set_attribute("llm.prompt_tokens", metrics.prompt_tokens)
            span.set_attribute("llm.completion_tokens", metrics.completion_tokens)
            span.set_attribute("llm.latency_ms", metrics.latency_ms)
            span.set_attribute("llm.cost_usd", metrics.cost_usd)
            span.set_attribute("llm.tokens_per_sec", metrics.tokens_per_sec)


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in PRICING_USD_PER_1K_TOKENS:
        return 0.0
    prompt_rate, completion_rate = PRICING_USD_PER_1K_TOKENS[model]
    return (prompt_tokens / 1000.0) * prompt_rate + (
        completion_tokens / 1000.0
    ) * completion_rate
