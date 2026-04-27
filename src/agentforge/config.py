"""Configuration & threshold knobs.

Centralizing thresholds here makes it easy to tune in eval and revert in production
without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Supervisor
    supervisor_intent_threshold: float = 0.70

    # Retrieval
    retriever_top_k: int = 8
    retriever_rerank_top_n: int = 4
    retriever_min_similarity: float = 0.62

    # Chunking
    chunking_max_tokens: int = 512
    chunking_overlap: int = 64

    # Critic / fallback
    critic_threshold: float = 0.75

    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai | vertex
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 30

    # Tracing
    otel_service_name: str = "agentforge"
    otel_endpoint: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")


CONFIG = Config()
