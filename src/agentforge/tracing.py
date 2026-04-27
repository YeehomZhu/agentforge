"""OpenTelemetry tracing with conversation_id propagation.

Every node and tool call is wrapped in a span carrying the `conversation_id` so a
single end-to-end trace can be reconstructed across processes/microservices.
"""
from __future__ import annotations

import contextvars
import functools
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from agentforge.config import CONFIG

_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "conversation_id", default=""
)

_provider = TracerProvider(
    resource=Resource.create({"service.name": CONFIG.otel_service_name})
)
_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer(__name__)


def new_conversation_id() -> str:
    cid = str(uuid.uuid4())
    _conversation_id.set(cid)
    return cid


def get_conversation_id() -> str:
    return _conversation_id.get()


F = TypeVar("F", bound=Callable[..., Any])


def traced(span_name: str) -> Callable[[F], F]:
    """Decorator: wrap a function in an OTel span tagged with conversation_id."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("conversation_id", get_conversation_id())
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
