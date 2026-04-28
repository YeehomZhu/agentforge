"""OAuth 2.0 tool calling — per-user token cache + auto-refresh + scope hygiene.

Why this exists:
    When an agent calls a third-party API on behalf of a user (Microsoft Graph,
    Google Workspace, Salesforce, etc.), it must use that user's OAuth token —
    never a service-account token, never another user's token. This module
    sketches the production pattern:

        1. Per-user token cache (encrypted at rest in real deployments).
        2. Automatic refresh before expiry, with a small safety margin.
        3. Forced re-auth if refresh_token is invalid or revoked.
        4. Scope minimization — only request what the tool needs.
        5. Fully traced — every token operation lands on the conversation_id span.

Production hardening (NOT in this reference):
    - Replace _InMemoryTokenStore with a vault-backed store (KMS / Secret Manager).
    - Sign refresh requests from a backend, never from agent code in untrusted env.
    - Rotate client_secret regularly and use short-lived tokens.
    - Audit every token use — who, when, which scope, which tool, which conversation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from agentforge.tracing import get_conversation_id, traced, tracer

# Refresh access_token this many seconds before nominal expiry.
REFRESH_SAFETY_MARGIN_SEC = 60


@dataclass
class OAuthToken:
    access_token: str
    refresh_token: str | None
    expires_at_epoch: float
    scopes: tuple[str, ...] = field(default_factory=tuple)

    def is_expired(self, now: float | None = None) -> bool:
        return (now or time.time()) >= (
            self.expires_at_epoch - REFRESH_SAFETY_MARGIN_SEC
        )


class TokenStore(Protocol):
    """Pluggable storage. Production = vault; tests = in-memory."""

    def get(self, user_id: str) -> OAuthToken | None: ...
    def put(self, user_id: str, token: OAuthToken) -> None: ...
    def delete(self, user_id: str) -> None: ...


class _InMemoryTokenStore:
    """Reference store for local development & tests. Never use in production."""

    def __init__(self) -> None:
        self._tokens: dict[str, OAuthToken] = {}

    def get(self, user_id: str) -> OAuthToken | None:
        return self._tokens.get(user_id)

    def put(self, user_id: str, token: OAuthToken) -> None:
        self._tokens[user_id] = token

    def delete(self, user_id: str) -> None:
        self._tokens.pop(user_id, None)


class ReAuthRequired(Exception):
    """Refresh failed; the user must re-authorize via the OAuth flow."""


class OAuthToolClient:
    """Provides a valid access_token for a (user, required_scopes) pair.

    The agent calls `get_access_token(user_id, required_scopes)` right before
    invoking a third-party tool. All scope-checking, refresh, and error handling
    is centralized here so tool code stays a thin HTTP wrapper.
    """

    def __init__(self, store: TokenStore | None = None) -> None:
        self._store = store or _InMemoryTokenStore()

    @traced("oauth.get_access_token")
    def get_access_token(
        self, user_id: str, required_scopes: tuple[str, ...]
    ) -> str:
        token = self._store.get(user_id)
        if token is None:
            raise ReAuthRequired(f"No token cached for user={user_id}")

        # Scope minimization check — never silently broaden.
        missing = set(required_scopes) - set(token.scopes)
        if missing:
            raise ReAuthRequired(
                f"Token for user={user_id} missing scopes: {sorted(missing)}"
            )

        if token.is_expired():
            token = self._refresh(user_id, token)

        return token.access_token

    @traced("oauth.refresh")
    def _refresh(self, user_id: str, token: OAuthToken) -> OAuthToken:
        if not token.refresh_token:
            self._store.delete(user_id)
            raise ReAuthRequired(f"No refresh_token for user={user_id}")

        # In production: POST to the IdP token endpoint with client_id /
        # client_secret / refresh_token. Here we stub the network call so the
        # module is testable without a live IdP.
        new_token = self._call_token_endpoint(token.refresh_token, token.scopes)
        if new_token is None:
            self._store.delete(user_id)
            raise ReAuthRequired(f"Refresh rejected for user={user_id}")

        self._store.put(user_id, new_token)
        with tracer.start_as_current_span("oauth.refresh.success") as span:
            span.set_attribute("conversation_id", get_conversation_id())
            span.set_attribute("oauth.user_id", user_id)
            span.set_attribute("oauth.expires_in_sec", new_token.expires_at_epoch - time.time())
        return new_token

    def _call_token_endpoint(
        self, refresh_token: str, scopes: tuple[str, ...]
    ) -> OAuthToken | None:
        """Stub — wire to real IdP (Entra ID, Google, Okta, ...) in production."""
        raise NotImplementedError(
            "Wire to identity provider's /token endpoint. "
            "See README → 'OAuth tool calling' for the full sequence."
        )


# ---------------------------------------------------------------------------
# Tiny example tool that uses the client. Real tools live in their own modules.
# ---------------------------------------------------------------------------


@traced("tool.user_calendar")
def list_user_calendar_events(
    client: OAuthToolClient, user_id: str
) -> list[dict]:
    """Example tool: fetch the user's next 10 calendar events.

    Demonstrates: required-scope declaration, token retrieval, traced call.
    Real implementation would `httpx.get(...)` the calendar API with the token.
    """
    required_scopes = ("calendar.readonly",)
    _access_token = client.get_access_token(user_id, required_scopes)
    raise NotImplementedError(
        "Wire to calendar API (Microsoft Graph / Google Calendar). "
        "Token would be sent as `Authorization: Bearer <_access_token>`."
    )
