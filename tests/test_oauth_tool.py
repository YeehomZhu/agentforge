"""Tests for OAuth tool client: scope checks, expiry, refresh failure paths."""
from __future__ import annotations

import time

import pytest

from agentforge.oauth_tool import (
    OAuthToken,
    OAuthToolClient,
    ReAuthRequired,
    _InMemoryTokenStore,
)


def _fresh_token(scopes: tuple[str, ...] = ("calendar.readonly",)) -> OAuthToken:
    return OAuthToken(
        access_token="at-123",
        refresh_token="rt-123",
        expires_at_epoch=time.time() + 3600,
        scopes=scopes,
    )


def test_returns_cached_token_when_valid() -> None:
    store = _InMemoryTokenStore()
    store.put("user-1", _fresh_token())
    client = OAuthToolClient(store=store)

    assert client.get_access_token("user-1", ("calendar.readonly",)) == "at-123"


def test_no_token_raises_reauth() -> None:
    client = OAuthToolClient(store=_InMemoryTokenStore())
    with pytest.raises(ReAuthRequired, match="No token cached"):
        client.get_access_token("user-1", ("calendar.readonly",))


def test_missing_scope_raises_reauth_and_does_not_silently_broaden() -> None:
    store = _InMemoryTokenStore()
    store.put("user-1", _fresh_token(scopes=("calendar.readonly",)))
    client = OAuthToolClient(store=store)

    with pytest.raises(ReAuthRequired, match="missing scopes"):
        client.get_access_token("user-1", ("calendar.readwrite",))


def test_expired_token_with_no_refresh_token_raises_reauth() -> None:
    store = _InMemoryTokenStore()
    store.put(
        "user-1",
        OAuthToken(
            access_token="at-old",
            refresh_token=None,
            expires_at_epoch=time.time() - 10,
            scopes=("calendar.readonly",),
        ),
    )
    client = OAuthToolClient(store=store)

    with pytest.raises(ReAuthRequired, match="No refresh_token"):
        client.get_access_token("user-1", ("calendar.readonly",))

    # Token should have been evicted.
    assert store.get("user-1") is None
