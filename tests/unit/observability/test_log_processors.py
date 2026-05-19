"""Unit tests for atm.observability.log_processors.

Tests cover:
1. Flat dict: api_key/token/password/secret values are redacted, others left intact
2. Nested dict: secrets at multiple depths are redacted
3. List of dicts: each dict's secrets are redacted
4. Mixed-case keys (API_Key, Token, passWORD) are matched case-insensitively
5. No-op for clean event_dict (no secret keys)
6. Cyclic dict (self-referencing): does not infinite-loop, returns safely
7. configure_structlog() is idempotent (call twice without raising)
8. Bootstrap respects ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1 env var
9. filter_secrets does NOT mutate the input dict
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def _call_filter(event_dict: dict[str, Any]) -> dict[str, Any]:
    """Invoke filter_secrets with dummy logger/method arguments."""
    from atm.observability.log_processors import filter_secrets

    return filter_secrets(None, "info", event_dict)


_REDACTED = "<redacted>"


def test_flat_dict_redacts_secret_keys() -> None:
    """Values whose keys match secret patterns are replaced with <redacted>."""
    event: dict[str, Any] = {"api_key": "my-secret", "name": "alice", "event": "login"}
    result = _call_filter(event)
    assert result["api_key"] == _REDACTED
    assert result["name"] == "alice"
    assert result["event"] == "login"


def test_flat_dict_redacts_token() -> None:
    event: dict[str, Any] = {"token": "tok_xyz", "user": "bob"}
    result = _call_filter(event)
    assert result["token"] == _REDACTED
    assert result["user"] == "bob"


def test_flat_dict_redacts_password() -> None:
    event: dict[str, Any] = {"password": "hunter2", "action": "auth"}
    result = _call_filter(event)
    assert result["password"] == _REDACTED
    assert result["action"] == "auth"


def test_flat_dict_redacts_secret() -> None:
    event: dict[str, Any] = {"secret": "shh", "level": "info"}
    result = _call_filter(event)
    assert result["secret"] == _REDACTED
    assert result["level"] == "info"


def test_nested_dict_secrets_redacted() -> None:
    """Secrets nested inside another dict are recursively redacted."""
    event: dict[str, Any] = {
        "config": {
            "api_key": "nested-secret",
            "host": "localhost",
        },
        "msg": "startup",
    }
    result = _call_filter(event)
    assert result["config"]["api_key"] == _REDACTED
    assert result["config"]["host"] == "localhost"
    assert result["msg"] == "startup"


def test_deeply_nested_dict_secrets_redacted() -> None:
    """Secrets at multiple depths are all redacted."""
    event: dict[str, Any] = {
        "outer": {
            "inner": {
                "password": "deep-secret",
                "ok": True,
            },
            "token": "mid-token",
        },
    }
    result = _call_filter(event)
    assert result["outer"]["inner"]["password"] == _REDACTED
    assert result["outer"]["inner"]["ok"] is True
    assert result["outer"]["token"] == _REDACTED


def test_list_of_dicts_secrets_redacted() -> None:
    """Each dict in a list has its secret values redacted."""
    event: dict[str, Any] = {
        "agents": [
            {"api_key": "k1", "id": "a1"},
            {"api_key": "k2", "id": "a2"},
            {"name": "no-secret", "id": "a3"},
        ]
    }
    result = _call_filter(event)
    assert result["agents"][0]["api_key"] == _REDACTED
    assert result["agents"][0]["id"] == "a1"
    assert result["agents"][1]["api_key"] == _REDACTED
    assert result["agents"][1]["id"] == "a2"
    assert result["agents"][2]["name"] == "no-secret"


def test_list_with_non_dict_items_unchanged() -> None:
    """Non-dict items in a list are passed through unchanged."""
    event: dict[str, Any] = {"tags": ["a", "b", 42]}
    result = _call_filter(event)
    assert result["tags"] == ["a", "b", 42]


def test_mixed_case_api_key() -> None:
    event: dict[str, Any] = {"API_Key": "val"}
    result = _call_filter(event)
    assert result["API_Key"] == _REDACTED


def test_mixed_case_token() -> None:
    event: dict[str, Any] = {"Token": "val"}
    result = _call_filter(event)
    assert result["Token"] == _REDACTED


def test_mixed_case_password() -> None:
    event: dict[str, Any] = {"passWORD": "val"}
    result = _call_filter(event)
    assert result["passWORD"] == _REDACTED


def test_mixed_case_secret() -> None:
    event: dict[str, Any] = {"MY_SECRET_VALUE": "shh"}
    result = _call_filter(event)
    assert result["MY_SECRET_VALUE"] == _REDACTED


def test_no_op_for_clean_dict() -> None:
    """A dict with no secret keys is returned unchanged (values preserved)."""
    event: dict[str, Any] = {"event": "heartbeat", "host": "server1", "count": 42}
    result = _call_filter(event)
    assert result == event


def test_cyclic_dict_does_not_infinite_loop() -> None:
    """A self-referencing dict must not cause infinite recursion."""
    event: dict[str, Any] = {"key": "value", "api_key": "secret"}
    event["self"] = event

    result = _call_filter(event)
    assert result["api_key"] == _REDACTED
    assert result["key"] == "value"


def test_configure_structlog_idempotent() -> None:
    """Calling configure_structlog() twice must not raise."""
    from atm.observability.log_processors import configure_structlog

    configure_structlog()
    configure_structlog()


def test_bootstrap_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1 is set, atm.__init__ skips configure_structlog.

    We test the conditional logic directly: with the env var set, importing (reloading)
    atm should not call configure_structlog.
    """
    import unittest.mock as mock

    import atm

    monkeypatch.setenv("ATM_DISABLE_STRUCTLOG_BOOTSTRAP", "1")

    with mock.patch("atm.observability.log_processors.configure_structlog") as mock_cfg:
        importlib.reload(atm)
        mock_cfg.assert_not_called()


def test_bootstrap_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ATM_DISABLE_STRUCTLOG_BOOTSTRAP is not set, atm.__init__ calls configure_structlog.

    Verifies the actual production import path: reloading the atm package while the
    env var is unset must invoke configure_structlog (the bootstrap block in
    src/atm/__init__.py uses ``from atm.observability.log_processors import …``).
    """
    import unittest.mock as mock

    import atm

    monkeypatch.delenv("ATM_DISABLE_STRUCTLOG_BOOTSTRAP", raising=False)

    with mock.patch("atm.observability.log_processors.configure_structlog") as mock_cfg:
        importlib.reload(atm)
        mock_cfg.assert_called_once()


def test_filter_secrets_does_not_mutate_input() -> None:
    """The original event_dict passed to filter_secrets must remain unchanged."""
    original: dict[str, Any] = {
        "api_key": "original-secret",
        "name": "unchanged",
        "nested": {"password": "nested-secret", "safe": "value"},
    }
    import copy

    original_copy = copy.deepcopy(original)

    _call_filter(original)

    assert original == original_copy, "filter_secrets must not mutate its input"


def test_filter_secrets_exported_from_observability() -> None:
    """filter_secrets is re-exported from atm.observability."""
    from atm.observability import filter_secrets  # noqa: F401


def test_configure_structlog_exported_from_observability() -> None:
    """configure_structlog is re-exported from atm.observability."""
    from atm.observability import configure_structlog  # noqa: F401
