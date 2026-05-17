"""Unit tests for atm.ui.auth — shared-secret authentication helpers.

Tests:
1. check_secret: matching secrets return True.
2. check_secret: mismatched secrets return False.
3. check_secret: empty entered vs non-empty expected returns False.
4. get_expected_secret: reads from human_cfg.shared_secret.
5. get_expected_secret: falls back to ATM_UI_SECRET env var.
6. get_expected_secret: returns None when neither is set.
7. login: succeeds when expected_secret is None (dev mode).
8. login: succeeds with correct secret.
9. login: fails with wrong secret.
10. login: fails with empty participant_id.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from atm.ui.auth import check_secret, get_expected_secret, login


# ---------------------------------------------------------------------------
# check_secret
# ---------------------------------------------------------------------------


class TestCheckSecret:
    def test_matching_secrets_return_true(self) -> None:
        assert check_secret("correct-secret", "correct-secret") is True

    def test_mismatched_secrets_return_false(self) -> None:
        assert check_secret("wrong", "correct-secret") is False

    def test_empty_entered_vs_nonempty_expected_is_false(self) -> None:
        assert check_secret("", "some-secret") is False

    def test_empty_both_is_true(self) -> None:
        assert check_secret("", "") is True

    def test_case_sensitive(self) -> None:
        assert check_secret("Secret", "secret") is False

    def test_whitespace_sensitive(self) -> None:
        assert check_secret("secret ", "secret") is False


# ---------------------------------------------------------------------------
# get_expected_secret
# ---------------------------------------------------------------------------


class TestGetExpectedSecret:
    def test_reads_from_human_cfg(self) -> None:
        cfg = MagicMock()
        cfg.shared_secret = "cfg-secret"
        assert get_expected_secret(cfg) == "cfg-secret"

    def test_falls_back_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATM_UI_SECRET", "env-secret")
        cfg = MagicMock()
        cfg.shared_secret = None
        assert get_expected_secret(cfg) == "env-secret"

    def test_returns_none_when_nothing_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATM_UI_SECRET", raising=False)
        cfg = MagicMock()
        cfg.shared_secret = None
        assert get_expected_secret(cfg) is None

    def test_returns_none_when_cfg_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATM_UI_SECRET", raising=False)
        assert get_expected_secret(None) is None

    def test_cfg_secret_takes_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATM_UI_SECRET", "env-secret")
        cfg = MagicMock()
        cfg.shared_secret = "cfg-secret"
        assert get_expected_secret(cfg) == "cfg-secret"

    def test_missing_shared_secret_attr_falls_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If human_cfg has no shared_secret attribute, fall back to env var."""
        monkeypatch.setenv("ATM_UI_SECRET", "env-fallback")
        cfg = object()  # plain object without shared_secret
        result = get_expected_secret(cfg)
        assert result == "env-fallback"


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


class TestLogin:
    def test_dev_mode_no_secret_required(self) -> None:
        """When expected_secret is None, any non-empty participant_id succeeds."""
        assert login("P01", "", None) is True
        assert login("P01", "anything", None) is True

    def test_correct_secret_succeeds(self) -> None:
        assert login("P01", "good-secret", "good-secret") is True

    def test_wrong_secret_fails(self) -> None:
        assert login("P01", "bad-secret", "good-secret") is False

    def test_empty_participant_id_always_fails(self) -> None:
        assert login("", "", None) is False
        assert login("", "good-secret", "good-secret") is False
        assert login("   ", "good-secret", "good-secret") is False

    def test_whitespace_participant_id_fails(self) -> None:
        assert login("  ", "", None) is False

    def test_non_empty_participant_id_with_no_secret_succeeds(self) -> None:
        assert login("alice", "any", None) is True


# ---------------------------------------------------------------------------
# Module importable without Streamlit runtime
# ---------------------------------------------------------------------------


def test_auth_importable_without_streamlit() -> None:
    """atm.ui.auth is importable and works without Streamlit installed/running."""
    from atm.ui.auth import check_secret, get_expected_secret, login  # noqa: F401

    assert callable(check_secret)
    assert callable(get_expected_secret)
    assert callable(login)
