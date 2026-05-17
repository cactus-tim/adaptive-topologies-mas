"""Shared-secret authentication helpers for the ATM Streamlit UI (M14 §UI).

Authentication strategy:
  - The proctor configures ``HumanCfg.shared_secret`` (or ``ATM_UI_SECRET``
    env var as fallback).
  - On the login page, the participant enters a secret.  The UI compares it
    to the expected secret using constant-time comparison (``hmac.compare_digest``).
  - On success, ``st.session_state.authenticated`` is set to ``True`` and the
    participant ID is stored.
  - Logout clears ``authenticated``, ``participant_id``, and ``current_row``.

All Streamlit imports are lazy (inside functions) so this module is importable
without the ``[ui]`` extra installed.
"""

from __future__ import annotations

import hmac
import os
from typing import Any


def check_secret(entered: str, expected: str) -> bool:
    """Compare *entered* to *expected* using constant-time comparison.

    Parameters
    ----------
    entered:
        The secret entered by the participant in the UI.
    expected:
        The expected shared secret from configuration.

    Returns
    -------
    bool
        ``True`` if the secrets match; ``False`` otherwise.
    """
    return hmac.compare_digest(
        entered.encode("utf-8"),
        expected.encode("utf-8"),
    )


def get_expected_secret(human_cfg: Any | None) -> str | None:
    """Resolve the expected shared secret from config or environment.

    Lookup order:
    1. ``human_cfg.shared_secret`` (when config is loaded)
    2. ``ATM_UI_SECRET`` environment variable
    3. ``None`` (no authentication required — dev mode)

    Parameters
    ----------
    human_cfg:
        A :class:`atm.experiment.config.HumanCfg` instance or ``None``.

    Returns
    -------
    str | None
        The shared secret, or ``None`` if not configured.
    """
    if human_cfg is not None:
        secret = getattr(human_cfg, "shared_secret", None)
        if secret:
            return str(secret)
    env_secret = os.environ.get("ATM_UI_SECRET")
    if env_secret:
        return env_secret
    return None


def login(participant_id: str, entered_secret: str, expected_secret: str | None) -> bool:
    """Attempt login for *participant_id*.

    When *expected_secret* is ``None`` (dev mode), login always succeeds.

    Parameters
    ----------
    participant_id:
        Participant identifier entered in the UI.
    entered_secret:
        Secret entered in the UI.
    expected_secret:
        Expected shared secret from configuration.

    Returns
    -------
    bool
        ``True`` if authentication succeeded.
    """
    if not participant_id.strip():
        return False
    if expected_secret is None:
        return True
    return check_secret(entered_secret, expected_secret)


def logout() -> None:
    """Clear authentication state from ``st.session_state``.

    Resets ``authenticated``, ``participant_id``, ``current_row``, and
    navigates back to the login page.
    """
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install the [ui] extra: "
            "uv pip install 'atm[ui]'"
        ) from exc

    st.session_state["authenticated"] = False
    st.session_state["participant_id"] = None
    st.session_state["current_row"] = None
    st.session_state["page"] = "login"
