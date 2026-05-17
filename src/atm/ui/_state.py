"""Session-state initialisation and async helpers for the ATM Streamlit UI (M14 §UI).

All Streamlit imports are lazy (inside functions) so this module can be imported
without the ``[ui]`` extra installed.

``_run_sync`` uses ``asyncio.get_event_loop().run_until_complete(coro)`` which
is safe after ``nest_asyncio.apply()`` has been called in ``app.py``.

Session state keys managed here:

``authenticated``   — bool; True after the shared-secret login succeeds.
``participant_id``  — str | None; set at login.
``page``            — str; current page name (``"login"``, ``"queue"``,
                      ``"response"``, ``"proctor"``).
``current_row``     — QueueRow | None; the queue row the participant is
                      currently responding to.
``queue``           — HumanRequestQueue | None; lazy-initialised queue helper.
``config_bundle``   — ExperimentConfig | None; loaded once at startup.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")

# Keys and their default values.  Using a dict makes it easy to iterate.
_DEFAULTS: dict[str, Any] = {
    "authenticated": False,
    "participant_id": None,
    "page": "login",
    "current_row": None,
    "queue": None,
    "config_bundle": None,
}


def init_session_state() -> None:
    """Initialise all ATM session state keys with their default values.

    Idempotent — keys that already exist are not overwritten.  Call this
    once at the top of ``app.py`` before routing to any page.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install the [ui] extra: uv pip install 'atm[ui]'"
        ) from exc

    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run *coro* synchronously inside the existing event loop.

    ``nest_asyncio.apply()`` (called at app startup) makes
    ``loop.run_until_complete`` safe even if there is already a running event
    loop (e.g. inside Streamlit's Tornado event loop).

    Parameters
    ----------
    coro:
        An awaitable coroutine to execute.

    Returns
    -------
    _T
        The value returned by the coroutine.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


def build_queue_from_env() -> Any:
    """Construct a HumanRequestQueue from the ATM_PG_DSN environment variable.

    Returns
    -------
    HumanRequestQueue
        Ready-to-use queue backed by the Postgres DB at ``ATM_PG_DSN``.

    Raises
    ------
    RuntimeError
        When ``ATM_PG_DSN`` is not set in the environment.
    """
    from atm.human._queue import HumanRequestQueue
    from atm.storage.session import (
        create_engine,
        create_session_factory,
    )

    dsn = os.environ.get("ATM_PG_DSN")
    if not dsn:
        raise RuntimeError(
            "ATM_PG_DSN environment variable is not set. "
            "Provide a PostgreSQL DSN such as "
            "postgresql+asyncpg://user:pass@host/db"
        )

    engine = create_engine(dsn)
    session_factory = create_session_factory(engine)
    return HumanRequestQueue(session_factory=session_factory)


def build_queue_from_config(human_cfg: Any) -> Any:
    """Construct a HumanRequestQueue from a ``HumanCfg`` instance.

    Falls back to :func:`build_queue_from_env` when
    ``human_cfg.queue_dsn`` is ``None``.

    Parameters
    ----------
    human_cfg:
        A :class:`atm.experiment.config.HumanCfg` instance.

    Returns
    -------
    HumanRequestQueue
        Ready-to-use queue.
    """
    if human_cfg.queue_dsn is not None:
        from atm.human._queue import HumanRequestQueue
        from atm.storage.session import (
            create_engine,
            create_session_factory,
        )

        engine = create_engine(human_cfg.queue_dsn)
        session_factory = create_session_factory(engine)
        return HumanRequestQueue(session_factory=session_factory)

    return build_queue_from_env()
