"""Streamlit HITL UI entrypoint for the ATM framework (M14 §UI).

Run with::

    streamlit run src/atm/ui/app.py

Environment variables:
    ATM_PG_DSN       — PostgreSQL async DSN (required for queue/DB features).
                       e.g. postgresql+asyncpg://atm:atm@localhost:5432/atm
    ATM_UI_SECRET    — Shared secret for participant login (optional; dev mode
                       skips auth when not set).
    ATM_CONFIG_PATH  — Path to an experiment config YAML to load
                       (optional; enables richer config-driven behaviour).

Page routing (via ``st.session_state.page``):
    "login"    — :func:`atm.ui.views.render_login`
    "queue"    — :func:`atm.ui.views.render_queue`
    "response" — :func:`atm.ui.views.render_response`
    "proctor"  — :func:`atm.ui.views.render_proctor`

nest_asyncio is applied first (before any other imports) so that
``asyncio.get_event_loop().run_until_complete(coro)`` is safe inside
Streamlit's Tornado event loop.
"""

import nest_asyncio

nest_asyncio.apply()

# ---------------------------------------------------------------------------
# Standard library + Streamlit
# ---------------------------------------------------------------------------

import os

import streamlit as st

# ---------------------------------------------------------------------------
# ATM UI helpers
# ---------------------------------------------------------------------------

from atm.ui._state import build_queue_from_env, init_session_state
from atm.ui.views import render_login, render_proctor, render_queue, render_response

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ATM HITL — Human Study",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Initialise session state keys with defaults (idempotent)
init_session_state()

# Lazy-init the queue helper once per browser session
if st.session_state.get("queue") is None:
    try:
        st.session_state["queue"] = build_queue_from_env()
    except RuntimeError:
        # ATM_PG_DSN not set — queue remains None; views show a warning
        pass

# ---------------------------------------------------------------------------
# Sidebar navigation (visible only when authenticated)
# ---------------------------------------------------------------------------

if st.session_state.get("authenticated"):
    with st.sidebar:
        st.markdown("## ATM HITL")
        st.markdown(f"Logged in as **{st.session_state.get('participant_id', '?')}**")
        st.divider()

        nav_page = st.radio(
            "Navigate",
            options=["queue", "proctor"],
            format_func=lambda p: {"queue": "📋 Task Queue", "proctor": "🔬 Proctor Panel"}[p],
            index=0 if st.session_state.get("page") != "proctor" else 1,
            key="sidebar_nav",
        )
        # Only update page via sidebar when not in middle of a response
        if st.session_state.get("page") not in ("response",):
            if nav_page != st.session_state.get("page"):
                st.session_state["current_row"] = None
                st.session_state["page"] = nav_page
                st.rerun()

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

page: str = st.session_state.get("page", "login")

if page == "login":
    render_login()
elif page == "queue":
    if not st.session_state.get("authenticated"):
        st.session_state["page"] = "login"
        st.rerun()
    else:
        render_queue()
elif page == "response":
    if not st.session_state.get("authenticated"):
        st.session_state["page"] = "login"
        st.rerun()
    else:
        current_row = st.session_state.get("current_row")
        if current_row is None:
            st.session_state["page"] = "queue"
            st.rerun()
        else:
            render_response(current_row)
elif page == "proctor":
    if not st.session_state.get("authenticated"):
        st.session_state["page"] = "login"
        st.rerun()
    else:
        render_proctor()
else:
    st.error(f"Unknown page: {page!r}. Redirecting to login.")
    st.session_state["page"] = "login"
    st.rerun()
