"""Page renderers for the ATM Streamlit HITL UI (M14 §UI).

Each ``render_*`` function renders a complete Streamlit page.  All Streamlit
imports are inside the functions so this module is importable without the
``[ui]`` extra installed.

Pages
-----
``render_login()``
    Shared-secret login form.  On success, writes to ``st.session_state``.

``render_queue()``
    Lists pending requests for the authenticated participant.  A "Claim"
    button transitions to the response page.

``render_response(queue_row)``
    Displays the context JSON (messages, scratchpad, tool calls) and a
    text area for the participant's action.  After submission, calls
    ``queue.submit_response``.

``render_proctor()``
    Lists ``study_sessions`` rows with "Start" / "End" controls.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atm.human._queue import QueueRow


# ---------------------------------------------------------------------------
# render_login
# ---------------------------------------------------------------------------


def render_login() -> None:
    """Render the login page.

    Reads ``st.session_state.config_bundle`` to obtain the ``HumanCfg``.
    On successful authentication, sets::

        st.session_state.authenticated = True
        st.session_state.participant_id = <entered_id>
        st.session_state.page = "queue"
    """
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Install the [ui] extra: uv pip install 'atm[ui]'") from exc

    from atm.ui.auth import get_expected_secret, login  # noqa: PLC0415

    st.title("ATM HITL — Participant Login")
    st.markdown(
        "Welcome to the Adaptive Topology MAS user study.  "
        "Enter your participant ID and the session secret provided by the proctor."
    )

    config_bundle = st.session_state.get("config_bundle")
    human_cfg = getattr(config_bundle, "human", None) if config_bundle else None
    expected_secret = get_expected_secret(human_cfg)

    with st.form("login_form"):
        participant_id = st.text_input("Participant ID", placeholder="e.g. P01")
        secret_input = st.text_input("Session secret", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if login(participant_id, secret_input, expected_secret):
            st.session_state["authenticated"] = True
            st.session_state["participant_id"] = participant_id.strip()
            st.session_state["page"] = "queue"
            st.rerun()
        else:
            st.error("Invalid participant ID or secret.  Please try again.")


# ---------------------------------------------------------------------------
# render_queue
# ---------------------------------------------------------------------------


def render_queue() -> None:
    """Render the queue page — list pending requests and allow claiming.

    Reads from ``st.session_state.queue`` and ``st.session_state.participant_id``.
    A "Claim" button calls ``queue.claim(...)`` and navigates to
    ``render_response``.
    """
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Install the [ui] extra: uv pip install 'atm[ui]'") from exc

    from atm.ui._state import _run_sync  # noqa: PLC0415
    from atm.ui.auth import logout  # noqa: PLC0415

    participant_id: str | None = st.session_state.get("participant_id")
    queue = st.session_state.get("queue")

    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.title(f"Pending Requests — {participant_id}")
    with col_logout:
        if st.button("Logout"):
            logout()
            st.rerun()

    if queue is None:
        st.warning(
            "Queue is not configured.  Set ``ATM_PG_DSN`` or provide a "
            "config bundle with ``human.queue_dsn``."
        )
        return

    with st.spinner("Fetching pending requests…"):
        rows = _run_sync(queue.fetch_pending(participant_id=participant_id, limit=20))

    if not rows:
        st.info("No pending requests.  Waiting for the runner to enqueue a task…")
        if st.button("Refresh"):
            st.rerun()
        return

    st.markdown(f"**{len(rows)} request(s) available.**")

    for row in rows:
        with st.expander(
            f"Request `{row.request_id}` — status: **{row.status}**",
            expanded=True,
        ):
            _render_context_summary(row.context_json)

            already_claimed = row.status == "claimed" and row.claimed_by == participant_id
            button_label = "Resume Response" if already_claimed else "Claim & Respond"

            if st.button(button_label, key=f"claim_{row.id}"):
                if not already_claimed:
                    claimed = _run_sync(queue.claim(row.id, participant_id=participant_id or ""))
                    if not claimed:
                        st.warning("Could not claim this request — it was already taken.")
                        st.rerun()
                        return
                st.session_state["current_row"] = row
                st.session_state["page"] = "response"
                st.rerun()


# ---------------------------------------------------------------------------
# render_response
# ---------------------------------------------------------------------------


def render_response(queue_row: "QueueRow") -> None:
    """Render the response form for a claimed queue row.

    Displays the full context (messages, scratchpad, tool calls) and
    a text area for the participant's action and rationale.  After
    submission, bundles the response (with TLX scores) and calls
    ``queue.submit_response``.

    Parameters
    ----------
    queue_row:
        The ``QueueRow`` obtained from :func:`render_queue` after claiming.
    """
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Install the [ui] extra: uv pip install 'atm[ui]'") from exc

    from atm.ui._state import _run_sync  # noqa: PLC0415
    from atm.ui.tlx import NasaTLXForm  # noqa: PLC0415

    queue = st.session_state.get("queue")
    participant_id: str | None = st.session_state.get("participant_id")

    col_title, col_back = st.columns([5, 1])
    with col_title:
        st.title(f"Respond — `{queue_row.request_id}`")
    with col_back:
        if st.button("← Back"):
            st.session_state["current_row"] = None
            st.session_state["page"] = "queue"
            st.rerun()

    # --- Context display ---
    ctx: dict[str, Any] = queue_row.context_json

    with st.expander("Context", expanded=True):
        question = ctx.get("question")
        if question:
            st.markdown(f"**Question:** {question}")

        role = ctx.get("role")
        if role:
            st.markdown(f"**Role:** `{role}`")

        allowed_actions = ctx.get("allowed_actions", [])
        if allowed_actions:
            st.markdown(f"**Allowed actions:** {', '.join(str(a) for a in allowed_actions)}")

        recent_messages = ctx.get("recent_messages", [])
        if recent_messages:
            st.markdown("**Recent messages:**")
            for msg in recent_messages:
                if isinstance(msg, dict):
                    role_label = msg.get("role", "?")
                    content = msg.get("content", "")
                    st.markdown(f"- **{role_label}**: {content}")
                else:
                    st.markdown(f"- {msg}")

        artifacts = ctx.get("artifacts", {})
        if artifacts:
            with st.expander("Artifacts / scratchpad"):
                for k, v in artifacts.items():
                    st.markdown(f"**{k}**")
                    if isinstance(v, str):
                        st.code(v)
                    else:
                        st.json(v)

        # Tool calls (if any in context)
        tool_calls = ctx.get("tool_calls", [])
        if tool_calls:
            with st.expander("Tool calls"):
                st.json(tool_calls)

    # --- Response form ---
    st.markdown("---")
    st.subheader("Your Response")

    action_options = [str(a) for a in ctx.get("allowed_actions", ["approve", "reject", "revise"])]
    action = st.selectbox("Action", options=action_options, key="resp_action")
    rationale = st.text_area(
        "Rationale / comment (optional)",
        placeholder="Explain your decision…",
        key="resp_rationale",
    )

    st.markdown("---")

    # --- NASA-TLX ---
    tlx_result = NasaTLXForm.render(key_prefix="resp_tlx")

    if tlx_result is not None:
        # Build study_session_id from config or generate a transient one
        config_bundle = st.session_state.get("config_bundle")
        human_cfg = getattr(config_bundle, "human", None) if config_bundle else None
        study_session_id: str | None = (
            getattr(human_cfg, "study_session_id", None) if human_cfg else None
        )
        if study_session_id is None:
            study_session_id = str(uuid.uuid4())

        # Strip raw_score from the per-scale dict (it goes alongside, not inside)
        tlx_scores = {k: v for k, v in tlx_result.items() if k != "raw_score"}

        response_payload: dict[str, Any] = {
            "action": str(action),
            "rationale": rationale.strip() if rationale else "",
            "tlx_scores": tlx_scores,
            "payload": {"study_session_id": study_session_id},
        }

        if queue is None:
            st.error("Queue not initialised — cannot submit response.")
            return

        with st.spinner("Submitting response…"):
            ok = _run_sync(
                queue.submit_response(
                    run_id=queue_row.run_id,
                    request_id=queue_row.request_id,
                    response_json=response_payload,
                )
            )

        if ok:
            st.success("Response submitted!  Thank you.")
            st.balloons()
            # Navigate back to queue
            st.session_state["current_row"] = None
            st.session_state["page"] = "queue"
            st.rerun()
        else:
            st.warning(
                "Could not submit — the response may have already been recorded.  "
                "Returning to queue."
            )
            st.session_state["current_row"] = None
            st.session_state["page"] = "queue"
            st.rerun()


# ---------------------------------------------------------------------------
# render_proctor
# ---------------------------------------------------------------------------


def render_proctor() -> None:
    """Render the proctor panel — manage study sessions.

    Lists ``study_sessions`` rows and provides "Start session" / "End session"
    buttons that write to ``study_sessions.status`` in Postgres.

    Access requires the same shared-secret authentication as participants.
    The proctor navigates to this page via the sidebar.
    """
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Install the [ui] extra: uv pip install 'atm[ui]'") from exc

    from atm.ui._state import _run_sync  # noqa: PLC0415

    st.title("Proctor Panel — Study Sessions")
    st.markdown(
        "Use this panel to start and end proctored study sessions.  "
        "Each session is associated with a participant and tracks consent, "
        "start/end times, and proctor notes."
    )

    queue = st.session_state.get("queue")
    if queue is None:
        st.warning(
            "Database not configured.  Set ``ATM_PG_DSN`` or load a config bundle."
        )
        return

    # Fetch sessions directly via the queue's session factory
    sessions = _run_sync(_fetch_study_sessions(queue))

    # --- New session form ---
    with st.expander("Start a new session", expanded=not sessions):
        with st.form("new_session_form"):
            new_participant_id = st.text_input("Participant ID", placeholder="e.g. P01")
            consent_given = st.checkbox("Participant has given informed consent", value=False)
            proctor_notes = st.text_area("Proctor notes (optional)")
            new_submitted = st.form_submit_button("Start Session")

        if new_submitted:
            if not new_participant_id.strip():
                st.error("Participant ID is required.")
            elif not consent_given:
                st.error("Consent must be confirmed before starting a session.")
            else:
                _run_sync(
                    _create_study_session(
                        queue,
                        participant_id=new_participant_id.strip(),
                        consent_given=consent_given,
                        proctor_notes=proctor_notes.strip() or None,
                    )
                )
                st.success(f"Session started for participant '{new_participant_id.strip()}'.")
                st.rerun()

    # --- Existing sessions ---
    if not sessions:
        st.info("No study sessions found.  Start the first one above.")
        return

    st.markdown(f"**{len(sessions)} session(s) found.**")

    for sess in sessions:
        _render_session_row(queue, sess)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_context_summary(ctx: dict[str, Any]) -> None:
    """Render a short summary of *ctx* in the queue listing."""
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Install the [ui] extra: uv pip install 'atm[ui]'") from exc

    question = ctx.get("question")
    if question:
        st.markdown(f"**{question}**")

    role = ctx.get("role")
    allowed_actions = ctx.get("allowed_actions", [])
    if role or allowed_actions:
        st.markdown(
            f"Role: `{role or '?'}` | Actions: {', '.join(str(a) for a in allowed_actions)}"
        )


def _render_session_row(queue: Any, sess: Any) -> None:
    """Render a single study-session row in the proctor panel."""
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Install the [ui] extra: uv pip install 'atm[ui]'") from exc

    from atm.ui._state import _run_sync  # noqa: PLC0415

    session_id = str(sess.get("id", "?"))
    participant = sess.get("participant_id", "?")
    status = sess.get("status", "?")
    started_at = sess.get("started_at")
    ended_at = sess.get("ended_at")

    label = f"{participant} — `{status}` (started: {started_at})"
    with st.expander(label, expanded=status == "active"):
        st.markdown(f"**Session ID:** `{session_id}`")
        if ended_at:
            st.markdown(f"**Ended at:** {ended_at}")
        notes = sess.get("proctor_notes")
        if notes:
            st.markdown(f"**Notes:** {notes}")

        col_end, col_pause = st.columns(2)
        with col_end:
            if status != "ended":
                if st.button("End session", key=f"end_{session_id}"):
                    _run_sync(
                        _update_session_status(
                            queue, uuid.UUID(session_id), "ended"
                        )
                    )
                    st.rerun()
        with col_pause:
            if status == "active":
                if st.button("Pause session", key=f"pause_{session_id}"):
                    _run_sync(
                        _update_session_status(
                            queue, uuid.UUID(session_id), "paused"
                        )
                    )
                    st.rerun()
            elif status == "paused":
                if st.button("Resume session", key=f"resume_{session_id}"):
                    _run_sync(
                        _update_session_status(
                            queue, uuid.UUID(session_id), "active"
                        )
                    )
                    st.rerun()


async def _fetch_study_sessions(queue: Any) -> list[dict[str, Any]]:
    """Query study_sessions table via the queue's session factory.

    Returns a list of plain dicts; each dict corresponds to one row.
    Columns: id, participant_id, status, consent_given, started_at,
    ended_at, proctor_notes.
    """
    import sqlalchemy as sa  # noqa: PLC0415

    from atm.storage.models import StudySession  # noqa: PLC0415

    stmt = sa.select(
        StudySession.id,
        StudySession.participant_id,
        StudySession.status,
        StudySession.consent_given,
        StudySession.started_at,
        StudySession.ended_at,
        StudySession.proctor_notes,
    ).order_by(StudySession.started_at.desc())

    async with queue._sf() as session:
        result = await session.execute(stmt)
        rows = result.fetchall()

    return [
        {
            "id": row.id,
            "participant_id": row.participant_id,
            "status": row.status,
            "consent_given": row.consent_given,
            "started_at": row.started_at,
            "ended_at": row.ended_at,
            "proctor_notes": row.proctor_notes,
        }
        for row in rows
    ]


async def _create_study_session(
    queue: Any,
    *,
    participant_id: str,
    consent_given: bool,
    proctor_notes: str | None,
) -> uuid.UUID:
    """Insert a new StudySession row and return its UUID."""
    from atm.storage.models import StudySession  # noqa: PLC0415

    new_id = uuid.uuid4()
    sess_obj = StudySession(
        id=new_id,
        participant_id=participant_id,
        consent_given=consent_given,
        proctor_notes=proctor_notes,
        status="active",
    )
    async with queue._sf() as session:
        session.add(sess_obj)
        await session.commit()

    return new_id


async def _update_session_status(
    queue: Any,
    session_id: uuid.UUID,
    new_status: str,
) -> None:
    """Update study_sessions.status for a given session_id."""
    import sqlalchemy as sa  # noqa: PLC0415

    from atm.storage.models import StudySession  # noqa: PLC0415

    stmt = (
        sa.update(StudySession)
        .where(StudySession.id == session_id)
        .values(status=new_status)
    )
    async with queue._sf() as session:
        await session.execute(stmt)
        await session.commit()
