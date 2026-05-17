"""Async PG-backed queue helper for the Streamlit HITL gateway (M14 §HITL).

HumanRequestQueue wraps the ``human_request_queue`` table and provides the full
request lifecycle:

  pending → claimed → answered  (happy path)
  pending → cancelled           (runner gives up or times out)

All public methods are idempotent and safe to call from multiple processes
simultaneously; atomicity is enforced at the database level via
``ON CONFLICT DO NOTHING`` (enqueue) and conditional ``UPDATE … WHERE …
RETURNING id`` (claim / submit_response / cancel).

Usage::

    from atm.storage.session import create_engine, create_session_factory
    from atm.human._queue import HumanRequestQueue

    engine = create_engine(dsn)
    queue = HumanRequestQueue(session_factory=create_session_factory(engine))

    await queue.enqueue(run_id, "req-1", {"question": "Proceed?"})
    rows = await queue.fetch_pending(limit=5)
    await queue.claim(rows[0].id, participant_id="alice")
    await queue.submit_response(run_id, "req-1", {"answer": "yes"})
    response = await queue.wait_for_response(run_id, "req-1", timeout_s=120)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atm.storage.models import HumanRequestQueue as HRQModel
from atm.storage.session import session_scope

__all__ = ["HumanRequestQueue", "QueueRow"]

# ---------------------------------------------------------------------------
# QueueRow — lightweight data transfer object
# ---------------------------------------------------------------------------


@dataclass
class QueueRow:
    """Minimal projection of a ``human_request_queue`` row returned to callers.

    Using a dataclass (rather than returning ORM instances) prevents accidental
    lazy-load exceptions and keeps the public API decoupled from the ORM model.
    """

    id: uuid.UUID
    run_id: uuid.UUID
    request_id: str
    context_json: dict[str, Any]
    status: str
    claimed_by: str | None


# ---------------------------------------------------------------------------
# HumanRequestQueue
# ---------------------------------------------------------------------------


class HumanRequestQueue:
    """Async PG queue between the runner process and the Streamlit UI.

    All methods accept / return plain Python values; the caller does not need
    to interact with SQLAlchemy objects directly.

    Parameters
    ----------
    session_factory:
        An ``async_sessionmaker[AsyncSession]`` bound to the async engine that
        connects to the database containing the ``human_request_queue`` table.
        Obtain one via ``create_session_factory(create_engine(dsn))``.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    # ------------------------------------------------------------------
    # enqueue
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        run_id: uuid.UUID,
        request_id: str,
        context_json: dict[str, Any],
    ) -> bool:
        """Insert a new request row idempotently.

        Parameters
        ----------
        run_id:
            UUID of the run this request belongs to (FK → runs.id).
        request_id:
            Caller-assigned stable idempotency key (unique within a run).
        context_json:
            Arbitrary context dict sent to the human participant.

        Returns
        -------
        bool
            ``True`` if a new row was inserted; ``False`` if a row with
            (run_id, request_id) already existed (``ON CONFLICT DO NOTHING``).
        """
        stmt = (
            pg_insert(HRQModel)
            .values(
                id=uuid.uuid4(),
                run_id=run_id,
                request_id=request_id,
                context_json=context_json,
                status="pending",
            )
            .on_conflict_do_nothing(
                constraint="uq_human_request_queue_run_request",
            )
        )
        async with session_scope(self._sf) as session:
            raw = await session.execute(stmt)
            return cast(CursorResult[Any], raw).rowcount == 1

    # ------------------------------------------------------------------
    # fetch_pending
    # ------------------------------------------------------------------

    async def fetch_pending(
        self,
        participant_id: str | None = None,
        limit: int = 10,
    ) -> list[QueueRow]:
        """Return pending (and optionally claimed-by-this-participant) queue rows.

        Parameters
        ----------
        participant_id:
            When provided, includes rows already claimed by this participant
            (status='claimed' AND claimed_by=participant_id) in addition to
            unclaimed pending rows (status='pending').  When ``None``, only
            status='pending' rows are returned.
        limit:
            Maximum number of rows to return.

        Returns
        -------
        list[QueueRow]
            Rows ordered by ``created_at`` ascending (oldest first).
        """
        # Build WHERE clause
        if participant_id is not None:
            where_clause = sa.or_(
                HRQModel.status == "pending",
                sa.and_(
                    HRQModel.status == "claimed",
                    HRQModel.claimed_by == participant_id,
                ),
            )
        else:
            where_clause = HRQModel.status == "pending"

        stmt = (
            sa.select(
                HRQModel.id,
                HRQModel.run_id,
                HRQModel.request_id,
                HRQModel.context_json,
                HRQModel.status,
                HRQModel.claimed_by,
            )
            .where(where_clause)
            .order_by(HRQModel.created_at.asc())
            .limit(limit)
        )

        async with self._sf() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        return [
            QueueRow(
                id=row.id,
                run_id=row.run_id,
                request_id=row.request_id,
                context_json=row.context_json,
                status=row.status,
                claimed_by=row.claimed_by,
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # claim
    # ------------------------------------------------------------------

    async def claim(
        self,
        queue_row_id: uuid.UUID,
        participant_id: str,
    ) -> bool:
        """Atomically claim a pending row for a participant.

        The UPDATE is conditional on ``status='pending'`` so that concurrent
        claims from multiple UI sessions are serialised at the database level —
        only one will succeed.

        Parameters
        ----------
        queue_row_id:
            Primary-key UUID of the row to claim.
        participant_id:
            Identifier of the participant claiming the row.

        Returns
        -------
        bool
            ``True`` if the row was successfully claimed; ``False`` if another
            process already claimed it (or it no longer exists).
        """
        stmt = (
            sa.update(HRQModel)
            .where(
                HRQModel.id == queue_row_id,
                HRQModel.status == "pending",
            )
            .values(
                status="claimed",
                claimed_by=participant_id,
                claimed_at=sa.func.now(),
            )
        )
        async with session_scope(self._sf) as session:
            raw = await session.execute(stmt)
            return cast(CursorResult[Any], raw).rowcount == 1

    # ------------------------------------------------------------------
    # submit_response
    # ------------------------------------------------------------------

    async def submit_response(
        self,
        run_id: uuid.UUID,
        request_id: str,
        response_json: dict[str, Any],
    ) -> bool:
        """Store the human's response; idempotent — second call returns False.

        The ``WHERE response_json IS NULL`` guard ensures that only the first
        call actually updates the row; subsequent calls with the same
        (run_id, request_id) are no-ops.

        Parameters
        ----------
        run_id:
            Run UUID identifying the queue row.
        request_id:
            Request idempotency key identifying the queue row.
        response_json:
            The human's response payload.

        Returns
        -------
        bool
            ``True`` if the row was updated (first response); ``False`` if
            response_json was already set (idempotent no-op).
        """
        stmt = (
            sa.update(HRQModel)
            .where(
                HRQModel.run_id == run_id,
                HRQModel.request_id == request_id,
                HRQModel.response_json.is_(None),
            )
            .values(
                response_json=response_json,
                status="answered",
                responded_at=sa.func.now(),
            )
        )
        async with session_scope(self._sf) as session:
            raw = await session.execute(stmt)
            return cast(CursorResult[Any], raw).rowcount == 1

    # ------------------------------------------------------------------
    # wait_for_response
    # ------------------------------------------------------------------

    async def wait_for_response(
        self,
        run_id: uuid.UUID,
        request_id: str,
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        """Poll until response_json is set, then return it.

        Polling interval is 1 second.  Returns ``None`` if the timeout expires
        before a response is recorded.

        Parameters
        ----------
        run_id:
            Run UUID identifying the queue row.
        request_id:
            Request idempotency key identifying the queue row.
        timeout_s:
            Maximum number of seconds to wait.

        Returns
        -------
        dict[str, Any] | None
            The response payload, or ``None`` on timeout.

        Note
        ----
        # TODO(M14.1): replace polling with PG LISTEN/NOTIFY
        """
        deadline = time.monotonic() + timeout_s
        poll_stmt = sa.select(HRQModel.response_json).where(
            HRQModel.run_id == run_id,
            HRQModel.request_id == request_id,
        )

        while True:
            async with self._sf() as session:
                result = await session.execute(poll_stmt)
                response_json: dict[str, Any] | None = result.scalar_one_or_none()

            if response_json is not None:
                return response_json

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            await asyncio.sleep(min(1.0, remaining))

    # ------------------------------------------------------------------
    # cancel
    # ------------------------------------------------------------------

    async def cancel(
        self,
        run_id: uuid.UUID,
        request_id: str,
    ) -> bool:
        """Cancel a pending/claimed request that has not yet been answered.

        The ``WHERE response_json IS NULL`` guard prevents cancellation of
        already-answered rows (idempotent behaviour).

        Parameters
        ----------
        run_id:
            Run UUID identifying the queue row.
        request_id:
            Request idempotency key identifying the queue row.

        Returns
        -------
        bool
            ``True`` if the row was cancelled; ``False`` if it was already
            answered (or does not exist).
        """
        stmt = (
            sa.update(HRQModel)
            .where(
                HRQModel.run_id == run_id,
                HRQModel.request_id == request_id,
                HRQModel.response_json.is_(None),
            )
            .values(status="cancelled")
        )
        async with session_scope(self._sf) as session:
            raw = await session.execute(stmt)
            return cast(CursorResult[Any], raw).rowcount == 1
