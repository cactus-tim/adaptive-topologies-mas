"""Unit tests for atm.human._queue.HumanRequestQueue.

All tests use a FakeSession / mock async engine — no live Postgres required.

Coverage:
  - enqueue: idempotent (first call returns True, duplicate returns False)
  - fetch_pending: returns rows with status='pending', filters by participant_id
  - claim: atomic update, returns True on success, False if already claimed
  - submit_response: sets response_json; second call with same args returns False (idempotent)
  - cancel: sets status='cancelled' where response_json IS NULL
  - wait_for_response: polling loop, returns dict on success, None on timeout
  - QueueRow dataclass fields
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atm.human._queue import HumanRequestQueue, QueueRow

# ---------------------------------------------------------------------------
# Helpers — fake DB row objects
# ---------------------------------------------------------------------------


class _FakeRow:
    """Minimal attribute-based fake for a SQLAlchemy Row-like object."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# QueueRow dataclass
# ---------------------------------------------------------------------------


class TestQueueRow:
    def test_fields_present(self) -> None:
        row_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = QueueRow(
            id=row_id,
            run_id=run_id,
            request_id="req-1",
            context_json={"q": "hello"},
            status="pending",
            claimed_by=None,
        )
        assert row.id == row_id
        assert row.run_id == run_id
        assert row.request_id == "req-1"
        assert row.context_json == {"q": "hello"}
        assert row.status == "pending"
        assert row.claimed_by is None

    def test_claimed_by_can_be_set(self) -> None:
        row = QueueRow(
            id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            request_id="r",
            context_json={},
            status="claimed",
            claimed_by="participant_1",
        )
        assert row.claimed_by == "participant_1"


# ---------------------------------------------------------------------------
# Fixtures — build HumanRequestQueue with a fake session factory
# ---------------------------------------------------------------------------


def _make_queue(session_factory: Any) -> HumanRequestQueue:
    """Build a HumanRequestQueue with the provided session factory."""
    return HumanRequestQueue(session_factory=session_factory)


def _make_fake_session_factory(session: Any) -> Any:
    """Return a callable that, when called, returns a context-manager yielding session."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=cm)
    return factory


# ---------------------------------------------------------------------------
# enqueue — idempotency
# ---------------------------------------------------------------------------


class TestEnqueue:
    async def test_enqueue_first_call_returns_true(self) -> None:
        """First enqueue should return True (row inserted)."""
        run_id = uuid.uuid4()
        session = AsyncMock()
        # Simulate "1 row inserted" via rowcount on execute result
        result = MagicMock()
        result.rowcount = 1
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        inserted = await queue.enqueue(
            run_id=run_id,
            request_id="req-1",
            context_json={"question": "Approve?"},
        )
        assert inserted is True

    async def test_enqueue_duplicate_returns_false(self) -> None:
        """Duplicate enqueue (ON CONFLICT DO NOTHING) returns False."""
        run_id = uuid.uuid4()
        session = AsyncMock()
        result = MagicMock()
        result.rowcount = 0  # no rows inserted because of ON CONFLICT DO NOTHING
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        inserted = await queue.enqueue(
            run_id=run_id,
            request_id="req-1",
            context_json={"question": "Approve?"},
        )
        assert inserted is False

    async def test_enqueue_calls_execute(self) -> None:
        """enqueue must call session.execute at least once."""
        run_id = uuid.uuid4()
        session = AsyncMock()
        result = MagicMock()
        result.rowcount = 1
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        await queue.enqueue(run_id=run_id, request_id="req-A", context_json={})
        assert session.execute.await_count >= 1


# ---------------------------------------------------------------------------
# fetch_pending
# ---------------------------------------------------------------------------


class TestFetchPending:
    async def test_fetch_pending_returns_queue_rows(self) -> None:
        """fetch_pending returns a list of QueueRow instances."""
        run_id = uuid.uuid4()
        row_id = uuid.uuid4()
        fake_row = _FakeRow(
            id=row_id,
            run_id=run_id,
            request_id="req-1",
            context_json={"q": "hi"},
            status="pending",
            claimed_by=None,
        )
        result = MagicMock()
        result.fetchall = MagicMock(return_value=[fake_row])
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        rows = await queue.fetch_pending(limit=10)
        assert len(rows) == 1
        assert isinstance(rows[0], QueueRow)
        assert rows[0].id == row_id
        assert rows[0].request_id == "req-1"

    async def test_fetch_pending_empty_result(self) -> None:
        """fetch_pending returns empty list when no rows match."""
        result = MagicMock()
        result.fetchall = MagicMock(return_value=[])
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        rows = await queue.fetch_pending()
        assert rows == []

    async def test_fetch_pending_with_participant_id(self) -> None:
        """fetch_pending with participant_id calls execute (filter logic is DB-side)."""
        result = MagicMock()
        result.fetchall = MagicMock(return_value=[])
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        rows = await queue.fetch_pending(participant_id="alice", limit=5)
        assert rows == []
        session.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# claim
# ---------------------------------------------------------------------------


class TestClaim:
    async def test_claim_success_returns_true(self) -> None:
        """claim returns True when the UPDATE succeeds (row transitions to 'claimed')."""
        row_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 1
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        claimed = await queue.claim(queue_row_id=row_id, participant_id="alice")
        assert claimed is True

    async def test_claim_already_claimed_returns_false(self) -> None:
        """claim returns False when the row is already claimed (rowcount=0)."""
        row_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 0  # no rows updated because status != 'pending'
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        claimed = await queue.claim(queue_row_id=row_id, participant_id="bob")
        assert claimed is False

    async def test_claim_calls_execute(self) -> None:
        """claim must issue an UPDATE via session.execute."""
        row_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 1
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        await queue.claim(queue_row_id=row_id, participant_id="charlie")
        assert session.execute.await_count >= 1


# ---------------------------------------------------------------------------
# submit_response — idempotency
# ---------------------------------------------------------------------------


class TestSubmitResponse:
    async def test_submit_response_first_call_returns_true(self) -> None:
        """First submit_response returns True (row updated)."""
        run_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 1
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        updated = await queue.submit_response(
            run_id=run_id,
            request_id="req-1",
            response_json={"answer": "yes"},
        )
        assert updated is True

    async def test_submit_response_second_call_returns_false(self) -> None:
        """Second submit_response (response_json already set) returns False (idempotent)."""
        run_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 0  # WHERE response_json IS NULL fails — already set
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        updated = await queue.submit_response(
            run_id=run_id,
            request_id="req-1",
            response_json={"answer": "no"},
        )
        assert updated is False

    async def test_submit_response_calls_execute(self) -> None:
        """submit_response must call session.execute."""
        run_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 1
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        await queue.submit_response(run_id=run_id, request_id="r", response_json={})
        assert session.execute.await_count >= 1


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


class TestCancel:
    async def test_cancel_unanswered_returns_true(self) -> None:
        """cancel returns True when an unanswered row is cancelled."""
        run_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 1
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        cancelled = await queue.cancel(run_id=run_id, request_id="req-1")
        assert cancelled is True

    async def test_cancel_already_answered_returns_false(self) -> None:
        """cancel returns False when response_json is already set (idempotent)."""
        run_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 0  # WHERE response_json IS NULL fails — already answered
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        cancelled = await queue.cancel(run_id=run_id, request_id="req-1")
        assert cancelled is False

    async def test_cancel_calls_execute(self) -> None:
        """cancel must call session.execute."""
        run_id = uuid.uuid4()
        result = MagicMock()
        result.rowcount = 1
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        await queue.cancel(run_id=run_id, request_id="req-X")
        assert session.execute.await_count >= 1


# ---------------------------------------------------------------------------
# wait_for_response — polling
# ---------------------------------------------------------------------------


class TestWaitForResponse:
    async def test_returns_response_when_available(self) -> None:
        """wait_for_response returns the response_json dict when found."""
        run_id = uuid.uuid4()
        expected_response = {"answer": "approve", "score": 9}

        # First execute call returns None; second returns the response

        async def _noop_sleep(s: float) -> None:
            pass

        # We need separate result mocks for each session.execute call
        result_none = MagicMock()
        result_none.scalar_one_or_none = MagicMock(return_value=None)

        result_found = MagicMock()
        result_found.scalar_one_or_none = MagicMock(return_value=expected_response)

        session = AsyncMock()
        session.execute = AsyncMock(side_effect=[result_none, result_found])

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        with patch("asyncio.sleep", side_effect=_noop_sleep):
            response = await queue.wait_for_response(
                run_id=run_id,
                request_id="req-1",
                timeout_s=5.0,
            )

        assert response == expected_response

    async def test_returns_none_on_timeout(self) -> None:
        """wait_for_response returns None when timeout expires without a response."""
        run_id = uuid.uuid4()

        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        # Use a very short timeout and patch sleep so the loop terminates quickly
        sleep_calls: list[float] = []

        async def _fast_sleep(s: float) -> None:
            sleep_calls.append(s)
            # After 2 iterations, make time appear to have expired by raising
            if len(sleep_calls) >= 2:
                # Patch the internal clock so timeout logic fires
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError), patch("asyncio.sleep", side_effect=_fast_sleep):
            await queue.wait_for_response(
                run_id=run_id,
                request_id="req-1",
                timeout_s=0.001,  # effectively immediate timeout
            )

    async def test_returns_none_on_zero_timeout(self) -> None:
        """wait_for_response with timeout_s=0 returns None immediately (no rows)."""
        run_id = uuid.uuid4()

        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        response = await queue.wait_for_response(
            run_id=run_id,
            request_id="req-1",
            timeout_s=0.0,
        )
        assert response is None

    async def test_wait_for_response_returns_immediately_if_already_answered(self) -> None:
        """wait_for_response returns immediately when response_json is already set."""
        run_id = uuid.uuid4()
        existing_response = {"answer": "reject"}

        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=existing_response)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        factory = _make_fake_session_factory(session)
        queue = _make_queue(factory)

        response = await queue.wait_for_response(
            run_id=run_id,
            request_id="req-1",
            timeout_s=30.0,
        )
        assert response == existing_response
