"""Integration tests for atm.human._queue.HumanRequestQueue against real PostgreSQL.

Requires ATM_ENABLE_PG_TESTS=1 and a live PostgreSQL instance (default DSN:
postgresql+asyncpg://atm:atm@localhost:5432/atm_test).

Lifecycle test:
  enqueue → fetch_pending → claim → submit_response → wait_for_response → assert response

Additional:
  - enqueue idempotency (same run_id + request_id twice → only one row)
  - cancel (status set to 'cancelled' where response_json IS NULL)
  - submit_response double-call returns False second time
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from atm.human._queue import HumanRequestQueue, QueueRow
from atm.storage.session import create_session_factory

# ---------------------------------------------------------------------------
# Helper: insert a minimal run row so FK constraint on human_request_queue.run_id
# is satisfied. Experiments table must also have a parent row.
# ---------------------------------------------------------------------------


async def _insert_run(engine: AsyncEngine) -> uuid.UUID:
    """Insert a minimal experiment + run row; return the run UUID."""
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO experiments (id, name, config_snapshot, status) "
                "VALUES (:id, :name, :cfg, :status)"
            ),
            {
                "id": str(exp_id),
                "name": f"test-exp-{exp_id}",
                "cfg": "{}",
                "status": "running",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO runs "
                "(id, exp_id, topology, task_id, agent_set, seed, model, status) "
                "VALUES (:id, :exp_id, :topo, :task, :agent_set, :seed, :model, :status)"
            ),
            {
                "id": str(run_id),
                "exp_id": str(exp_id),
                "topo": "chain",
                "task": "t1",
                "agent_set": "canonical_4",
                "seed": 0,
                "model": "fake:scripted",
                "status": "running",
            },
        )
    return run_id


# ---------------------------------------------------------------------------
# Full lifecycle test
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_queue_full_lifecycle(pg_engine_fast: AsyncEngine) -> None:
    """enqueue → fetch_pending → claim → submit_response → wait_for_response.

    This is the primary acceptance test for step 2.1.
    """
    run_id = await _insert_run(pg_engine_fast)
    request_id = f"req-{uuid.uuid4()}"
    context = {"question": "Should we proceed?", "phase": "planning"}
    response_data = {"answer": "yes", "reason": "looks good"}

    session_factory = create_session_factory(pg_engine_fast)
    queue = HumanRequestQueue(session_factory=session_factory)

    # 1. enqueue — first call should return True
    inserted = await queue.enqueue(
        run_id=run_id,
        request_id=request_id,
        context_json=context,
    )
    assert inserted is True, "First enqueue must return True (row inserted)"

    # 2. fetch_pending — should return our row
    rows = await queue.fetch_pending(limit=10)
    pending_ids = [r.request_id for r in rows]
    assert request_id in pending_ids, "fetch_pending must include our newly enqueued row"

    # Find our row
    our_row: QueueRow | None = next((r for r in rows if r.request_id == request_id), None)
    assert our_row is not None
    assert our_row.status == "pending"
    assert our_row.context_json == context

    # 3. claim — should return True (transitions to 'claimed')
    claimed = await queue.claim(queue_row_id=our_row.id, participant_id="participant_42")
    assert claimed is True, "claim must return True on first claim"

    # 4. submit_response — first call returns True
    updated = await queue.submit_response(
        run_id=run_id,
        request_id=request_id,
        response_json=response_data,
    )
    assert updated is True, "First submit_response must return True"

    # 5. wait_for_response — should return the response immediately (already answered)
    response = await queue.wait_for_response(
        run_id=run_id,
        request_id=request_id,
        timeout_s=10.0,
    )
    assert response is not None, "wait_for_response must not return None when answer exists"
    assert response.get("answer") == "yes"
    assert response.get("reason") == "looks good"


# ---------------------------------------------------------------------------
# Idempotency: enqueue same (run_id, request_id) twice
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_enqueue_idempotency(pg_engine_fast: AsyncEngine) -> None:
    """Enqueuing the same (run_id, request_id) twice yields only one row."""
    run_id = await _insert_run(pg_engine_fast)
    request_id = f"req-{uuid.uuid4()}"

    session_factory = create_session_factory(pg_engine_fast)
    queue = HumanRequestQueue(session_factory=session_factory)

    first = await queue.enqueue(run_id=run_id, request_id=request_id, context_json={"q": "a"})
    second = await queue.enqueue(run_id=run_id, request_id=request_id, context_json={"q": "a"})

    assert first is True, "First enqueue must return True"
    assert second is False, "Duplicate enqueue must return False (ON CONFLICT DO NOTHING)"

    # Confirm exactly one row in DB
    async with pg_engine_fast.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT COUNT(*) FROM human_request_queue "
                "WHERE run_id = :run_id AND request_id = :req_id"
            ),
            {"run_id": str(run_id), "req_id": request_id},
        )
        count = result.scalar()
    assert count == 1, "Exactly one row must exist after idempotent double-enqueue"


# ---------------------------------------------------------------------------
# submit_response idempotency: second call returns False
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_submit_response_idempotency(pg_engine_fast: AsyncEngine) -> None:
    """submit_response twice: first returns True, second returns False."""
    run_id = await _insert_run(pg_engine_fast)
    request_id = f"req-{uuid.uuid4()}"

    session_factory = create_session_factory(pg_engine_fast)
    queue = HumanRequestQueue(session_factory=session_factory)

    await queue.enqueue(run_id=run_id, request_id=request_id, context_json={})

    first = await queue.submit_response(
        run_id=run_id, request_id=request_id, response_json={"answer": "yes"}
    )
    second = await queue.submit_response(
        run_id=run_id, request_id=request_id, response_json={"answer": "no"}
    )

    assert first is True, "First submit_response must return True"
    assert second is False, "Second submit_response must return False (response_json already set)"

    # Confirm the original response_json is preserved
    async with pg_engine_fast.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT response_json FROM human_request_queue "
                "WHERE run_id = :run_id AND request_id = :req_id"
            ),
            {"run_id": str(run_id), "req_id": request_id},
        )
        row = result.fetchone()
    assert row is not None
    assert row[0]["answer"] == "yes", "First response_json must be preserved (idempotent)"


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_cancel_unanswered_row(pg_engine_fast: AsyncEngine) -> None:
    """cancel sets status='cancelled' on an unanswered row and returns True."""
    run_id = await _insert_run(pg_engine_fast)
    request_id = f"req-{uuid.uuid4()}"

    session_factory = create_session_factory(pg_engine_fast)
    queue = HumanRequestQueue(session_factory=session_factory)

    await queue.enqueue(run_id=run_id, request_id=request_id, context_json={})
    cancelled = await queue.cancel(run_id=run_id, request_id=request_id)
    assert cancelled is True

    # Verify status in DB
    async with pg_engine_fast.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT status FROM human_request_queue "
                "WHERE run_id = :run_id AND request_id = :req_id"
            ),
            {"run_id": str(run_id), "req_id": request_id},
        )
        row = result.fetchone()
    assert row is not None
    assert row[0] == "cancelled"


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_cancel_already_answered_row(pg_engine_fast: AsyncEngine) -> None:
    """cancel returns False when response_json is already set (row already answered)."""
    run_id = await _insert_run(pg_engine_fast)
    request_id = f"req-{uuid.uuid4()}"

    session_factory = create_session_factory(pg_engine_fast)
    queue = HumanRequestQueue(session_factory=session_factory)

    await queue.enqueue(run_id=run_id, request_id=request_id, context_json={})
    await queue.submit_response(
        run_id=run_id, request_id=request_id, response_json={"answer": "approved"}
    )

    cancelled = await queue.cancel(run_id=run_id, request_id=request_id)
    assert cancelled is False, "cancel must return False when response already set"


# ---------------------------------------------------------------------------
# claim atomicity: double-claim
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_claim_atomicity_double_claim(pg_engine_fast: AsyncEngine) -> None:
    """Two concurrent claims on the same row: exactly one succeeds."""
    run_id = await _insert_run(pg_engine_fast)
    request_id = f"req-{uuid.uuid4()}"

    session_factory = create_session_factory(pg_engine_fast)
    queue = HumanRequestQueue(session_factory=session_factory)

    await queue.enqueue(run_id=run_id, request_id=request_id, context_json={})
    rows = await queue.fetch_pending(limit=10)
    our_row = next(r for r in rows if r.request_id == request_id)

    # Both participants try to claim simultaneously
    results = await asyncio.gather(
        queue.claim(queue_row_id=our_row.id, participant_id="alice"),
        queue.claim(queue_row_id=our_row.id, participant_id="bob"),
        return_exceptions=True,
    )

    successes = [r for r in results if r is True]
    failures = [r for r in results if r is False]
    assert len(successes) == 1, "Exactly one claim must succeed"
    assert len(failures) == 1, "Exactly one claim must fail"
