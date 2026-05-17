"""E2E integration test — StreamlitHumanGateway + HumanRequestQueue via real PG (M14 §6.2).

Acceptance criteria (plan Step 6.2):
  - Full end-to-end path: queue → gateway → callback → DB write.
  - ``human_interactions`` row has non-null ``tlx_scores``, ``raw_tlx_score``,
    and ``study_session_id`` matching the submitted values.

Test strategy
-------------
Instead of running ``run_one`` (which requires a full ExperimentConfig + LLM
fixtures + LangGraph compilation), we drive the gateway layer directly:

  1. Insert Experiment + Run rows in PG to satisfy FK constraints.
  2. Build ``HumanRequestQueue`` + ``StreamlitHumanGateway`` from test config.
  3. Concurrently (``asyncio.gather``):
       a. ``gateway.request(ctx, request_id=...)`` — enqueues, then polls.
       b. ``FakeStreamlitClient.run()`` — polls fetch_pending → claim →
          submit_response with all 6 TLX scales.
  4. Dispatch ``human_response`` through ``ExperimentCallbackHandler`` (same
     pattern as test_m9_hitl.py Scenario A) to drive the DB write.
  5. Assert ``human_interactions`` has exactly 1 row with correct TLX data.

This approach validates the entire queue → gateway → callback → DB path while
remaining fully deterministic (no real LLM, no LangGraph graph compilation).

Marker: ``@pytest.mark.requires_postgres``
Skip gate: ``ATM_ENABLE_PG_TESTS != 1``
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableLambda
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from atm.core.types import HumanContext, HumanRole, Message, MessageKind
from atm.experiment.config import HumanCfg
from atm.human._queue import HumanRequestQueue
from atm.human.streamlit_gateway import StreamlitHumanGateway
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.models import Base, Experiment, HumanInteraction, Run, StudySession
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_session_factory, session_scope

# ---------------------------------------------------------------------------
# NASA-TLX payload submitted by the FakeStreamlitClient
# ---------------------------------------------------------------------------

_TLX_SCORES: dict[str, int] = {
    "mental_demand": 70,
    "physical_demand": 20,
    "temporal_demand": 50,
    "performance": 60,
    "effort": 65,
    "frustration": 40,
}

# Expected raw_tlx_score from NasaTLX(**_TLX_SCORES).raw_score:
#   (70 + 20 + 50 + (100-60) + 65 + 40) / 6 = 285 / 6 = 47.5
_EXPECTED_RAW_TLX: float = (
    _TLX_SCORES["mental_demand"]
    + _TLX_SCORES["physical_demand"]
    + _TLX_SCORES["temporal_demand"]
    + (100 - _TLX_SCORES["performance"])
    + _TLX_SCORES["effort"]
    + _TLX_SCORES["frustration"]
) / 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_experiment_run_and_session(
    factory: Any,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
    study_session_id: uuid.UUID,
    participant_id: str,
) -> None:
    """Insert Experiment + Run + StudySession rows to satisfy FK constraints.

    StudySession must be inserted before HumanInteraction because
    ``human_interactions.study_session_id`` has a FK to ``study_sessions.id``.
    """
    async with session_scope(factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"m14-e2e-test-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="chain",
                task_id="m14-e2e-task",
                agent_set="canonical_4",
                seed=42,
                model="fake:echo",
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
            )
        )
    async with session_scope(factory) as session:
        session.add(
            StudySession(
                id=study_session_id,
                participant_id=participant_id,
                consent_given=True,
                status="active",
                meta_json={},
            )
        )


def _make_human_context(run_id: uuid.UUID) -> HumanContext:
    """Build a minimal HumanContext for the test interaction."""
    msg = Message(
        sender="critic",
        kind=MessageKind.DECISION,
        content="Does the solution look correct? Please approve, reject, or abstain.",
    )
    return HumanContext(
        run_id=run_id,
        role=HumanRole.REVIEWER,
        question="Does the solution look correct?",
        recent_messages=(msg,),
        allowed_actions=("approve", "reject", "abstain"),
    )


# ---------------------------------------------------------------------------
# FakeStreamlitClient
# ---------------------------------------------------------------------------


class FakeStreamlitClient:
    """Simulates a Streamlit UI participant — polls queue and submits a response.

    Flow (mirrors Streamlit UI behaviour):
      1. Poll ``fetch_pending()`` until a row appears (up to ``max_polls``).
      2. ``claim()`` the row on behalf of ``participant_id``.
      3. ``submit_response()`` with action + TLX scores + study_session_id.
    """

    def __init__(
        self,
        queue: HumanRequestQueue,
        run_id: uuid.UUID,
        participant_id: str,
        study_session_id: str,
        *,
        poll_interval_s: float = 0.1,
        max_polls: int = 100,
    ) -> None:
        self._queue = queue
        self._run_id = run_id
        self._participant_id = participant_id
        self._study_session_id = study_session_id
        self._poll_interval_s = poll_interval_s
        self._max_polls = max_polls

    async def run(self) -> dict[str, Any] | None:
        """Poll → claim → submit; return the response dict submitted, or None on timeout."""
        # Step 1: poll until pending row appears
        for _ in range(self._max_polls):
            rows = await self._queue.fetch_pending(participant_id=self._participant_id, limit=5)
            if rows:
                break
            await asyncio.sleep(self._poll_interval_s)
        else:
            # Timed out waiting for a queue entry
            return None

        row = rows[0]

        # Step 2: claim
        await self._queue.claim(row.id, participant_id=self._participant_id)

        # Step 3: submit response with TLX scores and study_session_id
        response_payload: dict[str, Any] = {
            "action": "approve",
            "comment": "Looks good to me.",
            "payload": {"study_session_id": self._study_session_id},
            "study_session_id": self._study_session_id,
            "tlx_scores": _TLX_SCORES,
        }
        await self._queue.submit_response(
            run_id=self._run_id,
            request_id=row.request_id,
            response_json=response_payload,
        )
        return response_payload


# ---------------------------------------------------------------------------
# Main E2E test
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
async def test_streamlit_gateway_e2e_queue_to_db(
    pg_engine_fast: AsyncEngine,
    truncate_m14_tables: None,
    tmp_path: Path,
) -> None:
    """End-to-end: StreamlitGateway enqueues → FakeClient responds → CB writes DB row.

    Assertions:
      - ``human_interactions`` contains exactly 1 row for the test run_id.
      - ``tlx_scores`` JSON matches ``_TLX_SCORES``.
      - ``raw_tlx_score`` is not null and matches expected value.
      - ``study_session_id`` matches the UUID we submitted.
    """
    # -----------------------------------------------------------------------
    # Setup: identifiers
    # -----------------------------------------------------------------------
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    study_session_uuid = uuid.uuid4()
    study_session_id = str(study_session_uuid)
    participant_id = "user_test"
    request_id = "chain:0:reviewer"

    # -----------------------------------------------------------------------
    # Setup: DB rows + session factory
    # -----------------------------------------------------------------------
    session_factory = create_session_factory(pg_engine_fast)
    await _insert_experiment_run_and_session(
        session_factory,
        exp_id=exp_id,
        run_id=run_id,
        study_session_id=study_session_uuid,
        participant_id=participant_id,
    )

    # -----------------------------------------------------------------------
    # Setup: HumanCfg for streamlit gateway
    # -----------------------------------------------------------------------
    # We reuse the pg_engine_fast DSN (same DB) for the queue.
    # pg_engine_fast.url is a sqlalchemy URL object; convert to string.
    queue_dsn = str(pg_engine_fast.url)
    # Ensure the driver scheme is asyncpg-compatible.
    if queue_dsn.startswith("postgresql://") and "+asyncpg" not in queue_dsn:
        queue_dsn = queue_dsn.replace("postgresql://", "postgresql+asyncpg://", 1)

    human_cfg = HumanCfg(
        enabled=True,
        gateway="streamlit",
        queue_dsn=queue_dsn,
        study_session_id=study_session_id,
        participant_id=participant_id,
        timeout_s=10.0,
        timeout_policy="llm_fallback",
    )

    # -----------------------------------------------------------------------
    # Setup: queue + gateway
    # -----------------------------------------------------------------------
    queue = HumanRequestQueue(session_factory=session_factory)
    gateway = StreamlitHumanGateway(
        queue,
        human_cfg=human_cfg,
        study_session_id=study_session_id,
        participant_id=participant_id,
    )

    # -----------------------------------------------------------------------
    # Setup: fake Streamlit client
    # -----------------------------------------------------------------------
    fake_client = FakeStreamlitClient(
        queue=queue,
        run_id=run_id,
        participant_id=participant_id,
        study_session_id=study_session_id,
        poll_interval_s=0.1,
        max_polls=100,
    )

    # -----------------------------------------------------------------------
    # Setup: ExperimentCallbackHandler (for the DB write)
    # -----------------------------------------------------------------------
    parquet_writer = ParquetWriter(tmp_path, run_id, exp_id)
    handler = ExperimentCallbackHandler(
        run_id=run_id,
        exp_id=exp_id,
        session_factory=session_factory,
        parquet_writer=parquet_writer,
        budget_warn_threshold=Decimal("100"),
        budget_exceed_threshold=Decimal("1000"),
    )

    # -----------------------------------------------------------------------
    # Build HumanContext
    # -----------------------------------------------------------------------
    ctx = _make_human_context(run_id)

    # -----------------------------------------------------------------------
    # Coroutine A: gateway.request() — enqueues then polls for response
    # -----------------------------------------------------------------------
    async def _gateway_request() -> Any:
        return await gateway.request(ctx, request_id=request_id)

    # -----------------------------------------------------------------------
    # Coroutine B: FakeStreamlitClient — polls, claims, submits
    # -----------------------------------------------------------------------
    async def _fake_client_run() -> dict[str, Any] | None:
        return await fake_client.run()

    # -----------------------------------------------------------------------
    # Run both concurrently — gateway polls for a response while client submits
    # -----------------------------------------------------------------------
    gateway_result, client_result = await asyncio.gather(
        _gateway_request(),
        _fake_client_run(),
    )

    # Verify the gateway received a valid response (not a timeout)
    assert gateway_result is not None, "gateway.request() returned None unexpectedly"
    assert not gateway_result.timed_out, (
        f"gateway.request() timed out — FakeStreamlitClient may have failed to submit. "
        f"client_result={client_result!r}"
    )
    assert gateway_result.action == "approve"
    assert gateway_result.tlx_scores == _TLX_SCORES

    # Verify client submitted successfully
    assert client_result is not None, "FakeStreamlitClient failed to submit a response"

    # -----------------------------------------------------------------------
    # Dispatch human_request + human_response events via callback handler
    # to drive the DB write (mirrors test_m9_hitl.py Scenario A pattern).
    # -----------------------------------------------------------------------
    response_json = gateway_result.model_dump(mode="json")

    async def _dispatch_hitl_events(inputs: dict[str, Any]) -> dict[str, Any]:
        # Dispatch human_request (insert placeholder row)
        await adispatch_custom_event(
            "human_request",
            {
                "run_id": run_id,
                "request_id": request_id,
                "role": str(ctx.role.value),
                "context_json": ctx.model_dump(mode="json"),
                "requested_at": datetime.now(UTC),
            },
        )
        # Dispatch human_response (write TLX + study_session_id)
        await adispatch_custom_event(
            "human_response",
            {
                "run_id": run_id,
                "request_id": request_id,
                "answered_at": datetime.now(UTC),
                "response_json": response_json,
                "source": gateway_result.source,
                "timed_out": gateway_result.timed_out,
                "latency_s": 0.05,
            },
        )
        return inputs

    runnable: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
        _dispatch_hitl_events
    )
    await runnable.ainvoke(
        {},
        config={
            "callbacks": [handler],
            "metadata": {"is_root_run": True},
            "run_id": uuid.uuid4(),
        },
    )

    # Allow async DB writes to complete
    await asyncio.sleep(0.15)

    # -----------------------------------------------------------------------
    # Assert: human_interactions row has all three M14 fields
    # -----------------------------------------------------------------------
    async with session_scope(session_factory) as session:
        result = await session.execute(
            select(HumanInteraction).where(HumanInteraction.run_id == run_id)
        )
        rows = result.scalars().all()

    assert len(rows) == 1, (
        f"Expected exactly 1 human_interactions row for run_id={run_id}, "
        f"got {len(rows)}"
    )
    hi = rows[0]

    # 1. tlx_scores must not be null and must match submitted values
    assert hi.tlx_scores is not None, "human_interactions.tlx_scores is NULL — TLX not persisted"
    assert hi.tlx_scores == _TLX_SCORES, (
        f"tlx_scores mismatch: expected {_TLX_SCORES!r}, got {hi.tlx_scores!r}"
    )

    # 2. raw_tlx_score must be computed and not null
    assert hi.raw_tlx_score is not None, (
        "human_interactions.raw_tlx_score is NULL — NasaTLX aggregation not persisted"
    )
    assert abs(hi.raw_tlx_score - _EXPECTED_RAW_TLX) < 0.01, (
        f"raw_tlx_score mismatch: expected {_EXPECTED_RAW_TLX}, got {hi.raw_tlx_score}"
    )

    # 3. study_session_id must match what we submitted
    assert hi.study_session_id is not None, (
        "human_interactions.study_session_id is NULL — not propagated from payload"
    )
    assert str(hi.study_session_id) == study_session_id, (
        f"study_session_id mismatch: expected {study_session_id!r}, "
        f"got {hi.study_session_id!r}"
    )
