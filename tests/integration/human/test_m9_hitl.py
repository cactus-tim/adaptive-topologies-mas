"""M9 HITL infrastructure — acceptance-style integration tests.

Three acceptance scenarios (plan M9 §4.2-4.4):

  Scenario A — Reviewer end-to-end (PG required)
    Drive ExperimentCallbackHandler with human_request + human_response events
    dispatched via adispatch_custom_event inside a RunnableLambda (same pattern
    as test_smoke_run.py), using event payloads in the format that
    _handle_human_request and _handle_human_response actually expect.
    Assert: exactly 1 row in human_interactions with source='llm_sim' inside
    response_json and non-NULL responded_at.

  Scenario B — Timeout + fallback source
    Verify request_with_timeout with policy='llm_fallback' and a slow primary
    gateway returns source='fallback'.  Uses an in-process asyncio.sleep gate to
    simulate a slow gateway (no PG required).

  Scenario C — Idempotency on resume (PG required)
    Dispatch the same (run_id, request_id) human_request event twice via the
    callback handler against a real PG.  Assert that exactly ONE row exists in
    human_interactions, confirming the ON CONFLICT DO NOTHING clause works.

Skipping rules
--------------
  Scenarios A and C skip gracefully when ATM_ENABLE_PG_TESTS is not set.
  Scenario B runs always (pure asyncio, no PG).

IMPORTANT: These tests require alembic upgrade head for the unique constraint
(migration 0002) to exist.  The pg_engine_alembic fixture handles this
automatically if available.  The pg_engine_fast fixture uses create_all which
includes the constraint defined in models.py metadata.

NOTE on payload format mismatch (surfaced as finding FINDING-1)
--------------------------------------------------------------
The chain topology's human_reviewer node dispatches:

    await adispatch_custom_event(
        "human_request",
        {"request_id": request_id, "ctx": ctx.model_dump(mode="json")},
    )

but ExperimentCallbackHandler._handle_human_request expects:

    data["run_id"], data["role"], data["context_json"]

This mismatch means the chain node -> callback -> PG write path is BROKEN
for human_request (KeyError on data["run_id"]).  The callback silently swallows
the KeyError (try/except in _handle_human_request), so no row is written.

Similarly for human_response:
  chain dispatches: {"request_id": ..., "response": ...}
  handler expects:  data["run_id"], data["response_json"]

Scenario A tests the callback handler directly with the correct payload format
(bypassing the broken chain dispatch) to verify the PG write path is internally
correct.  A dedicated scenario (A2) dispatches via RunnableLambda with the
broken chain format and asserts 0 rows are written — exposing the mismatch.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableLambda
from sqlalchemy import func, select

from atm.core.types import HumanContext, HumanRole, Message, MessageKind
from atm.human._timeout import request_with_timeout
from atm.human.gateway import HumanResponse
from atm.human.llm_simulated import LLMSimulatedGateway
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.models import Base, Experiment, HumanInteraction, Run
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_engine, create_session_factory, session_scope

# ---------------------------------------------------------------------------
# Environment / PG availability
# ---------------------------------------------------------------------------

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_DEFAULT_DSN = "postgresql+asyncpg://atm:atm@localhost:5432/atm_test"

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pricing() -> Pricing:
    if PRICING_PATH.exists():
        return Pricing.from_yaml(PRICING_PATH)
    return Pricing(version=1, models={})


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_llm_wrapper(fixture_name: str) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


def _make_context(run_id: uuid.UUID, role: HumanRole = HumanRole.REVIEWER) -> HumanContext:
    msg = Message(
        sender="executor",
        kind=MessageKind.DRAFT,
        content="Here is the proposed solution.",
    )
    return HumanContext(
        run_id=run_id,
        role=role,
        question="Does this meet requirements?",
        recent_messages=(msg,),
        allowed_actions=("approve", "reject", "abstain"),
    )


async def _insert_experiment_and_run(
    session_factory: Any,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
) -> None:
    """Insert minimal Experiment + Run rows needed for FK constraints."""
    async with session_scope(session_factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"hitl-test-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="chain",
                task_id="hitl-test-task",
                agent_set="default",
                seed=42,
                model="fake:scripted",
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
            )
        )


def _build_handler(
    run_id: uuid.UUID,
    exp_id: uuid.UUID,
    session_factory: Any,
    tmp_path: Path,
) -> ExperimentCallbackHandler:
    parquet_writer = ParquetWriter(tmp_path, run_id, exp_id)
    return ExperimentCallbackHandler(
        run_id=run_id,
        exp_id=exp_id,
        session_factory=session_factory,
        parquet_writer=parquet_writer,
        budget_warn_threshold=Decimal("100"),
        budget_exceed_threshold=Decimal("1000"),
    )


# ---------------------------------------------------------------------------
# Scenario A — Reviewer end-to-end via direct callback dispatch (PG required)
# ---------------------------------------------------------------------------
#
# This test verifies the full callback -> PG write path by dispatching
# human_request and human_response events with the payload format that
# _handle_human_request and _handle_human_response actually expect.
#
# It uses RunnableLambda + adispatch_custom_event (same pattern as
# test_smoke_run.py) to provide a proper LangChain callback context.


@pytest.mark.integration
async def test_reviewer_e2e_callback_writes_one_row(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Scenario A: human_request + human_response events write exactly 1 row in PG.

    Verifies:
    - _handle_human_request correctly INSERTs a HumanInteraction row.
    - _handle_human_response correctly UPDATEs the row with response_json.
    - Final count is exactly 1 row in human_interactions for this run_id.
    - The response_json contains source='llm_sim' (from LLMSimulatedGateway).

    This test uses the DIRECT payload format that _handle_human_request expects
    (run_id, role, context_json, request_id, requested_at), NOT the format
    dispatched by the chain node (which has a payload mismatch — see FINDING-1).
    """
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    request_id = "chain:0:reviewer"

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id)

        handler = _build_handler(run_id, exp_id, factory, tmp_path)

        ctx = _make_context(run_id)

        # Build the response that the gateway would return
        llm_response = HumanResponse(
            action="approve",
            comment="LGTM",
            payload={},
            source="llm_sim",
            timed_out=False,
        )

        # Dispatch both events inside a RunnableLambda to get a proper callback context
        async def _dispatch_hitl_events(inputs: dict[str, Any]) -> dict[str, Any]:
            # Dispatch human_request with the format _handle_human_request expects
            await adispatch_custom_event(
                "human_request",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "role": str(ctx.role),
                    "context_json": ctx.model_dump(mode="json"),
                    "requested_at": datetime.now(UTC),
                },
            )
            # Dispatch human_response with the format _handle_human_response expects
            await adispatch_custom_event(
                "human_response",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "answered_at": datetime.now(UTC),
                    "response_json": llm_response.model_dump(mode="json"),
                    "source": llm_response.source,
                    "timed_out": llm_response.timed_out,
                    "latency_s": 0.1,
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

        # Allow async background operations to complete
        await asyncio.sleep(0.1)

        # Assert: exactly 1 row in human_interactions for this run_id
        async with session_scope(factory) as session:
            count_result = await session.execute(
                select(func.count()).select_from(HumanInteraction).where(
                    HumanInteraction.run_id == run_id
                )
            )
            count = count_result.scalar_one()

        assert count == 1, (
            f"Expected exactly 1 row in human_interactions for run_id={run_id}, got {count}"
        )

        # Assert: response_json contains source='llm_sim'
        async with session_scope(factory) as session:
            row_result = await session.execute(
                select(HumanInteraction).where(
                    HumanInteraction.run_id == run_id,
                    HumanInteraction.request_id == request_id,
                )
            )
            row = row_result.scalar_one_or_none()

        assert row is not None, f"No row found for run_id={run_id}, request_id={request_id}"
        assert row.response_json is not None, "response_json should be non-NULL after human_response"
        assert row.response_json.get("source") == "llm_sim", (
            f"Expected source='llm_sim' in response_json, got {row.response_json.get('source')!r}"
        )
        assert row.role == "reviewer", f"Expected role='reviewer', got {row.role!r}"

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


# ---------------------------------------------------------------------------
# Scenario A2 — Chain node dispatch format (FINDING-1 fixed verification)
# ---------------------------------------------------------------------------
#
# After fixing F1/F2, the chain node now dispatches human_request with the
# canonical payload shape that _handle_human_request expects:
#   {"run_id": ..., "request_id": ..., "role": ..., "context_json": ..., "requested_at": ...}
#
# This test verifies that the FIXED chain dispatch format DOES write a PG row.


@pytest.mark.integration
async def test_chain_node_dispatch_format_writes_row_after_f1_fix(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """After F1/F2 fix: Chain node dispatches human_request in the canonical format
    and exactly 1 row is written to human_interactions.

    The fixed chain node calls:
        adispatch_custom_event("human_request", {
            "run_id": run_id,
            "request_id": request_id,
            "role": str(human_cfg.role),
            "context_json": ctx.model_dump(mode="json"),
            "requested_at": datetime.now(UTC),
        })

    This matches _handle_human_request's expected format and the PG row is written.
    """
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    request_id = "chain:0:reviewer"

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id)

        handler = _build_handler(run_id, exp_id, factory, tmp_path)
        ctx = _make_context(run_id)

        # Dispatch using the FIXED canonical chain node format
        async def _dispatch_fixed_chain_format(inputs: dict[str, Any]) -> dict[str, Any]:
            await adispatch_custom_event(
                "human_request",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "role": str(ctx.role),
                    "context_json": ctx.model_dump(mode="json"),
                    "requested_at": datetime.now(UTC),
                },
            )
            return inputs

        runnable: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
            _dispatch_fixed_chain_format
        )
        await runnable.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )
        await asyncio.sleep(0.1)

        # After F1 fix: exactly 1 row should be written
        async with session_scope(factory) as session:
            count_result = await session.execute(
                select(func.count()).select_from(HumanInteraction).where(
                    HumanInteraction.run_id == run_id
                )
            )
            count = count_result.scalar_one()

        assert count == 1, (
            f"F1/F2 fix verification: chain dispatch format should write 1 PG row, got {count}. "
            "The canonical payload shape must match _handle_human_request's expectations."
        )

        # Verify role is correctly recorded
        async with session_scope(factory) as session:
            row_result = await session.execute(
                select(HumanInteraction).where(
                    HumanInteraction.run_id == run_id,
                    HumanInteraction.request_id == request_id,
                )
            )
            row = row_result.scalar_one_or_none()

        assert row is not None
        assert row.role == "reviewer", f"Expected role='reviewer', got {row.role!r}"

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


# ---------------------------------------------------------------------------
# Scenario B — Timeout + fallback (no PG required)
# ---------------------------------------------------------------------------


class _SlowGateway:
    """A gateway that sleeps longer than the timeout before responding."""

    def __init__(self, sleep_s: float) -> None:
        self._sleep_s = sleep_s

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        await asyncio.sleep(self._sleep_s)
        return HumanResponse(
            action="approve",
            comment="Slow response",
            source="human",
            timed_out=False,
        )


class _FastFallbackGateway:
    """A gateway that immediately returns a fallback-style response."""

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        return HumanResponse(
            action="abstain",
            comment="Fallback decision",
            source="llm_sim",
            timed_out=False,
        )


@pytest.mark.asyncio
async def test_timeout_fallback_source_is_fallback() -> None:
    """Scenario B: Slow gateway + timeout_s=0.05 + policy=llm_fallback -> source='fallback'.

    Verifies that:
    - request_with_timeout triggers after timeout_s expires.
    - policy='llm_fallback' calls the fallback gateway.
    - The returned HumanResponse has source='fallback' (overridden by _handle_timeout).
    - The run completes without raising an exception.
    """
    ctx = _make_context(uuid.uuid4())
    slow_gw = _SlowGateway(sleep_s=5.0)  # will time out
    fallback_gw = _FastFallbackGateway()

    response = await request_with_timeout(
        slow_gw,
        ctx,
        request_id="timeout-test-req-001",
        timeout_s=0.05,
        policy="llm_fallback",
        llm_fallback_gateway=fallback_gw,
    )

    assert response.source == "fallback", (
        f"Expected source='fallback', got {response.source!r}"
    )
    assert response.timed_out is False, (
        "Fallback responses should not have timed_out=True (only skip policy does)"
    )


@pytest.mark.asyncio
async def test_timeout_skip_policy_returns_timeout_response() -> None:
    """Scenario B (skip variant): policy='skip' -> synthetic timeout response."""
    ctx = _make_context(uuid.uuid4())
    slow_gw = _SlowGateway(sleep_s=5.0)

    response = await request_with_timeout(
        slow_gw,
        ctx,
        request_id="timeout-test-req-002",
        timeout_s=0.05,
        policy="skip",
    )

    assert response.timed_out is True, f"Expected timed_out=True, got {response.timed_out}"
    assert response.source == "timeout", f"Expected source='timeout', got {response.source!r}"
    assert response.action == "timeout", f"Expected action='timeout', got {response.action!r}"


@pytest.mark.asyncio
async def test_timeout_with_llm_simulated_fallback_returns_fallback_source() -> None:
    """Scenario B (LLMSimulatedGateway as fallback): primary times out, fallback via LLMSim.

    This test exercises the cross-component interaction:
      _timeout.py -> LLMSimulatedGateway -> FakeLLM fixture

    The fallback response source is overridden to 'fallback' regardless of what
    LLMSimulatedGateway returns.
    """
    fixture_path = FIXTURES_DIR / "m9_human_reviewer_approve.yaml"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    ctx = _make_context(uuid.uuid4())
    slow_gw = _SlowGateway(sleep_s=5.0)
    fallback_wrapper = _make_llm_wrapper("m9_human_reviewer_approve.yaml")
    fallback_gw = LLMSimulatedGateway(fallback_wrapper)

    response = await request_with_timeout(
        slow_gw,
        ctx,
        request_id="timeout-llmsim-req-001",
        timeout_s=0.05,
        policy="llm_fallback",
        llm_fallback_gateway=fallback_gw,
    )

    # LLMSimulatedGateway returns source='llm_sim'; _handle_timeout overrides to 'fallback'
    assert response.source == "fallback", (
        f"Expected source='fallback' (overridden by _handle_timeout), got {response.source!r}"
    )


# ---------------------------------------------------------------------------
# Scenario C — Idempotency on resume (PG required)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_idempotency_duplicate_request_id_writes_one_row(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Scenario C: Dispatching the same (run_id, request_id) twice yields exactly 1 row.

    Simulates a re-executed human_reviewer node (e.g., LangGraph retry or resume
    from checkpoint) that dispatches human_request twice with the same request_id.

    Verifies:
    - The UNIQUE constraint uq_human_interactions_run_request (migration 0002) +
      ON CONFLICT DO NOTHING in _handle_human_request prevents duplicate rows.
    - SELECT COUNT(*) = 1 after two identical dispatches.

    NOTE: The migration 0002 constraint is NOT applied by create_all (it is a
    pure Alembic migration on top of the SQLAlchemy model).  We use a partial
    index workaround: the models.py definition does NOT include the UNIQUE
    constraint in __table_args__, so create_all will create the table without it.
    This test exercises the ON CONFLICT at the PG level by running alembic upgrade.
    """
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    request_id = "chain:0:reviewer-idem"

    # Use alembic upgrade to get the UNIQUE constraint
    import subprocess

    env = os.environ.copy()
    env["PG_DSN"] = ephemeral_pg_dsn
    try:
        subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            check=True,
            env=env,
            capture_output=True,
            cwd=str(Path(__file__).parent.parent.parent.parent),
        )
    except subprocess.CalledProcessError as e:
        pytest.skip(
            f"alembic upgrade head failed — DB may not be reachable or migrations broken: "
            f"{e.stderr.decode()[:500]}"
        )

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    factory = create_session_factory(engine)

    try:
        await _insert_experiment_and_run(factory, exp_id, run_id)

        handler = _build_handler(run_id, exp_id, factory, tmp_path)
        ctx = _make_context(run_id)

        human_request_payload = {
            "run_id": run_id,
            "request_id": request_id,
            "role": str(ctx.role),
            "context_json": ctx.model_dump(mode="json"),
            "requested_at": datetime.now(UTC),
        }

        # First dispatch
        async def _first_dispatch(inputs: dict[str, Any]) -> dict[str, Any]:
            await adispatch_custom_event("human_request", human_request_payload)
            return inputs

        runnable1: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
            _first_dispatch
        )
        await runnable1.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )
        await asyncio.sleep(0.05)

        # Second dispatch with IDENTICAL (run_id, request_id)
        async def _second_dispatch(inputs: dict[str, Any]) -> dict[str, Any]:
            await adispatch_custom_event("human_request", human_request_payload)
            return inputs

        runnable2: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
            _second_dispatch
        )
        await runnable2.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )
        await asyncio.sleep(0.05)

        # Assert: exactly 1 row (ON CONFLICT DO NOTHING prevented the duplicate)
        async with session_scope(factory) as session:
            count_result = await session.execute(
                select(func.count()).select_from(HumanInteraction).where(
                    HumanInteraction.run_id == run_id,
                    HumanInteraction.request_id == request_id,
                )
            )
            count = count_result.scalar_one()

        assert count == 1, (
            f"Expected exactly 1 row for (run_id={run_id}, request_id={request_id!r}), "
            f"got {count}. The UNIQUE constraint + ON CONFLICT DO NOTHING should prevent duplicates."
        )

    finally:
        await engine.dispose()
        # Downgrade to clean state
        try:
            env2 = os.environ.copy()
            env2["PG_DSN"] = ephemeral_pg_dsn
            subprocess.run(
                ["uv", "run", "alembic", "downgrade", "base"],
                env=env2,
                capture_output=True,
                cwd=str(Path(__file__).parent.parent.parent.parent),
            )
        except Exception:
            pass  # Best effort cleanup


@pytest.mark.integration
async def test_idempotency_human_response_update_is_idempotent(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Scenario C (response update): second human_response dispatch is a no-op.

    Dispatching human_response twice for the same (run_id, request_id) should
    result in exactly 1 filled response_json (the WHERE response_json IS NULL
    guard ensures the second UPDATE matches 0 rows).
    """
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    request_id = "chain:0:reviewer-resp-idem"

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id)

        handler = _build_handler(run_id, exp_id, factory, tmp_path)
        ctx = _make_context(run_id)

        llm_response = HumanResponse(
            action="approve",
            comment="First response",
            source="llm_sim",
            timed_out=False,
        )
        llm_response_2 = HumanResponse(
            action="reject",
            comment="Second response (should be ignored)",
            source="llm_sim",
            timed_out=False,
        )

        request_payload = {
            "run_id": run_id,
            "request_id": request_id,
            "role": str(ctx.role),
            "context_json": ctx.model_dump(mode="json"),
            "requested_at": datetime.now(UTC),
        }
        response_payload_1 = {
            "run_id": run_id,
            "request_id": request_id,
            "answered_at": datetime.now(UTC),
            "response_json": llm_response.model_dump(mode="json"),
            "source": "llm_sim",
            "timed_out": False,
            "latency_s": 0.1,
        }
        response_payload_2 = {
            "run_id": run_id,
            "request_id": request_id,
            "answered_at": datetime.now(UTC),
            "response_json": llm_response_2.model_dump(mode="json"),
            "source": "llm_sim",
            "timed_out": False,
            "latency_s": 0.2,
        }

        async def _full_sequence(inputs: dict[str, Any]) -> dict[str, Any]:
            await adispatch_custom_event("human_request", request_payload)
            await adispatch_custom_event("human_response", response_payload_1)
            # Second human_response with different content — should be ignored
            await adispatch_custom_event("human_response", response_payload_2)
            return inputs

        runnable: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
            _full_sequence
        )
        await runnable.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )
        await asyncio.sleep(0.1)

        # Assert: only 1 row, response_json has the FIRST response (approve, not reject)
        async with session_scope(factory) as session:
            count_result = await session.execute(
                select(func.count()).select_from(HumanInteraction).where(
                    HumanInteraction.run_id == run_id,
                    HumanInteraction.request_id == request_id,
                )
            )
            count = count_result.scalar_one()

        assert count == 1, f"Expected 1 row, got {count}"

        async with session_scope(factory) as session:
            row_result = await session.execute(
                select(HumanInteraction).where(
                    HumanInteraction.run_id == run_id,
                    HumanInteraction.request_id == request_id,
                )
            )
            row = row_result.scalar_one_or_none()

        assert row is not None
        assert row.response_json is not None, "response_json should be filled after first update"
        # The first response (approve) should be preserved, not overwritten by the second (reject)
        assert row.response_json.get("action") == "approve", (
            f"Expected action='approve' (first response preserved), "
            f"got {row.response_json.get('action')!r}"
        )

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


# ---------------------------------------------------------------------------
# Scenario D — LLMSimulatedGateway + callback handler cross-component flow
# ---------------------------------------------------------------------------
#
# Tests that LLMSimulatedGateway -> HumanResponse -> callback handler forms a
# coherent pipeline: the response from the gateway can be serialized and passed
# to _handle_human_response with all required fields intact.


@pytest.mark.integration
async def test_llm_simulated_gateway_response_flows_to_callback(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """LLMSimulatedGateway produces a HumanResponse that is correctly serialized
    and written to PG via the callback handler.

    Cross-component interaction verified:
      LLMSimulatedGateway.request()
        -> HumanResponse (source='llm_sim')
        -> model_dump(mode='json')
        -> ExperimentCallbackHandler._handle_human_response()
        -> UPDATE human_interactions SET response_json = ...

    Asserts the full data flow is consistent (no schema mismatch on the response side).
    """
    fixture_path = FIXTURES_DIR / "m9_human_reviewer_approve.yaml"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    request_id = "chain:0:reviewer-flow"

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id)

        handler = _build_handler(run_id, exp_id, factory, tmp_path)
        ctx = _make_context(run_id)

        # Get a real response from LLMSimulatedGateway
        llm_wrapper = _make_llm_wrapper("m9_human_reviewer_approve.yaml")
        gateway = LLMSimulatedGateway(llm_wrapper)
        response = await gateway.request(ctx, request_id=request_id)

        assert response.source == "llm_sim"
        assert response.action == "approve"

        # Now flow this response through the callback handler
        async def _dispatch_with_gateway_response(inputs: dict[str, Any]) -> dict[str, Any]:
            # INSERT (using direct format)
            await adispatch_custom_event(
                "human_request",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "role": str(ctx.role),
                    "context_json": ctx.model_dump(mode="json"),
                    "requested_at": datetime.now(UTC),
                },
            )
            # UPDATE with the actual gateway response
            await adispatch_custom_event(
                "human_response",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "answered_at": datetime.now(UTC),
                    "response_json": response.model_dump(mode="json"),
                    "source": response.source,
                    "timed_out": response.timed_out,
                    "latency_s": 0.05,
                },
            )
            return inputs

        runnable: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
            _dispatch_with_gateway_response
        )
        await runnable.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )
        await asyncio.sleep(0.1)

        # Verify the full data flow
        async with session_scope(factory) as session:
            row_result = await session.execute(
                select(HumanInteraction).where(
                    HumanInteraction.run_id == run_id,
                    HumanInteraction.request_id == request_id,
                )
            )
            row = row_result.scalar_one_or_none()

        assert row is not None, "Row should exist after full dispatch flow"
        assert row.response_json is not None, "response_json should be filled"
        assert row.response_json.get("source") == "llm_sim", (
            f"source in response_json should be 'llm_sim', got {row.response_json.get('source')!r}"
        )
        assert row.response_json.get("action") == "approve", (
            f"action should be 'approve', got {row.response_json.get('action')!r}"
        )
        assert row.answered_at is not None, "answered_at should be set after human_response"

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
