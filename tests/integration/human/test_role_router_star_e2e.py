"""Integration test: RuleBasedRoleRouter end-to-end — multi-phase HITL with Postgres.

Step 7.1 (m9.2 plan): Verify that:
  1. >=2 distinct roles appear in human_interactions for a run when RuleBasedRoleRouter
     is used (DEFAULT_ROLE_TABLE: planning->coordinator, execution->peer, verification->reviewer).
  2. runs.human_role matches the LAST human_interactions.role (chronologically).
  3. runs.cognitive_load_proxy > 0.0 after finalization.

Implementation note — Star single-phase limitation:
  Star topology's human_reviewer node only fires during the verification phase
  (critic -> critic_postprocess -> human_reviewer -> coordinator).  Running Star
  end-to-end would produce only one distinct role (reviewer) for all HITL calls,
  making it impossible to exercise >=2 roles without patching the graph.

  Following the plan's documented fallback (m9.2-plan.md §7.1 CRITICAL note), we
  use direct event dispatching via RunnableLambda (same pattern as
  test_hitl_adaptive.py and test_hitl_cross_topology.py adaptive scenario).  This
  approach:
    - Tests the actual RuleBasedRoleRouter.decide() with real phase values.
    - Tests the SQL finalization (SELECT role FROM human_interactions ORDER BY
      requested_at DESC LIMIT 1) that runner.py performs at step 12a.
    - Tests human_sim_cognitive_load_proxy with real Postgres data.
    - Uses the established integration-test harness (ExperimentCallbackHandler,
      session_scope, ephemeral_pg_dsn fixture).

  We dispatch 3 human_request/response cycles with phases planning/execution/
  verification, letting RuleBasedRoleRouter assign coordinator/peer/reviewer.
  Then we manually replicate runner.py's step-12a finalization to update
  runs.human_role and runs.cognitive_load_proxy, and assert all three invariants.

Gate: ATM_ENABLE_PG_TESTS=1 required (uses ephemeral_pg_dsn fixture from conftest).
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
import sqlalchemy as sa
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableLambda
from sqlalchemy import select

from atm.core.types import HumanContext, HumanRole, Message, MessageKind, Phase
from atm.evaluation.metrics import human_sim_cognitive_load_proxy
from atm.human.llm_simulated import LLMSimulatedGateway
from atm.human.role_router import DEFAULT_ROLE_TABLE, RuleBasedRoleRouter
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.models import Base, Experiment, HumanInteraction, Run
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_engine, create_session_factory, session_scope

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"

_FIXTURE_NAME = "m9_2_star_role_router.yaml"

# Phases to simulate (in order) — maps to 3 distinct roles via DEFAULT_ROLE_TABLE:
#   planning   -> coordinator
#   execution  -> peer
#   verification -> reviewer
_SIMULATION_PHASES: list[Phase] = [Phase.PLANNING, Phase.EXECUTION, Phase.VERIFICATION]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pricing() -> Pricing:
    if PRICING_PATH.exists():
        return Pricing.from_yaml(PRICING_PATH)
    return Pricing(version=1, models={})


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_llm_wrapper(fixture_name: str) -> LLMWrapper:
    fixture_path = FIXTURES_DIR / fixture_name
    fake = FakeLLM(mode="scripted", fixture=fixture_path)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


def _make_human_context(
    run_id: uuid.UUID,
    role: HumanRole,
    phase: Phase,
    iter_total: int,
) -> HumanContext:
    """Build a minimal HumanContext for one simulated HITL interaction."""
    return HumanContext(
        run_id=run_id,
        role=role,
        question=f"Please review the current state at phase={phase.value} (iter={iter_total}).",
        recent_messages=(
            Message(
                sender="coordinator",
                kind=MessageKind.DRAFT,
                content=f"Phase {phase.value}: task in progress.",
            ),
        ),
        allowed_actions=("approve", "reject"),
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
                name=f"role-router-star-e2e-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="star",
                task_id="role-router-star-e2e-task",
                agent_set="default",
                seed=42,
                model="fake:scripted",
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
                # Initial static role from cfg (will be overwritten by dynamic finalization)
                human_role="reviewer",
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


async def _finalize_run(
    session_factory: Any,
    run_id: uuid.UUID,
) -> tuple[str | None, float | None]:
    """Replicate runner.py step-12a: query last role + compute cognitive_load_proxy.

    Returns:
        (dynamic_human_role, dynamic_cog_proxy) — mirrors what runner._update_run_success
        would receive.  Both may be None on DB/metric error (handled gracefully).
    """
    dynamic_human_role: str | None = None
    dynamic_cog_proxy: float | None = None

    # Step 12a-i: fetch last role from human_interactions (DESC requested_at)
    try:
        async with session_scope(session_factory) as session:
            last_role_result = await session.execute(
                sa.text(
                    "SELECT role FROM human_interactions"
                    " WHERE run_id = :rid"
                    " ORDER BY requested_at DESC LIMIT 1"
                ).bindparams(rid=run_id)
            )
            last_role_row = last_role_result.fetchone()
            if last_role_row is not None:
                dynamic_human_role = str(last_role_row[0])
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Failed to query last human_interactions.role: {exc}")

    # Step 12a-ii: compute cognitive_load_proxy
    try:
        async with session_scope(session_factory) as session:
            dynamic_cog_proxy = await human_sim_cognitive_load_proxy(session, run_id)
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Failed to compute cognitive_load_proxy: {exc}")

    # Persist both fields to runs row
    update_values: dict[str, object] = {
        "status": "completed",
        "finish_reason": "success",
        "cognitive_load_proxy": dynamic_cog_proxy,
    }
    if dynamic_human_role is not None:
        update_values["human_role"] = dynamic_human_role

    async with session_scope(session_factory) as session:
        await session.execute(
            sa.update(Run).where(Run.id == run_id).values(**update_values)
        )

    return dynamic_human_role, dynamic_cog_proxy


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_rule_based_role_router_multi_phase_writes_distinct_roles(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Star + RuleBasedRoleRouter end-to-end: >=2 distinct roles, correct human_role, cog_proxy > 0.

    Scenario:
      - RuleBasedRoleRouter with DEFAULT_ROLE_TABLE:
          planning   -> coordinator
          execution  -> peer
          verification -> reviewer
      - LLMSimulatedGateway (scripted fixture) returns "approve" for each phase.
      - 3 HITL interactions dispatched across phases: planning / execution / verification.

    Assertions:
      1. len(set(roles)) >= 2  — at least 2 distinct roles in human_interactions.
      2. runs.human_role == roles[-1]  — dynamic finalization wrote the LAST role.
      3. runs.cognitive_load_proxy > 0.0  — metric computed from 3 HITL rows.

    Implementation note:
      Star topology's human node only fires in the verification phase, making it
      impossible to exercise multiple distinct roles end-to-end without patching.
      Per m9.2-plan.md §7.1 CRITICAL note, we use direct event dispatch
      (same pattern as test_hitl_adaptive.py) to drive the full DB write + SQL
      finalization path while exercising the actual RuleBasedRoleRouter.
    """
    fixture_path = FIXTURES_DIR / _FIXTURE_NAME
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id)

        handler = _build_handler(run_id, exp_id, factory, tmp_path)

        # Build RuleBasedRoleRouter with DEFAULT_ROLE_TABLE (no custom table)
        role_router = RuleBasedRoleRouter()

        # Build LLMSimulatedGateway backed by scripted fixture
        gateway_wrapper = _make_llm_wrapper(_FIXTURE_NAME)
        gateway = LLMSimulatedGateway(llm=gateway_wrapper)

        # Shared state prototype (mirrors runner._build_initial_state)
        _shared_proto: dict[str, Any] = {
            "run_id": run_id,
            "task_id": "role-router-star-e2e-task",
            "task_input": "Classify the following sentence: integration test.",
            "phase": "planning",
            "iteration": 0,
            "iter_total": 0,
            "phase_started_at_iter": 0,
            "phase_history": [],
            "active_topology": "star",
            "topology_started_at_iter": 0,
            "topology_switch_count": 0,
            "topology_history": [],
            "final_answer": None,
            "signals": {},
            "broadcast_bus": [],
            "human_requests": [],
            "human_responses": [],
        }

        # Collect (phase, role, request_id, ctx, response) for all 3 interactions
        interactions: list[tuple[Phase, HumanRole, str, HumanContext, Any]] = []
        for i, phase in enumerate(_SIMULATION_PHASES):
            # RuleBasedRoleRouter.decide() — real call, not mocked
            phase_shared = dict(_shared_proto)
            phase_shared["phase"] = phase.value
            phase_shared["iter_total"] = i

            active_role: HumanRole = await role_router.decide(phase, phase_shared)
            request_id = f"star:{run_id}:{i}:reviewer"
            ctx = _make_human_context(run_id, active_role, phase, iter_total=i)

            # Gateway call — uses scripted fixture entries sequentially
            response = await gateway.request(ctx, request_id=request_id)

            interactions.append((phase, active_role, request_id, ctx, response))

        # --- Build async dispatch function ---
        # Dispatches all 3 human_request+response pairs inside a single RunnableLambda
        # so ExperimentCallbackHandler's callback context is active.

        async def _dispatch_all(inputs: dict[str, Any]) -> dict[str, Any]:
            """Dispatch human_request + human_response for each simulated phase."""
            for i, (_phase, active_role, req_id, ctx, resp) in enumerate(interactions):
                _requested_at = datetime.now(UTC)
                # Small sleep to ensure ascending requested_at for ORDER BY DESC check
                if i > 0:
                    await asyncio.sleep(0.01)

                # human_request event
                await adispatch_custom_event(
                    "human_request",
                    {
                        "run_id": run_id,
                        "request_id": req_id,
                        "role": str(
                            active_role.value
                            if hasattr(active_role, "value")
                            else active_role
                        ),
                        "context_json": ctx.model_dump(mode="json"),
                        "requested_at": _requested_at,
                    },
                )

                _answered_at = datetime.now(UTC)

                # human_response event
                await adispatch_custom_event(
                    "human_response",
                    {
                        "run_id": run_id,
                        "request_id": req_id,
                        "answered_at": _answered_at,
                        "response_json": resp.model_dump(mode="json"),
                        "source": getattr(resp, "source", "llm_simulated"),
                        "timed_out": getattr(resp, "timed_out", False),
                        "latency_s": (_answered_at - _requested_at).total_seconds(),
                    },
                )

            return inputs

        runnable: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
            _dispatch_all
        )
        await runnable.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )

        # Allow async background DB writes to complete
        await asyncio.sleep(0.3)

        # ── Fetch human_interactions ordered by requested_at ASC ─────────────
        async with session_scope(factory) as session:
            hi_result = await session.execute(
                select(HumanInteraction.role, HumanInteraction.requested_at)
                .where(HumanInteraction.run_id == run_id)
                .order_by(HumanInteraction.requested_at.asc())
            )
            hi_rows = hi_result.fetchall()

        roles = [row.role for row in hi_rows]

        # ── Finalize: replicate runner.py step-12a ────────────────────────────
        _dynamic_human_role, _dynamic_cog_proxy = await _finalize_run(factory, run_id)

        # ── Fetch updated run row ─────────────────────────────────────────────
        async with session_scope(factory) as session:
            run_result = await session.execute(select(Run).where(Run.id == run_id))
            run_row = run_result.scalar_one_or_none()

        assert run_row is not None, f"No run row found for run_id={run_id}"

        # ── Assertion 1: >=2 distinct roles in human_interactions ─────────────
        assert len(roles) >= 3, (
            f"Expected at least 3 human_interactions rows for run_id={run_id}, got {len(roles)}. "
            f"roles={roles}"
        )
        assert len(set(roles)) >= 2, (
            f"Expected >=2 distinct roles in human_interactions, got: {roles}. "
            f"RuleBasedRoleRouter should map planning->coordinator, "
            f"execution->peer, verification->reviewer."
        )

        # ── Assertion 2: runs.human_role == last interaction role ─────────────
        assert run_row.human_role == roles[-1], (
            f"runs.human_role={run_row.human_role!r} != last human_interactions.role={roles[-1]!r}. "
            f"Dynamic finalization should write the last role from human_interactions."
        )

        # ── Assertion 3: cognitive_load_proxy > 0.0 ──────────────────────────
        assert run_row.cognitive_load_proxy is not None, (
            f"runs.cognitive_load_proxy is None for run_id={run_id}. "
            f"Expected a positive float (3 HITL interactions => alpha*3 + ... > 0)."
        )
        assert run_row.cognitive_load_proxy > 0.0, (
            f"runs.cognitive_load_proxy={run_row.cognitive_load_proxy!r} is not > 0.0. "
            f"Expected positive value for 3 HITL interactions."
        )

        # ── Extra diagnostic: verify expected role sequence ───────────────────
        # DEFAULT_ROLE_TABLE: planning->coordinator, execution->peer, verification->reviewer
        expected_roles = ["coordinator", "peer", "reviewer"]
        assert roles == expected_roles, (
            f"Expected roles={expected_roles} (planning/execution/verification), "
            f"got {roles}. "
            f"DEFAULT_ROLE_TABLE: {DEFAULT_ROLE_TABLE}"
        )

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
