"""Postgres-backed integration test for the oracle labels pipeline.

Verifies the end-to-end producer/consumer contract:
  build_leave_one_out_oracle() (producer) →
  OracleTable.to_json_dict() (serializer) →
  OracleTopologyRouter (consumer)

Skip behaviour: transitively inherited from the ``session_factory_fast``
fixture chain (``session_factory_fast`` → ``pg_engine_fast`` → ``pg_dsn``).
``pg_dsn`` calls ``pytest.skip()`` when ``ATM_ENABLE_PG_TESTS=1`` is not set.
No manual ``pytest.skipif`` is needed or present.

Prerequisite for running: a live PostgreSQL instance reachable at $ATM_PG_DSN
(default: postgresql+asyncpg://atm:atm@localhost:5432/atm_test).
Enable with: ATM_ENABLE_PG_TESTS=1 uv run pytest tests/integration/analysis/ -v
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from atm.analysis.oracle import build_leave_one_out_oracle
from atm.core.state import SharedState
from atm.core.types import Phase
from atm.phases.topology_router import OracleTopologyRouter
from atm.storage.models import Experiment, Run
from atm.storage.session import session_scope

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


async def _seed_experiment(session_factory, *, name: str) -> uuid.UUID:
    """Insert one Experiment row and return its id."""
    exp_id = _new_uuid()
    async with session_scope(session_factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=name,
                config_snapshot={},
                status="running",
            )
        )
    return exp_id


async def _seed_run(
    session_factory,
    *,
    exp_id: uuid.UUID,
    task_id: str,
    topology: str,
    quality_score: float,
    budget_spent_usd: Decimal = Decimal("0"),
) -> uuid.UUID:
    """Insert one Run row and return its id."""
    run_id = _new_uuid()
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology=topology,
                task_id=task_id,
                agent_set="default",
                seed=42,
                model="gpt-4",
                models_by_role_json={},
                model_version_snapshot={},
                status="finished",
                quality_score=quality_score,
                budget_spent_usd=budget_spent_usd,
            )
        )
    return run_id


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
async def test_oracle_producer_consumer_contract(
    session_factory_fast,
) -> None:
    """End-to-end oracle pipeline: seed DB → build oracle → route via OracleTopologyRouter.

    Data layout
    -----------
    Task IDs (task_type inferred from prefix):
      "HumanEval/0"  → programming
      "HumanEval/1"  → programming
      "HumanEval/2"  → programming
      "GSM8K/0"      → reasoning

    Programming tasks:
      HumanEval/0: mesh(0.9), linear(0.4), supervisor(0.5)
      HumanEval/1: mesh(0.8), linear(0.3)
      HumanEval/2: supervisor(0.85), linear(0.2)

    LOO for HumanEval/0 (excluded): remaining programming rows from HumanEval/1 + HumanEval/2
      mesh:       [0.8]       → mean 0.800
      linear:     [0.3, 0.2]  → mean 0.250
      supervisor: [0.85]      → mean 0.850
      LOO winner for HumanEval/0 = "supervisor"

    Global programming type winner (all rows):
      mesh:       [0.9, 0.8]   → mean 0.850
      linear:     [0.4, 0.3, 0.2] → mean 0.300
      supervisor: [0.5, 0.85]  → mean 0.675
      type winner = "mesh"

    Reasoning tasks:
      GSM8K/0: debate(0.95), linear(0.1)
      LOO for GSM8K/0: no other reasoning rows → fallback to "linear" (default)
      type winner for reasoning = "debate"

    OracleTopologyRouter lookup order:
      1. by_task_id[task_id][phase]   (most specific)
      2. by_task_type[task_type][phase]  (requires state["task_type"] to be set explicitly)
      3. _default  (fallback)
    """
    exp_id = await _seed_experiment(session_factory_fast, name=f"oracle-integration-{_new_uuid()}")

    # --- programming rows ---
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="HumanEval/0",
        topology="mesh",
        quality_score=0.9,
        budget_spent_usd=Decimal("1.00"),
    )
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="HumanEval/0",
        topology="linear",
        quality_score=0.4,
        budget_spent_usd=Decimal("0.50"),
    )
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="HumanEval/0",
        topology="supervisor",
        quality_score=0.5,
        budget_spent_usd=Decimal("0.70"),
    )
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="HumanEval/1",
        topology="mesh",
        quality_score=0.8,
        budget_spent_usd=Decimal("0.90"),
    )
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="HumanEval/1",
        topology="linear",
        quality_score=0.3,
        budget_spent_usd=Decimal("0.40"),
    )
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="HumanEval/2",
        topology="supervisor",
        quality_score=0.85,
        budget_spent_usd=Decimal("0.80"),
    )
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="HumanEval/2",
        topology="linear",
        quality_score=0.2,
        budget_spent_usd=Decimal("0.30"),
    )

    # --- reasoning rows ---
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="GSM8K/0",
        topology="debate",
        quality_score=0.95,
        budget_spent_usd=Decimal("1.20"),
    )
    await _seed_run(
        session_factory_fast,
        exp_id=exp_id,
        task_id="GSM8K/0",
        topology="linear",
        quality_score=0.1,
        budget_spent_usd=Decimal("0.20"),
    )

    # --- Build OracleTable via LOO over real Postgres rows ---
    oracle = await build_leave_one_out_oracle(
        exp_id,
        session_factory=session_factory_fast,
    )

    # Verify the oracle contains expected entries
    assert "programming" in oracle.by_task_type, (
        f"Expected 'programming' in by_task_type, got keys: {list(oracle.by_task_type)}"
    )
    assert "reasoning" in oracle.by_task_type, (
        f"Expected 'reasoning' in by_task_type, got keys: {list(oracle.by_task_type)}"
    )

    # Global winner for programming = mesh (mean 0.85 vs supervisor 0.675 vs linear 0.30)
    assert oracle.by_task_type["programming"] == "mesh", (
        f"Expected programming type winner = 'mesh', got {oracle.by_task_type['programming']!r}"
    )

    # Global winner for reasoning = debate (only debate rows have high scores)
    assert oracle.by_task_type["reasoning"] == "debate", (
        f"Expected reasoning type winner = 'debate', got {oracle.by_task_type['reasoning']!r}"
    )

    # LOO winner for HumanEval/0 = supervisor (best in remainder = HumanEval/1 + HumanEval/2)
    assert oracle.by_task_id.get("HumanEval/0") == "supervisor", (
        f"LOO winner for HumanEval/0: expected 'supervisor', "
        f"got {oracle.by_task_id.get('HumanEval/0')!r}"
    )

    # LOO winner for GSM8K/0 = linear (no other reasoning rows → fallback to default)
    assert oracle.by_task_id.get("GSM8K/0") == "linear", (
        f"LOO fallback for GSM8K/0: expected 'linear', got {oracle.by_task_id.get('GSM8K/0')!r}"
    )

    # --- Serialize to router dict and construct OracleTopologyRouter ---
    router_dict = oracle.to_json_dict()

    # Verify the router dict has the required keys
    assert "by_task_type" in router_dict
    assert "by_task_id" in router_dict
    assert "_default" in router_dict

    # Verify nested phase structure: each value must be a dict mapping phase → topology
    assert isinstance(router_dict["by_task_type"].get("programming"), dict), (
        "by_task_type['programming'] must be a phase-keyed dict"
    )
    assert "planning" in router_dict["by_task_type"]["programming"], (
        "by_task_type['programming'] must have 'planning' key"
    )

    router = OracleTopologyRouter(router_dict)

    # --- Router test 1: by_task_id lookup for HumanEval/0 ---
    # Expected: supervisor (LOO winner for HumanEval/0)
    state_he0: SharedState = {
        "task_id": "HumanEval/0",
        "phase": Phase.PLANNING,
    }
    decision_he0 = await router.decide(state_he0)
    assert decision_he0.decided_by == "oracle", (
        f"Expected decided_by='oracle', got {decision_he0.decided_by!r}"
    )
    assert decision_he0.topology == "supervisor", (
        f"OracleTopologyRouter for HumanEval/0: expected 'supervisor', "
        f"got {decision_he0.topology!r}"
    )

    # --- Router test 2: by_task_id lookup for HumanEval/1 (LOO → mesh winner) ---
    # LOO for HumanEval/1: remaining programming rows = HumanEval/0 + HumanEval/2
    #   mesh:       [0.9]       → mean 0.900
    #   linear:     [0.4, 0.2]  → mean 0.300
    #   supervisor: [0.5, 0.85] → mean 0.675
    # Winner = mesh
    state_he1: SharedState = {
        "task_id": "HumanEval/1",
        "phase": Phase.EXECUTION,
    }
    decision_he1 = await router.decide(state_he1)
    assert decision_he1.decided_by == "oracle"
    assert decision_he1.topology == "mesh", (
        f"OracleTopologyRouter for HumanEval/1: expected 'mesh', got {decision_he1.topology!r}"
    )

    # --- Router test 3: by_task_type fallback for unknown task_id ---
    # Must explicitly set state["task_type"] because OracleTopologyRouter.decide()
    # line ~392 is a `pass` stub — it does NOT auto-infer task_type from task_id.
    state_unknown: SharedState = {
        "task_id": "HumanEval/999",  # not in by_task_id
        "phase": Phase.PLANNING,
        **{"task_type": "programming"},  # type: ignore[typeddict-item]  # explicitly set
    }
    decision_unknown = await router.decide(state_unknown)
    assert decision_unknown.decided_by == "oracle"
    # Falls through to by_task_type["programming"]["planning"] = mesh (global winner)
    assert decision_unknown.topology == "mesh", (
        f"by_task_type fallback for programming: expected 'mesh', got {decision_unknown.topology!r}"
    )

    # --- Router test 4: by_task_type fallback for reasoning type ---
    state_reasoning: SharedState = {
        "task_id": "GSM8K/999",  # not in by_task_id
        "phase": Phase.VERIFICATION,
        **{"task_type": "reasoning"},  # type: ignore[typeddict-item]  # explicitly set
    }
    decision_reasoning = await router.decide(state_reasoning)
    assert decision_reasoning.decided_by == "oracle"
    # by_task_type["reasoning"]["verification"] = debate (global winner for reasoning)
    assert decision_reasoning.topology == "debate", (
        f"by_task_type fallback for reasoning: expected 'debate', "
        f"got {decision_reasoning.topology!r}"
    )

    # --- Router test 5: _default fallback when neither task_id nor task_type is known ---
    state_default: SharedState = {
        "task_id": "unknown/task-xyz",
        "phase": Phase.PLANNING,
        # task_type is NOT set — falls through to _default
    }
    decision_default = await router.decide(state_default)
    assert decision_default.decided_by == "oracle"
    # _default in the oracle is "linear"
    assert decision_default.topology == "linear", (
        f"_default fallback: expected 'linear', got {decision_default.topology!r}"
    )
    assert "default" in decision_default.reason.lower(), (
        f"Reason should mention 'default': {decision_default.reason!r}"
    )

    # --- Router test 6: router_cost_usd is 0.0 (no LLM calls) ---
    assert decision_he0.router_cost_usd == 0.0, (
        f"Oracle router should have zero cost, got {decision_he0.router_cost_usd}"
    )
