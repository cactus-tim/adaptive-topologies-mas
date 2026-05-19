"""Tests verifying that run_one() calls compute_quality via the aggregator (M11 wiring).

Two tests:
  1. Inline-prompt task (not registered) → compute_quality is NOT called;
     quality_score == 0.0 in RunResult.
  2. Registered task path (patched resolve_spec returns a TaskSpec) →
     compute_quality IS called; quality_score reflects its return value.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import pytest

from atm.experiment.config import (
    AgentSetCfg,
    BudgetCfg,
    EvaluationCfg,
    ExperimentConfig,
    ModelCfg,
    ObservabilityCfg,
    TaskCfg,
    TopologyCfg,
)
from atm.experiment.runner import run_one


def _make_cfg(**overrides: Any) -> ExperimentConfig:
    """Minimal ExperimentConfig with fake:echo judge model for wiring tests."""
    defaults: dict[str, Any] = {
        "name": "wiring_exp",
        "seed": 42,
        "task": TaskCfg(name="fib_test", input="Write fib(n)"),
        "model": ModelCfg(default="fake:echo"),
        "agents": AgentSetCfg(set="canonical_4"),
        "topology": TopologyCfg(name="chain", max_iterations=3),
        "budget": BudgetCfg(per_call_usd=0.1, per_run_usd=5.0, per_experiment_usd=50.0),
        "observability": ObservabilityCfg(
            pg_dsn="postgresql+asyncpg://localhost/atm_test",
            parquet_dir="/tmp/atm_wiring_test_parquet",
        ),
        "evaluation": EvaluationCfg(judge_model="fake:echo"),
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_final_state(final_answer: str = "The answer is 42") -> dict[str, Any]:
    """Build a minimal mock graph state."""
    return {
        "shared": {
            "task_id": "fib_test",
            "task_input": "Write fib(n)",
            "phase": "done",
            "iteration": 1,
            "iter_total": 1,
            "active_topology": "chain",
            "final_answer": final_answer,
            "signals": {},
            "phase_started_at_iter": 0,
            "topology_started_at_iter": 0,
            "topology_history": [],
            "topology_switch_count": 0,
            "phase_history": [],
            "human_requests": [],
            "human_responses": [],
            "broadcast_bus": [],
        },
        "agents": {},
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


def _common_patches(
    mock_ce: Any,
    mock_csf: Any,
    exp_id: UUID,
    run_id: UUID,
    mock_ee: Any,
    mock_ir: Any,
    mock_pw_cls: Any,
    mock_pw: Any,
    mock_reg: Any,
    mock_cp_scope: Any,
    mock_graph: Any,
) -> None:
    """Wire up the standard infrastructure mocks."""
    mock_ee.return_value = exp_id
    mock_ir.return_value = run_id

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    mock_ce.return_value = mock_engine
    mock_csf.return_value = MagicMock()
    mock_pw_cls.return_value = mock_pw

    mock_topo_instance = Mock()
    mock_topo_instance.build = Mock(return_value=mock_graph)
    mock_topo_cls = Mock(return_value=mock_topo_instance)
    mock_reg.get.return_value = mock_topo_cls

    async def _aenter(self: Any) -> Any:
        return AsyncMock()

    async def _aexit(self: Any, *args: Any) -> bool:
        return False

    mock_cp_scope.return_value.__aenter__ = _aenter
    mock_cp_scope.return_value.__aexit__ = _aexit


@pytest.mark.asyncio
async def test_inline_prompt_short_circuits_to_zero() -> None:
    """run_one() with an unregistered task skips compute_quality; score is 0.0.

    The "fib_test" task is not registered in TASKS, so resolve_spec returns
    None and the runner short-circuits to quality_score=0.0 without calling
    compute_quality.
    """
    cfg = _make_cfg()
    final_state = _make_final_state()

    mock_pw = AsyncMock()
    mock_pw.close = AsyncMock()
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    with (
        patch("atm.experiment.runner.create_engine") as mock_ce,
        patch("atm.experiment.runner.create_session_factory") as mock_csf,
        patch("atm.experiment.runner.Base.metadata.create_all"),
        patch("atm.experiment.runner._ensure_experiment", new_callable=AsyncMock) as mock_ee,
        patch("atm.experiment.runner._insert_run", new_callable=AsyncMock) as mock_ir,
        patch("atm.experiment.runner._update_run_success", new_callable=AsyncMock),
        patch("atm.experiment.runner.ParquetWriter") as mock_pw_cls,
        patch("atm.experiment.runner.ExperimentCallbackHandler"),
        patch("atm.experiment.runner._build_agents", return_value={}),
        patch("atm.experiment.runner.TopologyRegistry") as mock_reg,
        patch("atm.experiment.runner.checkpointer_scope") as mock_cp_scope,
        patch("atm.experiment.runner._load_pricing", return_value=MagicMock()),
        patch("atm.experiment.runner.compute_quality", new_callable=AsyncMock) as mock_cq,
    ):
        _common_patches(
            mock_ce,
            mock_csf,
            exp_id,
            run_id,
            mock_ee,
            mock_ir,
            mock_pw_cls,
            mock_pw,
            mock_reg,
            mock_cp_scope,
            mock_graph,
        )
        result = await run_one(cfg)

    mock_cq.assert_not_called()
    assert result.metrics["quality_score"] == 0.0
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_registered_task_calls_compute_quality() -> None:
    """run_one() with a registered task spec calls compute_quality.

    We patch resolve_spec to return a fake TaskSpec and compute_quality to
    return (0.75, {}). The result must have quality_score == 0.75.
    """
    from atm.tasks.base import TaskSpec

    cfg = _make_cfg()
    final_state = _make_final_state(final_answer="some answer")

    fake_spec = TaskSpec(
        id="gsm8k:gsm8k_test_001",
        type="reasoning",
        input="What is 2+2?",
        expected="4",
        evaluator_key="gsm8k_numeric",
    )

    mock_pw = AsyncMock()
    mock_pw.close = AsyncMock()
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    with (
        patch("atm.experiment.runner.create_engine") as mock_ce,
        patch("atm.experiment.runner.create_session_factory") as mock_csf,
        patch("atm.experiment.runner.Base.metadata.create_all"),
        patch("atm.experiment.runner._ensure_experiment", new_callable=AsyncMock) as mock_ee,
        patch("atm.experiment.runner._insert_run", new_callable=AsyncMock) as mock_ir,
        patch("atm.experiment.runner._update_run_success", new_callable=AsyncMock),
        patch("atm.experiment.runner.ParquetWriter") as mock_pw_cls,
        patch("atm.experiment.runner.ExperimentCallbackHandler"),
        patch("atm.experiment.runner._build_agents", return_value={}),
        patch("atm.experiment.runner.TopologyRegistry") as mock_reg,
        patch("atm.experiment.runner.checkpointer_scope") as mock_cp_scope,
        patch("atm.experiment.runner._load_pricing", return_value=MagicMock()),
        patch("atm.experiment.runner.resolve_spec", return_value=fake_spec) as mock_rs,
        patch(
            "atm.experiment.runner.compute_quality",
            new_callable=AsyncMock,
            return_value=(0.75, {"details": "ok"}),
        ) as mock_cq,
    ):
        _common_patches(
            mock_ce,
            mock_csf,
            exp_id,
            run_id,
            mock_ee,
            mock_ir,
            mock_pw_cls,
            mock_pw,
            mock_reg,
            mock_cp_scope,
            mock_graph,
        )
        result = await run_one(cfg)

    mock_rs.assert_called_once_with(cfg.task)
    mock_cq.assert_called_once()
    call_args = mock_cq.call_args
    assert call_args[0][0] is fake_spec
    assert call_args[0][1] == "some answer"
    assert call_args[1]["run_seed"] == 42
    assert result.metrics["quality_score"] == 0.75
    assert result.status == "completed"
