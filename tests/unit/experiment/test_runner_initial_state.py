"""Unit tests for M9 Step 3.1 runner edits.

Three tests (acceptance criteria from plan step 3.1):
  1. _build_initial_state puts run_id into state["shared"]["run_id"]
  2. cfg.human=None (default) — topology.build receives human_cfg=None, no errors
  3. cfg.human=HumanCfg(enabled=True, ...) — topology.build receives the kwarg

Four tests added for fix-dabench-task Step 3.1:
  4. _build_initial_state augments task_input with DABench metadata
  5. _build_initial_state leaves task_input unchanged when metadata is empty
  6. _build_initial_state leaves task_input unchanged for irrelevant (non-DABench) metadata
  7. _pre_stage_workspace delegates to stage_workspace_for with the right arguments
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from atm.core.types import TaskSpec
from atm.experiment.config import (
    AgentSetCfg,
    BudgetCfg,
    ExperimentConfig,
    HumanCfg,
    ModelCfg,
    ObservabilityCfg,
    TaskCfg,
    TopologyCfg,
)
from atm.experiment.runner import _build_initial_state, _pre_stage_workspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(**overrides: Any) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing."""
    defaults: dict[str, Any] = {
        "name": "test_exp",
        "seed": 42,
        "task": TaskCfg(name="fib_test", input="Write fib(n)"),
        "model": ModelCfg(default="fake:echo"),
        "agents": AgentSetCfg(set="canonical_4"),
        "topology": TopologyCfg(name="chain", max_iterations=3),
        "budget": BudgetCfg(per_call_usd=0.1, per_run_usd=5.0, per_experiment_usd=50.0),
        "observability": ObservabilityCfg(
            pg_dsn="postgresql+asyncpg://localhost/atm_test",
            parquet_dir="/tmp/atm_test_parquet",
        ),
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_final_state(final_answer: str = "fib(10) = 55") -> dict[str, Any]:
    return {
        "shared": {
            "task_id": "fib_test",
            "task_input": "Write fib(n)",
            "phase": "done",
            "iteration": 1,
            "iter_total": 1,
            "active_topology": "chain",
            "final_answer": final_answer,
            "signals": {"critic_approved": True},
            "phase_started_at_iter": 0,
            "topology_started_at_iter": 0,
            "topology_history": [],
            "topology_switch_count": 0,
            "phase_history": ["execution"],
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


def _patch_run_one(
    mock_ce: Any,
    mock_csf: Any,
    exp_id: Any,
    run_id: Any,
    mock_ee: Any,
    mock_ir: Any,
    mock_pw: Any,
    mock_pw_cls: Any,
    mock_reg: Any,
    mock_cp_scope: Any,
    topo_instance: Any,
) -> None:
    """Wire up common mocks for run_one tests."""
    mock_ee.return_value = exp_id
    mock_ir.return_value = run_id

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    mock_ce.return_value = mock_engine
    mock_csf.return_value = MagicMock()
    mock_pw_cls.return_value = mock_pw

    # Registry returns a class whose instance is topo_instance
    mock_topo_cls = Mock(return_value=topo_instance)
    mock_reg.get.return_value = mock_topo_cls

    async def _cp_aenter(self: Any) -> Any:
        return AsyncMock()

    async def _cp_aexit(self: Any, *args: Any) -> bool:
        return False

    mock_cp_scope.return_value.__aenter__ = _cp_aenter
    mock_cp_scope.return_value.__aexit__ = _cp_aexit


# ---------------------------------------------------------------------------
# Test 1 — run_id present in initial_state["shared"]
# ---------------------------------------------------------------------------


def test_build_initial_state_contains_run_id() -> None:
    """_build_initial_state must put run_id into state['shared']['run_id']."""
    cfg = _make_cfg()
    run_id = uuid.uuid4()

    state = _build_initial_state(cfg, run_id)

    shared = state["shared"]
    assert "run_id" in shared, (
        f"Expected 'run_id' in state['shared'], got keys: {list(shared.keys())}"
    )
    assert shared["run_id"] == run_id, f"Expected run_id={run_id}, got {shared['run_id']}"


# ---------------------------------------------------------------------------
# Test 2 — cfg.human=None (default) propagates None to topology.build
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_human_cfg_none_passes_none_to_build() -> None:
    """When cfg.human is None, topology.build receives human_cfg=None without errors."""
    cfg = _make_cfg()  # human is None by default
    assert cfg.human is None

    final_state = _make_final_state(final_answer="fib(10) = 55")
    received_kwargs: dict[str, Any] = {}

    def _fake_build(agents: Any, topo_cfg: Any, **kwargs: Any) -> Any:
        received_kwargs.update(kwargs)
        mock = AsyncMock()
        mock.ainvoke = AsyncMock(return_value=final_state)
        return mock

    topo_instance = Mock()
    topo_instance.build = _fake_build

    mock_pw = AsyncMock()
    mock_pw.close = AsyncMock()

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
    ):
        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        _patch_run_one(
            mock_ce,
            mock_csf,
            exp_id,
            run_id,
            mock_ee,
            mock_ir,
            mock_pw,
            mock_pw_cls,
            mock_reg,
            mock_cp_scope,
            topo_instance,
        )
        from atm.experiment.runner import run_one

        result = await run_one(cfg)

    assert result.status == "completed"
    assert "human_cfg" in received_kwargs, (
        f"topology.build must receive human_cfg kwarg; got kwargs: {received_kwargs}"
    )
    assert received_kwargs["human_cfg"] is None, (
        f"human_cfg must be None when cfg.human is None; got: {received_kwargs['human_cfg']}"
    )


# ---------------------------------------------------------------------------
# Test 3 — cfg.human=HumanCfg(enabled=True) propagates to topology.build
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_human_cfg_enabled_propagates_to_build() -> None:
    """When cfg.human=HumanCfg(enabled=True), topology.build receives the object."""
    human_cfg = HumanCfg(enabled=True, gateway="llm_simulated")
    cfg = _make_cfg(human=human_cfg)
    assert cfg.human is not None
    assert cfg.human.enabled is True

    final_state = _make_final_state(final_answer="fib(10) = 55")
    received_kwargs: dict[str, Any] = {}

    def _fake_build(agents: Any, topo_cfg: Any, **kwargs: Any) -> Any:
        received_kwargs.update(kwargs)
        mock = AsyncMock()
        mock.ainvoke = AsyncMock(return_value=final_state)
        return mock

    topo_instance = Mock()
    topo_instance.build = _fake_build

    mock_pw = AsyncMock()
    mock_pw.close = AsyncMock()

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
    ):
        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        _patch_run_one(
            mock_ce,
            mock_csf,
            exp_id,
            run_id,
            mock_ee,
            mock_ir,
            mock_pw,
            mock_pw_cls,
            mock_reg,
            mock_cp_scope,
            topo_instance,
        )
        from atm.experiment.runner import run_one

        result = await run_one(cfg)

    assert result.status == "completed"
    assert "human_cfg" in received_kwargs, (
        f"topology.build must receive human_cfg kwarg; got kwargs: {received_kwargs}"
    )
    received_human = received_kwargs["human_cfg"]
    assert received_human is human_cfg, (
        f"topology.build must receive the exact HumanCfg instance; got: {received_human}"
    )
    assert received_human.enabled is True
    assert received_human.gateway == "llm_simulated"


# ---------------------------------------------------------------------------
# Test 4 — DABench metadata is appended to task_input (fix-dabench-task 3.1)
# ---------------------------------------------------------------------------


def test_build_initial_state_augments_dabench_metadata() -> None:
    """_build_initial_state must append [Task metadata] block for DABench specs."""
    # Arrange
    dabench_spec = TaskSpec(
        id="dabench/example",
        type="reasoning",
        input="What is the mean fare?",
        expected=None,
        metadata={
            "format": "@mean[1.0]",
            "constraints": "Round to 2 decimals",
            "file_name": "titanic.csv",
        },
        evaluator_key="dabench",
    )
    cfg = _make_cfg(task=TaskCfg(name="dabench", input=""))

    with patch("atm.experiment.runner.resolve_spec", return_value=dabench_spec):
        # Act
        state = _build_initial_state(cfg, run_id=uuid.uuid4())

    # Assert
    task_input: str = state["shared"]["task_input"]
    assert "[Task metadata]" in task_input, (
        f"Expected '[Task metadata]' block in task_input; got: {task_input!r}"
    )
    assert "Dataset file: titanic.csv" in task_input, (
        f"Expected 'Dataset file: titanic.csv' in task_input; got: {task_input!r}"
    )
    assert "Constraints: Round to 2 decimals" in task_input, (
        f"Expected 'Constraints: Round to 2 decimals' in task_input; got: {task_input!r}"
    )
    assert "Answer format" in task_input, (
        f"Expected 'Answer format' line in task_input; got: {task_input!r}"
    )
    assert "@mean[1.0]" in task_input, f"Expected '@mean[1.0]' in task_input; got: {task_input!r}"


# ---------------------------------------------------------------------------
# Test 5 — Empty metadata leaves task_input unchanged (fix-dabench-task 3.1)
# ---------------------------------------------------------------------------


def test_build_initial_state_no_metadata_unchanged() -> None:
    """_build_initial_state must NOT augment task_input when metadata is empty."""
    # Arrange
    plain_spec = TaskSpec(
        id="dabench/example",
        type="reasoning",
        input="Solve fibonacci",
        expected=None,
        metadata={},
        evaluator_key="dabench",
    )
    cfg = _make_cfg(task=TaskCfg(name="dabench", input=""))

    with patch("atm.experiment.runner.resolve_spec", return_value=plain_spec):
        # Act
        state = _build_initial_state(cfg, run_id=uuid.uuid4())

    # Assert
    task_input: str = state["shared"]["task_input"]
    assert task_input == "Solve fibonacci", (
        f"Expected task_input to equal 'Solve fibonacci' exactly; got: {task_input!r}"
    )
    assert "[Task metadata]" not in task_input, (
        f"Expected no '[Task metadata]' block when metadata is empty; got: {task_input!r}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Irrelevant (non-DABench) metadata leaves task_input unchanged
#           (fix-dabench-task 3.1)
# ---------------------------------------------------------------------------


def test_build_initial_state_irrelevant_metadata_unchanged() -> None:
    """_build_initial_state must NOT augment task_input for non-DABench metadata."""
    # Arrange — HumanEval-style spec: has metadata but no 'format' or 'file_name'
    humaneval_spec = TaskSpec(
        id="humaneval/HumanEval/0",
        type="programming",
        input="def fibonacci(n):",
        expected=None,
        metadata={"test": "assert fibonacci(10) == 55", "entry_point": "fibonacci"},
        evaluator_key="humaneval",
    )
    cfg = _make_cfg(task=TaskCfg(name="humaneval", input=""))

    with patch("atm.experiment.runner.resolve_spec", return_value=humaneval_spec):
        # Act
        state = _build_initial_state(cfg, run_id=uuid.uuid4())

    # Assert
    task_input: str = state["shared"]["task_input"]
    assert "[Task metadata]" not in task_input, (
        f"Expected no '[Task metadata]' block for HumanEval spec; got: {task_input!r}"
    )
    assert task_input == "def fibonacci(n):", (
        f"Expected task_input identical to spec.input for HumanEval; got: {task_input!r}"
    )


# ---------------------------------------------------------------------------
# Test 7 — _pre_stage_workspace calls stage_workspace_for (fix-dabench-task 3.1)
# ---------------------------------------------------------------------------


def test_pre_stage_workspace_invoked_on_resume_path(tmp_path: Path) -> None:
    """_pre_stage_workspace must delegate to stage_workspace_for with correct args."""
    # Arrange
    dabench_spec = TaskSpec(
        id="dabench/titanic/0",
        type="reasoning",
        input="What is the mean fare?",
        expected=None,
        metadata={
            "format": "@mean_fare[34.65]",
            "constraints": "Round to 2 decimals",
            "file_name": "titanic.csv",
        },
        evaluator_key="dabench",
    )
    cfg = _make_cfg(task=TaskCfg(name="dabench", input=""))
    run_id = uuid.uuid4()
    staged_file = Path("/tmp/titanic.csv")

    mock_stage = Mock(return_value=[staged_file])

    with (
        patch("atm.experiment.runner.resolve_spec", return_value=dabench_spec),
        patch("atm.experiment.runner.stage_workspace_for", mock_stage),
    ):
        # Act
        result = _pre_stage_workspace(cfg, run_id, tmp_path)

    # Assert — the mock was called exactly once with the resolved spec and workspace path
    mock_stage.assert_called_once_with(dabench_spec, tmp_path)
    assert result == [staged_file], (
        f"Expected _pre_stage_workspace to return the staged file list; got: {result!r}"
    )
