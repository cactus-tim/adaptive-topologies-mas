"""Unit tests for experiment runner — run_one() lifecycle.

Tests use heavily mocked dependencies to avoid any database or filesystem I/O.
All 6+ required tests are implemented:
  a. success path: status=completed, quality_score=0.0 (inline-prompt fib_test path)
  b. BudgetExceededError path: status=budget_exceeded, parquet closed before update
  c. generic Exception path: status=failed, still flushes parquet
  d. flush-before-update order (mock call order assertions)
  e. initial_state has all 14 shared keys
  f. git_sha fallback when subprocess raises

M11 note: The M6 "55-substring" evaluate() stub has been replaced by the
aggregator (compute_quality). Inline-prompt tasks (task.name not in TASKS
registry) short-circuit to quality_score=0.0 without calling the aggregator.
The "fib_test" fixture task is not registered, so quality_score == 0.0.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import pytest

from atm.core.errors import BudgetExceededError
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
from atm.experiment.runner import (
    RunResult,
    _build_initial_state,
    _get_git_sha,
    run_one,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(**overrides: Any) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing.

    M11: evaluation.judge_model is set to "fake:echo" so that the judge
    LLMWrapper construction in run_one() does not try to reach a real LLM API.
    The "fib_test" task is an unregistered inline-prompt task, so resolve_spec
    returns None and quality_score short-circuits to 0.0.
    """
    defaults: dict[str, Any] = {
        "name": "test_exp",
        "seed": 42,
        "task": TaskCfg(
            name="fib_test",
            input="Write fib(n)",
        ),
        "model": ModelCfg(default="fake:echo"),
        "agents": AgentSetCfg(set="canonical_4"),
        "topology": TopologyCfg(name="chain", max_iterations=3),
        "budget": BudgetCfg(per_call_usd=0.1, per_run_usd=5.0, per_experiment_usd=50.0),
        "observability": ObservabilityCfg(
            pg_dsn="postgresql+asyncpg://localhost/atm_test",
            parquet_dir="/tmp/atm_test_parquet",
        ),
        "evaluation": EvaluationCfg(judge_model="fake:echo"),
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_final_state(final_answer: str = "fib(10) = 55") -> dict[str, Any]:
    """Build a mock final graph state."""
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


# ---------------------------------------------------------------------------
# Test: _build_initial_state has all 14 shared keys
# ---------------------------------------------------------------------------


def test_initial_state_has_all_14_shared_keys() -> None:
    """_build_initial_state must populate all 14 SharedState keys."""
    cfg = _make_cfg()
    run_id = uuid.uuid4()
    state = _build_initial_state(cfg, run_id)

    shared = state["shared"]

    required_keys = {
        "task_id",
        "task_input",
        "phase",
        "iteration",
        "iter_total",
        "active_topology",
        "final_answer",
        "signals",
        "phase_started_at_iter",
        "topology_started_at_iter",
        "topology_history",
        "topology_switch_count",
        "phase_history",
        "human_requests",
        "human_responses",
        "broadcast_bus",
    }

    missing = required_keys - set(shared.keys())
    assert not missing, f"Missing keys in initial_state['shared']: {missing}"


def test_initial_state_correct_values() -> None:
    """_build_initial_state sets sensible defaults for shared keys."""
    cfg = _make_cfg()
    run_id = uuid.uuid4()
    state = _build_initial_state(cfg, run_id)
    shared = state["shared"]

    assert shared["task_id"] == "fib_test"
    assert shared["task_input"] == "Write fib(n)"
    assert shared["iteration"] == 0
    assert shared["iter_total"] == 0
    assert shared["active_topology"] == "chain"
    assert shared["final_answer"] == ""
    assert shared["signals"] == {}
    assert shared["phase_started_at_iter"] == 0
    assert shared["topology_switch_count"] == 0
    assert shared["topology_history"] == []
    assert shared["phase_history"] == []
    assert shared["human_requests"] == []
    assert shared["human_responses"] == []
    assert shared["broadcast_bus"] == []


# ---------------------------------------------------------------------------
# Test: git_sha fallback
# ---------------------------------------------------------------------------


def test_git_sha_fallback_when_subprocess_raises() -> None:
    """_get_git_sha returns None when subprocess raises any exception."""
    with patch("atm.experiment.runner.subprocess.run", side_effect=OSError("no git")):
        result = _get_git_sha()
    assert result is None


def test_git_sha_fallback_when_subprocess_nonzero() -> None:
    """_get_git_sha returns None when subprocess returns non-zero exit code."""
    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    with patch("atm.experiment.runner.subprocess.run", return_value=mock_result):
        result = _get_git_sha()
    assert result is None


def test_git_sha_returns_value_on_success() -> None:
    """_get_git_sha returns the stripped sha on success."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "abc1234\n"
    with patch("atm.experiment.runner.subprocess.run", return_value=mock_result):
        result = _get_git_sha()
    assert result == "abc1234"


# ---------------------------------------------------------------------------
# Mock setup helpers
# ---------------------------------------------------------------------------


def _make_mock_engine_and_session() -> tuple[MagicMock, AsyncMock, AsyncMock]:
    """Create mock engine, session_factory, and session."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.add = Mock()

    mock_factory = AsyncMock()
    mock_factory.return_value = mock_session
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()

    return mock_engine, mock_factory, mock_session


def _make_mock_graph(final_state: dict[str, Any]) -> AsyncMock:
    """Create a mock CompiledStateGraph."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=final_state)
    return mock_graph


def _make_mock_topology(mock_graph: AsyncMock) -> Mock:
    """Create a mock topology instance."""
    mock_topo = Mock()
    mock_topo.build = Mock(return_value=mock_graph)
    return mock_topo


def _make_mock_parquet_writer() -> AsyncMock:
    """Create a mock ParquetWriter."""
    mock_pw = AsyncMock()
    mock_pw.close = AsyncMock()
    mock_pw.flush = AsyncMock()
    return mock_pw


# ---------------------------------------------------------------------------
# Test: success path
# ---------------------------------------------------------------------------


def _patch_run_one_common(
    mock_ce: Any,
    mock_csf: Any,
    exp_id: UUID,
    run_id: UUID,
    mock_ee: Any,
    mock_ir: Any,
    mock_pw: Any,
    mock_pw_cls: Any,
    mock_reg: Any,
    mock_cp_scope: Any,
    mock_graph: Any,
) -> None:
    """Set up common mocks for run_one tests."""
    mock_ee.return_value = exp_id
    mock_ir.return_value = run_id

    # Engine — use a MagicMock with async begin() context manager
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    mock_ce.return_value = mock_engine

    # Session factory
    mock_csf.return_value = MagicMock()

    # Parquet writer
    mock_pw_cls.return_value = mock_pw

    # Topology registry
    # BUG-4 fix: runner now does cls = registry.get(name); instance = cls(); instance.build(...)
    # So get() must return a callable (class mock) whose return_value is the instance.
    mock_topo_instance = Mock()
    mock_topo_instance.build = Mock(return_value=mock_graph)
    mock_topo_cls = Mock(return_value=mock_topo_instance)
    mock_reg.get.return_value = mock_topo_cls

    # Checkpointer scope context manager
    mock_cp = AsyncMock()

    async def _cp_aenter(self: Any) -> Any:
        return mock_cp

    async def _cp_aexit(self: Any, *args: Any) -> bool:
        return False

    mock_cp_scope.return_value.__aenter__ = _cp_aenter
    mock_cp_scope.return_value.__aexit__ = _cp_aexit


@pytest.mark.asyncio
async def test_run_one_success_path() -> None:
    """Success path: status=completed, quality_score=1.0 when answer contains '55'."""
    cfg = _make_cfg()
    final_state = _make_final_state(final_answer="The answer is 55")

    with (
        patch("atm.experiment.runner.create_engine") as mock_ce,
        patch("atm.experiment.runner.create_session_factory") as mock_csf,
        patch("atm.experiment.runner.Base.metadata.create_all"),
        patch("atm.experiment.runner._ensure_experiment", new_callable=AsyncMock) as mock_ee,
        patch("atm.experiment.runner._insert_run", new_callable=AsyncMock) as mock_ir,
        patch("atm.experiment.runner._update_run_success", new_callable=AsyncMock) as mock_urs,
        patch("atm.experiment.runner.ParquetWriter") as mock_pw_cls,
        patch("atm.experiment.runner.ExperimentCallbackHandler"),
        patch("atm.experiment.runner._build_agents", return_value={}),
        patch("atm.experiment.runner.TopologyRegistry") as mock_reg,
        patch("atm.experiment.runner.checkpointer_scope") as mock_cp_scope,
        patch("atm.experiment.runner._load_pricing", return_value=MagicMock()),
    ):
        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        mock_pw = _make_mock_parquet_writer()
        mock_graph = _make_mock_graph(final_state)
        _patch_run_one_common(
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
            mock_graph,
        )
        result = await run_one(cfg)

    assert result.status == "completed"
    assert result.run_id == run_id
    assert result.exp_id == exp_id
    # M11: "fib_test" is not a registered task → inline-prompt path → quality_score=0.0
    assert result.metrics["quality_score"] == 0.0
    assert result.final_answer == "The answer is 55"
    mock_urs.assert_called_once()


# ---------------------------------------------------------------------------
# Test: BudgetExceededError path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_budget_exceeded_path() -> None:
    """BudgetExceededError: status=budget_exceeded, parquet closed before update."""
    cfg = _make_cfg()

    with (
        patch("atm.experiment.runner.create_engine") as mock_ce,
        patch("atm.experiment.runner.create_session_factory") as mock_csf,
        patch("atm.experiment.runner.Base.metadata.create_all"),
        patch("atm.experiment.runner._ensure_experiment", new_callable=AsyncMock) as mock_ee,
        patch("atm.experiment.runner._insert_run", new_callable=AsyncMock) as mock_ir,
        patch("atm.experiment.runner._update_run_failed", new_callable=AsyncMock) as mock_urf,
        patch("atm.experiment.runner.ParquetWriter") as mock_pw_cls,
        patch("atm.experiment.runner.ExperimentCallbackHandler"),
        patch("atm.experiment.runner._build_agents", return_value={}),
        patch("atm.experiment.runner.TopologyRegistry") as mock_reg,
        patch("atm.experiment.runner.checkpointer_scope") as mock_cp_scope,
        patch("atm.experiment.runner._load_pricing", return_value=MagicMock()),
    ):
        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        mock_pw = _make_mock_parquet_writer()

        # Graph raises BudgetExceededError
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            side_effect=BudgetExceededError(level="run", limit_usd=5.0, spent_usd=5.01)
        )
        _patch_run_one_common(
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
            mock_graph,
        )
        result = await run_one(cfg)

    assert result.status == "budget_exceeded"
    assert result.run_id == run_id
    # parquet.close() should have been called
    mock_pw.close.assert_called_once()
    # _update_run_failed should have been called with budget_exceeded status
    mock_urf.assert_called_once()
    call_kwargs = mock_urf.call_args
    assert call_kwargs[1]["status"] == "budget_exceeded"


# ---------------------------------------------------------------------------
# Test: generic Exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_generic_exception_path() -> None:
    """Generic Exception: status=failed, parquet still flushed, exception re-raised."""
    cfg = _make_cfg()

    with (
        patch("atm.experiment.runner.create_engine") as mock_ce,
        patch("atm.experiment.runner.create_session_factory") as mock_csf,
        patch("atm.experiment.runner.Base.metadata.create_all"),
        patch("atm.experiment.runner._ensure_experiment", new_callable=AsyncMock) as mock_ee,
        patch("atm.experiment.runner._insert_run", new_callable=AsyncMock) as mock_ir,
        patch("atm.experiment.runner._update_run_failed", new_callable=AsyncMock) as mock_urf,
        patch("atm.experiment.runner.ParquetWriter") as mock_pw_cls,
        patch("atm.experiment.runner.ExperimentCallbackHandler"),
        patch("atm.experiment.runner._build_agents", return_value={}),
        patch("atm.experiment.runner.TopologyRegistry") as mock_reg,
        patch("atm.experiment.runner.checkpointer_scope") as mock_cp_scope,
        patch("atm.experiment.runner._load_pricing", return_value=MagicMock()),
    ):
        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        mock_pw = _make_mock_parquet_writer()

        # Graph raises a generic exception
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph crashed"))
        _patch_run_one_common(
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
            mock_graph,
        )
        with pytest.raises(RuntimeError, match="graph crashed"):
            await run_one(cfg)

    # parquet should still have been closed
    mock_pw.close.assert_called_once()
    # _update_run_failed should have been called with failed status
    mock_urf.assert_called_once()
    call_kwargs = mock_urf.call_args
    assert call_kwargs[1]["status"] == "failed"


# ---------------------------------------------------------------------------
# Test: flush-before-update order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_before_update_order_on_success() -> None:
    """parquet.close() MUST be called before _update_run_success (flush invariant)."""
    cfg = _make_cfg()
    final_state = _make_final_state(final_answer="fib(10) = 55")
    call_order: list[str] = []

    with (
        patch("atm.experiment.runner.create_engine") as mock_ce,
        patch("atm.experiment.runner.create_session_factory") as mock_csf,
        patch("atm.experiment.runner.Base.metadata.create_all"),
        patch("atm.experiment.runner._ensure_experiment", new_callable=AsyncMock) as mock_ee,
        patch("atm.experiment.runner._insert_run", new_callable=AsyncMock) as mock_ir,
        patch("atm.experiment.runner._update_run_success", new_callable=AsyncMock) as mock_urs,
        patch("atm.experiment.runner.ParquetWriter") as mock_pw_cls,
        patch("atm.experiment.runner.ExperimentCallbackHandler"),
        patch("atm.experiment.runner._build_agents", return_value={}),
        patch("atm.experiment.runner.TopologyRegistry") as mock_reg,
        patch("atm.experiment.runner.checkpointer_scope") as mock_cp_scope,
        patch("atm.experiment.runner._load_pricing", return_value=MagicMock()),
    ):
        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        mock_pw = AsyncMock()

        async def track_close() -> None:
            call_order.append("parquet.close")

        mock_pw.close = track_close

        async def track_update_success(*args: Any, **kwargs: Any) -> None:
            call_order.append("update_run_success")

        mock_urs.side_effect = track_update_success

        mock_graph = _make_mock_graph(final_state)
        _patch_run_one_common(
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
            mock_graph,
        )
        await run_one(cfg)

    # parquet.close must come BEFORE update_run_success
    assert "parquet.close" in call_order
    assert "update_run_success" in call_order
    close_idx = call_order.index("parquet.close")
    update_idx = call_order.index("update_run_success")
    assert close_idx < update_idx, (
        f"parquet.close (idx={close_idx}) must come before update_run_success (idx={update_idx})"
    )


@pytest.mark.asyncio
async def test_flush_before_update_order_on_budget_exceeded() -> None:
    """parquet.close() MUST be called before _update_run_failed on BudgetExceededError."""
    cfg = _make_cfg()
    call_order: list[str] = []

    with (
        patch("atm.experiment.runner.create_engine") as mock_ce,
        patch("atm.experiment.runner.create_session_factory") as mock_csf,
        patch("atm.experiment.runner.Base.metadata.create_all"),
        patch("atm.experiment.runner._ensure_experiment", new_callable=AsyncMock) as mock_ee,
        patch("atm.experiment.runner._insert_run", new_callable=AsyncMock) as mock_ir,
        patch("atm.experiment.runner._update_run_failed", new_callable=AsyncMock) as mock_urf,
        patch("atm.experiment.runner.ParquetWriter") as mock_pw_cls,
        patch("atm.experiment.runner.ExperimentCallbackHandler"),
        patch("atm.experiment.runner._build_agents", return_value={}),
        patch("atm.experiment.runner.TopologyRegistry") as mock_reg,
        patch("atm.experiment.runner.checkpointer_scope") as mock_cp_scope,
        patch("atm.experiment.runner._load_pricing", return_value=MagicMock()),
    ):
        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        mock_pw = AsyncMock()

        async def track_close() -> None:
            call_order.append("parquet.close")

        mock_pw.close = track_close

        async def track_update_failed(*args: Any, **kwargs: Any) -> None:
            call_order.append("update_run_failed")

        mock_urf.side_effect = track_update_failed

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            side_effect=BudgetExceededError(level="run", limit_usd=5.0, spent_usd=5.01)
        )
        _patch_run_one_common(
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
            mock_graph,
        )
        await run_one(cfg)

    assert "parquet.close" in call_order
    assert "update_run_failed" in call_order
    close_idx = call_order.index("parquet.close")
    update_idx = call_order.index("update_run_failed")
    assert close_idx < update_idx, (
        f"parquet.close (idx={close_idx}) must come before update_run_failed (idx={update_idx})"
    )


# ---------------------------------------------------------------------------
# Test: RunResult model
# ---------------------------------------------------------------------------


def test_run_result_fields() -> None:
    """RunResult Pydantic model has the required fields."""
    run_id = uuid.uuid4()
    exp_id = uuid.uuid4()
    result = RunResult(
        run_id=run_id,
        exp_id=exp_id,
        status="completed",
        metrics={"quality_score": 1.0, "cost_usd": 0.01, "iters": 3},
        final_answer="The answer is 55",
    )
    assert result.run_id == run_id
    assert result.exp_id == exp_id
    assert result.status == "completed"
    assert result.metrics["quality_score"] == 1.0
    assert result.metrics["cost_usd"] == 0.01
    assert result.metrics["iters"] == 3
    assert result.final_answer == "The answer is 55"


# ---------------------------------------------------------------------------
# Test: _build_agents uses role name as dict key (BUG-1 regression)
# ---------------------------------------------------------------------------


def test_build_agents_keys_by_role_name() -> None:
    """_build_agents must return dict keyed by role name (not 'role_agent').

    BUG-1 fix: topology nodes look up agents by role (e.g. agents["critic"]),
    not by the old convention of f"{role}_agent".
    """
    from atm.experiment.runner import _build_agents

    cfg = _make_cfg()

    # Build a minimal pricing and budget so LLMWrapper doesn't fail
    from atm.llm.budget import BudgetTracker
    from atm.llm.fake import FakeLLM
    from atm.llm.pricing import Pricing
    from atm.llm.wrapper import LLMWrapper

    fake_llm = LLMWrapper(
        model_id="fake:echo",
        pricing=Pricing(version=1, models={}),
        budget=BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0),
        llm=FakeLLM(mode="echo"),
    )
    llms = {
        "planner": fake_llm,
        "executor": fake_llm,
        "critic": fake_llm,
        "researcher": fake_llm,
    }

    # Find the conf dir
    from pathlib import Path as _Path

    conf_dir = _Path(__file__).parent.parent.parent.parent / "conf"

    agents = _build_agents(cfg, llms, conf_dir=conf_dir)

    # Keys must be role names, not "planner_agent" etc.
    for key in agents:
        assert not key.endswith("_agent"), (
            f"_build_agents dict key must be role name, got '{key}'. "
            "Expected one of: 'planner', 'executor', 'critic', 'researcher'."
        )
    # At least the three core roles should be present
    for role in ["planner", "executor", "critic"]:
        assert role in agents, (
            f"Expected agents['{role}'] to exist; got keys: {list(agents.keys())}"
        )


def test_build_agents_critic_is_critic_subclass() -> None:
    """_build_agents must instantiate Critic for the 'critic' role (BUG-3 fix).

    The Critic subclass overrides step() to emit DECISION messages.
    """
    from atm.agents.critic import Critic
    from atm.experiment.runner import _build_agents
    from atm.llm.budget import BudgetTracker
    from atm.llm.fake import FakeLLM
    from atm.llm.pricing import Pricing
    from atm.llm.wrapper import LLMWrapper

    cfg = _make_cfg()

    fake_llm = LLMWrapper(
        model_id="fake:echo",
        pricing=Pricing(version=1, models={}),
        budget=BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0),
        llm=FakeLLM(mode="echo"),
    )
    llms = {
        "planner": fake_llm,
        "executor": fake_llm,
        "critic": fake_llm,
        "researcher": fake_llm,
    }

    from pathlib import Path as _Path

    conf_dir = _Path(__file__).parent.parent.parent.parent / "conf"

    agents = _build_agents(cfg, llms, conf_dir=conf_dir)

    assert "critic" in agents, "agents['critic'] must exist"
    assert isinstance(agents["critic"], Critic), (
        f"agents['critic'] must be Critic instance, got {type(agents['critic'])}"
    )


# ---------------------------------------------------------------------------
# Test: run_one instantiates topology class (BUG-4 regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_instantiates_topology_class() -> None:
    """run_one must call topology_cls() before topology_instance.build().

    BUG-4 fix: TopologyRegistry.get() returns the CLASS, not an instance.
    Runner must instantiate it: cls = get(name); instance = cls(); instance.build(...)
    """
    cfg = _make_cfg()
    final_state = _make_final_state(final_answer="The answer is 55")

    instantiation_log: list[str] = []

    class _FakeTopo:
        def __init__(self) -> None:
            instantiation_log.append("instantiated")

        def build(self, agents: Any, cfg: Any, **kw: Any) -> AsyncMock:
            instantiation_log.append("build_called")
            mock = AsyncMock()
            mock.ainvoke = AsyncMock(return_value=final_state)
            return mock

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
        mock_pw = _make_mock_parquet_writer()
        mock_ee.return_value = exp_id
        mock_ir.return_value = run_id

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_ce.return_value = mock_engine
        mock_csf.return_value = MagicMock()
        mock_pw_cls.return_value = mock_pw

        # Registry returns the CLASS (not an instance) — runner must instantiate it
        mock_reg.get.return_value = _FakeTopo

        mock_cp = AsyncMock()

        async def _cp_aenter(self: Any) -> Any:
            return mock_cp

        async def _cp_aexit(self: Any, *args: Any) -> bool:
            return False

        mock_cp_scope.return_value.__aenter__ = _cp_aenter
        mock_cp_scope.return_value.__aexit__ = _cp_aexit

        await run_one(cfg)

    # The topology class must have been instantiated (constructor called)
    assert "instantiated" in instantiation_log, (
        "Topology class must be instantiated (cls()) before build() is called. "
        f"Calls seen: {instantiation_log}"
    )
    assert "build_called" in instantiation_log, (
        "topology_instance.build() must be called after instantiation."
    )


# ---------------------------------------------------------------------------
# Test: F4 — human_gateway_llm is built and passed to topology.build when needed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_passes_human_gateway_llm_when_human_enabled() -> None:
    """F4 fix: when cfg.human.enabled=True and gateway='llm_simulated', run_one must
    build a separate LLMWrapper for the HITL gateway and pass it as
    human_gateway_llm=... kwarg to topology_instance.build().

    Without this fix, LLMSimulatedGateway is constructed with llm=None and crashes
    on the first ainvoke with AttributeError.
    """
    from atm.experiment.config import HumanCfg

    cfg = _make_cfg(human=HumanCfg(enabled=True, gateway="llm_simulated", timeout_policy="skip"))
    final_state = _make_final_state(final_answer="The answer is 55")

    build_kwargs_log: list[dict[str, Any]] = []

    class _FakeTopoF4:
        def __init__(self) -> None:
            pass

        def build(self, agents: Any, cfg: Any, **kw: Any) -> AsyncMock:
            build_kwargs_log.append(dict(kw))
            mock = AsyncMock()
            mock.ainvoke = AsyncMock(return_value=final_state)
            return mock

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
        patch("atm.experiment.runner.build_llm") as mock_build_llm,
    ):
        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        mock_pw = _make_mock_parquet_writer()
        mock_ee.return_value = exp_id
        mock_ir.return_value = run_id

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_ce.return_value = mock_engine
        mock_csf.return_value = MagicMock()
        mock_pw_cls.return_value = mock_pw
        mock_reg.get.return_value = _FakeTopoF4

        # build_llm is called both for regular LLM wrappers AND for human_gateway_llm.
        # We use a fake LLMWrapper for each call.
        from atm.llm.budget import BudgetTracker
        from atm.llm.fake import FakeLLM
        from atm.llm.pricing import Pricing
        from atm.llm.wrapper import LLMWrapper

        def _make_fake_wrapper(*args: Any, **kwargs: Any) -> LLMWrapper:
            return LLMWrapper(
                model_id="fake:echo",
                pricing=Pricing(version=1, models={}),
                budget=BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0),
                llm=FakeLLM(mode="echo"),
            )

        mock_build_llm.side_effect = _make_fake_wrapper

        mock_cp = AsyncMock()

        async def _cp_aenter(self: Any) -> Any:
            return mock_cp

        async def _cp_aexit(self: Any, *args: Any) -> bool:
            return False

        mock_cp_scope.return_value.__aenter__ = _cp_aenter
        mock_cp_scope.return_value.__aexit__ = _cp_aexit

        await run_one(cfg)

    # build() must have been called with human_gateway_llm kwarg
    assert len(build_kwargs_log) == 1, (
        f"Expected exactly 1 build() call, got {len(build_kwargs_log)}"
    )
    build_kwargs = build_kwargs_log[0]

    assert "human_gateway_llm" in build_kwargs, (
        "F4 regression: human_gateway_llm not passed to topology.build(). "
        f"Actual kwargs: {list(build_kwargs.keys())}"
    )
    assert build_kwargs["human_gateway_llm"] is not None, (
        "F4 regression: human_gateway_llm passed as None to topology.build()."
    )


@pytest.mark.asyncio
async def test_run_one_does_not_pass_human_gateway_llm_when_human_disabled() -> None:
    """When cfg.human is None (default), human_gateway_llm must be None (no extra
    build_llm call for HITL)."""
    cfg = _make_cfg()  # human=None by default
    final_state = _make_final_state(final_answer="The answer is 55")

    build_kwargs_log: list[dict[str, Any]] = []

    class _FakeTopoNoHuman:
        def __init__(self) -> None:
            pass

        def build(self, agents: Any, cfg: Any, **kw: Any) -> AsyncMock:
            build_kwargs_log.append(dict(kw))
            mock = AsyncMock()
            mock.ainvoke = AsyncMock(return_value=final_state)
            return mock

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
        mock_pw = _make_mock_parquet_writer()
        mock_ee.return_value = exp_id
        mock_ir.return_value = run_id

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_ce.return_value = mock_engine
        mock_csf.return_value = MagicMock()
        mock_pw_cls.return_value = mock_pw
        mock_reg.get.return_value = _FakeTopoNoHuman

        mock_cp = AsyncMock()

        async def _cp_aenter(self: Any) -> Any:
            return mock_cp

        async def _cp_aexit(self: Any, *args: Any) -> bool:
            return False

        mock_cp_scope.return_value.__aenter__ = _cp_aenter
        mock_cp_scope.return_value.__aexit__ = _cp_aexit

        await run_one(cfg)

    assert len(build_kwargs_log) == 1
    build_kwargs = build_kwargs_log[0]
    # human_gateway_llm should be None when human is not configured
    assert build_kwargs.get("human_gateway_llm") is None, (
        "When cfg.human is None, human_gateway_llm must be None (not built). "
        f"Got: {build_kwargs.get('human_gateway_llm')!r}"
    )
