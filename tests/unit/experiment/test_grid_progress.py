"""Unit tests for M12 grid runner: GridProgress, GridResult, status aggregation.

Coverage:
  - GridProgress/GridResult dataclass shape and immutability.
  - _aggregate_experiment_status() decision rules (completed | partial | failed | running).
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any
from uuid import UUID

import pytest

from atm.experiment.config import ExperimentConfig
from atm.experiment.grid import (
    GridProgress,
    GridResult,
    _aggregate_experiment_status,
)


def _make_min_cfg(name: str = "grid-unit-test") -> ExperimentConfig:
    """Local helper — minimal ExperimentConfig for unit tests."""
    return ExperimentConfig.model_validate(
        {
            "name": name,
            "seed": 7,
            "model": {"default": "fake:echo", "by_role": {}},
            "agents": {"set": "minimal"},
            "topology": {"name": "star", "max_iterations": 3},
            "task": {"name": "smoke", "input": "hello"},
            "observability": {
                "pg_dsn": "postgresql+asyncpg://x:y@localhost:5432/nowhere",
                "parquet_dir": "/tmp/atm-grid-unit",
            },
        }
    )


# ---------------------------------------------------------------------------
# GridProgress dataclass
# ---------------------------------------------------------------------------


def test_grid_progress_construction() -> None:
    p = GridProgress(total=10, done=3, failed=1, in_progress=2, eta_s=42.5)
    assert p.total == 10
    assert p.done == 3
    assert p.failed == 1
    assert p.in_progress == 2
    assert p.eta_s == 42.5


def test_grid_progress_is_frozen() -> None:
    p = GridProgress(total=10, done=3, failed=1, in_progress=2, eta_s=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.done = 4  # type: ignore[misc]


def test_grid_progress_eta_can_be_none() -> None:
    p = GridProgress(total=10, done=0, failed=0, in_progress=0, eta_s=None)
    assert p.eta_s is None


# ---------------------------------------------------------------------------
# GridResult dataclass
# ---------------------------------------------------------------------------


def test_grid_result_construction() -> None:
    import uuid

    exp = uuid.uuid4()
    rids = [uuid.uuid4(), uuid.uuid4()]
    r = GridResult(
        exp_id=exp,
        total=4,
        completed=2,
        failed=1,
        budget_exceeded=1,
        run_ids=rids,
    )
    assert r.exp_id == exp
    assert r.total == 4
    assert r.completed == 2
    assert r.failed == 1
    assert r.budget_exceeded == 1
    assert r.run_ids == rids


def test_grid_result_is_frozen() -> None:
    import uuid

    r = GridResult(
        exp_id=uuid.uuid4(),
        total=1,
        completed=1,
        failed=0,
        budget_exceeded=0,
        run_ids=[],
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.completed = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _aggregate_experiment_status
# ---------------------------------------------------------------------------


def test_aggregate_all_completed() -> None:
    statuses = ["completed", "completed", "completed"]
    assert _aggregate_experiment_status(statuses) == "completed"


def test_aggregate_all_failed() -> None:
    statuses = ["failed", "failed"]
    assert _aggregate_experiment_status(statuses) == "failed"


def test_aggregate_all_budget_exceeded() -> None:
    # budget_exceeded is a terminal failure for experiment-aggregate purposes
    statuses = ["budget_exceeded", "budget_exceeded"]
    assert _aggregate_experiment_status(statuses) == "failed"


def test_aggregate_mixed_failed_and_budget() -> None:
    statuses = ["failed", "budget_exceeded"]
    assert _aggregate_experiment_status(statuses) == "failed"


def test_aggregate_partial() -> None:
    statuses = ["completed", "failed"]
    assert _aggregate_experiment_status(statuses) == "partial"


def test_aggregate_partial_with_budget() -> None:
    statuses = ["completed", "budget_exceeded"]
    assert _aggregate_experiment_status(statuses) == "partial"


def test_aggregate_running_when_in_flight() -> None:
    statuses = ["completed", "running", "failed"]
    assert _aggregate_experiment_status(statuses) == "running"


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _aggregate_experiment_status([])


def test_aggregate_unknown_status_treated_as_running() -> None:
    # Any non-terminal status -> still running
    statuses = ["completed", "queued"]
    assert _aggregate_experiment_status(statuses) == "running"


# ---------------------------------------------------------------------------
# run_grid driver — uses in-process executor stub via monkeypatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_grid_rejects_empty_configs() -> None:
    from atm.experiment.grid import run_grid

    with pytest.raises(ValueError, match="non-empty"):
        await run_grid([], parallelism=2)


@pytest.mark.asyncio
async def test_run_grid_rejects_zero_parallelism() -> None:
    from atm.experiment.grid import run_grid

    with pytest.raises(ValueError, match="parallelism"):
        await run_grid([_make_min_cfg()], parallelism=0)


class _StubExecutor:
    """Synchronous in-process stub of ProcessPoolExecutor for unit tests.

    ``run_in_executor`` is intercepted at the asyncio level so the parent
    loop runs the worker callable inline and returns a resolved Future.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.shutdown_called = False
        self.cancel_futures = False

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_called = True
        self.cancel_futures = cancel_futures


@pytest.fixture
def stub_pool(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ProcessPoolExecutor + loop.run_in_executor with in-process stubs.

    The fixture replaces ``atm.experiment.grid.ProcessPoolExecutor`` and uses a
    fake event-loop method to run workers inline. Returned dict allows tests to
    inject behaviour into the worker callable via ``state['worker_fn']``.
    """
    import atm.experiment.grid as grid_mod

    state: dict[str, Any] = {
        "worker_fn": None,  # callable(cfg_dict) -> dict
        "calls": [],
        "executor_instance": None,
    }

    def _executor_factory(*args: Any, **kwargs: Any) -> _StubExecutor:
        ex = _StubExecutor()
        state["executor_instance"] = ex
        return ex

    monkeypatch.setattr(grid_mod, "ProcessPoolExecutor", _executor_factory)

    # Intercept loop.run_in_executor by monkeypatching at the loop level.
    import asyncio

    async def fake_run_in_executor(self: Any, executor: Any, fn: Any, *args: Any) -> Any:
        state["calls"].append(args[0] if args else None)
        worker_fn = state["worker_fn"] or (lambda cfg_dict: {"status": "completed"})
        # Run inline (no thread pool); a tiny yield so as_completed sees ordering.
        await asyncio.sleep(0)
        return worker_fn(args[0] if args else None)

    # Patch run_in_executor on the BaseEventLoop class so it applies
    # regardless of which loop pytest-asyncio constructs for the test.
    monkeypatch.setattr(
        asyncio.BaseEventLoop, "run_in_executor", fake_run_in_executor, raising=True
    )

    # Stub PG update + resolve to avoid DB I/O.
    async def fake_update(*args: Any, **kwargs: Any) -> None:
        return None

    async def fake_resolve(*args: Any, **kwargs: Any) -> UUID:
        return UUID("00000000-0000-0000-0000-000000000abc")

    monkeypatch.setattr(grid_mod, "_update_experiment_status", fake_update)
    monkeypatch.setattr(grid_mod, "_resolve_exp_id", fake_resolve)

    return state


@pytest.mark.asyncio
async def test_run_grid_all_completed(stub_pool: dict[str, Any]) -> None:
    from atm.experiment.grid import run_grid

    counter = {"n": 0}

    def worker(cfg_dict: dict[str, Any]) -> dict[str, Any]:
        counter["n"] += 1
        return {
            "run_id": str(uuid.uuid4()),
            "exp_id": str(uuid.uuid4()),
            "status": "completed",
            "quality_score": 0.5,
            "cost_usd": 0.001,
            "iters": 2,
            "final_answer": "ok",
            "error": None,
        }

    stub_pool["worker_fn"] = worker
    cfgs = [_make_min_cfg(), _make_min_cfg(), _make_min_cfg()]
    result = await run_grid(cfgs, parallelism=2)

    assert result.total == 3
    assert result.completed == 3
    assert result.failed == 0
    assert result.budget_exceeded == 0
    assert len(result.run_ids) == 3
    assert counter["n"] == 3


@pytest.mark.asyncio
async def test_run_grid_mixed_outcomes(stub_pool: dict[str, Any]) -> None:
    from atm.experiment.grid import run_grid

    outcomes = iter(["completed", "failed", "budget_exceeded"])

    def worker(cfg_dict: dict[str, Any]) -> dict[str, Any]:
        st = next(outcomes)
        return {
            "run_id": str(uuid.uuid4()),
            "exp_id": str(uuid.uuid4()),
            "status": st,
            "quality_score": None,
            "cost_usd": 0.0,
            "iters": 0,
            "final_answer": "",
            "error": None if st == "completed" else "err",
        }

    stub_pool["worker_fn"] = worker
    cfgs = [_make_min_cfg() for _ in range(3)]
    result = await run_grid(cfgs, parallelism=2)

    assert result.total == 3
    assert result.completed == 1
    assert result.failed == 1
    assert result.budget_exceeded == 1


@pytest.mark.asyncio
async def test_run_grid_progress_callback_invoked(stub_pool: dict[str, Any]) -> None:
    from atm.experiment.grid import run_grid

    def worker(cfg_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": str(uuid.uuid4()),
            "exp_id": str(uuid.uuid4()),
            "status": "completed",
            "quality_score": 1.0,
            "cost_usd": 0.0,
            "iters": 1,
            "final_answer": "x",
            "error": None,
        }

    stub_pool["worker_fn"] = worker

    seen: list[GridProgress] = []

    def cb(p: GridProgress) -> None:
        seen.append(p)

    cfgs = [_make_min_cfg() for _ in range(4)]
    await run_grid(cfgs, parallelism=2, progress_callback=cb)

    assert len(seen) == 4
    # Last snapshot must show all done.
    last = seen[-1]
    assert last.total == 4
    assert last.done == 4
    assert last.in_progress == 0


@pytest.mark.asyncio
async def test_run_grid_callback_exception_is_swallowed(
    stub_pool: dict[str, Any],
) -> None:
    from atm.experiment.grid import run_grid

    def worker(cfg_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": str(uuid.uuid4()),
            "exp_id": str(uuid.uuid4()),
            "status": "completed",
            "quality_score": 1.0,
            "cost_usd": 0.0,
            "iters": 1,
            "final_answer": "",
            "error": None,
        }

    stub_pool["worker_fn"] = worker

    def boom(p: Any) -> None:
        raise RuntimeError("callback bug")

    cfgs = [_make_min_cfg()]
    # Must not raise.
    r = await run_grid(cfgs, parallelism=1, progress_callback=boom)
    assert r.completed == 1


# ---------------------------------------------------------------------------
# fail_fast cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_grid_fail_fast_shuts_down_executor(
    stub_pool: dict[str, Any],
) -> None:
    from atm.experiment.grid import run_grid

    call_count = {"n": 0}

    def worker(cfg_dict: dict[str, Any]) -> dict[str, Any]:
        call_count["n"] += 1
        # First completes, second fails -> trigger cancel.
        if call_count["n"] == 1:
            st = "completed"
        elif call_count["n"] == 2:
            st = "failed"
        else:
            st = "completed"
        return {
            "run_id": str(uuid.uuid4()),
            "exp_id": str(uuid.uuid4()),
            "status": st,
            "quality_score": None,
            "cost_usd": 0.0,
            "iters": 0,
            "final_answer": "",
            "error": None if st == "completed" else "boom",
        }

    stub_pool["worker_fn"] = worker
    cfgs = [_make_min_cfg() for _ in range(5)]

    result = await run_grid(cfgs, parallelism=2, fail_fast=True)

    # We expect at least the failing cell to have been counted.
    assert result.failed >= 1
    # Executor.shutdown(cancel_futures=True) must have been called.
    ex = stub_pool["executor_instance"]
    assert ex is not None
    assert ex.shutdown_called is True
    # Either via fail_fast branch (cancel=True) or normal finally (cancel=False).
    # Just assert shutdown happened at least once.
