"""Unit tests for _run_cell_worker — picklability + dict-shape contract.

These tests run in-process (no actual ProcessPoolExecutor spawn) by monkeypatching
``atm.experiment.runner.run_one`` to a deterministic stub.
"""

from __future__ import annotations

import pickle
import uuid
from typing import Any
from uuid import UUID

import pytest

from atm.experiment.config import ExperimentConfig
from atm.experiment.runner import RunResult


def _make_min_cfg() -> ExperimentConfig:
    """Build a minimal-but-valid ExperimentConfig (no PG round-trip)."""
    return ExperimentConfig.model_validate(
        {
            "name": "grid-worker-test",
            "seed": 7,
            "model": {
                "default": "fake:echo",
                "by_role": {},
            },
            "agents": {"set": "minimal"},
            "topology": {"name": "star", "max_iterations": 3},
            "task": {"name": "smoke", "input": "hello"},
            "observability": {
                "pg_dsn": "postgresql+asyncpg://x:y@localhost:5432/nowhere",
                "parquet_dir": "/tmp/atm-grid-worker-test",
            },
        }
    )


# ---------------------------------------------------------------------------
# Picklability — cfg.model_dump must round-trip through pickle
# ---------------------------------------------------------------------------


def test_cfg_dump_is_picklable() -> None:
    """ExperimentConfig.model_dump(mode='python') must be picklable
    (so ProcessPoolExecutor can ship it to workers)."""
    cfg = _make_min_cfg()
    dumped = cfg.model_dump(mode="python")
    blob = pickle.dumps(dumped)
    restored = pickle.loads(blob)
    # Must round-trip through model_validate too.
    cfg2 = ExperimentConfig.model_validate(restored)
    assert cfg2.name == cfg.name
    assert cfg2.seed == cfg.seed
    assert cfg2.topology.name == cfg.topology.name


# ---------------------------------------------------------------------------
# _run_cell_worker — happy path (mocked run_one)
# ---------------------------------------------------------------------------


def test_run_cell_worker_returns_dict_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker should serialize RunResult into a picklable dict with the documented keys."""
    fake_run_id = uuid.uuid4()
    fake_exp_id = uuid.uuid4()

    async def fake_run_one(cfg: ExperimentConfig) -> RunResult:
        return RunResult(
            run_id=fake_run_id,
            exp_id=fake_exp_id,
            status="completed",
            metrics={"quality_score": 0.75, "cost_usd": 0.002, "iters": 4},
            final_answer="42",
        )

    monkeypatch.setattr("atm.experiment.runner.run_one", fake_run_one)

    from atm.experiment.grid import _run_cell_worker

    cfg = _make_min_cfg()
    result = _run_cell_worker(cfg.model_dump(mode="python"))

    # Shape contract.
    expected_keys = {
        "run_id",
        "exp_id",
        "status",
        "quality_score",
        "cost_usd",
        "iters",
        "final_answer",
        "error",
    }
    assert set(result.keys()) == expected_keys
    assert result["run_id"] == str(fake_run_id)
    assert result["exp_id"] == str(fake_exp_id)
    assert result["status"] == "completed"
    assert result["quality_score"] == 0.75
    assert result["cost_usd"] == pytest.approx(0.002)
    assert result["iters"] == 4
    assert result["final_answer"] == "42"
    assert result["error"] is None

    # Whole dict must be picklable.
    pickle.dumps(result)


# ---------------------------------------------------------------------------
# _run_cell_worker — run_one raises
# ---------------------------------------------------------------------------


def test_run_cell_worker_catches_run_one_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_one(cfg: ExperimentConfig) -> RunResult:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("atm.experiment.runner.run_one", fake_run_one)

    from atm.experiment.grid import _run_cell_worker

    cfg = _make_min_cfg()
    result = _run_cell_worker(cfg.model_dump(mode="python"))

    assert result["status"] == "failed"
    assert result["run_id"] is None
    assert result["error"] is not None
    assert "kaboom" in result["error"]


# ---------------------------------------------------------------------------
# _run_cell_worker — config validation failure
# ---------------------------------------------------------------------------


def test_run_cell_worker_handles_invalid_cfg_dict() -> None:
    from atm.experiment.grid import _run_cell_worker

    bad_dict: dict[str, Any] = {"this": "is not a valid cfg"}
    result = _run_cell_worker(bad_dict)
    assert result["status"] == "failed"
    assert result["error"] is not None
    assert "config validation failed" in result["error"]


# ---------------------------------------------------------------------------
# Result dict round-trips through pickle (it must, to come back from worker)
# ---------------------------------------------------------------------------


def test_run_cell_worker_result_is_picklable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_one(cfg: ExperimentConfig) -> RunResult:
        return RunResult(
            run_id=uuid.uuid4(),
            exp_id=uuid.uuid4(),
            status="budget_exceeded",
            metrics={"quality_score": None, "cost_usd": 10.0, "iters": 2},
            final_answer="",
        )

    monkeypatch.setattr("atm.experiment.runner.run_one", fake_run_one)

    from atm.experiment.grid import _run_cell_worker

    cfg = _make_min_cfg()
    result = _run_cell_worker(cfg.model_dump(mode="python"))
    blob = pickle.dumps(result)
    restored = pickle.loads(blob)
    assert restored["status"] == "budget_exceeded"
    # UUID can be re-parsed
    UUID(restored["run_id"])
