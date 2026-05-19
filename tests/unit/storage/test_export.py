"""Unit tests for atm.storage.export.

Smoke-level — exercises the conversion functions on plain objects
mirroring the SQLAlchemy schema (so we don't need a real PG to test).
The full end-to-end path is covered by an integration-style test that
runs against atm-postgres when ATM_ENABLE_PG_TESTS=1.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest

from atm.storage.export import (
    _experiment_to_dict,
    _run_to_row,
    _to_jsonable,
    export_experiment,
)


class TestToJsonable:
    def test_decimal_to_float(self) -> None:
        assert _to_jsonable(Decimal("1.5")) == 1.5

    def test_datetime_to_iso(self) -> None:
        dt = datetime(2026, 5, 17, 10, 30, tzinfo=UTC)
        assert _to_jsonable(dt) == dt.isoformat()

    def test_uuid_to_str(self) -> None:
        u = uuid4()
        assert _to_jsonable(u) == str(u)

    def test_nested_dict(self) -> None:
        d = {"a": Decimal("2"), "b": {"c": uuid4()}}
        out = _to_jsonable(d)
        assert out["a"] == 2.0
        assert isinstance(out["b"]["c"], str)

    def test_nested_list(self) -> None:
        lst = [Decimal("3"), datetime(2026, 1, 1)]
        out = _to_jsonable(lst)
        assert out[0] == 3.0
        assert isinstance(out[1], str)

    def test_passthrough_primitives(self) -> None:
        assert _to_jsonable("hello") == "hello"
        assert _to_jsonable(42) == 42
        assert _to_jsonable(None) is None


class TestExperimentToDict:
    def test_full_row(self) -> None:
        exp = SimpleNamespace(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            name="e3_pilot",
            config_snapshot={"key": "value"},
            git_sha="abc123",
            started_at=datetime(2026, 5, 17, 10, tzinfo=UTC),
            finished_at=datetime(2026, 5, 17, 12, tzinfo=UTC),
            total_cost_usd=Decimal("12.34"),
            status="completed",
        )
        out = _experiment_to_dict(exp)  # type: ignore[arg-type]
        assert out["id"] == "12345678-1234-5678-1234-567812345678"
        assert out["name"] == "e3_pilot"
        assert out["total_cost_usd"] == 12.34
        assert out["status"] == "completed"
        json.dumps(out)

    def test_handles_none_finished_at(self) -> None:
        exp = SimpleNamespace(
            id=uuid4(),
            name="x",
            config_snapshot=None,
            git_sha=None,
            started_at=datetime.now(UTC),
            finished_at=None,
            total_cost_usd=None,
            status="running",
        )
        out = _experiment_to_dict(exp)  # type: ignore[arg-type]
        assert out["finished_at"] is None
        assert out["total_cost_usd"] == 0.0


class TestRunToRow:
    def test_full_row(self) -> None:
        run = SimpleNamespace(
            id=UUID("11111111-1111-1111-1111-111111111111"),
            exp_id=UUID("22222222-2222-2222-2222-222222222222"),
            topology="adaptive",
            task_id="humaneval/0",
            agent_set="canonical_4",
            human_role="reviewer",
            seed=42,
            model="cerebras:gpt-oss-120b",
            models_by_role_json={"planner": "x"},
            model_version_snapshot={"v": "1"},
            sandbox_image_digest=None,
            status="completed",
            finish_reason="completed",
            budget_spent_usd=Decimal("0.0123"),
            quality_score=0.95,
            wall_time_s=42.0,
            iterations=6,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            error=None,
            cognitive_load_proxy=0.5,
            replay_of=None,
            host="host-1",
            process_pid=12345,
        )
        row = _run_to_row(run)  # type: ignore[arg-type]
        assert row["id"] == "11111111-1111-1111-1111-111111111111"
        assert row["topology"] == "adaptive"
        assert row["budget_spent_usd"] == 0.0123
        assert row["quality_score"] == 0.95
        assert json.loads(row["models_by_role_json"]) == {"planner": "x"}

    def test_handles_nullable_fields(self) -> None:
        run = SimpleNamespace(
            id=uuid4(),
            exp_id=uuid4(),
            topology="star",
            task_id="t/0",
            agent_set="canonical_4",
            human_role=None,
            seed=1,
            model="m",
            models_by_role_json=None,
            model_version_snapshot=None,
            sandbox_image_digest=None,
            status="failed",
            finish_reason="error",
            budget_spent_usd=None,
            quality_score=None,
            wall_time_s=None,
            iterations=None,
            started_at=datetime.now(UTC),
            finished_at=None,
            error="boom",
            cognitive_load_proxy=None,
            replay_of=None,
            host=None,
            process_pid=None,
        )
        row = _run_to_row(run)  # type: ignore[arg-type]
        assert row["quality_score"] is None
        assert row["budget_spent_usd"] == 0.0
        assert row["human_role"] is None
        assert row["error"] == "boom"


class _ScalarResult:
    """Mimic the .scalar_one_or_none() / .scalars().all() result API."""

    def __init__(self, exp: Any = None, runs: list[Any] | None = None) -> None:
        self._exp = exp
        self._runs = runs or []

    def scalar_one_or_none(self) -> Any:
        return self._exp

    def scalars(self) -> Any:
        return SimpleNamespace(all=lambda: list(self._runs))


class _FakeSession:
    """Async session stub — first execute returns exp, second returns runs."""

    def __init__(self, exp: Any, runs: list[Any]) -> None:
        self._exp = exp
        self._runs = runs
        self._call_count = 0

    async def execute(self, _stmt: Any) -> _ScalarResult:
        self._call_count += 1
        if self._call_count == 1:
            return _ScalarResult(exp=self._exp)
        return _ScalarResult(runs=self._runs)


class _AsyncCtx:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, *_args: Any) -> None:
        pass


def _make_factory(exp: Any, runs: list[Any]) -> Any:
    session = _FakeSession(exp, runs)
    return lambda: _AsyncCtx(session)


@pytest.mark.asyncio
async def test_export_experiment_writes_both_files(tmp_path: Path) -> None:
    exp_uuid = UUID("12345678-1234-5678-1234-567812345678")
    exp = SimpleNamespace(
        id=exp_uuid,
        name="e3_smoke",
        config_snapshot={},
        git_sha=None,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        total_cost_usd=Decimal("0.5"),
        status="completed",
    )
    runs = [
        SimpleNamespace(
            id=uuid4(),
            exp_id=exp_uuid,
            topology="adaptive",
            task_id=f"t/{i}",
            agent_set="canonical_4",
            human_role="reviewer",
            seed=42 + i,
            model="cerebras:gpt-oss-120b",
            models_by_role_json={},
            model_version_snapshot={},
            sandbox_image_digest=None,
            status="completed",
            finish_reason="completed",
            budget_spent_usd=Decimal("0.05"),
            quality_score=1.0,
            wall_time_s=10.0,
            iterations=3,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            error=None,
            cognitive_load_proxy=None,
            replay_of=None,
            host="h",
            process_pid=1,
        )
        for i in range(3)
    ]

    sf = _make_factory(exp, runs)
    out_dir = await export_experiment(exp_uuid, session_factory=sf, root=tmp_path)

    assert out_dir == tmp_path / "experiments" / str(exp_uuid)
    assert (out_dir / "experiment.json").exists()
    assert (out_dir / "_runs.parquet").exists()

    exp_data = json.loads((out_dir / "experiment.json").read_text())
    assert exp_data["name"] == "e3_smoke"
    assert exp_data["total_cost_usd"] == 0.5

    df = pd.read_parquet(out_dir / "_runs.parquet")
    assert len(df) == 3
    assert set(df["topology"]) == {"adaptive"}
    assert df["quality_score"].tolist() == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_export_experiment_missing_raises(tmp_path: Path) -> None:
    sf = _make_factory(None, [])
    with pytest.raises(ValueError, match="experiment not found"):
        await export_experiment(uuid4(), session_factory=sf, root=tmp_path)


@pytest.mark.asyncio
async def test_export_experiment_no_runs_empty_parquet(tmp_path: Path) -> None:
    exp_uuid = uuid4()
    exp = SimpleNamespace(
        id=exp_uuid,
        name="empty",
        config_snapshot={},
        git_sha=None,
        started_at=datetime.now(UTC),
        finished_at=None,
        total_cost_usd=Decimal("0"),
        status="failed",
    )
    sf = _make_factory(exp, [])
    out_dir = await export_experiment(exp_uuid, session_factory=sf, root=tmp_path)
    df = pd.read_parquet(out_dir / "_runs.parquet")
    assert len(df) == 0
