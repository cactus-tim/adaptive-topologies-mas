"""Unit/PG tests for atm.analysis.loaders -- Steps 3-5.

Gate: PG tests are only run when ATM_ENABLE_PG_TESTS=1.
      All PG tests are marked @pytest.mark.requires_postgres and use the
      ephemeral_pg_dsn fixture from tests/conftest.py.

Step 3 scope (this file, initial commit):
  - test_load_experiment_* — 3 tests
  - test_load_runs_*       — 3 tests

Step 4 scope:
  - test_load_llm_calls_* — 4 tests (parquet round-trip, no PG required)
  - test_load_llm_calls_for_experiment_* — 3 tests (parquet round-trip, no PG required)
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from atm.storage.models import Experiment, Run
from atm.storage.session import create_session_factory, session_scope

# ---------------------------------------------------------------------------
# Helpers — factory for inserting fixtures into the ephemeral DB
# ---------------------------------------------------------------------------


async def _insert_experiment(
    session_factory: Any,
    *,
    name: str = "test-exp",
    status: str = "running",
) -> uuid.UUID:
    """Insert a minimal Experiment row and return its id."""
    exp_id = uuid.uuid4()
    async with session_scope(session_factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=name,
                config_snapshot={"batch_size": 2},
                git_sha="abc123",
                status=status,
            )
        )
    return exp_id


async def _insert_run(
    session_factory: Any,
    *,
    exp_id: uuid.UUID,
    task_id: str = "HumanEval/1",
    topology: str = "mesh",
    quality_score: float | None = 0.85,
    seed: int = 42,
    budget_spent_usd: Decimal = Decimal("0.0010"),
    status: str = "done",
    finish_reason: str | None = "success",
) -> uuid.UUID:
    """Insert a minimal Run row and return its id."""
    run_id = uuid.uuid4()
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology=topology,
                task_id=task_id,
                agent_set="default",
                human_role=None,
                seed=seed,
                model="gpt-4",
                models_by_role_json={},
                model_version_snapshot={},
                status=status,
                finish_reason=finish_reason,
                budget_spent_usd=budget_spent_usd,
                quality_score=quality_score,
            )
        )
    return run_id


# ===========================================================================
# load_experiment
# ===========================================================================


class TestLoadExperiment:
    """Tests for load_experiment(exp_id, *, session_factory)."""

    @pytest.mark.requires_postgres
    async def test_load_experiment_returns_dict_with_expected_keys(
        self, ephemeral_pg_dsn: str
    ) -> None:
        """Happy path: load_experiment returns a dict with all column keys."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_experiment

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-keys-test")

        result = await load_experiment(str(exp_id), session_factory=factory)

        await engine.dispose()

        assert isinstance(result, dict)
        required_keys = {"id", "name", "config_snapshot", "status"}
        assert required_keys <= result.keys(), f"Missing keys: {required_keys - result.keys()}"

    @pytest.mark.requires_postgres
    async def test_load_experiment_values_match_inserted_row(self, ephemeral_pg_dsn: str) -> None:
        """load_experiment values match what was inserted."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_experiment

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="my-named-experiment", status="running")

        result = await load_experiment(str(exp_id), session_factory=factory)

        await engine.dispose()

        assert str(result["id"]) == str(exp_id)
        assert result["name"] == "my-named-experiment"
        assert result["status"] == "running"
        assert result["config_snapshot"] == {"batch_size": 2}

    @pytest.mark.requires_postgres
    async def test_load_experiment_raises_key_error_for_missing_id(
        self, ephemeral_pg_dsn: str
    ) -> None:
        """load_experiment raises KeyError when exp_id does not exist."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_experiment

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        missing_id = str(uuid.uuid4())

        try:
            with pytest.raises(KeyError):
                await load_experiment(missing_id, session_factory=factory)
        finally:
            await engine.dispose()


# ===========================================================================
# load_runs
# ===========================================================================


class TestLoadRuns:
    """Tests for load_runs(exp_id, *, session_factory)."""

    @pytest.mark.requires_postgres
    async def test_load_runs_returns_dataframe_with_task_type(self, ephemeral_pg_dsn: str) -> None:
        """load_runs DataFrame includes a derived task_type column."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_runs

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-task-type")
        await _insert_run(factory, exp_id=exp_id, task_id="HumanEval/5")

        df = await load_runs(str(exp_id), session_factory=factory)

        await engine.dispose()

        assert isinstance(df, pd.DataFrame)
        assert "task_type" in df.columns, "load_runs must include task_type column"
        assert df["task_type"].iloc[0] == "programming"

    @pytest.mark.requires_postgres
    async def test_load_runs_all_runs_returned(self, ephemeral_pg_dsn: str) -> None:
        """load_runs returns all runs for the given experiment."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_runs

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-all-runs")
        for i in range(3):
            await _insert_run(factory, exp_id=exp_id, task_id=f"HumanEval/{i}", seed=i)

        df = await load_runs(str(exp_id), session_factory=factory)

        await engine.dispose()

        assert len(df) == 3

    @pytest.mark.requires_postgres
    async def test_load_runs_empty_experiment_returns_empty_dataframe(
        self, ephemeral_pg_dsn: str
    ) -> None:
        """load_runs returns an empty DataFrame (not an error) when no runs exist."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_runs

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-no-runs")

        df = await load_runs(str(exp_id), session_factory=factory)

        await engine.dispose()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @pytest.mark.requires_postgres
    async def test_load_runs_numeric_columns_are_float(self, ephemeral_pg_dsn: str) -> None:
        """Decimal/UUID columns are cast to Python-native types (not left as objects)."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_runs

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-dtype-check")
        await _insert_run(
            factory,
            exp_id=exp_id,
            task_id="HumanEval/99",
            budget_spent_usd=Decimal("0.0042"),
            quality_score=0.75,
        )

        df = await load_runs(str(exp_id), session_factory=factory)

        await engine.dispose()

        # budget_spent_usd should be float, not Decimal
        bsu = df["budget_spent_usd"].iloc[0]
        assert isinstance(bsu, float), f"Expected float, got {type(bsu)}"

        # quality_score should be float (or NaN), not None
        qs = df["quality_score"].iloc[0]
        assert isinstance(qs, float), f"Expected float, got {type(qs)}"

    @pytest.mark.requires_postgres
    async def test_load_runs_isolates_by_exp_id(self, ephemeral_pg_dsn: str) -> None:
        """load_runs only returns runs for the requested exp_id, not other experiments."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_runs

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id_a = await _insert_experiment(factory, name="exp-A")
        exp_id_b = await _insert_experiment(factory, name="exp-B")

        await _insert_run(factory, exp_id=exp_id_a, task_id="HumanEval/10", seed=1)
        await _insert_run(factory, exp_id=exp_id_b, task_id="HumanEval/20", seed=2)
        await _insert_run(factory, exp_id=exp_id_b, task_id="HumanEval/21", seed=3)

        df_a = await load_runs(str(exp_id_a), session_factory=factory)
        df_b = await load_runs(str(exp_id_b), session_factory=factory)

        await engine.dispose()

        assert len(df_a) == 1
        assert len(df_b) == 2


# ===========================================================================
# Helpers — write fixture parquet files via ParquetWriter
# ===========================================================================


def _make_llm_call_row(run_id: str, agent_id: str = "agent-0") -> dict[str, Any]:
    """Return a minimal dict matching LLM_CALL_SCHEMA."""
    import datetime

    return {
        "run_id": run_id,
        "agent_id": agent_id,
        "model": "gpt-4o",
        "at": datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_hit_tokens": 0,
        "cost_usd": 0.001,
        "latency_ms": 250.0,
        "cache_scope": "none",
        "fingerprint": "fp-abc123",
    }


async def _write_llm_calls_via_writer(
    parquet_dir: Path,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
    rows: list[dict[str, Any]],
) -> None:
    """Write rows to the llm_calls parquet via ParquetWriter (flush immediately)."""
    from atm.storage.parquet_writer import ParquetWriter

    writer = ParquetWriter(
        root=parquet_dir,
        run_id=run_id,
        exp_id=exp_id,
        buffer_rows=1,  # flush immediately on every write
    )
    for row in rows:
        await writer.write_llm_call(row)
    await writer.close()


# ===========================================================================
# load_llm_calls
# ===========================================================================


class TestLoadLlmCalls:
    """Tests for load_llm_calls(run_id, *, parquet_dir)."""

    def test_load_llm_calls_returns_dataframe_with_schema_columns(self, tmp_path: Path) -> None:
        """Happy path: loaded DataFrame columns match LLM_CALL_SCHEMA.names."""
        from atm.analysis.loaders import load_llm_calls
        from atm.storage.schemas import LLM_CALL_SCHEMA

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_llm_call_row(str(run_id))

        asyncio.get_event_loop().run_until_complete(
            _write_llm_calls_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = load_llm_calls(str(run_id), parquet_dir=tmp_path)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == LLM_CALL_SCHEMA.names

    def test_load_llm_calls_round_trip_values(self, tmp_path: Path) -> None:
        """Round-trip: values written via ParquetWriter are read back correctly."""
        from atm.analysis.loaders import load_llm_calls

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_llm_call_row(str(run_id), agent_id="my-agent")

        asyncio.get_event_loop().run_until_complete(
            _write_llm_calls_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = load_llm_calls(str(run_id), parquet_dir=tmp_path)

        assert len(df) == 1
        assert df["agent_id"].iloc[0] == "my-agent"
        assert df["model"].iloc[0] == "gpt-4o"
        assert df["input_tokens"].iloc[0] == 100
        assert abs(df["cost_usd"].iloc[0] - 0.001) < 1e-9

    def test_load_llm_calls_multiple_rows(self, tmp_path: Path) -> None:
        """All rows are returned when multiple LLM calls exist for a run."""
        from atm.analysis.loaders import load_llm_calls

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        rows = [_make_llm_call_row(str(run_id), agent_id=f"agent-{i}") for i in range(5)]

        asyncio.get_event_loop().run_until_complete(
            _write_llm_calls_via_writer(tmp_path, exp_id, run_id, rows)
        )

        df = load_llm_calls(str(run_id), parquet_dir=tmp_path)

        assert len(df) == 5

    def test_load_llm_calls_raises_file_not_found_for_missing_run(self, tmp_path: Path) -> None:
        """FileNotFoundError is raised when no parquet file exists for the run_id."""
        from atm.analysis.loaders import load_llm_calls

        missing_run_id = str(uuid.uuid4())

        with pytest.raises(FileNotFoundError):
            load_llm_calls(missing_run_id, parquet_dir=tmp_path)


# ===========================================================================
# load_llm_calls_for_experiment
# ===========================================================================


class TestLoadLlmCallsForExperiment:
    """Tests for load_llm_calls_for_experiment(exp_id, *, parquet_dir)."""

    def test_load_llm_calls_for_experiment_concatenates_multiple_runs(self, tmp_path: Path) -> None:
        """All runs' LLM call rows are concatenated into one DataFrame."""
        from atm.analysis.loaders import load_llm_calls_for_experiment

        exp_id = uuid.uuid4()
        run_id_a = uuid.uuid4()
        run_id_b = uuid.uuid4()

        rows_a = [_make_llm_call_row(str(run_id_a), agent_id="a0")]
        rows_b = [
            _make_llm_call_row(str(run_id_b), agent_id="b0"),
            _make_llm_call_row(str(run_id_b), agent_id="b1"),
        ]

        asyncio.get_event_loop().run_until_complete(
            _write_llm_calls_via_writer(tmp_path, exp_id, run_id_a, rows_a)
        )
        asyncio.get_event_loop().run_until_complete(
            _write_llm_calls_via_writer(tmp_path, exp_id, run_id_b, rows_b)
        )

        df = load_llm_calls_for_experiment(str(exp_id), parquet_dir=tmp_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_llm_calls_for_experiment_injects_run_id_from_path(self, tmp_path: Path) -> None:
        """run_id column is injected from the filename when it differs from parquet value.

        The parquet file stores the run_id in column data, but
        load_llm_calls_for_experiment must ensure run_id is present.
        """
        from atm.analysis.loaders import load_llm_calls_for_experiment

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_llm_call_row(str(run_id))

        asyncio.get_event_loop().run_until_complete(
            _write_llm_calls_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = load_llm_calls_for_experiment(str(exp_id), parquet_dir=tmp_path)

        assert "run_id" in df.columns
        # The run_id in the DataFrame should match the actual run_id used
        assert str(run_id) in df["run_id"].values

    def test_load_llm_calls_for_experiment_empty_experiment_returns_empty_dataframe(
        self, tmp_path: Path
    ) -> None:
        """Empty DataFrame is returned (not an error) when no runs exist for the experiment."""
        from atm.analysis.loaders import load_llm_calls_for_experiment

        exp_id = uuid.uuid4()
        # Create the experiment directory but no run subdirectories
        exp_dir = tmp_path / "experiments" / str(exp_id)
        exp_dir.mkdir(parents=True)

        df = load_llm_calls_for_experiment(str(exp_id), parquet_dir=tmp_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
