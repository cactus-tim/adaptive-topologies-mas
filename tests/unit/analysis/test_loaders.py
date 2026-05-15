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
import datetime
import json
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from atm.storage.models import Experiment, Run
from atm.storage.session import create_session_factory, session_scope

_UTC = datetime.UTC

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


# ===========================================================================
# Helpers — write topology_transitions and phases via ParquetWriter
# ===========================================================================


def _make_topology_transition_row(run_id: str) -> dict[str, Any]:
    """Return a minimal dict matching TOPOLOGY_TRANSITION_SCHEMA field names."""
    return {
        "run_id": run_id,
        "from_topology": "chain",
        "to_topology": "mesh",
        "phase_at_decision": "execution",
        "iter_within_phase": 1,
        "iter_within_topology": 2,
        "decided_by": "llm_router",
        "reason": "quality dropped below threshold",
        "considered_alternatives_json": json.dumps(["chain", "star"]),
        "guards_applied_json": json.dumps(["min_agents_guard"]),
        "signals_snapshot_json": json.dumps({"quality": 0.4, "cost": 0.01}),
        "router_cost_usd": 0.0002,
        "at": datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=_UTC),
    }


def _make_phase_row(run_id: str) -> dict[str, Any]:
    """Return a minimal dict matching PHASE_SCHEMA field names."""
    return {
        "run_id": run_id,
        "phase_name": "execution",
        "from_phase": "planning",
        "started_at": datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=_UTC),
        "ended_at": datetime.datetime(2024, 1, 1, 12, 5, 0, tzinfo=_UTC),
        "entry_reason": "planning complete",
        "topology_used": "chain",
        "decided_by": "rule",
    }


async def _write_transitions_via_writer(
    parquet_dir: Path,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
    rows: list[dict[str, Any]],
) -> None:
    """Write topology_transition rows to parquet via ParquetWriter."""
    from atm.storage.parquet_writer import ParquetWriter

    writer = ParquetWriter(
        root=parquet_dir,
        run_id=run_id,
        exp_id=exp_id,
        buffer_rows=1,
    )
    for row in rows:
        await writer.write_topology_transition(row)
    await writer.close()


async def _write_phases_via_writer(
    parquet_dir: Path,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
    rows: list[dict[str, Any]],
) -> None:
    """Write phase rows to parquet via ParquetWriter."""
    from atm.storage.parquet_writer import ParquetWriter

    writer = ParquetWriter(
        root=parquet_dir,
        run_id=run_id,
        exp_id=exp_id,
        buffer_rows=1,
    )
    for row in rows:
        await writer.write_phase(row)
    await writer.close()


# ===========================================================================
# load_topology_transitions — parquet source
# ===========================================================================


class TestLoadTopologyTransitionsParquet:
    """Parquet-source tests for load_topology_transitions (no PG required)."""

    def test_load_topology_transitions_parquet_returns_dataframe(self, tmp_path: Path) -> None:
        """Happy path: parquet-source returns a DataFrame with expected columns."""
        from atm.analysis.loaders import load_topology_transitions

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_topology_transition_row(str(run_id))

        asyncio.get_event_loop().run_until_complete(
            _write_transitions_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = asyncio.get_event_loop().run_until_complete(
            load_topology_transitions(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_load_topology_transitions_parquet_signals_snapshot_is_dict(
        self, tmp_path: Path
    ) -> None:
        """signals_snapshot column must be a Python dict, not a JSON string."""
        from atm.analysis.loaders import load_topology_transitions

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_topology_transition_row(str(run_id))

        asyncio.get_event_loop().run_until_complete(
            _write_transitions_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = asyncio.get_event_loop().run_until_complete(
            load_topology_transitions(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        val = df["signals_snapshot"].iloc[0]
        assert isinstance(val, dict), f"Expected dict, got {type(val)}: {val!r}"
        assert val["quality"] == pytest.approx(0.4)

    def test_load_topology_transitions_parquet_considered_alternatives_is_list(
        self, tmp_path: Path
    ) -> None:
        """considered_alternatives column must be a Python list, not a JSON string."""
        from atm.analysis.loaders import load_topology_transitions

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_topology_transition_row(str(run_id))

        asyncio.get_event_loop().run_until_complete(
            _write_transitions_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = asyncio.get_event_loop().run_until_complete(
            load_topology_transitions(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        val = df["considered_alternatives"].iloc[0]
        assert isinstance(val, list), f"Expected list, got {type(val)}: {val!r}"
        assert "chain" in val

    def test_load_topology_transitions_parquet_guards_applied_is_list(
        self, tmp_path: Path
    ) -> None:
        """guards_applied column must be a Python list, not a JSON string."""
        from atm.analysis.loaders import load_topology_transitions

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_topology_transition_row(str(run_id))

        asyncio.get_event_loop().run_until_complete(
            _write_transitions_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = asyncio.get_event_loop().run_until_complete(
            load_topology_transitions(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        val = df["guards_applied"].iloc[0]
        assert isinstance(val, list), f"Expected list, got {type(val)}: {val!r}"
        assert "min_agents_guard" in val

    def test_load_topology_transitions_parquet_multiple_runs_concatenated(
        self, tmp_path: Path
    ) -> None:
        """Transitions from multiple runs are concatenated into one DataFrame."""
        from atm.analysis.loaders import load_topology_transitions

        exp_id = uuid.uuid4()
        run_id_a = uuid.uuid4()
        run_id_b = uuid.uuid4()

        asyncio.get_event_loop().run_until_complete(
            _write_transitions_via_writer(
                tmp_path, exp_id, run_id_a, [_make_topology_transition_row(str(run_id_a))]
            )
        )
        asyncio.get_event_loop().run_until_complete(
            _write_transitions_via_writer(
                tmp_path,
                exp_id,
                run_id_b,
                [
                    _make_topology_transition_row(str(run_id_b)),
                    _make_topology_transition_row(str(run_id_b)),
                ],
            )
        )

        df = asyncio.get_event_loop().run_until_complete(
            load_topology_transitions(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        assert len(df) == 3

    def test_load_topology_transitions_parquet_empty_returns_empty_dataframe(
        self, tmp_path: Path
    ) -> None:
        """Empty DataFrame returned when no parquet files exist for the experiment."""
        from atm.analysis.loaders import load_topology_transitions

        exp_id = uuid.uuid4()

        df = asyncio.get_event_loop().run_until_complete(
            load_topology_transitions(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ===========================================================================
# load_topology_transitions — PG source
# ===========================================================================


class TestLoadTopologyTransitionsPG:
    """PG-source tests for load_topology_transitions (gated on ATM_ENABLE_PG_TESTS=1)."""

    @pytest.mark.requires_postgres
    async def test_load_topology_transitions_pg_returns_dataframe(
        self, ephemeral_pg_dsn: str
    ) -> None:
        """Happy path: PG-source returns a DataFrame with expected columns."""
        import decimal

        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_topology_transitions
        from atm.storage.models import TopologyTransition

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-transitions-pg")
        run_id = await _insert_run(factory, exp_id=exp_id)

        # Insert a topology transition row
        async with session_scope(factory) as session:
            session.add(
                TopologyTransition(
                    run_id=run_id,
                    from_topology="chain",
                    to_topology="mesh",
                    phase_at_decision="execution",
                    iter_within_phase=1,
                    iter_within_topology=2,
                    decided_by="llm_router",
                    reason="quality low",
                    considered_alternatives=["chain", "star"],
                    guards_applied=["guard1"],
                    signals_snapshot={"quality": 0.4},
                    router_cost_usd=decimal.Decimal("0.0002"),
                )
            )

        df = await load_topology_transitions(str(exp_id), source="pg", session_factory=factory)

        await engine.dispose()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @pytest.mark.requires_postgres
    async def test_load_topology_transitions_pg_signals_snapshot_is_dict(
        self, ephemeral_pg_dsn: str
    ) -> None:
        """PG-source: signals_snapshot must be a Python dict (JSONB returns native type)."""
        import decimal

        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_topology_transitions
        from atm.storage.models import TopologyTransition

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-transitions-pg-dict")
        run_id = await _insert_run(factory, exp_id=exp_id, seed=99)

        async with session_scope(factory) as session:
            session.add(
                TopologyTransition(
                    run_id=run_id,
                    from_topology=None,
                    to_topology="chain",
                    phase_at_decision="planning",
                    iter_within_phase=0,
                    iter_within_topology=0,
                    decided_by="initial",
                    reason="initial topology",
                    considered_alternatives=[],
                    guards_applied=[],
                    signals_snapshot={"cost": 0.0, "quality": 1.0},
                    router_cost_usd=decimal.Decimal("0"),
                )
            )

        df = await load_topology_transitions(str(exp_id), source="pg", session_factory=factory)

        await engine.dispose()

        val = df["signals_snapshot"].iloc[0]
        assert isinstance(val, dict), f"Expected dict, got {type(val)}: {val!r}"

    @pytest.mark.requires_postgres
    async def test_load_topology_transitions_pg_and_parquet_same_column_set(
        self, ephemeral_pg_dsn: str, tmp_path: Path
    ) -> None:
        """PG and parquet sources produce DataFrames with the same column names."""
        import decimal

        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_topology_transitions
        from atm.storage.models import TopologyTransition

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-colset-check")
        run_id = await _insert_run(factory, exp_id=exp_id, seed=7)

        async with session_scope(factory) as session:
            session.add(
                TopologyTransition(
                    run_id=run_id,
                    from_topology="chain",
                    to_topology="mesh",
                    phase_at_decision="execution",
                    iter_within_phase=1,
                    iter_within_topology=1,
                    decided_by="rule",
                    reason="rule triggered",
                    considered_alternatives=["mesh"],
                    guards_applied=[],
                    signals_snapshot={"k": "v"},
                    router_cost_usd=decimal.Decimal("0"),
                )
            )

        df_pg = await load_topology_transitions(str(exp_id), source="pg", session_factory=factory)

        # Write same data to parquet (use await since we're in an async test)
        parquet_row = _make_topology_transition_row(str(run_id))
        await _write_transitions_via_writer(tmp_path, exp_id, run_id, [parquet_row])
        df_pq = await load_topology_transitions(str(exp_id), source="parquet", parquet_dir=tmp_path)

        await engine.dispose()

        assert sorted(df_pg.columns.tolist()) == sorted(df_pq.columns.tolist()), (
            f"Column mismatch: PG={sorted(df_pg.columns.tolist())!r}, "
            f"Parquet={sorted(df_pq.columns.tolist())!r}"
        )


# ===========================================================================
# load_phases — parquet source
# ===========================================================================


class TestLoadPhasesParquet:
    """Parquet-source tests for load_phases (no PG required)."""

    def test_load_phases_parquet_returns_dataframe(self, tmp_path: Path) -> None:
        """Happy path: parquet-source returns a DataFrame."""
        from atm.analysis.loaders import load_phases

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_phase_row(str(run_id))

        asyncio.get_event_loop().run_until_complete(
            _write_phases_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = asyncio.get_event_loop().run_until_complete(
            load_phases(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_load_phases_parquet_columns_present(self, tmp_path: Path) -> None:
        """Parquet-source DataFrame includes required phase columns."""
        from atm.analysis.loaders import load_phases

        exp_id = uuid.uuid4()
        run_id = uuid.uuid4()
        row = _make_phase_row(str(run_id))

        asyncio.get_event_loop().run_until_complete(
            _write_phases_via_writer(tmp_path, exp_id, run_id, [row])
        )

        df = asyncio.get_event_loop().run_until_complete(
            load_phases(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        required_cols = {"run_id", "phase_name", "from_phase", "started_at", "ended_at",
                         "entry_reason", "topology_used", "decided_by"}
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_load_phases_parquet_multiple_runs_concatenated(self, tmp_path: Path) -> None:
        """Phases from multiple runs are concatenated."""
        from atm.analysis.loaders import load_phases

        exp_id = uuid.uuid4()
        run_id_a = uuid.uuid4()
        run_id_b = uuid.uuid4()

        asyncio.get_event_loop().run_until_complete(
            _write_phases_via_writer(tmp_path, exp_id, run_id_a, [_make_phase_row(str(run_id_a))])
        )
        asyncio.get_event_loop().run_until_complete(
            _write_phases_via_writer(
                tmp_path,
                exp_id,
                run_id_b,
                [_make_phase_row(str(run_id_b)), _make_phase_row(str(run_id_b))],
            )
        )

        df = asyncio.get_event_loop().run_until_complete(
            load_phases(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        assert len(df) == 3

    def test_load_phases_parquet_empty_returns_empty_dataframe(self, tmp_path: Path) -> None:
        """Empty DataFrame returned when no phase parquet files exist."""
        from atm.analysis.loaders import load_phases

        exp_id = uuid.uuid4()

        df = asyncio.get_event_loop().run_until_complete(
            load_phases(str(exp_id), source="parquet", parquet_dir=tmp_path)
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ===========================================================================
# load_phases — PG source
# ===========================================================================


class TestLoadPhasesPG:
    """PG-source tests for load_phases (gated on ATM_ENABLE_PG_TESTS=1)."""

    @pytest.mark.requires_postgres
    async def test_load_phases_pg_returns_dataframe(self, ephemeral_pg_dsn: str) -> None:
        """Happy path: PG-source returns a DataFrame with expected rows."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_phases
        from atm.storage.models import Phase

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-phases-pg")
        run_id = await _insert_run(factory, exp_id=exp_id)

        async with session_scope(factory) as session:
            session.add(
                Phase(
                    run_id=run_id,
                    phase_name="execution",
                    from_phase="planning",
                    entry_reason="planning done",
                    topology_used="chain",
                    decided_by="rule",
                )
            )

        df = await load_phases(str(exp_id), source="pg", session_factory=factory)

        await engine.dispose()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @pytest.mark.requires_postgres
    async def test_load_phases_pg_and_parquet_same_column_set(
        self, ephemeral_pg_dsn: str, tmp_path: Path
    ) -> None:
        """PG and parquet sources produce identical column sets for phases."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_phases
        from atm.storage.models import Phase

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-phases-colset")
        run_id = await _insert_run(factory, exp_id=exp_id, seed=11)

        async with session_scope(factory) as session:
            session.add(
                Phase(
                    run_id=run_id,
                    phase_name="planning",
                    from_phase=None,
                    entry_reason="initial",
                    topology_used="chain",
                    decided_by="rule",
                )
            )

        df_pg = await load_phases(str(exp_id), source="pg", session_factory=factory)

        # Use await since we're in an async test
        parquet_row = _make_phase_row(str(run_id))
        await _write_phases_via_writer(tmp_path, exp_id, run_id, [parquet_row])
        df_pq = await load_phases(str(exp_id), source="parquet", parquet_dir=tmp_path)

        await engine.dispose()

        assert sorted(df_pg.columns.tolist()) == sorted(df_pq.columns.tolist()), (
            f"PG cols={sorted(df_pg.columns.tolist())!r}, "
            f"Parquet cols={sorted(df_pq.columns.tolist())!r}"
        )


# ===========================================================================
# load_human_interactions — PG only
# ===========================================================================


class TestLoadHumanInteractions:
    """PG-only tests for load_human_interactions (gated on ATM_ENABLE_PG_TESTS=1)."""

    @pytest.mark.requires_postgres
    async def test_load_human_interactions_returns_dataframe(self, ephemeral_pg_dsn: str) -> None:
        """Happy path: returns a DataFrame with expected columns."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_human_interactions
        from atm.storage.models import HumanInteraction

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-hitl")
        run_id = await _insert_run(factory, exp_id=exp_id)

        async with session_scope(factory) as session:
            session.add(
                HumanInteraction(
                    run_id=run_id,
                    role="reviewer",
                    context_json={"task": "code review"},
                    response_json={"answer": "lgtm"},
                    tlx_scores={"mental_demand": 5, "effort": 4},
                    raw_tlx_score=4.5,
                    request_id="req-001",
                )
            )

        df = await load_human_interactions(str(exp_id), session_factory=factory)

        await engine.dispose()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @pytest.mark.requires_postgres
    async def test_load_human_interactions_required_columns(self, ephemeral_pg_dsn: str) -> None:
        """DataFrame must contain all required columns from the plan spec."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_human_interactions
        from atm.storage.models import HumanInteraction

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-hitl-cols")
        run_id = await _insert_run(factory, exp_id=exp_id, seed=5)

        async with session_scope(factory) as session:
            session.add(
                HumanInteraction(
                    run_id=run_id,
                    role="reviewer",
                    context_json={},
                    request_id="req-002",
                )
            )

        df = await load_human_interactions(str(exp_id), session_factory=factory)

        await engine.dispose()

        # Columns from plan spec: id, run_id, role, requested_at, answered_at,
        # raw_tlx_score, tlx_scores, request_id
        required = {"id", "run_id", "role", "requested_at", "answered_at",
                    "raw_tlx_score", "tlx_scores", "request_id"}
        missing = required - set(df.columns)
        assert not missing, f"Missing required columns: {missing}"

    @pytest.mark.requires_postgres
    async def test_load_human_interactions_raw_tlx_score_nan_when_null(
        self, ephemeral_pg_dsn: str
    ) -> None:
        """raw_tlx_score is NaN (float) when the DB value is NULL."""
        import math

        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_human_interactions
        from atm.storage.models import HumanInteraction

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-hitl-nan")
        run_id = await _insert_run(factory, exp_id=exp_id, seed=8)

        async with session_scope(factory) as session:
            session.add(
                HumanInteraction(
                    run_id=run_id,
                    role="operator",
                    context_json={},
                    raw_tlx_score=None,  # NULL in DB
                    request_id="req-003",
                )
            )

        df = await load_human_interactions(str(exp_id), session_factory=factory)

        await engine.dispose()

        val = df["raw_tlx_score"].iloc[0]
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert math.isnan(val), f"Expected NaN, got {val}"

    @pytest.mark.requires_postgres
    async def test_load_human_interactions_raw_tlx_score_float_when_set(
        self, ephemeral_pg_dsn: str
    ) -> None:
        """raw_tlx_score is a Python float when the DB value is non-NULL."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_human_interactions
        from atm.storage.models import HumanInteraction

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-hitl-float")
        run_id = await _insert_run(factory, exp_id=exp_id, seed=9)

        async with session_scope(factory) as session:
            session.add(
                HumanInteraction(
                    run_id=run_id,
                    role="operator",
                    context_json={},
                    raw_tlx_score=3.75,
                    request_id="req-004",
                )
            )

        df = await load_human_interactions(str(exp_id), session_factory=factory)

        await engine.dispose()

        val = df["raw_tlx_score"].iloc[0]
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert abs(val - 3.75) < 1e-9

    @pytest.mark.requires_postgres
    async def test_load_human_interactions_empty_experiment_returns_empty_dataframe(
        self, ephemeral_pg_dsn: str
    ) -> None:
        """Empty DataFrame returned when no human interactions exist for the experiment."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from atm.analysis.loaders import load_human_interactions

        engine = create_async_engine(ephemeral_pg_dsn, echo=False)
        factory = create_session_factory(engine)

        exp_id = await _insert_experiment(factory, name="exp-hitl-empty")

        df = await load_human_interactions(str(exp_id), session_factory=factory)

        await engine.dispose()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
