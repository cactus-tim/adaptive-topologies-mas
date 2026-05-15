"""Data loaders — experiment artifacts from Postgres and Parquet into pandas DataFrames.

Public surface (to be implemented in Steps 3-5):
    load_experiment            — load a single experiment row as a dict
    load_runs                  — load all Run rows for an experiment as a DataFrame
    load_llm_calls             — load LLM call Parquet for a single run
    load_llm_calls_for_experiment — concat LLM call Parquets for all runs in an experiment
    load_topology_transitions  — load TopologyTransition rows (PG or Parquet)
    load_phases                — load Phase rows (PG or Parquet)
    load_human_interactions    — load HumanInteraction rows (PG only)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from atm.analysis.oracle import _infer_task_type


async def load_experiment(
    exp_id: str,
    *,
    session_factory: Any,
) -> dict[str, Any]:
    """Load a single experiment row from Postgres as a plain dict.

    Args:
        exp_id:          Experiment UUID string.
        session_factory: Async SQLAlchemy sessionmaker.

    Returns:
        Dict with experiment column values.

    Raises:
        KeyError: If no experiment with the given exp_id exists.
    """
    from sqlalchemy import select

    from atm.storage.models import Experiment

    exp_uuid = uuid.UUID(exp_id) if not isinstance(exp_id, uuid.UUID) else exp_id

    async with session_factory() as session:
        result = await session.execute(select(Experiment).where(Experiment.id == exp_uuid))
        row = result.scalar_one_or_none()

    if row is None:
        raise KeyError(f"No experiment found with id={exp_id!r}")

    return {
        "id": row.id,
        "name": row.name,
        "config_snapshot": row.config_snapshot,
        "git_sha": row.git_sha,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "total_cost_usd": float(row.total_cost_usd) if row.total_cost_usd is not None else None,
        "status": row.status,
    }


async def load_runs(
    exp_id: str,
    *,
    session_factory: Any,
) -> pd.DataFrame:
    """Load all Run rows for an experiment from Postgres.

    The returned DataFrame includes a derived ``task_type`` column inferred
    from ``task_id`` via ``_infer_task_type``.

    Args:
        exp_id:          Experiment UUID string.
        session_factory: Async SQLAlchemy sessionmaker.

    Returns:
        DataFrame with one row per run.
    """
    from sqlalchemy import select

    from atm.storage.models import Run

    exp_uuid = uuid.UUID(exp_id) if not isinstance(exp_id, uuid.UUID) else exp_id

    async with session_factory() as session:
        result = await session.execute(select(Run).where(Run.exp_id == exp_uuid))
        run_objects: list[Run] = list(result.scalars().all())

    if not run_objects:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for run in run_objects:
        rows.append(
            {
                "id": str(run.id),
                "exp_id": str(run.exp_id),
                "topology": run.topology,
                "task_id": run.task_id,
                "task_type": _infer_task_type(run.task_id),
                "agent_set": run.agent_set,
                "human_role": run.human_role,
                "seed": run.seed,
                "model": run.model,
                "models_by_role_json": run.models_by_role_json,
                "model_version_snapshot": run.model_version_snapshot,
                "sandbox_image_digest": run.sandbox_image_digest,
                "status": run.status,
                "finish_reason": run.finish_reason,
                "budget_spent_usd": float(run.budget_spent_usd)
                if run.budget_spent_usd is not None
                else float("nan"),
                "quality_score": float(run.quality_score)
                if run.quality_score is not None
                else float("nan"),
                "cognitive_load_proxy": float(run.cognitive_load_proxy)
                if run.cognitive_load_proxy is not None
                else float("nan"),
                "wall_time_s": run.wall_time_s,
                "iterations": run.iterations,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "error": run.error,
                "replay_of": str(run.replay_of) if run.replay_of is not None else None,
                "host": run.host,
                "process_pid": run.process_pid,
            }
        )

    return pd.DataFrame(rows)


def load_llm_calls(
    run_id: str,
    *,
    parquet_dir: Path,
) -> pd.DataFrame:
    """Load the LLM call Parquet for a single run.

    The file is located by globbing for the run_id under all experiment
    subdirectories:
    ``parquet_dir/experiments/*/runs/{run_id}/llm_calls.parquet``

    Args:
        run_id:      Run UUID string (used to locate the Parquet file).
        parquet_dir: Root directory containing per-run Parquet files.

    Returns:
        DataFrame with columns matching ``LLM_CALL_SCHEMA.names``.

    Raises:
        FileNotFoundError: If the Parquet file for ``run_id`` does not exist.
    """
    import pyarrow.parquet as pq

    pattern = f"experiments/*/runs/{run_id}/llm_calls.parquet"
    matches = list(parquet_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No llm_calls.parquet found for run_id={run_id!r} "
            f"under {parquet_dir!r} (pattern: {pattern!r})"
        )

    table = pq.read_table(matches[0])  # type: ignore[no-untyped-call]
    result: pd.DataFrame = table.to_pandas()
    return result


def load_llm_calls_for_experiment(
    exp_id: str,
    *,
    parquet_dir: Path,
) -> pd.DataFrame:
    """Concatenate LLM call Parquets for all runs in an experiment.

    Globs all run directories under the experiment and concatenates their
    llm_calls.parquet files. The ``run_id`` column is injected from the
    directory name when not already present in the parquet data.

    File layout:
        ``parquet_dir/experiments/{exp_id}/runs/{run_id}/llm_calls.parquet``

    Args:
        exp_id:      Experiment UUID string (used to glob matching files).
        parquet_dir: Root directory containing per-run Parquet files.

    Returns:
        Concatenated DataFrame with one row per LLM call across all runs.
        Returns an empty DataFrame when no runs exist for the experiment.
    """
    import pyarrow.parquet as pq

    exp_runs_dir = parquet_dir / "experiments" / exp_id / "runs"
    pattern = "*/llm_calls.parquet"
    parquet_files = list(exp_runs_dir.glob(pattern)) if exp_runs_dir.exists() else []

    if not parquet_files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for parquet_file in parquet_files:
        # The run_id is the name of the parent directory (the run UUID)
        run_id_from_path = parquet_file.parent.name
        table = pq.read_table(parquet_file)  # type: ignore[no-untyped-call]
        df: pd.DataFrame = table.to_pandas()
        # Ensure run_id column is present and reflects the directory-derived run_id
        df["run_id"] = run_id_from_path
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


async def load_topology_transitions(
    exp_id: str,
    *,
    source: Literal["pg", "parquet"] = "pg",
    session_factory: Any = None,
    parquet_dir: Path | None = None,
) -> pd.DataFrame:
    """Load TopologyTransition rows for an experiment.

    Supports two sources:
    - ``"pg"``      : reads from Postgres (requires ``session_factory``).
    - ``"parquet"`` : reads from Parquet files (requires ``parquet_dir``).

    Both sources return identical column sets. JSONB/JSON-string fields
    (``signals_snapshot``, ``considered_alternatives``, ``guards_applied``)
    are decoded to native Python types (dict/list), never left as JSON strings.

    Args:
        exp_id:          Experiment UUID string.
        source:          Data source selector.
        session_factory: Async SQLAlchemy sessionmaker (required for ``"pg"``).
        parquet_dir:     Root Parquet directory (required for ``"parquet"``).

    Returns:
        DataFrame with one row per topology transition.
    """
    import json

    if source == "parquet":
        if parquet_dir is None:
            raise ValueError("parquet_dir is required when source='parquet'")
        import pyarrow.parquet as pq

        exp_runs_dir = parquet_dir / "experiments" / exp_id / "runs"
        parquet_files = (
            list(exp_runs_dir.glob("*/topology_transitions.parquet"))
            if exp_runs_dir.exists()
            else []
        )

        if not parquet_files:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for pq_file in parquet_files:
            table = pq.read_table(pq_file)  # type: ignore[no-untyped-call]
            df: pd.DataFrame = table.to_pandas()
            # Decode JSON-string columns into native Python types
            df["signals_snapshot"] = df["signals_snapshot_json"].apply(json.loads)
            df["considered_alternatives"] = df["considered_alternatives_json"].apply(json.loads)
            df["guards_applied"] = df["guards_applied_json"].apply(json.loads)
            # Drop the raw _json columns
            df = df.drop(
                columns=[
                    "signals_snapshot_json",
                    "considered_alternatives_json",
                    "guards_applied_json",
                ]
            )
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    else:  # source == "pg"
        if session_factory is None:
            raise ValueError("session_factory is required when source='pg'")

        from sqlalchemy import select

        from atm.storage.models import Run, TopologyTransition

        exp_uuid = uuid.UUID(exp_id) if not isinstance(exp_id, uuid.UUID) else exp_id

        async with session_factory() as session:
            # Join through runs to filter by experiment
            result = await session.execute(
                select(TopologyTransition)
                .join(Run, TopologyTransition.run_id == Run.id)
                .where(Run.exp_id == exp_uuid)
            )
            transitions = list(result.scalars().all())

        if not transitions:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for t in transitions:
            rows.append(
                {
                    "run_id": str(t.run_id),
                    "from_topology": t.from_topology,
                    "to_topology": t.to_topology,
                    "phase_at_decision": t.phase_at_decision,
                    "iter_within_phase": t.iter_within_phase,
                    "iter_within_topology": t.iter_within_topology,
                    "decided_by": t.decided_by,
                    "reason": t.reason,
                    # PG returns native Python types for ARRAY and JSONB
                    "considered_alternatives": list(t.considered_alternatives),
                    "guards_applied": list(t.guards_applied),
                    "signals_snapshot": dict(t.signals_snapshot),
                    "router_cost_usd": float(t.router_cost_usd)
                    if t.router_cost_usd is not None
                    else float("nan"),
                    "at": t.at,
                }
            )

        return pd.DataFrame(rows)


async def load_phases(
    exp_id: str,
    *,
    source: Literal["pg", "parquet"] = "pg",
    session_factory: Any = None,
    parquet_dir: Path | None = None,
) -> pd.DataFrame:
    """Load Phase rows for an experiment.

    Supports two sources (same contract as ``load_topology_transitions``).

    Args:
        exp_id:          Experiment UUID string.
        source:          Data source selector.
        session_factory: Async SQLAlchemy sessionmaker (required for ``"pg"``).
        parquet_dir:     Root Parquet directory (required for ``"parquet"``).

    Returns:
        DataFrame with one row per phase.
    """
    if source == "parquet":
        if parquet_dir is None:
            raise ValueError("parquet_dir is required when source='parquet'")
        import pyarrow.parquet as pq

        exp_runs_dir = parquet_dir / "experiments" / exp_id / "runs"
        parquet_files = list(exp_runs_dir.glob("*/phases.parquet")) if exp_runs_dir.exists() else []

        if not parquet_files:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for pq_file in parquet_files:
            table = pq.read_table(pq_file)  # type: ignore[no-untyped-call]
            df: pd.DataFrame = table.to_pandas()
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    else:  # source == "pg"
        if session_factory is None:
            raise ValueError("session_factory is required when source='pg'")

        from sqlalchemy import select

        from atm.storage.models import Phase, Run

        exp_uuid = uuid.UUID(exp_id) if not isinstance(exp_id, uuid.UUID) else exp_id

        async with session_factory() as session:
            result = await session.execute(
                select(Phase).join(Run, Phase.run_id == Run.id).where(Run.exp_id == exp_uuid)
            )
            phases = list(result.scalars().all())

        if not phases:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for p in phases:
            rows.append(
                {
                    "run_id": str(p.run_id),
                    "phase_name": p.phase_name,
                    "from_phase": p.from_phase,
                    "started_at": p.started_at,
                    "ended_at": p.ended_at,
                    "entry_reason": p.entry_reason,
                    "topology_used": p.topology_used,
                    "decided_by": p.decided_by,
                }
            )

        return pd.DataFrame(rows)


async def load_human_interactions(
    exp_id: str,
    *,
    session_factory: Any,
) -> pd.DataFrame:
    """Load HumanInteraction rows for an experiment from Postgres.

    The ``raw_tlx_score`` column is cast to float; NULL values become NaN.

    Args:
        exp_id:          Experiment UUID string.
        session_factory: Async SQLAlchemy sessionmaker.

    Returns:
        DataFrame with columns: id, run_id, role, requested_at, answered_at,
        raw_tlx_score, tlx_scores, request_id.
    """
    from sqlalchemy import select

    from atm.storage.models import HumanInteraction, Run

    exp_uuid = uuid.UUID(exp_id) if not isinstance(exp_id, uuid.UUID) else exp_id

    async with session_factory() as session:
        result = await session.execute(
            select(HumanInteraction)
            .join(Run, HumanInteraction.run_id == Run.id)
            .where(Run.exp_id == exp_uuid)
        )
        interactions = list(result.scalars().all())

    if not interactions:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for hi in interactions:
        raw_tlx = hi.raw_tlx_score
        # Cast to float; None (NULL in DB) becomes NaN
        raw_tlx_float = float("nan") if raw_tlx is None else float(raw_tlx)

        rows.append(
            {
                "id": str(hi.id),
                "run_id": str(hi.run_id),
                "role": hi.role,
                "requested_at": hi.requested_at,
                "answered_at": hi.answered_at,
                "raw_tlx_score": raw_tlx_float,
                "tlx_scores": hi.tlx_scores,
                "request_id": hi.request_id,
            }
        )

    return pd.DataFrame(rows)
