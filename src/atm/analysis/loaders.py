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

    Args:
        run_id:      Run UUID string (used to locate the Parquet file).
        parquet_dir: Root directory containing per-run Parquet files.

    Returns:
        DataFrame with columns matching ``LLM_CALL_SCHEMA.names``.

    Raises:
        FileNotFoundError: If the Parquet file for ``run_id`` does not exist.
    """
    raise NotImplementedError("load_llm_calls — implemented in Step 4")


def load_llm_calls_for_experiment(
    exp_id: str,
    *,
    parquet_dir: Path,
) -> pd.DataFrame:
    """Concatenate LLM call Parquets for all runs in an experiment.

    Injects a ``run_id`` column from the filename when not already present.

    Args:
        exp_id:      Experiment UUID string (used to glob matching files).
        parquet_dir: Root directory containing per-run Parquet files.

    Returns:
        Concatenated DataFrame with one row per LLM call across all runs.
    """
    raise NotImplementedError("load_llm_calls_for_experiment — implemented in Step 4")


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

    Both sources return identical column sets. JSONB ``signals_snapshot``
    is decoded to a Python dict (not left as a JSON string).

    Args:
        exp_id:          Experiment UUID string.
        source:          Data source selector.
        session_factory: Async SQLAlchemy sessionmaker (required for ``"pg"``).
        parquet_dir:     Root Parquet directory (required for ``"parquet"``).

    Returns:
        DataFrame with one row per topology transition.
    """
    raise NotImplementedError("load_topology_transitions — implemented in Step 5")


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
    raise NotImplementedError("load_phases — implemented in Step 5")


async def load_human_interactions(
    exp_id: str,
    *,
    session_factory: Any,
) -> pd.DataFrame:
    """Load HumanInteraction rows for an experiment from Postgres.

    The ``raw_tlx_score`` column is cast to float; empty strings become NaN.

    Args:
        exp_id:          Experiment UUID string.
        session_factory: Async SQLAlchemy sessionmaker.

    Returns:
        DataFrame with one row per human interaction.
    """
    raise NotImplementedError("load_human_interactions — implemented in Step 5")
