"""Export PG aggregate tables (experiments, runs) to disk artifacts.

Per-run streaming events (messages, llm_calls, budget_events, tool_calls,
tool_results, human_interactions, scratchpads, phases, topology_transitions)
are already written to ``data/experiments/experiments/{exp_id}/runs/{run_id}/``
by the writers in :mod:`atm.storage.parquet_writer` and friends.

What is NOT written there: the aggregate ``runs`` row (with
``quality_score``, ``budget_spent_usd``, ``wall_time_s``, ``status``, etc.)
and the ``experiments`` row. Those live only in PG. This module snapshots
them out so that:

  * cross-machine consolidation works without a PG dump/restore;
  * the experiment is self-contained on disk for paper artifacts;
  * a PG container loss does not erase quality_score / cost.

Public API:
  - ``export_experiment(exp_id, session_factory, root) -> Path`` —
    async; writes ``experiment.json`` (single dict) and ``_runs.parquet``
    (one row per Run) under ``root/experiments/{exp_id}/``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import select

from atm.storage.models import Experiment, Run

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    """Convert non-JSON-serializable values (Decimal, datetime, UUID) to JSON-safe types."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _experiment_to_dict(exp: Experiment) -> dict[str, Any]:
    return {
        "id": str(exp.id),
        "name": exp.name,
        "config_snapshot": _to_jsonable(exp.config_snapshot or {}),
        "git_sha": exp.git_sha,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
        "total_cost_usd": float(exp.total_cost_usd) if exp.total_cost_usd is not None else 0.0,
        "status": exp.status,
    }


def _run_to_row(run: Run) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "exp_id": str(run.exp_id),
        "topology": run.topology,
        "task_id": run.task_id,
        "agent_set": run.agent_set,
        "human_role": run.human_role,
        "seed": run.seed,
        "model": run.model,
        "models_by_role_json": json.dumps(_to_jsonable(run.models_by_role_json or {})),
        "model_version_snapshot": json.dumps(_to_jsonable(run.model_version_snapshot or {})),
        "sandbox_image_digest": run.sandbox_image_digest,
        "status": run.status,
        "finish_reason": run.finish_reason,
        "budget_spent_usd": float(run.budget_spent_usd)
        if run.budget_spent_usd is not None
        else 0.0,
        "quality_score": float(run.quality_score) if run.quality_score is not None else None,
        "wall_time_s": float(run.wall_time_s) if run.wall_time_s is not None else None,
        "iterations": int(run.iterations) if run.iterations is not None else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error": run.error,
        "cognitive_load_proxy": float(run.cognitive_load_proxy)
        if run.cognitive_load_proxy is not None
        else None,
        "replay_of": str(run.replay_of) if run.replay_of else None,
        "host": run.host,
        "process_pid": int(run.process_pid) if run.process_pid is not None else None,
    }


async def export_experiment(
    exp_id: str | UUID,
    *,
    session_factory: Any,
    root: Path | str = Path("data/experiments"),
) -> Path:
    """Export an experiment's aggregate PG rows to disk.

    Writes two files under ``root/experiments/{exp_id}/``:

      * ``experiment.json`` — single JSON object mirroring the experiments row
      * ``_runs.parquet``   — one row per Run (snappy compression)

    The directory is created if missing. Existing files are overwritten —
    safe to call on the same exp_id multiple times to refresh the snapshot.

    Args:
        exp_id:           Experiment UUID (str or UUID).
        session_factory:  Async SQLAlchemy sessionmaker / async_sessionmaker.
        root:             Root of the on-disk experiment tree
                          (default: data/experiments — matches parquet_writer).

    Returns:
        Path to the per-experiment directory (``root/experiments/{exp_id}``).
    """
    exp_uuid = exp_id if isinstance(exp_id, UUID) else UUID(str(exp_id))

    async with session_factory() as session:
        exp_result = await session.execute(select(Experiment).where(Experiment.id == exp_uuid))
        exp: Experiment | None = exp_result.scalar_one_or_none()
        if exp is None:
            raise ValueError(f"experiment not found: exp_id={exp_uuid}")

        runs_result = await session.execute(select(Run).where(Run.exp_id == exp_uuid))
        runs: list[Run] = list(runs_result.scalars().all())

    root_path = Path(root)
    exp_dir = root_path / "experiments" / str(exp_uuid)
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_json_path = exp_dir / "experiment.json"
    exp_json_path.write_text(json.dumps(_experiment_to_dict(exp), indent=2, sort_keys=True))

    runs_parquet_path = exp_dir / "_runs.parquet"
    rows = [_run_to_row(r) for r in runs]
    df = pd.DataFrame(rows)
    df.to_parquet(runs_parquet_path, compression="snappy", index=False)

    logger.info(
        "exported experiment %s: runs=%d, dir=%s",
        exp_uuid,
        len(runs),
        exp_dir,
    )
    return exp_dir
