"""M12 grid runner — parallel execution driver for ExperimentConfig sweeps.

Public API:
  - run_grid(configs, *, parallelism, fail_fast=False, progress_callback=None) -> GridResult
  - GridProgress (frozen dataclass): live progress snapshot
  - GridResult (frozen dataclass): terminal aggregate

Worker model:
  ProcessPoolExecutor(max_workers=parallelism); each cell runs in a child process
  via ``_run_cell_worker(cfg_dict)`` which re-validates the cfg dict and invokes
  ``asyncio.run(run_one(cfg))``. Child process records its host + pid in
  ``runs.host`` / ``runs.process_pid`` (via runner._register_worker_identity).

Status aggregation rules (``_aggregate_experiment_status``):
  - all "completed"                        → "completed"
  - all in {"failed", "budget_exceeded"}   → "failed"
  - any in-flight (non-terminal)           → "running"
  - mix of "completed" and failure-class   → "partial"

fail_fast semantics:
  When fail_fast=True, the first cell to report "failed" or "budget_exceeded"
  triggers ``executor.shutdown(wait=False, cancel_futures=True)``. PENDING
  futures are cancelled; RUNNING workers continue to completion (cannot be
  preempted from the parent process).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import CancelledError as FuturesCancelledError
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker

from atm.experiment.config import ExperimentConfig
from atm.storage.checkpointer import build_checkpointer
from atm.storage.export import export_experiment
from atm.storage.models import Experiment
from atm.storage.session import create_engine, create_session_factory, session_scope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridProgress:
    """Live snapshot of grid-run progress, emitted to ``progress_callback``.

    Fields:
        total:       Total number of cells in the grid.
        done:        Number of cells that reached a terminal status.
        failed:      Number of cells in {"failed", "budget_exceeded"} terminal state.
        in_progress: Number of cells currently running or pending.
        eta_s:       Estimated remaining wall-clock seconds, or None if no
                     samples are available (e.g. before the first cell finishes).
    """

    total: int
    done: int
    failed: int
    in_progress: int
    eta_s: float | None


@dataclass(frozen=True)
class GridResult:
    """Terminal aggregate of a grid run.

    Fields:
        exp_id:           UUID of the parent experiment row.
        total:            Number of cells the grid attempted (before cancellation).
        completed:        Cells with terminal status="completed".
        failed:           Cells with terminal status="failed".
        budget_exceeded:  Cells with terminal status="budget_exceeded".
        run_ids:          UUIDs of all runs that reached _insert_run (may be < total
                          when fail_fast cancels cells before they touch the DB).
    """

    exp_id: UUID
    total: int
    completed: int
    failed: int
    budget_exceeded: int
    run_ids: list[UUID]


_TERMINAL_SUCCESS: frozenset[str] = frozenset({"completed"})
_TERMINAL_FAILURE: frozenset[str] = frozenset({"failed", "budget_exceeded"})
_TERMINAL: frozenset[str] = _TERMINAL_SUCCESS | _TERMINAL_FAILURE


def _aggregate_experiment_status(statuses: list[str]) -> str:
    """Aggregate per-cell statuses into an experiment-level status.

    Rules:
      - empty input            → ValueError (never call with no cells).
      - any non-terminal cell  → "running" (still in flight).
      - all terminal_success   → "completed".
      - all terminal_failure   → "failed".
      - mix of success/failure → "partial".

    Unknown / non-terminal statuses (e.g. "running", "queued") are treated as
    in-flight, yielding "running".
    """
    if not statuses:
        raise ValueError("_aggregate_experiment_status: empty status list")

    if any(s not in _TERMINAL for s in statuses):
        return "running"

    has_success = any(s in _TERMINAL_SUCCESS for s in statuses)
    has_failure = any(s in _TERMINAL_FAILURE for s in statuses)

    if has_success and has_failure:
        return "partial"
    if has_success:
        return "completed"
    return "failed"


def _run_cell_worker(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Run a single grid cell in a child process.

    The cfg is passed as a plain dict (output of ``ExperimentConfig.model_dump(mode="python")``)
    to avoid Pydantic v2 pickle quirks. The worker re-validates it back into a
    full ``ExperimentConfig`` and invokes ``asyncio.run(run_one(cfg))``.

    Never raises — all exceptions are caught and returned as a dict with
    ``status="failed"`` so the parent ProcessPoolExecutor never sees an
    un-picklable BaseException through the queue.

    Returns a picklable dict:
        {
            "run_id":       str | None,
            "exp_id":       str | None,
            "status":       "completed" | "failed" | "budget_exceeded",
            "quality_score": float | None,
            "cost_usd":     float,
            "iters":        int,
            "final_answer": str,
            "error":        str | None,
        }
    """
    import traceback

    from atm.experiment.config import ExperimentConfig
    from atm.experiment.runner import run_one

    class _HttpxAcloseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "Event loop is closed" in msg:
                return False
            return not ("Task exception was never retrieved" in msg and "aclose" in msg)

    logging.getLogger("asyncio").addFilter(_HttpxAcloseFilter())

    try:
        cfg = ExperimentConfig.model_validate(cfg_dict)
    except Exception as exc:
        return {
            "run_id": None,
            "exp_id": None,
            "status": "failed",
            "quality_score": None,
            "cost_usd": 0.0,
            "iters": 0,
            "final_answer": "",
            "error": f"config validation failed: {exc}",
        }

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_one(cfg))
        except Exception:
            return {
                "run_id": None,
                "exp_id": None,
                "status": "failed",
                "quality_score": None,
                "cost_usd": 0.0,
                "iters": 0,
                "final_answer": "",
                "error": traceback.format_exc()[:4000],
            }
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if pending:
                loop.run_until_complete(
                    asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=5.0,
                    )
                )
        except TimeoutError:
            pass
        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        try:
            asyncio.set_event_loop(None)
        finally:
            loop.close()

    metrics = result.metrics or {}
    return {
        "run_id": str(result.run_id),
        "exp_id": str(result.exp_id),
        "status": result.status,
        "quality_score": metrics.get("quality_score"),
        "cost_usd": float(metrics.get("cost_usd", 0.0) or 0.0),
        "iters": int(metrics.get("iters", 0) or 0),
        "final_answer": result.final_answer,
        "error": None,
    }


async def _update_experiment_status(
    session_factory: async_sessionmaker[Any],
    exp_id: UUID,
    new_status: str,
) -> None:
    """UPDATE experiments.status. Never raises (DB failure -> warning)."""
    try:
        async with session_scope(session_factory) as session:
            await session.execute(
                sa.update(Experiment).where(Experiment.id == exp_id).values(status=new_status)
            )
    except Exception as exc:
        logger.warning(
            "experiments.status UPDATE failed: exp_id=%s status=%s err=%s",
            exp_id,
            new_status,
            str(exc)[:200],
        )


def _compute_eta_s(done: int, total: int, started_monotonic: float) -> float | None:
    """Linear-extrapolation ETA: (elapsed / done) * remaining. None if done==0."""
    if done <= 0:
        return None
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    remaining = max(0, total - done)
    if remaining == 0:
        return 0.0
    return (elapsed / done) * remaining


async def run_grid(
    configs: list[ExperimentConfig],
    *,
    parallelism: int,
    fail_fast: bool = False,
    progress_callback: Callable[[GridProgress], None] | None = None,
) -> GridResult:
    """Drive a list of ExperimentConfig cells through a ProcessPoolExecutor.

    Args:
        configs:           Non-empty list of fully-validated ExperimentConfig cells
                           (produced by ``load_grid_configs``).
        parallelism:       Max concurrent worker processes (>= 1).
        fail_fast:         If True, on first failed/budget_exceeded cell trigger
                           ``executor.shutdown(wait=False, cancel_futures=True)``.
        progress_callback: Optional callback invoked after every cell completes
                           with a ``GridProgress`` snapshot.

    Returns:
        GridResult — terminal aggregate.

    Raises:
        ValueError: If ``configs`` is empty or ``parallelism`` < 1.
    """
    if not configs:
        raise ValueError("run_grid: configs must be non-empty")
    if parallelism < 1:
        raise ValueError(f"run_grid: parallelism must be >= 1, got {parallelism}")

    total = len(configs)

    pg_dsn = configs[0].observability.pg_dsn
    exp_name = configs[0].name

    cfg_dicts = [cfg.model_dump(mode="python") for cfg in configs]

    counters = {
        "completed": 0,
        "failed": 0,
        "budget_exceeded": 0,
    }
    statuses: list[str] = []
    run_ids: list[UUID] = []

    started_monotonic = time.monotonic()
    loop = asyncio.get_running_loop()

    try:
        _warm_saver, _warm_pool = await build_checkpointer(pg_dsn, max_size=1, min_size=1)
        await _warm_pool.close()
    except Exception as exc:
        logger.debug("checkpointer warmup skipped (%s); workers will retry", exc)

    executor = ProcessPoolExecutor(
        max_workers=parallelism,
        mp_context=mp.get_context("spawn"),
    )
    cancelled = False

    try:
        futures = [
            loop.run_in_executor(executor, _run_cell_worker, cfg_dict) for cfg_dict in cfg_dicts
        ]

        for fut in asyncio.as_completed(futures):
            try:
                result = await fut
            except (asyncio.CancelledError, FuturesCancelledError):
                continue
            except Exception as exc:
                logger.error("worker raised through pool: %s", exc)
                counters["failed"] += 1
                statuses.append("failed")
                continue

            status = str(result.get("status") or "failed")
            statuses.append(status)
            if status in counters:
                counters[status] += 1
            else:
                counters["failed"] += 1

            rid_str = result.get("run_id")
            if rid_str:
                with contextlib.suppress(Exception):
                    run_ids.append(UUID(rid_str))

            done = sum(counters.values())
            failed_total = counters["failed"] + counters["budget_exceeded"]
            in_progress = total - done
            if progress_callback is not None:
                try:
                    progress_callback(
                        GridProgress(
                            total=total,
                            done=done,
                            failed=failed_total,
                            in_progress=in_progress,
                            eta_s=_compute_eta_s(done, total, started_monotonic),
                        )
                    )
                except Exception:
                    logger.exception("progress_callback raised; ignoring")

            if fail_fast and status in _TERMINAL_FAILURE and not cancelled:
                cancelled = True
                logger.warning(
                    "fail_fast: cell %s status=%s — cancelling pending futures",
                    rid_str,
                    status,
                )
                for f in futures:
                    done_fn = getattr(f, "done", None)
                    cancel_fn = getattr(f, "cancel", None)
                    if done_fn is None or cancel_fn is None:
                        continue
                    try:
                        if not done_fn():
                            cancel_fn()
                    except Exception:
                        pass
                executor.shutdown(wait=False, cancel_futures=True)
    finally:
        executor.shutdown(wait=True)

    exp_id = await _resolve_exp_id(pg_dsn, exp_name)

    if statuses and exp_id is not None:
        agg = _aggregate_experiment_status(statuses)
        if agg in _TERMINAL or agg == "partial":
            engine = create_engine(pg_dsn)
            try:
                sf = create_session_factory(engine)
                await _update_experiment_status(sf, exp_id, agg)

                try:
                    parquet_root = (
                        configs[0].observability.parquet_dir if configs else "data/experiments"
                    )
                    await export_experiment(exp_id, session_factory=sf, root=parquet_root)
                except Exception:
                    logger.debug(
                        "auto-export_experiment failed; aggregate not snapshotted",
                        exc_info=True,
                    )
            finally:
                await engine.dispose()

    return GridResult(
        exp_id=exp_id or UUID("00000000-0000-0000-0000-000000000000"),
        total=total,
        completed=counters["completed"],
        failed=counters["failed"],
        budget_exceeded=counters["budget_exceeded"],
        run_ids=run_ids,
    )


async def _resolve_exp_id(pg_dsn: str, exp_name: str) -> UUID | None:
    """Look up experiments.id by name. Returns None if not found or DB unreachable."""
    engine = create_engine(pg_dsn)
    try:
        sf = create_session_factory(engine)
        try:
            async with session_scope(sf) as session:
                row = await session.execute(
                    sa.select(Experiment.id).where(Experiment.name == exp_name)
                )
                scalar: UUID | None = row.scalar_one_or_none()
                return scalar
        except Exception as exc:
            logger.warning("resolve_exp_id failed: name=%s err=%s", exp_name, str(exc)[:200])
            return None
    finally:
        await engine.dispose()
