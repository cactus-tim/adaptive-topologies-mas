"""Zombie-run reconciliation for ``atm grid`` and ops use (m12-resume-replay G5).

A "zombie" run is a row in the ``runs`` table whose ``status='running'`` but
whose owning OS process is no longer alive. This typically happens when a
worker is SIGKILLed (OOM, host reboot) and never gets a chance to write the
terminal status. Without reconciliation, ``atm grid`` would either skip the
slot ("already running") or double-launch — both wrong.

Public API
----------
ZombieRow         — per-row classification result
ReconcileReport   — aggregate report (scanned, zombies, actions)
reconcile_zombies — async function: scans + (optionally) marks rows as failed

Heuristic
---------
- If ``runs.host`` is None → cannot verify → ``no_pid`` zombie.
- If ``runs.host`` != ``current_host`` (default ``socket.gethostname()``) →
  cross-host PID checks are unreliable, treat as ``host_mismatch`` zombie.
  (Future M13 work could add a heartbeat table for cross-host verification.)
- If ``runs.host`` matches AND ``runs.process_pid`` is set → check with
  ``_pid_alive(pid)`` (uses ``psutil.pid_exists`` when available, falls back
  to ``os.kill(pid, 0)`` on POSIX). Dead pid → ``pid_dead`` zombie.
- If ``runs.process_pid`` is None → ``no_pid`` zombie.

Mutation policy
---------------
- ``allow_force_resume=False`` (default) → UPDATE each zombie's row to
  ``status='failed', finish_reason='zombie'``. Idempotent: if the row's
  status changed between the SELECT and the UPDATE (e.g. a sibling worker
  wrote 'completed'), the UPDATE's ``WHERE status='running'`` predicate
  filters it out — no-op.
- ``allow_force_resume=True`` → only classify and log; do not mutate. Used
  by ``atm grid --force-resume`` when the operator wants to ``atm resume``
  the surviving rows manually.
"""

from __future__ import annotations

import errno
import os
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import sqlalchemy as sa
import structlog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


ZombieReason = Literal["pid_dead", "host_mismatch", "no_pid"]
ActionTaken = Literal["marked_failed", "kept_force_resume"]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZombieRow:
    """One row that was classified as a zombie during reconcile."""

    run_id: UUID
    host: str | None
    pid: int | None
    reason: ZombieReason


@dataclass
class ReconcileReport:
    """Aggregate result of a reconcile pass over one experiment."""

    scanned: int = 0
    zombies: list[ZombieRow] = field(default_factory=list)
    actions: dict[UUID, ActionTaken] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PID liveness probe
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` corresponds to a live process on this host.

    Prefers :func:`psutil.pid_exists` (cross-platform and more accurate); falls
    back to ``os.kill(pid, 0)`` (POSIX) when ``psutil`` is not installed.

    ``os.kill(pid, 0)``:
        - returns silently if the pid exists and the caller has permission to
          send it a signal (including the no-op signal 0);
        - raises ``ProcessLookupError`` (ESRCH) if no such pid;
        - raises ``PermissionError`` (EPERM) if the pid exists but is owned
          by another user — for reconcile purposes this still counts as
          *alive* (process is running, just not ours to signal).
    """
    if pid <= 0:
        return False

    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        try:
            return bool(psutil.pid_exists(pid))
        except Exception:
            # Fall through to os.kill probe on unexpected psutil failure.
            pass

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM means the pid exists but is owned by another uid — still alive.
        return True
    except OSError as exc:  # pragma: no cover — defensive
        if getattr(exc, "errno", None) == errno.ESRCH:
            return False
        return True
    return True


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify_row(
    *,
    run_id: UUID,
    host: str | None,
    pid: int | None,
    current_host: str,
) -> ZombieRow | None:
    """Return a ZombieRow if the row should be classified as a zombie, else None."""
    if host is None:
        return ZombieRow(run_id=run_id, host=host, pid=pid, reason="no_pid")
    if host != current_host:
        return ZombieRow(run_id=run_id, host=host, pid=pid, reason="host_mismatch")
    if pid is None:
        return ZombieRow(run_id=run_id, host=host, pid=pid, reason="no_pid")
    if not _pid_alive(pid):
        return ZombieRow(run_id=run_id, host=host, pid=pid, reason="pid_dead")
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def reconcile_zombies(
    session_factory: async_sessionmaker[AsyncSession],
    exp_id: UUID,
    *,
    allow_force_resume: bool = False,
    current_host: str | None = None,
) -> ReconcileReport:
    """Scan ``runs`` for ``exp_id`` and mark zombies as failed.

    Parameters
    ----------
    session_factory:
        Async SQLAlchemy session factory.
    exp_id:
        Experiment UUID whose runs should be scanned.
    allow_force_resume:
        When True, classify zombies but do not mutate the DB — only log.
        Returned report still lists them so callers can act.
    current_host:
        Override for the host comparison (default ``socket.gethostname()``).
        Tests pass an explicit value to bypass the live hostname.

    Returns
    -------
    ReconcileReport
        Contains ``scanned``, list of ``ZombieRow``s, and a per-run-id
        mapping of the action that was applied.
    """
    host_name = current_host if current_host is not None else socket.gethostname()
    report = ReconcileReport()

    # Lazy import inside the function to avoid a top-level cycle with
    # storage.models (which itself imports SQLAlchemy and would be loaded
    # even when reconcile is never called).
    from atm.storage.models import Run
    from atm.storage.session import session_scope

    # 1) SELECT phase — fetch candidate rows. Use a parameter-bound textual
    # SELECT against ORM columns: we don't need ORM identity tracking and
    # this keeps the query path predictable. Result columns are positional.
    select_stmt = (
        sa.select(Run.id, Run.host, Run.process_pid)
        .where(Run.exp_id == exp_id)
        .where(Run.status == "running")
    )

    async with session_scope(session_factory) as session:
        result = await session.execute(select_stmt)
        rows = result.all()

    report.scanned = len(rows)

    # 2) Classify in-memory (no DB calls during the pid_alive probe).
    candidates: list[ZombieRow] = []
    for run_id, host, pid in rows:
        zombie = _classify_row(
            run_id=run_id, host=host, pid=pid, current_host=host_name
        )
        if zombie is not None:
            candidates.append(zombie)
    report.zombies = candidates

    # 3) Mutate phase — single UPDATE per zombie, gated on status='running'
    # so a concurrent winner is not clobbered. Skip when force_resume.
    for zombie in candidates:
        if allow_force_resume:
            report.actions[zombie.run_id] = "kept_force_resume"
            logger.info(
                "reconcile: zombie kept (force_resume)",
                run_id=str(zombie.run_id),
                host=zombie.host,
                pid=zombie.pid,
                reason=zombie.reason,
            )
            continue

        update_stmt = (
            sa.update(Run)
            .where(Run.id == zombie.run_id)
            .where(Run.status == "running")
            .values(status="failed", finish_reason="zombie")
        )
        async with session_scope(session_factory) as session:
            await session.execute(update_stmt)

        report.actions[zombie.run_id] = "marked_failed"
        logger.info(
            "reconcile: zombie marked failed",
            run_id=str(zombie.run_id),
            host=zombie.host,
            pid=zombie.pid,
            reason=zombie.reason,
        )

    logger.info(
        "reconcile complete",
        exp_id=str(exp_id),
        scanned=report.scanned,
        zombies=len(report.zombies),
        force_resume=allow_force_resume,
    )
    return report
