"""Aggregator — compute quality score and persist it to the database.

Public API:
  - ``compute_quality(spec, answer, *, sandbox, judge_llm, run_seed)``
      → ``tuple[float | None, dict[str, Any]]``
      Never raises; returns (None, {"error": ...}) on any failure.

  - ``persist_quality(session, run_id, quality_score)``
      → None
      Idempotent UPDATE of runs.quality_score; no-ops if run_id not found.

  - ``aggregate_run(session, run_id, spec, answer, *, sandbox, judge_llm, run_seed)``
      → ``float | None``
      Orchestrator: calls compute_quality then persist_quality.

Architecture notes:
  - Judge LLM calls share the run's BudgetTracker (passed in as ``judge_llm``).
    BudgetTracker lives until run_one() returns; judge calls happen before
    that, so budget attribution is correct (context.md Decision note).
  - ``compute_quality`` delegates to ``score_ground_truth`` from ground_truth.py,
    which dispatches to the appropriate M10 evaluator based on spec.evaluator_key.
  - Any exception in ``compute_quality`` is caught; (None, {"error": ...}) is
    returned so that run finalisation never breaks on evaluation failure.
"""

from __future__ import annotations

import traceback
import uuid
from typing import Any

import sqlalchemy as sa
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from atm.evaluation.ground_truth import score_ground_truth
from atm.tasks.base import LLMLike, TaskSpec

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


async def compute_quality(
    spec: TaskSpec,
    answer: str,
    *,
    sandbox: Any | None = None,
    judge_llm: LLMLike | None = None,
    run_seed: int = 0,
) -> tuple[float | None, dict[str, Any]]:
    """Compute quality score for ``answer`` against ``spec``.

    Delegates to ``score_ground_truth`` from ground_truth.py, which dispatches
    to the appropriate M10 evaluator based on ``spec.evaluator_key``.

    This function NEVER raises. Any exception is caught and returned as
    ``(None, {"error": "<traceback>"})`` so that run finalisation remains
    stable even if evaluation fails.

    Args:
        spec:      The task specification; drives evaluator dispatch.
        answer:    The final answer string produced by the agent topology.
        sandbox:   A SubprocessSandbox instance — required for humaneval_pytest.
                   Judge LLM calls share the run's BudgetTracker; pass the
                   same LLMWrapper used during the run so costs are attributed
                   to the correct run budget.
        judge_llm: An LLMLike instance — required for creative_judge /
                   analysis_hybrid.  Judge calls share the run's BudgetTracker.
        run_seed:  Seed for self-consistency judge call reproducibility
                   (reserved for future SelfConsistentJudge wiring).

    Returns:
        A tuple ``(score, details)`` where:
          - ``score``: float in [0.0, 1.0] on success, None on any error.
          - ``details``: dict from EvalResult.details on success, or
                         ``{"error": "<traceback>"}`` on failure.
    """
    try:
        result = await score_ground_truth(
            spec,
            answer,
            sandbox=sandbox,
            judge_llm=judge_llm,
        )
        return result.score, dict(result.details)
    except Exception:
        tb = traceback.format_exc()
        logger.warning(
            "compute_quality failed",
            task_id=spec.id,
            evaluator_key=spec.evaluator_key,
            error=tb,
        )
        return None, {"error": tb}


async def persist_quality(
    session: AsyncSession,
    run_id: uuid.UUID,
    quality_score: float | None,
) -> None:
    """Idempotent UPDATE of runs.quality_score for the given run_id.

    Issues a raw SQL UPDATE so that ORM lazy-load restrictions on the Run
    model do not interfere.  If the run_id does not exist in the table,
    the UPDATE affects 0 rows and the call is a no-op.

    Args:
        session:       An open AsyncSession (caller owns transaction).
        run_id:        UUID of the run row to update.
        quality_score: The new quality score (may be None → SQL NULL).
    """
    stmt = (
        sa.update(sa.table("runs", sa.column("id"), sa.column("quality_score")))
        .where(sa.column("id") == run_id)
        .values(quality_score=quality_score)
    )
    await session.execute(stmt)
    logger.debug("persist_quality", run_id=str(run_id), quality_score=quality_score)


async def aggregate_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    spec: TaskSpec,
    answer: str,
    *,
    sandbox: Any | None = None,
    judge_llm: LLMLike | None = None,
    run_seed: int = 0,
) -> float | None:
    """Orchestrate quality computation and persistence for a single run.

    Calls ``compute_quality`` then ``persist_quality``.  Always persists —
    even if quality is None (SQL NULL) — so that the column is always updated
    on run completion.

    Args:
        session:   An open AsyncSession (caller owns transaction scope).
        run_id:    UUID of the run row to update.
        spec:      The task specification.
        answer:    The final answer string.
        sandbox:   SubprocessSandbox — forwarded to compute_quality.
        judge_llm: LLMLike — forwarded to compute_quality.
        run_seed:  Run seed — forwarded to compute_quality.

    Returns:
        The computed quality score (float) or None on evaluation failure.
    """
    score, details = await compute_quality(
        spec,
        answer,
        sandbox=sandbox,
        judge_llm=judge_llm,
        run_seed=run_seed,
    )
    if "error" in details:
        logger.warning(
            "aggregate_run: compute_quality returned error",
            run_id=str(run_id),
            error=details["error"][:200],
        )
    await persist_quality(session, run_id, score)
    return score
