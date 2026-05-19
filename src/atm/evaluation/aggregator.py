"""Aggregator — compute quality score and persist it to the database."""

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

    Never raises — any exception is caught and returned as ``(None, {"error": "..."})``.

    Args:
        spec:      Task specification (drives evaluator dispatch via ``evaluator_key``).
        answer:    Final answer string produced by the agent topology.
        sandbox:   Required for ``humaneval_pytest`` evaluator.
        judge_llm: Required by judge-based evaluators.
        run_seed:  Seed for reproducibility (forwarded to judge evaluators).

    Returns:
        ``(score, details)`` — score is float in [0, 1] or None on failure.
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
    """Idempotent UPDATE of runs.quality_score. No-op if run_id not found."""
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
    """Compute quality and persist it; always persists even if score is None."""
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
