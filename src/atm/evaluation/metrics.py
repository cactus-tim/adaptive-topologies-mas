"""Evaluation metrics for ATM experiment pipeline (m9.2).

Public API:
    human_sim_cognitive_load_proxy -- NASA-TLX proxy based on HITL interaction data.
    _load_weights                  -- Load alpha/beta/gamma weights from YAML (internal).

Formula:
    cognitive_load_proxy = alpha * count_interrupts
                         + beta  * mean(len(json.dumps(context_json)))
                         + gamma * mean(latency_s)

where latency_s = (answered_at - requested_at).total_seconds() if answered_at else 0.0.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import UUID

import sqlalchemy as sa
from omegaconf import OmegaConf
from sqlalchemy.ext.asyncio import AsyncSession

from atm.storage.models import HumanInteraction

# ---------------------------------------------------------------------------
# Weights file
# ---------------------------------------------------------------------------

WEIGHTS_PATH = Path("conf/evaluation/cognitive_load.yaml")

DEFAULT_WEIGHTS: dict[str, float] = {"alpha": 1.0, "beta": 0.001, "gamma": 0.1}

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_weights(path: Path = WEIGHTS_PATH) -> dict[str, float]:
    """Load alpha/beta/gamma weights from *path*.

    Falls back to :data:`DEFAULT_WEIGHTS` if the file is missing or malformed.
    """
    try:
        cfg = OmegaConf.load(path)
        return {
            "alpha": float(cfg.alpha),
            "beta": float(cfg.beta),
            "gamma": float(cfg.gamma),
        }
    except Exception as exc:
        _log.warning(
            "cognitive_load_proxy: could not load weights from %s (%s); using defaults",
            path,
            exc,
        )
        return dict(DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Public metric
# ---------------------------------------------------------------------------


async def human_sim_cognitive_load_proxy(
    session: AsyncSession,
    run_id: str | UUID,
) -> float:
    """Compute the NASA-TLX cognitive load proxy for a given run.

    Queries ``human_interactions`` for *run_id*, then applies:

        alpha * count_interrupts
        + beta  * mean(len(json.dumps(context_json)))
        + gamma * mean(latency_s)

    Args:
        session: Open async SQLAlchemy session.
        run_id:  Run identifier (UUID or str representation).

    Returns:
        Cognitive load proxy as a non-negative float.  Returns ``0.0`` when
        there are no HITL interactions for the given run.
    """
    rid: UUID = UUID(str(run_id)) if not isinstance(run_id, UUID) else run_id

    stmt = sa.select(
        HumanInteraction.id,
        HumanInteraction.context_json,
        HumanInteraction.requested_at,
        HumanInteraction.answered_at,
    ).where(HumanInteraction.run_id == rid)

    result = await session.execute(stmt)
    rows = result.fetchall()

    if not rows:
        return 0.0

    count_interrupts = len(rows)

    ctx_lens: list[int] = []
    latencies: list[float] = []

    for _, context_json, requested_at, answered_at in rows:
        ctx_lens.append(len(json.dumps(context_json, default=str)))
        if answered_at is not None:
            latencies.append((answered_at - requested_at).total_seconds())
        else:
            latencies.append(0.0)

    mean_context_len = sum(ctx_lens) / count_interrupts
    mean_latency = sum(latencies) / count_interrupts

    weights = _load_weights()
    alpha = weights["alpha"]
    beta = weights["beta"]
    gamma = weights["gamma"]

    return alpha * count_interrupts + beta * mean_context_len + gamma * mean_latency
