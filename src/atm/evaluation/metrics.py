"""Pure metric functions for RQ1-RQ4 (stateless, side-effect-free)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from uuid import UUID

import sqlalchemy as sa
from omegaconf import OmegaConf
from sqlalchemy.ext.asyncio import AsyncSession

from atm.evaluation.tlx import NasaTLX, aggregate_tlx
from atm.storage.models import HumanInteraction

_EPS: float = 1e-6

WEIGHTS_PATH = Path("conf/evaluation/cognitive_load.yaml")

DEFAULT_WEIGHTS: dict[str, float] = {"alpha": 1.0, "beta": 0.001, "gamma": 0.1}

_log = logging.getLogger(__name__)


def aggregate_quality(scores: list[float]) -> float:
    """Return the arithmetic mean of *scores*.

    Returns 0.0 for an empty list (no division).
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def humaneval_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k from Chen et al. 2021 (arXiv:2107.03374).

    Args:
        n: Total number of samples generated per problem.
        c: Number of samples that pass all unit tests (correct solutions).
        k: Number of samples selected for evaluation (pass@k).

    Returns:
        Estimated probability that at least one of k randomly selected
        samples is correct, computed as:

            1 - prod_{i=0}^{c-1} (1 - k / (n - i))

        Special cases:
        - c == 0          → 0.0 (no correct samples exist)
        - n - c < k       → 1.0 (fewer wrong than k; guaranteed to pick one correct)
        - c == n          → 1.0 (all samples are correct)
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / (n - i) for i in range(c))


def cost_per_quality(cost: float, quality: float) -> float:
    """Return *cost* divided by *quality*, clamped at eps to avoid division by zero.

    Args:
        cost:    Total monetary / token cost of a run (non-negative).
        quality: Quality score in [0, 1].

    Returns:
        cost / max(quality, 1e-6)
    """
    return cost / max(quality, _EPS)


def time_per_quality(time_s: float, quality: float) -> float:
    """Return *time_s* divided by *quality*, clamped at eps to avoid division by zero.

    Args:
        time_s:  Wall-clock duration of a run in seconds (non-negative).
        quality: Quality score in [0, 1].

    Returns:
        time_s / max(quality, 1e-6)
    """
    return time_s / max(quality, _EPS)


def aggregate_human_load(tlx_list: list[NasaTLX]) -> float:
    """Return the mean NASA-TLX raw_score across *tlx_list*.

    Delegates to :func:`atm.evaluation.tlx.aggregate_tlx`.

    Returns 0.0 for an empty list.
    """
    if not tlx_list:
        return 0.0
    return aggregate_tlx(tlx_list)


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


async def human_sim_cognitive_load_proxy(
    session: AsyncSession,
    run_id: str | UUID,
) -> float:
    """Compute the NASA-TLX cognitive load proxy for a given run.

    Queries ``human_interactions`` for *run_id*, then applies:

        alpha * count_interrupts
        + beta  * mean(len(json.dumps(context_json)))
        + gamma * mean(latency_s)

    Returns 0.0 when there are no HITL interactions for the run.
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
