"""Pure metric functions for RQ1-RQ4.

All functions are stateless and side-effect-free.

Quality metrics (RQ1):
    aggregate_quality      — arithmetic mean of per-run quality scores
    humaneval_pass_at_k    — unbiased pass@k estimator (Chen et al. 2021)

Efficiency metrics (RQ2):
    cost_per_quality       — cost / max(quality, eps)
    time_per_quality       — latency / max(quality, eps)

Human-load metric (RQ4):
    aggregate_human_load   — mean NASA-TLX raw_score across a list of ratings
                             (delegates to atm.evaluation.tlx.aggregate_tlx)

Reference:
    Chen et al. 2021, "Evaluating Large Language Models Trained on Code"
    arXiv:2107.03374 — Appendix A, unbiased estimator of pass@k.
"""

from __future__ import annotations

import math

from atm.evaluation.tlx import NasaTLX, aggregate_tlx

_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


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
    # Unbiased estimator: iterate c times, each factor is (1 - k/(n-i))
    # Equivalent to 1 - C(n-c, k) / C(n, k) but numerically stable for large n.
    return 1.0 - math.prod(1.0 - k / (n - i) for i in range(c))


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Human load
# ---------------------------------------------------------------------------


def aggregate_human_load(tlx_list: list[NasaTLX]) -> float:
    """Return the mean NASA-TLX raw_score across *tlx_list*.

    Delegates to :func:`atm.evaluation.tlx.aggregate_tlx`.

    Returns 0.0 for an empty list.
    """
    if not tlx_list:
        return 0.0
    return aggregate_tlx(tlx_list)
