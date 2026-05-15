"""Cost / token estimator for an experiment grid (M12 — m12-estimate-status-cli).

Provides a single async entry point, :func:`estimate_grid`, that walks a list
of :class:`ExperimentConfig` cells and produces a :class:`GridEstimate` with
per-cell token / USD projections plus per-topology and grand-total roll-ups.

Two estimation sources are supported per cell:

* ``historical_avg`` — when ``cfg.estimate.use_historical=True`` AND we have
  at least one ``completed`` run for the same ``(topology, task)`` pair in
  the ``runs`` PG table, we take ``AVG(budget_spent_usd)`` as the cost
  estimate (and reconstruct synthetic token counts from
  ``AVG(iterations) * calls_per_iter * heuristic_tokens_per_call`` for the
  display column).

* ``heuristic`` — pure formula fallback. Tokens =
  ``calls_per_iter * heuristic_tokens_per_call * topology.max_iterations``;
  output tokens = input // 3 (typical ratio); cost via
  :meth:`Pricing.estimate`.

The historical SELECT uses bound parameters (no f-string SQL); the function
accepts ``session_factory=None`` so the CLI can degrade to a heuristic-only
estimate when the database is unreachable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atm.experiment.config import EstimateCfg, ExperimentConfig
from atm.llm.pricing import Pricing
from atm.storage.models import Run

EstimateSource = Literal["historical_avg", "heuristic"]


@dataclass(frozen=True, slots=True)
class CellEstimate:
    """Projected cost for one (topology, task, seed) cell of the grid."""

    topology: str
    task: str
    seed: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float
    source: EstimateSource


@dataclass(frozen=True, slots=True)
class GridEstimate:
    """Aggregate projection for an entire grid."""

    cells: list[CellEstimate]
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    per_topology: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _load_historical_avgs(
    session: AsyncSession,
    pairs: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], tuple[float, float]]:
    """Return ``{(topology, task): (avg_cost_usd, avg_iters)}`` for completed runs.

    Pairs with no completed runs are simply absent from the result dict, so
    callers fall back to the heuristic path.

    Uses :func:`sqlalchemy.select` with bound ``IN`` filters — no string
    interpolation, immune to SQL injection.
    """
    unique_pairs = sorted({(t, k) for t, k in pairs})
    if not unique_pairs:
        return {}

    topologies = sorted({t for t, _ in unique_pairs})
    tasks = sorted({k for _, k in unique_pairs})

    # Single query for the cartesian superset; we filter to the requested
    # pairs in Python so the SQL stays simple and uses bound parameters only.
    from sqlalchemy import func

    stmt = (
        select(
            Run.topology,
            Run.task_id,
            func.avg(Run.budget_spent_usd).label("avg_cost"),
            func.avg(Run.iterations).label("avg_iters"),
        )
        .where(Run.status == "completed")
        .where(Run.topology.in_(topologies))
        .where(Run.task_id.in_(tasks))
        .group_by(Run.topology, Run.task_id)
    )

    result = await session.execute(stmt)
    out: dict[tuple[str, str], tuple[float, float]] = {}
    requested = set(unique_pairs)
    for row in result.all():
        key = (row.topology, row.task_id)
        if key not in requested:
            continue
        avg_cost = float(row.avg_cost) if row.avg_cost is not None else 0.0
        avg_iters = float(row.avg_iters) if row.avg_iters is not None else 0.0
        if avg_cost > 0.0:
            out[key] = (avg_cost, avg_iters)
    return out


def _heuristic_cell(
    cfg: ExperimentConfig,
    pricing: Pricing,
    est_cfg: EstimateCfg,
) -> tuple[int, int, float]:
    """Compute (input_tokens, output_tokens, cost_usd) via the heuristic path."""
    est_input = (
        est_cfg.calls_per_iter * est_cfg.heuristic_tokens_per_call * cfg.topology.max_iterations
    )
    est_output = est_input // 3
    try:
        cost = pricing.estimate(
            cfg.model.default,
            prompt_tokens=est_input,
            completion_tokens=est_output,
        )
    except Exception:
        # Unknown model in pricing table — fall back to zero rather than
        # crashing the whole estimate (mirrors `_load_pricing` fallback).
        cost = 0.0
    return est_input, est_output, cost


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def estimate_grid(
    configs: list[ExperimentConfig],
    session_factory: async_sessionmaker[AsyncSession] | None,
    pricing: Pricing,
    cfg: EstimateCfg,
) -> GridEstimate:
    """Project token/USD costs for an entire grid.

    Args:
        configs:         List of ``ExperimentConfig`` cells (typically from
                         :func:`load_grid_configs`).
        session_factory: Async session factory for PG reads. May be ``None``
                         if ``cfg.use_historical`` is False or the caller
                         could not establish a DB connection — the function
                         then degrades to heuristic-only estimates.
        pricing:         Pricing table for ``Pricing.estimate`` calls.
        cfg:             Heuristic + historical-toggle parameters
                         (``ExperimentConfig.estimate``).

    Returns:
        Aggregated :class:`GridEstimate` (empty cells list if *configs* is
        empty).
    """
    if not configs:
        return GridEstimate(
            cells=[],
            total_cost_usd=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            per_topology={},
        )

    # ------------------------------------------------------------------
    # Phase 1: load historical averages (one round-trip).
    # ------------------------------------------------------------------
    historical: dict[tuple[str, str], tuple[float, float]] = {}
    if cfg.use_historical and session_factory is not None:
        pairs = [(c.topology.name, c.task.name) for c in configs]
        async with session_factory() as session:
            historical = await _load_historical_avgs(session, pairs)

    # ------------------------------------------------------------------
    # Phase 2: walk cells, classify, compute.
    # ------------------------------------------------------------------
    cells: list[CellEstimate] = []
    per_topology: dict[str, float] = defaultdict(float)
    total_in = 0
    total_out = 0
    total_cost = 0.0

    for c in configs:
        key = (c.topology.name, c.task.name)
        if key in historical:
            avg_cost, avg_iters = historical[key]
            # Synthetic token counts for the display table (rough — historical
            # cost is what we actually report).
            iters_for_tokens = avg_iters if avg_iters > 0 else float(c.topology.max_iterations)
            est_input = int(cfg.calls_per_iter * cfg.heuristic_tokens_per_call * iters_for_tokens)
            est_output = est_input // 3
            est_cost = avg_cost
            source: EstimateSource = "historical_avg"
        else:
            est_input, est_output, est_cost = _heuristic_cell(c, pricing, cfg)
            source = "heuristic"

        cell = CellEstimate(
            topology=c.topology.name,
            task=c.task.name,
            seed=c.seed,
            est_input_tokens=est_input,
            est_output_tokens=est_output,
            est_cost_usd=est_cost,
            source=source,
        )
        cells.append(cell)
        per_topology[c.topology.name] += est_cost
        total_in += est_input
        total_out += est_output
        total_cost += est_cost

    return GridEstimate(
        cells=cells,
        total_cost_usd=total_cost,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        per_topology=dict(per_topology),
    )
