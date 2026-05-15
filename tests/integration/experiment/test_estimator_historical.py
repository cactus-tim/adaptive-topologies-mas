"""PG-gated integration test for ``estimate_grid`` historical-avg path.

Seeds an experiment + a handful of completed runs in PostgreSQL, then calls
``estimate_grid`` with a matching ``(topology, task)`` config and asserts:

  * the cell's ``source`` is ``"historical_avg"``;
  * ``est_cost_usd`` equals ``AVG(runs.budget_spent_usd)`` for that pair;
  * cells without historical data fall back to the heuristic path.

Skipped unless ``ATM_ENABLE_PG_TESTS=1``.
"""

from __future__ import annotations

import os
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from atm.experiment.config import EstimateCfg, ExperimentConfig
from atm.experiment.estimator import estimate_grid
from atm.experiment.loader import load_config
from atm.llm.pricing import ModelPricing, Pricing
from atm.storage.models import Experiment, Run

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
pytestmark = pytest.mark.requires_postgres

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "experiment"
VALID_MINIMAL = FIXTURES_DIR / "valid_minimal.yaml"


def _make_pricing() -> Pricing:
    return Pricing(
        version=1,
        models={"fake:scripted": ModelPricing(input_per_1k=0.001, output_per_1k=0.002)},
    )


def _cfg_for(topology: str, task: str) -> ExperimentConfig:
    base = load_config(str(VALID_MINIMAL))
    return base.model_copy(
        update={
            "topology": base.topology.model_copy(update={"name": topology}),
            "task": base.task.model_copy(update={"name": task}),
        }
    )


@pytest.mark.asyncio
async def test_estimator_picks_historical_avg(  # type: ignore[no-untyped-def]
    pg_engine_fast,
    session_factory_fast,
) -> None:
    """Seed 3 completed runs → estimator returns historical_avg with matching cost."""
    exp_id = uuid.uuid4()

    async with session_factory_fast() as session:
        session.add(
            Experiment(
                id=exp_id,
                name="hist_test",
                status="running",
            )
        )
        for cost, iters in [(0.05, 4), (0.06, 5), (0.07, 6)]:
            session.add(
                Run(
                    id=uuid.uuid4(),
                    exp_id=exp_id,
                    topology="star",
                    task_id="humaneval",
                    agent_set="canonical_4",
                    seed=42,
                    model="fake:scripted",
                    status="completed",
                    budget_spent_usd=Decimal(str(cost)),
                    iterations=iters,
                )
            )
        await session.commit()

    cfg = _cfg_for("star", "humaneval")
    est_cfg = EstimateCfg(use_historical=True)
    result = await estimate_grid(
        configs=[cfg],
        session_factory=session_factory_fast,
        pricing=_make_pricing(),
        cfg=est_cfg,
    )

    assert len(result.cells) == 1
    cell = result.cells[0]
    assert cell.source == "historical_avg"
    expected_avg = (0.05 + 0.06 + 0.07) / 3
    assert cell.est_cost_usd == pytest.approx(expected_avg, rel=1e-4)


@pytest.mark.asyncio
async def test_estimator_falls_back_to_heuristic_without_data(  # type: ignore[no-untyped-def]
    pg_engine_fast,
    session_factory_fast,
) -> None:
    """Empty DB → heuristic source, cost from Pricing.estimate."""
    cfg = _cfg_for("chain", "no_historical_task")
    est_cfg = EstimateCfg(use_historical=True)
    pricing = _make_pricing()

    result = await estimate_grid(
        configs=[cfg],
        session_factory=session_factory_fast,
        pricing=pricing,
        cfg=est_cfg,
    )
    assert len(result.cells) == 1
    assert result.cells[0].source == "heuristic"


@pytest.mark.asyncio
async def test_estimator_mixed_grid_partial_history(  # type: ignore[no-untyped-def]
    pg_engine_fast,
    session_factory_fast,
) -> None:
    """Mixed grid: one (topology, task) has history, one does not."""
    exp_id = uuid.uuid4()
    async with session_factory_fast() as session:
        session.add(Experiment(id=exp_id, name="mix_test", status="running"))
        session.add(
            Run(
                id=uuid.uuid4(),
                exp_id=exp_id,
                topology="star",
                task_id="humaneval",
                agent_set="canonical_4",
                seed=42,
                model="fake:scripted",
                status="completed",
                budget_spent_usd=Decimal("0.10"),
                iterations=5,
            )
        )
        # A non-completed run for chain — must NOT contribute.
        session.add(
            Run(
                id=uuid.uuid4(),
                exp_id=exp_id,
                topology="chain",
                task_id="humaneval",
                agent_set="canonical_4",
                seed=42,
                model="fake:scripted",
                status="running",
                budget_spent_usd=Decimal("99.99"),
                iterations=99,
            )
        )
        await session.commit()

    cfg_star = _cfg_for("star", "humaneval")
    cfg_chain = _cfg_for("chain", "humaneval")
    result = await estimate_grid(
        configs=[cfg_star, cfg_chain],
        session_factory=session_factory_fast,
        pricing=_make_pricing(),
        cfg=EstimateCfg(use_historical=True),
    )
    cells_by_topo = {c.topology: c for c in result.cells}
    assert cells_by_topo["star"].source == "historical_avg"
    assert cells_by_topo["star"].est_cost_usd == pytest.approx(0.10, rel=1e-4)
    assert cells_by_topo["chain"].source == "heuristic"  # only "running" doesn't count
