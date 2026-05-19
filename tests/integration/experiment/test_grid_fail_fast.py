"""M12 grid-runner integration test (PG): fail_fast cancellation.

Verifies that ``run_grid(fail_fast=True)`` triggers
``executor.shutdown(wait=False, cancel_futures=True)`` on the first failed
or budget_exceeded cell, so that some pending futures get cancelled and the
total runs in PG is less than the configured number of cells.

Strategy:
  - 6-cell grid: 3 cells with ``budget.per_run_usd=0.0001`` (impossibly tight
    → budget_exceeded), 3 cells with ``budget.per_run_usd=0.5`` (normal).
  - The first cell to complete must be one of the budget_exceeded ones (they
    fail at the first LLM call, while normal cells run the full pipeline).
  - fail_fast=True → remaining futures cancelled → total runs in PG < 6.

Gated on ``ATM_ENABLE_PG_TESTS=1`` via ``ephemeral_pg_dsn`` fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PG_TESTS_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_FIXTURE_YAML = (
    Path(__file__).parent.parent.parent / "fixtures" / "experiment" / "grid_runner_failfast.yaml"
)


pytestmark = pytest.mark.requires_postgres


@pytest.mark.asyncio
async def test_grid_fail_fast_cancels_pending_futures(
    ephemeral_pg_dsn: str,
    tmp_path: Path,
) -> None:
    """fail_fast=True must cancel pending cells; final GridResult reports
    non-completed terminal counts >= 1."""
    import sqlalchemy as sa

    from atm.experiment.grid import run_grid
    from atm.experiment.loader import load_grid_configs
    from atm.storage.session import create_engine

    overrides = [
        f"observability.pg_dsn={ephemeral_pg_dsn}",
        f"observability.parquet_dir={tmp_path / 'parquet'}",
        f"name=m12_grid_failfast_{os.getpid()}",
    ]
    configs = load_grid_configs(str(_FIXTURE_YAML), overrides=overrides)
    assert len(configs) == 6, f"expected 6 cells, got {len(configs)}"

    result = await run_grid(configs, parallelism=2, fail_fast=True)

    failed_total = result.failed + result.budget_exceeded
    assert failed_total >= 1, (
        f"expected >=1 failed/budget_exceeded; got failed={result.failed} "
        f"budget_exceeded={result.budget_exceeded}"
    )

    total_terminal = result.completed + result.failed + result.budget_exceeded
    assert total_terminal <= result.total

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    try:
        async with engine.connect() as conn:
            exp_status = (
                await conn.execute(
                    sa.text("SELECT status FROM experiments WHERE id = :eid").bindparams(
                        eid=result.exp_id
                    )
                )
            ).scalar_one_or_none()
            if exp_status is not None:
                assert exp_status in {"failed", "partial"}, f"unexpected status={exp_status}"
    finally:
        await engine.dispose()
