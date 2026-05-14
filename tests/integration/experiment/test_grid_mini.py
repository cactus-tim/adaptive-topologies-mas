"""M12 grid-runner integration test (PG): mini 2x1x2 = 4-cell grid.

Verifies the end-to-end M12 grid pipeline:
  - ``load_grid_configs`` expands the YAML's grid.sweep x seeds into 4 cells.
  - ``run_grid(parallelism=2)`` spawns child processes via ProcessPoolExecutor;
    each invokes ``run_one`` against the ephemeral PG.
  - All 4 ``runs`` rows present with terminal ``status='completed'``.
  - Each row has ``host`` and ``process_pid`` populated (M12 reconcile prereq).
  - The aggregate ``experiments.status='completed'``.

Gated on ``ATM_ENABLE_PG_TESTS=1`` via ``ephemeral_pg_dsn`` fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PG_TESTS_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_FIXTURE_YAML = (
    Path(__file__).parent.parent.parent / "fixtures" / "experiment" / "grid_runner_mini.yaml"
)


pytestmark = pytest.mark.requires_postgres


@pytest.mark.asyncio
async def test_grid_mini_all_cells_complete_with_host_pid(
    ephemeral_pg_dsn: str,
    tmp_path: Path,
) -> None:
    """Run a 4-cell mini-grid via run_grid and assert all PG side-effects."""
    import sqlalchemy as sa

    from atm.experiment.grid import run_grid
    from atm.experiment.loader import load_grid_configs
    from atm.storage.session import create_engine

    overrides = [
        f"observability.pg_dsn={ephemeral_pg_dsn}",
        f"observability.parquet_dir={tmp_path / 'parquet'}",
        # Force unique experiment name per test to avoid cross-run collisions.
        f"name=m12_grid_mini_{os.getpid()}",
    ]
    configs = load_grid_configs(str(_FIXTURE_YAML), overrides=overrides)

    assert len(configs) == 4, f"expected 4 cells, got {len(configs)}"

    # parallelism=2 keeps PG connection load reasonable.
    result = await run_grid(configs, parallelism=2, fail_fast=False)

    # Aggregate counts.
    assert result.total == 4
    assert result.completed == 4, (
        f"expected all 4 cells completed; got completed={result.completed} "
        f"failed={result.failed} budget_exceeded={result.budget_exceeded}"
    )
    assert len(result.run_ids) == 4

    # DB-side verification.
    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    try:
        async with engine.connect() as conn:
            # Per-run host/pid populated.
            rows = (
                await conn.execute(
                    sa.text(
                        "SELECT id, status, host, process_pid FROM runs WHERE exp_id = :eid"
                    ).bindparams(eid=result.exp_id)
                )
            ).fetchall()
            assert len(rows) == 4, f"expected 4 run rows; got {len(rows)}"
            for row in rows:
                assert row.status == "completed", f"run {row.id} status={row.status}"
                assert row.host is not None and row.host != "", f"run {row.id} host not populated"
                assert row.process_pid is not None and row.process_pid > 0, (
                    f"run {row.id} process_pid not populated"
                )

            # Experiment-aggregate status.
            exp_status = (
                await conn.execute(
                    sa.text("SELECT status FROM experiments WHERE id = :eid").bindparams(
                        eid=result.exp_id
                    )
                )
            ).scalar_one()
            assert exp_status == "completed", f"experiment status={exp_status}"
    finally:
        await engine.dispose()
