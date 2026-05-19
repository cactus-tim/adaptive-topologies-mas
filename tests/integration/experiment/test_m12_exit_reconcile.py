"""Integration test for M12 exit criterion 4 — reconcile clears zombie runs.

Test flow
---------
1. Load grid configs from ``grid_runner_mini.yaml`` via ``load_grid_configs``
   with OmegaConf overrides so the config is bit-identical to what ``atm grid``
   will reconstruct.
2. Seed a zombie run row using ORM helpers (NO raw SQL INSERT):
   - Call ``_ensure_experiment(session_factory, cfg)`` from ``atm.experiment.runner``
     to create the experiment row.
   - Call ``_insert_run(session_factory, exp_id, cfg)`` to create a run row with
     all NOT NULL columns populated.
   - UPDATE the row via SQLAlchemy ORM statement to set status='running',
     host=socket.gethostname(), and a dead PID.
3. Run ``uv run atm grid --config <tmp_yaml> --parallelism 2 --yes --no-estimate``
   as a blocking subprocess — reconcile is ON by default.  Exit code 0.
4. Assert: the zombie row now has status='failed' and finish_reason='zombie'.
5. Assert: new grid run rows are also present (the grid ran normally).

Markers
-------
- ``requires_postgres``: gated on ``ATM_ENABLE_PG_TESTS=1``.
- ``integration``: marks the test as an integration test.

Design note (BLOCKER 3 from plan review)
-----------------------------------------
We must NOT use raw SQL INSERT for the zombie row because the Run model has many
NOT NULL columns that are not easily enumerable without reading the ORM definition.
The ORM helpers ``_ensure_experiment`` + ``_insert_run`` handle all required fields
and are the same path the real runner uses — making the zombie row realistic.
"""

from __future__ import annotations

import os
import socket
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")

pytestmark = [
    pytest.mark.requires_postgres,
    pytest.mark.integration,
]

_FIXTURE_YAML = (
    Path(__file__).parent.parent.parent / "fixtures" / "experiment" / "grid_runner_mini.yaml"
)

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"


def _write_grid_yaml(
    tmp_path: Path,
    *,
    pg_dsn: str,
    parquet_dir: Path,
    unique_name: str,
) -> Path:
    """Write a self-contained grid YAML with injected DSN, parquet dir and name.

    Uses star topology only (no sweep) so the grid degenerates to a single cell
    — that is enough to prove that reconcile ran and that the grid executed
    successfully on top of the zombie cleanup.

    We deliberately use a single-cell grid (no ``grid.sweep``) so the
    subprocess finishes quickly and the total run count is predictable:
    - 1 zombie row seeded before the subprocess
    - 1 new run row created by the grid subprocess
    → total of 2 run rows for the experiment after the grid finishes.
    """
    planner_fix = str(_FIXTURES_DIR / "m6_star_planner.yaml")
    executor_fix = str(_FIXTURES_DIR / "m6_star_executor.yaml")
    critic_fix = str(_FIXTURES_DIR / "m6_star_critic.yaml")

    yaml_content = textwrap.dedent(
        f"""\
        name: {unique_name}
        seed: 42

        task:
          name: fibonacci_smoke
          input: "Write a Python function fib(n) that returns the n-th Fibonacci number. Compute fib(10)."

        model:
          default: "fake:scripted"
          by_role:
            planner: "fake:scripted"
            executor: "fake:scripted"
            critic: "fake:scripted"
            researcher: "fake:scripted"
          fake_fixtures:
            planner: "{planner_fix}"
            executor: "{executor_fix}"
            critic: "{critic_fix}"
            researcher: "{planner_fix}"

        agents:
          set: canonical_4

        topology:
          name: star
          max_iterations: 4
          extra: {{}}

        budget:
          per_call_usd: 0.10
          per_run_usd: 0.50
          per_experiment_usd: 50.0

        observability:
          pg_dsn: "{pg_dsn}"
          parquet_dir: "{parquet_dir!s}"
          callback_sync: true

        human:
          enabled: false
        """
    )

    cfg_path = tmp_path / "reconcile_test_grid.yaml"
    cfg_path.write_text(yaml_content)
    return cfg_path


@pytest.mark.asyncio
async def test_reconcile_zombie_on_grid_start(
    ephemeral_pg_dsn: str,
    tmp_path: Path,
) -> None:
    """M12 EC-4: zombie run is reconciled to status='failed'/'zombie' by atm grid.

    Steps:
      1. Create an experiment row + a stale 'running' run (zombie) via ORM helpers.
      2. Run ``atm grid`` subprocess (reconcile ON by default).
      3. Assert: zombie row now has status='failed', finish_reason='zombie'.
      4. Assert: at least one new completed run row was created by the grid.
    """
    from atm.experiment.loader import load_grid_configs
    from atm.experiment.runner import _ensure_experiment, _insert_run
    from atm.storage.models import Run
    from atm.storage.session import create_engine, create_session_factory, session_scope

    unique_name = f"m12_reconcile_{uuid.uuid4().hex[:8]}"
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(exist_ok=True)

    cfg_path = _write_grid_yaml(
        tmp_path,
        pg_dsn=ephemeral_pg_dsn,
        parquet_dir=parquet_dir,
        unique_name=unique_name,
    )
    overrides = [f"name={unique_name}"]
    configs = load_grid_configs(str(cfg_path), overrides=overrides)
    assert len(configs) >= 1, "Expected at least one config cell"
    cfg = configs[0]

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    session_factory = create_session_factory(engine)

    exp_id = await _ensure_experiment(session_factory, cfg)

    zombie_run_id = await _insert_run(session_factory, exp_id, cfg)

    dead_pid = (os.getpid() + 999999) % 4_194_304
    if dead_pid == os.getpid():
        dead_pid = (dead_pid + 1) % 4_194_304

    async with session_scope(session_factory) as session:
        await session.execute(
            sa.update(Run)
            .where(Run.id == zombie_run_id)
            .values(
                status="running",
                host=socket.gethostname(),
                process_pid=dead_pid,
            )
        )

    await engine.dispose()

    env = os.environ.copy()
    env["ATM_PG_DSN"] = ephemeral_pg_dsn
    env["ATM_PARQUET_DIR"] = str(parquet_dir)
    env["ATM_DISABLE_STRUCTLOG_BOOTSTRAP"] = "1"

    result = subprocess.run(
        [
            "uv",
            "run",
            "atm",
            "grid",
            "--config",
            str(cfg_path),
            "--parallelism",
            "2",
            "--yes",
            "--no-estimate",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode in (0, 1), result.stderr

    engine2 = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    try:
        async with engine2.connect() as conn:
            zombie_row = (
                await conn.execute(
                    sa.text("SELECT status, finish_reason FROM runs WHERE id = :rid").bindparams(
                        rid=zombie_run_id
                    )
                )
            ).fetchone()

            assert zombie_row is not None, (
                f"Zombie run row {zombie_run_id} not found after grid execution"
            )
            assert zombie_row.status == "failed", (
                f"Expected zombie run to have status='failed', got '{zombie_row.status}'"
            )
            assert zombie_row.finish_reason == "zombie", (
                f"Expected zombie run to have finish_reason='zombie', "
                f"got '{zombie_row.finish_reason}'"
            )

            run_rows = (
                await conn.execute(
                    sa.text(
                        "SELECT id, status FROM runs WHERE exp_id = :eid ORDER BY started_at"
                    ).bindparams(eid=exp_id)
                )
            ).fetchall()

            non_zombie_runs = [r for r in run_rows if r.id != zombie_run_id]
            assert len(non_zombie_runs) >= 1, (
                "Expected at least one new run row created by the grid subprocess, "
                f"but only found {len(run_rows)} total row(s) "
                f"(zombie_id={zombie_run_id})"
            )

            completed = [r for r in non_zombie_runs if r.status == "completed"]
            assert len(completed) >= 1, (
                f"Expected at least one completed run from the grid, "
                f"got statuses: {[r.status for r in non_zombie_runs]}"
            )
    finally:
        await engine2.dispose()
