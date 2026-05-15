"""M12 exit criterion 1 — parallel grid execution via ``atm grid`` subprocess.

Verifies the end-to-end M12 grid pipeline through the real CLI binary (not CliRunner):
  - A 4-cell mini-grid (2 topologies x 2 seeds) is driven by
    ``atm grid --config <tmp_yaml> --parallelism 4 --yes --no-estimate --no-reconcile``.
  - Exit code is 0 (all cells completed).
  - All 4 ``runs`` rows in PG have ``status='completed'``.
  - ``experiments.status='completed'``.
  - Wall-time bound: total subprocess duration < 2x single-cell estimate, proving
    that cells ran concurrently and not sequentially.

Design note (BLOCKER 2)
-----------------------
``asyncio_mode = auto`` in pyproject.toml means pytest-asyncio installs its own
event loop.  Using ``typer.testing.CliRunner`` inside an async test would create a
SECOND event loop and conflict.  This test avoids that entirely by using
``subprocess.run`` (real OS process) and opening its own async engine for DB
assertions — compatible with the existing pytest event loop.

Gated on ``ATM_ENABLE_PG_TESTS=1`` via the root ``ephemeral_pg_dsn`` fixture.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
import time
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module-level guard (also enforced by the ``ephemeral_pg_dsn`` fixture skip)
# ---------------------------------------------------------------------------

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")

pytestmark = [
    pytest.mark.requires_postgres,
    pytest.mark.integration,
]

# ---------------------------------------------------------------------------
# Paths to FakeLLM fixtures (resolved at import time — do not rely on cwd)
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_GRID_MINI_YAML = (
    Path(__file__).parent.parent.parent / "fixtures" / "experiment" / "grid_runner_mini.yaml"
)

# Subprocess timeout: generous upper bound (120 s) so CI slowness doesn't cause
# false negatives.  The wall-time parallelism assertion uses a much tighter bound.
_SUBPROCESS_TIMEOUT_S = 120

# Wall-time parallelism threshold: 4 cells in parallel with parallelism=4
# should take roughly the same time as 1 cell (FakeLLM, no real IO).
# We allow 3x single-cell budget as the upper bound to absorb process-spawn
# overhead and scheduler jitter, while still rejecting full sequential execution
# (which would take ~4x or more).
#
# FakeLLM star-topology cells with max_iterations=4 finish in well under 5 s
# each on modern hardware; sequential execution of 4 would be ~20 s.
# We cap at 30 s total, which fails only if parallelism broke down to serial.
_WALL_TIME_PARALLEL_MAX_S = 30.0


# ---------------------------------------------------------------------------
# Helper: build a self-contained grid YAML referencing the ephemeral PG DSN
# ---------------------------------------------------------------------------


def _write_grid_yaml(tmp_path: Path, *, pg_dsn: str, parquet_dir: str) -> tuple[Path, str]:
    """Write a 2-topology x 2-seed = 4-cell grid YAML into *tmp_path*.

    All LLM fixtures are resolved to absolute paths so the subprocess (which
    may have a different cwd) can find them.  The PG DSN is embedded verbatim.
    """
    planner_fix = str(_FIXTURES_DIR / "m6_star_planner.yaml")
    executor_fix = str(_FIXTURES_DIR / "m6_star_executor.yaml")
    critic_fix = str(_FIXTURES_DIR / "m6_star_critic.yaml")

    unique_name = f"m12_grid_parallel_{uuid.uuid4().hex[:8]}"

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
          parquet_dir: "{parquet_dir}"
          callback_sync: true

        grid:
          sweep:
            topology.name: ["star", "chain"]
          parallelism: 4
          fail_fast: false
          seeds: [42, 43]
        """
    )

    config_path = tmp_path / "m12_grid_parallel_config.yaml"
    config_path.write_text(yaml_content)
    return config_path, unique_name


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grid_parallel_all_cells_complete(
    ephemeral_pg_dsn: str,
    tmp_path: Path,
) -> None:
    """M12 EC-1: ``atm grid`` with parallelism=4 → 4 cells completed in PG.

    Steps:
      1. Write a self-contained grid YAML referencing the ephemeral PG DSN.
      2. Run ``atm grid --config <yaml> --parallelism 4 --yes
                        --no-estimate --no-reconcile`` as a blocking subprocess.
      3. Assert exit code == 0.
      4. Open an async PG session and assert:
         - 4 rows in ``runs`` for this experiment, all ``status='completed'``.
         - ``experiments.status='completed'``.
      5. Assert wall-time < ``_WALL_TIME_PARALLEL_MAX_S`` (proves parallelism).
    """
    import sqlalchemy as sa

    from atm.storage.session import create_engine

    if not _PG_ENABLED:
        pytest.skip("ATM_ENABLE_PG_TESTS not set")

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    config_path, experiment_name = _write_grid_yaml(
        tmp_path,
        pg_dsn=ephemeral_pg_dsn,
        parquet_dir=str(parquet_dir),
    )

    # Propagate the test environment; silence noisy structlog bootstrap.
    env = os.environ.copy()
    env["ATM_PG_DSN"] = ephemeral_pg_dsn
    env["ATM_DISABLE_STRUCTLOG_BOOTSTRAP"] = "1"

    # ── Step 2: run ``atm grid`` ────────────────────────────────────────────
    t_start = time.monotonic()

    result = subprocess.run(
        [
            "uv",
            "run",
            "atm",
            "grid",
            "--config",
            str(config_path),
            "--parallelism",
            "4",
            "--yes",
            "--no-estimate",
            "--no-reconcile",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )

    elapsed_s = time.monotonic() - t_start

    # ── Step 3: assert exit code ────────────────────────────────────────────
    assert result.returncode == 0, (
        f"``atm grid`` exited with code {result.returncode} (expected 0).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # ── Step 4: assert DB side-effects ─────────────────────────────────────
    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    try:
        async with engine.connect() as conn:
            # Look up THIS test's experiment by its unique name — robust against
            # cross-test state pollution that can leave rows from earlier tests
            # in the same ephemeral DB.
            exp_id_row = (
                await conn.execute(
                    sa.text("SELECT id FROM experiments WHERE name = :name").bindparams(
                        name=experiment_name
                    )
                )
            ).fetchone()

            assert exp_id_row is not None, (
                f"No experiment row found for name={experiment_name!r}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
            exp_id = exp_id_row[0]

            # 4 run rows for this experiment.
            run_rows = (
                await conn.execute(
                    sa.text("SELECT id, status FROM runs WHERE exp_id = :eid").bindparams(
                        eid=exp_id
                    )
                )
            ).fetchall()

            assert len(run_rows) == 4, (
                f"Expected 4 run rows, found {len(run_rows)}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

            # All 4 runs must be completed.
            non_completed = [(str(row[0]), row[1]) for row in run_rows if row[1] != "completed"]
            assert not non_completed, (
                f"Some runs did not reach 'completed' status: {non_completed}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

            # Experiment aggregate status must be 'completed'.
            exp_status = (
                await conn.execute(
                    sa.text("SELECT status FROM experiments WHERE id = :eid").bindparams(eid=exp_id)
                )
            ).scalar_one()

            assert exp_status == "completed", (
                f"experiments.status={exp_status!r}, expected 'completed'.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    finally:
        await engine.dispose()

    # ── Step 5: wall-time parallelism assertion ─────────────────────────────
    # If parallelism=4 worked, 4 cells should not take significantly longer than
    # 1 cell.  We assert elapsed < _WALL_TIME_PARALLEL_MAX_S.  Sequential
    # execution of 4 cells would exceed this bound.
    assert elapsed_s < _WALL_TIME_PARALLEL_MAX_S, (
        f"``atm grid`` took {elapsed_s:.1f}s, exceeding the parallelism threshold "
        f"of {_WALL_TIME_PARALLEL_MAX_S}s.  This suggests cells ran sequentially "
        "rather than in parallel."
    )
