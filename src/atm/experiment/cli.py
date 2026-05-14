"""Typer CLI for the ATM experiment runner.

Usage:
    atm run      --config conf/experiments/smoke.yaml [+key=val ...]
    atm grid     --config conf/experiments/grid.yaml  [--parallelism N] [--fail-fast]
                  [--yes] [--no-reconcile] [--force-resume] [--resume-incomplete]
                  [--no-estimate]
    atm estimate --config conf/experiments/smoke.yaml
    atm status   --config conf/experiments/smoke.yaml
    atm resume   --run-id <uuid> --config conf/experiments/smoke.yaml
    atm replay   --run-id <uuid> --config conf/experiments/smoke.yaml
    atm reconcile --config conf/experiments/smoke.yaml [--force-resume]

Exit codes:
    0 — completed
    1 — failed (generic error)
    2 — budget_exceeded
    3 — config/load error (ValidationError, FileNotFoundError, etc.)
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from atm.experiment.config import load_config
from atm.experiment.runner import run_one

app = typer.Typer(name="atm", no_args_is_help=True)


# ---------------------------------------------------------------------------
# run — single experiment
# ---------------------------------------------------------------------------


@app.command("run")
def run(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML"),
    ],
    override: Annotated[
        list[str] | None,
        typer.Argument(help="OmegaConf dotlist overrides like +topology.name=star"),
    ] = None,
) -> None:
    """Run a single experiment from a YAML config file."""
    # Load and validate config
    try:
        cfg = load_config(str(config), overrides=override or [])
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    # Run the experiment
    result = asyncio.run(run_one(cfg))

    typer.echo(
        f"Run {result.run_id}: "
        f"status={result.status}, "
        f"quality={result.metrics.get('quality_score', 'n/a')}, "
        f"cost=${result.metrics.get('cost_usd', 0):.4f}, "
        f"iters={result.metrics.get('iters', 0)}"
    )

    if result.status == "completed":
        raise typer.Exit(0)
    elif result.status == "budget_exceeded":
        raise typer.Exit(2)
    else:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# grid — sweep over a parameter grid
# ---------------------------------------------------------------------------


@app.command("grid")
def grid(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to grid experiment YAML"),
    ],
    parallelism: Annotated[
        int,
        typer.Option("--parallelism", "-p", help="Number of parallel workers (default: 1)"),
    ] = 1,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Abort grid on first cell failure"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts"),
    ] = False,
    no_reconcile: Annotated[
        bool,
        typer.Option(
            "--no-reconcile",
            help="Skip zombie-run reconciliation on start",
        ),
    ] = False,
    force_resume: Annotated[
        bool,
        typer.Option(
            "--force-resume",
            help=(
                "When combined with --resume-incomplete, mark orphaned 'running' runs as "
                "'failed' before resuming (no-op without --resume-incomplete)"
            ),
        ),
    ] = False,
    resume_incomplete: Annotated[
        bool,
        typer.Option(
            "--resume-incomplete",
            help=(
                "Resume any failed/running runs for this experiment that have a LangGraph "
                "checkpoint before launching new grid cells"
            ),
        ),
    ] = False,
    no_estimate: Annotated[
        bool,
        typer.Option(
            "--no-estimate",
            help="Skip cost estimation pre-flight check",
        ),
    ] = False,
) -> None:
    """Run a parameter grid sweep from a YAML config file.

    Before launching the grid, this command optionally:
      1. Reconciles zombie runs for this experiment (--no-reconcile to skip).
      2. Estimates total grid cost and confirms if over budget (--no-estimate to skip).
      3. Resumes incomplete runs from LangGraph checkpoints (--resume-incomplete).

    The --force-resume flag is a no-op without --resume-incomplete: it only controls
    whether orphaned 'running' rows are marked 'failed' before resuming.
    """
    # Load and validate config
    try:
        cfg = load_config(str(config))
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    asyncio.run(
        _grid_async(
            cfg=cfg,
            parallelism=parallelism,
            fail_fast=fail_fast,
            yes=yes,
            no_reconcile=no_reconcile,
            force_resume=force_resume,
            resume_incomplete=resume_incomplete,
            no_estimate=no_estimate,
        )
    )


async def _grid_async(
    cfg: object,
    parallelism: int,
    fail_fast: bool,
    yes: bool,
    no_reconcile: bool,
    force_resume: bool,
    resume_incomplete: bool,
    no_estimate: bool,
) -> None:
    """Async implementation of the grid command."""
    from atm.experiment.config import ExperimentConfig
    from atm.experiment.grid import run_grid  # type: ignore[import-untyped]
    from atm.experiment.reconcile import reconcile_zombies  # type: ignore[import-untyped]
    from atm.storage.session import create_engine, create_session_factory

    if not isinstance(cfg, ExperimentConfig):
        typer.echo("Internal error: config is not an ExperimentConfig", err=True)
        raise typer.Exit(1)

    pg_dsn = cfg.observability.pg_dsn

    # --- Step 1: Reconcile zombies for this experiment (unless --no-reconcile) ---
    exp_id: uuid.UUID | None = None
    if not no_reconcile:
        import sqlalchemy as sa

        from atm.storage.models import Experiment

        engine = create_engine(pg_dsn)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                result = await session.execute(
                    sa.select(Experiment.id).where(Experiment.name == cfg.name)
                )
                row = result.fetchone()
                if row is not None:
                    exp_id = row[0]
        finally:
            await engine.dispose()

        if exp_id is not None:
            engine = create_engine(pg_dsn)
            session_factory = create_session_factory(engine)
            try:
                report = await reconcile_zombies(
                    session_factory,
                    exp_id,
                    allow_force_resume=force_resume,
                )
                if report:
                    typer.echo(f"Reconcile report for experiment '{cfg.name}':")
                    typer.echo(report)
            finally:
                await engine.dispose()
        else:
            typer.echo(
                f"Experiment '{cfg.name}' not found in DB — skipping reconcile (no zombies possible)."
            )

    # --- Step 2: Estimate pre-flight (unless --no-estimate) ---
    if not no_estimate:
        from atm.experiment.estimator import estimate_grid  # type: ignore[import-untyped]

        try:
            estimate = await estimate_grid(cfg)
            total_cost = estimate.get("total_cost_usd", 0.0)
            typer.echo(f"Estimated grid cost: ${total_cost:.4f}")

            budget_limit = cfg.budget.per_experiment_usd * 1.0
            if total_cost > budget_limit and not yes:
                typer.confirm(
                    f"Estimated cost ${total_cost:.4f} exceeds budget "
                    f"${budget_limit:.4f}. Continue?",
                    abort=True,
                )
        except Exception as exc:
            typer.echo(f"Warning: cost estimation failed: {exc}", err=True)

    # --- Step 3: Resume incomplete runs (if --resume-incomplete) ---
    if resume_incomplete:
        import sqlalchemy as sa

        from atm.experiment.runner import resume_one  # type: ignore[attr-defined]
        from atm.storage.checkpointer import build_checkpointer
        from atm.storage.models import Experiment, Run

        engine = create_engine(pg_dsn)
        session_factory = create_session_factory(engine)
        try:
            # Lookup exp_id if not already found
            if exp_id is None:
                async with session_factory() as session:
                    result = await session.execute(
                        sa.select(Experiment.id).where(Experiment.name == cfg.name)
                    )
                    row = result.fetchone()
                    if row is not None:
                        exp_id = row[0]

            if exp_id is not None:
                # Ensure checkpoints table exists via build_checkpointer before querying it
                _saver, pool = await build_checkpointer(pg_dsn, max_size=1, min_size=1)
                try:
                    # Find runs with status in ('failed', 'running') for this exp_id
                    async with session_factory() as session:
                        runs_result = await session.execute(
                            sa.select(Run.id).where(
                                Run.exp_id == exp_id,
                                Run.status.in_(["failed", "running"]),
                            )
                        )
                        resumable_run_ids = [r[0] for r in runs_result.fetchall()]

                    # Filter to runs that have a checkpoint
                    if resumable_run_ids:
                        typer.echo(
                            f"Found {len(resumable_run_ids)} incomplete run(s) to resume."
                        )
                        for run_id in resumable_run_ids:
                            # Check if checkpoint exists for this run
                            has_checkpoint = False
                            async with pool.connection() as conn:
                                cp_result = await conn.execute(
                                    "SELECT 1 FROM checkpoints WHERE thread_id = %s LIMIT 1",
                                    (str(run_id),),
                                )
                                has_checkpoint = cp_result.fetchone() is not None

                            if has_checkpoint:
                                typer.echo(f"Resuming run {run_id}...")
                                try:
                                    await resume_one(run_id, cfg)
                                    typer.echo(f"Run {run_id} resumed successfully.")
                                except Exception as exc:
                                    typer.echo(
                                        f"Warning: failed to resume run {run_id}: {exc}",
                                        err=True,
                                    )
                finally:
                    await pool.close()
        finally:
            await engine.dispose()

    # --- Step 4: Run the grid ---
    await run_grid(cfg, parallelism=parallelism, fail_fast=fail_fast)


# ---------------------------------------------------------------------------
# estimate — cost estimation for a config
# ---------------------------------------------------------------------------


@app.command("estimate")
def estimate(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML"),
    ],
) -> None:
    """Estimate the cost of an experiment or grid without running it."""
    try:
        cfg = load_config(str(config))
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    async def _run() -> None:
        from atm.experiment.estimator import estimate_grid

        result = await estimate_grid(cfg)
        total_cost = result.get("total_cost_usd", 0.0)
        cell_count = result.get("cell_count", 1)
        typer.echo(f"Cells: {cell_count}, Estimated total cost: ${total_cost:.4f}")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# status — show experiment status from DB
# ---------------------------------------------------------------------------


@app.command("status")
def status(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML"),
    ],
) -> None:
    """Show the status of runs for an experiment."""
    try:
        cfg = load_config(str(config))
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    async def _run() -> None:
        import sqlalchemy as sa

        from atm.experiment.config import ExperimentConfig
        from atm.storage.models import Experiment, Run
        from atm.storage.session import create_engine, create_session_factory

        if not isinstance(cfg, ExperimentConfig):
            typer.echo("Internal error: config is not an ExperimentConfig", err=True)
            raise typer.Exit(1)

        engine = create_engine(cfg.observability.pg_dsn)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                exp_result = await session.execute(
                    sa.select(Experiment.id, Experiment.status).where(
                        Experiment.name == cfg.name
                    )
                )
                exp_row = exp_result.fetchone()
                if exp_row is None:
                    typer.echo(f"Experiment '{cfg.name}' not found in DB.")
                    return

                exp_id, exp_status = exp_row[0], exp_row[1]
                typer.echo(f"Experiment: {cfg.name}  id={exp_id}  status={exp_status}")

                runs_result = await session.execute(
                    sa.select(Run.id, Run.status, Run.topology, Run.task_id, Run.seed).where(
                        Run.exp_id == exp_id
                    )
                )
                runs = runs_result.fetchall()
                if not runs:
                    typer.echo("  No runs found.")
                    return
                for r in runs:
                    typer.echo(
                        f"  Run {r[0]}: status={r[1]}  topology={r[2]}"
                        f"  task={r[3]}  seed={r[4]}"
                    )
        finally:
            await engine.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# resume — resume a specific run from its LangGraph checkpoint
# ---------------------------------------------------------------------------


@app.command("resume")
def resume(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="UUID of the run to resume"),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML"),
    ],
) -> None:
    """Resume a single run from its last LangGraph checkpoint."""
    try:
        cfg = load_config(str(config))
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    async def _run() -> None:
        from atm.experiment.runner import resume_one  # type: ignore[attr-defined]

        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError as exc:
            typer.echo(f"Invalid run-id: {exc}", err=True)
            raise typer.Exit(1) from exc

        await resume_one(run_uuid, cfg)
        typer.echo(f"Run {run_id} resumed.")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# replay — replay a run deterministically from Parquet logs
# ---------------------------------------------------------------------------


@app.command("replay")
def replay(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="UUID of the run to replay"),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML"),
    ],
) -> None:
    """Replay a run deterministically using recorded LLM responses from Parquet logs."""
    try:
        cfg = load_config(str(config))
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    async def _run() -> None:
        from atm.experiment.runner import replay_one  # type: ignore[attr-defined]

        try:
            run_uuid = uuid.UUID(run_id)
        except ValueError as exc:
            typer.echo(f"Invalid run-id: {exc}", err=True)
            raise typer.Exit(1) from exc

        await replay_one(run_uuid, cfg)
        typer.echo(f"Run {run_id} replayed.")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# reconcile — reconcile zombie runs for an experiment
# ---------------------------------------------------------------------------


@app.command("reconcile")
def reconcile(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML"),
    ],
    force_resume: Annotated[
        bool,
        typer.Option(
            "--force-resume",
            help="Mark orphaned 'running' runs as 'failed' before reconciling",
        ),
    ] = False,
) -> None:
    """Reconcile zombie (orphaned running) runs for an experiment."""
    try:
        cfg = load_config(str(config))
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    async def _run() -> None:
        import sqlalchemy as sa

        from atm.experiment.config import ExperimentConfig
        from atm.experiment.reconcile import reconcile_zombies
        from atm.storage.models import Experiment
        from atm.storage.session import create_engine, create_session_factory

        if not isinstance(cfg, ExperimentConfig):
            typer.echo("Internal error: config is not an ExperimentConfig", err=True)
            raise typer.Exit(1)

        engine = create_engine(cfg.observability.pg_dsn)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                result = await session.execute(
                    sa.select(Experiment.id).where(Experiment.name == cfg.name)
                )
                row = result.fetchone()
                if row is None:
                    typer.echo(
                        f"Experiment '{cfg.name}' not found in DB — nothing to reconcile."
                    )
                    return
                exp_id = row[0]

            report = await reconcile_zombies(
                session_factory,
                exp_id,
                allow_force_resume=force_resume,
            )
            typer.echo(f"Reconcile report for experiment '{cfg.name}':")
            typer.echo(report or "No zombies found.")
        finally:
            await engine.dispose()

    asyncio.run(_run())
