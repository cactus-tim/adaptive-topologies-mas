"""Typer CLI for the ATM experiment runner.

Usage::

    atm run      --config conf/experiments/smoke.yaml [+key=val ...]
    atm grid     --config conf/experiments/sweep.yaml [--parallelism N] [--fail-fast] [--yes]
    atm estimate --config conf/experiments/grid.yaml
    atm status   --exp-id <uuid>
    atm status   --exp-name my_exp --json

Exit codes (``atm run``):
    0 — completed
    1 — failed (generic error)
    2 — budget_exceeded
    3 — config/load error (ValidationError, FileNotFoundError, etc.)

Exit codes (grid, ``atm grid``):
    0 — all cells completed
    1 — some cells failed (partial)
    2 — all cells failed
    3 — config/load error
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from sqlalchemy import Integer, case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from atm.experiment.config import load_config
from atm.experiment.estimator import GridEstimate, estimate_grid
from atm.experiment.loader import load_grid_configs
from atm.experiment.runner import _load_pricing, run_one
from atm.llm.pricing import Pricing
from atm.storage.models import Experiment, Run
from atm.storage.session import create_engine, create_session_factory

app = typer.Typer(name="atm", no_args_is_help=True)


# ---------------------------------------------------------------------------
# atm run
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
    estimate: Annotated[
        bool,
        typer.Option(
            "--estimate/--no-estimate",
            help="Print a cost estimate before running; confirm if > budget/2.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the cost-confirm prompt when --estimate is set.",
        ),
    ] = False,
) -> None:
    """Run a single experiment from a YAML config file."""
    # Load and validate config
    try:
        cfg = load_config(str(config), overrides=override or [])
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    # Optional pre-flight estimate + confirm gate
    if estimate:
        pricing = _load_pricing()
        engine, session_factory = _maybe_open_session(cfg.observability.pg_dsn)
        try:
            grid_est = asyncio.run(
                estimate_grid(
                    configs=[cfg],
                    session_factory=session_factory if cfg.estimate.use_historical else None,
                    pricing=pricing,
                    cfg=cfg.estimate,
                )
            )
        finally:
            if engine is not None:
                asyncio.run(engine.dispose())

        typer.echo(
            f"Estimate: ${grid_est.total_cost_usd:.4f} "
            f"(input={grid_est.total_input_tokens}, output={grid_est.total_output_tokens})"
        )
        threshold = cfg.budget.per_experiment_usd * 0.5
        if grid_est.total_cost_usd > threshold and not yes:
            confirmed = typer.confirm(
                f"Estimated cost ${grid_est.total_cost_usd:.4f} exceeds 50% of "
                f"per_experiment_usd budget (${cfg.budget.per_experiment_usd:.2f}). Proceed?",
                default=False,
            )
            if not confirmed:
                typer.echo("Aborted by user.", err=True)
                raise typer.Exit(3)

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
# atm estimate
# ---------------------------------------------------------------------------


@app.command("estimate")
def estimate_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML (single-cell or grid)."),
    ],
    override: Annotated[
        list[str] | None,
        typer.Argument(help="OmegaConf dotlist overrides like +topology.name=star"),
    ] = None,
) -> None:
    """Print a per-cell + grand-total cost estimate for a config / grid.

    Uses ``runs.budget_spent_usd`` averages for ``(topology, task)`` pairs
    with at least one completed historical run; falls back to the heuristic
    formula otherwise. Database access is optional — if the configured PG
    DSN is unreachable, the command degrades to heuristic-only.
    """
    try:
        configs = load_grid_configs(str(config), overrides=override or [])
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    if not configs:
        typer.echo("No configs to estimate.")
        raise typer.Exit(0)

    pricing = _load_pricing()
    base_cfg = configs[0]
    use_hist = base_cfg.estimate.use_historical

    engine: AsyncEngine | None = None
    session_factory: async_sessionmaker[Any] | None = None
    if use_hist:
        engine, session_factory = _maybe_open_session(base_cfg.observability.pg_dsn)

    try:
        grid_est = asyncio.run(
            estimate_grid(
                configs=configs,
                session_factory=session_factory,
                pricing=pricing,
                cfg=base_cfg.estimate,
            )
        )
    finally:
        if engine is not None:
            asyncio.run(engine.dispose())

    _print_estimate_table(grid_est)
    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# atm grid — M12 parallel sweep driver
# ---------------------------------------------------------------------------


@app.command("grid")
def grid(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to experiment YAML with optional grid: block"),
    ],
    parallelism: Annotated[
        int | None,
        typer.Option(
            "--parallelism",
            "-p",
            help="Max concurrent worker processes. Overrides cfg.grid.parallelism if set.",
        ),
    ] = None,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast",
            help="Stop the sweep on first failed/budget_exceeded cell.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the interactive confirmation prompt."),
    ] = False,
    override: Annotated[
        list[str] | None,
        typer.Argument(help="OmegaConf dotlist overrides like +topology.name=star"),
    ] = None,
) -> None:
    """Run an experiment grid in parallel via ProcessPoolExecutor.

    The config may declare a ``grid:`` block (handled by ``load_grid_configs``);
    if absent, the grid degenerates to a single cell.
    """
    # Step 1: load and expand configs.
    try:
        configs = load_grid_configs(str(config), overrides=override or [])
    except (ValidationError, FileNotFoundError, Exception) as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(3) from exc

    if not configs:
        typer.echo("No cells produced by load_grid_configs — empty grid?", err=True)
        raise typer.Exit(3)

    n_cells = len(configs)

    # Resolve effective parallelism: CLI flag > cfg.grid.parallelism > default 4.
    effective_parallelism: int = (
        parallelism
        if parallelism is not None
        else (configs[0].grid.parallelism if configs[0].grid is not None else 4)
    )
    effective_fail_fast: bool = fail_fast or (
        configs[0].grid.fail_fast if configs[0].grid is not None else False
    )

    typer.echo(
        f"Grid: {n_cells} cell(s), parallelism={effective_parallelism}, "
        f"fail_fast={effective_fail_fast}"
    )

    # Step 2: interactive confirm (skipped with --yes or in non-TTY).
    if not yes and sys.stdin.isatty() and not typer.confirm("Proceed?", default=True):
        typer.echo("Aborted.", err=True)
        raise typer.Exit(3)

    # Step 3: drive run_grid with live-progress callback.
    # Imported lazily so `atm grid --help` doesn't require sqlalchemy etc.
    from atm.experiment.grid import GridProgress, run_grid

    def _on_progress(p: GridProgress) -> None:
        eta = f"{p.eta_s:.0f}s" if p.eta_s is not None else "?"
        msg = (
            f"\r[{p.done}/{p.total}] done={p.done - p.failed} failed={p.failed} "
            f"in_progress={p.in_progress} eta={eta}    "
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    result = asyncio.run(
        run_grid(
            configs,
            parallelism=effective_parallelism,
            fail_fast=effective_fail_fast,
            progress_callback=_on_progress,
        )
    )

    # Final newline after the \r-overwrite progress line.
    sys.stdout.write("\n")
    sys.stdout.flush()

    typer.echo(
        f"GridResult: exp_id={result.exp_id} "
        f"total={result.total} completed={result.completed} "
        f"failed={result.failed} budget_exceeded={result.budget_exceeded}"
    )

    # Exit code: 0=all completed, 1=partial, 2=all failed, 3 already handled.
    failed_total = result.failed + result.budget_exceeded
    if result.completed == result.total:
        raise typer.Exit(0)
    if failed_total == result.total:
        raise typer.Exit(2)
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# atm status
# ---------------------------------------------------------------------------


@app.command("status")
def status_cmd(
    exp_id: Annotated[
        str | None,
        typer.Option("--exp-id", help="Experiment UUID."),
    ] = None,
    exp_name: Annotated[
        str | None,
        typer.Option("--exp-name", help="Experiment name (unique)."),
    ] = None,
    pg_dsn: Annotated[
        str | None,
        typer.Option(
            "--pg-dsn",
            help="PostgreSQL DSN; falls back to ATM_PG_DSN env var, then default.",
            envvar="ATM_PG_DSN",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a formatted table."),
    ] = False,
) -> None:
    """Print aggregated status of an experiment (cells done/failed/running, cost, ETA)."""
    if not pg_dsn:
        typer.echo(
            "Config error: --pg-dsn is required (or set ATM_PG_DSN).",
            err=True,
        )
        raise typer.Exit(3)

    # Validate --exp-id as UUID early to avoid leaking DB error messages back
    # to the user. UUID parsing is purely string-based; no DB call yet.
    if exp_id is not None:
        try:
            uuid.UUID(exp_id)
        except ValueError as exc:
            typer.echo(f"Config error: --exp-id must be a valid UUID ({exc})", err=True)
            raise typer.Exit(3) from exc

    engine = create_engine(pg_dsn)
    factory = create_session_factory(engine)

    try:
        row = asyncio.run(_query_status(factory, exp_id=exp_id, exp_name=exp_name))
    finally:
        asyncio.run(engine.dispose())

    if row is None:
        typer.echo("No matching experiment found.", err=True)
        raise typer.Exit(1)

    if as_json:
        typer.echo(json.dumps(row, default=str))
    else:
        _print_status_table(row)

    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _maybe_open_session(
    pg_dsn: str,
) -> tuple[AsyncEngine | None, async_sessionmaker[Any] | None]:
    """Open (engine, session_factory) or return ``(None, None)`` on failure.

    The estimator's historical path is best-effort: if the DSN is malformed
    or the server is unreachable we still want ``atm estimate`` to succeed
    with the heuristic fallback, so failures are swallowed here.
    """
    try:
        engine = create_engine(pg_dsn)
        factory = create_session_factory(engine)
        return engine, factory
    except Exception:  # pragma: no cover — defensive
        return None, None


async def _query_status(
    factory: async_sessionmaker[Any],
    *,
    exp_id: str | None,
    exp_name: str | None,
) -> dict[str, Any] | None:
    """Run the aggregated SELECT and return a plain dict (or None)."""
    async with factory() as session:
        # Resolve the experiment row first (single experiment).
        exp_stmt = select(Experiment)
        if exp_id is not None:
            exp_stmt = exp_stmt.where(Experiment.id == exp_id)
        elif exp_name is not None:
            exp_stmt = exp_stmt.where(Experiment.name == exp_name)
        else:
            exp_stmt = exp_stmt.order_by(desc(Experiment.started_at)).limit(1)

        exp = (await session.execute(exp_stmt)).scalar_one_or_none()
        if exp is None:
            return None

        # Aggregate over runs. We use ``case`` expressions (parameterized,
        # no string interpolation) to count rows per status.
        completed_expr = func.sum(case((Run.status == "completed", 1), else_=0)).label("completed")
        failed_expr = func.sum(case((Run.status == "failed", 1), else_=0)).label("failed")
        running_expr = func.sum(case((Run.status == "running", 1), else_=0)).label("running")
        agg_stmt = select(
            func.count(Run.id).label("total"),
            func.coalesce(completed_expr, 0).label("completed_coalesced"),
            func.coalesce(failed_expr, 0).label("failed_coalesced"),
            func.coalesce(running_expr, 0).label("running_coalesced"),
            func.avg(Run.quality_score).label("avg_quality"),
            func.coalesce(func.sum(Run.budget_spent_usd), 0).label("total_cost"),
        ).where(Run.exp_id == exp.id)
        agg = (await session.execute(agg_stmt)).one()

    return {
        "id": str(exp.id),
        "name": exp.name,
        "status": exp.status,
        "started_at": exp.started_at,
        "finished_at": exp.finished_at,
        "total": int(agg.total or 0),
        "completed": int(agg.completed_coalesced or 0),
        "failed": int(agg.failed_coalesced or 0),
        "running": int(agg.running_coalesced or 0),
        "avg_quality": float(agg.avg_quality) if agg.avg_quality is not None else None,
        "total_cost": float(agg.total_cost) if agg.total_cost is not None else 0.0,
    }


def _print_estimate_table(grid_est: GridEstimate) -> None:
    """Pretty-print a GridEstimate to stdout via typer.echo."""
    typer.echo(
        f"{'topology':<14}{'task':<24}{'seed':>6}  "
        f"{'tokens_in':>10}{'tokens_out':>12}  {'cost_usd':>10}  source"
    )
    typer.echo("-" * 88)
    for c in grid_est.cells:
        typer.echo(
            f"{c.topology:<14}{c.task:<24}{c.seed:>6}  "
            f"{c.est_input_tokens:>10}{c.est_output_tokens:>12}  "
            f"${c.est_cost_usd:>9.4f}  {c.source}"
        )
    typer.echo("-" * 88)
    typer.echo(f"TOTAL{' ' * 53}${grid_est.total_cost_usd:>9.4f}")
    if grid_est.per_topology:
        typer.echo("")
        typer.echo("Per topology:")
        for topo, cost in sorted(grid_est.per_topology.items()):
            typer.echo(f"  {topo:<14}${cost:.4f}")


def _print_status_table(row: dict[str, Any]) -> None:
    """Pretty-print the status row to stdout via typer.echo."""
    typer.echo(f"Experiment: {row['name']} ({row['id']})")
    typer.echo(f"  status     : {row['status']}")
    typer.echo(f"  started_at : {row['started_at']}")
    typer.echo(f"  finished_at: {row['finished_at']}")
    typer.echo(f"  total cells: {row['total']}")
    typer.echo(
        f"  completed  : {row['completed']}  failed: {row['failed']}  running: {row['running']}"
    )
    if row["avg_quality"] is not None:
        typer.echo(f"  avg quality: {row['avg_quality']:.4f}")
    else:
        typer.echo("  avg quality: n/a")
    typer.echo(f"  total cost : ${row['total_cost']:.4f}")


__all__ = ["Integer", "Pricing", "app"]
