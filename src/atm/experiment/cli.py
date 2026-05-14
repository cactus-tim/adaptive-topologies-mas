"""Typer CLI for the ATM experiment runner.

Usage:
    atm run  --config conf/experiments/smoke.yaml [+key=val ...]
    atm grid --config conf/experiments/sweep.yaml [--parallelism N] [--fail-fast] [--yes]

Exit codes (single-run, ``atm run``):
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
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from atm.experiment.config import load_config
from atm.experiment.loader import load_grid_configs
from atm.experiment.runner import run_one

app = typer.Typer(name="atm", no_args_is_help=True)


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
    effective_fail_fast: bool = (
        fail_fast or (configs[0].grid.fail_fast if configs[0].grid is not None else False)
    )

    typer.echo(
        f"Grid: {n_cells} cell(s), parallelism={effective_parallelism}, "
        f"fail_fast={effective_fail_fast}"
    )

    # Step 2: interactive confirm (skipped with --yes or in non-TTY).
    if not yes and sys.stdin.isatty():
        if not typer.confirm("Proceed?", default=True):
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
