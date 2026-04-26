"""Typer CLI for the ATM experiment runner.

Usage:
    atm run --config conf/experiments/smoke.yaml [+key=val ...]

Exit codes:
    0 — completed
    1 — failed (generic error)
    2 — budget_exceeded
    3 — config/load error (ValidationError, FileNotFoundError, etc.)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from atm.experiment.config import load_config
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
