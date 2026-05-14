"""Typer CLI for the ATM experiment runner.

Usage:
    atm run --config conf/experiments/smoke.yaml [+key=val ...]
    atm resume --run-id <uuid> [--force]
    atm replay <run_id> [--mode deterministic|semantic] [--output-config-only]
    atm reconcile --exp-id <uuid> [--dry-run]

Exit codes:
    0 — completed
    1 — failed (generic error)
    2 — budget_exceeded
    3 — config/load error (ValidationError, FileNotFoundError, etc.)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer
from pydantic import ValidationError

from atm.experiment.config import load_config
from atm.experiment.runner import replay_one, resume_one, run_one

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
# m12-resume-replay: atm resume / atm replay / atm reconcile
# ---------------------------------------------------------------------------


@app.command("resume")
def resume(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="UUID of the run row to resume"),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Skip the live-pid check on the original worker.",
        ),
    ] = False,
) -> None:
    """Resume a run from its last LangGraph PG checkpoint.

    The run's ExperimentConfig is rehydrated from ``experiments.config_snapshot``;
    LangGraph continues from the latest checkpointed state under
    ``thread_id=str(run_id)``.
    """
    try:
        rid = UUID(run_id)
    except ValueError as exc:
        typer.echo(f"Invalid run_id (must be UUID): {exc}", err=True)
        raise typer.Exit(3) from exc

    try:
        result = asyncio.run(resume_one(rid, force=force))
    except LookupError as exc:
        typer.echo(f"Resume error: {exc}", err=True)
        raise typer.Exit(3) from exc
    except RuntimeError as exc:
        typer.echo(f"Resume blocked: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        f"Resumed run {result.run_id}: status={result.status}, "
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


@app.command("replay")
def replay(
    run_id: Annotated[
        str,
        typer.Argument(help="UUID of the original run to replay"),
    ],
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Replay mode: 'deterministic' (FakeLLM from parquet) or 'semantic'.",
        ),
    ] = "deterministic",
    output_config_only: Annotated[
        bool,
        typer.Option(
            "--output-config-only",
            help="Print the rehydrated cfg JSON and exit (no run).",
        ),
    ] = False,
) -> None:
    """Replay an existing run deterministically (or semantically).

    A NEW run row is inserted with ``replay_of=<original_run_id>``. The new
    run lives in the same experiment as the original.
    """
    if mode not in ("deterministic", "semantic"):
        typer.echo(f"Invalid --mode {mode!r} (use deterministic|semantic)", err=True)
        raise typer.Exit(3)

    try:
        rid = UUID(run_id)
    except ValueError as exc:
        typer.echo(f"Invalid run_id (must be UUID): {exc}", err=True)
        raise typer.Exit(3) from exc

    if output_config_only:
        # Lightweight path — just rehydrate cfg from snapshot and print.
        async def _dump() -> str:
            # Without cfg we don't have a DSN. Require ATM_PG_DSN env or fail
            # gracefully — exposing the typer error.
            import os

            from atm.experiment.runner import (
                _fetch_experiment_snapshot,
                _fetch_run_row,
                _load_cfg_from_snapshot,
                create_engine_for_dsn_discovery,
            )

            dsn_env = os.environ.get("ATM_PG_DSN")
            if dsn_env is None:
                raise RuntimeError("--output-config-only requires ATM_PG_DSN env variable.")
            dsn: str = dsn_env
            # Build a minimal cfg shim purely for DSN-discovery.
            from atm.experiment.config import ObservabilityCfg

            class _Shim:
                observability = ObservabilityCfg(pg_dsn=dsn, parquet_dir="/tmp")

            bootstrap = create_engine_for_dsn_discovery(_Shim())  # type: ignore[arg-type]
            assert bootstrap is not None
            engine, session_factory = bootstrap
            try:
                run_row = await _fetch_run_row(session_factory, rid)
                snap = await _fetch_experiment_snapshot(session_factory, run_row["exp_id"])
                cfg = _load_cfg_from_snapshot(snap)
                return cfg.model_dump_json(indent=2)
            finally:
                await engine.dispose()

        try:
            payload = asyncio.run(_dump())
        except Exception as exc:
            typer.echo(f"Replay config dump failed: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(payload)
        raise typer.Exit(0)

    try:
        result = asyncio.run(replay_one(rid, mode=mode))  # type: ignore[arg-type]
    except LookupError as exc:
        typer.echo(f"Replay error: {exc}", err=True)
        raise typer.Exit(3) from exc
    except FileNotFoundError as exc:
        typer.echo(f"Replay error: {exc}", err=True)
        raise typer.Exit(3) from exc
    except ValueError as exc:
        typer.echo(f"Replay error: {exc}", err=True)
        raise typer.Exit(3) from exc

    typer.echo(
        f"Replayed run {result.run_id} (from {rid}): "
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


@app.command("reconcile")
def reconcile(
    exp_id: Annotated[
        str,
        typer.Option("--exp-id", help="UUID of the experiment to reconcile."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Classify zombies but do not mutate the DB."),
    ] = False,
) -> None:
    """Scan an experiment for zombie runs and (optionally) mark them failed.

    A run is a zombie when ``runs.status='running'`` but the owning OS process
    is no longer alive (or unreachable). Default mutates rows to
    ``status='failed', finish_reason='zombie'``. ``--dry-run`` only classifies
    and prints; nothing is written.
    """
    import os

    try:
        eid = UUID(exp_id)
    except ValueError as exc:
        typer.echo(f"Invalid exp_id (must be UUID): {exc}", err=True)
        raise typer.Exit(3) from exc

    dsn = os.environ.get("ATM_PG_DSN")
    if dsn is None:
        typer.echo(
            "atm reconcile requires ATM_PG_DSN env variable "
            "(no YAML config is loaded for this command).",
            err=True,
        )
        raise typer.Exit(3)

    async def _go() -> dict[str, object]:
        from atm.experiment.reconcile import reconcile_zombies
        from atm.storage.session import create_engine, create_session_factory

        engine = create_engine(dsn)
        session_factory = create_session_factory(engine)
        try:
            report = await reconcile_zombies(session_factory, eid, allow_force_resume=dry_run)
            return {
                "scanned": report.scanned,
                "zombies": [
                    {
                        "run_id": str(z.run_id),
                        "host": z.host,
                        "pid": z.pid,
                        "reason": z.reason,
                    }
                    for z in report.zombies
                ],
                "actions": {str(k): v for k, v in report.actions.items()},
            }
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(_go())
    except Exception as exc:
        typer.echo(f"Reconcile failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(0)
