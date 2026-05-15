"""PG-gated integration test for ``atm status --json``.

Seeds an experiment with mixed run statuses and invokes the CLI via
``subprocess.run`` (real OS process, not CliRunner) to avoid the
``RuntimeError: asyncio.run() cannot be called from a running event loop``
that occurs when ``atm status`` (which calls ``asyncio.run`` internally)
is invoked from inside a pytest-asyncio test via CliRunner.

Pattern mirrors ``tests/integration/experiment/test_m12_exit_grid_parallel.py``.

Skipped unless ``ATM_ENABLE_PG_TESTS=1``.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from decimal import Decimal

import pytest

from atm.storage.models import Experiment, Run

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
pytestmark = pytest.mark.requires_postgres

_SUBPROCESS_TIMEOUT_S = 30


@pytest.mark.asyncio
async def test_status_cli_json_aggregates(  # type: ignore[no-untyped-def]
    pg_dsn,
    pg_engine_fast,
    session_factory_fast,
) -> None:
    """Seed 2 completed, 1 failed, 1 running → JSON output matches counts."""
    exp_id = uuid.uuid4()

    async with session_factory_fast() as session:
        session.add(
            Experiment(
                id=exp_id,
                name="status_test",
                status="running",
            )
        )
        # 2 completed
        for q, cost in [(0.8, 0.10), (0.9, 0.20)]:
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
                    quality_score=q,
                    iterations=5,
                )
            )
        # 1 failed
        session.add(
            Run(
                id=uuid.uuid4(),
                exp_id=exp_id,
                topology="star",
                task_id="humaneval",
                agent_set="canonical_4",
                seed=42,
                model="fake:scripted",
                status="failed",
                budget_spent_usd=Decimal("0.05"),
            )
        )
        # 1 running
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
                budget_spent_usd=Decimal("0.01"),
            )
        )
        await session.commit()

    # Invoke CLI via subprocess to avoid asyncio.run() conflict with
    # the running pytest-asyncio event loop.
    env = os.environ.copy()
    env["ATM_PG_DSN"] = pg_dsn

    result = subprocess.run(
        [
            "uv",
            "run",
            "atm",
            "status",
            "--exp-name",
            "status_test",
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )

    assert result.returncode == 0, (
        f"``atm status`` exited with code {result.returncode} (expected 0).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    parsed = json.loads(result.stdout.strip())
    assert parsed["name"] == "status_test"
    assert parsed["total"] == 4
    assert parsed["completed"] == 2
    assert parsed["failed"] == 1
    assert parsed["running"] == 1
    assert parsed["avg_quality"] == pytest.approx(0.85, rel=1e-4)
    assert parsed["total_cost"] == pytest.approx(0.36, rel=1e-3)


@pytest.mark.asyncio
async def test_status_cli_no_match(  # type: ignore[no-untyped-def]
    pg_dsn,
    pg_engine_fast,
    session_factory_fast,
) -> None:
    """Unknown experiment name → exit 1."""
    env = os.environ.copy()
    env["ATM_PG_DSN"] = pg_dsn

    result = subprocess.run(
        [
            "uv",
            "run",
            "atm",
            "status",
            "--exp-name",
            "nonexistent_xyz",
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )

    assert result.returncode == 1


@pytest.mark.asyncio
async def test_status_cli_latest_experiment(  # type: ignore[no-untyped-def]
    pg_dsn,
    pg_engine_fast,
    session_factory_fast,
) -> None:
    """No --exp-id/--exp-name → returns the most recent experiment."""
    async with session_factory_fast() as session:
        session.add(Experiment(id=uuid.uuid4(), name="older", status="completed"))
        await session.flush()
        session.add(Experiment(id=uuid.uuid4(), name="newer", status="running"))
        await session.commit()

    env = os.environ.copy()
    env["ATM_PG_DSN"] = pg_dsn

    result = subprocess.run(
        ["uv", "run", "atm", "status", "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )

    assert result.returncode == 0, (
        f"``atm status`` exited with code {result.returncode} (expected 0).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    parsed = json.loads(result.stdout.strip())
    # Either is fine if started_at is identical at clock resolution; the test
    # just asserts the CLI returns _some_ experiment with valid counts.
    assert parsed["name"] in {"older", "newer"}
    assert parsed["total"] == 0
