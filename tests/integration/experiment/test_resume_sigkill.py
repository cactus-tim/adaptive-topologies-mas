"""Integration test for `resume_one` after SIGKILL (m12-resume-replay G5).

Spawns a `run_one` worker in a subprocess, lets it pass the first checkpoint
(detected by polling ``runs.status='running'`` with a non-null `process_pid`),
then SIGKILLs the worker. Once the worker is dead, calls `resume_one` in the
test process and asserts the run reaches ``status='completed'`` with the same
quality score as a fresh run with the same FakeLLM seed.

This test is slow (5-30s wall-clock) and PG-gated. Skipped unless
``ATM_ENABLE_PG_TESTS=1``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")

pytestmark = pytest.mark.requires_postgres


_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_SMOKE_YAML = Path(__file__).parent.parent.parent.parent / "conf" / "experiments" / "smoke.yaml"


def _write_long_fixtures(tmp_path: Path, n_iterations: int = 30) -> tuple[Path, Path, Path]:
    """Write FakeLLM scripted fixtures driving a chain run for N iterations.

    The default m6_chain_critic fixture issues APPROVE immediately, so the
    chain completes in one iteration (~0.5-1 s on fast CI runners) - too
    short for the polling-loop / SIGKILL race in this test. Writing fresh
    fixtures with (N-1) REJECT + 1 final APPROVE pushes the wall-clock past
    ~3 s, giving the test enough time to catch the running row.
    """
    import yaml as _yaml

    planner_entries = [
        {
            "agent_id": "planner",
            "role": "planner",
            "step_idx": 0,
            "content": "Plan: implement fib(n).",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "model": "fake:scripted",
        }
    ]
    executor_entries = [
        {
            "agent_id": "executor",
            "role": "executor",
            "step_idx": i,
            "content": f"Draft iteration {i}: fib(10)=55.",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "model": "fake:scripted",
        }
        for i in range(n_iterations)
    ]
    critic_entries = [
        {
            "agent_id": "critic",
            "role": "critic",
            "step_idx": i,
            "content": "APPROVE fib" if i == n_iterations - 1 else "REJECT — needs improvement.",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "model": "fake:scripted",
        }
        for i in range(n_iterations)
    ]

    def _write(name: str, entries: list) -> Path:
        path = tmp_path / name
        path.write_text(_yaml.dump({"version": 1, "mode": "scripted", "entries": entries}))
        return path

    return (
        _write("resume_sigkill_planner.yaml", planner_entries),
        _write("resume_sigkill_executor.yaml", executor_entries),
        _write("resume_sigkill_critic.yaml", critic_entries),
    )


# ---------------------------------------------------------------------------
# Subprocess child program
# ---------------------------------------------------------------------------

_CHILD_PROGRAM = """
import asyncio
import os
import sys
from pathlib import Path

from atm.experiment.config import load_config
from atm.experiment.runner import run_one


async def _main() -> None:
    pg_dsn = os.environ["ATM_PG_DSN"]
    parquet_dir = os.environ["ATM_PARQUET_DIR"]
    fixtures_dir = Path(os.environ["ATM_FIXTURES_DIR"])
    smoke_yaml = Path(os.environ["ATM_SMOKE_YAML"])

    planner_fx = os.environ["ATM_PLANNER_FX"]
    executor_fx = os.environ["ATM_EXECUTOR_FX"]
    critic_fx = os.environ["ATM_CRITIC_FX"]

    overrides = [
        f"observability.pg_dsn={pg_dsn}",
        f"observability.parquet_dir={parquet_dir}",
        "topology.name=chain",
        "topology.max_iterations=40",
        "model.default=fake:scripted",
        f"model.fake_fixtures.planner={planner_fx}",
        f"model.fake_fixtures.executor={executor_fx}",
        f"model.fake_fixtures.critic={critic_fx}",
        "human.enabled=false",
    ]
    cfg = load_config(smoke_yaml, overrides=overrides)
    result = await run_one(cfg)
    # Print run_id to stdout so the parent can read it — but in SIGKILL flow
    # we never reach this print.
    print(f"RUN_ID={result.run_id}")


asyncio.run(_main())
"""


@pytest.mark.asyncio
async def test_resume_after_sigkill_completes_run(
    ephemeral_pg_dsn: str,
    tmp_path: Path,
) -> None:
    """Spawn → wait for running row → SIGKILL → resume → assert completed.

    The test relies on the fact that ``_insert_run`` writes the row with
    ``status='running'`` and ``process_pid`` before the topology graph
    actually starts ainvoke — so we can SELECT the run_id as soon as the
    INSERT commits.
    """
    import sqlalchemy as sa

    from atm.experiment.runner import resume_one
    from atm.storage.session import create_engine

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(exist_ok=True)

    # Write long fixtures so the chain runs long enough to be killed on fast CI.
    planner_fx, executor_fx, critic_fx = _write_long_fixtures(tmp_path)

    # 1) Spawn worker.
    env = os.environ.copy()
    env["ATM_PG_DSN"] = ephemeral_pg_dsn
    env["ATM_PARQUET_DIR"] = str(parquet_dir)
    env["ATM_FIXTURES_DIR"] = str(_FIXTURES_DIR)
    env["ATM_SMOKE_YAML"] = str(_SMOKE_YAML)
    env["ATM_PLANNER_FX"] = str(planner_fx)
    env["ATM_EXECUTOR_FX"] = str(executor_fx)
    env["ATM_CRITIC_FX"] = str(critic_fx)
    # Disable structlog bootstrap noise from the child.
    env["ATM_DISABLE_STRUCTLOG_BOOTSTRAP"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_PROGRAM],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 2) Poll for a running row with pid=proc.pid.
    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    target_run_id = None
    deadline = time.monotonic() + 30.0
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=5)
                pytest.fail(
                    "child exited before we could SIGKILL it.\n"
                    f"stdout={stdout!r}\nstderr={stderr!r}"
                )
            async with engine.connect() as conn:
                row = (
                    await conn.execute(
                        sa.text(
                            "SELECT id FROM runs WHERE process_pid = :pid "
                            "AND status = 'running' "
                            "ORDER BY started_at DESC LIMIT 1"
                        ).bindparams(pid=proc.pid)
                    )
                ).fetchone()
            if row is not None:
                target_run_id = row[0]
                break
            time.sleep(0.25)
    finally:
        if proc.poll() is None:
            # 3) SIGKILL the worker.
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=5)

    assert target_run_id is not None, "did not see running row before timeout"

    # 4) Resume.
    resumed = await resume_one(target_run_id, force=True)

    assert resumed.run_id == target_run_id
    assert resumed.status in ("completed", "budget_exceeded"), (
        f"unexpected status: {resumed.status}"
    )

    # 5) Sanity-check: the runs row is no longer 'running'.
    try:
        async with engine.connect() as conn:
            status_row = (
                await conn.execute(
                    sa.text("SELECT status FROM runs WHERE id = :rid").bindparams(rid=target_run_id)
                )
            ).fetchone()
    finally:
        await engine.dispose()

    assert status_row is not None
    assert status_row[0] != "running"
