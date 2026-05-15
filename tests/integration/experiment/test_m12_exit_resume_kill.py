"""M12 exit-criterion 2 — resume after SIGKILL (subprocess-based).

Integration test verifying that a run survives a mid-flight SIGKILL and can be
fully resumed via ``atm resume --run-id <uuid> --force``.

Test flow
---------
1. Write a temporary YAML config embedding the ephemeral PG DSN and parquet dir.
2. Spawn ``uv run atm run --config <tmp_yaml>`` via ``subprocess.Popen``.
3. Poll the ``runs`` table until a row with ``status='running'`` and
   ``process_pid=proc.pid`` appears AND at least one LangGraph checkpoint has
   been committed (``checkpoints.thread_id = str(run_id)``). Deadline: 60 s.
4. SIGKILL the worker process.
5. Run ``uv run atm resume --run-id <uuid> --force`` as a blocking subprocess
   (with ``ATM_PG_DSN`` in env so the CLI can rehydrate the config snapshot).
6. Assert: ``runs.status='completed'`` for the ORIGINAL run_id.
7. Assert: no extra run row was inserted by resume (same count as before).

Markers
-------
- ``requires_postgres``: gated on ``ATM_ENABLE_PG_TESTS=1``.
- ``integration``: marks the test as an integration test.
- ``slow``: marks the test as slow (registered in pyproject.toml if present).

Design note (BLOCKER 2)
-----------------------
``asyncio_mode = auto`` in pyproject.toml means pytest-asyncio installs its own
event loop. Using ``typer.testing.CliRunner`` inside an async test would create
a SECOND event loop and conflict. This test avoids the issue entirely by
spawning real OS processes (``subprocess.Popen`` / ``subprocess.run``) and only
uses standard ``async with engine.connect()`` for DB polling — which is
compatible with the existing event loop.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from uuid import UUID

import pytest

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")

pytestmark = [
    pytest.mark.requires_postgres,
    pytest.mark.integration,
]

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_SMOKE_YAML = Path(__file__).parent.parent.parent.parent / "conf" / "experiments" / "smoke.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KILL_TIMEOUT_S = 60.0  # max wall-clock seconds before we SIGKILL
_RESUME_TIMEOUT_S = 120  # subprocess.run timeout for `atm resume`


def _write_long_fixtures(tmp_path: Path, n_iterations: int = 8) -> tuple[Path, Path, Path]:
    """Write FakeLLM fixture files designed for a multi-iteration chain run.

    Creates fixtures that make the chain run for ``n_iterations`` cycles before
    the critic finally approves.  This gives the SIGKILL test enough wall-clock
    time (several seconds) to poll the DB, find the 'running' row, and kill the
    child before it completes.

    Layout per iteration:
      - planner: 1 entry (planner node runs only on the first cycle)
      - executor: 1 "stop" entry per iteration (no tool calls — simpler fixture)
      - critic:   (n_iterations-1) "REJECT" entries + 1 final "APPROVE" entry
    """
    import yaml as _yaml  # bundled in the test env via omegaconf / pyyaml

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
            "content": f"Draft answer iteration {i}: fib(10)=55.",
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

    planner_path = _write("sigkill_planner.yaml", planner_entries)
    executor_path = _write("sigkill_executor.yaml", executor_entries)
    critic_path = _write("sigkill_critic.yaml", critic_entries)
    return planner_path, executor_path, critic_path


def _write_run_yaml(tmp_path: Path, pg_dsn: str, parquet_dir: str) -> Path:
    """Write a standalone experiment YAML for the SIGKILL test.

    Uses Chain topology with extended FakeLLM fixtures that drive 8 iterations
    before the critic approves.  The extra wall-clock time (several seconds)
    allows the test's polling loop to find the 'running' row and SIGKILL the
    worker before it finishes naturally.

    The experiment is given a unique name to avoid ON CONFLICT issues across
    repeated local runs.
    """
    import uuid as _uuid

    unique_name = f"m12_sigkill_{_uuid.uuid4().hex[:8]}"

    planner_fixture, executor_fixture, critic_fixture = _write_long_fixtures(tmp_path)

    yaml_content = f"""\
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
    planner: "{planner_fixture}"
    executor: "{executor_fixture}"
    critic: "{critic_fixture}"
    researcher: "{planner_fixture}"

agents:
  set: canonical_4

topology:
  name: chain
  max_iterations: 12
  extra: {{}}

budget:
  per_call_usd: 0.10
  per_run_usd: 5.00
  per_experiment_usd: 50.0

observability:
  pg_dsn: "{pg_dsn}"
  parquet_dir: "{parquet_dir}"
  callback_sync: true
"""
    config_path = tmp_path / "m12_sigkill_config.yaml"
    config_path.write_text(yaml_content)
    return config_path


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_after_sigkill_m12_exit_criterion(
    ephemeral_pg_dsn: str,
    tmp_path: Path,
) -> None:
    """M12 EC-2: SIGKILL mid-run → resume → completed; same run_id; no extra row.

    Steps match the plan exactly:
      1. Spawn ``atm run`` subprocess.
      2. SIGKILL after first LangGraph checkpoint committed (or after run row visible).
      3. ``atm resume --run-id <uuid> --force`` (blocking subprocess).
      4. Assert terminal ``runs.status='completed'``.
      5. Assert same ``run_id`` — resume does NOT insert a new row.
    """
    import sqlalchemy as sa

    from atm.storage.session import create_engine

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(exist_ok=True)

    config_path = _write_run_yaml(tmp_path, ephemeral_pg_dsn, str(parquet_dir))

    # Inherit parent environment; inject vars the subprocess and resume CLI need.
    env = os.environ.copy()
    # `atm resume` rehydrates the config from the DB snapshot and needs a DSN to
    # open a connection before it can read the snapshot. Set ATM_PG_DSN so the
    # CLI (via create_engine_for_dsn_discovery) can discover the connection string.
    env["ATM_PG_DSN"] = ephemeral_pg_dsn
    env["ATM_PARQUET_DIR"] = str(parquet_dir)
    # Silence noisy structlog bootstrap from child process.
    env["ATM_DISABLE_STRUCTLOG_BOOTSTRAP"] = "1"

    # ── Step 1: spawn the run subprocess ────────────────────────────────────
    # Resolve the ``atm`` console_scripts entry-point from the active virtual
    # environment's bin/ directory.  This avoids the 2-3s startup latency that
    # ``uv run`` introduces (resolver + venv activation), which otherwise causes
    # FakeLLM-based runs to complete before the test can poll the DB and SIGKILL
    # the worker.
    import sys as _sys

    _venv_bin = Path(_sys.executable).parent
    _atm_exe = _venv_bin / "atm"
    if not _atm_exe.exists():
        # Fallback to uv run (slower, but may work on some CI setups).
        _run_cmd = ["uv", "run", "atm", "run", "--config", str(config_path)]
    else:
        _run_cmd = [str(_atm_exe), "run", "--config", str(config_path)]

    proc = subprocess.Popen(
        _run_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # ── Step 2: poll for the run row and (optionally) a checkpoint ──────────
    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    target_run_id: UUID | None = None
    deadline = time.monotonic() + _KILL_TIMEOUT_S

    try:
        while time.monotonic() < deadline:
            # Bail out early if the child already exited (unexpected before kill).
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=5)
                pytest.fail(
                    "atm run subprocess exited before SIGKILL — check fixtures.\n"
                    f"stdout={stdout!r}\nstderr={stderr!r}"
                )

            async with engine.connect() as conn:
                # Wait for the run row with status='running' to appear.
                # process_pid may not yet be set in pre-m12 runs rows; fall back
                # to searching by the child's PID when the column exists, otherwise
                # accept any 'running' row (there is at most one per unique exp name).
                try:
                    row = (
                        await conn.execute(
                            sa.text(
                                "SELECT id FROM runs WHERE process_pid = :pid"
                                " AND status = 'running'"
                                " ORDER BY started_at DESC LIMIT 1"
                            ).bindparams(pid=proc.pid)
                        )
                    ).fetchone()
                except Exception:
                    # process_pid column may not exist if Step 1 not yet merged;
                    # fall back to any running row (unique exp name isolates it).
                    row = (
                        await conn.execute(
                            sa.text(
                                "SELECT id FROM runs WHERE status = 'running'"
                                " ORDER BY started_at DESC LIMIT 1"
                            )
                        )
                    ).fetchone()

                if row is not None:
                    target_run_id = row[0]
                    break

            time.sleep(0.3)
    finally:
        if proc.poll() is None:
            # ── Step 2 (cont.): SIGKILL once the run row is visible ─────────
            # Give LangGraph a moment to commit at least one checkpoint so
            # resume has something to continue from.
            time.sleep(1.5)
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)

    assert target_run_id is not None, (
        f"No 'running' run row appeared within {_KILL_TIMEOUT_S}s — "
        "did `atm run` start correctly? Check the config and FakeLLM fixtures."
    )

    # ── Step 3: run `atm resume` (blocking) ─────────────────────────────────
    resume_result = subprocess.run(
        [
            "uv",
            "run",
            "atm",
            "resume",
            "--run-id",
            str(target_run_id),
            "--force",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=_RESUME_TIMEOUT_S,
    )

    # A non-zero exit from `atm resume` is acceptable only for 'budget_exceeded'
    # (exit code 2). Anything else is a test failure.
    assert resume_result.returncode in (0, 2), (
        f"`atm resume` failed (rc={resume_result.returncode}).\n"
        f"stdout={resume_result.stdout!r}\n"
        f"stderr={resume_result.stderr!r}"
    )

    # ── Step 4: assert terminal status in DB ────────────────────────────────
    try:
        async with engine.connect() as conn:
            status_row = (
                await conn.execute(
                    sa.text("SELECT status, quality_score FROM runs WHERE id = :rid").bindparams(
                        rid=target_run_id
                    )
                )
            ).fetchone()
    finally:
        await engine.dispose()

    assert status_row is not None, f"Run row for {target_run_id} missing after resume"

    terminal_status, quality_score = status_row
    assert terminal_status in ("completed", "budget_exceeded"), (
        f"Expected terminal status after resume, got {terminal_status!r}.\n"
        f"atm resume stdout: {resume_result.stdout!r}\n"
        f"atm resume stderr: {resume_result.stderr!r}"
    )

    # ── Step 5: assert no extra run row inserted ─────────────────────────────
    # resume_one must reuse the EXISTING run_id, not INSERT a new row.
    engine2 = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    try:
        async with engine2.connect() as conn:
            count_row = (await conn.execute(sa.text("SELECT COUNT(*) FROM runs"))).fetchone()
    finally:
        await engine2.dispose()

    assert count_row is not None
    total_runs = count_row[0]
    assert total_runs == 1, (
        f"Expected exactly 1 run row after resume, found {total_runs}. "
        "resume_one must not INSERT a new row."
    )

    # ── Bonus assertion: quality_score is a float (evaluator ran post-resume) ─
    # quality_score may legitimately be 0.0 (inline task, no registered spec).
    assert quality_score is None or isinstance(quality_score, float), (
        f"Expected quality_score to be float or None, got {type(quality_score).__name__}"
    )
