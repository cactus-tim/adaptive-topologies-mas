# m12-grid-integration — Context

## SESSION PROGRESS (2026-05-14)

### COMPLETED
- (nothing yet)

### IN PROGRESS
- Not started

### BLOCKERS
- None

## Quick Resume

1. Read this file
2. Check `m12-grid-integration-tasks.md` for what's next
3. Read `m12-grid-integration-plan.md` Phase 1 for strategy
4. Start with: Add four CLI flags to `grid()` in `src/atm/experiment/cli.py`

## Key Files

**`src/atm/experiment/cli.py`**
- Role: Typer CLI application; contains `atm grid` command (lines 207-313), `atm reconcile` (lines 539-605), and all other `atm` subcommands
- Planned change: Add `--no-reconcile`, `--force-resume`, `--resume-incomplete`, `--no-estimate` flags; insert three pre-flight behaviours (reconcile-on-start, estimate, resume-incomplete) between lines 268 and 271; add `_collect_resume_targets` private async helper
- Status: NOT STARTED

**`tests/integration/experiment/test_m12_exit_grid_parallel.py`**
- Role: NEW — exit criterion 1 acceptance test (parallel grid execution)
- Planned change: Create file; subprocess invocation of `atm grid` with 4-cell mini grid; assert 4 completed rows and wall-time < 60s
- Status: NOT STARTED

**`tests/integration/experiment/test_m12_exit_resume_kill.py`**
- Role: NEW — exit criterion 2 acceptance test (resume after SIGKILL)
- Planned change: Create file; spawn `atm run` subprocess, poll for running row, SIGKILL, then `atm resume --run-id <uuid> --force`, assert in-place update and quality parity
- Status: NOT STARTED

**`tests/integration/experiment/test_m12_exit_replay_bitwise.py`**
- Role: NEW — exit criterion 3 acceptance test (bitwise replay determinism)
- Planned change: Create file; run `run_one` in-process, replay via `atm replay` subprocess, assert `replay_of` FK, `final_answer` identity, parquet equal modulo `[run_id, message_id, at]`
- Status: NOT STARTED

**`tests/integration/experiment/test_m12_exit_reconcile.py`**
- Role: NEW — exit criterion 4 acceptance test (zombie reconcile-on-start)
- Planned change: Create file; seed experiment + zombie row via `_ensure_experiment`/`_insert_run` (NOT raw SQL); invoke `atm grid`; assert zombie `status='failed', finish_reason='zombie'` and 4 fresh cells completed
- Status: NOT STARTED

**`README.md`**
- Role: Project documentation (bilingual: Russian narrative, English code blocks)
- Planned change: Add CLI section with seven command descriptions and worked grid YAML + command sequence; update status note to M12 complete
- Status: NOT STARTED

## Decisions

**Reconcile scope resolution**
- Decision: Look up existing experiment row by `cfg.name` (unique constraint) before calling `reconcile_zombies`. If row exists, reconcile its runs. If not (first launch), skip reconcile entirely.
- Rationale: `reconcile_zombies` requires `exp_id`, but `run_grid` creates the experiment row. This approach keeps reconcile per-experiment without restructuring `run_grid`. Name uniqueness is enforced by `_ensure_experiment`'s ON CONFLICT DO NOTHING logic.

**Subprocess invocation for all integration tests**
- Decision: All four exit-criterion tests use `["uv", "run", "atm", "<command>", ...]` subprocess calls, NOT `CliRunner`.
- Rationale: `pyproject.toml` declares `asyncio_mode = "auto"`. `CliRunner.invoke` calls `asyncio.run(run_grid(...))` synchronously, which on Python 3.11 raises `RuntimeError: asyncio.run() cannot be called from a running event loop` when invoked from inside a pytest-asyncio-managed test. `src/atm/__main__.py` does not exist, so `python -m atm` is also unavailable; `uv run atm` is the correct entry point.

**Checkpoint table DDL guarantee**
- Decision: Call `build_checkpointer(dsn, max_size=1, min_size=1)` and immediately close the returned pool before any SELECT against the `checkpoints` table in the resume-incomplete branch.
- Rationale: The `checkpoints` table is NOT in ATM's SQLAlchemy `Base.metadata`. `ephemeral_pg_dsn` schemas do not include LangGraph tables. `build_checkpointer` invokes `saver.setup()` internally, which is idempotent DDL (CREATE TABLE IF NOT EXISTS). Without this step, a SELECT on `checkpoints` raises `ProgrammingError 42P01` on fresh schemas.

**Zombie test setup: ORM helpers only, no raw SQL**
- Decision: Use `_ensure_experiment(session_factory, cfg)` and `_insert_run(session_factory, eid, cfg, ...)` from `runner.py` to seed test data, then UPDATE to `status='running'` with a dead pid.
- Rationale: `runs` table has 6 NOT NULL columns without server defaults (`topology`, `task_id`, `agent_set`, `seed`, `model`, `status`). Raw SQL INSERT trips NOT NULL violations. ORM helpers populate all required columns from `cfg` correctly.

**Cost-warning threshold**
- Decision: Estimate prompt triggers when `grid_est.total_cost_usd > configs[0].budget.per_experiment_usd` (1.0x multiplier).
- Rationale: Grid runs multiple cells; a stricter threshold than `atm run`'s 0.5x is intentional per reviewer confirmation.

**Single shared engine per `grid()` invocation**
- Decision: Open one `_maybe_open_session` engine at the top of `grid()` if `pg_dsn` is available, reuse for reconcile/estimate/resume probes, dispose once in `finally`.
- Rationale: Prevents connection leaks from multiple `_maybe_open_session` calls; simplifies lifecycle management.

## Constraints

- `src/atm/__main__.py` does not exist — subprocess invocations must use `["uv", "run", "atm", ...]`.
- `atm replay` does NOT have a `--yes` flag — Step 4 test invokes without it.
- `--resume-incomplete` resume loop is sequential for M12 (parallelism deferred to M13).
- All test experiment names must use `os.getpid()` suffix to avoid collision across pytest-xdist workers.
- MESSAGE_SCHEMA timestamp column is `at` (not `created_at`) — parquet diff must drop `['run_id', 'message_id', 'at']`.
- Exit code for the reconcile test is accepted as `in (0, 1)` — partial status (1 zombie + 4 completed) can produce either code depending on `_aggregate_experiment_status` logic; assert DB state, not exit code.
