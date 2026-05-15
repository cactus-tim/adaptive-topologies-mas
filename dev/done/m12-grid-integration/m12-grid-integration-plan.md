# m12-grid-integration — Plan

## Executive Summary

Final wiring block for M12. Extends the existing `atm grid` Typer command in `src/atm/experiment/cli.py` with three pre-flight behaviours (reconcile-on-start, estimate pre-flight, resume-incomplete) plus four new CLI flags (`--no-reconcile`, `--force-resume`, `--resume-incomplete`, `--no-estimate`). Writes four PG-gated integration tests that exercise the four M12 exit criteria end-to-end. Refreshes README with the consolidated CLI surface. No new business logic — purely glue between already-merged primitives (`run_grid`, `reconcile_zombies`, `estimate_grid`, `resume_one`).

## Current State

- `atm grid` command exists in `src/atm/experiment/cli.py` (lines 207-313): loads configs via `load_grid_configs`, optionally prompts via `typer.confirm`, drives `run_grid` with a progress callback, exits with codes 0/1/2/3.
- Reconcile / estimate / resume-incomplete are NOT yet wired into `atm grid`.
- `atm reconcile` exists separately (lines 539-605) using `reconcile_zombies(session_factory, eid, allow_force_resume=dry_run)`.
- All four predecessor blocks have merged their public APIs: `run_grid`, `reconcile_zombies`, `estimate_grid`, `resume_one`, `replay_one`.
- Four integration test files for M12 exit criteria do not yet exist.
- README is stale (shows M0 status).

## Proposed Approach

Wire the three pre-flight behaviours into `atm grid` in a fixed order:

1. Resolve experiment ID by name (SELECT from `experiments` table).
2. Reconcile zombies (skipped if `--no-reconcile` or no existing experiment row).
3. Estimate cost (skipped if `--no-estimate` or single-cell grid; prompts user if cost exceeds budget).
4. Resume incomplete runs (only if `--resume-incomplete` and an experiment row exists).
5. Run the grid as before.

All four new integration tests use subprocess invocation (`uv run atm ...`) to avoid the `asyncio_mode=auto` + `CliRunner` conflict under Python 3.11. Each test maps 1:1 to an M12 exit criterion and is PG-gated via `ephemeral_pg_dsn`.

Key resolution for reconcile scope: reconcile is per-experiment and requires `exp_id`, but `run_grid` creates the experiment row. Solution: look up existing experiment row by `cfg.name` (unique constraint). If found, reconcile. If not (first launch), skip.

## Implementation Phases

### Phase 1: CLI wiring (~3h)
**Goal:** Extend `atm grid` with four new flags and three pre-flight behaviours.

- [ ] 1.1 Add four CLI flags to `grid()` signature in `src/atm/experiment/cli.py`
  - File: `src/atm/experiment/cli.py`
  - Acceptance: `uv run atm grid --help` shows `--no-reconcile`, `--force-resume`, `--resume-incomplete`, `--no-estimate`

- [ ] 1.2 Implement experiment-name lookup (SELECT by name), reconcile pass, estimate pre-flight, and resume-incomplete branch with `_collect_resume_targets` helper
  - File: `src/atm/experiment/cli.py`
  - Acceptance: Calling `atm grid --config ...` on a fresh DB skips reconcile, prints estimate for >1 cell, runs; calling on existing experiment prints `Reconciled N zombie run(s)...`

- [ ] 1.3 Manage connection lifecycle: single shared engine opened at top of `grid()`, disposed once in `finally`; `build_checkpointer` called-and-closed for checkpoint DDL guarantee before resume-incomplete SELECT
  - File: `src/atm/experiment/cli.py`
  - Acceptance: No connection leaks; `ruff check` and `mypy` clean on `cli.py`

### Phase 2: Integration tests — wave 1 (parallel) (~2h)
**Goal:** Write three of the four exit-criterion tests that have no dependency on Phase 1's new flags.

- [ ] 2.1 Create `test_m12_exit_resume_kill.py` — spawns `atm run` via subprocess, polls for running row, SIGKILLs, then invokes `atm resume --run-id <uuid> --force`, asserts resumed row updated in-place with quality parity
  - File: `tests/integration/experiment/test_m12_exit_resume_kill.py`
  - Acceptance: Passes under `ATM_ENABLE_PG_TESTS=1`; no new run row inserted by resume; quality_score matches baseline

- [ ] 2.2 Create `test_m12_exit_replay_bitwise.py` — runs `run_one` in-process, then invokes `atm replay <run_id> --mode deterministic` via subprocess, asserts `replay_of` FK, `final_answer` identity, and `messages.parquet` content equal modulo `[run_id, message_id, at]` columns
  - File: `tests/integration/experiment/test_m12_exit_replay_bitwise.py`
  - Acceptance: Passes under `ATM_ENABLE_PG_TESTS=1`; parquet diff clean; `replay_of` set correctly

- [ ] 2.3 Update `README.md` with CLI section (seven commands, worked grid YAML + command sequence, updated status note)
  - File: `README.md`
  - Acceptance: Visual inspection; no broken YAML; Russian narrative preserved; code blocks in English

### Phase 3: Integration tests — wave 2 (depend on Phase 1) (~1.5h)
**Goal:** Write the two exit-criterion tests that consume the new CLI flags from Phase 1.

- [ ] 3.1 Create `test_m12_exit_grid_parallel.py` — invokes `atm grid` via subprocess with `--no-reconcile --no-estimate --parallelism 4 --yes`, asserts 4 completed run rows, experiment completed, wall-time < 60s
  - File: `tests/integration/experiment/test_m12_exit_grid_parallel.py`
  - Acceptance: Passes under `ATM_ENABLE_PG_TESTS=1`; returncode 0; 4 completed rows; `experiments.status='completed'`

- [ ] 3.2 Create `test_m12_exit_reconcile.py` — seeds experiment + zombie run via `_ensure_experiment`/`_insert_run` (NOT raw SQL), invokes `atm grid` (reconcile ON by default), asserts zombie row is `status='failed', finish_reason='zombie'` and 4 fresh cells completed
  - File: `tests/integration/experiment/test_m12_exit_reconcile.py`
  - Acceptance: Passes under `ATM_ENABLE_PG_TESTS=1`; zombie state correct from DB; exit code in (0, 1)

### Phase 4: Verification sweep (~0.5h)
**Goal:** Full unit + integration suite green.

- [ ] 4.1 Run ruff + mypy on changed files, then full unit suite, then full integration suite under `ATM_ENABLE_PG_TESTS=1`
  - File: N/A (verification only)
  - Acceptance: All tests green; no regressions in existing `test_grid_mini.py`, `test_resume_sigkill.py`, `test_replay_deterministic.py`, `test_status_cli.py`, `test_estimator_historical.py`

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/experiment/cli.py` | Extend `grid()` with 4 flags + 3 pre-flight behaviours + `_collect_resume_targets` helper | Core M12 wiring |
| `tests/integration/experiment/test_m12_exit_grid_parallel.py` | NEW — exit criterion 1 acceptance test | M12 sign-off |
| `tests/integration/experiment/test_m12_exit_resume_kill.py` | NEW — exit criterion 2 acceptance test | M12 sign-off |
| `tests/integration/experiment/test_m12_exit_replay_bitwise.py` | NEW — exit criterion 3 acceptance test | M12 sign-off |
| `tests/integration/experiment/test_m12_exit_reconcile.py` | NEW — exit criterion 4 acceptance test | M12 sign-off |
| `README.md` | Refresh CLI section, add CLI reference table and worked grid example | User-facing docs |

## Dependencies & Order Constraints

- Phase 1 (CLI wiring) must complete before Phase 3 tests (which use `--no-reconcile`, `--no-estimate`, and the reconcile-on-start behaviour).
- Phase 2 tests and README update can run in parallel with Phase 1.
- Phase 4 (verification) runs last, after all phases complete.
- No changes to `grid.py`, `runner.py`, `reconcile.py`, `estimator.py`, `loader.py` — their public APIs are consumed as-is.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Connection leaks from `_maybe_open_session` calls | Medium | Silent resource exhaustion | Open one shared engine per `grid()` invocation; dispose in `finally` |
| `asyncio.run()` conflict under `asyncio_mode=auto` if CliRunner is used | High | RuntimeError, test failure | All integration tests use subprocess invocation (`uv run atm ...`) |
| SIGKILL timing race in resume-kill test (child not yet in `running` state) | Medium | Flaky test | 60s polling deadline with 250ms interval; descriptive failure message with subprocess stdout/stderr |
| Pid reuse on test host for zombie test | Low | Wrong zombie classification | 3-retry loop with `_pid_alive` check; `pytest.skip` if exhausted |
| `checkpoints` table not present on fresh schema before resume-incomplete SELECT | High | `ProgrammingError 42P01` | Call `build_checkpointer(dsn)` (idempotent `saver.setup()`) before any SELECT on `checkpoints` |
| `uv` not on PATH in some CI environments | Low | Subprocess invocation fails | Document fallback: `[sys.executable, "-c", "from atm.experiment.cli import app; app()"]` with argv patching |
| Experiment-name collision across pytest-xdist workers | Low | Test interference | Use `name=..._os.getpid()` in all test experiments |

## Out of Scope

- Changes to `grid.py`, `runner.py`, `reconcile.py`, `estimator.py`, `loader.py` (APIs consumed as-is).
- DB schema / migration changes.
- `dev/codebase-map.md` update (handled by user post-merge via background agent).
- Moving `dev/active/m12-grid-integration/` to `dev/done/` (handled by orchestrator, not this plan).
- `--resume-incomplete` parallelism (sequential for M12; revisit in M13).
- New unit tests for `cli.py` changes (logic is thin glue; integration tests cover the contract).

## Timeline

- Total: ~7h
- Phase 1: ~3h
- Phase 2 (parallel with Phase 1 where possible): ~2h
- Phase 3: ~1.5h
- Phase 4: ~0.5h
- Created: 2026-05-14
