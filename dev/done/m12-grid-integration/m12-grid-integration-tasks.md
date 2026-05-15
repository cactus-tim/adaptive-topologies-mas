# m12-grid-integration — Tasks

## Phase 1: CLI wiring - NOT STARTED

- [ ] 1.1 Add four CLI flags to `grid()` signature — `src/atm/experiment/cli.py`
  - Acceptance: `uv run atm grid --help` shows `--no-reconcile`, `--force-resume`, `--resume-incomplete`, `--no-estimate`

- [ ] 1.2 Implement experiment-name lookup, reconcile pass, estimate pre-flight, and resume-incomplete branch with `_collect_resume_targets` helper — `src/atm/experiment/cli.py`
  - Acceptance: Fresh DB skips reconcile and prints estimate (>1 cell) before running; existing experiment prints `Reconciled N zombie run(s)...`; `--resume-incomplete` resumes failed-with-checkpoint rows before fresh cells

- [ ] 1.3 Manage connection lifecycle: single shared engine, `build_checkpointer` setup for checkpoint DDL guarantee, `finally` dispose — `src/atm/experiment/cli.py`
  - Acceptance: No connection leaks; `ruff check src/atm/experiment/cli.py` and `mypy src/atm/experiment/cli.py` both clean

## Phase 2: Integration tests wave 1 (parallel with Phase 1) - NOT STARTED

- [ ] 2.1 Create exit criterion 2 test — resume after SIGKILL — `tests/integration/experiment/test_m12_exit_resume_kill.py`
  - Acceptance: Passes under `ATM_ENABLE_PG_TESTS=1`; no new run row inserted by resume; quality_score matches baseline; SIGKILL poll timeout 60s

- [ ] 2.2 Create exit criterion 3 test — replay bitwise — `tests/integration/experiment/test_m12_exit_replay_bitwise.py`
  - Acceptance: Passes under `ATM_ENABLE_PG_TESTS=1`; `replay_of` FK set; `final_answer` identical; `messages.parquet` equal modulo `[run_id, message_id, at]` columns

- [ ] 2.3 Refresh README CLI section — `README.md`
  - Acceptance: Seven commands listed with one-line descriptions; worked grid YAML + command sequence present; Russian narrative preserved; no broken YAML

## Phase 3: Integration tests wave 2 (after Phase 1) - NOT STARTED

- [ ] 3.1 Create exit criterion 1 test — parallel grid execution — `tests/integration/experiment/test_m12_exit_grid_parallel.py`
  - Acceptance: Passes under `ATM_ENABLE_PG_TESTS=1`; returncode 0; 4 rows `status='completed'`; `experiments.status='completed'`; wall-time < 60s

- [ ] 3.2 Create exit criterion 4 test — zombie reconcile-on-start — `tests/integration/experiment/test_m12_exit_reconcile.py`
  - Acceptance: Passes under `ATM_ENABLE_PG_TESTS=1`; zombie row has `status='failed', finish_reason='zombie'`; 4 fresh cells completed; CLI exit code in (0, 1)

## Phase 4: Verification sweep - NOT STARTED

- [ ] 4.1 Run full lint + type-check + unit + integration suite
  - Acceptance: `ruff check` clean; `mypy` clean; `uv run pytest tests/unit/ -q` green; `ATM_ENABLE_PG_TESTS=1 uv run pytest tests/integration/experiment/ -v` green (all four new tests + all existing tests)

---
## Stats
- Total: 11 tasks · ~7h
- Done: 0 / 11

## How to Update
Check off tasks with [x] and update `m12-grid-integration-context.md` SESSION PROGRESS section after each milestone. Update phase headers: all tasks done -> "- COMPLETE", some done -> "- IN PROGRESS".
