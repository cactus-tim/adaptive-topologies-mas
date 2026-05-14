# m12-resume-replay — Implementation Plan

Class: complex. Needs Integration Tests: yes.

## Steps (sequential)

1. **Step 1 — config_snapshot widening (TDD)**
   - Modify `_ensure_experiment` in `src/atm/experiment/runner.py`: write full `cfg.model_dump(mode="json")` instead of the sparse 4-field dict.
   - Helper: `_load_cfg_from_snapshot(snapshot: dict) -> ExperimentConfig` in runner.py (rehydrates ExperimentConfig from JSON dump).
   - Unit test: `tests/unit/experiment/test_config_snapshot_widening.py`.

2. **Step 2 — host/pid capture on `_insert_run` (simple)**
   - At run start, populate `runs.host = socket.gethostname()` and `runs.process_pid = os.getpid()`.
   - Unit test extends existing runner tests OR new `tests/unit/experiment/test_run_host_pid.py`.

3. **Step 3 — `build_llm` replay support (TDD)**
   - Add `replay_source: str | Path | None = None` kwarg to `build_llm`.
   - When `model_id == "fake:replay"` and `replay_source` given → load parquet via `pyarrow.parquet.read_table`, instantiate `FakeLLM(mode="replay", replay_table=table)`.
   - Unit test: `tests/unit/llm/test_factory_replay.py`.

4. **Step 4 — `reconcile.py` module (TDD)**
   - New file `src/atm/experiment/reconcile.py` with `ZombieRow`, `ReconcileReport`, `reconcile_zombies()`.
   - Heuristic: host mismatch → zombie; host match + pid given → `_pid_alive(pid)` using `os.kill(pid, 0)`; pid None → no_pid zombie.
   - When `allow_force_resume=False`: UPDATE `runs SET status='failed', finish_reason='zombie'`.
   - When `allow_force_resume=True`: only log (no mutation).
   - Unit test: `tests/unit/experiment/test_reconcile.py` with mocked session.

5. **Step 5 — `resume_one(run_id, cfg, *, force)` in runner.py (TDD-light)**
   - SELECT runs row by id; reconstruct `ExperimentConfig` via `_load_cfg_from_snapshot`.
   - Reset run status to 'running'; re-build topology with `thread_id=str(run_id)`.
   - On success: `_update_run_success` (run_id existing, NOT new).
   - `--force` skips pid-alive check.

6. **Step 6 — `replay_one(original_run_id, mode)` in runner.py (TDD-light)**
   - SELECT original run; verify `model_version_snapshot` parity (deterministic mode) — abort if mismatch unless semantic.
   - Override `cfg.model.default = "fake:replay"`; map each role to original parquet path.
   - INSERT new run with `replay_of=<original_run_id>`, `exp_id` parent equality enforced.
   - Build LLM wrappers using `replay_source` per role.

7. **Step 7 — CLI commands (`atm resume`, `atm replay`, `atm reconcile`)** (simple)
   - Add to `src/atm/experiment/cli.py`.
   - `atm resume --run-id <uuid> [--force]`
   - `atm replay <run_id> [--mode deterministic|semantic] [--output-config-only]`
   - `atm reconcile --exp-id <uuid> [--dry-run]`

8. **Step 8 — Integration tests (PG, gated by `ATM_ENABLE_PG_TESTS`)**
   - `tests/integration/experiment/test_resume_sigkill.py` — slow, requires PG.
   - `tests/integration/experiment/test_replay_deterministic.py` — requires PG.

9. **Step 9 — Verification**
   - ruff format + check, mypy on touched files.
   - Smoke `uv run pytest tests/unit/ -q`.

## Key decisions (per dec)

- **Option (a)** for config_snapshot — widen to `cfg.model_dump(mode="json")`.
- **No psutil** — use `os.kill(pid, 0)` POSIX fallback.
- `replay_of` enforces `exp_id` parent equality.
- `--force` bypasses pid-alive check; reconcile is opt-in.
- LangGraph `thread_id=str(run_id)` semantics — resume continues from last checkpoint.

## Migration

No new Alembic migration needed — `experiments.config_snapshot` is already `JSONB`; we just write a richer payload. ORM schema unchanged.
