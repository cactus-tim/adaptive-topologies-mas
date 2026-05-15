# m12-grid-integration — final wiring + exit-criteria integration tests

## What is this block

Финальный сшивающий блок. Берёт `run_grid` (B), `reconcile_zombies` (C), `estimate_grid` (D) и собирает их в единый `atm grid` flow, который удовлетворяет всем exit-criteria M12. Никакой новой бизнес-логики — только wiring, флаги CLI, и intеграционные тесты, гоняющие сценарии целиком.

## In scope

1. **`atm grid` wiring:**
   - Reconcile-on-start (default ON):
     - Before launching ProcessPoolExecutor, call `reconcile_zombies(session_factory, exp_id, allow_force_resume=False)`.
     - Print report: `Reconciled N zombie runs (marked failed): [run_id, run_id, ...]`.
     - Flag `--no-reconcile` disables (for ops emergencies).
     - Flag `--force-resume` switches to `allow_force_resume=True` (logs but doesn't mark failed).
   - Estimate pre-flight (default ON for grids > 1 cell):
     - Before ProcessPoolExecutor, call `estimate_grid(configs, ...)`.
     - If `total_cost_usd > cfg.budget.per_experiment_usd` → print warning, `typer.confirm(...)` unless `--yes`.
     - Always print one-line total: `Estimated grid cost: $X.XX over N cells`.
   - Resume support inside grid:
     - Flag `--resume-incomplete` — at start, after reconcile, find any `runs.status IN ('failed','running')` for this `exp_id` that have a checkpoint (`SELECT FROM checkpoints WHERE thread_id = run_id::text`). For each, call `resume_one(run_id, cfg)` from C instead of `run_one(cfg)` from B.
     - This is how E2/E5 (HITL) recovers after process death.

2. **CLI flag inventory** (final `atm grid` signature):
   ```
   atm grid --config <path>
       [+key=val ...]                # OmegaConf overrides
       [--parallelism N]             # override GridCfg.parallelism
       [--fail-fast]                 # abort on first failure
       [--no-reconcile]              # skip zombie cleanup
       [--force-resume]              # reconcile but don't mark failed
       [--resume-incomplete]         # resume failed runs from checkpoint
       [--no-estimate]               # skip cost pre-flight
       [--yes]                       # auto-confirm cost prompt
   ```

3. **End-to-end integration tests covering all four exit criteria:**

   - **`tests/integration/experiment/test_m12_exit_grid_parallel.py`** (PG):
     - Load mini grid (2 topologies × 2 tasks × 1 seed = 4 cells) with FakeLLM.
     - `atm grid --config <tmp_yaml> --parallelism 4 --yes`.
     - Assert: 4 rows in `runs`, all `status='completed'`, `experiments.status='completed'`.
     - Wall-time bound: should complete in < 2× single-run-time (proves parallelism).

   - **`tests/integration/experiment/test_m12_exit_resume_kill.py`** (PG, slow):
     - Spawn `atm run` subprocess with a fixture that has ≥ 3 LLM calls.
     - SIGKILL after 1.5 calls (use a barrier file or sleep).
     - Run `atm resume --run-id <uuid>`.
     - Assert: terminal status `completed`; `quality_score` matches a fresh-run baseline.

   - **`tests/integration/experiment/test_m12_exit_replay_bitwise.py`** (PG):
     - Run a complete cell with FakeLLM scripted.
     - Run `atm replay <run_id> --mode deterministic --yes`.
     - Assert:
       - new run row with `replay_of=<orig>` present.
       - new `runs.final_answer == orig.final_answer`.
       - `messages` parquet content equal (modulo id/timestamp fields).
       - `runs.model_version_snapshot == orig.model_version_snapshot`.

   - **`tests/integration/experiment/test_m12_exit_reconcile.py`** (PG):
     - Manually INSERT a `runs` row with `status='running'`, `host=socket.gethostname()`, `pid=<dead_pid>` (use `os.getpid() + 999999` or fork-then-exit child).
     - `atm grid --config ...` (with reconcile default ON).
     - Assert: the zombie row is now `status='failed'`, `finish_reason='zombie'`.
     - Assert: new grid runs execute normally on top.

4. **Documentation:**
   - Update `README.md` (or `dev/cli.md` if it exists) with the four commands:
     - `atm run`, `atm grid`, `atm estimate`, `atm status`, `atm resume`, `atm replay`, `atm reconcile`.
   - Include 1 worked example: minimal grid YAML + commands sequence.

5. **`dev/codebase-map.md` update:**
   - Mark M12 complete.
   - Add line to `experiment/` section listing new modules: `grid.py`, `estimator.py`, `reconcile.py`, `loader.py`.

## Out of scope

- Anything not on the M12 exit criteria checklist (PLAN.md lines 664-668).
- Performance tuning beyond proving parallelism works.
- Cross-experiment dashboards / analysis — M13.

## Inputs / preconditions

- All three precursor blocks merged: **m12-config-schema, m12-grid-runner, m12-resume-replay, m12-estimate-status-cli**.
- All public APIs from those blocks importable and tested.

## Outputs / contract

After this block:

```bash
# All four M12 exit criteria pass:
$ atm grid --config exp1.yaml --parallelism 4    # 4 parallel runs to PG
$ atm resume --run-id <uuid>                      # killed-run reaches END
$ atm replay <run_id> --mode deterministic        # bit-identical final state
$ atm grid --config exp1.yaml                     # reconcile clears zombies
```

`dev/codebase-map.md` reflects M12 complete state.

## References

- `/home/cactustim/agents/feat/m12/arch/PLAN.md` §M12 Exit (lines 664-668).
- `/home/cactustim/agents/feat/m12/arch/experiment_plan.md` — E1-E5 motivation for resume/replay (HITL recovery).
- All four predecessor dec files in `/home/cactustim/agents/feat/m12/dev/dec/`.

## Depends On

- `m12-config-schema`
- `m12-grid-runner`
- `m12-resume-replay`
- `m12-estimate-status-cli`

## Suggested run-task class

**standard** — wiring + four integration tests. Logic-light but test-heavy. ~300-500 LOC, mostly tests + CLI plumbing.

## Notes / open questions

- **Decision:** `--resume-incomplete` is opt-in, not default. Default behavior on grid restart is "reconcile-mark-failed then start fresh cells" — predictable for new users.
- **Decision:** Reconcile is per-experiment, not global. Avoids stepping on other experiments' running rows.
- Assumption: SIGKILL test uses a pytest fixture that orchestrates subprocess + barrier file. If the runner needs adjustments (e.g. an env var like `ATM_TEST_PAUSE_AFTER_N_CALLS=2`) — add minimally in m12-grid-runner or here, document.
- Open: cost-prompt threshold of `per_experiment_usd * 1.0` (not 0.5) for grids. Single-run had 0.5 in m12-estimate-status-cli; grid is more expensive so warn only when total exceeds budget cap entirely. Reasonable default.
