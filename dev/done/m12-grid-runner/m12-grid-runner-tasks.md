# Tasks: m12-grid-runner

## Implementation order (sequential, single worktree)

### Step 1 — Wire host/pid into run_one (simple)
- [ ] В `src/atm/experiment/runner.py` после `_insert_run(...)` добавить UPDATE `runs.host = socket.gethostname()`, `runs.process_pid = os.getpid()`.
- [ ] Импорты: `import os, socket`.
- [ ] Обернуть в try/except + warning (как остальные пост-update'ы), чтобы DB-сбой не падал run.
- [ ] Smoke: `uv run pytest tests/unit/experiment/ -q`.

### Step 2 — GridProgress / GridResult / status aggregation (tdd)
- [ ] `tests/unit/experiment/test_grid_progress.py`: тесты для:
  - dataclass instantiation + frozen
  - `_aggregate_experiment_status({...})` → "completed" | "partial" | "failed" | "running"
- [ ] `src/atm/experiment/grid.py`: import stubs, dataclasses, helper.
- [ ] Smoke pytest.

### Step 3 — _run_cell_worker (tdd)
- [ ] `tests/unit/experiment/test_grid_worker.py`: тест что `_run_cell_worker(cfg.model_dump(mode="python"))` возвращает dict с `run_id`, `status`, `quality_score`, etc. Используем FakeLLM-fixture.
  - Test ОЧЕНЬ дорогой если он реально дёргает run_one → используем monkeypatch на `run_one` чтобы вернуть fake RunResult и проверить только сериализационную обёртку (picklability + dict-shape).
- [ ] `_run_cell_worker(cfg_dict: dict) -> dict`: 
  - re-validate via `ExperimentConfig.model_validate`
  - `asyncio.run(run_one(cfg))`
  - return picklable dict: `{"run_id": str, "exp_id": str, "status": str, "metrics": dict, "final_answer": str, "error": str|None}`
  - try/except — catch all, return error dict (worker never raises through pickle boundary).
- [ ] Smoke pytest.

### Step 4 — run_grid() core (tdd, no fail_fast yet)
- [ ] `tests/unit/experiment/test_grid_progress.py` — добавить тесты:
  - `run_grid([])` raises ValueError
  - `run_grid([cfg, cfg])` with mocked pool returns GridResult с правильными счётчиками
  - progress_callback вызывается на каждом completed future
- [ ] `run_grid(configs, *, parallelism, fail_fast=False, progress_callback=None) -> GridResult`:
  - asyncio + `loop.run_in_executor(ProcessPoolExecutor(max_workers=parallelism), _run_cell_worker, cfg_dict)`
  - Wait via `asyncio.as_completed`
  - Track counters; emit progress_callback
  - After all done: aggregate status via `_aggregate_experiment_status`, UPDATE experiments.status
  - Return GridResult
- [ ] Smoke pytest.

### Step 5 — fail_fast cancellation (tdd)
- [ ] Тест: с mocked pool, одна cell возвращает status="failed"; `fail_fast=True` → ostavshiesya pending futures cancelled, counted в GridResult как not-started (failed counter unchanged for them).
- [ ] Patch run_grid: при первом fail, `executor.shutdown(wait=False, cancel_futures=True)`; собираем уже-running results.
- [ ] Smoke pytest.

### Step 6 — atm grid CLI (simple)
- [ ] В `src/atm/experiment/cli.py` добавить `@app.command("grid")`:
  - args: `--config / -c`, `--parallelism / -p` (overrides cfg.grid.parallelism), `--fail-fast`, `--yes`, optional `[+key=val ...]` overrides.
  - Calls `load_grid_configs`, prints cell count, askConfirm if not `--yes`.
  - Calls `asyncio.run(run_grid(...))` with progress_callback printing live status via `\r`.
  - Exit codes 0/1/2/3 per plan.
- [ ] Smoke: `uv run python -m atm.experiment.cli grid --help` (or via entry point `atm grid --help`).

### Step 7 — Debate super-step parallelism audit test (tdd)
- [ ] `tests/unit/topology/test_debate_super_step_parallelism.py`:
  - Build debate topology with FakeLLM that records `time.monotonic()` on each `ainvoke`.
  - Run one round.
  - Assert `|t_debater_pro - t_debater_contra| < threshold` (e.g. 50ms) — i.e. они стартуют в одном super-step'е.
- [ ] Add explanatory docstring: «Mesh by design serial round-robin; Debate uses LangGraph super-step fan-out — this test asserts the latter still holds.»

### Step 8 — Integration test: grid_mini (PG)
- [ ] `tests/fixtures/experiment/grid_runner_mini.yaml` — minimal grid config (2 topologies × 2 tasks × 1 seed = 4 cells), FakeLLM scripted.
- [ ] `tests/integration/experiment/test_grid_mini.py`:
  - `@pytest.mark.requires_postgres`, `@pytest.mark.pg`
  - load_grid_configs → run_grid(parallelism=2)
  - Assert 4 rows в `runs`, all `status='completed'`, all имеют `host` и `process_pid`.
  - Assert `experiments.status='completed'`.
- [ ] Collect-only verify.

### Step 9 — Integration test: fail_fast (PG)
- [ ] `tests/integration/experiment/test_grid_fail_fast.py`:
  - monkeypatch одну cell чтобы run_one бросило (например невалидный task input).
  - run_grid(parallelism=2, fail_fast=True).
  - Assert GridResult.failed >= 1, total cells started ≤ all (some cancelled).
- [ ] Collect-only verify.

## Verification (Phase 5)

- [ ] `uv run ruff format src/atm/experiment/grid.py src/atm/experiment/cli.py src/atm/experiment/runner.py tests/unit/experiment/test_grid_progress.py tests/unit/experiment/test_grid_worker.py tests/unit/topology/test_debate_super_step_parallelism.py tests/integration/experiment/test_grid_mini.py tests/integration/experiment/test_grid_fail_fast.py`
- [ ] `uv run ruff check --fix` on same files
- [ ] `uv run mypy src/atm/experiment/grid.py src/atm/experiment/cli.py src/atm/experiment/runner.py`
- [ ] `uv run pytest tests/unit/experiment/ tests/unit/topology/test_debate_super_step_parallelism.py -q`
- [ ] PG integration collect-only:  `uv run pytest tests/integration/experiment/test_grid_mini.py tests/integration/experiment/test_grid_fail_fast.py --collect-only`

## Phase 6 — Done

- [ ] `mkdir -p dev/done && mv dev/active/m12-grid-runner dev/done/m12-grid-runner`
- [ ] Cleanup `.draft-*`, `.escalation-*`
- [ ] Final commit message.
