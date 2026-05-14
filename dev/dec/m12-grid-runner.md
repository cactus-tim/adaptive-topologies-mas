# m12-grid-runner — Grid driver, ProcessPoolExecutor, parallel agents audit

## What is this block

Core execution engine of M12. Реализует `experiment/grid.py` — берёт список `ExperimentConfig` cells (от `load_grid_configs`), гоняет их через `ProcessPoolExecutor`, аггрегирует прогресс в PG (`experiments.status` + per-run rows). Заодно — аудит того, что внутри одного run'а `asyncio.gather` правда параллелит agent-узлы в Mesh/Debate broadcast'е (это уже должно быть в M6 runner, но проверяется в этом блоке). Регистрирует `host`/`process_pid` в `runs` при старте worker'а (поля от блока A).

CLI: `atm grid --config <path> [+key=val ...]` — синхронная обёртка вокруг grid driver'а, печатает live-progress, выходит с exit-кодом по агрегированному статусу.

## In scope

1. **`src/atm/experiment/grid.py` (NEW FILE):**

   ```python
   async def run_grid(
       configs: list[ExperimentConfig],
       *,
       parallelism: int,
       fail_fast: bool = False,
       progress_callback: Callable[[GridProgress], None] | None = None,
   ) -> GridResult: ...
   ```

   - **Worker model:** `concurrent.futures.ProcessPoolExecutor(max_workers=parallelism)`. Каждая cell — отдельный child process, вызывает `asyncio.run(run_one(cfg))`. Process isolation = чистый event loop, нет cross-run state leakage, корректные signal handlers.
   - **Worker entrypoint:** module-level function `_run_cell_worker(cfg_dict: dict) -> dict` (top-level, picklable). Re-validates `cfg_dict` через `ExperimentConfig.model_validate` чтобы избежать pickle Pydantic v2 quirks.
   - **Host/PID capture:** worker записывает `socket.gethostname()` + `os.getpid()` в `runs.host`/`runs.process_pid` в первом DB-UPDATE после `_insert_run`. Extend `runner._insert_run` чтобы принимать optional `host`/`pid` kwargs (или сделать отдельный `_register_worker_identity` хелпер).
   - **Progress aggregation:**
     - Главный процесс держит `dict[run_id, RunStatus]` в памяти.
     - После каждого `future.result()` обновляет `experiments.status` через `UPDATE experiments SET status = ... WHERE id = exp_id`:
       - `running` пока хотя бы один cell `running`.
       - `completed` если все `completed`.
       - `partial` если есть и `completed` и `failed`.
       - `failed` если все `failed`.
   - **GridProgress dataclass:**
     ```python
     @dataclass
     class GridProgress:
         total: int
         done: int
         failed: int
         in_progress: int
         eta_s: float | None
     ```
   - **GridResult:**
     ```python
     @dataclass
     class GridResult:
         exp_id: UUID
         total: int
         completed: int
         failed: int
         budget_exceeded: int
         run_ids: list[UUID]
     ```

2. **`atm grid` CLI command** (in `src/atm/experiment/cli.py`):
   - Signature: `atm grid --config <path> [+key=val ...] [--parallelism N] [--fail-fast]`.
   - Loads via `load_grid_configs`, prints expected cell count, asks `--yes` flag or interactive confirm (`typer.confirm`).
   - Live progress via `tqdm` или ручной `\r`-overwrite (без новых deps; tqdm уже косвенно есть).
   - Exit codes: 0 = all completed; 1 = some failed; 2 = all failed; 3 = config error.

3. **Parallel agents audit (asyncio inside run):**
   - Read `src/atm/topology/mesh.py` and `src/atm/topology/debate.py`.
   - Verify Mesh broadcast and Debate parallel-stance use `asyncio.gather` (not sequential await in a loop).
   - If sequential — fix to `asyncio.gather(*[agent.step(...) for ...])`. Document in changelog of the block.
   - Add unit test `tests/unit/topology/test_mesh_concurrency.py` that asserts gather pattern: instrument LLMWrapper to record call-start timestamps; assert max overlap ≥ 2 within tolerance.

4. **Tests:**
   - `tests/unit/experiment/test_grid_progress.py`: GridProgress derivation, status aggregation rules (`completed`/`partial`/`failed`).
   - `tests/integration/experiment/test_grid_mini.py` (PG): 2 topologies × 2 tasks × 1 seed = 4 cells with FakeLLM; assert all 4 rows present in `runs`; `experiments.status == 'completed'`; `host` and `process_pid` populated.
   - `tests/integration/experiment/test_grid_fail_fast.py` (PG): inject one failing cell; assert `fail_fast=True` cancels pending futures.

## Out of scope

- `atm resume`, `atm replay`, reconcile-on-grid-start — m12-resume-replay.
- `atm estimate`, `atm status` — m12-estimate-status-cli.
- `atm grid --estimate` pre-flight cost gate — m12-grid-integration (wiring stage).
- Cross-host distributed execution (Ray/Dask) — out of scope entirely; M12 is single-host process pool.

## Inputs / preconditions

- m12-config-schema **must be merged**: this block imports `load_grid_configs`, `GridCfg`, and relies on `runs.host`/`runs.process_pid` columns.
- `experiment/runner.py::run_one` is the per-cell entrypoint, unchanged in signature.
- Existing FakeLLM fixtures sufficient for integration tests (per codebase-map; reuse `m11_gsm8k_e2e_executor.yaml` or add minimal star/chain fixtures).

## Outputs / contract

```python
# Public API
from atm.experiment.grid import run_grid, GridProgress, GridResult

# CLI
$ atm grid --config conf/experiments/exp1.yaml --parallelism 4
# stdout (live): "[3/8] running | done=2 failed=0 eta=42s"
# exit 0 / 1 / 2 / 3
```

DB side-effects after `run_grid` returns:
- exactly `len(configs)` rows in `runs` for the experiment, each with terminal `status` (completed | failed | budget_exceeded).
- `experiments.status` ∈ {completed, partial, failed}.
- Every row has `host` and `process_pid` populated (for reconcile in m12-resume-replay).

## References

- `/home/cactustim/agents/feat/m12/arch/PLAN.md` §M12 lines 651-654.
- `/home/cactustim/agents/feat/m12/arch/arch.md` §11 (storage) and §12 (experiment).
- `/home/cactustim/agents/feat/m12/src/atm/experiment/runner.py` — `run_one` lifecycle (worker calls this).
- `/home/cactustim/agents/feat/m12/src/atm/topology/mesh.py` and `debate.py` — for asyncio audit.
- Python docs: `concurrent.futures.ProcessPoolExecutor` — for handling pool shutdown / fail_fast cancellation.

## Depends On

- `m12-config-schema` — needs `GridCfg`, `load_grid_configs`, and `runs.host/process_pid` columns.

## Suggested run-task class

**complex** — ProcessPool + asyncio inside subprocess + PG aggregation + signal handling on fail_fast. ~600-900 LOC including tests.

## Notes / open questions

- **Decision:** Process pool over thread pool — child process can have its own asyncio loop without contention. Picklability of `ExperimentConfig` is handled by serializing to dict and re-validating in worker.
- **Decision:** Worker writes `host`/`pid` directly to PG (not via main process channel). Avoids IPC complexity.
- **Decision:** `parallelism` from CLI flag overrides `GridCfg.parallelism`. If neither set → default 4.
- Assumption: tqdm dependency is acceptable; if not, fall back to plain `print(..., end="\r")`. Check `pyproject.toml` before adding.
- Risk: SubprocessSandbox inside workers — each child spins its own sandbox; may stress filesystem under high parallelism. Watch workspace dir scoping (already per-run_id, should be safe).
