# План: m12-grid-runner

## Цель блока

Реализовать grid driver для M12: `run_grid()` API + `atm grid` CLI, прогоняющий список `ExperimentConfig` через `ProcessPoolExecutor`, агрегирующий статусы в PG (`experiments.status` + per-run rows), записывающий `host`/`process_pid` для каждого worker'а. Плюс — audit (без правок) Mesh/Debate на корректность параллельного выполнения агентов внутри одного run'а.

## Класс задачи: complex

ProcessPool + asyncio inside subprocess + PG-aggregation + signal handling on fail_fast + audit. ~600-900 LOC.

## Зависимости

- **m12-config-schema (merged):** даёт `load_grid_configs`, `GridCfg`, колонки `runs.host`/`runs.process_pid` (Alembic 0004).
- **experiment/runner.py::run_one** — per-cell entrypoint, signature не меняется.

## Файлы

### NEW

- `src/atm/experiment/grid.py` — `run_grid()`, `_run_cell_worker()`, `GridProgress`, `GridResult`, status aggregation helpers.
- `tests/unit/experiment/test_grid_progress.py` — derivation/aggregation rules.
- `tests/unit/experiment/test_grid_worker.py` — `_run_cell_worker` picklability + invokes `run_one` correctly.
- `tests/unit/topology/test_mesh_concurrency.py` — concurrency audit assertion (debate fan-out happens in same LangGraph super-step). Renamed scope: this asserts debate super-step parallelism (mesh is by-design serial round-robin per arch).
- `tests/integration/experiment/test_grid_mini.py` (PG) — 2 top × 2 task × 1 seed mini grid; verifies all rows + host/pid + experiments.status.
- `tests/integration/experiment/test_grid_fail_fast.py` (PG) — inject failing cell, assert `fail_fast=True` cancels.
- `tests/fixtures/experiment/grid_runner_mini.yaml` — config for integration tests (uses FakeLLM, star+chain × commongen+gsm8k).

### MODIFIED

- `src/atm/experiment/cli.py` — add `atm grid` subcommand with `--config / --parallelism / --fail-fast / --yes` flags + live progress.
- `src/atm/experiment/runner.py` — extend `_insert_run` to accept optional `host`/`pid` kwargs (or add `_register_worker_identity` helper called from `run_one` post-insert).

### READ-ONLY (audit)

- `src/atm/topology/mesh.py` — confirm dispatcher-based round-robin (not parallel).
- `src/atm/topology/debate.py` — confirm planner → (debater_pro, debater_contra) fan-out via LangGraph super-step.

## Контракт API

```python
# src/atm/experiment/grid.py
@dataclass(frozen=True)
class GridProgress:
    total: int
    done: int
    failed: int
    in_progress: int
    eta_s: float | None

@dataclass(frozen=True)
class GridResult:
    exp_id: UUID
    total: int
    completed: int
    failed: int
    budget_exceeded: int
    run_ids: list[UUID]

async def run_grid(
    configs: list[ExperimentConfig],
    *,
    parallelism: int,
    fail_fast: bool = False,
    progress_callback: Callable[[GridProgress], None] | None = None,
) -> GridResult: ...
```

## Решения / Decisions

1. **Worker model:** `concurrent.futures.ProcessPoolExecutor(max_workers=parallelism)` + `loop.run_in_executor(pool, _run_cell_worker, cfg_dict)`. Main process collects futures.
2. **Picklability:** worker принимает `dict` (от `cfg.model_dump(mode="python")`) и пере-валидирует через `ExperimentConfig.model_validate` чтобы избежать Pydantic v2 pickle quirks.
3. **Host/PID capture:** в worker'е, сразу после `_insert_run`, делаем `UPDATE runs SET host=:h, process_pid=:p WHERE id=:rid`. Чтобы не править `_insert_run` (риск breaking change), сделаем отдельный хелпер `_register_worker_identity(session_factory, run_id, host, pid)` и вызовем его из worker'а **перед** invocation топологии. Альтернатива: внутри `_run_cell_worker` после `asyncio.run(run_one(cfg))` — но нам нужно зарегистрировать pid даже если run упадёт. Поэтому делаем `monkey-style override`: worker делает `await _insert_run` сам? Нет — этого не хочется. Лучше: модифицируем `runner.run_one` чтобы оно после `_insert_run` сразу UPDATE'ило host/pid из `os.getpid()`+`socket.gethostname()`. Это null-op для не-grid-режима (тоже полезно для observability single-run). Безопасно: чистое расширение без сигнатурных breaking changes.
4. **Progress aggregation:** main holds `dict[future, run_idx]` + counters; on `future.result()` callback пересчитывает progress + вызывает `progress_callback`. После всех done — `UPDATE experiments.status`:
   - `completed` если все `completed` (включая `budget_exceeded` как success-terminal? — нет, budget_exceeded считается failure для experiments-aggregate: completed = только status=='completed'; failed = status in {'failed','budget_exceeded'}). Будем считать:
     - `completed` → all cells `status='completed'`
     - `failed` → all cells `status` in `{'failed','budget_exceeded'}`
     - `partial` → mix
5. **fail_fast:** при первом непогашенном `failed` future — `executor.shutdown(wait=False, cancel_futures=True)`, переводим остаток в `failed` через DB (если есть `run_id`) или пропускаем (если cell ещё не дошёл до `_insert_run`). Возвращаем `GridResult` со скорректированными счётчиками.
6. **CLI:** `atm grid` использует `tqdm` если есть, иначе `print(..., end='\r')`. Лёгкая проверка через `importlib.util.find_spec`. Без новых deps. (Проверить pyproject — tqdm обычно есть как transitive.)
7. **Audit (Mesh/Debate):** ничего не правим в коде топологий. В тесте `test_mesh_concurrency.py` (точнее — переименуем в `test_debate_super_step_parallelism.py` чтобы не врать названием) ставим time-stamped instrumentation на FakeLLM, прогоняем debate и проверяем что debater_pro и debater_contra стартанули в пределах epsilon (одного super-step'а LangGraph). Mesh-параллелизма нет by design; задокументировано в коде/комментариях.

## Exit codes (CLI)

- 0 — all cells `completed`
- 1 — some cells failed (partial)
- 2 — all cells failed
- 3 — config/load error

## Шаги

| # | Type | Шаг | Файлы | Depends |
|---|------|-----|-------|---------|
| 1 | simple | Минимальный hook в `runner._insert_run` (или новый `_register_worker_identity` сразу после) — UPDATE host/pid после insert | runner.py | — |
| 2 | tdd | `GridProgress`/`GridResult` dataclasses + status aggregation helper `_aggregate_experiment_status()` | grid.py, test_grid_progress.py | 1 |
| 3 | tdd | `_run_cell_worker(cfg_dict)` top-level worker entrypoint | grid.py, test_grid_worker.py | 2 |
| 4 | tdd | `run_grid()` async wrapper над ProcessPoolExecutor (без fail_fast) | grid.py, test_grid_progress.py | 3 |
| 5 | tdd | fail_fast cancellation in run_grid | grid.py, test_grid_progress.py | 4 |
| 6 | simple | `atm grid` CLI command в cli.py | cli.py | 4, 5 |
| 7 | tdd | Audit test для Debate super-step parallelism | test_debate_super_step_parallelism.py | — (parallel) |
| 8 | simple | Integration test `test_grid_mini.py` (PG) + fixture | test_grid_mini.py, grid_runner_mini.yaml | 4 |
| 9 | simple | Integration test `test_grid_fail_fast.py` (PG) | test_grid_fail_fast.py | 5 |

## Needs Integration Tests: YES

PG-gated (`@pytest.mark.requires_postgres`, `pytest.mark.pg`). Активируются под `ATM_ENABLE_PG_TESTS=1`. Если PG локально нет — `--collect-only` для верификации parsing/skip.

## Out of scope

- `atm resume`/`atm replay` — m12-resume-replay.
- `atm estimate`/`atm status` — m12-estimate-status-cli.
- `--estimate` pre-flight gate — m12-grid-integration.
- Cross-host (Ray/Dask) — не делаем.

## Риски

- **Pydantic v2 pickle:** mitigated через model_dump + model_validate в worker.
- **SubprocessSandbox под parallelism>=4:** workspace scoped per-run_id, должно быть ОК.
- **PG connection pool exhaustion:** каждый worker создаёт свой engine через `create_engine`. При parallelism=4 это 4 пула × pool_size — следить за PG `max_connections`. Тесты используют parallelism=2.
- **Сигналы / shutdown:** `cancel_futures=True` отменяет только PENDING futures. RUNNING workers продолжают; их статус останется тем, который успеет записать `_update_run_*`. Документируем в docstring `fail_fast`.

## NOTES.md

Будет создан в worktree если потребуется фиксировать judgment calls.
