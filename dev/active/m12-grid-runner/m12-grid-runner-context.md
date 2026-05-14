# Context: m12-grid-runner

## Зачем нужен блок

M12 продолжает M11's reproducibility-focused milestone. Single-run pipeline (`atm run`) уже работает; теперь нужен **grid-runner** — drive параллельные прогоны cells (топология × задача × seed) через child processes, агрегировать статусы в PG. Это даст возможность запускать experiment sweeps в одну команду (`atm grid --config conf/sweep.yaml --parallelism 8`) и затем анализировать через M13.

## Архитектурный контекст

- arch.md §11: `experiments` table = aggregate row per (config-snapshot, name); `runs` table = one row per cell.
- arch.md §12: experiment runner — Pydantic config + OmegaConf + Typer CLI.
- M12-config-schema (merged) даёт:
  - `GridCfg.sweep: dict[str, list[scalar]]`, `parallelism`, `fail_fast`, `seeds`.
  - `load_grid_configs(path, overrides) -> list[ExperimentConfig]` — раскрывает grid.sweep × seeds.
  - Alembic 0004: `runs.host`, `runs.process_pid`, composite index, `replay_of` (нам нужны только host/pid).

## Текущая поверхность (что есть до m12-grid-runner)

- `experiment/config.py`: `ExperimentConfig`, `GridCfg`, `EstimateCfg`.
- `experiment/loader.py`: `load_config`, `load_grid_configs`.
- `experiment/runner.py`: `run_one(cfg)` async fn — полная lifecycle (ensure_experiment, insert_run, build topology, ainvoke, evaluate, update). Возвращает `RunResult`.
- `experiment/cli.py`: `atm run --config <path> [+overrides]`.
- `storage/models.py`: `Run` has `host: str|None`, `process_pid: int|None` (M12-A).
- `topology/mesh.py`: dispatcher-based round-robin (one agent per super-step, not concurrent broadcast). By design.
- `topology/debate.py`: LangGraph super-step fan-out — planner → (debater_pro, debater_contra) in one tick. LangGraph schedules these concurrently within the super-step.

## Что строим

```
                  atm grid --config sweep.yaml --parallelism 4
                                │
                                ▼
              load_grid_configs(path) → list[ExperimentConfig]
                                │
                                ▼
                  run_grid(configs, parallelism=N, fail_fast=False)
                                │
                                ▼
            ProcessPoolExecutor(max_workers=N) — spawn child for each cell
                                │
        ┌────────────────┬──────┴──────┬────────────────┐
        ▼                ▼             ▼                ▼
   _run_cell_worker  _run_cell_worker ...           _run_cell_worker
   (process A)       (process B)                    (process N)
        │                │
        ▼                ▼
   asyncio.run(run_one(cfg))
        │
        ▼
   PG INSERT runs (status=running, host=hostname, process_pid=os.getpid())
   → ainvoke topology → evaluate → UPDATE status=completed/failed
        │
        ▼
   return {status, run_id, ...}  (picklable dict)
                                │
                                ▼
                    Main aggregates progress
                                │
                                ▼
              UPDATE experiments.status (completed|partial|failed)
                                │
                                ▼
                  Return GridResult to caller / CLI
```

## Picklability constraints

`ExperimentConfig` is Pydantic v2 — pickling Pydantic models через ProcessPoolExecutor может быть нестабильно (v2 had pickle issues с custom validators в ранних релизах). Стратегия: main pickles `cfg.model_dump(mode="python")` (plain dict), worker re-validates через `ExperimentConfig.model_validate(d)`. Это **гарантированно picklable** (только built-ins + UUID).

## Host/PID capture

В архитектуре дальше (m12-resume-replay) нужно знать, какой worker зарегистрировал run, чтобы reconcile (детектировать crashed workers по mismatch с running PIDs). Самый чистый путь — расширить `run_one` чтобы оно само писало `os.getpid()` + `socket.gethostname()` сразу после `_insert_run`. Это и в single-run полезно (легче дебажить логи). Этот хелпер — простой `UPDATE Run SET host=..., process_pid=... WHERE id=:rid`.

## Audit (Mesh + Debate)

**Mesh:** Round-robin (`dispatcher_node` → один `_make_agent_node` per super-step → `mesh_broadcast`). Параллельности агентов нет — by design, для определённости семантики consensus voting. Документируется тестом-комментарием.

**Debate:** `planner → debater_pro / debater_contra` — LangGraph фан-аут в одном super-step'е. LangGraph runtime запускает both nodes concurrently (через `asyncio.gather` под капотом, см. langgraph internals). Это уже работает; тест верифицирует через `FakeLLM` instrumentation что start-timestamps debater_pro и debater_contra близки.

## Внешние ссылки

- `arch.md` §11 (storage), §12 (experiment runner).
- `PLAN.md` §M12 lines 651-654.
- Python docs: `concurrent.futures.ProcessPoolExecutor`, `asyncio.get_event_loop().run_in_executor`.
- LangGraph: super-step semantics.

## Тестовая стратегия

- **Unit:** dataclass derivation, status aggregation rules, worker picklability, fail_fast cancellation logic (mockованный pool).
- **Integration (PG):**
  - `test_grid_mini.py`: 2×2×1 = 4 cells, FakeLLM, parallelism=2, проверяем что все 4 строки `runs` присутствуют, `host`/`process_pid` populated, `experiments.status='completed'`.
  - `test_grid_fail_fast.py`: monkeypatch `run_one` чтобы одна cell бросала, fail_fast=True; проверяем что pending future cancelled (counted as not-started) или failed.

PG-gated через `ATM_ENABLE_PG_TESTS=1`. Без PG локально — `pytest --collect-only` валидирует skip.
