# M13 — Analysis Tooling — Context

## SESSION PROGRESS (2026-05-15)

### COMPLETED
- Step 1.1: Added `[project.optional-dependencies] analysis` group (matplotlib>=3.9, nbformat>=5.10, pandas>=2.2, seaborn>=0.13) and same direct entries in `[dependency-groups] dev` in pyproject.toml. `uv sync --extra analysis` and `uv sync --group dev` both exit 0; all four imports verified.
- Step 2.1: Created `loaders.py` (6 loader stubs), `metrics.py` (7 metric stubs), `plots.py` (9 plot stubs with `matplotlib.use("Agg")` as first statement). Updated `__init__.py` with sorted `__all__` of 26 symbols (4 oracle + 7 metrics + 9 plots + 6 loaders). All imports verified; `pytest tests/unit/analysis/test_oracle.py` green (9 passed); ruff clean.
- Step 3.1: Implemented `load_experiment` (PG-backed async loader returning dict; raises KeyError on missing ID) and `load_runs` (PG-backed async loader returning DataFrame with derived `task_type` column via `_infer_task_type`; casts Decimal to float). Created `tests/unit/analysis/test_loaders.py` with 8 PG-gated tests (requires_postgres marker); all 8 pass with ATM_ENABLE_PG_TESTS=1 (PG at localhost:5433); skip cleanly without the flag.
- Step 4.1: Implemented `load_llm_calls` (glob `experiments/*/runs/{run_id}/llm_calls.parquet`, raises FileNotFoundError when missing) and `load_llm_calls_for_experiment` (glob all run dirs under experiment, concat with run_id injection from directory name). Added 7 parquet round-trip tests (no PG required); all 7 pass.
- Step 5.1: Implemented `load_topology_transitions` (PG+parquet; parquet path decodes `*_json` columns via `json.loads()`), `load_phases` (PG+parquet), `load_human_interactions` (PG-only; `raw_tlx_score` None→NaN). Added 20 tests: 10 parquet (always-run), 8 PG-gated, 2 column-set parity. All pass (26 passed, 18 skipped without PG; 35 passed with ATM_ENABLE_PG_TESTS=1).
- Step 6.1: Implemented all 7 metric functions in `src/atm/analysis/metrics.py` (pure functions, no side effects). Created `tests/unit/analysis/test_metrics.py` with 32 unit tests covering happy path + edge cases for all 7 functions; divergent-fixture test confirms per-task_id filtering in `compute_oracle_gap_loo`. All 32 pass + 9 oracle tests pass (41 total).
- Step 7.1: Implemented `plot_pareto` (quality vs cost scatter with numpy bootstrap 95% CI error bars, adaptive highlighted with star marker), `plot_topology_task_heatmap` (seaborn heatmap with lazy import), `plot_phase_timeline` (broken_barh Gantt chart per run_id) in `src/atm/analysis/plots.py`. Created `tests/unit/analysis/test_plots.py` with 28 smoke tests (figure returns, Axes counts, labels, edge cases for empty/missing columns). `matplotlib.use("Agg")` is first call. autouse fixture closes all figures after each test. All 28 pass; ruff clean.
- Step 8.1: Implemented 5 RQ2/G11 plot functions in `src/atm/analysis/plots.py`: `plot_transition_timeline_quality` (step plot + twinx quality overlay per run), `plot_guard_override_rate` (horizontal barh, override rate per to_topology), `plot_router_cost_share` (stacked bar router vs worker cost per run), `plot_time_per_topology` (mean duration barh per topology; fallback to count when no duration column), `plot_oracle_gap_loo` (histogram + axvline at 0; accepts Series or DataFrame). Added 41 RQ2/G11 smoke tests to `test_plots.py` (7-8 per function + 5 exports checks); all 69 tests pass; ruff clean.

### COMPLETED
- Step 9.1: Implemented `plot_cognitive_load_boxplot(runs_df, human_interactions_df, *, role=None)` in plots.py (2-panel figure: raw_tlx_score boxplot + cognitive_load_proxy boxplot using pure matplotlib ax.boxplot to avoid seaborn PendingDeprecationWarning). Implemented `plot_oracle_vs_router(runs_df, oracle_table, *, router_col="topology")` in oracle.py with `from __future__ import annotations` + `TYPE_CHECKING` guard (matplotlib imported lazily inside function body only). Added `plot_oracle_vs_router` to `__init__.py` exports. Created `test_oracle_plot.py` (12 tests) and `test_oracle_no_matplotlib_import.py` (4 tests using subprocess isolation). Added 11 cognitive load tests to `test_plots.py`. 154 passed, 18 skipped (PG-gated).

- Step 11.1: Normalized `__all__` in `__init__.py` (28 symbols, alphabetically sorted). Fixed lint/type issues: RUF022 (sorted __all__), UP037 (remove quotes from return type in oracle.py), B905 (zip strict= in plots.py), plt.Axes→matplotlib.axes.Axes, aggfunc type ignore, int(Hashable) cast, unstack fill_value int, pyarrow no-untyped-call type ignores. Added pandas-stubs to dev deps. Added per-file-ignores for test_oracle_no_matplotlib_import.py. ruff check/format: clean; mypy: clean; pytest: 159 passed, 18 skipped (PG-gated); full suite: 1759 passed.

- Lint run (m13 surface): auto-fixed `scripts/gen_analysis_notebook.py` ruff format + 8 mypy `no-untyped-call`/`no-any-return` errors from untyped nbformat API (added `# type: ignore` annotations). All clean: ruff check, ruff format, mypy 0 errors on 6 source files.

### IN PROGRESS
- Step 12.1: Codebase-map update (running in parallel)

### BLOCKERS
- None

## Quick Resume

1. Прочитай этот файл
2. Проверь `m13-tasks.md` — что следующее по чеклисту
3. Прочитай `m13-plan.md` Phase 1 для стратегии
4. Начни с: Step 1 — добавить optional-dep group `analysis` + direct dev deps в `pyproject.toml`

## Key Files

**`/home/cactustim/agents/feat/m13/pyproject.toml`**
- Role: управление зависимостями проекта (uv)
- Planned change: новый `[project.optional-dependencies] analysis` с matplotlib/seaborn/nbformat/pandas; те же пакеты добавить прямо в `[dependency-groups] dev`
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/src/atm/analysis/__init__.py`**
- Role: публичный API пакета `atm.analysis`
- Planned change: добавить `__all__` с 23 символами (6 loaders + 7 metrics + 10 plots)
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/src/atm/analysis/loaders.py`**
- Role: загрузчики данных из Postgres и Parquet в pandas DataFrame
- Planned change: новый файл — 6 функций: `load_experiment`, `load_runs`, `load_llm_calls`, `load_topology_transitions`, `load_phases`, `load_human_interactions`
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/src/atm/analysis/metrics.py`**
- Role: чистые функции производных метрик из experiment_plan.md §4
- Planned change: новый файл — 7 функций: `compute_guard_override_rate`, `compute_router_cost_share`, `compute_time_per_topology`, `compute_oracle_gap_loo`, `compute_oracle_gap_manual`, `compute_hurt_rate`, `compute_topology_switch_counts`
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/src/atm/analysis/plots.py`**
- Role: plot-функции для RQ1/RQ2/RQ3/RQ4; принимают DataFrame, возвращают Figure
- Planned change: новый файл — 9 функций (10-я `plot_oracle_vs_router` живёт в `oracle.py`); `matplotlib.use("Agg")` до `import matplotlib.pyplot as plt` — первые строки модуля
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/src/atm/analysis/oracle.py`**
- Role: OracleTable, LOO-builder, load_oracle_table (M8.7 done)
- Planned change: добавить `plot_oracle_vs_router` с `TYPE_CHECKING`-паттерном (matplotlib import только внутри функции, не на top-level)
- Status: NOT STARTED (existing M8.7 content untouched)

**`/home/cactustim/agents/feat/m13/scripts/gen_analysis_notebook.py`**
- Role: CLI-скрипт генерации `notebooks/analysis_template.ipynb`
- Planned change: новый файл; флаг `--check` для CI drift-detection
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/notebooks/analysis_template.ipynb`**
- Role: шаблонный notebook для анализа результатов экспериментов
- Planned change: новый файл, генерируется скриптом; секции RQ1/RQ2/RQ3/RQ4
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/tests/unit/analysis/test_loaders.py`**
- Role: unit + PG-gated тесты для всех 6 loaders
- Planned change: новый файл
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/tests/unit/analysis/test_metrics.py`**
- Role: unit-тесты для 7 metric-функций (≥14 тестов)
- Planned change: новый файл
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/tests/unit/analysis/test_plots.py`**
- Role: smoke-тесты для 9 plot-функций в `plots.py`
- Planned change: новый файл
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/tests/unit/analysis/test_oracle_plot.py`**
- Role: тесты для `plot_oracle_vs_router` в `oracle.py`
- Planned change: новый файл
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/tests/unit/analysis/test_oracle_no_matplotlib_import.py`**
- Role: регрессионный тест — oracle.py не импортирует matplotlib на top-level
- Planned change: новый файл; чистит `sys.modules` и проверяет `"matplotlib" not in sys.modules`
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/tests/unit/analysis/test_notebook_generator.py`**
- Role: 4 теста для генератора: generates, idempotent, committed-matches, rq2-covers-all-metrics
- Planned change: новый файл
- Status: NOT STARTED

**`/home/cactustim/agents/feat/m13/dev/codebase-map.md`**
- Role: карта кодовой базы для разработчиков
- Planned change: добавить M13 analysis surface; исправить устаревшие колонки Run (reviewer N6): `created_at`→`started_at`, `exit_code`→`finish_reason`, убрать `tokens_in`/`cost_usd`, добавить `budget_spent_usd`
- Status: NOT STARTED

## Decisions

**Matplotlib Agg backend**
- Decision: `matplotlib.use("Agg")` форсируется первыми строками `plots.py` до `import matplotlib.pyplot as plt`
- Rationale: `pytest.ini_options filterwarnings=["error"]` превращает GUI-предупреждения в ошибки; headless CI не имеет дисплея

**oracle.py остаётся import-лёгким**
- Decision: `plot_oracle_vs_router` добавляется в `oracle.py`, но matplotlib импортируется только внутри функции через `TYPE_CHECKING`-паттерн
- Rationale: `oracle.py` используется M8.7 router-кодом в runtime; тянуть matplotlib в runtime недопустимо; regression test (Step 9) закрепляет это

**Loaders без внутренних вызовов plot/metrics**
- Decision: plot-функции принимают готовые DataFrame, никогда не вызывают loaders внутри
- Rationale: детерминированная тестируемость; notebook-ячейки явно показывают data flow

**Dual-source loaders (PG + parquet)**
- Decision: `load_topology_transitions` и `load_phases` поддерживают `source=Literal["pg","parquet"]` с соответствующим kwarg
- Rationale: parquet-поток доступен без PG (оффлайн анализ); оба источника должны давать идентичные column sets

**Sequenced loader steps (не параллельно)**
- Decision: Steps 3→4→5 последовательны, хотя логически независимы
- Rationale: все три модифицируют одни и те же файлы `loaders.py` + `test_loaders.py`; merge-конфликты не стоят экономии времени

**compute_oracle_gap_loo — per-task_id, не per-task_type**
- Decision: для каждого запуска с `task_id=t` oracle-качество ищется через `runs_df[runs_df["task_id"] == t]`, NOT `task_type`
- Rationale: reviewer N2 явно потребовал; включён divergent-fixture тест

**compute_hurt_rate — caller вычисляет best_static_df**
- Decision: `compute_hurt_rate(runs_df, best_static_df)` требует предвычисленный best_static_df от caller
- Rationale: тестируемость (вставить любой fixture); избегает hardcoded список static топологий внутри функции

**Bootstrap CI для Pareto (no scipy)**
- Decision: доверительные полосы через numpy bootstrap с `n_bootstrap=1000`
- Rationale: scipy не добавляется как зависимость; numpy уже в main deps; ~50ms на группу приемлемо для ≤20 групп

**Notebook-drift check по cell sources, не полным байтам**
- Decision: `test_committed_notebook_matches_generator` сравнивает только cell source-строки
- Rationale: kernel-metadata варьируется между машинами; content-только сравнение надёжнее

**seaborn — ленивый импорт**
- Decision: `import seaborn as sns` — внутри тел функций с `try/except ImportError: sns = None`
- Rationale: non-analysis потребители `atm.analysis` не платят import cost; seaborn опциональна для headless

## Constraints

- `conf/oracle/type_level_manual.yaml` содержит TODO-заглушки до M14+; `compute_oracle_gap_manual` возвращает NaN для таких задач — задокументированное поведение
- Phase end-timestamp column name нужно верифицировать в `models.py:233-280` первым действием Step 5 (скорее всего `ended_at`)
- `task_type` нет в таблице `runs` — выводится через `_infer_task_type(task_id)` в `load_runs`; schema-change не входит в M13
- `filterwarnings=["error"]` в pytest — любое matplotlib-предупреждение = падение теста
