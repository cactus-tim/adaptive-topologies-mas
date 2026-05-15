# M13 — Analysis Tooling (loaders, plots, oracle, template notebook) — Plan

## Executive Summary

Построить слой анализа, который превращает сохранённые артефакты экспериментов
(таблицы Postgres + потоки Parquet) в pandas-DataFrame, графики RQ1/RQ2/RQ3/RQ4,
производные G11-метрики и шаблонный Jupyter-notebook для дипломной защиты.

## Current State

- `src/atm/analysis/` уже существует с `oracle.py` (M8.7 завершён): `OracleTable`,
  `build_loo_from_rows`, `build_leave_one_out_oracle`, `load_oracle_table`,
  `_infer_task_type`.
- `notebooks/` содержит только `.gitkeep`.
- `pyproject.toml` имеет `pandas>=2.2` в `dev` и `pyarrow>=16` в main deps;
  matplotlib/seaborn/nbformat отсутствуют.
- Все upstream-схемы (Run, TopologyTransition, HumanInteraction с `raw_tlx_score`,
  `runs.cognitive_load_proxy`, Phase, parquet-раскладка) подтверждены в `f4a26ef`.

## Proposed Approach

- Новые модули: `loaders.py`, `metrics.py`, `plots.py` в `src/atm/analysis/`.
- Расширение `oracle.py` через `TYPE_CHECKING`-паттерн (без matplotlib на top-level).
- Скрипт-генератор `scripts/gen_analysis_notebook.py` + коммитированный
  `notebooks/analysis_template.ipynb`.
- Unit-тесты для каждой публичной функции (≥1 happy path + ≥1 edge case).
- PG-тесты гейтированы через `ATM_ENABLE_PG_TESTS=1` и `ephemeral_pg_dsn`.
- Agg-бэкенд matplotlib форсируется в `plots.py` первыми строками модуля.
- Seaborn импортируется лениво внутри тел функций.
- Metrics — чистые функции без побочных эффектов; plots принимают готовые DataFrame.

## Implementation Phases

### Phase 1: Dependencies (~0.5h)
**Goal:** Добавить matplotlib/seaborn/nbformat в pyproject.toml.

- [ ] 1.1 Добавить optional-dep group `analysis` + прямые записи в `[dependency-groups] dev`
  - File: `pyproject.toml`
  - Acceptance: `uv sync --extra analysis` exit 0; `python -c "import matplotlib, seaborn, nbformat"` работает в синкнутом venv; `uv sync --group dev` тоже включает эти пакеты

### Phase 2: Skeleton (~0.5h)
**Goal:** Создать заглушки модулей и обновить `__init__.py`.

- [ ] 2.1 Создать `loaders.py`, `metrics.py`, `plots.py` (стабы с `NotImplementedError`); обновить `__init__.py`
  - Files: `src/atm/analysis/loaders.py`, `src/atm/analysis/metrics.py`, `src/atm/analysis/plots.py`, `src/atm/analysis/__init__.py`
  - Acceptance: `from atm.analysis import load_runs, plot_pareto, compute_hurt_rate` — ImportError не падает; `pytest tests/unit/analysis/test_oracle.py` зелёный

### Phase 3: PG loaders — experiment + runs (~1h)
**Goal:** Реализовать первые два async PG-загрузчика с TDD.

- [ ] 3.1 Реализовать `load_experiment` + `load_runs` (PG-backed) + тесты
  - Files: `src/atm/analysis/loaders.py`, `tests/unit/analysis/test_loaders.py`
  - Acceptance: `pytest tests/unit/analysis/test_loaders.py -m requires_postgres` зелёный при `ATM_ENABLE_PG_TESTS=1`; пропускается без него

### Phase 4: Parquet loader — llm_calls (~0.75h)
**Goal:** Реализовать parquet-загрузчик llm_calls.

- [ ] 4.1 Реализовать `load_llm_calls` + `load_llm_calls_for_experiment` + тесты
  - Files: `src/atm/analysis/loaders.py`, `tests/unit/analysis/test_loaders.py`
  - Acceptance: round-trip test c `tmp_path` и fixture через `ParquetWriter`; столбцы совпадают с `LLM_CALL_SCHEMA.names`; multi-run glob конкатенирует с инжекцией `run_id`

### Phase 5: Dual-source loaders — transitions + phases + human interactions (~1.25h)
**Goal:** Реализовать три оставшихся загрузчика (G11 + RQ4).

- [ ] 5.1 Реализовать `load_topology_transitions`, `load_phases`, `load_human_interactions` + тесты
  - Files: `src/atm/analysis/loaders.py`, `tests/unit/analysis/test_loaders.py`
  - Acceptance: PG-тесты и parquet-тесты проходят; `df["signals_snapshot"].iloc[0]` — dict, не JSON-строка; `raw_tlx_score` — float с NaN для пустых строк

### Phase 6: Metrics (parallel with Phase 7) (~1.5h)
**Goal:** Реализовать все 7 производных метрик из experiment_plan.md §4.

- [ ] 6.1 Реализовать 7 metric-функций + тесты (≥14 unit-тестов)
  - Files: `src/atm/analysis/metrics.py`, `tests/unit/analysis/test_metrics.py`
  - Acceptance: `pytest tests/unit/analysis/test_metrics.py -v` зелёный; тест `oracle_gap_loo` divergent-fixture подтверждает per-`task_id` фильтрацию

### Phase 7: RQ1 plots (parallel with Phase 6) (~1h)
**Goal:** Реализовать три RQ1 plot-функции.

- [ ] 7.1 Реализовать `plot_pareto`, `plot_topology_task_heatmap`, `plot_phase_timeline` + тесты
  - Files: `src/atm/analysis/plots.py`, `tests/unit/analysis/test_plots.py`
  - Acceptance: `pytest tests/unit/analysis/test_plots.py::test_plot_pareto -v` зелёный; нет предупреждений при `filterwarnings=["error"]`

### Phase 8: RQ2 plots (~1.25h)
**Goal:** Реализовать пять G11/RQ2 plot-функций.

- [ ] 8.1 Реализовать `plot_transition_timeline_quality`, `plot_guard_override_rate`, `plot_router_cost_share`, `plot_time_per_topology`, `plot_oracle_gap_loo` + тесты
  - Files: `src/atm/analysis/plots.py`, `tests/unit/analysis/test_plots.py`
  - Acceptance: `pytest tests/unit/analysis/test_plots.py -k "rq2 or g11" -v` зелёный

### Phase 9: Cognitive-load plot + oracle plot + import guard (~1h)
**Goal:** Реализовать последние plot-функции + TYPE_CHECKING-паттерн + регрессионный тест.

- [ ] 9.1 Реализовать `plot_cognitive_load_boxplot` (plots.py), `plot_oracle_vs_router` (oracle.py) + TYPE_CHECKING + import-bloat тест
  - Files: `src/atm/analysis/plots.py`, `src/atm/analysis/oracle.py`, `src/atm/analysis/__init__.py`, `tests/unit/analysis/test_plots.py`, `tests/unit/analysis/test_oracle_plot.py`, `tests/unit/analysis/test_oracle_no_matplotlib_import.py`
  - Acceptance: `test_oracle_module_does_not_import_matplotlib` зелёный; `plot_cognitive_load_boxplot` — Figure с 2 Axes

### Phase 10: Notebook generator (~1.25h)
**Goal:** Написать скрипт-генератор и скоммитить шаблонный notebook.

- [ ] 10.1 Написать `scripts/gen_analysis_notebook.py` + сгенерировать `notebooks/analysis_template.ipynb` + тесты
  - Files: `scripts/gen_analysis_notebook.py`, `notebooks/analysis_template.ipynb`, `tests/unit/analysis/test_notebook_generator.py`
  - Acceptance: `pytest tests/unit/analysis/test_notebook_generator.py -v` зелёный (4 теста); RQ2-cell содержит все 8 символов из §4

### Phase 10 (parallel): Lint + exports (~0.5h)
**Goal:** Финальный `__init__.py` + lint/type/test gate.

- [ ] 11.1 Нормализовать `__all__` (23 символа), прогнать ruff + mypy + pytest
  - Files: `src/atm/analysis/__init__.py`
  - Acceptance: все команды exit 0: ruff check, ruff format --check, mypy, pytest (с и без PG)

### Phase 10 (parallel): Codebase-map update (~0.25h)
**Goal:** Задокументировать M13 и исправить устаревшие колонки Run.

- [ ] 12.1 Обновить `dev/codebase-map.md`: M13 surface + исправить колонки Run
  - File: `dev/codebase-map.md`
  - Acceptance: Run-раздел показывает `started_at, finish_reason, budget_spent_usd` (без `created_at, exit_code, tokens_in, cost_usd`); M13 analysis section присутствует

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `pyproject.toml` | Добавить optional group `analysis` + direct dev deps | matplotlib/seaborn/nbformat нужны для plots/notebook |
| `src/atm/analysis/__init__.py` | Обновить `__all__` — 23 символа | Публичный API analysis-слоя |
| `src/atm/analysis/loaders.py` | Новый файл — 6 loader-функций | Загрузка артефактов из PG и Parquet |
| `src/atm/analysis/metrics.py` | Новый файл — 7 metric-функций | Производные RQ2/G11 метрики из experiment_plan.md §4 |
| `src/atm/analysis/plots.py` | Новый файл — 9 plot-функций | Визуализация RQ1/RQ2/RQ3/RQ4 |
| `src/atm/analysis/oracle.py` | Расширить `plot_oracle_vs_router` + TYPE_CHECKING | G9 plot рядом с типом данных; oracle.py остаётся import-лёгким |
| `scripts/gen_analysis_notebook.py` | Новый файл — генератор notebook | Воспроизводимая генерация шаблона |
| `notebooks/analysis_template.ipynb` | Новый файл — генерированный шаблон | Единая точка входа для анализа результатов |
| `tests/unit/analysis/test_loaders.py` | Новый файл | TDD для loaders |
| `tests/unit/analysis/test_metrics.py` | Новый файл | TDD для metrics |
| `tests/unit/analysis/test_plots.py` | Новый файл | TDD для plots |
| `tests/unit/analysis/test_oracle_plot.py` | Новый файл | TDD для plot_oracle_vs_router |
| `tests/unit/analysis/test_oracle_no_matplotlib_import.py` | Новый файл | Регрессионный тест import-bloat |
| `tests/unit/analysis/test_notebook_generator.py` | Новый файл | TDD для генератора |
| `dev/codebase-map.md` | Обновить — M13 surface + исправить Run columns | Баг reviewer N6 |

## Dependencies & Order Constraints

Wave-порядок (последовательный там, где файлы конфликтуют):

```
Wave 1:  Step 1 (pyproject)
Wave 2:  Step 2 (skeleton)
Wave 3:  Step 3 (loaders: load_experiment + load_runs)
Wave 4:  Step 4 (loaders: load_llm_calls)
Wave 5:  Step 5 (loaders: transitions + phases + human_interactions)
Wave 6:  Step 6 (metrics) ‖ Step 7 (RQ1 plots)   ← disjoint files, параллельно
Wave 7:  Step 8 (RQ2 plots)
Wave 8:  Step 9 (cognitive-load + oracle + import guard)
Wave 9:  Step 10 (notebook generator)
Wave 10: Step 11 (lint/init) ‖ Step 12 (codebase-map)  ← disjoint files, параллельно
```

Steps 3–5 принудительно последовательны (один файл `loaders.py` + `test_loaders.py`) по рекомендации reviewer для избежания merge-конфликтов.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| matplotlib Agg-backend не форсирован → GUI-warning → test failure | Medium | High | `matplotlib.use("Agg")` первыми строками `plots.py` (Step 2) |
| Decimal/UUID типы утекают в pandas → object-dtype | Medium | Medium | Явные касты в loaders + dtype-assertions в тестах |
| JSONB-поля из parquet — строки вместо dict/list | High | High | JSON-decode в parquet-path loaders; тесты в обоих источниках |
| oracle.py начинает импортировать matplotlib → регрессия | Low | Medium | `TYPE_CHECKING` паттерн + `test_oracle_no_matplotlib_import` (Step 9) |
| `conf/oracle/type_level_manual.yaml` содержит TODO → NaN | High (now) | Low | Документированное поведение; M14+ заполнит файл |
| Дрейф notebook vs генератор | Medium | Low | `test_committed_notebook_matches_generator` (Step 10) |
| `Phase` конец-timestamp имя колонки не `ended_at` | Low | Medium | Прочитать `models.py:233-280` первым действием Step 5 |

## Out of Scope

- Cohen's d и paired t-test — в ячейках notebook через scipy.stats, не в `metrics.py`
- HTTP/gRPC endpoints — M13 только read-side analysis
- Alembic migrations — нет изменений схемы
- Изменения runtime (`run_one`, CLI, topology, orchestrator)
- Integration tests (Needs Integration Tests = no)
- `jupyter` как зависимость (только `nbformat`; devs ставят jupyter сами)

## Timeline

- Total: ~10h
- Steps: 12 steps, 10 waves
- Created: 2026-05-15
