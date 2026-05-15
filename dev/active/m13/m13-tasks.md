# M13 — Analysis Tooling — Tasks

## Wave 1 — Dependencies (Step 1) COMPLETE

- [x] 1.1 Добавить optional-dep group `analysis` + прямые записи в `[dependency-groups] dev`
  - Type: simple
  - Depends On: —
  - Can-Parallel-With: Step 2 (разные части проекта)
  - File: `pyproject.toml`
  - Acceptance: `uv sync --extra analysis` exit 0; `python -c "import matplotlib, seaborn, nbformat, pandas"` работает; `uv sync --group dev` также включает эти пакеты

## Wave 2 — Skeleton (Step 2) COMPLETE

- [x] 2.1 Создать `loaders.py`, `metrics.py`, `plots.py` (стабы); обновить `__init__.py`
  - Type: simple
  - Depends On: 1.1
  - Can-Parallel-With: —
  - Files: `src/atm/analysis/loaders.py` (new), `src/atm/analysis/metrics.py` (new), `src/atm/analysis/plots.py` (new), `src/atm/analysis/__init__.py`
  - Acceptance: `from atm.analysis import load_runs, plot_pareto, compute_hurt_rate, compute_oracle_gap_manual` — не падает; `pytest tests/unit/analysis/test_oracle.py` зелёный; `plots.py` начинается с `matplotlib.use("Agg")` до `import matplotlib.pyplot`

## Wave 3 — PG loaders: experiment + runs (Step 3) COMPLETE

- [x] 3.1 Реализовать `load_experiment` + `load_runs` (async PG) + тесты
  - Type: tdd
  - Depends On: 2.1
  - Can-Parallel-With: —
  - Files: `src/atm/analysis/loaders.py`, `tests/unit/analysis/test_loaders.py` (new)
  - Acceptance: `pytest tests/unit/analysis/test_loaders.py -m requires_postgres` зелёный при `ATM_ENABLE_PG_TESTS=1`; пропускается без него; DataFrame имеет колонку `task_type`

## Wave 4 — Parquet loader: llm_calls (Step 4) COMPLETE

- [x] 4.1 Реализовать `load_llm_calls` + `load_llm_calls_for_experiment` + тесты
  - Type: tdd
  - Depends On: 3.1
  - Can-Parallel-With: —
  - Files: `src/atm/analysis/loaders.py`, `tests/unit/analysis/test_loaders.py`
  - Acceptance: round-trip через `ParquetWriter`; столбцы == `LLM_CALL_SCHEMA.names`; multi-run glob конкатенирует с инжекцией `run_id`; `FileNotFoundError` при отсутствующем файле

## Wave 5 — Dual-source loaders: transitions + phases + human_interactions (Step 5) COMPLETE

- [x] 5.1 Реализовать `load_topology_transitions` (PG+parquet), `load_phases` (PG+parquet), `load_human_interactions` (PG-only) + тесты
  - Type: tdd
  - Depends On: 4.1
  - Can-Parallel-With: —
  - Files: `src/atm/analysis/loaders.py`, `tests/unit/analysis/test_loaders.py`
  - Acceptance: parquet-тесты всегда запускаются; PG-тесты гейтированы; `df["signals_snapshot"].iloc[0]` — dict; `raw_tlx_score` — float с NaN; PG и parquet дают идентичные column sets для transitions

## Wave 6 — Metrics (Step 6) COMPLETE  ← параллельно с Wave 6 Plot

- [x] 6.1 Реализовать 7 metric-функций + ≥14 unit-тестов
  - Type: tdd
  - Depends On: 5.1
  - Can-Parallel-With: Step 7 (disjoint files: `metrics.py`+`test_metrics.py` vs `plots.py`+`test_plots.py`)
  - Files: `src/atm/analysis/metrics.py`, `tests/unit/analysis/test_metrics.py` (new)
  - Acceptance: `pytest tests/unit/analysis/test_metrics.py -v` зелёный; divergent-fixture тест для `compute_oracle_gap_loo` проходит (per-task_id != per-task_type); docstring `compute_time_per_topology` явно описывает non-contiguous суммирование

## Wave 6 — RQ1 Plots (Step 7) NOT STARTED  ← параллельно с Wave 6 Metrics

- [ ] 7.1 Реализовать `plot_pareto`, `plot_topology_task_heatmap`, `plot_phase_timeline` + smoke-тесты
  - Type: tdd
  - Depends On: 2.1
  - Can-Parallel-With: Step 6 (disjoint files)
  - Files: `src/atm/analysis/plots.py`, `tests/unit/analysis/test_plots.py` (new)
  - Acceptance: `pytest tests/unit/analysis/test_plots.py::test_plot_pareto -v` зелёный; `isinstance(fig, plt.Figure)` и `len(fig.axes) >= 1`; `plt.close(fig)` в teardown; нет предупреждений

## Wave 7 — RQ2 Plots (Step 8) NOT STARTED

- [ ] 8.1 Реализовать `plot_transition_timeline_quality`, `plot_guard_override_rate`, `plot_router_cost_share`, `plot_time_per_topology`, `plot_oracle_gap_loo` + тесты
  - Type: tdd
  - Depends On: 7.1
  - Can-Parallel-With: —
  - Files: `src/atm/analysis/plots.py`, `tests/unit/analysis/test_plots.py`
  - Acceptance: `pytest tests/unit/analysis/test_plots.py -k "rq2 or g11" -v` зелёный; twinx legend имеет ≥2 entries

## Wave 8 — Cognitive-load + oracle plot + import guard (Step 9) NOT STARTED

- [ ] 9.1 Реализовать `plot_cognitive_load_boxplot` (plots.py) + `plot_oracle_vs_router` (oracle.py) + TYPE_CHECKING + import-bloat regression test
  - Type: tdd
  - Depends On: 8.1
  - Can-Parallel-With: —
  - Files: `src/atm/analysis/plots.py`, `src/atm/analysis/oracle.py`, `src/atm/analysis/__init__.py`, `tests/unit/analysis/test_plots.py`, `tests/unit/analysis/test_oracle_plot.py` (new), `tests/unit/analysis/test_oracle_no_matplotlib_import.py` (new)
  - Acceptance: `test_oracle_module_does_not_import_matplotlib` зелёный; `plot_cognitive_load_boxplot` — Figure с 2 Axes; NaN-строки фильтруются без ошибок

## Wave 9 — Notebook generator (Step 10) NOT STARTED

- [ ] 10.1 Написать `scripts/gen_analysis_notebook.py` + сгенерировать и скоммитить `notebooks/analysis_template.ipynb` + тесты
  - Type: tdd
  - Depends On: 9.1
  - Can-Parallel-With: —
  - Files: `scripts/gen_analysis_notebook.py` (new), `notebooks/analysis_template.ipynb` (new), `tests/unit/analysis/test_notebook_generator.py` (new)
  - Acceptance: `pytest tests/unit/analysis/test_notebook_generator.py -v` зелёный (4 теста); `nbformat.validate()` проходит; все 8 RQ2-символов присутствуют в RQ2-ячейках; notebook идемпотентен

## Wave 10 — Lint + exports (Step 11) NOT STARTED  ← параллельно с Wave 10 Docs

- [ ] 11.1 Нормализовать `__all__` (23 символа) в `__init__.py`; прогнать ruff + mypy + pytest
  - Type: simple
  - Depends On: 10.1
  - Can-Parallel-With: Step 12 (disjoint files)
  - File: `src/atm/analysis/__init__.py`
  - Acceptance: `uv run ruff check src/atm/analysis tests/unit/analysis` exit 0; `uv run mypy src/atm/analysis` exit 0; `uv run pytest tests/unit/analysis -v` exit 0; `uv run pytest -q` (весь suite) exit 0

## Wave 10 — Codebase-map update (Step 12) NOT STARTED  ← параллельно с Wave 10 Lint

- [ ] 12.1 Обновить `dev/codebase-map.md`: M13 analysis surface + исправить колонки Run
  - Type: simple
  - Depends On: 10.1
  - Can-Parallel-With: Step 11 (disjoint file)
  - File: `dev/codebase-map.md`
  - Acceptance: Run-раздел показывает реальные колонки (`started_at`, `finish_reason`, `budget_spent_usd`); нет `created_at`, `exit_code`, `tokens_in`, `cost_usd`; M13 loaders/metrics/plots секция добавлена

---

## Stats

- Total: 12 tasks (Steps 1–12, один task per step) · ~10h
- Done: 6 / 12

## How to Update

После каждого завершённого шага:
1. Замени `[ ]` на `[x]` для выполненных tasks
2. Обнови заголовок Phase: все done → `COMPLETE`, часть → `IN PROGRESS`
3. Обнови счётчик Done в Stats
4. Замени SESSION PROGRESS в `m13-context.md`
