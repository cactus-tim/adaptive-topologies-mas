# M11 — Evaluation Framework — Plan

## Executive Summary

Построить пакет `src/atm/evaluation/` с полным набором метрик для RQ1–RQ4 дипломной работы. Пакет состоит из пяти новых модулей, которые компонуют поверх M10 EVALUATORS (не дублируют), добавляют post-hoc LLM-судей (rubric / pairwise с проверкой позиционного смещения / self-consistency), агрегатор качества, NASA-TLX модель и чистые функции метрик. Агрегатор вшивается в `experiment/runner.py` взамен заглушки M6, записывая `runs.quality_score` — контракт, который читают M12 и M13.

## Current State

- M10 полностью слит; в `src/atm/tasks/` работают четыре эвалюатора (`humaneval_pytest`, `mmlu_exact_match`, `creative_judge`, `analysis_hybrid`) с Protocol `Evaluator` и `EvalResult.score ∈ [0,1]`.
- `src/atm/experiment/_evaluator.py` содержит заглушку M6 (`evaluate(...) → 1.0 if "55" in answer else 0.0`), которая должна быть удалена.
- `runs.quality_score` (DOUBLE PRECISION nullable) существует в схеме БД; раннер обновляет колонку в `_update_run_success` / `_update_run_failed`.
- `human_interactions.tlx_scores` (JSONB) и `raw_tlx_score` (DOUBLE PRECISION) существуют; M9 пишет их inline; M11 даёт только модель и хелпер.
- `LLMWrapper` + `BudgetTracker` работают; трекер живёт одну итерацию `run_one` — judge-вызовы из агрегатора корректно атрибутируются к RUN-бюджету.
- `SubprocessSandbox` в `src/atm/tools/sandbox/subprocess_sandbox.py` — дёшево создавать per-run.

## Proposed Approach

1. **Compose, not duplicate**: `ground_truth.py` — тонкий фасад над M10; `judges.py` добавляет только post-hoc протоколы (pairwise + self-consistency), которых в M10 нет.
2. **Flat dispatch** в `ground_truth.py` (`if/elif` вместо registry lookup) — полностью type-checkable.
3. **Pairwise position-bias mitigation** (arXiv 2406.07791): swap test (AB + BA); победитель объявляется только при совпадении обоих порядков; явная функция `_to_answer_relative(slot_winner, swapped)` исключает ошибки перевода.
4. **Self-consistency seeding**: seed для каждого вызова `i` = `hash((run_seed, i)) & 0xFFFFFFFF`, обеспечивает воспроизводимость.
5. **Aggregator never crashes**: `compute_quality` ловит все исключения и возвращает `(None, {"error": ...})`; раннер получает `None` и записывает SQL NULL — run finalisation не ломается.
6. **`resolve_spec`** (рек. #5) — отдельный хелпер в `atm.tasks.__init__`, возвращает `TaskSpec | None`; `None` = M6 inline-prompt путь → `quality_score=0.0` без вызова агрегатора.
7. **`EvaluationCfg`** (рек. #6) — новый раздел конфига (`judge_model`, `judge_self_consistency_n`); `Field(default_factory=...)` → обратная совместимость.
8. **TDD на каждый шаг**; интеграционные тесты под `ATM_INTEGRATION_PG=1`; маркер `integration` (новый `requires_pg` не добавляем).

## Implementation Phases

### Phase 1: Scaffolding + TLX + Ground Truth + Judges (~3h)
**Goal:** Создать скелет пакета и три независимых модуля первого слоя.

- [ ] 1.1 Создать скелет пакета `src/atm/evaluation/` + реализовать `tlx.py` + написать 6 unit-тестов
  - Files: `src/atm/evaluation/__init__.py`, `src/atm/evaluation/tlx.py`, `tests/unit/evaluation/__init__.py`, `tests/unit/evaluation/test_tlx.py`
  - Acceptance: `uv run pytest tests/unit/evaluation/test_tlx.py -q` — 6 green; `uv run mypy --strict src/atm/evaluation/tlx.py` — clean

- [ ] 1.2 Реализовать `ground_truth.py` (фасад над M10) + 5 unit-тестов
  - Files: `src/atm/evaluation/ground_truth.py`, `tests/unit/evaluation/test_ground_truth.py`
  - Acceptance: `uv run pytest tests/unit/evaluation/test_ground_truth.py -q` — 5 green; mypy strict clean

- [ ] 1.3 Реализовать `judges.py` (rubric / self-consistent / pairwise с position-bias swap) + 10 unit-тестов + 3 yaml-фикстуры
  - Files: `src/atm/evaluation/judges.py`, `tests/unit/evaluation/test_judges.py`, `tests/fixtures/llm/m11_judge_pairwise_ab.yaml`, `tests/fixtures/llm/m11_judge_pairwise_swap_disagree.yaml`, `tests/fixtures/llm/m11_judge_self_consistency.yaml`
  - Acceptance: `uv run pytest tests/unit/evaluation/test_judges.py -q` — 10 green; mypy strict clean

### Phase 2: Metrics + Aggregator (~2.5h)
**Goal:** Чистые функции метрик и async-агрегатор с PG-записью.

- [ ] 2.1 Реализовать `metrics.py` (pure functions: quality / efficiency / time / human / RQ2) + 12 unit-тестов
  - Files: `src/atm/evaluation/metrics.py`, `tests/unit/evaluation/test_metrics.py`
  - Acceptance: `uv run pytest tests/unit/evaluation/test_metrics.py -q` — 12 green; mypy strict + ruff clean
  - Note: зависит от Phase 1.1 (импорт `NasaTLX`)

- [ ] 2.2 Реализовать `aggregator.py` + `atm.tasks.resolve_spec` + `EvaluationCfg` + тесты (unit + integration)
  - Files: `src/atm/evaluation/aggregator.py`, `src/atm/tasks/__init__.py` (add `resolve_spec`), `src/atm/experiment/config.py` (add `EvaluationCfg`), `tests/unit/evaluation/test_aggregator_pure.py`, `tests/unit/tasks/test_resolve_spec.py`, `tests/unit/experiment/test_config_evaluation_cfg.py`, `tests/integration/conftest.py` (new), `tests/integration/evaluation/__init__.py`, `tests/integration/evaluation/test_aggregator.py`
  - Acceptance: unit-тесты green; `ATM_INTEGRATION_PG=1 uv run pytest tests/integration/evaluation/test_aggregator.py -q` — 4 green; `ATM_INTEGRATION_PG=1 uv run pytest tests/integration/storage -q` — по-прежнему green; mypy strict clean
  - Note: зависит от Phase 1.2 (использует `score_ground_truth`)

### Phase 3: Runner Wiring (~1.5h)
**Goal:** Заменить M6-заглушку в `runner.py`; удалить `_evaluator.py`.

- [ ] 3.1 Подключить агрегатор в `experiment/runner.py` + удалить `src/atm/experiment/_evaluator.py` + тест wiring + аудит M6 smoke-тестов
  - Files: `src/atm/experiment/runner.py` (3 callsite swap + imports + judge wrapper + sandbox), `src/atm/experiment/_evaluator.py` (delete), `tests/unit/experiment/test_runner_evaluation_wiring.py`, `tests/unit/experiment/test_runner_*.py` (audit existing)
  - Acceptance: `uv run pytest tests/unit/experiment -q` — all green; `uv run mypy --strict src/atm/experiment/runner.py` — clean; `grep -rn "from atm.experiment._evaluator" src/ tests/` — пусто

### Phase 4: Finalisation (~1h)
**Goal:** Финальные экспорты, e2e интеграционный тест, обновление codebase-map.

- [ ] 4.1 Финализировать `src/atm/evaluation/__init__.py` + e2e integration test + обновить `dev/codebase-map.md`
  - Files: `src/atm/evaluation/__init__.py` (rewrite с `__all__`), `tests/integration/evaluation/test_aggregator_e2e.py`, `dev/codebase-map.md`
  - Acceptance: `uv run pytest tests/unit -q` — ~40 новых тестов green; `ATM_INTEGRATION_PG=1 uv run pytest tests/integration -q` — all green; `uv run mypy --strict src/atm` — clean; `uv run ruff check src/atm tests` — clean

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/evaluation/__init__.py` | создать → переписать | скелет пакета + итоговый `__all__` |
| `src/atm/evaluation/tlx.py` | создать | NasaTLX Pydantic + aggregate_tlx |
| `src/atm/evaluation/ground_truth.py` | создать | фасад над M10 EVALUATORS |
| `src/atm/evaluation/judges.py` | создать | rubric / pairwise / self-consistency |
| `src/atm/evaluation/metrics.py` | создать | pure functions, RQ1–RQ4 метрики |
| `src/atm/evaluation/aggregator.py` | создать | compute_quality + persist_quality + aggregate_run |
| `src/atm/tasks/__init__.py` | изменить | добавить `resolve_spec` |
| `src/atm/experiment/config.py` | изменить | добавить `EvaluationCfg` + поле на `ExperimentConfig` |
| `src/atm/experiment/runner.py` | изменить | 3 callsite swap + judge wrapper + sandbox + импорты |
| `src/atm/experiment/_evaluator.py` | удалить | заглушка M6 заменена агрегатором |
| `tests/unit/evaluation/__init__.py` | создать | пакет тестов + import atm.tasks |
| `tests/unit/evaluation/test_tlx.py` | создать | 6 тестов |
| `tests/unit/evaluation/test_ground_truth.py` | создать | 5 тестов |
| `tests/unit/evaluation/test_judges.py` | создать | 10 тестов |
| `tests/unit/evaluation/test_metrics.py` | создать | 12 тестов |
| `tests/unit/evaluation/test_aggregator_pure.py` | создать | 4 теста (no PG) |
| `tests/unit/tasks/test_resolve_spec.py` | создать | 3 теста |
| `tests/unit/experiment/test_config_evaluation_cfg.py` | создать | 2 теста |
| `tests/unit/experiment/test_runner_evaluation_wiring.py` | создать | 1–2 теста |
| `tests/unit/experiment/test_runner_*.py` | аудит/правка | заменить "55-substring" assertions |
| `tests/integration/conftest.py` | создать | promote pg_dsn / pg_engine_fast / pg_engine_alembic / session_factory_fast |
| `tests/integration/storage/conftest.py` | изменить | thin re-export из родительского conftest |
| `tests/integration/evaluation/__init__.py` | создать | empty |
| `tests/integration/evaluation/test_aggregator.py` | создать | 4 PG-теста |
| `tests/integration/evaluation/test_aggregator_e2e.py` | создать | 1 e2e контрактный тест |
| `tests/fixtures/llm/m11_judge_pairwise_ab.yaml` | создать | FakeLLM скрипт для pairwise AB |
| `tests/fixtures/llm/m11_judge_pairwise_swap_disagree.yaml` | создать | FakeLLM скрипт, swap-disagree сценарий |
| `tests/fixtures/llm/m11_judge_self_consistency.yaml` | создать | 3 скриптовых ответа для self-consistency |
| `dev/codebase-map.md` | изменить | секция "Evaluation Framework (M11)" |

## Dependencies & Order Constraints

```
Волна 1 (параллельно): Step 1 (tlx) ‖ Step 2 (ground_truth) ‖ Step 3 (judges)
Волна 2 (параллельно): Step 4 (metrics, depends 1) ‖ Step 5 (aggregator, depends 2)
Волна 3 (последовательно): Step 6 (runner wiring, depends 5)
Волна 4 (последовательно): Step 7 (finalise, depends 1–6)
```

- Step 4 зависит от Step 1 (импорт `NasaTLX`).
- Step 5 зависит от Step 2 (импорт `score_ground_truth`).
- Step 6 зависит от Step 5 (агрегатор + `resolve_spec` + `EvaluationCfg` должны существовать).
- Step 7 зависит от всех предыдущих.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Slot→answer translation ошибка в pairwise (silent wrong tie/agree) | Medium | High | Явный тест `_to_answer_relative`; swap-consistent кейс явно различает slot-relative vs answer-relative winners |
| Продвижение `pg_engine_alembic` ломает storage-тесты | Low | High | `pytest --collect-only tests/integration/storage` после продвижения; storage conftest становится thin re-export |
| M6 smoke-тесты assert `quality_score == 1.0` | Medium | Medium | Аудит grep; замена на `in (0.0, None)` для inline-prompt fixtures |
| Реальный `openai:gpt-4o` judge вызывается в тестах (API key) | Low | Medium | Все тест-конфиги переопределяют `evaluation.judge_model = "fake:echo"` |
| `humaneval_pass_at_k` off-by-one | Low | Medium | Тест против reference-значения (n=20, c=2, k=10 → 0.6316) |
| `filterwarnings = "error"` в pyproject.toml (DeprecationWarning → fail) | Low | Low | mypy strict + ruff до коммита; no `import *` |
| `judge_self_consistency_n` добавлен в конфиг но не потребляется runner-ом в M11 | Low | Low | Документировано в Open Questions; plumbing готов, wiring — follow-up |

## Out of Scope

- TLX backfill: M9 пишет `raw_tlx_score` inline; M11 даёт только модель + хелпер, без backfill-хелпера.
- RQ2 PG-запросы в M11: `metrics.py` содержит pure functions; реальные запросы к `topology_transitions` — на M13.
- Self-consistency wiring в `run_one` (конфиг есть, runner пока не вызывает `SelfConsistentJudge` — deferred).
- Пирwise tie semantics в Bradley-Terry (M13).
- Новый pytest-маркер `requires_pg` — используем существующий `integration`.

## Timeline

- Total: ~8h
- Phases: 1 (3h) → 2 (2.5h) → 3 (1.5h) → 4 (1h)
- Created: 2026-05-12
