# M11 — Evaluation Framework — Tasks

## Phase 1: Scaffolding + TLX + Ground Truth + Judges — COMPLETE

- [x] 1.1 Создать скелет `src/atm/evaluation/` + реализовать `tlx.py` + 6 unit-тестов
  - Type: tdd
  - Depends On: none
  - Can-Parallel-With: 1.2, 1.3
  - Files: `src/atm/evaluation/__init__.py` (new placeholder), `src/atm/evaluation/tlx.py` (new), `tests/unit/evaluation/__init__.py` (new), `tests/unit/evaluation/test_tlx.py` (new)
  - Acceptance: `uv run pytest tests/unit/evaluation/test_tlx.py -q` — 6 green; `uv run mypy --strict src/atm/evaluation/tlx.py` — clean
  - Notes: `NasaTLX(frozen=True)` — 6 int fields ge=0 le=100; `@property raw_score` = `(mental + physical + temporal + (100-performance) + effort + frustration) / 6`; `aggregate_tlx([])` не делит на ноль

- [x] 1.2 Реализовать `ground_truth.py` (flat dispatch над M10) + 5 unit-тестов
  - Type: tdd
  - Depends On: none
  - Can-Parallel-With: 1.1, 1.3
  - Files: `src/atm/evaluation/ground_truth.py` (new), `tests/unit/evaluation/test_ground_truth.py` (new)
  - Acceptance: `uv run pytest tests/unit/evaluation/test_ground_truth.py -q` — 5 green; mypy strict clean
  - Notes: `async def score_ground_truth(spec, answer, *, sandbox=None, judge_llm=None) -> EvalResult`; flat if/elif по `spec.evaluator_key`; unknown key → `KeyError`; judge required but None → `ValueError`

- [x] 1.3 Реализовать `judges.py` + 10 unit-тестов + 3 yaml-фикстуры FakeLLM
  - Type: tdd
  - Depends On: none
  - Can-Parallel-With: 1.1, 1.2
  - Files: `src/atm/evaluation/judges.py` (new, ~150 LoC), `tests/unit/evaluation/test_judges.py` (new), `tests/fixtures/llm/m11_judge_pairwise_ab.yaml` (new), `tests/fixtures/llm/m11_judge_pairwise_swap_disagree.yaml` (new), `tests/fixtures/llm/m11_judge_self_consistency.yaml` (new)
  - Acceptance: `uv run pytest tests/unit/evaluation/test_judges.py -q` — 10 green; mypy strict clean
  - Notes: `PairwiseResult(winner, swap_consistent, reason_ab, reason_ba)`; `_to_answer_relative(slot_winner, swapped)`; self-consistency seed = `hash((run_seed, i)) & 0xFFFFFFFF`; score scale 0..10 (M10 convention); agent_ids: `rubric_judge`, `self_consistency_judge`, `pairwise_judge_ab`, `pairwise_judge_ba`

## Phase 2: Metrics + Aggregator — COMPLETE

- [x] 2.1 Реализовать `metrics.py` (pure functions) + 12 unit-тестов
  - Type: tdd
  - Depends On: 1.1 (импорт NasaTLX)
  - Can-Parallel-With: 2.2
  - Files: `src/atm/evaluation/metrics.py` (new, ~120 LoC), `tests/unit/evaluation/test_metrics.py` (new)
  - Acceptance: `uv run pytest tests/unit/evaluation/test_metrics.py -q` — 12 green; mypy strict + ruff clean
  - Notes: `humaneval_pass_at_k` — unbiased estimator Chen et al. 2021 via `math.prod`; тест n=20,c=2,k=10 → 0.6316; `cost_per_quality(quality=0)` использует `eps=1e-6`; все empty-list inputs → 0.0

- [x] 2.2 Реализовать `aggregator.py` + `resolve_spec` + `EvaluationCfg` + unit + integration тесты
  - Type: tdd
  - Depends On: 1.2 (использует score_ground_truth)
  - Can-Parallel-With: 2.1
  - Files: `src/atm/evaluation/aggregator.py` (new, ~120 LoC), `src/atm/tasks/__init__.py` (modify — add resolve_spec), `src/atm/experiment/config.py` (modify — add EvaluationCfg + field), `tests/unit/evaluation/test_aggregator_pure.py` (new, 4 tests), `tests/unit/tasks/test_resolve_spec.py` (new, 3 tests), `tests/unit/experiment/test_config_evaluation_cfg.py` (new, 2 tests), `tests/integration/conftest.py` (new), `tests/integration/evaluation/__init__.py` (new), `tests/integration/evaluation/test_aggregator.py` (new, 4 PG tests)
  - Acceptance: unit green; `ATM_INTEGRATION_PG=1 uv run pytest tests/integration/evaluation/test_aggregator.py -q` — 4 green; `ATM_INTEGRATION_PG=1 uv run pytest tests/integration/storage -q` — по-прежнему green; mypy strict clean
  - Notes: `compute_quality` → `(None, {"error":...})` при сбое (не бросает); `persist_quality` идемпотентен; `resolve_spec` — None если `task.input` set + name не в TASKS; `EvaluationCfg.judge_model = "openai:gpt-4o"`, `judge_self_consistency_n: int = Field(3, ge=1, le=10)`; `tests/integration/storage/conftest.py` → thin re-export

## Phase 3: Runner Wiring — NOT STARTED

- [ ] 3.1 Подключить агрегатор в `runner.py` + удалить `_evaluator.py` + тест wiring + аудит M6 smoke-тестов
  - Type: simple
  - Depends On: 2.2 (aggregator, resolve_spec, EvaluationCfg exist)
  - Can-Parallel-With: none
  - Files: `src/atm/experiment/runner.py` (modify — 3 callsites + imports + judge LLMWrapper + SubprocessSandbox), `src/atm/experiment/_evaluator.py` (DELETE), `tests/unit/experiment/test_runner_evaluation_wiring.py` (new, 1–2 tests), `tests/unit/experiment/test_runner_*.py` (audit — patch "55-substring" assertions)
  - Acceptance: `uv run pytest tests/unit/experiment -q` — all green; `uv run mypy --strict src/atm/experiment/runner.py` — clean; `grep -rn "from atm.experiment._evaluator" src/ tests/` — пусто; `grep -rn "_evaluator" src/` — пусто
  - Notes: inline-prompt path (`spec is None`) → `quality_score=0.0` + debug log; judge wrapper из `cfg.evaluation.judge_model`; один `SubprocessSandbox()` per run (дёшево, no-op constructor); все 3 callsite (success/BudgetExceeded/Exception) обёрнуты в try/except → None на сбой

## Phase 4: Finalisation — NOT STARTED

- [ ] 4.1 Финализировать `__init__.py` + e2e integration test + обновить codebase-map
  - Type: simple
  - Depends On: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1
  - Can-Parallel-With: none
  - Files: `src/atm/evaluation/__init__.py` (rewrite with __all__ ~15 symbols), `tests/integration/evaluation/test_aggregator_e2e.py` (new, 1 e2e test, @pytest.mark.integration), `dev/codebase-map.md` (edit — add M11 section)
  - Acceptance: `uv run pytest tests/unit -q` — ~40 новых тестов green; `ATM_INTEGRATION_PG=1 uv run pytest tests/integration -q` — all green; `uv run mypy --strict src/atm` — clean; `uv run ruff check src/atm tests` — clean
  - Notes: e2e test — FakeLLM scripted MMLU → answer "B" → `runs.quality_score == 1.0`; `evaluation.judge_model = "fake:echo"` в тест-конфиге; `__all__` — алфавитный порядок как в `atm/tasks/__init__.py`; `pyproject.toml` маркеры НЕ изменяем

---

## Stats

- Total: 7 tasks · ~8h
- Done: 3 / 7

## How to Update

После завершения каждой задачи:
1. Отметить `[x]` вместо `[ ]`.
2. Обновить заголовок фазы: все задачи выполнены → `COMPLETE`; часть → `IN PROGRESS`.
3. Обновить секцию `SESSION PROGRESS` в `m11-context.md`.
4. Обновить счётчик `Done: N / 7` в Stats.
