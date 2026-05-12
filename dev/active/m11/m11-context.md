# M11 — Evaluation Framework — Context

## SESSION PROGRESS (2026-05-12)

### COMPLETED
- Step 1.1: scaffold `src/atm/evaluation/__init__.py` + implement `tlx.py` (NasaTLX Pydantic model + aggregate_tlx) + create `tests/unit/evaluation/__init__.py` + `tests/unit/evaluation/test_tlx.py` (8 tests green, mypy strict clean, ruff clean)
- Step 1.2: implement `src/atm/evaluation/ground_truth.py` (flat if/elif dispatch over M10 evaluators) + `tests/unit/evaluation/test_ground_truth.py` (5 tests green); merged feat/m11 into worktree to bring in M10 task modules

- Step 1.3: judges.py — RubricJudge, PairwiseJudge, SelfConsistentJudge + PairwiseResult;
  _to_answer_relative helper; 3 FakeLLM YAML fixtures; 10 unit tests green;
  mypy --strict clean; ruff clean. Worktree branch: worktree-agent-a21c790c7fdcd2ade.

- Step 2.1: metrics.py (pure functions: quality / efficiency / time / human / RQ2) + 12 unit tests — DONE (merged from feat/m11)
- Step 2.2: aggregator.py + resolve_spec + EvaluationCfg + unit + integration tests — DONE (merged from feat/m11)
- Step 3.1: Wire aggregator into runner.py + delete _evaluator.py + test wiring + M6 smoke audit
  - Replaced `from atm.experiment._evaluator import evaluate` with `compute_quality` + `resolve_spec` + `SubprocessSandbox`
  - Added judge LLMWrapper built from `cfg.evaluation.judge_model` (fake:echo in tests)
  - Three callsites updated (success / BudgetExceeded / Exception) with try/except + inline-prompt short-circuit
  - _update_run_success / _update_run_failed signatures updated to `quality_score: float | None`
  - Deleted `src/atm/experiment/_evaluator.py`
  - Removed 4 now-stale evaluate() tests from test_runner.py; updated quality_score == 1.0 → 0.0
  - Created `tests/unit/experiment/test_runner_evaluation_wiring.py` (2 tests)
  - All 1065 unit tests green; mypy --strict clean; ruff clean

### IN PROGRESS
- Step 4.1 (finalise __init__.py + e2e test + codebase-map) — next

### BLOCKERS
- Нет

---

## Quick Resume

1. Прочитать этот файл.
2. Открыть `m11-tasks.md` — найти первую незавершённую задачу.
3. Прочитать `m11-plan.md` Phase 1 за стратегией.
4. Начать с: создания `src/atm/evaluation/__init__.py` (пустой заглушка) + `src/atm/evaluation/tlx.py` + `tests/unit/evaluation/__init__.py` + `tests/unit/evaluation/test_tlx.py`.

---

## Key Files

Каждый файл, которого касается M11 (создать / изменить / удалить):

**`src/atm/evaluation/__init__.py`**
- Роль: точка входа пакета evaluation
- Запланированное изменение: создать пустую заглушку (Step 1), переписать с `__all__` в Step 7
- Статус: DONE (placeholder `# M11 evaluation`)

**`src/atm/evaluation/tlx.py`**
- Роль: Pydantic-модель `NasaTLX` (6 шкал 0–100) + хелпер `aggregate_tlx`
- Запланированное изменение: создать
- Статус: DONE

**`src/atm/evaluation/ground_truth.py`**
- Роль: thin facade над M10 EVALUATORS; `score_ground_truth(spec, answer, ...)` — flat dispatch по `evaluator_key`
- Запланированное изменение: создать
- Статус: DONE

**`src/atm/evaluation/judges.py`**
- Роль: post-hoc LLM-as-judge — `RubricJudge`, `SelfConsistentJudge`, `PairwiseJudge` + `PairwiseResult`
- Запланированное изменение: создать (~150 LoC)
- Статус: NOT STARTED

**`src/atm/evaluation/metrics.py`**
- Роль: pure functions для RQ1–RQ4 (quality / efficiency / time / human / RQ2)
- Запланированное изменение: создать (~120 LoC)
- Статус: NOT STARTED

**`src/atm/evaluation/aggregator.py`**
- Роль: async `compute_quality` + `persist_quality` + `aggregate_run`; никогда не бросает исключений
- Запланированное изменение: создать (~120 LoC)
- Статус: NOT STARTED

**`src/atm/tasks/__init__.py`**
- Роль: реестр задач; экспортирует `TASKS`, `EVALUATORS`
- Запланированное изменение: добавить `resolve_spec(task_cfg) -> TaskSpec | None`
- Статус: NOT STARTED

**`src/atm/experiment/config.py`**
- Роль: Pydantic-конфиги эксперимента
- Запланированное изменение: добавить `EvaluationCfg(judge_model, judge_self_consistency_n)` + поле `evaluation: EvaluationCfg` на `ExperimentConfig`
- Статус: NOT STARTED

**`src/atm/experiment/runner.py`**
- Роль: главный раннер; `run_one(cfg) -> RunResult`
- Запланированное изменение: 3 callsite-замены (`evaluate(...)` → `compute_quality(...)`); добавить judge LLMWrapper + SubprocessSandbox; убрать import `_evaluator`
- Статус: DONE

**`src/atm/experiment/_evaluator.py`**
- Роль: заглушка M6 (`evaluate(...) → 1.0 if "55" in answer else 0.0`)
- Запланированное изменение: УДАЛИТЬ в Step 6
- Статус: DONE (DELETED)

**`tests/unit/evaluation/__init__.py`**
- Роль: пакет unit-тестов evaluation; содержит `import atm.tasks  # noqa: F401`
- Запланированное изменение: создать
- Статус: DONE

**`tests/unit/evaluation/test_tlx.py`**
- Роль: 8 unit-тестов NasaTLX + aggregate_tlx (6+ required)
- Запланированное изменение: создать
- Статус: DONE

**`tests/unit/evaluation/test_ground_truth.py`**
- Роль: 5 unit-тестов score_ground_truth (dispatch + missing deps)
- Запланированное изменение: создать
- Статус: DONE

**`tests/unit/evaluation/test_judges.py`**
- Роль: 10 unit-тестов rubric / self-consistency / pairwise / budget
- Запланированное изменение: создать
- Статус: NOT STARTED

**`tests/unit/evaluation/test_metrics.py`**
- Роль: 12 unit-тестов pure functions (pass@k, division-by-zero, RQ2 helpers)
- Запланированное изменение: создать
- Статус: NOT STARTED

**`tests/unit/evaluation/test_aggregator_pure.py`**
- Роль: 4 in-memory теста aggregator (failure modes, no PG)
- Запланированное изменение: создать
- Статус: NOT STARTED

**`tests/unit/tasks/test_resolve_spec.py`**
- Роль: 3 теста resolve_spec (registered / unregistered / seed determinism)
- Запланированное изменение: создать
- Статус: NOT STARTED

**`tests/unit/experiment/test_config_evaluation_cfg.py`**
- Роль: 2 теста EvaluationCfg (defaults + override)
- Запланированное изменение: создать
- Статус: NOT STARTED

**`tests/unit/experiment/test_runner_evaluation_wiring.py`**
- Роль: 1–2 теста wiring (monkeypatched compute_quality вызывается run_one)
- Запланированное изменение: создать
- Статус: DONE (2 tests)

**`tests/unit/experiment/test_runner_*.py`**
- Роль: существующие smoke-тесты раннера
- Запланированное изменение: аудит; замена `quality_score == 1.0` на 0.0 для inline-prompt fixtures; remove 4 now-stale evaluate() tests
- Статус: DONE

**`tests/integration/conftest.py`**
- Роль: общие PG-фикстуры для всего integration-поддерева
- Запланированное изменение: СОЗДАТЬ; поднять `pg_dsn`, `pg_engine_fast`, `pg_engine_alembic`, `session_factory_fast` из storage/conftest.py
- Статус: NOT STARTED

**`tests/integration/storage/conftest.py`**
- Роль: фикстуры для storage integration-тестов
- Запланированное изменение: thin re-export из `tests/integration/conftest.py`
- Статус: NOT STARTED

**`tests/integration/evaluation/__init__.py`**
- Роль: пакет integration-тестов evaluation
- Запланированное изменение: создать (пустой)
- Статус: NOT STARTED

**`tests/integration/evaluation/test_aggregator.py`**
- Роль: 4 PG-теста aggregator (MMLU correct/wrong, HumanEval canonical, idempotency)
- Запланированное изменение: создать; маркер `@pytest.mark.integration`
- Статус: NOT STARTED

**`tests/integration/evaluation/test_aggregator_e2e.py`**
- Роль: 1 e2e контрактный тест (`run_one → runs.quality_score == 1.0` для scripted-correct MMLU)
- Запланированное изменение: создать; маркер `@pytest.mark.integration`
- Статус: NOT STARTED

**`tests/fixtures/llm/m11_judge_pairwise_ab.yaml`**
- Роль: FakeLLM-скрипт для pairwise AB + BA orderings
- Запланированное изменение: создать
- Статус: NOT STARTED

**`tests/fixtures/llm/m11_judge_pairwise_swap_disagree.yaml`**
- Роль: FakeLLM-скрипт для swap-disagree сценария
- Запланированное изменение: создать
- Статус: NOT STARTED

**`tests/fixtures/llm/m11_judge_self_consistency.yaml`**
- Роль: 3 скриптовых ответа для self-consistency judge
- Запланированное изменение: создать
- Статус: NOT STARTED

**`dev/codebase-map.md`**
- Роль: карта кодовой базы
- Запланированное изменение: добавить раздел "Evaluation Framework (M11)"
- Статус: NOT STARTED

---

## Decisions

**Архитектурные решения:**

- Decision: Compose, not duplicate — `ground_truth.py` как thin facade над M10 EVALUATORS
  - Rationale: M10 уже содержит 4 эвалюатора с единым Protocol. Дублирование логики создало бы расхождение. Фасад изолирует импорт M10 за одним интерфейсом.

- Decision: Flat `if/elif` dispatch в `ground_truth.py` (рек. #5)
  - Rationale: `EVALUATORS.get(...)` lookup — не type-checkable; flat chain полностью проверяется mypy и проще отлаживается.

- Decision: Pairwise swap test с явным `_to_answer_relative(slot_winner, swapped)` (рек. #4)
  - Rationale: Позиционное смещение LLM-судей задокументировано (arXiv 2406.07791); swap test — рекомендованная митигация. Явный хелпер исключает silent bugs при трансляции slot-relative → answer-relative.

- Decision: `compute_quality` возвращает `(None, {"error": ...})`, никогда не бросает (рек. fix #10)
  - Rationale: сбой оценки качества не должен ломать finalisation запуска; `runs.quality_score = NULL` — допустимый результат.

- Decision: Judge LLM использует тот же `BudgetTracker` что и in-loop агенты (рек. fix #7)
  - Rationale: `BudgetTracker` живёт до возврата `run_one`; judge-вызовы происходят до этого момента → корректная атрибуция к RUN-бюджету. Отдельный evaluation-бюджет не нужен.

- Decision: Score scale 0..10 (M10 `_invoke_judge` convention), не 0..5 как в arch.md §13.2
  - Rationale: консистентность с M10 означает, что смена scale в будущем затронет один файл (`_judge.py`), а не два. Отклонение задокументировано в `judges.py` docstring.

- Decision: `resolve_spec` возвращает `None` для M6 inline-prompt задач (rек. #5)
  - Rationale: позволяет runner-у short-circuit к `quality_score=0.0` без регистрации noop-эвалюатора — EVALUATORS registry остаётся чистым.

- Decision: Продвижение PG-фикстур в `tests/integration/conftest.py` (рек. #1)
  - Rationale: evaluation integration-тесты нужны те же фикстуры что и storage; дублирование в conftest хуже чем единый родительский conftest.

## Constraints

- `pyproject.toml` имеет `filterwarnings = "error"` — никаких stray DeprecationWarning.
- `mypy --strict` и `ruff` обязательны для всех новых/изменённых файлов.
- Новый pytest-маркер `requires_pg` НЕ добавляем — используем `integration`.
- `pytest-asyncio` `asyncio_mode="auto"` уже настроен.
- Никаких новых third-party зависимостей — Pydantic v2 + stdlib + numpy (если нужен).
- M9 владеет записью `human_interactions.tlx_scores`; M11 только предоставляет модель.
- RQ2 PG-запросы к `topology_transitions` — в M13, не в M11.

## Open Questions (из плана)

1. **TLX backfill ownership**: M9 пишет `raw_tlx_score` inline; backfill-хелпер не нужен.
2. **Pairwise tie в Bradley-Terry (M13)**: ties дают 0.5 или исключаются — решение на этапе M13.
3. **`judge_self_consistency_n` wiring**: конфиг добавлен, но runner не вызывает `SelfConsistentJudge` напрямую в M11 — deferred follow-up.
