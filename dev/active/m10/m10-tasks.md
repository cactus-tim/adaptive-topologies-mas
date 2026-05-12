# M10 Tasks & Datasets — Tasks

## Phase 1: Scaffolding (datasets dep + dirs) COMPLETE

- [x] 1.1 Добавить `"datasets>=2.20,<4"` в `[project.dependencies]` в `pyproject.toml` — `pyproject.toml`
  - Acceptance: `uv run python -c "import datasets; print(datasets.__version__)"` успешно
- [x] 1.2 Создать `tests/unit/tasks/__init__.py` с `from atm import tasks  # noqa: F401` — `tests/unit/tasks/__init__.py`
  - Acceptance: файл существует; `@register_*` декораторы активируются при pytest collection
- [x] 1.3 Создать placeholder-файлы `tests/fixtures/tasks/.gitkeep` и `conf/tasks/.gitkeep` — новые файлы
  - Acceptance: директории появляются в git
- [x] 1.4 Запустить `uv lock` для обновления lockfile
  - Acceptance: `uv sync` проходит; `uv run pytest tests/unit -q` зелёный (нет регрессий)

## Phase 2: Foundation (base.py) COMPLETE

- [x] 2.1 Написать тесты `tests/unit/tasks/test_base.py` (TDD, перед реализацией) — `tests/unit/tasks/test_base.py`
  - Acceptance: 5+ тестов написаны и падают (red): register+get, get неизвестного KeyError, sample детерминированность, sample overflow, EvalResult score validation
- [x] 2.2 Реализовать `src/atm/tasks/base.py` — `src/atm/tasks/base.py`
  - Acceptance: `LLMLike` Protocol, `Evaluator` Protocol, `EvalResult` (frozen Pydantic, `score: float = Field(..., ge=0, le=1)`), `EvaluatorRegistry`, `TaskLoader` Protocol, `TaskRegistry` (sort by id, `random.Random(seed)`), `TASKS`, `EVALUATORS`; 5+ тестов зелёные; mypy --strict чистый
- [x] 2.3 Обновить `src/atm/tasks/__init__.py` с базовыми re-exports — `src/atm/tasks/__init__.py`
  - Acceptance: `from atm.tasks import TASKS, EVALUATORS, Evaluator, EvalResult, TaskLoader, LLMLike` работает

## Phase 3: Cache Layer + Judge Helper COMPLETE

- [x] 3.1 Написать тесты `tests/unit/tasks/test_cache.py` (TDD) — `tests/unit/tasks/test_cache.py`
  - Acceptance: 7 тестов написаны и падают (red); покрывают все 4 требуемых сценария + 3 дополнительных
- [x] 3.2 Реализовать `src/atm/tasks/_cache.py` — `src/atm/tasks/_cache.py`
  - Acceptance: `write_cache` с atomic rename (`os.replace`), bytes-metadata (footgun documented), `dataset_revision: str | None`; 7 тестов зелёные: write→read round-trip, is_cached False→True, no .tmp after success, FileNotFoundError on missing, returns Path, revision OK, cache_dir() type
- [x] 3.3 Реализовать `src/atm/tasks/_judge.py` с `_invoke_judge` хелпером — `src/atm/tasks/_judge.py`
  - Acceptance: `_invoke_judge(llm_like: LLMLike, *, prompt: str, agent_id: str) -> tuple[float, str, str | None]`; JSON parse + regex fallback; Steps 6 и 7 импортируют отсюда; mypy strict чистый

## Phase 4: Loaders & Evaluators (параллельно) COMPLETE

### 4a — HumanEval
- [x] 4.1 Создать `tests/fixtures/tasks/humaneval_sample.json` (2 строки) — `tests/fixtures/tasks/humaneval_sample.json`
  - Acceptance: валидный JSON, поля `task_id, prompt, canonical_solution, test, entry_point`
- [x] 4.2 Написать тесты `tests/unit/tasks/test_humaneval.py` (TDD) — `tests/unit/tasks/test_humaneval.py`
  - Acceptance: 6 тестов написаны и падают (red)
- [x] 4.3 Реализовать `src/atm/tasks/humaneval.py` — `src/atm/tasks/humaneval.py`
  - Acceptance: `_strip_code_fences` хелпер + тест на него; `HumanEvalLoader` (`"humaneval"`); `HumanEvalEvaluator` (`"humaneval_pytest"`, DI sandbox); 6 тестов зелёные: loader mock, кэш-хит counter, canonical→passed=True, empty→False, SyntaxError→False, `_strip_code_fences` тест

### 4b — MMLU
- [x] 5.1 Создать `tests/fixtures/tasks/mmlu_sample.json` (3 строки) — `tests/fixtures/tasks/mmlu_sample.json`
  - Acceptance: валидный JSON, поля `question_id, question, options, answer, answer_index, category`
- [x] 5.2 Написать тесты `tests/unit/tasks/test_mmlu.py` (TDD) — `tests/unit/tasks/test_mmlu.py`
  - Acceptance: 5 тестов написаны и падают (red)
- [x] 5.3 Реализовать `src/atm/tasks/mmlu.py` — `src/atm/tasks/mmlu.py`
  - Acceptance: `MMLULoader` (`"mmlu"`, `limit=500`); `MMLUEvaluator` (`"mmlu_exact_match"`, regex A-J); 5 тестов зелёные: loader, кэш-хит, exact "A"→match, "The answer is B."→match, "C" vs "A"→no match

### 4c — Creative
- [x] 6.1 Создать `conf/tasks/creative_prompts.yaml` (8 промптов) — `conf/tasks/creative_prompts.yaml`
  - Acceptance: 8 записей с `id, prompt, rubric`; валидируется при загрузке (минимум 5)
- [x] 6.2 Создать `tests/fixtures/llm/m10_judge_creative.yaml` — `tests/fixtures/llm/m10_judge_creative.yaml`
  - Acceptance: entry для `agent_id="creative_judge"`, content=`'{"score": 8, "reasoning": "..."}'`
- [x] 6.3 Написать тесты `tests/unit/tasks/test_creative.py` (TDD) — `tests/unit/tasks/test_creative.py`
  - Acceptance: 4 теста написаны и падают (red)
- [x] 6.4 Реализовать `src/atm/tasks/creative.py` — `src/atm/tasks/creative.py`
  - Acceptance: `CreativeLoader` (`"creative"`); `CreativeJudgeEvaluator` (`"creative_judge"`, `judge_llm: LLMLike`, использует `_judge._invoke_judge`); 4 теста зелёные

### 4d — Analysis
- [x] 7.1 Создать `conf/tasks/analysis_prompts.yaml` (6 задач) — `conf/tasks/analysis_prompts.yaml`
  - Acceptance: 6 записей с `id, prompt, data_csv, expected_pattern, rubric, structural_checks`
- [x] 7.2 Создать `tests/fixtures/llm/m10_judge_analysis.yaml` — `tests/fixtures/llm/m10_judge_analysis.yaml`
  - Acceptance: entry для analysis judge
- [x] 7.3 Написать тесты `tests/unit/tasks/test_analysis.py` (TDD) — `tests/unit/tasks/test_analysis.py`
  - Acceptance: 4 теста написаны и падают (red)
- [x] 7.4 Реализовать `src/atm/tasks/analysis.py` — `src/atm/tasks/analysis.py`
  - Acceptance: `AnalysisLoader` (`"analysis"`); `AnalysisHybridEvaluator` (`"analysis_hybrid"`, `judge_llm: LLMLike`, использует `_judge._invoke_judge`, score = 0.5*struct + 0.5*judge); 4 теста зелёные

## Phase 5: Finalization COMPLETE

- [x] 8.1 Финализировать `src/atm/tasks/__init__.py` с guarded imports всех 4 loader-модулей — `src/atm/tasks/__init__.py`
  - Acceptance: все 4 loader-а зарегистрированы в `TASKS`; `__all__` перечисляет публичные символы
- [x] 8.2 Добавить re-exports `TaskRegistry`, `EvaluatorRegistry`, `TASKS`, `EVALUATORS` в `src/atm/__init__.py` — `src/atm/__init__.py`
  - Acceptance: `from atm import TASKS, EVALUATORS` работает
- [x] 8.3 Создать `tests/unit/tasks/test_registry_smoke.py` — автотест с pre-seeded fixture-кэшем — `tests/unit/tasks/test_registry_smoke.py`
  - Acceptance: тест `TASKS.get("humaneval").sample(10, seed=42)` зелёный без сетевых вызовов; детерминированный результат
- [x] 8.4 Обновить `dev/codebase-map.md` секцию Tasks & Evaluation — `dev/codebase-map.md`
  - Acceptance: секция содержит 4 loader-имени, evaluator-ключи, `data/cache/tasks/`, dependency `datasets`
- [x] 8.5 Запустить финальный test suite + mypy + ruff
  - Acceptance: `uv run pytest tests/unit -q` — 0 failures; `uv run mypy src/atm --strict` — clean; `uv run ruff check src/atm tests` — clean

---

## Stats

- Total: 28 tasks · ~5h
- Done: 28 / 28

## How to Update

Check off tasks with `[x]` and update `m10-context.md` SESSION PROGRESS after each milestone.

Phase completion markers:
- All tasks in phase done → change `NOT STARTED` to `COMPLETE`
- Some tasks done → change `NOT STARTED` to `IN PROGRESS`
