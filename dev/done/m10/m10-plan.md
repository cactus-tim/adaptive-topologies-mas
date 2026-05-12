# M10 Tasks & Datasets — Plan

## Executive Summary

M10 поставляет инфраструктуру задач для экспериментов: registry loader-ов и evaluator-ов, четыре loader-модуля (HumanEval, MMLU, Creative, Analysis), четыре evaluator-а разных типов (sandbox-исполнение, exact-match, LLM-judge, hybrid), parquet-кэш HF датасетов, YAML-промпты для creative/analysis задач, и полный unit-тест покрытие без сетевых вызовов.

## Current State

- `src/atm/tasks/__init__.py` — пустой M0 skeleton.
- `TaskSpec` уже существует в `src/atm/core/types.py:263` (frozen Pydantic, поля: `id`, `type`, `input`, `expected`, `metadata`, `evaluator_key`). Не пересоздаём.
- `LLMWrapper` в `src/atm/llm/wrapper.py`, `FakeLLM` в `src/atm/llm/fake.py`.
- `DockerSandbox` / `SubprocessSandbox` в `src/atm/tools/sandbox/`.
- `datasets` library отсутствует в `pyproject.toml` — нужно добавить.
- `data/` уже глобально gitignored.

## Proposed Approach

5-волновая структура. Wave 4 — четыре параллельных loader/evaluator-модуля, изолированных по файлам. Shared judge-логика вынесена в `_judge.py` (создаётся в Wave 3 вместе с `_cache.py`) чтобы избежать race condition в Wave 4. `LLMLike` Protocol введён в `base.py` для DI judge-LLM без привязки к конкретному классу. Parquet-кэш использует atomic rename + pyarrow bytes-metadata + опциональное поле `dataset_revision`.

## Implementation Phases

### Phase 1: Scaffolding (~0.5h)
**Goal:** Добавить `datasets` dependency и создать структуру каталогов.

- [ ] 1.1 Добавить `"datasets>=2.20,<4"` в `[project.dependencies]` в `pyproject.toml` (алфавитный порядок, после `asyncpg`)
  - File: `pyproject.toml`
  - Acceptance: `uv run python -c "import datasets; print(datasets.__version__)"` успешно
- [ ] 1.2 Создать `tests/unit/tasks/__init__.py` с импортом `from atm import tasks  # noqa: F401` (активирует `@register_*` декораторы при pytest collection)
  - File: `tests/unit/tasks/__init__.py`
  - Acceptance: файл существует, импорт работает без ошибок
- [ ] 1.3 Создать placeholder-файлы для новых директорий
  - File: `tests/fixtures/tasks/.gitkeep`, `conf/tasks/.gitkeep`
  - Acceptance: директории появляются в git
- [ ] 1.4 Запустить `uv lock` для обновления lockfile
  - Acceptance: `uv sync` проходит, `uv run pytest tests/unit -q` зелёный (нет регрессий)

Note: `.gitignore` не редактируется — `data/` уже глобально gitignored.

---

### Phase 2: Foundation (~1h)
**Goal:** Реализовать публичный API модуля задач: Protocols, Registries, EvalResult.

- [ ] 2.1 Написать тесты `tests/unit/tasks/test_base.py` (TDD, перед реализацией)
  - File: `tests/unit/tasks/test_base.py`
  - Acceptance: тесты написаны, падают (red)
- [ ] 2.2 Реализовать `src/atm/tasks/base.py`:
  - `LLMLike` — runtime-checkable Protocol: `async ainvoke(messages, *, agent_id, ...) -> LLMResponse`; используется в сигнатурах evaluator-конструкторов Steps 6/7
  - `Evaluator` — runtime-checkable Protocol: `async evaluate(spec: TaskSpec, answer: str, artifacts: dict[str, Any]) -> EvalResult`
  - `EvalResult` — frozen Pydantic: `score: float = Field(..., ge=0, le=1)`, `passed: bool`, `details: dict[str, Any]`, `evaluator_key: str`
  - `EvaluatorRegistry` — `@register_evaluator(name)`, `get(name) -> Evaluator` (KeyError если нет)
  - `TaskLoader` — Protocol: `def load(cache_dir: Path | None = None) -> list[TaskSpec]`
  - `TaskRegistry` — `@register_task(name)`, `get(name)`, `.sample(n, seed)` детерминировано через `random.Random(seed)` (sort by id before sample); ленивая загрузка + LRU-кэш в `self._cache`
  - Экспортировать инстансы `TASKS = TaskRegistry()`, `EVALUATORS = EvaluatorRegistry()`
  - File: `src/atm/tasks/base.py`
  - Acceptance: 5+ тестов зелёные; mypy --strict чистый
- [ ] 2.3 Обновить `src/atm/tasks/__init__.py` с re-exports
  - File: `src/atm/tasks/__init__.py`
  - Acceptance: `from atm.tasks import TASKS, EVALUATORS, Evaluator, EvalResult, TaskLoader, LLMLike` работает

---

### Phase 3: Cache Layer + Judge Helper (~1h)
**Goal:** Реализовать parquet-кэш и shared judge-хелпер (до Wave 4, чтобы избежать race).

- [ ] 3.1 Написать тесты `tests/unit/tasks/test_cache.py` (TDD)
  - File: `tests/unit/tasks/test_cache.py`
  - Acceptance: тесты написаны, падают (red)
- [ ] 3.2 Реализовать `src/atm/tasks/_cache.py`:
  - `cache_path(name, cache_dir) -> Path`
  - `is_cached(name, cache_dir) -> bool`
  - `write_cache(name, rows, cache_dir, *, dataset_revision: str | None = None) -> Path` — atomic rename (`os.replace`); schema metadata как `bytes` (pyarrow requirement: keys и values должны быть `bytes`, иначе runtime error); metadata содержит `created_at_utc`, `dataset_id`, `row_count`, `dataset_revision`
  - `read_cache(name, cache_dir) -> list[dict]`
  - File: `src/atm/tasks/_cache.py`
  - Acceptance: 4 теста зелёные (write→read round-trip, is_cached False/True, atomic rename не оставляет .tmp, read отсутствующего raises FileNotFoundError)
- [ ] 3.3 Реализовать `src/atm/tasks/_judge.py` с хелпером:
  - `_invoke_judge(llm_like: LLMLike, *, prompt: str, agent_id: str) -> tuple[float, str, str | None]` — парсит JSON из LLM ответа; regex fallback `re.search(r'\{.*\}', text, re.DOTALL)` при json.loads failure; возвращает `(score_normalized, reasoning, error_or_None)`
  - File: `src/atm/tasks/_judge.py`
  - Acceptance: Steps 6 и 7 импортируют этот хелпер вместо локальной реализации; mypy strict чистый

---

### Phase 4: Loaders & Evaluators (параллельно, ~2h)
**Goal:** Реализовать все четыре loader+evaluator пары.

#### Step 4 — HumanEval
- [ ] 4.1 Создать `tests/fixtures/tasks/humaneval_sample.json` (2 строки HF dataset)
  - File: `tests/fixtures/tasks/humaneval_sample.json`
  - Acceptance: валидный JSON с полями `task_id, prompt, canonical_solution, test, entry_point`
- [ ] 4.2 Написать тесты `tests/unit/tasks/test_humaneval.py` (TDD)
  - File: `tests/unit/tasks/test_humaneval.py`
  - Acceptance: тесты написаны, падают (red)
- [ ] 4.3 Реализовать `src/atm/tasks/humaneval.py`:
  - `_strip_code_fences(text: str) -> str` — убирает markdown ` ```python ... ``` ` обёртки из LLM-ответов
  - `HumanEvalLoader` зарегистрированный как `"humaneval"`: load → cache check → HF load → `TaskSpec(id=f"humaneval/{task_id}", type="programming", ...)`
  - `HumanEvalEvaluator` зарегистрированный как `"humaneval_pytest"`: принимает `sandbox` через DI (default `SubprocessSandbox`); payload = `f"{answer}\n\n{spec.metadata['test']}\n\ncheck({spec.metadata['entry_point']})"` → `await sandbox.execute("python", payload, timeout=10)`
  - File: `src/atm/tasks/humaneval.py`
  - Acceptance: 6 тестов зелёные: loader с mock, кэш-хит (counter), evaluator canonical→passed=True, evaluator пустая строка→False, evaluator SyntaxError→False, `_strip_code_fences` тест

#### Step 5 — MMLU
- [ ] 5.1 Создать `tests/fixtures/tasks/mmlu_sample.json` (3 строки HF dataset)
  - File: `tests/fixtures/tasks/mmlu_sample.json`
  - Acceptance: валидный JSON с полями `question_id, question, options, answer, answer_index, category`
- [ ] 5.2 Написать тесты `tests/unit/tasks/test_mmlu.py` (TDD)
  - File: `tests/unit/tasks/test_mmlu.py`
  - Acceptance: тесты написаны, падают (red)
- [ ] 5.3 Реализовать `src/atm/tasks/mmlu.py`:
  - `MMLULoader` зарегистрированный как `"mmlu"`: `load(cache_dir, limit=500)`; формат options `"A. opt0\nB. opt1\n..."`; `TaskSpec(id=f"mmlu/{question_id}", type="qa", expected=answer, evaluator_key="mmlu_exact_match")`
  - `MMLUEvaluator` зарегистрированный как `"mmlu_exact_match"`: нормализованный exact match через `re.match(r"\s*(?:answer:?\s*)?\(?([A-Ja-j])\)?", answer.strip())`; покрывает A–J (10-way MMLU-Pro)
  - File: `src/atm/tasks/mmlu.py`
  - Acceptance: 5 тестов зелёные: loader, кэш-хит, exact "A"→match, "The answer is B."→match, "C" vs "A"→no match

#### Step 6 — Creative
- [ ] 6.1 Создать `conf/tasks/creative_prompts.yaml` (8 промптов, минимум 5 требование §M10)
  - File: `conf/tasks/creative_prompts.yaml`
  - Acceptance: 8 записей с полями `id, prompt, rubric`; валидируется при загрузке
- [ ] 6.2 Создать `tests/fixtures/llm/m10_judge_creative.yaml` (scripted FakeLLM response)
  - File: `tests/fixtures/llm/m10_judge_creative.yaml`
  - Acceptance: содержит entry для `agent_id="creative_judge"`, content=`'{"score": 8, "reasoning": "..."}'`
- [ ] 6.3 Написать тесты `tests/unit/tasks/test_creative.py` (TDD)
  - File: `tests/unit/tasks/test_creative.py`
  - Acceptance: тесты написаны, падают (red)
- [ ] 6.4 Реализовать `src/atm/tasks/creative.py`:
  - `CreativeLoader` зарегистрированный как `"creative"`: YAML → `TaskSpec(id=f"creative/{id}", type="creative", expected=None, evaluator_key="creative_judge")`; кэш не используется (источник локальный)
  - `CreativeJudgeEvaluator` зарегистрированный как `"creative_judge"`: принимает `judge_llm: LLMLike`; использует `_invoke_judge` из `tasks._judge`; `passed = score >= 0.6`
  - File: `src/atm/tasks/creative.py`
  - Acceptance: 4 теста зелёные: loader 8 TaskSpec, judge FakeLLM→score=0.8/passed=True, malformed JSON→score=0.0/error в details, score=4→0.4/passed=False

#### Step 7 — Analysis
- [ ] 7.1 Создать `conf/tasks/analysis_prompts.yaml` (6 задач с inline data_csv и structural_checks)
  - File: `conf/tasks/analysis_prompts.yaml`
  - Acceptance: 6 записей с полями `id, prompt, data_csv, expected_pattern, rubric, structural_checks`
- [ ] 7.2 Создать `tests/fixtures/llm/m10_judge_analysis.yaml` (scripted FakeLLM response)
  - File: `tests/fixtures/llm/m10_judge_analysis.yaml`
  - Acceptance: содержит entry для analysis judge
- [ ] 7.3 Написать тесты `tests/unit/tasks/test_analysis.py` (TDD)
  - File: `tests/unit/tasks/test_analysis.py`
  - Acceptance: тесты написаны, падают (red)
- [ ] 7.4 Реализовать `src/atm/tasks/analysis.py`:
  - `AnalysisLoader` зарегистрированный как `"analysis"`: YAML → `TaskSpec(id=f"analysis/{id}", type="analysis", ...)`
  - `AnalysisHybridEvaluator` зарегистрированный как `"analysis_hybrid"`: structural_check (substring в answer.lower()) → score_struct; LLM-judge через `_invoke_judge` из `tasks._judge` → score_judge; `score = 0.5 * score_struct + 0.5 * score_judge`; принимает `judge_llm: LLMLike`
  - File: `src/atm/tasks/analysis.py`
  - Acceptance: 4 теста зелёные: loader 6 TaskSpec, правильный паттерн+good judge→1.0, struct FAIL+good judge→0.5, both FAIL→0.0

---

### Phase 5: Finalization (~0.5h)
**Goal:** Финализировать exports, smoke-тест registry, обновить codebase-map.

- [ ] 8.1 Финализировать `src/atm/tasks/__init__.py` с guarded imports всех 4 loader-модулей (паттерн из `topology/__init__.py`); `__all__` перечисляет все публичные символы
  - File: `src/atm/tasks/__init__.py`
  - Acceptance: `from atm.tasks import TASKS, EVALUATORS` работает и все loader-ы зарегистрированы
- [ ] 8.2 Добавить re-exports в `src/atm/__init__.py`: `TaskRegistry`, `EvaluatorRegistry`, `TASKS`, `EVALUATORS`
  - File: `src/atm/__init__.py`
  - Acceptance: `from atm import TASKS, EVALUATORS` работает
- [ ] 8.3 Создать `tests/unit/tasks/test_registry_smoke.py` — автоматический smoke-тест с pre-seeded fixture-кэшем (не сетевой):
  - Fixture кэш: `tmp_path` + копия `humaneval_sample.json` записанная через `write_cache`
  - Тест: `TASKS.get("humaneval").sample(10, seed=42)` возвращает ≤2 TaskSpec (по fixture), детерминированно
  - File: `tests/unit/tasks/test_registry_smoke.py`
  - Acceptance: тест зелёный, без сетевых вызовов
- [ ] 8.4 Обновить `dev/codebase-map.md` секцию Tasks & Evaluation
  - File: `dev/codebase-map.md`
  - Acceptance: секция содержит 4 loader-имени, evaluator-ключи, путь к кэшу, dependency `datasets`
- [ ] 8.5 Запустить финальный test suite: `uv run pytest tests/unit -q`, `uv run mypy src/atm --strict`, `uv run ruff check src/atm tests`
  - Acceptance: 0 failures, mypy clean, ruff clean

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `pyproject.toml` | `+datasets>=2.20,<4` | HF dataset loader |
| `src/atm/tasks/__init__.py` | rewrite skeleton → full exports + guarded imports | публичный API модуля |
| `src/atm/tasks/base.py` | new | `LLMLike`, `Evaluator`, `EvalResult`, `EvaluatorRegistry`, `TaskLoader`, `TaskRegistry`, `TASKS`, `EVALUATORS` |
| `src/atm/tasks/_cache.py` | new | parquet-кэш с atomic rename |
| `src/atm/tasks/_judge.py` | new | shared `_invoke_judge` хелпер |
| `src/atm/tasks/humaneval.py` | new | loader + sandbox evaluator + `_strip_code_fences` |
| `src/atm/tasks/mmlu.py` | new | loader + exact-match evaluator |
| `src/atm/tasks/creative.py` | new | YAML loader + LLM-judge evaluator |
| `src/atm/tasks/analysis.py` | new | YAML loader + hybrid evaluator |
| `src/atm/__init__.py` | +re-exports | `TASKS`, `EVALUATORS` доступны с уровня пакета |
| `conf/tasks/creative_prompts.yaml` | new | 8 creative промптов с rubric |
| `conf/tasks/analysis_prompts.yaml` | new | 6 analysis задач с structural_checks |
| `tests/unit/tasks/__init__.py` | new | `from atm import tasks` для активации декораторов |
| `tests/unit/tasks/test_base.py` | new | unit-тесты registry/protocol |
| `tests/unit/tasks/test_cache.py` | new | unit-тесты parquet-кэша |
| `tests/unit/tasks/test_humaneval.py` | new | unit-тесты HumanEval (6 тестов incl. `_strip_code_fences`) |
| `tests/unit/tasks/test_mmlu.py` | new | unit-тесты MMLU |
| `tests/unit/tasks/test_creative.py` | new | unit-тесты creative judge |
| `tests/unit/tasks/test_analysis.py` | new | unit-тесты analysis hybrid |
| `tests/unit/tasks/test_registry_smoke.py` | new | smoke-тест registry с fixture-кэшем |
| `tests/fixtures/tasks/humaneval_sample.json` | new | 2 строки HF dataset для mock |
| `tests/fixtures/tasks/mmlu_sample.json` | new | 3 строки HF dataset для mock |
| `tests/fixtures/llm/m10_judge_creative.yaml` | new | FakeLLM fixture для creative judge |
| `tests/fixtures/llm/m10_judge_analysis.yaml` | new | FakeLLM fixture для analysis judge |
| `dev/codebase-map.md` | update Tasks & Evaluation section | статус M10 |

## Dependencies & Order Constraints

- Phase 1 (scaffolding) должна быть завершена до Phase 2 (нужны `tests/unit/tasks/` и `datasets` dep).
- Phase 2 (`base.py`) должна быть завершена до Phase 3 (`_cache.py` и `_judge.py` импортируют из `base.py`).
- Phase 3 должна быть завершена до Phase 4 (все loader-ы зависят от `_cache.py`; Steps 6 и 7 зависят от `_judge.py`).
- Phase 4 (все 4 шага) должна быть завершена до Phase 5 (финальные exports и smoke-тест).
- Внутри Phase 4: Steps 4, 5, 6, 7 независимы, трогают непересекающиеся файлы.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `datasets` тянет тяжёлые transitive deps (pandas, fsspec) — медленный CI install | Medium | Low | Pinned версия `>=2.20,<4`, no extras |
| pyarrow schema metadata keys/values должны быть `bytes` — footgun | High | Medium | Явно документировать в `_cache.py`; тест на round-trip поймает регрессию |
| SubprocessSandbox запускает untrusted код на host | Low (тесты используют trusted fixture) | High | Только DockerSandbox в prod; SubprocessSandbox только с trusted fixture кодом |
| `filterwarnings = "error"` в pyproject.toml может сломать тесты из-за DeprecationWarning из `datasets` | Medium | Medium | Добавить `ignore::DeprecationWarning:datasets` в filterwarnings если нужно |
| Race condition в Wave 4 (Steps 6+7 оба хотят `_invoke_judge`) | Было Medium | Resolved | `_judge.py` создаётся в Phase 3 до Wave 4 |
| Cyclic import между tasks модулями | Low | High | Guarded imports в `__init__.py`; evaluator instances не создаются при импорте |

## Out of Scope

- Реальный Docker sandbox для HumanEval — только SubprocessSandbox в тестах; DockerSandbox для prod-runner в M11.
- Acceptance integration тесты (Needs Integration Tests = no).
- Оптимизация hybrid evaluator weights (0.5/0.5 — ресёрч choice, настраивается в M11).
- Авторизация HF Hub (оба датасета публичные, HF_TOKEN опционален).
- Рефакторинг judge-логики в `evaluation/judges.py` — запланирован на M11.

## Timeline

- Total: ~5h
- Wave 1: ~0.5h
- Wave 2: ~1h
- Wave 3: ~1h
- Wave 4: ~2h (параллельно)
- Wave 5: ~0.5h
- Created: 2026-05-12
