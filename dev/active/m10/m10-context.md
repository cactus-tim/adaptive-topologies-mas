# M10 Tasks & Datasets — Context

## SESSION PROGRESS (2026-05-12)

### COMPLETED
- Phase 1 (Scaffolding): добавлен `datasets>=2.20,<4` в pyproject.toml, создан `tests/unit/tasks/__init__.py`, созданы `.gitkeep` для `tests/fixtures/tasks/` и `conf/tasks/`, обновлён uv lockfile (datasets 3.6.0 установлен), 866 unit-тестов зелёных.
- Phase 2 (Foundation): реализован `src/atm/tasks/base.py` — `LLMLike`, `Evaluator`, `EvalResult`, `TaskLoader`, `EvaluatorRegistry`, `TaskRegistry`, `TASKS`, `EVALUATORS`; написаны 13 unit-тестов в `tests/unit/tasks/test_base.py`; обновлён `src/atm/tasks/__init__.py` с re-exports; mypy --strict + ruff clean; 879 unit-тестов зелёных.
- Phase 3 (Cache + Judge): реализован `src/atm/tasks/_cache.py` — atomic write/read via os.replace, pyarrow bytes-metadata, `dataset_revision`; написан `src/atm/tasks/_judge.py` — `_invoke_judge` с JSON parse + regex fallback; 7 тестов в `tests/unit/tasks/test_cache.py`; mypy --strict + ruff clean; 983 unit-тестов зелёных.
- Phase 4 (Loaders): реализованы все 4 loader/evaluator пары — HumanEval, MMLU, Creative, Analysis; все тесты зелёные.
- Phase 5 (Finalization): финализирован `src/atm/tasks/__init__.py` с guarded imports всех 4 loader-модулей; добавлен re-export `TASKS, EVALUATORS` в `src/atm/__init__.py`; создан `tests/unit/tasks/test_registry_smoke.py` (4 теста: loader registration, no-network sample, determinism, different seeds); обновлён `dev/codebase-map.md` секция Tasks & Evaluation; 1013 unit-тестов зелёных; mypy --strict clean; ruff clean.

### IN PROGRESS
- Все 5 фаз complete; готово к verification и PR

### BLOCKERS
- None

## Quick Resume

1. Прочитай этот файл
2. Проверь `m10-tasks.md` — что делать следующим
3. Прочитай `m10-plan.md` Phase 1 для стратегии
4. Начни с: `pyproject.toml` — добавить `"datasets>=2.20,<4"` + `uv lock`

## Key Files

**`pyproject.toml`**
- Role: объявление зависимостей проекта
- Planned change: добавить `"datasets>=2.20,<4"` в `[project.dependencies]`
- Status: NOT STARTED

**`src/atm/tasks/__init__.py`**
- Role: публичный API модуля tasks — re-exports + guarded imports loader-модулей
- Planned change: переписать пустой skeleton в полный модуль с `__all__`
- Status: NOT STARTED

**`src/atm/tasks/base.py`**
- Role: Protocol-ы, Registry-ы, EvalResult, модульные инстансы `TASKS` и `EVALUATORS`
- Planned change: создать с нуля (`LLMLike`, `Evaluator`, `EvalResult`, `EvaluatorRegistry`, `TaskLoader`, `TaskRegistry`)
- Status: NOT STARTED

**`src/atm/tasks/_cache.py`**
- Role: parquet-кэш HF датасетов — atomic write/read, избегает повторных сетевых запросов
- Planned change: создать с нуля
- Status: DONE (Phase 3)

**`src/atm/tasks/_judge.py`**
- Role: shared хелпер `_invoke_judge` для LLM-judge evaluator-ов (creative + analysis)
- Planned change: создать с нуля; Steps 6 и 7 импортируют отсюда, а не дублируют логику
- Status: DONE (Phase 3)

**`src/atm/tasks/humaneval.py`**
- Role: HumanEval loader + sandbox evaluator
- Planned change: создать; включает `_strip_code_fences` хелпер
- Status: NOT STARTED

**`src/atm/tasks/mmlu.py`**
- Role: MMLU-Pro loader + exact-match evaluator (A-J 10-way)
- Planned change: создать
- Status: NOT STARTED

**`src/atm/tasks/creative.py`**
- Role: Creative YAML loader + LLM-judge evaluator (использует `_judge._invoke_judge`)
- Planned change: создать
- Status: NOT STARTED

**`src/atm/tasks/analysis.py`**
- Role: Analysis YAML loader + hybrid evaluator (structural + LLM-judge через `_judge._invoke_judge`)
- Planned change: создать
- Status: NOT STARTED

**`src/atm/__init__.py`**
- Role: top-level пакетный re-export
- Planned change: добавить `TaskRegistry`, `EvaluatorRegistry`, `TASKS`, `EVALUATORS`
- Status: NOT STARTED

**`conf/tasks/creative_prompts.yaml`**
- Role: 8 creative промптов с rubric (источник данных для CreativeLoader)
- Planned change: создать
- Status: NOT STARTED

**`conf/tasks/analysis_prompts.yaml`**
- Role: 6 analysis задач с inline data_csv и structural_checks
- Planned change: создать
- Status: NOT STARTED

**`tests/unit/tasks/__init__.py`**
- Role: активация `@register_*` декораторов при pytest collection
- Planned change: создать с `from atm import tasks  # noqa: F401`
- Status: NOT STARTED

**`tests/unit/tasks/test_registry_smoke.py`**
- Role: автоматический smoke-тест registry с pre-seeded fixture-кэшем (не сетевой)
- Planned change: создать в Phase 5; тест `TASKS.get("humaneval").sample(10, seed=42)`
- Status: NOT STARTED

**`dev/codebase-map.md`**
- Role: карта кодовой базы
- Planned change: обновить секцию Tasks & Evaluation — статус M10, список loader-имён, evaluator-ключей, путь к кэшу
- Status: NOT STARTED

## Decisions

**LLMLike Protocol для judge DI**
- Decision: ввести `LLMLike` как runtime-checkable Protocol в `base.py` вместо использования конкретного `LLMWrapper` или `Any`
- Rationale: позволяет DI `FakeLLM` в тестах и `LLMWrapper` в production без привязки к классу; соответствует duck-typing природе Python

**`_judge.py` создаётся в Phase 3 (Wave 3)**
- Decision: `_invoke_judge` хелпер вынесен в отдельный `_judge.py` и создаётся в Phase 3, а не inline в creative/analysis
- Rationale: Steps 6 и 7 выполняются параллельно в Wave 4; если оба реализуют inline, возникает risk дублирования или race condition при рефакторинге. Создание в Phase 3 устраняет это.

**JSON parse fallback в `_invoke_judge`**
- Decision: `json.loads` + regex fallback `re.search(r'\{.*\}', text, re.DOTALL)` для noisy LLM outputs
- Rationale: реальные LLM-ы часто оборачивают JSON в markdown или добавляют prefix-текст

**Parquet metadata как bytes**
- Decision: все keys и values в pyarrow schema metadata должны быть `bytes` (например `b"created_at_utc"`)
- Rationale: pyarrow требует bytes для metadata; строки вызывают runtime TypeError. Footgun задокументирован явно в `_cache.py`

**`dataset_revision: str | None` в parquet metadata**
- Decision: добавить опциональное поле `dataset_revision` в write_cache signature и в Parquet metadata
- Rationale: reproducibility — знать какая revision HF датасета была закэширована

**SubprocessSandbox только с trusted кодом**
- Decision: HumanEval evaluator в тестах использует SubprocessSandbox с trusted canonical_solution из fixture
- Rationale: DockerSandbox требует Docker daemon; в тестах безопасно использовать SubprocessSandbox только с trusted кодом из fixture

**TASKS, EVALUATORS как модульные инстансы**
- Decision: `TASKS = TaskRegistry()` и `EVALUATORS = EvaluatorRegistry()` экспортируются как инстансы на уровне модуля; декораторы используют эти инстансы
- Rationale: singleton pattern через module-level instance (как в `topology/__init__.py`); не глобальный singleton

## Constraints

- `TaskSpec` уже существует в `src/atm/core/types.py:263` — не пересоздавать, только импортировать
- HF `load_dataset` вызовы запрещены в unit-тестах (`filterwarnings = "error"` может поймать неожиданные сетевые вызовы); мокировать через `monkeypatch`
- `@pytest.mark.network` тесты запускаются только вручную с `ATM_ENABLE_NETWORK_TESTS=1`
- Каждый тест должен использовать `tmp_path` fixture для кэша, не реальный `data/cache/`
- mypy --strict обязателен для всех новых файлов в `src/atm/tasks/`
- MMLU-Pro использует 10-way multi-choice A-J (не A-D как оригинальный MMLU)

## Amendments Applied

1. **Step 3 расширен `_judge.py`**: в Phase 3 добавляется `src/atm/tasks/_judge.py` с `_invoke_judge(llm_like, *, prompt, agent_id) -> tuple[float, str, str | None]`. Steps 6 и 7 импортируют этот хелпер вместо локальной inline реализации. Устраняет risk race condition в Wave 4.

2. **`LLMLike` Protocol в `base.py`**: Step 2 вводит `LLMLike` — runtime-checkable Protocol с `async ainvoke(messages, *, agent_id, ...) -> LLMResponse`. Используется в сигнатурах конструкторов `CreativeJudgeEvaluator` и `AnalysisHybridEvaluator` вместо конкретного `LLMWrapper` или `Any`.

3. **Smoke-тест как автотест**: Step 8 smoke "TASKS.get('humaneval').sample(10, seed=42)" вынесен в `tests/unit/tasks/test_registry_smoke.py` с pre-seeded fixture-кэшем. Не сетевой, запускается автоматически в test suite.

4. **Parquet bytes metadata + `dataset_revision`**: `write_cache` явно использует `bytes` для metadata keys/values (footgun pyarrow задокументирован). Добавлено опциональное `dataset_revision: str | None` для reproducibility.

5. **Без правки `.gitignore`**: `data/` уже глобально gitignored; Step 1 не редактирует `.gitignore`.

6. **`tests/unit/tasks/__init__.py` с импортом tasks**: файл содержит `from atm import tasks  # noqa: F401` чтобы `@register_*` декораторы активировались при pytest collection.

7. **`_strip_code_fences` в humaneval + тест**: хелпер убирает markdown ``` ```python ... ``` ``` обёртки из LLM-ответов; тест на него включён в `test_humaneval.py` (Step 4, 6 тестов итого вместо 5).
