# Add Cerebras Provider — Context

## SESSION PROGRESS (2026-04-26)

### COMPLETED
- Step 1.1: Added `langchain-cerebras>=0.5,<0.6` to `pyproject.toml` (alphabetically between `langchain-anthropic` and `langchain-openai`), regenerated `uv.lock` (resolved `langchain-cerebras v0.5.0`), ran `uv sync`; verified import OK and `langchain-core` stays at `0.3.84`
- Step 1.2: Added `cerebras:llama3.1-8b` and `cerebras:gpt-oss-120b` entries to `conf/pricing.yaml`; both entries have `# VERIFY` markers on price values, cache fields set to `0.0`, and `# DEPRECATION 2026-05-27` comment above the 8B entry; Pricing.from_yaml loads both keys OK; no 70b IDs present
- Step 2.1+2.2 (TDD combined): Confirmed via introspection that `ChatCerebras` uses `api_key` kwarg (not `cerebras_api_key`). Wrote `TestBuildCerebras` (8 test cases — 7 methods, 1 parametrized with 2 values) in `tests/unit/llm/test_providers.py`; confirmed red. Created `src/atm/llm/providers/cerebras.py` with `build_cerebras` factory matching `build_openai` pattern; all 8 test cases pass, 34/34 total providers tests pass. `mypy` and `ruff` clean.
- Step 3.1+3.2: Added `build_cerebras` re-export to `providers/__init__.py` and `atm/llm/__init__.py` (alphabetical order in both import and `__all__`); added `CEREBRAS_API_KEY=` placeholder to `.env.example`; updated `dev/codebase-map.md` with provider entry (both models + deprecation), `build_cerebras` in exports list, and `langchain-cerebras` in external dependencies. All 8 TestBuildCerebras tests pass; ruff clean; both import paths verified.

### IN PROGRESS
- Phase 4: Final verification (Step 4.1 — full test suite + lint + mypy)

### BLOCKERS
- None

## Quick Resume
1. Read this file
2. Check `add-cerebras-provider-tasks.md` for what's next
3. Read `add-cerebras-provider-plan.md` Phase 1 for strategy
4. Start with: Task 1.1 — добавить `langchain-cerebras>=0.5,<0.6` в `pyproject.toml` и запустить `uv lock`

## Key Files

**`pyproject.toml`**
- Role: манифест проекта, секция `[project] dependencies`
- Planned change: добавить `"langchain-cerebras>=0.5,<0.6"` в алфавитном порядке между `langchain-anthropic` и `langchain-openai`
- Status: NOT STARTED

**`uv.lock`**
- Role: детерминистичный lockfile
- Planned change: регенерировать через `uv lock` после правки `pyproject.toml`
- Status: NOT STARTED

**`conf/pricing.yaml`**
- Role: ценовые данные для `LLMWrapper.cost_usd`
- Planned change: добавить два блока — `cerebras:llama3.1-8b` и `cerebras:gpt-oss-120b` с `# VERIFY`-маркерами и `# DEPRECATION 2026-05-27` для 8B-модели
- Status: DONE

**`src/atm/llm/providers/cerebras.py`**
- Role: фабрика провайдера Cerebras (новый файл)
- Planned change: создать по шаблону `openai.py`; функция `build_cerebras(model_id, opts) -> BaseChatModel`; `setdefault("api_key", "EMPTY")`; docstring с supported/removed model IDs
- Status: DONE

**`src/atm/llm/providers/__init__.py`**
- Role: публичный API пакета `atm.llm.providers`
- Planned change: добавить `from atm.llm.providers.cerebras import build_cerebras` и `"build_cerebras"` в `__all__` (алфавитный порядок)
- Status: DONE

**`src/atm/llm/__init__.py`**
- Role: публичный API пакета `atm.llm`
- Planned change: добавить `build_cerebras` в import-блок из `atm.llm.providers` и в `__all__`
- Status: DONE

**`tests/unit/llm/test_providers.py`**
- Role: unit-тесты провайдеров LLM
- Planned change: добавить класс `TestBuildCerebras` (7 тестов); существующие классы не трогать
- Status: DONE (8 test cases — 7 methods including 1 parametrized with 2 values)

**`dev/codebase-map.md`**
- Role: архитектурная карта кодовой базы
- Planned change: добавить cerebras в список провайдеров, обе активные модели, зависимость `langchain-cerebras`
- Status: DONE

**`.env.example`** (if exists)
- Role: шаблон переменных окружения
- Planned change: добавить строку `CEREBRAS_API_KEY=`
- Status: DONE

## Decisions

- **Модельный набор**: только `llama3.1-8b` + `gpt-oss-120b`. Обе 70B-модели (`llama3.1-70b` удалена 2025-01-17, `llama-3.3-70b` удалена 2026-02-16) исключены по подтверждённому live-статусу платформы и явному решению пользователя ("достаточно разбивки 70b+"). Dead model IDs нигде в коде/конфигах не появляются.
  - Rationale: нерабочие model IDs в конфиге создают misleading developer experience и потенциальные 404 в runtime.

- **Pin `langchain-cerebras>=0.5,<0.6`**: верхняя граница `<0.6` предотвращает case upgrade к версиям, нарушающим текущий pin `langchain-core<0.4`. Проверено через PyPI metadata 2026-04-26: `0.5.x` требует `langchain-core>=0.3.29,<0.4.0` — точное попадание в lock проекта (`0.3.84`). Утверждение ревьювера о том, что 0.5.0 требует `langchain-core>=1.1.0` — опровергнуто как hallucination.
  - Rationale: стабильность транзитивных зависимостей критична; неожиданный upgrade `langchain-core` сломает весь LLM stack.

- **Без изменений `factory.py` / `wrapper.py`**: `init_chat_model` LangChain резолвит префикс `cerebras:` нативно после установки пакета. `_count_prompt_tokens` heuristic (4 chars/token) — приемлем для Llama/gpt-oss. `_detect_and_parse_usage` использует OpenAI-shape — Cerebras API совместим.
  - Rationale: минимальный footprint изменений; существующий code path уже обрабатывает Cerebras корректно.

- **`setdefault("api_key", "EMPTY")`**: по аналогии с `build_openai` — предотвращает падение при конструкции в тестах без env-var. Имя kwarg требует подтверждения через `inspect.signature(ChatCerebras.__init__)` перед написанием кода.
  - Rationale: инвариант "no network call on construction" должен держаться для всех провайдеров.

- **Без интеграционных тестов**: контракт `LLMWrapper.ainvoke` уже покрыт через FakeLLM; новый провайдер наследует покрытие. Тесты с реальным API потребовали бы ключа и были бы flaky.

## Constraints

- `langchain-core` должен оставаться в диапазоне `0.3.x` (текущий lock `0.3.84`).
- `langchain-openai` должен оставаться в диапазоне `0.3.x` (текущий lock `0.3.35`).
- Никаких dead model IDs в `pricing.yaml`, `cerebras.py`, тестах, или конфигах.
- `# VERIFY`-маркеры в `pricing.yaml` — обязательны; цены должны быть сверены с Cerebras Cloud dashboard перед merge.
- Deprecation `llama3.1-8b` — 2026-05-27; cleanup-задача создаётся отдельно.

## Amendments Applied

- **Модельный набор сужен**: исходный запрос включал `llama-3.1-8b + llama-3.1-70b + llama-3.3-70b`. После проверки live-статуса обе 70B-модели подтверждены как удалённые. Пользователь явно подтвердил сокращение до 2 активных моделей.
- **Корректный ID 8B-модели**: `llama3.1-8b` (без дефиса перед версией) — именно такой ID на Cerebras platform, не `llama-3.1-8b`.
- **Опровержение hallucination ревьювера**: утверждение "0.5.0 requires langchain-core>=1.1.0" опровергнуто прямой проверкой PyPI metadata; pin `>=0.5,<0.6` подтверждён корректным.
