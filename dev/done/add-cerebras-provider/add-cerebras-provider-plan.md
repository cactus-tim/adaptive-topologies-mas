# Add Cerebras Provider — Plan

## Executive Summary

Добавить нового LLM-провайдера Cerebras в `atm.llm.providers`, повторяя паттерн существующих провайдеров (`build_openai`, `build_anthropic`, `build_vllm`). Поддерживаются **две активные модели** на платформе Cerebras по состоянию на 2026-04-26: `llama3.1-8b` (8B tier, upcoming deprecation 2026-05-27) и `gpt-oss-120b` (120B tier — единственная активная модель уровня 70B+ после удаления обеих 70B-моделей). Провайдер берёт API-ключ из `CEREBRAS_API_KEY`, имеет фабричную функцию `build_cerebras`, регистрируется в публичном API и маршрутизируется через префикс `cerebras:` в LangChain `init_chat_model`.

## Current State

- В `src/atm/llm/providers/` существуют три провайдера: `openai.py`, `anthropic.py`, `vllm.py`.
- Все следуют паттерну: `build_*(model_id: str, opts: dict[str, Any]) -> BaseChatModel`.
- `atm.llm.factory.build_llm` делегирует `LLMWrapper`, который использует `init_chat_model` из LangChain. Префикс `cerebras:` поддерживается `init_chat_model` нативно после установки `langchain-cerebras`.
- `conf/pricing.yaml` содержит записи для OpenAI, Anthropic, vLLM и fake-моделей. Cerebras-записей нет.
- `langchain-cerebras` не установлен в проекте.

## Proposed Approach

1. Установить `langchain-cerebras>=0.5,<0.6` — единственная версия, совместимая с текущим pin'ом `langchain-core>=0.3,<0.4` (lock: 0.3.84). Версии 0.6+ снимают upper bound; 0.8+ требуют `langchain-core>=1.1.0`.
2. Добавить ценовые записи в `conf/pricing.yaml` для двух активных модель-ID: `cerebras:llama3.1-8b` и `cerebras:gpt-oss-120b`. Dead model IDs (`llama3.1-70b`, `llama-3.3-70b`) в YAML не добавляются.
3. Реализовать `build_cerebras` в `src/atm/llm/providers/cerebras.py` по шаблону `openai.py` с unit-тестами в `TestBuildCerebras`.
4. Реэкспортировать `build_cerebras` через `providers/__init__.py` и `atm/llm/__init__.py`.
5. Обновить документацию (`.env.example`, `dev/codebase-map.md`).
6. Прогнать полный тест-suite + lint + mypy.

`init_chat_model` в LangChain резолвит префикс `cerebras:` автоматически — изменений в `factory.py` и `wrapper.py` не требуется.

## Implementation Phases

### Phase 1: Install dependency + add pricing (~0.25h)
**Goal:** Подготовить внешнюю зависимость и ценовые данные параллельно.

- [ ] 1.1 Добавить `langchain-cerebras>=0.5,<0.6` в `pyproject.toml` и обновить lockfile
  - File: `pyproject.toml`, `uv.lock`
  - Acceptance: `uv sync` завершается без ошибок; `python -c "import langchain_cerebras"` без ImportError; `langchain-core` остаётся в диапазоне `0.3.x`

- [ ] 1.2 Добавить ценовые записи Cerebras в `conf/pricing.yaml`
  - File: `conf/pricing.yaml`
  - Acceptance: `Pricing.from_yaml("conf/pricing.yaml")` загружает оба ключа `cerebras:llama3.1-8b` и `cerebras:gpt-oss-120b` без KeyError; `grep -E '70b' conf/pricing.yaml | grep cerebras` возвращает пусто

### Phase 2: Implement provider factory (~0.5h)
**Goal:** Создать `build_cerebras` с полным unit-покрытием.

- [ ] 2.1 Создать `src/atm/llm/providers/cerebras.py` с функцией `build_cerebras`
  - File: `src/atm/llm/providers/cerebras.py`
  - Acceptance: файл содержит `build_cerebras`; introspection `ChatCerebras.__init__` выполнен до написания кода; `mypy src/atm/llm/providers/cerebras.py` чистый

- [ ] 2.2 Добавить класс `TestBuildCerebras` в `tests/unit/llm/test_providers.py`
  - File: `tests/unit/llm/test_providers.py`
  - Acceptance: 7 тестов в классе; `pytest tests/unit/llm/test_providers.py::TestBuildCerebras` — все pass (после шага 3.1)

### Phase 3: Wire re-exports + docs (~0.25h)
**Goal:** Включить `build_cerebras` в публичный API и актуализировать документацию.

- [ ] 3.1 Реэкспортировать `build_cerebras` в `providers/__init__.py` и `atm/llm/__init__.py`
  - File: `src/atm/llm/providers/__init__.py`, `src/atm/llm/__init__.py`
  - Acceptance: `from atm.llm import build_cerebras` работает; `from atm.llm.providers import build_cerebras` работает; `ruff check` чистый

- [ ] 3.2 Обновить `.env.example` (если существует) и `dev/codebase-map.md`
  - File: `.env.example` (if exists), `dev/codebase-map.md`
  - Acceptance: `grep -i cerebras dev/codebase-map.md` находит обновлённые строки; провайдер и обе модели присутствуют в карте

### Phase 4: Final verification (~0.25h)
**Goal:** Убедиться, что весь suite зелёный и в кодовой базе нет dead model IDs.

- [ ] 4.1 Запустить полный тест-suite + lint + mypy
  - File: (read-only)
  - Acceptance: `uv run pytest tests/unit/llm/ -v` — все pass; `ruff check src/ tests/` — clean; `mypy src/atm/llm/` — clean; `grep -rE '(llama3\.1-70b|llama-3\.3-70b)' src/ tests/ conf/` — ничего кроме "Removed" warning-секции в docstring

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `pyproject.toml` | Добавить зависимость `langchain-cerebras>=0.5,<0.6` | Установка LangChain-интеграции Cerebras |
| `uv.lock` | Regenerate через `uv lock` | Детерминистичный lockfile |
| `conf/pricing.yaml` | Добавить 2 записи (`cerebras:llama3.1-8b`, `cerebras:gpt-oss-120b`) | Cost-tracking для LLMWrapper |
| `src/atm/llm/providers/cerebras.py` | Новый файл — `build_cerebras` factory | Провайдер Cerebras по паттерну проекта |
| `src/atm/llm/providers/__init__.py` | Добавить import + `__all__` entry | Публичный API providers |
| `src/atm/llm/__init__.py` | Добавить re-export + `__all__` entry | Публичный API `atm.llm` |
| `tests/unit/llm/test_providers.py` | Добавить класс `TestBuildCerebras` (7 тестов) | Unit coverage провайдера |
| `dev/codebase-map.md` | Добавить cerebras в список провайдеров | Актуальность документации |
| `.env.example` | Добавить `CEREBRAS_API_KEY=` (если файл существует) | Developer onboarding |

## Dependencies & Order Constraints

- Phase 1 (1.1 и 1.2) — **параллельно**: `pyproject.toml` и `pricing.yaml` независимы.
- Phase 2 (2.1 и 2.2) — **после 1.1**: нужна установленная `langchain-cerebras` для introspection; 2.1 и 2.2 можно делать вместе (один файл — один commit).
- Phase 3 (3.1) — **после 2.1**: нужно существование `cerebras.py` для импорта в `__init__.py`.
- Phase 3 (3.2) — **после 3.1**: документация финализируется после wire-up.
- Phase 4 — **после всего**: финальный gate.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `langchain-cerebras 0.5.x` тянет конфликтующий `pydantic`/`httpx` | Low | High | `uv lock` сообщит конфликт немедленно; откат — удалить строку из `pyproject.toml` |
| `ChatCerebras.__init__` использует `cerebras_api_key` вместо `api_key` | Medium | Low | Mandatory introspect `inspect.signature` перед написанием кода |
| `ChatCerebras` eager-валидирует API-ключ при конструкции | Low | Medium | Если `api_key="EMPTY"` не работает — использовать `monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")` |
| Цены в `pricing.yaml` неточные | High | Low | `# VERIFY` markers; developer сверяет с Cerebras Cloud dashboard перед merge |
| `llama3.1-8b` устареет 2026-05-27 | Certain | Low | Deprecation marker в YAML + docstring; cleanup — отдельная задача |

## Out of Scope

- Изменения в `factory.py` или `wrapper.py` — `init_chat_model` резолвит `cerebras:` автоматически.
- Изменения в `_count_prompt_tokens` — heuristic 4 chars/token достаточен для Llama/gpt-oss.
- Интеграционные тесты с реальным Cerebras API — требуют валидного ключа, flaky/дорогие.
- Добавление dead model IDs (`llama3.1-70b` удалён 2025-01-17, `llama-3.3-70b` удалён 2026-02-16).
- Стратегия замены `llama3.1-8b` после 2026-05-27 — отдельная задача.
- Расширение tiktoken-покрытия для Cerebras-моделей — отдельная задача при необходимости.

## Timeline
- Total: ~1.25h
- Created: 2026-04-26
