# Add Cerebras Provider — Tasks

## Phase 1: Install dependency + add pricing (parallel) COMPLETE

- [x] 1.1 Добавить `"langchain-cerebras>=0.5,<0.6"` в `pyproject.toml` и запустить `uv lock` — `pyproject.toml`, `uv.lock`
  - Acceptance: `uv sync` без ошибок; `python -c "import langchain_cerebras"` без ImportError; `uv tree | grep langchain-core` показывает `0.3.x`; `grep langchain-cerebras uv.lock` показывает запись `0.5.x`
  - Note: добавлять в алфавитном порядке между `langchain-anthropic` и `langchain-openai`; inline comment `# pinned to 0.5.x — 0.6+ drops langchain-core<0.4 upper bound`

- [x] 1.2 Добавить ценовые записи в `conf/pricing.yaml` — `conf/pricing.yaml`
  - Acceptance: `Pricing.from_yaml("conf/pricing.yaml")` загружает `cerebras:llama3.1-8b` и `cerebras:gpt-oss-120b`; `grep -E '70b' conf/pricing.yaml | grep cerebras` возвращает пусто; цены помечены `# VERIFY`; `llama3.1-8b`-блок содержит `# DEPRECATION 2026-05-27`
  - Note: цены-заглушки (`0.0001` и `0.00085`/`0.00120`) с `# VERIFY` — developer сверяет с Cerebras Cloud dashboard перед merge; cache-поля = `0.0`

## Phase 2: Implement provider factory COMPLETE

- [x] 2.1 Создать `src/atm/llm/providers/cerebras.py` с `build_cerebras` — `src/atm/llm/providers/cerebras.py`
  - Acceptance: файл содержит `build_cerebras(model_id: str, opts: dict[str, Any]) -> BaseChatModel`; `mypy src/atm/llm/providers/cerebras.py` чистый; `ruff check` чистый; docstring содержит supported + removed model IDs
  - Note: **обязательный первый шаг** — `python -c "import inspect; from langchain_cerebras import ChatCerebras; print(inspect.signature(ChatCerebras.__init__))"` для подтверждения имени kwarg (`api_key` vs `cerebras_api_key`); `setdefault("api_key", "EMPTY")` или аналог

- [x] 2.2 Добавить `TestBuildCerebras` в `tests/unit/llm/test_providers.py` — `tests/unit/llm/test_providers.py`
  - Acceptance: класс содержит 7 тестов: `test_returns_base_chat_model`, `test_returns_chat_cerebras_instance`, `test_strips_cerebras_prefix`, `test_forwards_temperature`, `test_no_api_call_on_construction`, `test_api_key_is_empty_by_default`, `test_supports_active_models`; существующие классы не тронуты
  - Note: `test_supports_active_models` итерирует только по `("cerebras:llama3.1-8b", "cerebras:gpt-oss-120b")` — dead IDs отсутствуют

## Phase 3: Wire re-exports + docs COMPLETE

- [x] 3.1 Реэкспортировать `build_cerebras` в `providers/__init__.py` и `atm/llm/__init__.py` — `src/atm/llm/providers/__init__.py`, `src/atm/llm/__init__.py`
  - Acceptance: `from atm.llm import build_cerebras` без ImportError; `from atm.llm.providers import build_cerebras` без ImportError; `ruff check` чистый; `pytest tests/unit/llm/test_providers.py::TestBuildCerebras` все pass
  - Note: соблюдать алфавитный порядок в `__all__`; `build_cerebras` идёт между `build_anthropic` и `build_openai`

- [x] 3.2 Обновить `dev/codebase-map.md` и `.env.example` (если существует) — `dev/codebase-map.md`, `.env.example`
  - Acceptance: `grep -i cerebras dev/codebase-map.md` находит провайдер, обе модели, зависимость; `grep CEREBRAS_API_KEY .env.example` (если файл есть) находит запись

## Phase 4: Final verification COMPLETE

- [x] 4.1 Запустить полный тест-suite + lint + mypy — (read-only)
  - Acceptance: `uv run pytest tests/unit/llm/ -v` — все pass; `uv run ruff check src/ tests/` — clean; `uv run mypy src/atm/llm/` — clean; `grep -rE '(llama3\.1-70b|llama-3\.3-70b)' src/ tests/ conf/` — только "Removed" warning-секция в docstring `cerebras.py`

---
## Stats
- Total: 7 tasks · ~1.25h
- Done: 6 / 7

## How to Update
Check off tasks with [x] and update `add-cerebras-provider-context.md` SESSION PROGRESS after each milestone.
Phase header: all tasks done -> "COMPLETE", some done -> "IN PROGRESS".
