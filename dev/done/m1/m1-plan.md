# M1 — Core types & state — Plan

## Executive Summary

Заложить фундамент `src/atm/core/` — Pydantic v2 типы данных, LangGraph-совместимый TypedDict state с кастомными reducer'ами и иерархию ошибок. Все последующие milestones (M2 LLM wrapper, M3 storage, M5 agents, M6+ topology, M8 adaptive) импортируют из этого модуля, поэтому API должен быть стабильным и полностью соответствовать `dev/arch.md §3`. SQLAlchemy-модель и Alembic-миграция для `topology_transitions` не создаются — это отложено до M8.

## Current State

M0 завершён: `pyproject.toml` настроен (Python ≥3.11, ruff strict, mypy strict, pytest с `filterwarnings=["error"]` и `asyncio_mode=auto`), пакет `src/atm` пустой — каталог `core/` ещё не создан. Smoke-test в `tests/` зелёный. Зависимость `pydantic` в `pyproject.toml` отсутствует (`dependencies = []`).

## Proposed Approach

Четыре волны параллельной работы:

1. **Wave 1 — Step 0**: Добавить `pydantic>=2.7,<3` в зависимости, создать каркас каталогов.
2. **Wave 2 — Steps 1/2/3 (параллельно)**: Три независимых файла — `errors.py`, `types.py`, `reducers.py` — плюс их unit-тесты. Шаг 3 использует только dict/namedtuple-заглушки (не импортирует types.py), что гарантирует независимость.
3. **Wave 3 — Step 4**: `state.py` — импортирует из types.py и reducers.py; TypedDict-ы с Annotated-reducer'ами.
4. **Wave 4 — Step 5**: Публичный API (`__init__.py`), интеграционный тест, обновление codebase-map.md.

Ключевые архитектурные решения: TypedDict (не Pydantic BaseModel) для state-контейнеров; `datetime.now(timezone.utc)` вместо устаревшего `utcnow()`; `dedup_by_id_reducer` — фабрика замыканий с left-wins при коллизии id; `Message.to_lc`/`from_lc` — stub'ы, бросающие `NotImplementedError("...M2...")`.

## Implementation Phases

### Phase 1: Подготовка — зависимости и каркас (~0.5h)
**Goal:** Добавить Pydantic v2 и создать пустые модули-заглушки.

- [ ] 1.1 Добавить `pydantic>=2.7,<3` через `uv add`
  - File: `pyproject.toml`, `uv.lock`
  - Acceptance: `uv run python -c "import pydantic; assert pydantic.VERSION.startswith('2.')"` — успех; M0 smoke-test проходит.
- [ ] 1.2 Создать каталог `src/atm/core/` с пустым `__init__.py`
  - File: `src/atm/core/__init__.py`
  - Acceptance: файл существует; `from atm.core import` не падает с ImportError.
- [ ] 1.3 Создать `tests/unit/__init__.py` и `tests/unit/core/__init__.py` (если ещё не существуют)
  - File: `tests/unit/__init__.py`, `tests/unit/core/__init__.py`
  - Acceptance: оба файла существуют; pytest находит каталог.

### Phase 2: Три независимых модуля (Wave 2) (~2.5h, параллельно)
**Goal:** Реализовать errors.py, types.py, reducers.py с TDD-покрытием.

- [ ] 2.1 `src/atm/core/errors.py` — иерархия исключений
  - File: `src/atm/core/errors.py`, `tests/unit/core/test_errors.py`
  - Acceptance: 5+ тестов зелёные; `uv run mypy src/atm/core/errors.py` — no issues; `BudgetExceededError`, `PhaseError`, `ToolError`, `AtmError` импортируются.
- [ ] 2.2 `src/atm/core/types.py` — 17 Pydantic-классов и enums
  - File: `src/atm/core/types.py`, `tests/unit/core/test_types.py`
  - Acceptance: 15+ тестов зелёные; mypy strict чист; `Message(sender="a", kind="request", content="hi")` — создаётся без ошибок; no DeprecationWarning.
- [ ] 2.3 `src/atm/core/reducers.py` — `merge_agent_states` + `dedup_by_id_reducer`
  - File: `src/atm/core/reducers.py`, `tests/unit/core/test_reducers.py`
  - Acceptance: 21+ тестов зелёные (10 для merge_agent_states + 10 для dedup + 1 extra); mypy strict чист; тесты используют только dict/namedtuple (без импорта types.py).

### Phase 3: TypedDict-ы состояния (~1h)
**Goal:** Определить AgentState, SharedState, GraphState с Annotated-reducer'ами.

- [ ] 3.1 `src/atm/core/state.py` — три TypedDict-а
  - File: `src/atm/core/state.py`, `tests/unit/core/test_state.py`
  - Acceptance: 8+ тестов зелёные (включая MC-3 тест имени closure `_reduce` и MC-4 mypy smoke-check); `typing.get_type_hints(GraphState, include_extras=True)` возвращает Annotated-метаданные с callable reducer'ами.

### Phase 4: Публичный API и финализация (~0.5h)
**Goal:** Один импорт `from atm.core import ...` покрывает всё нужное; полный pytest-прогон зелёный.

- [ ] 4.1 Наполнить `src/atm/core/__init__.py` реэкспортами и `__all__`
  - File: `src/atm/core/__init__.py`
  - Acceptance: `from atm.core import GraphState, Message, BudgetExceededError, merge_agent_states` — работает; ~25 имён в `__all__`.
- [ ] 4.2 `tests/unit/core/test_public_api.py` — IR-4 + MC-5 + end-to-end reducer тест
  - File: `tests/unit/core/test_public_api.py`
  - Acceptance: все тесты зелёные; MC-5 (dedup с Pydantic Message) и IR-4 (__all__ non-None coverage) проходят.
- [ ] 4.3 Обновить `dev/codebase-map.md` — статус M1
  - File: `dev/codebase-map.md`
  - Acceptance: строка «M1 in progress» заменена на «M1 complete».
- [ ] 4.4 Финальный полный прогон
  - Acceptance: `uv run pytest -q` — все тесты passed, 0 warnings; `uv run mypy src/atm` — no issues; `uv run ruff check src/ tests/` — clean.

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `pyproject.toml` | Добавить `pydantic>=2.7,<3` в dependencies | Runtime-зависимость для типов |
| `uv.lock` | Обновить lockfile | Фиксация версий |
| `src/atm/core/__init__.py` | Создать; наполнить реэкспортами + `__all__` | Публичный API модуля |
| `src/atm/core/errors.py` | Создать | Иерархия исключений фреймворка |
| `src/atm/core/types.py` | Создать | 17 Pydantic-классов и enums |
| `src/atm/core/reducers.py` | Создать | LangGraph-reducer'ы для state merge |
| `src/atm/core/state.py` | Создать | TypedDict-контейнеры состояния |
| `tests/unit/core/test_errors.py` | Создать | Unit-тесты иерархии ошибок |
| `tests/unit/core/test_types.py` | Создать | Unit-тесты Pydantic-моделей |
| `tests/unit/core/test_reducers.py` | Создать | Unit-тесты reducer-инвариантов |
| `tests/unit/core/test_state.py` | Создать | Unit-тесты TypedDict-ов и Annotated |
| `tests/unit/core/test_public_api.py` | Создать | Integration-тест публичного API |
| `dev/codebase-map.md` | Обновить статус | Фиксация завершения M1 |

## Dependencies & Order Constraints

- Phase 1 (Step 0) — блокирует всё остальное: нужен pydantic + каталоги.
- Phase 2 шаги 2.1/2.2/2.3 — полностью параллельны между собой (независимые файлы; шаг 2.3 не импортирует types.py).
- Phase 3 (Step 4, state.py) — требует завершения 2.2 (types.py) и 2.3 (reducers.py).
- Phase 4 (Step 5, public API) — требует завершения всех предыдущих фаз.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| mypy strict с `Annotated[..., closure]` требует `# type: ignore[valid-type]` | Средняя | Низкий | MC-4: добавлять `# type: ignore` с обязательным комментарием-обоснованием только при реальной ошибке |
| `pydantic` подтягивает несовместимую `typing-extensions` | Низкая | Средний | IR-5: добавить `typing-extensions>=4.10` только если mypy strict заблокирует |
| `DeprecationWarning` из `datetime.utcnow` в pytest | Высокая (если не исправить) | Высокий | Использовать `datetime.now(timezone.utc)` во всех `default_factory` |
| Коллизия имён в `__all__` с будущими модулями | Низкая | Низкий | grep перед финализацией; IR-4 тест поймает в CI |
| Неверное направление left/right в dedup_reducer | Средняя | Высокий | MC-2 усиленный тест: проверка не только id, но и payload left-wins |
| Циклический импорт reducers.py ↔ state.py | Средняя | Высокий | Forward-ref через `TYPE_CHECKING` гвард в reducers.py |

## Out of Scope

- SQLAlchemy-модель и Alembic-миграция для `topology_transitions` — отложено до M8.
- `Message.to_lc` / `Message.from_lc` реализация — M2 (сейчас stub с NotImplementedError).
- `model_validator`-ы на инварианты (например `total_tokens = prompt + completion`) — M2.
- LangGraph `StateGraph` инстанциирование и тестирование — LangGraph не устанавливается на M1.
- `EvaluatorSpec` Protocol для `TaskSpec.evaluator_key` — M10 (реестр).
- Реэкспорт на уровне `src/atm/__init__.py` — минимизируем поверхность; только `atm.core`.
- `typing-extensions` как явная зависимость — только при реальной ошибке mypy (IR-5).

## Timeline
- Total: ~4.5h
- Created: 2026-04-23
