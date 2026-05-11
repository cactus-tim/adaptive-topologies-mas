# m8-foundation — Контекст

## ПРОГРЕСС СЕССИИ (2026-05-11)

### ВЫПОЛНЕНО
- **Wave 1** (5/13): аудит M1/M3-артефактов (.audit-baseline.md), TopologyDecision+PhaseDecision в core/types.py, re-export в core/__init__.py, 7 тестов в test_types.py, no-op подтверждение reducer-инвариантов с docstring-указателями. 177/177 в core suite.
- **Wave 2** (8/13): phases/manager.py — PhaseRouter Protocol, PhaseGuard type alias, PhaseLimits (frozen Pydantic), RuleBasedPhaseRouter (4 встроенных guards: ready_for_execution, ready_for_verification, critic_approved, iter_caps). 17 тестов TestRuleBased + import test. Полный suite 886/886.

### В ПРОЦЕССЕ
- Wave 3 (Step 3.1+3.2): LLMPhaseRouter с AsyncMock-тестами (append к phases/manager.py)

### БЛОКЕРЫ
- Нет

## Быстрое возобновление

1. Прочитать этот файл
2. Проверить `m8-foundation-tasks.md` — что делать следующим
3. Прочитать `m8-foundation-plan.md` Фаза 1 — стратегия
4. Начать с: аудита существующих артефактов M1–M3 (Step 1.1) и создания `TopologyDecision`/`PhaseDecision` (Step 1.2) — параллельно

## Ключевые файлы

**`src/atm/core/types.py`**
- Роль: все domain frozen-типы проекта
- Запланированное изменение: добавить `TopologyDecision` и `PhaseDecision` в конец файла (после `BudgetEvent`)
- Статус: НЕ НАЧАТО

**`src/atm/core/__init__.py`**
- Роль: публичный re-export API модуля `atm.core`
- Запланированное изменение: добавить `TopologyDecision`, `PhaseDecision` в imports + `__all__`
- Статус: НЕ НАЧАТО

**`src/atm/core/state.py`**
- Роль: `SharedState` TypedDict для LangGraph
- Запланированное изменение: аудит (без изменений если всё OK; иначе минорная правка docstring)
- Статус: ТОЛЬКО АУДИТ

**`src/atm/core/reducers.py`**
- Роль: `dedup_by_id_reducer` фабрика + `merge_agent_states`
- Запланированное изменение: аудит (без функциональных изменений)
- Статус: ТОЛЬКО АУДИТ

**`src/atm/storage/models.py`**
- Роль: ORM-модель `TopologyTransition` (SQLAlchemy 2.x)
- Запланированное изменение: аудит (без изменений)
- Статус: ТОЛЬКО АУДИТ

**`alembic/versions/0001_initial_business_schema.py`**
- Роль: DDL для таблицы `topology_transitions` + индексы
- Запланированное изменение: аудит (без изменений)
- Статус: ТОЛЬКО АУДИТ

**`src/atm/phases/manager.py`**
- Роль: основная реализация Phase FSM — Protocol, PhaseGuard, PhaseLimits, RuleBasedPhaseRouter, LLMPhaseRouter
- Запланированное изменение: **новый файл** — создать с нуля
- Статус: НЕ НАЧАТО

**`src/atm/phases/__init__.py`**
- Роль: публичный экспорт пакета `atm.phases`
- Запланированное изменение: перезаписать M0 skeleton с полным `__all__` и docstring
- Статус: НЕ НАЧАТО

**`tests/unit/core/test_types.py`**
- Роль: юнит-тесты domain-типов
- Запланированное изменение: добавить ~6 тестов для `TopologyDecision` и `PhaseDecision`
- Статус: НЕ НАЧАТО

**`tests/unit/core/test_reducers.py`**
- Роль: юнит-тесты reducer'ов
- Запланированное изменение: добавить explicit invariant-тесты если не покрыты (ожидаемо no-op)
- Статус: ТОЛЬКО АУДИТ

**`tests/unit/core/test_state.py`**
- Роль: юнит-тесты SharedState
- Запланированное изменение: дополнить покрытие если нужно
- Статус: ТОЛЬКО АУДИТ

**`tests/unit/phases/__init__.py`**
- Роль: маркер пакета тестов
- Запланированное изменение: **новый файл** (пустой)
- Статус: НЕ НАЧАТО

**`tests/unit/phases/test_manager.py`**
- Роль: TDD-тесты для RuleBasedPhaseRouter и LLMPhaseRouter
- Запланированное изменение: **новый файл** — 13+ тестов
- Статус: НЕ НАЧАТО

**`dev/codebase-map.md`**
- Роль: карта кодовой базы проекта
- Запланированное изменение: обновить секцию Phase Manager (M0 → M8.1+M8.2 complete)
- Статус: НЕ НАЧАТО

## Решения

### topology_history: list[str] vs list[TopologyTransition]
- Решение: оставить `list[str]` в `SharedState` — соответствует arch.md §3.2 line 422
- Обоснование: полные `TopologyTransition` объекты уже хранятся в `GraphState["topology_transitions"]` через dedup-reducer. `SharedState` нужен только короткий хвост имён для cooldown-check (arch.md §8bis.3). Дублирование избыточно. Подтверждено ревьюером в review §6, §13.
- Последствие: update dec-файла под arch.md — tech-debt вне scope этого блока.

### PhaseGuard сигнатура: bool vs Phase | None
- Решение: `PhaseGuard = Callable[[GraphState], bool]` (§8.1 — source of truth)
- Обоснование: arch.md §8.2 line 1642 sketch (`Callable[[GraphState], Phase | None]`) считается устаревшим (obsolete). Bool-семантика: (1) разделение responsibility — guard проверяет, router решает; (2) удобство композиции multiple guards per phase; (3) §8.1 — более детальное описание.
- Статус: BLOCKING-1, зафиксировано ревьюером. Downstream блоки m8-routing и m8-adaptive должны использовать этот контракт.

### Тестирование LLMPhaseRouter: AsyncMock vs FakeLLM
- Решение: использовать `unittest.mock.AsyncMock` для всех 5 тестов LLMPhaseRouter
- Обоснование: `FakeLLM(mode='scripted')` требует YAML-фикстуру обязательно (`src/atm/llm/providers/fake.py` lines 99–105). Параметра `responses=[...]` не существует. AsyncMock проще, надёжнее для тестирования fallback-сценариев.
- Статус: BLOCKING-2, зафиксировано ревьюером.

### decided_by="rule" для fallback-вывода LLMPhaseRouter
- Решение: использовать Literal["rule"] — соответствует arch.md §8.2 Literal
- Обоснование: `decided_by` фиксирует, кто в итоге принял решение, не маршрут попадания. Differentiation между прямым Rule и fallback — через WARNING-логи. Literal "fallback" несовместим с зафиксированным типом.
- Статус: зафиксировано ревьюером в review §7.5.

### router_cost_usd валидация: @field_validator vs NonNegativeFloat
- Решение: `@field_validator("router_cost_usd")` с проверкой `>= 0`
- Обоснование: следуем convention существующего `TopologyTransition` в `types.py` (не использует `NonNegativeFloat`). Однородность кодовой базы.

### Конфликт имён Phase
- Решение: в `phases/manager.py` импортировать строго `from atm.core.types import Phase` (StrEnum); `storage/models.py` не импортировать
- Обоснование: в `storage/models.py` есть ORM-модель `Phase` (PhaseRow). Смешивание приведёт к mypy ошибкам и runtime путанице.

## Ограничения

- Python 3.11+, Pydantic v2, pytest-asyncio (asyncio_mode="auto"), uv
- Новых Alembic-миграций не добавлять — DDL уже в M3 (`0001_initial_business_schema.py`)
- LangGraph-совместимая семантика `SharedState` (TypedDict, `total=False`) сохраняется; M5–M7 графы не трогать
- Conventions: ruff + mypy --strict; все новые модули обязательно проходят mypy
- `decide()` у PhaseRouter — всегда `async`; guards — sync
- `LLMPhaseRouter.decide()` никогда не raise наружу — fallback totally robust
