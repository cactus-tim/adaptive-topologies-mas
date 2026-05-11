# m8-foundation — План

## Краткое содержание

Заложить фундамент для адаптивной топологии L2 (Wave 1 = M8.1 + M8.2):

1. Добавить frozen-типы `TopologyDecision` и `PhaseDecision` в `core/types.py` (`TopologyTransition` уже существует с M3).
2. Провести аудит `SharedState`: подтвердить наличие всех необходимых полей с правильными типами и backward-compat через `total=False`.
3. Провести аудит `dedup_by_id_reducer`: фабрика и инварианты реализованы в M2; убедиться в покрытии property-style тестами явно.
4. Провести аудит Alembic-миграции и ORM-модели `TopologyTransition` (созданы в M3); проверить совместимость с актуальным Pydantic-типом.
5. Реализовать `phases/manager.py`: `RuleBasedPhaseRouter` (4 guards) + `LLMPhaseRouter` (JSON-парсинг, монотонность, fallback на Rule).
6. Написать юнит-тесты: 4 guards (positive/negative), rollback-attempt → fallback, malformed-JSON → fallback, монотонность.

## Текущее состояние

- `src/atm/core/types.py` — содержит `Phase` StrEnum, `PhaseTransition`, `TopologyTransition` (frozen, все 13 полей). `TopologyDecision` и `PhaseDecision` **отсутствуют** — нужно создать.
- `src/atm/core/state.py` — `SharedState: TypedDict, total=False` содержит все 7 нужных полей. `topology_history: list[str]` (не `list[TopologyTransition]` — решение принято в пользу arch.md §3.2, см. Решения).
- `src/atm/core/reducers.py` — `dedup_by_id_reducer` реализован с инвариантами. 14 тестов уже покрывают idempotent/associative/empty-neutral.
- `alembic/versions/0001_initial_business_schema.py` + `src/atm/storage/models.py` — миграция и ORM модель `TopologyTransition` полны и совместимы с Pydantic-типом. Новых миграций не требуется.
- `src/atm/phases/__init__.py` — пустой M0 skeleton.

## Предлагаемый подход

**Wave 1 (параллельно):** Steps 1, 2, 3 работают с непересекающимися файлами.

**Wave 2:** Step 4 создаёт `phases/manager.py` с `RuleBasedPhaseRouter` (зависит от `PhaseDecision` из Step 2).

**Wave 3:** Step 5 дополняет `phases/manager.py` классом `LLMPhaseRouter` (зависит от Step 4).

**Wave 4:** Step 6 — финальная санация экспортов, mypy, ruff, обновление codebase-map.

Ключевые архитектурные решения:
- `PhaseGuard = Callable[[GraphState], bool]` (bool-семантика из §8.1, sketch §8.2 line 1642 считается устаревшим).
- Тесты `LLMPhaseRouter` используют `unittest.mock.AsyncMock` (не FakeLLM, который требует YAML-фикстур).
- `decided_by="rule"` для fallback-вывода LLMPhaseRouter (per arch.md §8.2 Literal).

## Фазы реализации

### Фаза 1: Аудит и типы (Wave 1, параллельно) (~2.5h)
**Цель:** Зафиксировать baseline существующих артефактов M1–M3, создать два новых frozen-типа, расширить тесты reducer при необходимости.

- [ ] 1.1 Аудит SharedState, TopologyTransition (Pydantic + ORM + Alembic), dedup_by_id_reducer
  - Файл: `src/atm/core/state.py`, `src/atm/core/types.py`, `src/atm/core/reducers.py`, `src/atm/storage/models.py`, `alembic/versions/0001_initial_business_schema.py`, `tests/unit/core/test_state.py`, `tests/unit/core/test_reducers.py`, `tests/unit/core/test_types.py`
  - Приёмочный критерий: письменный baseline чек-лист; `uv run pytest tests/unit/core -q` — зелёный

- [ ] 1.2 Добавить `TopologyDecision` и `PhaseDecision` в `core/types.py` + re-export в `core/__init__.py`
  - Файл: `src/atm/core/types.py`, `src/atm/core/__init__.py`
  - Приёмочный критерий: `from atm.core.types import TopologyDecision, PhaseDecision` работает; mypy --strict зелёный

- [ ] 1.3 Написать ~6 тестов для новых типов в `test_types.py`
  - Файл: `tests/unit/core/test_types.py`
  - Приёмочный критерий: frozen-immutability, defaults, Literal-validation, `router_cost_usd >= 0`; `uv run pytest tests/unit/core/test_types.py -q` зелёный

- [ ] 1.4 Аудит и при необходимости дополнение тестов reducer (explicit invariants)
  - Файл: `tests/unit/core/test_reducers.py`
  - Приёмочный критерий: 3 инварианта явно читаемы в `pytest -v`; ожидаемо no-op если уже покрыты

### Фаза 2: RuleBasedPhaseRouter (Wave 2) (~2h)
**Цель:** Создать новый модуль `phases/manager.py` с Protocol, PhaseLimits, 4 guards и детерминированным маршрутизатором.

- [ ] 2.1 TDD: написать тесты TestRuleBased (8+ кейсов) до реализации
  - Файл: `tests/unit/phases/__init__.py` (новый пустой), `tests/unit/phases/test_manager.py` (новый)
  - Приёмочный критерий: тесты существуют и падают до реализации (или файл создан с pending-маркером)

- [ ] 2.2 Реализовать `PhaseRouter` Protocol, `PhaseGuard`, `_phase_order`, `PhaseLimits`, `RuleBasedPhaseRouter` в `phases/manager.py`
  - Файл: `src/atm/phases/manager.py` (новый)
  - Приёмочный критерий: `uv run pytest tests/unit/phases/test_manager.py::TestRuleBased -v` зелёный; mypy --strict зелёный

- [ ] 2.3 Обновить `phases/__init__.py` — добавить предварительный экспорт
  - Файл: `src/atm/phases/__init__.py`
  - Приёмочный критерий: `from atm.phases import RuleBasedPhaseRouter, PhaseLimits, PhaseGuard, PhaseRouter` работает

### Фаза 3: LLMPhaseRouter (Wave 3) (~1.5h)
**Цель:** Дополнить `phases/manager.py` классом с JSON-парсингом, валидацией монотонности и fallback-семантикой.

- [ ] 3.1 TDD: написать 5 тестов TestLLMRouter (AsyncMock) до реализации
  - Файл: `tests/unit/phases/test_manager.py`
  - Приёмочный критерий: тесты покрывают happy-path, rollback-attempt, malformed-JSON, unknown-phase, missing-field

- [ ] 3.2 Реализовать `LLMPhaseRouter` в `phases/manager.py`
  - Файл: `src/atm/phases/manager.py`
  - Приёмочный критерий: все 5 тестов зелёные; fallback не raise; WARNING логируется; mypy --strict зелёный

### Фаза 4: Финализация (Wave 4) (~0.5h)
**Цель:** Убедиться в целостности публичного API, чистоте линтеров, отсутствии регрессий M5–M7.

- [ ] 4.1 Финализировать `phases/__init__.py`: docstring + `__all__`
  - Файл: `src/atm/phases/__init__.py`
  - Приёмочный критерий: все публичные символы перечислены в `__all__`

- [ ] 4.2 Полный suite + линтеры + обновление codebase-map
  - Файл: `dev/codebase-map.md` (секция Phase Manager)
  - Приёмочный критерий: `uv run pytest tests/unit -q` 0 failures; `uv run mypy src/atm --strict`; `uv run ruff check src/atm tests` — чисто

## Ключевые файлы

| Файл | Изменение | Причина |
|------|-----------|---------|
| `src/atm/core/types.py` | Добавить `TopologyDecision`, `PhaseDecision` | Новые frozen-типы для Phase FSM |
| `src/atm/core/__init__.py` | Re-export новых типов | Публичный API `atm.core` |
| `src/atm/core/state.py` | Аудит (без изменений если всё OK) | Подтвердить backward-compat |
| `src/atm/core/reducers.py` | Аудит (без функциональных изменений) | Подтвердить покрытие инвариантов |
| `src/atm/storage/models.py` | Аудит (без изменений) | Проверить совпадение схемы с Pydantic |
| `alembic/versions/0001_initial_business_schema.py` | Аудит (без изменений) | Проверить полноту DDL |
| `src/atm/phases/manager.py` | **Новый файл** — основная реализация | RuleBasedPhaseRouter + LLMPhaseRouter |
| `src/atm/phases/__init__.py` | Полная перезапись из M0 skeleton | Экспорт публичного API phases |
| `tests/unit/core/test_types.py` | Добавить ~6 тестов | Покрытие новых типов |
| `tests/unit/core/test_reducers.py` | Дополнить при необходимости | Explicit invariant-тесты |
| `tests/unit/core/test_state.py` | Дополнить при необходимости | Покрытие полей SharedState |
| `tests/unit/phases/__init__.py` | **Новый файл** (пустой) | Создание test-пакета |
| `tests/unit/phases/test_manager.py` | **Новый файл** | Тесты RuleBased + LLMRouter |
| `dev/codebase-map.md` | Обновить секцию Phase Manager | Актуализация статуса |

## Зависимости и порядок выполнения

- Steps 1.1, 1.2/1.3, 1.4 — полностью параллельны (disjoint файлы)
- Step 2.x — зависит от завершения 1.2 (нужен `PhaseDecision`)
- Step 3.x — зависит от завершения 2.x (нужен `RuleBasedPhaseRouter` + `_phase_order`)
- Step 4.x — зависит от завершения 3.x

## Риски

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Найдено расхождение между ORM-моделью и Pydantic-типом (помимо Decimal→float) | Низкая | Высокое | Таблица сверки колонок в Step 1.1; при расхождении — replan |
| Конфликт имён `Phase` (StrEnum vs ORM PhaseRow) | Средняя | Среднее | Импортировать строго `from atm.core.types import Phase`; storage не импортировать в manager.py |
| Rollback через guards (`next_phase < current_phase`) | Средняя | Среднее | `assert _phase_order(decision.next_phase) >= _phase_order(current_phase)` в конце `decide()` |
| AsyncMock не валидирует сигнатуру `ainvoke` | Низкая | Низкое | Канонический вызов в production и тестах одинаков |
| Поломка legacy тестов M5–M7 правкой `core/__init__.py` | Низкая | Высокое | Полный `uv run pytest tests/unit -q` в финале |

## Вне scope

- Изменение `state.py` по `topology_history` (tech-debt, отдельный блок)
- Обновление dec-файла под arch.md §3.2 (tech-debt, отдельный блок)
- Новые Alembic-миграции (все нужные DDL уже в M3)
- Интеграционные тесты (нет нового HTTP/gRPC endpoint; acceptance-тесты — задача m8-adaptive)
- Рефакторинг `PhaseLimits` в `phases/config.py` (может понадобиться в m8-routing)

## Временные оценки

- Итого: ~6.5h
- Создан: 2026-05-11
