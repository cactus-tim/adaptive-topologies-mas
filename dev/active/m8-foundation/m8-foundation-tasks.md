# m8-foundation — Задачи

## Фаза 1: Аудит и типы (Wave 1, параллельно) — НЕ НАЧАТО

- [ ] 1.1 Аудит SharedState, TopologyTransition (Pydantic + ORM + Alembic), dedup_by_id_reducer — `src/atm/core/state.py`, `src/atm/core/types.py`, `src/atm/core/reducers.py`, `src/atm/storage/models.py`, `alembic/versions/0001_initial_business_schema.py`, `tests/unit/core/`
  - Приёмочный критерий: письменный baseline чек-лист (13 колонок сверены), `uv run pytest tests/unit/core -q` зелёный

- [ ] 1.2 Добавить `TopologyDecision` и `PhaseDecision` в `core/types.py` (frozen Pydantic v2, валидация `router_cost_usd >= 0` через `@field_validator`) — `src/atm/core/types.py`
  - Приёмочный критерий: классы созданы, frozen=True, поля соответствуют arch.md §8.2 и §8bis.1

- [ ] 1.3 Re-export новых типов в `core/__init__.py` — `src/atm/core/__init__.py`
  - Приёмочный критерий: `from atm.core import TopologyDecision, PhaseDecision` работает; `__all__` дополнен; `uv run mypy src/atm/core/types.py --strict` зелёный

- [ ] 1.4 Написать ~6 тестов для новых типов — `tests/unit/core/test_types.py`
  - Приёмочный критерий: frozen-immutability, defaults, Literal-validation, `router_cost_usd >= 0`; `uv run pytest tests/unit/core/test_types.py -q` зелёный

- [ ] 1.5 Аудит и дополнение тестов reducer (explicit invariants) — `tests/unit/core/test_reducers.py`
  - Приёмочный критерий: 3 инварианта явно читаемы в `pytest -v`; ожидаемо no-op (comment-pointer) если уже покрыты; `uv run pytest tests/unit/core/test_reducers.py -v` зелёный

## Фаза 2: RuleBasedPhaseRouter (Wave 2) — НЕ НАЧАТО

- [ ] 2.1 Создать `tests/unit/phases/__init__.py` (пустой) и написать тесты TestRuleBased (TDD) — `tests/unit/phases/__init__.py`, `tests/unit/phases/test_manager.py`
  - Приёмочный критерий: 8+ тест-функций созданы; 4 guards × positive/negative + edge (terminal done, stay planning); тесты падают до реализации

- [ ] 2.2 Реализовать `PhaseRouter` Protocol, `PhaseGuard` type alias, `_phase_order` helper, `PhaseLimits`, `RuleBasedPhaseRouter` — `src/atm/phases/manager.py`
  - Приёмочный критерий: `uv run pytest tests/unit/phases/test_manager.py::TestRuleBased -v` зелёный; `uv run mypy src/atm/phases/manager.py --strict` зелёный

- [ ] 2.3 Добавить предварительный экспорт в `phases/__init__.py` — `src/atm/phases/__init__.py`
  - Приёмочный критерий: `from atm.phases import RuleBasedPhaseRouter, PhaseLimits, PhaseGuard, PhaseRouter` работает

## Фаза 3: LLMPhaseRouter (Wave 3) — НЕ НАЧАТО

- [ ] 3.1 Написать 5 тестов TestLLMRouter (AsyncMock) — `tests/unit/phases/test_manager.py`
  - Приёмочный критерий: покрыты happy-path, rollback-attempt→fallback, malformed-JSON→fallback, unknown-phase→fallback, missing-field→fallback; используется `unittest.mock.AsyncMock`

- [ ] 3.2 Реализовать `LLMPhaseRouter` в `phases/manager.py` — `src/atm/phases/manager.py`
  - Приёмочный критерий: все 5 тестов зелёные; fallback не raise; WARNING логируется через `logging.getLogger(__name__)`; `uv run mypy src/atm/phases/manager.py --strict` зелёный; `uv run ruff check src/atm/phases/manager.py` чисто

## Фаза 4: Финализация (Wave 4) — НЕ НАЧАТО

- [ ] 4.1 Финализировать `phases/__init__.py`: docstring + `__all__` — `src/atm/phases/__init__.py`
  - Приёмочный критерий: `__all__ = ["RuleBasedPhaseRouter", "LLMPhaseRouter", "PhaseLimits", "PhaseGuard", "PhaseRouter"]`; модуль корректно импортируется

- [ ] 4.2 Обновить секцию Phase Manager в `dev/codebase-map.md` — `dev/codebase-map.md`
  - Приёмочный критерий: статус изменён с "M0 skeleton, M8 not started" на "M8.1 + M8.2 complete"; перечислены exports

- [ ] 4.3 Прогнать полный suite и линтеры — `tests/unit/`
  - Приёмочный критерий: `uv run pytest tests/unit -q` 0 failures; `uv run mypy src/atm --strict`; `uv run ruff check src/atm tests` — чисто

---

## Статистика

- Всего: 13 задач · ~6.5h
- Выполнено: 0 / 13

## Как обновлять

После выполнения каждой задачи отмечать `[x]` вместо `[ ]`.
После завершения фазы обновить заголовок: `НЕ НАЧАТО` → `В ПРОЦЕССЕ` → `ВЫПОЛНЕНО`.
Обновлять `SESSION PROGRESS` в `m8-foundation-context.md` после каждого milestone.
