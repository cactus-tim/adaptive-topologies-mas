# M1 — Core types & state — Tasks

## Phase 1: Подготовка — зависимости и каркас COMPLETE
- [x] 1.1 Добавить `pydantic>=2.7,<3` через `uv add 'pydantic>=2.7,<3'`; зафиксировать `uv.lock` — `pyproject.toml`, `uv.lock`
  - Acceptance: `uv run python -c "import pydantic; assert pydantic.VERSION.startswith('2.')"` — успех; `uv run pytest -q` M0 smoke-test проходит
- [x] 1.2 Создать `src/atm/core/__init__.py` (пустой) — `src/atm/core/__init__.py`
  - Acceptance: файл существует; `python -c "from atm.core import *"` не падает
- [x] 1.3 Создать `tests/unit/__init__.py` и `tests/unit/core/__init__.py` — `tests/unit/__init__.py`, `tests/unit/core/__init__.py`
  - Acceptance: оба файла существуют; `uv run pytest tests/unit/` не даёт collection errors

## Phase 2: Три независимых модуля (Wave 2 — параллельно) IN PROGRESS
- [x] 2.1 `src/atm/core/errors.py` — иерархия исключений (`AtmError`, `BudgetExceededError`, `PhaseError`, `ToolError`) + `tests/unit/core/test_errors.py` — `src/atm/core/errors.py`, `tests/unit/core/test_errors.py`
  - Acceptance: 5+ тестов зелёные; `uv run mypy src/atm/core/errors.py` — no issues; `uv run ruff check src/atm/core/errors.py` — clean
- [x] 2.2 `src/atm/core/types.py` — 17 Pydantic-классов + enums (по `arch.md §3.1`); `_utcnow()` вместо `utcnow`; `to_lc`/`from_lc` stubs + `tests/unit/core/test_types.py` — `src/atm/core/types.py`, `tests/unit/core/test_types.py`
  - Acceptance: 15+ тестов зелёные (включая CI-2 стаб-тесты и IR-3 JSON roundtrip); mypy strict чист; no DeprecationWarning
- [x] 2.3 `src/atm/core/reducers.py` — `merge_agent_states` + `dedup_by_id_reducer` фабрика; тесты только dict/namedtuple (без импорта types.py) + `tests/unit/core/test_reducers.py` — `src/atm/core/reducers.py`, `tests/unit/core/test_reducers.py`
  - Acceptance: 21+ тестов зелёные; `_get()` helper поддерживает dict и getattr; closure названа `_reduce`; mypy strict чист

## Phase 3: TypedDict-ы состояния COMPLETE
- [x] 3.1 `src/atm/core/state.py` — `AgentState`, `SharedState` (включая все L2-поля), `GraphState` с Annotated-reducer'ами; MC-4 mypy smoke-check + `tests/unit/core/test_state.py` — `src/atm/core/state.py`, `tests/unit/core/test_state.py`
  - Acceptance: 8+ тестов зелёные; MC-3 тест (имя closure `_reduce`) проходит; `typing.get_type_hints(GraphState, include_extras=True)` возвращает callable в метаданных; `uv run mypy src/atm/core/state.py` — no issues (или `# type: ignore[valid-type]` с комментарием если MC-4 потребует)

## Phase 4: Публичный API и финализация NOT STARTED
- [ ] 4.1 Наполнить `src/atm/core/__init__.py` реэкспортами и `__all__` (~25 имён из types/state/reducers/errors) — `src/atm/core/__init__.py`
  - Acceptance: `from atm.core import GraphState, Message, BudgetExceededError, merge_agent_states` — работает; `len(__all__) >= 25`
- [ ] 4.2 `tests/unit/core/test_public_api.py` — IR-4 (`__all__` non-None coverage), MC-5 (dedup с Pydantic Message), end-to-end reducer тест, agents merge тест — `tests/unit/core/test_public_api.py`
  - Acceptance: все тесты зелёные; MC-5 left-wins для Message с одинаковым id; IR-4 каждое имя в `__all__` not None
- [ ] 4.3 Обновить `dev/codebase-map.md` — заменить «M1 in progress» на «M1 complete» — `dev/codebase-map.md`
  - Acceptance: строка обновлена; git diff показывает только эту замену
- [ ] 4.4 Финальный полный прогон: `uv run pytest -q` + `uv run mypy src/atm` + `uv run ruff check src/ tests/`
  - Acceptance: pytest — все тесты passed, 0 warnings; mypy — no issues; ruff — clean

---
## Stats
- Total: 11 tasks · ~4.5h
- Done: 0 / 11

## How to Update
Check off tasks with `[x]` and update `m1-context.md` SESSION PROGRESS after each milestone.

Phase header format:
- All tasks done → `Phase N: [Name] COMPLETE`
- Some tasks done → `Phase N: [Name] IN PROGRESS`
- Nothing done → `Phase N: [Name] NOT STARTED`
