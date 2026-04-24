# M1 — Core types & state — Context

## SESSION PROGRESS (2026-04-23)

### COMPLETED
- Step 0 (Phase 1): добавлен pydantic==2.13.3 через `uv add 'pydantic>=2.7,<3'`; созданы `src/atm/core/__init__.py` (пустой) и `tests/unit/core/__init__.py`; все три acceptance criteria подтверждены; M0 smoke tests 2 passed
- Step 2.1 (Phase 2): реализован `src/atm/core/errors.py` — иерархия AtmError/BudgetExceededError/PhaseError/ToolError; создан `tests/unit/core/test_errors.py` (20 тестов); mypy strict — no issues; ruff — clean
- Step 2.2 (Phase 2): реализован `src/atm/core/types.py` — 17 классов/enums: 4 StrEnum (AgentRole/HumanRole/Phase/MessageKind) + 13 frozen Pydantic v2 models; `_utcnow()` с `datetime.UTC`; `to_lc`/`from_lc` stubs с NotImplementedError("...M2..."); `TYPE_CHECKING` guard для BaseMessage; `tests/unit/core/test_types.py` — 31 тест, все зелёные; mypy strict — no issues; ruff — clean; 0 DeprecationWarning
- Step 2.3 (Phase 2): создан `src/atm/core/reducers.py` с `merge_agent_states` и `dedup_by_id_reducer`; написан `tests/unit/core/test_reducers.py` (24 теста); все проходят; mypy strict — no issues; ruff — clean

- Step 3.1 (Phase 3): реализован `src/atm/core/state.py` — три TypedDict: AgentState/SharedState/GraphState; GraphState использует Annotated с reducer'ами (merge_agent_states и dedup_by_id_reducer); написан `tests/unit/core/test_state.py` (24 теста, все зелёные); mypy strict — no issues; ruff — clean; MC-3 тест closure-имён проходит

- Code-review fixes (2026-04-23): все 6 findings из .code-review.md устранены:
  - Major #1: `SharedState.topology` → `active_topology` (state.py:57); тест test_state.py:74 ужесточён до строгой проверки "active_topology"
  - Major #2: добавлен `sort_by` к трём dedup-reducer-ам (messages→created_at, budget_events→at, topology_transitions→at); llm_calls оставлен без sort_by (CI-1 path a); добавлены два теста в test_public_api.py
  - Minor #3: dev/codebase-map.md — "M1 in progress" → "M1 complete"; раздел Exports расширен до 26 имён
  - Minor #4: `Message.from_lc` — убраны дефолты sender="" и kind=MessageKind.REQUEST; тест обновлён: вызов с явными kwargs + matcher "M2"
  - Minor #5: test_types.py:115,139 — `pytest.raises((TypeError, Exception))` → `pytest.raises(pydantic.ValidationError)`
  - Minor #6: `reducers._get` — тип возврата `Any` → `Hashable`; cast() для явного контракта; sort key покрыт type: ignore[arg-type, return-value] с обоснованием

### IN PROGRESS
- M1 complete — все шаги выполнены, все code-review findings закрыты

### BLOCKERS
- None

## Quick Resume
1. Read this file
2. Check `m1-tasks.md` for what's next
3. Read `m1-plan.md` Phase 1 for strategy
4. Start with: Phase 1 — `uv add 'pydantic>=2.7,<3'` и создание `src/atm/core/__init__.py` + `tests/unit/core/__init__.py`

## Key Files

**`pyproject.toml`**
- Role: Манифест проекта; определяет runtime-зависимости
- Planned change: Добавить `pydantic>=2.7,<3` в `[project].dependencies`
- Status: NOT STARTED

**`src/atm/core/__init__.py`**
- Role: Публичный API модуля `atm.core`; реэкспортирует ~25 имён через `__all__`
- Planned change: Создать пустым на Phase 1; наполнить на Phase 4
- Status: NOT STARTED

**`src/atm/core/errors.py`**
- Role: Иерархия исключений фреймворка: `AtmError`, `BudgetExceededError`, `PhaseError`, `ToolError`
- Planned change: Создать (новый файл)
- Status: DONE

**`src/atm/core/types.py`**
- Role: 17 Pydantic v2 классов и enums по `arch.md §3.1`; все frozen
- Planned change: Создать (новый файл); `utcnow` заменить на `datetime.now(timezone.utc)`
- Status: DONE (31 тестов, все зелёные)

**`src/atm/core/reducers.py`**
- Role: Два LangGraph-совместимых reducer'а: `merge_agent_states` и фабрика `dedup_by_id_reducer`
- Planned change: Создать (новый файл); forward-ref на AgentState через TYPE_CHECKING
- Status: COMPLETE

**`src/atm/core/state.py`**
- Role: Три TypedDict-а: `AgentState`, `SharedState`, `GraphState`; GraphState использует Annotated с reducer'ами
- Planned change: Создать (новый файл); импортирует из types.py и reducers.py
- Status: DONE (24 тестов, все зелёные)

**`tests/unit/core/test_errors.py`**
- Role: Unit-тесты иерархии исключений (5+ тестов)
- Planned change: Создать
- Status: DONE (20 тестов, все зелёные)

**`tests/unit/core/test_types.py`**
- Role: Unit-тесты Pydantic-моделей (15+ тестов, включая CI-2 стаб-тесты и IR-3 JSON roundtrip)
- Planned change: Создать
- Status: DONE (31 тестов, все зелёные)

**`tests/unit/core/test_reducers.py`**
- Role: Unit-тесты reducer-инвариантов (21+ тестов); только dict/namedtuple, без импорта types.py
- Planned change: Создать
- Status: COMPLETE (24 тестов)

**`tests/unit/core/test_state.py`**
- Role: Unit-тесты TypedDict-ов (8+ тестов, включая MC-3 тест имени closure)
- Planned change: Создать
- Status: DONE (24 тестов, все зелёные)

**`tests/unit/core/test_public_api.py`**
- Role: IR-4 (__all__ non-None coverage), MC-5 (dedup с Pydantic Message), end-to-end reducer тест
- Planned change: Создать
- Status: NOT STARTED

**`dev/codebase-map.md`**
- Role: Карта кодовой базы; содержит статус каждого milestone
- Planned change: Заменить «M1 in progress» на «M1 complete» в финале
- Status: NOT STARTED

## Decisions

**TypedDict для state-контейнеров (не Pydantic BaseModel)**
- Decision: `AgentState`, `SharedState`, `GraphState` — stdlib `typing.TypedDict`; Pydantic-модели только для данных внутри state
- Rationale: Избегает overhead валидации на каждый тик LangGraph; совместимость с `langchain.create_agent`; закреплено в `arch.md §17 #1`

**`datetime.now(timezone.utc)` вместо `datetime.utcnow()`**
- Decision: Везде в `default_factory` использовать `lambda`/`_utcnow()` с `timezone.utc`
- Rationale: `utcnow()` deprecated в Python 3.12+; pytest-конфиг `filterwarnings=["error"]` превращает DeprecationWarning в ошибку

**`dedup_by_id_reducer` — left-wins при коллизии id**
- Decision: При пересечении id побеждает элемент из `left` (старое состояние)
- Rationale: Гарантирует идемпотентность `reducer(x, x) == x`; закреплено в `arch.md §3.3bis`

**`Message.to_lc`/`from_lc` — stub'ы с NotImplementedError**
- Decision: Сигнатуры присутствуют, тело бросает `NotImplementedError` с подстрокой "M2"
- Rationale: Фиксирует контракт из `arch.md §3.1` в одном месте; M1 не тянет `langchain-core`; CI-2 тест защищает от регрессии

**`LLMResponse` без поля `started_at` на M1 (CI-1, path a)**
- Decision: Не добавлять `started_at`; `GraphState.llm_calls` использует `dedup_by_id_reducer("id")` без `sort_by`
- Rationale: Сохраняет 1:1 соответствие с `arch.md §3.2`; `started_at` добавится в M2 как явная эволюция контракта

**`reducers.py` — отдельный файл, не часть `state.py`**
- Decision: Чистые функции в отдельном модуле; `state.py` импортирует из него
- Rationale: Устраняет circular-risk при расширениях; соответствует явному указанию `dev/PLAN.md §8`

**`ToolError.cause` — Python exception chaining, не конструктор-аргумент**
- Decision: `.cause` — read-only свойство над `self.__cause__`; устанавливается через `raise ToolError(...) from exc`
- Rationale: Идиоматичный Python-паттерн; traceback автоматически содержит цепочку

## Constraints

- Pydantic v2 только; совместимость с v1 не нужна
- Python 3.11 (`requires-python = ">=3.11"`); stdlib `typing.TypedDict` достаточен
- `filterwarnings=["error"]` в pytest — любой DeprecationWarning = ошибка
- ruff strict (`select = ["E","W","F","I","UP","B","SIM","C4","N","RUF"]`) + mypy strict на `src/atm`
- `from __future__ import annotations` обязательно во всех трёх src-файлах
- Никаких `# type: ignore` без комментария-обоснования
- LangGraph не устанавливается на M1; только структурная совместимость
- `typing-extensions` не добавлять превентивно (IR-5: только при реальной ошибке mypy)

## Amendments Applied
- Нет (USER_AMENDMENTS: none)
