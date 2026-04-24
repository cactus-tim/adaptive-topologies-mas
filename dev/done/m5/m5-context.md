# M5 — Agent framework + scratchpad policy C — Context

## Task Summary

Реализовать базовый класс `Agent` (LangGraph-совместимая нода), политику скратчпада C (append-only journal + windowed view + optional summarization), tool-calling loop, 5 ролевых подклассов (`Planner`, `Researcher`, `Executor`, `Critic`, `Debater`) и их YAML-конфиги. LangGraph-топология появится в M6; M5 только создаёт абстракцию.

**Final Class**: `complex`
**Needs Integration Tests**: нет

---

## SESSION PROGRESS (2026-04-24)

### COMPLETED
- Step 1 (Wave 1): AgentConfig pydantic v2 frozen model + load_agent_config YAML loader; 20 tests passing; YAML fixtures minimal_valid.yaml + debater_with_stance.yaml created
- Step 2 (Wave 2): _tokens.py estimate_prompt_tokens helper; tiktoken for openai:*, len//4 heuristic fallback; lru_cache(maxsize=8) on encoder; 11 tests passing
- Step 4-code (Wave 2): 5 role subclasses (Planner, Researcher, Executor, Critic, Debater) + 5 YAML configs in conf/agents/; all configs load via load_agent_config; Debater stance substitution + ValueError guard implemented; syntax checks pass; 20 config tests still green
- Step 3 (Wave 3): Agent base class with scratchpad policy C + tool-loop + summarizer integration; _StepOutcome/AgentView frozen dataclasses; C1 accumulation fix; __init__.py exports; 5 fixture YAML files; 18 new tests (49 total agent tests passing); 90 LLM tests passing; ruff clean
- Step 4-tests (Wave 4): test_debater_stance.py (8 tests) + test_role_configs_load.py (33 tests); full suite 90 tests passing
- Step 5 (Wave 4): m5_executor_exit.yaml fixture (2 entries: tool_calls + stop) + test_executor_exit_criterion.py with inline FakeCodeRunTool; proves end-to-end tool-loop: scratchpad 4 events, 1 code_run call (C1), tool_result ok=True stdout="2\n", DRAFT message mentions "2"; 91 total agent tests passing

### IN PROGRESS
- All waves complete; M5 fully done

### BLOCKERS
- Нет

---

## Quick Resume

1. Прочитать этот файл.
2. Открыть `m5-tasks.md` — найти первую незаполненную задачу.
3. Прочитать `m5-plan.md` Phase 1 для стратегии.
4. Начать с: **Step 1 — создать `src/atm/agents/__init__.py` (пустой) + `src/atm/agents/config.py` + `AgentConfig` pydantic-модель**.

---

## Key Files

### Новые файлы (создаются в M5)

**`src/atm/agents/__init__.py`**
- Role: корень пакета; re-export всех публичных символов
- Planned change: создаётся пустым в Step 1; **заполняется только в Step 3** (после merge Step 4 code)
- Status: DONE (empty) — Step 1 complete
- OWNER RULE: Steps 2 и 4 MUST NOT трогать этот файл

**`src/atm/agents/config.py`**
- Role: `AgentConfig` (pydantic v2 `frozen=True`) + `load_agent_config(path) -> AgentConfig`
- Planned change: NEW — Step 1
- Status: DONE — Step 1 complete

**`src/atm/agents/_tokens.py`**
- Role: `estimate_prompt_tokens(messages, model_id) -> int`; tiktoken для OpenAI, heuristic fallback
- Planned change: NEW — Step 2
- Status: DONE — Step 2 complete

**`src/atm/agents/base.py`**
- Role: `Agent`, `_StepOutcome`, `AgentView`; все ключевые алгоритмы (step, tool-loop, window, summarizer)
- Planned change: NEW — Step 3
- Status: DONE — Step 3 complete

**`src/atm/agents/planner.py`**
- Role: `class Planner(Agent): pass`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`src/atm/agents/researcher.py`**
- Role: `class Researcher(Agent): pass`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`src/atm/agents/executor.py`**
- Role: `class Executor(Agent): pass`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`src/atm/agents/critic.py`**
- Role: `class Critic(Agent): pass`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`src/atm/agents/debater.py`**
- Role: `Debater.__init__` — подстановка `{{stance}}` через `cfg.model_copy(update=...)`; `ValueError` при неверном stance
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`conf/agents/planner.yaml`**
- Role: конфиг Planner: `window_size=15`, `tools: [todo_write, plan_update]`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`conf/agents/researcher.yaml`**
- Role: конфиг Researcher: `window_size=12`, `tools: [semantic_search, duckduckgo_search, url_fetch, file_read]`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`conf/agents/executor.yaml`**
- Role: конфиг Executor: `window_size=12`, `tools: [code_run, file_write, file_read, calculator]`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`conf/agents/critic.yaml`**
- Role: конфиг Critic: `window_size=15`, `tools: [test_run, diff, lint]`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

**`conf/agents/debater.yaml`**
- Role: конфиг Debater: `window_size=12`, `params.stance: pro`
- Planned change: NEW — Step 4 (code)
- Status: NOT STARTED

### Существующие файлы (модифицируются)

**`src/atm/llm/wrapper.py`**
- Role: `LLMWrapper.ainvoke`; основной LLM-клиент проекта
- Planned change: +1 line `@property model_id` — Step 3
- Status: DONE — Step 3 complete (model_id property already present; no regressions in 90 LLM tests)
- RISK: регрессия существующих LLM-тестов; верификация `uv run pytest tests/unit/llm/ -v`

### Тестовые файлы

**`tests/unit/agents/test_config.py`** — Step 1, >= 7 тестов
**`tests/unit/agents/test_tokens.py`** — Step 2, 5 тестов
**`tests/unit/agents/test_agent_basic.py`** — Step 3
**`tests/unit/agents/test_agent_tool_loop.py`** — Step 3 (C1 assertion: `len(tool_calls)==2`)
**`tests/unit/agents/test_agent_scratchpad_window.py`** — Step 3
**`tests/unit/agents/test_agent_summarizer.py`** — Step 3 (summarizer wrapped in LLMWrapper)
**`tests/unit/agents/test_agent_budget_propagation.py`** — Step 3
**`tests/unit/agents/test_debater_stance.py`** — Step 4 (tests)
**`tests/unit/agents/test_role_configs_load.py`** — Step 4 (tests)
**`tests/unit/agents/test_executor_exit_criterion.py`** — Step 5

### Фикстуры

**`tests/fixtures/agents/minimal_valid.yaml`** — Step 1
**`tests/fixtures/agents/debater_with_stance.yaml`** — Step 1
**`tests/fixtures/llm/m5_agent_basic.yaml`** — Step 3
**`tests/fixtures/llm/m5_agent_tool_loop.yaml`** — Step 3
**`tests/fixtures/llm/m5_agent_scratchpad.yaml`** — Step 3
**`tests/fixtures/llm/m5_agent_summarizer_primary.yaml`** — Step 3
**`tests/fixtures/llm/m5_agent_summarizer_secondary.yaml`** — Step 3 (`agent_id: p1_summarizer`)
**`tests/fixtures/llm/m5_executor_exit.yaml`** — Step 5

---

## Decisions

### 1. ToolRegistry-not-list
- Decision: `Agent` принимает `ToolRegistry`, не `list[Tool]`
- Rationale: согласовано с M4 интерфейсом; `ToolRegistry.ainvoke_by_name` инкапсулирует dispatch и error-handling

### 2. `_StepOutcome` dataclass (C2 fix)
- Decision: `@dataclass(frozen=True)` с полями `response`, `scratchpad_events`, `tool_calls`, `tool_results`
- Rationale: заменяет ad-hoc tuple; типизировано, immutable, call-sites синхронизированы

### 3. tiktoken + heuristic fallback
- Decision: tiktoken для провайдера `openai:*`, `len(text)//4 + 4` для остальных; `lru_cache(maxsize=8)` на encoder
- Rationale: tiktoken работает без сети (bundled BPE); heuristic достаточна для conservative budget trigger (~12k); погрешность ~20-30% приемлема; NOT для биллинга

### 4. events-based `window_size` (M3 fix)
- Decision: `window_size` = количество событий скратчпада, не логических шагов
- Rationale: 1 логический шаг = до 3 событий (reasoning + tool_call + observation); role configs widened: planner=15, critic=15, остальные=12
- Docstring в `_build_prompt` явно документирует семантику

### 5. structlog для неизвестных tools (M2 fix)
- Decision: `structlog.get_logger(__name__).warning(...)` в `_build_tools_schema()` вместо `warnings.warn`
- Rationale: `filterwarnings = ["error"]` в pyproject.toml превращает Python-warnings в исключения под pytest; structlog уже в deps и роутится в observability infra

### 6. summarizer = второй LLMWrapper (M5 fix)
- Decision: `summarizer_llm: LLMWrapper | None` — полноценный обёрнутый LLMWrapper
- Rationale: даёт единый интерфейс для budget/retry; `agent_id = f"{primary_agent_id}_summarizer"` — хардкод в `_maybe_summarize`; тесты оборачивают FakeLLM в LLMWrapper с loose BudgetTracker

### 7. Debater stance при `__init__` (M7 draft decision)
- Decision: `{{stance}}` подставляется один раз в `Debater.__init__` через `cfg.model_copy(update={"system_prompt": resolved})`
- Rationale: AgentConfig frozen — нельзя мутировать; `model_copy` создаёт новый объект; prompt неизменен после инициализации

### 8. `__init__.py` ownership (M7 fix)
- Decision: Step 1 создаёт пустым; **только Step 3** добавляет экспорты
- Rationale: предотвращает merge-конфликты при параллельном выполнении Steps 2 и 4; Steps 2/4 используют fully-qualified imports

### 9. Delta-only return из `Agent.step()`
- Decision: возвращать только новые элементы списков (scratchpad_events за текущий step) + абсолютные значения счётчиков
- Rationale: reducer `merge_agent_states` делает list-concat → при возврате полного списка будут дубли; для счётчиков reducer делает `max()` → нужны абсолютные значения

### 10. Summarizer вызывается один раз за `step()` (M1 fix)
- Decision: `_maybe_summarize` вызывается до tool-loop; intra-loop truncation до 2000 chars; TODO-комментарий для M6/M7
- Rationale: re-entrant summarization внутри loop — сложно; TODO фиксирует долг; truncation как mitigation

---

## Constraints

- `pytest-asyncio` в режиме `asyncio_mode = "auto"` (async-тесты без декоратора).
- `filterwarnings = ["error"]` — Python `warnings.warn` = исключение под pytest; везде structlog.
- `AgentConfig` frozen — изменения только через `model_copy(update=...)`.
- `Agent.step()` возвращает delta, не полный state — иначе дубли через reducer.
- `src/atm/agents/__init__.py` — трогает только Step 3.
- `src/atm/llm/wrapper.py` — трогает только Step 3 (одна строка).
- FakeLLM scripted fixture: `agent_id` в fixture ДОЛЖЕН совпадать с `agent_id` переданным в `Agent()` constructor (primary) или его `_summarizer`-вариантом.
- Реальные LLM-вызовы и docker в тестах не используются.
