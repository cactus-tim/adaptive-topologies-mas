# M5 — Agent framework + scratchpad policy C — Plan

## Executive Summary

Реализовать milestone M5: базовый класс `Agent`, LangGraph-совместимую ноду с политикой скратчпада C (append-only journal + windowed view + optional summarization), tool-calling-loop, 5 ролевых подклассов (`Planner`, `Researcher`, `Executor`, `Critic`, `Debater`) и их YAML-конфиги. LangGraph-топология появится только в M6; здесь создаётся только абстракция, на которую M6/M7 и Adaptive (§7.7) опираются.

**Final Class**: `complex`
**Rationale**: ~10 новых файлов (base.py + config.py + 5 ролевых модулей + 5 YAML + >=5 тестов), несколько взаимозависимых алгоритмов (tool-loop, token-budget trigger, окно скратчпада, summarizer), задаёт абстракцию для M6/M7.

**Needs Integration Tests**: нет — `Agent` — чисто внутренняя абстракция, внешних HTTP/gRPC/БД-контрактов не создаёт.

---

## Current State

Модуль `src/atm/agents/` отсутствует. Зависимые системы:
- `src/atm/core/types.py` — `Message`, `ToolCall`, `ToolResult`, `LLMResponse`, `AgentRole` (frozen dataclasses).
- `src/atm/core/state.py` — `AgentState` (TypedDict), `GraphState`, reducer `merge_agent_states` (list-concat для списков, `max()` для счётчиков).
- `src/atm/core/errors.py` — `BudgetExceededError`.
- `src/atm/llm/wrapper.py` — `LLMWrapper.ainvoke`; НЕ имеет публичного `model_id` property.
- `src/atm/llm/fake.py` — `FakeLLM(mode="scripted", ...)` индексирует fixture по `agent_id` + `step_idx`.
- `src/atm/tools/base.py` — `ToolRegistry.ainvoke_by_name`; `.get(name)` бросает `ToolError` при отсутствии.
- `src/atm/tools/defaults.py` — `build_default_registry(...)` — 12 tools.

---

## Proposed Approach

### Ключевые дизайн-решения

1. **ToolRegistry-not-list**: `Agent` принимает `ToolRegistry` (не `list[Tool]`), согласовано с M4.
2. **`_StepOutcome` dataclass**: `frozen=True` dataclass вместо ad-hoc tuple; поля: `response`, `scratchpad_events`, `tool_calls`, `tool_results`.
3. **tiktoken + heuristic fallback**: `_tokens.py` — tiktoken для OpenAI-family, `len(text)//4` для остальных; +4 overhead per message (OpenAI cookbook convention).
4. **events-based `window_size`**: `window_size` — количество событий скратчпада (не логических шагов); 1 логический шаг = до 3 событий (`reasoning`, `tool_call`, `observation`).
5. **structlog для неизвестных tools**: в `_build_tools_schema()` — `structlog.get_logger(__name__).warning(...)` вместо `warnings.warn` (последнее становится исключением под `filterwarnings = ["error"]`).
6. **summarizer = второй LLMWrapper**: `summarizer_llm` — полноценный `LLMWrapper` с собственным loose `BudgetTracker`, не голый FakeLLM.
7. **Debater stance при `__init__`**: `{{stance}}` подставляется один раз в конструкторе `Debater.__init__` через `cfg.model_copy(update={"system_prompt": resolved})`.
8. **`__init__.py` ownership**: Step 1 создаёт пустым, **только Step 3** наполняет экспортами; Steps 2 и 4 не трогают.
9. **Delta-only return**: `Agent.step()` возвращает только дельту (новые элементы списков, абсолютные значения счётчиков) — reducer `merge_agent_states` сольёт.
10. **Summarizer вызывается один раз за `step()`** до tool-loop; intra-loop prompt-growth митигирован truncation tool-result до 2000 chars; TODO-комментарий фиксирует ограничение для M6/M7.

---

## Implementation Phases

### Phase 1 (Wave 1): AgentConfig + YAML loader (~1h)
**Goal:** Создать pydantic-модель `AgentConfig` и `load_agent_config`; получить работающий пакет `atm.agents`.

- [ ] 1.1 Создать пакет `src/atm/agents/` с пустым `__init__.py`
  - File: `src/atm/agents/__init__.py`
  - Acceptance: файл существует, пустой; `from atm.agents import AgentConfig` падает (пока нет экспортов — это нормально на этой стадии)

- [ ] 1.2 Написать `AgentConfig` (pydantic v2 `frozen=True`) + `load_agent_config`
  - File: `src/atm/agents/config.py`
  - Acceptance: поля `role`, `system_prompt`, `window_size` (>=1), `max_tool_iters` (>=0), `temperature`, `summarizer_model_id`, `context_token_budget` (>=1), `tools`, `params`; `model_copy(update=...)` работает; YAML-файл загружается

- [ ] 1.3 Написать тесты на `AgentConfig`
  - File: `tests/unit/agents/__init__.py`, `tests/unit/agents/test_config.py`
  - Acceptance: >= 7 тестов проходят: `test_config_defaults`, `test_config_validates_window_size_lt_1`, `test_config_validates_context_token_budget_lt_1`, `test_config_frozen_cannot_mutate`, `test_config_model_copy_update_works`, `test_load_agent_config_minimal`, `test_load_agent_config_with_params_stance`

- [ ] 1.4 Создать YAML-фикстуры для тестов конфига
  - Files: `tests/fixtures/agents/minimal_valid.yaml`, `tests/fixtures/agents/debater_with_stance.yaml`
  - Acceptance: `load_agent_config("minimal_valid.yaml")` возвращает валидный `AgentConfig`

**Verification**: `uv run pytest tests/unit/agents/test_config.py -v` — все >= 7 тестов зелёные.
**Rollback**: удалить `src/atm/agents/`, `tests/unit/agents/`, `tests/fixtures/agents/`.

---

### Phase 2 (Wave 2, параллельно): `_tokens.py` + ролевые подклассы (code) (~1h)

*Оба шага выполняются параллельно после Wave 1.*

#### Step 2: `_tokens.py` (~0.5h)
**Goal:** Готовый helper оценки промпт-токенов.

- [ ] 2.1 Написать `estimate_prompt_tokens(messages, model_id) -> int`
  - File: `src/atm/agents/_tokens.py`
  - Acceptance: tiktoken для `openai:*`, `len//4` heuristic для остальных; `lru_cache(maxsize=8)` на encoder; пустой список → 1; не падает на неизвестном provider
  - Implementation Note: `_PER_MESSAGE_OVERHEAD = 4`; комментарий со ссылкой на OpenAI cookbook; `provider = model_id.split(":", 1)[0] if ":" in model_id else "unknown"`

- [ ] 2.2 Написать тесты на `_tokens.py`
  - File: `tests/unit/agents/test_tokens.py`
  - Acceptance: `test_estimate_empty_returns_one`, `test_estimate_monotonic`, `test_estimate_fallback_for_unknown_provider`, `test_estimate_openai_uses_tiktoken` (two-sided: `abs(tiktoken_est - heuristic_est) / heuristic_est > 0.15`), `test_estimate_with_lrucache_hits`

**Verification**: `uv run pytest tests/unit/agents/test_tokens.py -v`.
**NOT TOUCHED**: `src/atm/agents/__init__.py`.

#### Step 4 — code only: 5 ролевых подклассов + YAML-конфиги (~0.5h)
**Goal:** 5 тонких подклассов готовы для импорта в Step 3.

- [ ] 4.1 Создать `Planner`, `Researcher`, `Executor`, `Critic` (пустые subclass)
  - Files: `src/atm/agents/planner.py`, `src/atm/agents/researcher.py`, `src/atm/agents/executor.py`, `src/atm/agents/critic.py`
  - Acceptance: каждый файл содержит `class Role(Agent): pass` с docstring

- [ ] 4.2 Создать `Debater` с подстановкой stance
  - File: `src/atm/agents/debater.py`
  - Acceptance: `Debater.__init__` делает `cfg.model_copy(update={"system_prompt": resolved_prompt})`; `ValueError` при неверном stance; `"{{stance}}"` не остаётся в `_system_prompt`
  - Implementation Note: `stance in ("pro", "contra")` — единственные допустимые значения

- [ ] 4.3 Написать YAML-конфиги для всех 5 ролей
  - Files: `conf/agents/planner.yaml`, `conf/agents/researcher.yaml`, `conf/agents/executor.yaml`, `conf/agents/critic.yaml`, `conf/agents/debater.yaml`
  - Acceptance: `window_size` = 15/12/12/15/12 соответственно; все файлы проходят `load_agent_config`
  - Implementation Note: planner `tools: [todo_write, plan_update]`; researcher `[semantic_search, duckduckgo_search, url_fetch, file_read]`; executor `[code_run, file_write, file_read, calculator]`; critic `[test_run, diff, lint]`; debater `[duckduckgo_search, url_fetch]`

**NOT TOUCHED**: `src/atm/agents/__init__.py`.

---

### Phase 3 (Wave 3): `Agent` базовый класс + scratchpad policy C + tool-loop (~3h)
**Goal:** Полностью реализованный `Agent.step()` со всеми алгоритмами; `__init__.py` заполнен экспортами.

- [ ] 3.1 Добавить `model_id` property в `LLMWrapper`
  - File: `src/atm/llm/wrapper.py`
  - Acceptance: `agent.llm.model_id == "fake:deterministic"`; `uv run pytest tests/unit/llm/ -v` остаётся зелёным

- [ ] 3.2 Реализовать `base.py`: `_StepOutcome`, `AgentView`, `Agent.__init__`, `Agent.step`
  - File: `src/atm/agents/base.py`
  - Acceptance: `Agent.step(state)` возвращает delta-dict с ключами `agents`, `messages`, `llm_calls`; `step_count` = абсолютное значение; `tool_calls` аккумулируются по всем итерациям (C1 fix)
  - Implementation Note: `_StepOutcome(frozen=True)` dataclass; `step()` — `async def step(state: GraphState) -> dict[str, Any]`

- [ ] 3.3 Реализовать `_build_prompt(view)` со scratchpad windowing
  - File: `src/atm/agents/base.py`
  - Acceptance: `scratchpad[-window_size:]` — хвост событий; system-msg + task + inbox + summary + tail; docstring документирует "events, not logical steps"

- [ ] 3.4 Реализовать `_run_tool_loop(messages)` с аккумуляцией tool_calls (C1)
  - File: `src/atm/agents/base.py`
  - Acceptance: `tool_calls_accum` и `tool_results_accum` не сбрасываются между итерациями; `ToolError` превращается в `ToolResult(ok=False)`; tool-result content truncated до 2000 chars; TODO-комментарий про re-entrant summarization

- [ ] 3.5 Реализовать `_maybe_summarize(view)` + `_build_tools_schema()`
  - File: `src/atm/agents/base.py`
  - Acceptance: summarizer вызывается если `estimate_prompt_tokens > context_token_budget` И `summarizer_llm is not None`; `_build_tools_schema` логирует через structlog при неизвестном tool-name (НЕ `warnings.warn`)

- [ ] 3.6 Заполнить `__init__.py` полными экспортами (после мерджа Step 4 code)
  - File: `src/atm/agents/__init__.py`
  - Acceptance: `from atm.agents import Agent, AgentConfig, load_agent_config, Planner, Researcher, Executor, Critic, Debater` работает; `__all__` задан

- [ ] 3.7 Написать FakeLLM-фикстуры для тестов Agent
  - Files: `tests/fixtures/llm/m5_agent_basic.yaml`, `m5_agent_tool_loop.yaml`, `m5_agent_scratchpad.yaml`, `m5_agent_summarizer_primary.yaml`, `m5_agent_summarizer_secondary.yaml`
  - Acceptance: каждый fixture задаёт `version: 1, mode: scripted, entries: [...]`; summarizer secondary индексируется по `agent_id: p1_summarizer`

- [ ] 3.8 Написать тесты `test_agent_basic.py`
  - File: `tests/unit/agents/test_agent_basic.py`
  - Acceptance: `test_happy_path_no_tools` (step_count==1, kind==DRAFT, tool_calls==[]), `test_llm_wrapper_exposes_model_id`

- [ ] 3.9 Написать тесты `test_agent_tool_loop.py`
  - File: `tests/unit/agents/test_agent_tool_loop.py`
  - Acceptance: `test_tool_call_then_final` (4 scratchpad events); `test_tool_calls_accumulate_across_iters` — **C1 assertion**: `len(tool_calls) == 2` при 2 итерациях с tool-calls

- [ ] 3.10 Написать тесты `test_agent_scratchpad_window.py`
  - File: `tests/unit/agents/test_agent_scratchpad_window.py`
  - Acceptance: `window_size=3` → prompt содержит R2/R3/R4, не содержит R0/R1; оригинальный scratchpad не мутирован

- [ ] 3.11 Написать тесты `test_agent_summarizer.py`
  - File: `tests/unit/agents/test_agent_summarizer.py`
  - Acceptance: `test_summarizer_invoked_when_over_budget` (summary == "SUMMARY_TEXT"); `test_summarizer_not_invoked_when_under_budget`; `test_summarizer_noop_when_summarizer_llm_is_none`
  - Implementation Note: summarizer_llm = `FakeLLM` обёрнутый в `LLMWrapper` с loose `BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0)`

- [ ] 3.12 Написать тест `test_agent_budget_propagation.py`
  - File: `tests/unit/agents/test_agent_budget_propagation.py`
  - Acceptance: `BudgetExceededError` из LLMWrapper не перехватывается Agent-ом, propagates up; `pytest.raises(BudgetExceededError)`

**Verification**:
- `uv run pytest tests/unit/agents/test_agent_*.py -v` — все зелёные
- `uv run pytest tests/unit/llm/ -v` — не регрессирует
- `uv run ruff check src/atm/agents/ src/atm/llm/wrapper.py`
- `uv run mypy src/atm/agents/`
**Rollback**: `git checkout -- src/atm/agents/base.py src/atm/agents/__init__.py tests/unit/agents/test_agent_*.py tests/fixtures/llm/m5_agent_*.yaml src/atm/llm/wrapper.py`

---

### Phase 4 (Wave 4, параллельно): тесты ролей + exit-criterion (~1h)

*Оба шага выполняются параллельно после Wave 3.*

#### Step 4 — tests: Debater + role configs (~0.5h)
**Goal:** Подтвердить корректность 5 ролевых подклассов и конфигов.

- [ ] 4.4 Написать тесты `test_debater_stance.py`
  - File: `tests/unit/agents/test_debater_stance.py`
  - Acceptance: `test_debater_pro_prompt_contains_pro` (`"Stance: pro"` в prompt, нет `{{stance}}`); `test_debater_contra_prompt_contains_contra`; `test_debater_invalid_stance_raises` (`ValueError`)

- [ ] 4.5 Написать тесты `test_role_configs_load.py`
  - File: `tests/unit/agents/test_role_configs_load.py`
  - Acceptance: parametrized по 5 YAML, каждый → валидный `AgentConfig`, `role` совпадает с basename файла

**Verification**: `uv run pytest tests/unit/agents/test_debater_stance.py tests/unit/agents/test_role_configs_load.py -v`.

#### Step 5: Exit-criterion test для Executor (~0.5h)
**Goal:** Доказать end-to-end корректность tool-loop на реалистичном сценарии.

- [ ] 5.1 Написать FakeLLM-фикстуру для Executor
  - File: `tests/fixtures/llm/m5_executor_exit.yaml`
  - Acceptance: 2 entries — step_0 `finish_reason: tool_calls` (code_run), step_1 `finish_reason: stop`

- [ ] 5.2 Написать тест `test_executor_exit_criterion.py`
  - File: `tests/unit/agents/test_executor_exit_criterion.py`
  - Acceptance: `step()` завершается; scratchpad >= 3 events; `tool_calls` 1 entry (C1 validation); `tool_results[0].ok == True`, `output == {"stdout": "2\n"}`; outbox DRAFT содержит `"output was 2"`
  - Implementation Note: inline `FakeCodeRunTool` с `async def ainvoke(args) -> ToolResult`; зарегистрировать в `ToolRegistry`

**Verification**: `uv run pytest tests/unit/agents/test_executor_exit_criterion.py -v`.
**Rollback**: удалить тест + фикстуру.

---

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/agents/__init__.py` | NEW (empty → populated by Step 3) | Package root; re-exports всех публичных символов |
| `src/atm/agents/config.py` | NEW | `AgentConfig` pydantic model + `load_agent_config` |
| `src/atm/agents/_tokens.py` | NEW | `estimate_prompt_tokens` с tiktoken+heuristic |
| `src/atm/agents/base.py` | NEW | `Agent`, `_StepOutcome`, `AgentView`, все алгоритмы |
| `src/atm/agents/planner.py` | NEW | `class Planner(Agent): pass` |
| `src/atm/agents/researcher.py` | NEW | `class Researcher(Agent): pass` |
| `src/atm/agents/executor.py` | NEW | `class Executor(Agent): pass` |
| `src/atm/agents/critic.py` | NEW | `class Critic(Agent): pass` |
| `src/atm/agents/debater.py` | NEW | `Debater.__init__` с stance-substitution |
| `src/atm/llm/wrapper.py` | MODIFIED | +1 line: `@property model_id` |
| `conf/agents/planner.yaml` | NEW | Role config `window_size=15` |
| `conf/agents/researcher.yaml` | NEW | Role config `window_size=12` |
| `conf/agents/executor.yaml` | NEW | Role config `window_size=12` |
| `conf/agents/critic.yaml` | NEW | Role config `window_size=15` |
| `conf/agents/debater.yaml` | NEW | Role config `window_size=12`, `params.stance: pro` |
| `tests/unit/agents/test_config.py` | NEW | >= 7 тестов на AgentConfig |
| `tests/unit/agents/test_tokens.py` | NEW | 5 тестов на estimate_prompt_tokens |
| `tests/unit/agents/test_agent_basic.py` | NEW | Happy-path + model_id |
| `tests/unit/agents/test_agent_tool_loop.py` | NEW | Tool-loop + C1 accumulation assertion |
| `tests/unit/agents/test_agent_scratchpad_window.py` | NEW | Window semantics |
| `tests/unit/agents/test_agent_summarizer.py` | NEW | 3 summarizer cases |
| `tests/unit/agents/test_agent_budget_propagation.py` | NEW | BudgetExceededError propagation |
| `tests/unit/agents/test_debater_stance.py` | NEW | 3 stance cases |
| `tests/unit/agents/test_role_configs_load.py` | NEW | Parametrized 5-role load |
| `tests/unit/agents/test_executor_exit_criterion.py` | NEW | Exit-criterion proof |
| `tests/fixtures/agents/minimal_valid.yaml` | NEW | Минимальная фикстура конфига |
| `tests/fixtures/agents/debater_with_stance.yaml` | NEW | Фикстура с params.stance |
| `tests/fixtures/llm/m5_agent_basic.yaml` | NEW | 1 scripted entry для happy-path |
| `tests/fixtures/llm/m5_agent_tool_loop.yaml` | NEW | 2-3 entries для tool-loop |
| `tests/fixtures/llm/m5_agent_scratchpad.yaml` | NEW | Entries для window-test |
| `tests/fixtures/llm/m5_agent_summarizer_primary.yaml` | NEW | Primary LLM entries |
| `tests/fixtures/llm/m5_agent_summarizer_secondary.yaml` | NEW | Summarizer entries (`agent_id: p1_summarizer`) |
| `tests/fixtures/llm/m5_executor_exit.yaml` | NEW | 2 entries для exit-criterion |

---

## Dependencies & Order Constraints

```
Step 1 (Wave 1)
  └─> Step 2 (Wave 2, parallel) — _tokens.py
  └─> Step 4-code (Wave 2, parallel) — role modules + YAMLs
        └─> Step 3 (Wave 3, solo) — base.py + __init__.py (after both Wave 2 branches merged)
              └─> Step 4-tests (Wave 4, parallel) — debater/role tests
              └─> Step 5 (Wave 4, parallel) — executor exit-criterion
```

**Hard constraints**:
- `src/atm/agents/__init__.py` — создаётся Step 1 (пустым), заполняется **только Step 3**. Steps 2 и 4 MUST NOT трогать.
- `src/atm/llm/wrapper.py` — трогается только Step 3 (одна property).
- Fixture-файлы `tests/fixtures/llm/` — не пересекаются между шагами (`m5_agent_*` для Step 3, `m5_debater_*` для Step 4, `m5_executor_*` для Step 5).

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| R1.1: mutable-default trap в pydantic v2 | Medium | Low | `Field(default_factory=list/dict)` |
| R2.1: tiktoken сетевой запрос в CI | Low | Medium | Использовать только `gpt-4o-mini` (BPE bundled) |
| R3.1: reducer list-concat — возврат full list вместо delta | Medium | High | Тесты проверяют длины; документировано в context.md |
| R3.2: `summary_before_window` reducer right-wins — пустая строка перезапишет | Low | Medium | В delta кладём существующее значение если не обновляли |
| R3.3: счётчики — возвращать абсолютные, не инкремент | Medium | High | Явно задокументировано; тест step_count==1 |
| R3.4: `model_id` property конфликтует с LLM-тестами | Low | Medium | Verification: `uv run pytest tests/unit/llm/ -v` |
| R3.5: summarizer бросает BudgetExceededError — propagates | Low | Low | Желательное поведение; no try/except в _maybe_summarize |
| R3.6: пустой tools_schema — передавать `tools=None`, не `[]` | Low | Medium | Явная проверка `if tools_schema else None` |
| R3.8: `__init__.py` imports требуют role-классов до финализации | Medium | Low | Wave scheduling — Step 4 code в Wave 2, Step 3 финализирует __init__ в Wave 3 после merge |
| R4.1: `AgentConfig` frozen + `model_copy` | Low | Low | `test_config_model_copy_update_works` в Step 1 |
| R4.2: `warnings.warn` становится исключением под pytest | High | High | structlog вместо warnings.warn |
| R5.1: сигнатурный дрейф Step 3 vs Step 5 | Low | Medium | Wave ordering — Step 5 зависит от Step 3 |
| R5.2: невалидный `ToolSchema.parameters` | Low | Low | Minimal `{"type":"object","properties":{}}` |

---

## Out of Scope

- LangGraph `StateGraph` и топология — M6.
- `shared.signals` emission, `phase_emit`, interrupt/HITL — M6/M8.
- `tool_result_as_tool_message: bool` флаг — YAGNI, M6+.
- Summarizer prompt YAML-override — M6+.
- Re-entrant summarization внутри tool-loop — M6/M7.
- Real DockerSandbox в тестах — вне M5 scope.
- Observability `dispatch_custom_event` — M6.

---

## Timeline

- Total: ~6h
- Wave 1: ~1h (Step 1)
- Wave 2: ~1h (Steps 2 + 4-code, параллельно)
- Wave 3: ~3h (Step 3, solo — ядро M5)
- Wave 4: ~1h (Steps 4-tests + 5, параллельно)
- Created: 2026-04-24
