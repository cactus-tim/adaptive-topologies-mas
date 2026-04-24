# M5 — Agent framework + scratchpad policy C — Tasks

## Legend
- `[ ]` pending
- `[x]` done
- `[~]` skipped

---

## Wave 1 — Foundation (sole) COMPLETE

*Prerequisite for everything. No parallelism.*

### Step 1: AgentConfig + YAML loader
**Type**: tdd | **Depends On**: — | **Can-Parallel-With**: none

- [x] 1.1 Создать пакет `src/atm/agents/` с пустым `__init__.py`
  - File: `src/atm/agents/__init__.py`
  - Acceptance: файл создан; **Step 1 ONLY** creates it empty

- [x] 1.2 Написать `AgentConfig` (pydantic v2 `frozen=True`) + `load_agent_config`
  - File: `src/atm/agents/config.py`
  - Acceptance: все поля по спеке (`window_size>=1`, `max_tool_iters>=0`, `context_token_budget>=1`); `Field(default_factory=...)` для `tools`/`params`; `model_copy(update=...)` работает

- [x] 1.3 Написать тесты AgentConfig (>= 7 штук)
  - Files: `tests/unit/agents/__init__.py`, `tests/unit/agents/test_config.py`
  - Acceptance: `test_config_defaults`, `test_config_validates_window_size_lt_1`, `test_config_validates_context_token_budget_lt_1`, `test_config_frozen_cannot_mutate`, `test_config_model_copy_update_works`, `test_load_agent_config_minimal`, `test_load_agent_config_with_params_stance` — все зелёные

- [x] 1.4 Создать YAML-фикстуры для конфига
  - Files: `tests/fixtures/agents/minimal_valid.yaml`, `tests/fixtures/agents/debater_with_stance.yaml`
  - Acceptance: `load_agent_config` загружает каждый без ошибок

**Verification**: `uv run pytest tests/unit/agents/test_config.py -v`

---

## Wave 2 — Parallel (after Wave 1) IN PROGRESS

*Steps 2 и 4-code выполняются параллельно. Они не пересекаются по файлам.*

### Step 2: `_tokens.py` helper
**Type**: tdd | **Depends On**: 1 (package exists) | **Can-Parallel-With**: 4-code

- [x] 2.1 Написать `estimate_prompt_tokens(messages, model_id) -> int`
  - File: `src/atm/agents/_tokens.py`
  - Acceptance: tiktoken для `openai:*`; `len(text)//4` fallback; `_PER_MESSAGE_OVERHEAD = 4`; `lru_cache(maxsize=8)` на encoder; пустой список → 1; не падает на неизвестных providers
  - Note: комментарий-ссылка на OpenAI cookbook convention; НЕ импортировать `_count_prompt_tokens` из `llm/wrapper.py`

- [x] 2.2 Написать тесты `test_tokens.py` (5 штук)
  - File: `tests/unit/agents/test_tokens.py`
  - Acceptance: `test_estimate_empty_returns_one`, `test_estimate_monotonic`, `test_estimate_fallback_for_unknown_provider`, `test_estimate_openai_uses_tiktoken` (two-sided: `abs(tiktoken_est - heuristic_est) / heuristic_est > 0.15`), `test_estimate_with_lrucache_hits`

- **NOT TOUCHED**: `src/atm/agents/__init__.py`

**Verification**: `uv run pytest tests/unit/agents/test_tokens.py -v`

---

### Step 4 (code only): 5 ролевых подклассов + YAML-конфиги
**Type**: simple | **Depends On**: 1 | **Can-Parallel-With**: 2

- [x] 4.1 Создать `Planner`, `Researcher`, `Executor`, `Critic` (пустые subclasses)
  - Files: `src/atm/agents/planner.py`, `src/atm/agents/researcher.py`, `src/atm/agents/executor.py`, `src/atm/agents/critic.py`
  - Acceptance: `class Role(Agent): pass` + docstring в каждом файле; используют `from atm.agents.base import Agent`

- [x] 4.2 Создать `Debater` с подстановкой stance
  - File: `src/atm/agents/debater.py`
  - Acceptance: `__init__` делает `cfg.model_copy(update={"system_prompt": resolved})`; `ValueError` при stance не в `{"pro","contra"}`; `{{stance}}` не остаётся в `_system_prompt`

- [x] 4.3 Написать YAML-конфиги (5 файлов)
  - Files: `conf/agents/planner.yaml`, `conf/agents/researcher.yaml`, `conf/agents/executor.yaml`, `conf/agents/critic.yaml`, `conf/agents/debater.yaml`
  - Acceptance: `window_size` = planner 15, researcher 12, executor 12, critic 15, debater 12; debater содержит `params: {stance: pro}` и `{{stance}}` в system_prompt; все 5 проходят `load_agent_config`

- **NOT TOUCHED**: `src/atm/agents/__init__.py`

*(Тесты для Step 4 — в Wave 4; они зависят от Step 3)*

---

## Wave 3 — Core (solo, after Wave 2 merged) COMPLETE

*Ключевой шаг. Единственный. Финализирует `__init__.py`.*

### Step 3: Базовый `Agent` + scratchpad policy C + tool-loop
**Type**: tdd | **Depends On**: 1, 2 | **Can-Parallel-With**: none в Wave 3

- [x] 3.1 Добавить `@property model_id` в `LLMWrapper`
  - File: `src/atm/llm/wrapper.py`
  - Acceptance: `agent.llm.model_id == "fake:deterministic"`; `uv run pytest tests/unit/llm/ -v` зелёный

- [x] 3.2 Реализовать `_StepOutcome`, `AgentView`, `Agent.__init__`, `Agent.step`
  - File: `src/atm/agents/base.py`
  - Acceptance: `step()` async, возвращает delta-dict (`agents`, `messages`, `llm_calls`); `step_count` = абсолютное значение

- [x] 3.3 Реализовать `_build_prompt(view)` со scratchpad windowing
  - File: `src/atm/agents/base.py`
  - Acceptance: `scratchpad[-window_size:]`; docstring документирует "events, not logical steps"; оригинальный scratchpad не мутируется

- [x] 3.4 Реализовать `_run_tool_loop(messages)` с аккумуляцией (C1 fix)
  - File: `src/atm/agents/base.py`
  - Acceptance: `tool_calls_accum` накапливается по всем итерациям (не сбрасывается); `ToolError` → `ToolResult(ok=False)`; tool-result content truncated до 2000 chars; TODO-комментарий про re-entrant summarization

- [x] 3.5 Реализовать `_maybe_summarize` + `_build_tools_schema`
  - File: `src/atm/agents/base.py`
  - Acceptance: `_maybe_summarize` вызывает summarizer только если `estimate > budget` И `summarizer_llm is not None`; `_build_tools_schema` использует `structlog.warning` для неизвестных tool-names (НЕ `warnings.warn`)

- [x] 3.6 Заполнить `__init__.py` экспортами (после merge Step 4 code)
  - File: `src/atm/agents/__init__.py`
  - Acceptance: `from atm.agents import Agent, AgentConfig, load_agent_config, Planner, Researcher, Executor, Critic, Debater` работает; `__all__` задан

- [x] 3.7 Создать FakeLLM-фикстуры для тестов Agent
  - Files: `tests/fixtures/llm/m5_agent_basic.yaml`, `m5_agent_tool_loop.yaml`, `m5_agent_scratchpad.yaml`, `m5_agent_summarizer_primary.yaml`, `m5_agent_summarizer_secondary.yaml`
  - Acceptance: summarizer secondary fixture использует `agent_id: p1_summarizer`

- [x] 3.8 `test_agent_basic.py` — happy-path + model_id
  - File: `tests/unit/agents/test_agent_basic.py`
  - Acceptance: `test_happy_path_no_tools` (step_count==1, kind==DRAFT, tool_calls==[]), `test_llm_wrapper_exposes_model_id`

- [x] 3.9 `test_agent_tool_loop.py` — tool-loop + C1 accumulation
  - File: `tests/unit/agents/test_agent_tool_loop.py`
  - Acceptance: `test_tool_call_then_final` (4 scratchpad events); **`test_tool_calls_accumulate_across_iters`** — `len(tool_calls) == 2` (C1 assertion)

- [x] 3.10 `test_agent_scratchpad_window.py` — window semantics
  - File: `tests/unit/agents/test_agent_scratchpad_window.py`
  - Acceptance: window_size=3 → prompt содержит R2/R3/R4, не R0/R1; scratchpad не мутирован

- [x] 3.11 `test_agent_summarizer.py` — 3 summarizer cases
  - File: `tests/unit/agents/test_agent_summarizer.py`
  - Acceptance: `test_summarizer_invoked_when_over_budget` (summary=="SUMMARY_TEXT"); `test_summarizer_not_invoked_when_under_budget`; `test_summarizer_noop_when_summarizer_llm_is_none`
  - Note: summarizer_llm = FakeLLM обёрнут в LLMWrapper с loose BudgetTracker

- [x] 3.12 `test_agent_budget_propagation.py` — no-swallow
  - File: `tests/unit/agents/test_agent_budget_propagation.py`
  - Acceptance: `pytest.raises(BudgetExceededError)` при `_RaisingLLM`; `.level == "run"`

**Verification**:
- `uv run pytest tests/unit/agents/test_agent_*.py -v`
- `uv run pytest tests/unit/llm/ -v`
- `uv run ruff check src/atm/agents/ src/atm/llm/wrapper.py`
- `uv run mypy src/atm/agents/`

---

## Wave 4 — Parallel finish (after Wave 3) IN PROGRESS

*Step 4-tests и Step 5 выполняются параллельно.*

### Step 4 (tests): Debater + role configs
**Type**: tdd | **Depends On**: 3, 4-code | **Can-Parallel-With**: 5

- [x] 4.4 `test_debater_stance.py` — 3 случая
  - File: `tests/unit/agents/test_debater_stance.py`
  - Acceptance: `test_debater_pro_prompt_contains_pro` (`"Stance: pro"` есть, `"{{stance}}"` нет); `test_debater_contra_prompt_contains_contra`; `test_debater_invalid_stance_raises` (`ValueError`)

- [x] 4.5 `test_role_configs_load.py` — parametrized 5 ролей
  - File: `tests/unit/agents/test_role_configs_load.py`
  - Acceptance: parametrize по 5 YAML; каждый → `AgentConfig`; `role` == basename файла

**Verification**: `uv run pytest tests/unit/agents/test_debater_stance.py tests/unit/agents/test_role_configs_load.py -v`

---

### Step 5: Exit-criterion test (Executor + fake code_run)
**Type**: tdd | **Depends On**: 3, 4 | **Can-Parallel-With**: 4-tests

- [x] 5.1 Создать FakeLLM-фикстуру `m5_executor_exit.yaml`
  - File: `tests/fixtures/llm/m5_executor_exit.yaml`
  - Acceptance: 2 entries — step_0 `finish_reason: tool_calls` (code_run с `{lang: python, code: "print(1+1)"}`), step_1 `finish_reason: stop` с текстом про "output was 2"

- [x] 5.2 Написать `test_executor_exit_criterion.py`
  - File: `tests/unit/agents/test_executor_exit_criterion.py`
  - Acceptance: `step()` завершается; scratchpad >= 3 events; `tool_calls` 1 entry (C1); `tool_results[0].ok==True`, `output=={"stdout":"2\n"}`; outbox DRAFT содержит `"output was 2"`
  - Note: inline `FakeCodeRunTool` с `ClassVar[ToolSchema]`; зарегистрировать в `ToolRegistry`

**Verification**: `uv run pytest tests/unit/agents/test_executor_exit_criterion.py -v`

---

## Stats

- Total: 18 tasks (1.1–1.4, 2.1–2.2, 4.1–4.3, 3.1–3.12, 4.4–4.5, 5.1–5.2) · ~6h
- Done: 20 / 20 (all waves complete)
- Waves: 4 (max parallel width: 2)

## How to Update

После завершения каждого шага:
1. Отметить задачи `[x]` в этом файле.
2. Обновить заголовок фазы: все задачи сделаны → `COMPLETE`, часть → `IN PROGRESS`.
3. Обновить `SESSION PROGRESS` в `m5-context.md`.
4. Обновить `Stats.Done` счётчик.
