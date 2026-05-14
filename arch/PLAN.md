# Adaptive Topologies MAS — План реализации

> Техническая часть дипломной работы: фреймворк для экспериментов с адаптивными топологиями мульти-агентных LLM-систем + human-in-the-loop. План покрывает ядро агентов и инфраструктуру экспериментов. Тексты, benchmark-датасет и дизайн-гайды — отдельно, после технички.

## 1. Executive summary

Строим модульный фреймворк на **LangGraph** с чёткими слоями: LLM → Agent → Topology → Phase Manager → Orchestrator. Параллельно — слой инфраструктуры: конфиги (YAML+Pydantic+OmegaConf), хранилище (Postgres + Parquet), наблюдаемость (свой LangGraph callback), бюджет-гварды, runner для грида с async + multiprocessing. Этапы идут последовательно с ранним end-to-end (Star+Chain на одной задаче) к концу M6, дальше добираются остальные топологии и адаптация.

## 2. Архитектура (высокоуровневая)

```
┌──────────────────────────────────────────────────────────────┐
│                  Experiment Runner (CLI)                     │
│        load configs → build graph → run → persist            │
└───────────────┬──────────────────────────────────┬───────────┘
                │                                  │
        ┌───────▼────────┐                ┌────────▼─────────┐
        │  Orchestrator  │                │   ObserveLayer    │
        │  (LangGraph)   │                │  callbacks, trace │
        └───┬────────┬───┘                └────────┬──────────┘
            │        │                             │
    ┌───────▼──┐   ┌─▼──────────────┐              │
    │ Phase    │   │  Topology      │              │
    │ Manager  │   │  (Star/Chain/  │              │
    │ (FSM)    │   │  Mesh/Debate/  │              │
    │          │   │  Hier/Adaptive)│              │
    └──────────┘   └────────┬───────┘              │
                            │                      │
                   ┌────────▼────────┐             │
                   │   Agent Layer   │             │
                   │ (Planner/Res/   │             │
                   │  Exec/Critic/   │             │
                   │  Debater)       │             │
                   └───┬──────────┬──┘             │
                       │          │                │
              ┌────────▼──┐   ┌───▼─────────┐      │
              │ LLMWrapper│   │ Tools       │      │
              │ (+budget  │   │ (global +   │      │
              │  +cache)  │   │  local +    │      │
              │           │   │  DockerSbx) │      │
              └───────────┘   └─────────────┘      │
                                                   │
              ┌────────────────────────┐           │
              │    Human Gateway       │           │
              │ (LLM-sim / CLI / Web)  │           │
              └────────────────────────┘           │
                                                   │
    ┌──────────────────────────────────────────────▼─────┐
    │            Storage: Postgres + Parquet             │
    └─────────────────────────────────────────────────────┘
```

### Ключевые принципы

- **Каждая топология = отдельный класс** со своим методом `build(agents) → CompiledStateGraph`. Adaptive — мета-граф поверх subgraph-ов.
- **Per-agent state** (inbox/outbox/scratchpad) с кастомным reducer.
- **Scratchpad-политика C**: пишется всегда (для аудита), в промпт — окно последних K шагов + опциональный summarizer.
- **Провайдер LLM — абстрагирован** через `LLMWrapper` поверх LangChain chat models (primary: OpenAI; switch на Anthropic/vLLM/Ollama через конфиг).
- **HITL — через Protocol `HumanGateway`**, LangGraph `interrupt()` + Postgres checkpointer.
- **Observability = свой LangGraph `AsyncCallbackHandler`**, пишет в parquet + Postgres асинхронно с буфером и обязательным sync-flush на границах run-а, transitions и ошибок (подробнее — `arch.md §10.3`, `§17/#2`).
- **Budget — three-tier** (per-call / per-run / per-experiment), dry-run для оценки грида.

## 3. Технологический стек

| Слой | Выбор |
|---|---|
| Язык / runtime | Python 3.11+ |
| Package manager | uv |
| Граф / оркестрация | LangGraph |
| LLM SDK | LangChain chat models (`init_chat_model`) |
| Модели (primary) | OpenAI GPT-4/GPT-4-mini |
| Модели (fallback/experiments) | Anthropic Claude, vLLM, Ollama (через конфиг) |
| Конфиги | YAML + Pydantic + OmegaConf |
| БД метаданных | PostgreSQL 16 |
| ORM | SQLAlchemy 2.x (async) |
| Migrations | Alembic |
| Bulk storage | Apache Parquet (PyArrow) |
| Checkpointer | `langgraph-checkpoint-postgres` |
| Tools: sandbox | Docker-контейнеры (свой `DockerSandbox`) |
| Tools: search | DuckDuckGo (бесплатно) |
| Human UI | Streamlit (этап 2) + CLI (этап 1) |
| Параллелизм | asyncio (внутри run) + ProcessPoolExecutor (grid) |
| CLI | Typer |
| Линт / тип | ruff + mypy |
| Тесты | pytest + pytest-asyncio |
| Анализ | pandas + matplotlib + jupyter |

## 4. Структура репозитория

```
adaptive-topologies-mas/
├── pyproject.toml                  # uv + deps
├── uv.lock
├── PLAN.md                         # этот файл
├── README.md
├── docker-compose.yml              # postgres + (sandbox infra)
├── .env.example
├── alembic.ini
├── alembic/                        # migrations
│
├── conf/                           # YAML configs
│   ├── base.yaml
│   ├── topology/                   # star.yaml chain.yaml ...
│   ├── agents/                     # planner.yaml ...
│   ├── task/                       # humaneval.yaml mmlu.yaml ...
│   ├── model/                      # gpt4.yaml claude.yaml ...
│   └── experiment/                 # grid definitions
│
├── src/atm/                        # "Adaptive Topologies MAS"
│   ├── __init__.py
│   ├── core/                       # базовые типы и state
│   │   ├── types.py                # Message, ToolCall, Phase, HumanRole
│   │   ├── state.py                # AgentState, GraphState, reducer
│   │   └── errors.py
│   ├── llm/                        # LLMWrapper, budget, pricing
│   │   ├── wrapper.py
│   │   ├── budget.py
│   │   ├── pricing.py
│   │   └── providers/              # openai.py anthropic.py vllm.py
│   ├── tools/
│   │   ├── base.py                 # Tool Protocol, registry
│   │   ├── global_tools.py         # calculator, search, url_fetch
│   │   ├── local_tools.py          # role-specific
│   │   └── sandbox/
│   │       ├── base.py
│   │       ├── docker_sandbox.py
│   │       └── subprocess_sandbox.py  # optional dev mode
│   ├── agents/
│   │   ├── base.py                 # Agent class
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── executor.py
│   │   ├── critic.py
│   │   └── debater.py
│   ├── topology/
│   │   ├── base.py                 # abstract Topology + TopologyRegistry
│   │   ├── star.py
│   │   ├── chain.py
│   │   ├── mesh.py
│   │   ├── debate.py
│   │   ├── hierarchical.py
│   │   └── adaptive.py             # мета-граф
│   ├── phases/
│   │   ├── manager.py              # PhaseState FSM
│   │   └── router.py               # LLM-router для переходов
│   ├── human/
│   │   ├── gateway.py              # Protocol
│   │   ├── llm_simulated.py
│   │   ├── cli_gateway.py
│   │   └── streamlit_gateway.py    # M+
│   ├── storage/
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── session.py
│   │   ├── parquet_writer.py
│   │   └── checkpointer.py         # обёртка над langgraph-checkpoint-postgres
│   ├── observability/
│   │   ├── callbacks.py            # LangGraph BaseCallbackHandler
│   │   └── tracer.py
│   ├── tasks/
│   │   ├── base.py                 # TaskSpec
│   │   ├── registry.py
│   │   ├── humaneval.py
│   │   ├── mmlu.py
│   │   ├── creative.py
│   │   └── analysis.py
│   ├── evaluation/
│   │   ├── judges.py               # LLM-as-judge
│   │   ├── ground_truth.py         # programming test runners
│   │   ├── metrics.py              # quality, efficiency, time, human
│   │   └── tlx.py                  # NASA-TLX для human-factors
│   ├── experiment/
│   │   ├── config.py               # Pydantic схемы всех конфигов
│   │   ├── loader.py               # OmegaConf → Pydantic
│   │   ├── runner.py               # один run
│   │   ├── grid.py                 # grid с process pool
│   │   └── cli.py                  # typer entrypoint
│   └── analysis/
│       ├── loaders.py              # read_runs(), read_llm_calls()
│       └── plots.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── notebooks/                      # для анализа
│
└── dev/                            # dev-docs, заметки
```

## 5. Модель данных

### 5.1 Postgres (SQLAlchemy)

| Таблица | Ключевые поля | Назначение |
|---|---|---|
| `experiments` | id, name, config_snapshot (jsonb), git_sha, started_at, finished_at, total_cost | Метаданные грид-запуска |
| `runs` | id, exp_id, topology, task_id, agent_set, human_role, seed, model, **models_by_role_json**, **model_version_snapshot**, **sandbox_image_digest**, status, **finish_reason**, budget_spent_usd, quality_score, wall_time_s, iterations | Один прогон. Жирным — поля для reproducibility bundle (arch.md §14.4) и attribution остановки (arch.md §7.1 stopping precedence) |
| `phases` | id, run_id, phase_name, started_at, ended_at, entry_reason, topology_used | Для Adaptive: переходы |
| `human_interactions` | id, run_id, role, requested_at, answered_at, context_ref, answer_ref, tlx_scores (jsonb) | HITL-логи |
| `checkpoints` | таблицы LangGraph checkpointer | Автоматически |
| `budget_events` | id, run_id, level (call/run/exp), event (warn/exceed), value_usd, at | Аудит бюджетов |

### 5.2 Parquet (файловая структура)

```
data/
├── experiments/{exp_id}/
│   ├── metadata.json              # дубль конфига
│   └── runs/{run_id}/
│       ├── llm_calls.parquet      # каждый вызов LLM: in/out tokens, cost, latency, agent_id, model
│       ├── messages.parquet       # межагентские сообщения (граф-edges по времени)
│       ├── tool_calls.parquet     # все вызовы инструментов: tool, args, result, latency
│       ├── scratchpads/{agent_id}.parquet  # per-agent внутренние размышления
│       └── phases.parquet         # дубль из PG для удобства pandas
```

### 5.3 Конфиги (YAML) — пример

```yaml
# conf/base.yaml
experiment:
  name: "m6-star-chain-humaneval"
  seed: 42
  budget:
    per_call_usd: 0.10
    per_run_usd: 0.50
    per_experiment_usd: 50.0

agents:
  set: "canonical_4"              # из conf/agents/canonical_4.yaml
  scratchpad:
    policy: "window_with_summary" # вариант C
    window_size: 3
    summarizer_model: "gpt-4o-mini"

topology:
  name: "star"                    # из conf/topology/star.yaml
  max_iterations: 10

task:
  name: "humaneval"
  split: "test"
  limit: 50

model:
  # per-role модели; дешёвые на worker-ролях, умные на проверке/судействе.
  # Полная схема — arch.md §12.1 ModelCfg.
  default: "openai:gpt-4o-mini"           # fallback для неуказанных ролей
  by_role:
    planner:      "openai:gpt-4o-mini"
    researcher:   "openai:gpt-4o-mini"
    executor:     "openai:gpt-4o-mini"
    critic:       "openai:gpt-4o"
    debater:      "openai:gpt-4o-mini"
    coordinator:  "openai:gpt-4o"
  judge:      "openai:gpt-4o"             # LLM-as-judge в evaluation (не Critic-агент)
  summarizer: "openai:gpt-4o-mini"        # scratchpad policy C
  router:     "openai:gpt-4o-mini"        # PhaseRouter/TopologyRouter при llm-режиме
  prompt_cache_scope: "per_run"           # per_run | per_task | off (arch.md §12.1, §17)
  provider_opts:
    prompt_cache: true

observability:
  callback_sync: false          # async с буфером + mandatory flush (см. arch.md §10.3); true — только для отладочных unit-тестов
  parquet_dir: "data/experiments"
  pg_dsn: ${oc.env:PG_DSN}
```

## 6. Топологии — спецификация

Общий интерфейс:

```python
class Topology(Protocol):
    name: str
    def build(self, agents: list[Agent], cfg: TopologyConfig) -> CompiledStateGraph: ...
```

| Топология | Структура | Условие выхода | Нюансы |
|---|---|---|---|
| **Star** | Coordinator — центральная нода; conditional edges Coord ↔ {Planner, Executor, Critic} | Critic.approve или max_iter | Coord агрегирует, принимает решения о следующем шаге |
| **Chain** | Planner → Executor → Critic → [loop if not approved / END] | Critic.approve или max_iter | Линейный pipeline; при отклонении возврат к Executor |
| **Mesh** | Broadcast-bus; все агенты читают bus, пишут bus; round-robin активации | `max_rounds` ИЛИ `consensus_threshold` голосов | Круги фиксированы; опц. голосование за финал |
| **Debate** | Debater_pro + Debater_contra (параллельно) → Critic (judge) | `max_rounds` или Judge.decide | Debater-ы параметризуются `stance` |
| **Hierarchical** | Top-Coord → 2 Sub-Coord → воркеры; **ровно 2 уровня**; каждый sub — compiled subgraph | Top-Coord.finalize | Subgraph per sub-команда (LangGraph feature); 3-й уровень не предусмотрен (arch.md §7.6) |
| **Adaptive (L2)** | Мета-граф: START → PhaseRouter → TopologyRouter → {5 subgraph} → TransitionGate → loop/END. Топология меняется и внутри фазы по сигналам агентов. Tick-granularity. Monotonic phases. Superset of 7 agent roles. | `phase == done` или budget/iter exceeded | 3 режима TopologyRouter (rule / llm / oracle); SwitchGuards против thrashing; `TopologyTransition` пишется каждый тик (включая no-change). Подробности — `arch.md §7.7, §8, §8bis` |

**Условия активации агентов в Mesh:**
- round-robin по умолчанию, опц. priority-based (Critic первым если есть сообщения с `type=draft`)

**Stopping criteria — единая precedence для всех топологий** (полная спека — arch.md §7.1):

```
1. BudgetExceededError         → status='budget_exceeded' (hard stop)
2. Global max_iter             → status='completed', finish_reason='max_iter'
3. Topology-specific success   → critic.approved / consensus / judge.decide / finalize
4. Topology-specific max       → max_rounds / max_exec_iter (последний страж)
```

Реализация — helper `_should_stop(state) -> (bool, reason)`, общий для всех топологий; `reason` пишется в `runs.finish_reason`.

**PhaseManager FSM** (для Adaptive и не только):
- States: `planning` → `execution` → `verification` → `done`
- Transitions: триггер от агента (`emit_phase_change`), rule-based (после N итераций), LLM-router (`should_switch_phase(state)`)
- Каждый transition записывается в `phases` таблицу

## 7. Роли агентов и их базовые промпты

5 ролей — 5 `Agent`-ов с фиксированными `role_id`, но параметризуемыми `system_prompt`, `tools_local`, `llm_config`.

| Роль | Системный промпт (тезис) | Local tools |
|---|---|---|
| Planner | Декомпозирует задачу, выдаёт пошаговый план | `todo_write`, `plan_update` |
| Researcher | Ищет факты, извлекает данные, обобщает | `semantic_search`, `arxiv_search` (stub) |
| Executor | Выполняет план — пишет код/текст/решение | `code_run` (Docker), `file_write` |
| Critic | Проверяет, ищет ошибки, предлагает правки | `test_run`, `diff`, `lint` |
| Debater | Защищает позицию (pro/contra), опровергает оппонента | (параметр `stance`), `search` |

Фиксированные наборы 4 агента (в конфиге):
- **default**: Planner, Researcher, Executor, Critic (для Chain/Star/Mesh/Hierarchical)
- **debate**: Planner, Debater(pro), Debater(contra), Critic-as-judge
- **Hierarchical**: Coordinator, Executor ×2 (с разными personas), Critic — coordinator роль берёт на себя функции Planner

## 8. Milestone-план

> Каждый milestone имеет: **цель** · **deliverable** · **задачи (чеклист)** · **зависимости** · **exit criteria**. Параллелимые потоки отмечены `∥`.

### M0 — Bootstrap проекта (0.5 дня) ✅

- **Цель:** рабочий скелет с линтом, тестами, БД, docker-compose
- **Зависимости:** —
- **Задачи:**
  - [x] `uv init` + `pyproject.toml` с базовыми зависимостями
  - [x] Структура каталогов из секции 4
  - [x] `docker-compose.yml` с Postgres 16
  - [x] `alembic init` + пустая initial migration
  - [x] ruff + mypy конфиг
  - [x] pytest скелет + 1 smoke-тест
  - [x] `.env.example`, `README.md` с quickstart
- **Exit:** `docker-compose up -d && uv run pytest` — зелёно ✅

### M1 — Core types & state (1 день) ✅

- **Цель:** базовые типы данных и LangGraph-совместимый state
- **Зависимости:** M0
- **Задачи:**
  - [x] `core/types.py`: `Message`, `ToolCall`, `Phase`, `HumanRole`, `AgentRole` (Pydantic)
  - [x] `core/state.py`: `AgentState`, `SharedState`, `GraphState` (TypedDict), `merge_agent_states` reducer
  - [x] `core/errors.py`: `BudgetExceededError`, `PhaseError`, `ToolError`
  - [x] unit-тесты на reducer (idempotent, commutative, handles empty)
- **Exit:** reducer корректно мёрджит state от 2+ агентов, тесты зелёные ✅

### M2 — LLM layer (2 дня) ✅

- **Цель:** унифицированный `LLMWrapper` с usage/cost/budget/retry
- **Зависимости:** M1
- **Задачи:**
  - [x] `llm/pricing.py` + `conf/pricing.yaml` (per-model стоимости)
  - [x] `llm/budget.py`: `BudgetTracker` (three-tier), события в `budget_events`
  - [x] `llm/wrapper.py`: `LLMWrapper.ainvoke(messages, tools, **opts) -> LLMResponse`
    - retry с exponential backoff на 429/5xx
    - usage tracking (in/out/cache tokens)
    - cost calculation
    - prompt caching (cache_control для Anthropic)
  - [x] `llm/providers/openai.py` (wrapper вокруг `init_chat_model("openai:gpt-4o")`)
  - [x] `llm/providers/anthropic.py`
  - [x] `llm/providers/vllm.py` (для локальных через OpenAI-compatible endpoint)
  - [x] `FakeLLM` (контракт — arch.md §4.4):
    - режимы: `scripted` (YAML-fixture `(role, step_idx) → response`), `replay` (из llm_calls.parquet), `echo`
    - **streaming не поддерживается** (`astream` → `NotImplementedError`)
    - tool_calls сценируются в fixture; cost/tokens эмулируются
    - fixture-файлы в `tests/fixtures/llm/<test_name>.yaml`
  - [x] Unit-тесты: usage tracking, budget cutoff, retry, FakeLLM определённость
- **Exit:** `LLMWrapper("openai:gpt-4o-mini").ainvoke(...)` возвращает ответ + корректный cost; budget-exceed останавливает; FakeLLM даёт битово идентичные ответы на идентичной fixture+seed ✅

### M3 — Storage & Observability (2 дня) ∥ частично с M2 ✅

- **Цель:** запись всех данных эксперимента
- **Зависимости:** M0 (PG), M1
- **Задачи:**
  - [x] `storage/models.py`: SQLAlchemy-модели всех таблиц (см. §5.1)
  - [x] Alembic migration для всех таблиц
  - [x] `storage/session.py`: async engine, session factory
  - [x] `storage/parquet_writer.py`: write_llm_call, write_message, write_tool_call, write_scratchpad, write_phase — с in-memory буфером и `flush()` (инварианты flush — arch.md §11.2, §17/#2)
  - [x] `observability/callbacks.py`: `ExperimentCallbackHandler(AsyncCallbackHandler)` — on_llm_start/end/error, on_tool_start/end, on_chain_start/end, on_custom_event → Postgres + Parquet (async); sync-flush на границах run-а и transitions
  - [x] `storage/checkpointer.py`: обёртка `langgraph-checkpoint-postgres` с **отдельным** async-engine (autocommit=True); бизнес-БД идёт через отдельный SQLAlchemy engine (two pools — arch.md §11.3, §17/#3)
  - [x] Unit-тесты writer-ов + integration-тест: smoke-run фиктивного графа → проверка, что всё записалось
- **Exit:** 1 fake-run создаёт корректные строки в PG + parquet-файлы читаются pandas-ом ✅

### M4 — Tools layer (2 дня) ∥ с M3 ✅

- **Цель:** рабочий набор инструментов, включая безопасный sandbox для кода
- **Зависимости:** M1
- **Задачи:**
  - [x] `tools/base.py`: `Tool` Protocol (`name`, `schema`, `ainvoke`), `ToolRegistry`
  - [x] Global: `calculator`, `duckduckgo_search`, `url_fetch`, `file_read`
  - [x] `tools/sandbox/base.py`: `CodeSandbox` Protocol (`execute(lang, code, files, timeout) -> ExecResult`)
  - [x] `tools/sandbox/docker_sandbox.py` (hardening — arch.md §5.2, §17/#7):
    - префетч базовых образов (python:3.11-slim, node:20-slim)
    - tmpfs для рабочей директории + read-only rootfs
    - `cap_drop: [ALL]`, `security_opt: ["no-new-privileges", "seccomp=<custom-profile>"]` (блок ptrace/mount/unshare/keyctl/bpf)
    - resource limits (cpu, mem, pids, network=none)
    - опциональный rootless-режим (для prod)
    - cleanup контейнеров по timeout
  - [x] `tools/sandbox/subprocess_sandbox.py` — для dev-режима, без изоляции
  - [x] Local tools:
    - Executor: `code_run` (через sandbox), `file_write`
    - Critic: `test_run` (sandbox), `diff`, `lint` (ruff/pylint subprocess)
    - Researcher: `semantic_search` (stub на FAISS + локальный corpus)
    - Planner: `todo_write`, `plan_update` (пишут в shared state)
  - [x] Integration-тесты: "напиши hello world" через `code_run` → ожидаемый stdout
  - [x] Тест на изоляцию: попытка `rm -rf /` в песочнице → не ломает host
- **Exit:** `DockerSandbox.execute("python", "print(1+1)")` возвращает "2", попытка вредоносного кода не даёт эффекта на host ✅

### M5 — Agent framework + scratchpad policy C (2 дня) ✅

- **Цель:** готовый `Agent` класс, оборачиваемый в LangGraph-ноду
- **Зависимости:** M2, M4
- **Задачи:**
  - [x] `agents/base.py`:
    - `Agent(role, system_prompt, llm, tools, cfg)`
    - `async def step(state: GraphState) -> GraphState` — LangGraph-совместимая сигнатура
    - scratchpad policy C:
      - пишет reasoning в `AgentState.scratchpad` всегда
      - `_build_prompt(state)`: берёт последние `window_size` своих шагов + весь inbox
      - при превышении context-window: зовёт `summarizer_model` для сжатия старых шагов → `summary_before_window`
    - tool-calling loop (до N tool-calls подряд, потом final answer)
    - усечение по бюджету
  - [x] 5 конкретных классов (`Planner`, `Researcher`, `Executor`, `Critic`, `Debater`) с базовыми промптами в `conf/agents/*.yaml`
  - [x] `Debater` принимает параметр `stance: "pro" | "contra"`
  - [x] Unit-тесты:
    - Agent с FakeLLM делает корректный tool-call
    - scratchpad window: на 5-й итерации в промпте только последние 3 шага
    - summarizer вызывается при превышении лимита
- **Exit:** `Executor` успешно пишет простую функцию и вызывает `code_run`, scratchpad растёт, window работает ✅

### M6 — Topology framework + Star + Chain + end-to-end (3 дня) ✅

- **Цель:** первый рабочий end-to-end на одной задаче
- **Зависимости:** M5, M3
- **Задачи:**
  - [x] `topology/base.py`: абстрактный `Topology`, `TopologyRegistry`
  - [x] `topology/star.py`: Coord-центрированный граф, conditional edges
  - [x] `topology/chain.py`: линейный граф с retry-loop
  - [x] `experiment/runner.py` (минимальная версия): load config → build agents → build topology → run → записать результаты
  - [x] `experiment/cli.py`: `atm run --config conf/experiments/smoke.yaml`
  - [x] Integration: full-run HumanEval-подобной задачи ("напиши fibonacci") через Star и через Chain
  - [x] Проверка: все LLM-calls в parquet, сумма costs корректна, финальный ответ в `runs.quality_score`
- **Exit:** `atm run --config smoke.yaml` — Star и Chain оба успешно проходят задачу, данные в PG + parquet ✅

### M7 — Mesh + Debate + Hierarchical (3 дня) ∥ ✅

- **Цель:** ещё 3 топологии
- **Зависимости:** M6
- **Задачи:**
  - [x] `topology/mesh.py`: broadcast-bus, round-robin, `max_rounds` / `consensus_threshold`
  - [x] `topology/debate.py`: 2 Debater-а параллельно → Critic-judge
  - [x] `topology/hierarchical.py`: subgraph'ы через LangGraph, 2 уровня
  - [x] Integration-тесты: каждая топология на одной задаче
- **Exit:** все 5 статических топологий работают end-to-end ✅

### M8 — Adaptive topology (L2): PhaseManager + TopologyRouter + Guards + SignalBus (5–6 дней)

- **Цель:** runtime-переключение топологий внутри и между фазами (архитектура L2, см. `arch.md §7.7, §8, §8bis`). Сердце RQ2.
- **Зависимости:** M7 (все 5 статических топологий готовы)
- **Декомпозиция (реализуется в порядке):**

  **M8.1 — Core reducers & types (0.5 дня)**
  - [ ] `core/types.py`: `TopologyTransition`, `TopologyDecision`, `PhaseDecision` (Pydantic, frozen)
  - [ ] `core/state.py`: расширить `SharedState` (signals, iter_total, phase_started_at_iter, topology_started_at_iter, topology_switch_count, topology_history)
  - [ ] `core/reducers.py`: `dedup_by_id_reducer(key, sort_by)` + unit-тесты на инварианты (idempotent, associative, empty-neutral)
  - [ ] Alembic migration: таблица `topology_transitions`
  - [ ] SQLAlchemy model: `TopologyTransition`

  **M8.2 — PhaseManager (Monotonic FSM) (0.5 дня)**
  - [ ] `phases/manager.py`: `RuleBasedPhaseRouter` (guards: `ready_for_execution`, `ready_for_verification`, `critic_approved`, iter caps)
  - [ ] `phases/manager.py`: `LLMPhaseRouter` с fallback на Rule (парсит JSON, валидирует монотонность)
  - [ ] Unit-тесты: попытка rollback возвращает фallback; все guard'ы покрыты

  **M8.3 — TopologyRouter (3 режима) (1.5 дня)**
  - [ ] `phases/topology_router.py`: `Protocol TopologyRouter`
  - [ ] `RuleBasedTopologyRouter`: таблица (phase × signals) → topology из `arch.md §7.7`
  - [ ] `LLMTopologyRouter`: prompt-template + Pydantic-валидация + router_cost_usd bookkeeping
  - [ ] `OracleTopologyRouter`: читает `oracle_table.json` по task_id или task_type (см. M8.7)
  - [ ] Unit-тесты с FakeLLM: каждый режим — детерминированный выбор на фиксированном state

  **M8.4 — SwitchGuards (0.5 дня)**
  - [ ] `phases/guards.py`: `SwitchGuards` + `GuardedRouter` декоратор
  - [ ] Реализовать все 4 guard'а (min_dwell, cooldown, max_per_run, max_per_phase)
  - [ ] Unit-тесты: каждый guard блокирует как ожидается, `considered_alternatives` сохраняется

  **M8.5 — SignalBus & emission helpers (0.5 дня)**
  - [ ] `phases/signals.py`: конвенции ключей (`stuck`, `rejected_count`, `needs_debate`, `ready_for_*`)
  - [ ] helper `emit_signal(state, key, value)` для агентов — обновляет state + `dispatch_custom_event("signal_emit", ...)`
  - [ ] Обновить Critic: эмитит `rejected_count` инкремент и `critic_approved` при approve
  - [ ] Обновить Executor: эмитит `stuck=True` после N неудачных code_run, `ready_for_verification` после первого успеха
  - [ ] Обновить Planner: эмитит `ready_for_execution` при финализации плана

  **M8.6 — TransitionGate + Adaptive meta-graph (1 день)**
  - [x] `topology/adaptive.py`: meta-граф с PhaseRouter, TopologyRouter, TransitionGate, 5 subgraph-узлов
  - [x] `TransitionGate` как чистая функция — реализует state-transfer таблицу из `arch.md §7.7`
  - [x] Dispatch событий: `phase_transition` (только при advance), `topology_transition` (каждый тик)
  - [x] Subgraph-узлы оборачивают компилированные графы из M6/M7 (`dispatch_topology` с lazy cache)
  - [x] Superset agent roster: все 7 ролей инстанциируются при topology.name=='adaptive'
  - [x] Integration-тест (FakeLLM): задача, где сценарий заставляет switch-и (`stuck` → mesh); проверить:
    - последовательность `topology_transitions` соответствует ожиданиям
    - `phase` монотонна
    - `messages` без дублей (dedup reducer работает)
    - guards срабатывают на ожидаемых сценариях (guard_override)
  - [x] `conf/experiments/adaptive_smoke.yaml` — конфиг smoke-теста

  **M8.7 — Oracle labels pipeline (0.5 дня) — ОБЯЗАТЕЛЬНО до M12**
  - [ ] `analysis/oracle.py`: `build_leave_one_out_oracle(exp_id) → OracleTable` — читает runs из E1, агрегирует по task_type без target task
  - [ ] `conf/oracle/type_level_manual.yaml`: 4 task_type × 3 phase = 12 клеток ручной разметки (заготовка с TODO на заполнение)
  - [ ] Unit-тест: `OracleTopologyRouter` с known table → stable decisions
  - **Note (audit G9):** ранее помечено как «условно после E1» — оракл-таблица нужна для E3, который запускается из M12, поэтому **выполнить до M12**; ручную type_level разметку заполнить параллельно с E1.

- **Exit criteria M8:**
  - `atm run --config conf/experiments/adaptive_smoke.yaml` — прогоняет 1 задачу через Adaptive, делает ≥1 реальный topology switch внутри execution-фазы, все transitions в PG и parquet
  - `SELECT COUNT(*) FROM topology_transitions WHERE decided_by='guard_override'` > 0 в тесте со специально сконструированной thrashing-провокацией
  - monotonicity invariant: `SELECT phase FROM phases ORDER BY at` — строго возрастает
  - `messages` после run'а без дубликатов (тест на `SELECT COUNT(*) = COUNT(DISTINCT message_id)`)

### M9 — Human Gateway + LLM-simulator (2 дня)

- **Цель:** HITL через внешний gateway, первая реализация — LLM-симулятор
- **Зависимости:** M6, M3 (checkpointer)
- **Задачи:**
  - [ ] `human/gateway.py`:
    - `class HumanGateway(Protocol): async def request(ctx, *, request_id) -> HumanResponse` с идемпотентностью по `(run_id, request_id)` (arch.md §9.1)
    - `HumanContext` / `HumanResponse` модели (поля `timed_out`, `source` — arch.md §3.1)
  - [ ] `human/llm_simulated.py`: LLM с системным промптом "ты — {role}, оцениваешь/руководишь/критикуешь..."
  - [ ] `human/cli_gateway.py` (минимальный rich-prompt, для отладки)
  - [ ] **Timeout + fallback-policy** (arch.md §9.1):
    - `asyncio.wait_for` с `deadline_s = HumanCfg.timeout_s` (default 900)
    - policies: `fail` / `llm_fallback` / `skip` — обрабатывается **топологией**, не gateway-ем
    - при timeout `HumanResponse(timed_out=True, source='timeout', action='timeout')`; `llm_fallback` → повторный вызов через LLMSimulatedGateway с `source='fallback'`
  - [ ] Интеграция в топологии: `interrupt()` в местах human-участия, затем resume из checkpointer; обработка timeout-ответа по `timeout_policy`
  - [ ] 5 ролей (Coordinator, Reviewer, Judge, Peer, Monitor) — системные промпты + правила активации в каждой топологии
  - [ ] Integration-тесты:
    - run с Reviewer через LLMSimulatedGateway — interaction пишется в `human_interactions`
    - timeout-тест: CLIGateway + deadline 1s + policy='llm_fallback' → run завершается успешно с `source='fallback'`
    - idempotency-тест: resume после interrupt не даёт дубль-записи в `human_interactions`
- **Exit:** Chain + human-as-Reviewer работает через LLM-симулятора; timeout+fallback отрабатывает; одна запись в `human_interactions` на один логический interrupt

### M9.1 — HITL integration в остальные топологии (3–4 дня)

> M9 поставил HITL-инфраструктуру и интеграцию **только в Chain** — это reference-implementation проверяющий контракт end-to-end. Расширение в остальные 5 топологий вынесено отдельно, потому что каждая требует собственного дизайн-решения о точке вставки human-узла, конфликте с Coordinator/Judge и правилах активации.

- **Цель:** подключить `human_reviewer` к Star/Mesh/Debate/Hierarchical/Adaptive с per-topology дизайн-решениями. Инфраструктура (Protocol, gateways, callback writer, timeout/fallback, idempotency, `human_gateway_llm` в Runner) уже готова в M9 — задача только в graph-wiring и unit/integration тестах.
- **Зависимости:** M9 (инфраструктура), M7 (статические топологии), M8 (Adaptive meta-graph)
- **Задачи:**
  - [ ] **Star + HITL** (0.5 дня): human_reviewer после Coordinator-decision, перед next-step dispatch. Policy: human-override > coordinator. Конфиг: `human_role=Coordinator` или `Reviewer`.
  - [ ] **Mesh + HITL** (0.5 дня): human как Peer-bus-participant; активируется на N-м round'е или по signal `consensus_pending`. Round-robin honor.
  - [ ] **Debate + HITL** (0.5 дня): human как Judge (заменяет/дополняет Critic-judge). Конфиг выбора: `judge=critic|human|both`.
  - [ ] **Hierarchical + HITL** (0.5–1 день): human на уровне Top-Coordinator (default) или per-sub-team (config-driven). Subgraph-HITL-test (важно — checkpointer + subgraph interrupt — flag в M9 рисках).
  - [ ] **Adaptive + HITL** (1 день): human как **side-input** для TopologyRouter (advisory) и/или для TransitionGate (override). Default: advisory; override — через флаг `human_can_override_router=true`.
    - **Priority specification (audit G4):** при наличии human_override решение принимается в порядке `human_override → guard → router`. Если все три согласны — `decided_by='router'`. Если guard блокирует router и human согласен с router — `decided_by='guard_override'`. Если human override включён и router/guard расходятся с human — `decided_by='human_override'`. Это значение пишется в `TopologyTransition.decided_by` для RQ2-анализа.
    - **Subgraph-interrupt тест:** interrupt() инициируется внутри активного subgraph-узла (НЕ из parent meta-graph). Resume через checkpointer корректно возвращается в тот же subgraph; ровно ОДНА запись в `human_interactions` после resume.
  - [ ] Расширить `human/__init__.py` re-export'ы (если потребуются helper'ы для конкретных топологий).
  - [ ] **Per-topology integration тесты** (1 файл/топология; маркер `@pytest.mark.requires_postgres`).
- **Exit:**
  - Все 5 топологий имеют HITL-вариант с тестами end-to-end (FakeLLM gateway, PG required).
  - Default-режим (без `human_cfg`) для каждой топологии — байт-в-байт идентичен поведению до M9.1.
  - `runs.human_role` корректно заполняется для каждой топологии.
  - **Adaptive+HITL (audit G4):** subgraph-level interrupt-resume test проходит — ровно 1 строка в `human_interactions`, корректная атрибуция `TopologyTransition.decided_by` в одном из трёх режимов (router / guard_override / human_override).

### M9.2 — Adaptive Role Router для RQ4 (2 дня)

> **Audit G3.** experiment_plan.md §5 (E4) требует 3 стратегии динамической смены HumanRole по фазе: `role_fixed_best`, `role_by_phase_rule`, `role_by_phase_llm`. Без этого компонента RQ4 — центральный вопрос диплома — **неотвечаем**. Инфраструктура M9 даёт только фиксированную роль.

- **Цель:** Router выбирает активную `HumanRole` по фазе/контексту; ablation `fixed | rule | llm` для E4.
- **Зависимости:** M9 (HumanGateway), M8 (PhaseManager FSM)
- **Задачи:**
  - [ ] `human/role_router.py`: `HumanRoleRouter` Protocol — `async def decide(phase: Phase, state: SharedState) -> HumanRole`.
  - [ ] `FixedRoleRouter(role: HumanRole)` — back-compat для M9/M9.1 (всегда возвращает фиксированную роль).
  - [ ] `RuleBasedRoleRouter` — таблица `phase → HumanRole`. Дефолт: `planning→Coordinator, execution→Peer, verification→Reviewer`. Конфиг через `conf/human/role_table.yaml`.
  - [ ] `LLMRoleRouter` — LLM получает state+phase, возвращает роль (Pydantic-валидация выхода). Аналогично `LLMPhaseRouter` (M8.2) — с fallback на rule.
  - [ ] Расширить `HumanCfg`: добавить `role_router: Literal["fixed", "rule", "llm"] = "fixed"`; `role_table: dict | None` (для rule); `role_router_model: str | None`.
  - [ ] Обновить топологии M9.1, чтобы перед запросом human узел спрашивал у router'а активную роль (если `role_router != "fixed"`).
  - [ ] `evaluation/metrics.py::human_sim_cognitive_load_proxy(run_id) → float` — формула из experiment_plan.md §5: `α·count(interrupts) + β·mean(context_len) + γ·mean(latency_s)`. Веса α/β/γ — в `conf/evaluation/cognitive_load.yaml`.
  - [ ] Unit-тесты: каждый router → детерминированный выбор на фиксированном state; LLMRoleRouter с FakeLLM-fixture даёт стабильное решение; cognitive_load_proxy на известном run'е даёт ожидаемое число.
- **Exit:**
  - Все три стратегии работают end-to-end в Chain (минимум — расширение опционально для остальных топологий).
  - `runs.human_role` отражает финальную роль; для динамических router'ов — последнюю активную в фазе.
  - `human_sim_cognitive_load_proxy` рассчитывается и записывается в `runs` (новая колонка `cognitive_load_proxy DOUBLE PRECISION`).

### M10 — Tasks & datasets (2 дня) ∥ с M9

- **Цель:** набор задач для эксперимента (4 типа) — синхронизировано с experiment_plan.md §0.
- **Зависимости:** M5
- **Audit G1 — task-mix синхронизация:** experiment_plan.md §0 определяет набор {**HumanEval** (programming), **GSM8K** (reasoning), **CommonGen** (creative), **InfiAgent-DABench** (decision)}. MMLU-Pro в эксперименте не используется. `TaskSpec.type` должен поддерживать значения `programming | reasoning | creative | decision` (расширить enum из arch.md §3.1, где `"qa"` → переименовать в `"reasoning"`).
- **Задачи:**
  - [ ] `tasks/base.py`: `TaskSpec(id, type, input, expected?, evaluator_key)`; `type: Literal["programming", "reasoning", "creative", "decision"]`.
  - [ ] `tasks/humaneval.py`: HuggingFace `openai_humaneval`, evaluator = прогон pytest в sandbox (`type="programming"`).
  - [ ] `tasks/gsm8k.py`: HuggingFace `gsm8k`, evaluator = numeric exact match (`type="reasoning"`). **Заменяет старый план `tasks/mmlu.py`**.
  - [ ] `tasks/commongen.py`: HuggingFace `common_gen`, evaluator = ROUGE-L + concept coverage check (`type="creative"`). **Заменяет старый план `tasks/creative.py` с ручными промптами**.
  - [ ] `tasks/dabench.py`: InfiAgent-DABench (data-analysis decision tasks), evaluator = LLM-judge + structural check на корректность выбора метода (`type="decision"`). **Заменяет старый план `tasks/analysis.py`**.
  - [ ] (Опционально) `tasks/mmlu.py` — оставить как дополнительный датасет для ablation, но НЕ в основном E1.
  - [ ] Parquet-кеш датасетов (`data/datasets/<name>.parquet`).
  - [ ] Тесты: каждая task загружается, evaluator на известных примерах даёт ожидаемые результаты.
- **Exit:** `TaskRegistry.get("humaneval").sample(10)`, `TaskRegistry.get("gsm8k").sample(10)`, `TaskRegistry.get("commongen").sample(10)`, `TaskRegistry.get("dabench").sample(10)` — все возвращают задачи, evaluator на pinned-примере даёт correct verdict.

### M11 — Evaluation framework + audit catch-ups (3 дня)

> Расширен катч-ап задачами из audit (G2/G6/G7/G8/G10), которые должны были быть в M0/M3/M4 но не попали туда. M11 — последний удобный milestone до E1, чтобы их закрыть.

- **Цель:** полноценный расчёт метрик для RQ1–RQ4 + закрытие audit-долгов до запуска грид-экспериментов.
- **Зависимости:** M10
- **Задачи (M11 core — evaluation):**
  - [ ] `evaluation/metrics.py`: функции per-metric (quality, efficiency, time, human; `human_sim_cognitive_load_proxy` см. M9.2).
  - [ ] `evaluation/judges.py`: LLM-judge с rubric, pairwise-сравнения, self-consistency.
  - [ ] `evaluation/ground_truth.py`: test-runners для programming (HumanEval pytest), numeric matcher (GSM8K), ROUGE+coverage (CommonGen), DABench structural check.
  - [ ] `evaluation/tlx.py`: NASA-TLX опросник (6 шкал), агрегация в `raw_tlx_score`.
  - [ ] Aggregator: post-run обновляет `runs.quality_score` и через `EvaluatorRegistry` (см. ниже G8) находит правильный evaluator по `TaskSpec.evaluator_key`.
  - [ ] Тесты на известных примерах (правильный код → quality=1.0).

- **Audit catch-up задачи (CRITICAL/IMPORTANT — закрыть до M12/E1):**
  - [ ] **G8 — `EvaluatorRegistry`** (`evaluation/registry.py`): аналогично `TaskRegistry`. Регистрирует `HumanEvalRunner`, `GSM8KMatcher`, `CommonGenJudge`, `DABenchEvaluator`. `EvaluatorRegistry.get(key) → Evaluator`. Обновить M10 task-файлы, чтобы при импорте каждый регистрировал свой evaluator (или сделать центральный bootstrap в M11). Без этого `TaskSpec.evaluator_key` не матчится и aggregator падает в hardcoded if/elif.
  - [ ] **G6 — `raw_tlx_score` колонка**: Alembic migration `0003_human_interactions_raw_tlx.py` → добавить `raw_tlx_score DOUBLE PRECISION NULL` в `human_interactions`. `evaluation/tlx.py` aggregator после `NasaTLX.raw_score` → `UPDATE human_interactions SET raw_tlx_score=... WHERE id=...`. Это нужно для M13 box-plot'ов и PG-фильтра E5.
  - [ ] **G2 — Reproducibility bundle**: добавить в `experiment/runner.py` (расширение текущего runner'а M6):
    - `seed_all(seed)` хелпер (зерно для random, numpy, torch если есть);
    - `experiments.git_sha`: `subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()` при старте `atm grid`/`atm run`;
    - `runs.model_version_snapshot`: первый ответ LLMWrapper → парсить `model_version` (или `system_fingerprint` у OpenAI) → записать;
    - `runs.sandbox_image_digest`: при `DockerSandbox.start()` → `docker inspect <image> -f '{{.Id}}'` → записать;
    - `structlog` init с `filter_secrets` процессором (arch.md §14.5); подключить в M0-bootstrap (импортный side-effect через `atm/__init__.py`).
  - [ ] **G7 — `conf/tools_policy.yaml`**: создать файл по спеке arch.md §5.3 (global + per_role tools); рефактор `ToolRegistry.tools_for(role)` чтобы читать из этого конфига, а не из хардкода. Расширить структуру §4 PLAN.md (минор G12 заодно).
  - [ ] **G10 — CI/CD pipeline**: `.github/workflows/ci.yml` с тремя jobs:
    - `lint`: `uv run ruff check && uv run mypy src`
    - `unit`: `uv run pytest tests/unit/ -m "not requires_postgres and not live"`
    - `integration`: `uv run pytest tests/integration/ -m requires_postgres` с postgres service в job (alembic upgrade head в setup).
    - Документировать в `README.md` или `dev/ci.md` как гонять каждый тип локально + флаг `--run-live` для `@pytest.mark.live`.

- **Exit:**
  - Все метрики в `runs` после run'а заполнены (`quality_score`, `cognitive_load_proxy`, `wall_time_s`, `budget_spent_usd`).
  - `EvaluatorRegistry.get(task.evaluator_key)` возвращает корректный evaluator для всех 4 task-типов из M10.
  - `human_interactions.raw_tlx_score` — заполнен в тестовом run'е с симулированным TLX.
  - `atm run --seed 42` — два прогона с тем же seed дают идентичные `messages` (FakeLLM + deterministic seed) и идентичный `quality_score`.
  - `experiments.git_sha`, `runs.model_version_snapshot`, `runs.sandbox_image_digest` — все заполняются в smoke-run'е.
  - `conf/tools_policy.yaml` существует; `ToolRegistry` читает оттуда.
  - GitHub Actions: PR triggers all 3 jobs, все зелёные на main.

### M12 — Experiment runner + grid + budget dry-run (2 дня)

- **Цель:** полноценный запуск грида
- **Зависимости:** M6 (runner скелет), M11
- **Задачи:**
  - [ ] `experiment/config.py`: полная Pydantic-схема
  - [ ] `experiment/loader.py`: OmegaConf (композиция + override из CLI) → Pydantic валидация
  - [ ] `experiment/grid.py`:
    - парсит "sweep"-секцию конфига (`topology: [star, chain, mesh]`) в декартово произведение
    - `ProcessPoolExecutor` для параллельных runs
    - agg progress через PG (запись в `experiments.status`)
  - [ ] Внутри run'а — asyncio для parallel agents (Debate, Mesh broadcast)
  - [ ] Dry-run: пройти по grid, для каждой задачи оценить средние токены (из preceding runs или heuristic), выдать ожидаемую стоимость
  - [ ] CLI commands: `atm run`, `atm grid`, `atm estimate`, `atm status`
  - [ ] **G5 — `atm resume` + `atm replay` + reconcile**:
    - `atm resume --run-id <uuid> [--force]` — продолжает run из последнего checkpoint'а LangGraph PG checkpointer. Критично для HITL-экспериментов E2/E5 (живой человек ответил, но процесс упал).
    - `atm replay <run_id> [--mode deterministic|semantic]` — детерминированный реплей: FakeLLM в режиме `replay` читает из `llm_calls.parquet` исходного run'а; новый run пишется как `replay_of=<original_run_id>`. Использует `runs.model_version_snapshot` для верификации pin'а.
    - Reconcile в `atm grid`: при старте проверять `SELECT id FROM runs WHERE status='running' AND exp_id=...`; если процесс не жив (pid недоступен) → сбросить в `status='failed'` или allow `--force-resume`.
  - [ ] Integration-тест: mini-grid (2 топологии × 2 задачи × 1 seed), без реальных LLM (FakeLLM).
  - [ ] Integration-тест G5: запустить run, прервать на середине через SIGKILL → `atm resume --run-id ...` доводит до конца, итоговое quality_score не меняется при FakeLLM-режиме.
- **Exit:**
  - `atm grid --config exp1.yaml` запускает параллельно 4 runs, все пишутся в PG.
  - `atm resume --run-id <uuid>` доводит killed-run до END.
  - `atm replay <run_id> --mode deterministic` даёт битово идентичный финальный state.
  - Reconcile при перезапуске grid'а не оставляет zombie `status='running'` строк.

### M13 — Analysis tooling (1.5 дня)

- **Цель:** jupyter-шаблоны для анализа и графиков для диплома; RQ2-специфичный анализ adaptive поведения.
- **Зависимости:** M12, M8.7 (oracle pipeline — обязательно)
- **Задачи:**
  - [ ] `analysis/loaders.py`: `load_experiment(exp_id)`, `load_llm_calls()`, `load_runs()` → pandas.
  - [ ] **G11 — `load_topology_transitions(exp_id) → pd.DataFrame`** — RQ2-loader для adaptive runs. Поля: `run_id, at_iter, from_topology, to_topology, decided_by, signals_snapshot, router_cost_usd`. Без него аналитический ноутбук E3 пишется на лету.
  - [ ] `analysis/plots.py`:
    - quality vs cost (Pareto) с confidence bands для adaptive vs static.
    - heatmap topology × task_type (RQ1).
    - phase-transition timelines для Adaptive.
    - **G11 — topology-transition timeline × quality correlation** (RQ2-специфичный plot).
    - **G11 — derived-метрики plots**: `guard_override_rate`, `router_cost_share`, `time_per_topology`, `oracle_gap_loo` (формулы — experiment_plan.md §4).
    - human cognitive load boxplots (использует `human_interactions.raw_tlx_score` из M11 G6 + `runs.cognitive_load_proxy` из M9.2).
  - [ ] **G9 — `analysis/oracle.py` loader + plot**: `load_oracle_table()` + plot «leave-one-out oracle quality vs adaptive router quality» для RQ2 (показывает gap между «идеальным» оракулом и реальным router'ом).
  - [ ] Шаблонный notebook `notebooks/analysis_template.ipynb` с секциями RQ1/RQ2/RQ3/RQ4.
- **Exit:**
  - После grid'а одной ячейкой генерируется сравнительный Pareto-график.
  - RQ2-метрики (guard_override_rate, router_cost_share, oracle_gap_loo) считаются из `topology_transitions` без ручного SQL.
  - `load_topology_transitions(exp_id)` возвращает корректный DataFrame для E3-runs.

### M14+ (этап 2 диплома) — отдельно, после техники

- Streamlit UI для human-as-user (реальные люди)
- NASA-TLX сбор в UI
- Замена LLMSimulatedGateway на живого человека
- Proctoring-протокол для реальных user studies

## 9. Critical path и параллельные потоки

```
M0 → M1 → M2 ─┬─ M3 ─┐
              │       ├→ M5 → M6 → M7 → M8 → M12 → M13
              └─ M4 ──┘           ↘
                                    M9 (∥ M7/8)
                                    M10 (∥ M8/9)
                                    M11 (после M10)
```

- Критичный путь: M0→M1→M2→M5→M6→M7→M8→M8.7→M11→M12→M13 (M8.7 включён в крит-путь после audit G9)
- Параллелимо: M3/M4 после M1; M9 после M6; M10 после M5; M9.1 после M9 (∥ M10/M11); M9.2 после M9 (∥ M10)
- ~~M8.7 (Oracle pipeline) зависит от результатов E1-pilot~~ — **обязательно до M12** (audit G9); E3 без oracle-таблицы невозможен.
- M9.1 (HITL для 5 остальных топологий) — необязательно для критичного пути, но без него RQ-эксперименты с human-as-Reviewer ограничены Chain'ом. Рекомендуется выполнить до M12.
- M9.2 (AdaptiveRoleRouter) — **необходимо** для RQ4 (E4); без него центральный вопрос диплома неотвечаем.
- M11 расширен с 2 до 3 дней — включает audit catch-ups (EvaluatorRegistry, raw_tlx_score migration, reproducibility bundle, tools_policy.yaml, CI/CD).
- M12 включает `atm resume`/`atm replay`/reconcile (audit G5) — критично для HITL-экспериментов после краша.
- M13 расширен с 1 до 1.5 дней — включает topology_transitions loaders/plots для RQ2 (audit G9+G11).
- Оценка по человеко-дням: **~40 дней** на одного исполнителя (M8 — 5–6 дней L2; M9.1 — 3–4 дня; M9.2 — 2 дня; M11 — 3 дня; M12 — 3 дня с G5; M13 — 1.5 дня), можно ужать до ~28–30 с распараллеливанием.

## 10. Риски и их митигации

| Риск | Вероятность | Импакт | Митигация |
|---|---|---|---|
| Стоимость грида взрывается | высокая | высокий | dry-run estimator обязательно до запуска; three-tier budget; prompt caching; GPT-4o-mini как baseline, GPT-4o только для критичных |
| LangGraph checkpointer glitches при subgraph + interrupt | средняя | средний | ранний integration-тест HITL в M9, subgraph-HITL тест отдельно |
| Docker sandbox на сервере требует особой настройки | средняя | средний | fallback subprocess_sandbox для dev; инструкция по production-setup с пользователем `unprivileged`; rootless docker |
| Per-agent nested state ломает LangGraph reducer | низкая | высокий | прототип reducer в M1 на проверочных кейсах перед тем как писать остальное |
| LLM-judge не стабилен для creative tasks | высокая | средний | pairwise-сравнения + self-consistency (multi-run judge) + sanity-check на human subset |
| Adaptive router нестабилен | средняя | средний | rule-based как fallback; LLM-router — отдельный флаг конфига; ablation в грид-e |
| Rate limits OpenAI на больших гридах | высокая | низкий | async-семафор + process-level ограничение; batching через OpenAI Batch API для не-time-critical runs |

## 11. Open points (решим по ходу)

- ~~**Task mix в benchmark-датасете**~~ — **[Resolved 2026-05-12 audit G1: HumanEval + GSM8K + CommonGen + InfiAgent-DABench согласно experiment_plan.md §0; зафиксировано в M10]**
- **Протокол human studies** (сколько участников, какие задачи, IRB) — отдельно на этапе 2 диплома.
- **Adaptive router: rule-based vs LLM-based** — сравнить в ablation, выбрать по результатам.
- ~~Hierarchical — 2 vs 3 уровня~~ — **[Resolved: ровно 2 уровня, arch.md §7.6, §18/#5]**
- **LLM-judge: self-consistency + pairwise — как комбинировать** — pilot на M11.
- **Parquet row-group tuning** — профилирование на M13.
- **Streamlit vs Gradio** для human UI — посмотрим на M14.
- **Ray** для масштабного грида — включим если станет узким местом на M12.
- **Adaptive role router strategy** — `fixed | rule | llm` — ablation в E4 (см. M9.2).

## 11bis. Audit log (2026-05-12)

Аудит PLAN.md выявил 16 пробелов (4 CRITICAL + 7 IMPORTANT + 5 MINOR). Закрыты в этом обновлении:

| ID | Severity | Где исправлено |
|----|----------|----------------|
| G1 | CRITICAL | M10 — task-mix синхронизирован с experiment_plan.md §0 |
| G2 | CRITICAL | M11 catch-up — seed_all, git_sha, model_version_snapshot, sandbox_image_digest, structlog filter_secrets |
| G3 | CRITICAL | **Новый M9.2** — AdaptiveRoleRouter (3 стратегии для RQ4) |
| G4 | CRITICAL | M9.1 расширен — priority spec `human_override → guard → router`, subgraph-interrupt тест в Exit |
| G5 | IMPORTANT | M12 — `atm resume` + `atm replay` + reconcile |
| G6 | IMPORTANT | M11 catch-up — `raw_tlx_score` колонка + Alembic migration 0003 + aggregator UPDATE |
| G7 | IMPORTANT | M11 catch-up — `conf/tools_policy.yaml` + рефактор ToolRegistry |
| G8 | IMPORTANT | M11 catch-up — `EvaluatorRegistry` |
| G9 | IMPORTANT | M8.7 — `условно` снято, обязательно до M12; M13 — oracle plot loader |
| G10 | IMPORTANT | M11 catch-up — GitHub Actions CI с lint/unit/integration jobs |
| G11 | IMPORTANT | M13 — `load_topology_transitions` + RQ2-specific plots |
| G12–G16 | MINOR | Отложены, в текущем апдейте не правились |

Полный отчёт: `dev/audit-plan-gaps.md`.

## 12. Что дальше по этому плану

Сразу после утверждения:
1. **M0**: инициализируем репозиторий (`uv`, docker-compose, структура, миграция)
2. **M1**: пишем core types и state с reducer-тестами

Каждый milestone будем вести с чеклистом (task tracking), по завершении — короткий review против exit criteria.
