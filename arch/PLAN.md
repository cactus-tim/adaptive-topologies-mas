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
  - [ ] `topology/adaptive.py`: meta-граф с PhaseRouter, TopologyRouter, TransitionGate, 5 subgraph-узлов
  - [ ] `TransitionGate` как чистая функция — реализует state-transfer таблицу из `arch.md §7.7`
  - [ ] Dispatch событий: `phase_transition` (только при advance), `topology_transition` (каждый тик)
  - [ ] Subgraph-узлы оборачивают компилированные графы из M6/M7 (`node_for_topology(name)`)
  - [ ] Superset agent roster: все 7 ролей инстанциируются при topology.name=='adaptive'
  - [ ] Integration-тест (FakeLLM): задача, где сценарий заставляет switch-и (`stuck` → mesh, `rejected_count≥3` → debate); проверить:
    - последовательность `topology_transitions` соответствует ожиданиям
    - `phase` монотонна
    - `messages` без дублей (dedup reducer работает)
    - guards срабатывают на ожидаемых сценариях

  **M8.7 — Oracle labels pipeline (0.5 дня, условно после E1 pilot)**
  - [ ] `analysis/oracle.py`: `build_leave_one_out_oracle(exp_id) → OracleTable` — читает runs из E1, агрегирует по task_type без target task
  - [ ] `conf/oracle/type_level_manual.yaml`: 4 task_type × 3 phase = 12 клеток ручной разметки (заготовка с TODO на заполнение)
  - [ ] Unit-тест: `OracleTopologyRouter` с known table → stable decisions

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

### M10 — Tasks & datasets (2 дня) ∥ с M9

- **Цель:** набор задач для эксперимента (4 типа)
- **Зависимости:** M5
- **Задачи:**
  - [ ] `tasks/base.py`: `TaskSpec(id, type, input, expected?, evaluator)`
  - [ ] `tasks/humaneval.py`: HuggingFace `openai_humaneval`, evaluator = прогон pytest в sandbox
  - [ ] `tasks/mmlu.py`: MMLU-Pro подмножество, evaluator = exact match
  - [ ] `tasks/creative.py`: open-ended (5-10 ручных промптов), evaluator = LLM-judge по rubric
  - [ ] `tasks/analysis.py`: набор задач типа "найди паттерн в данных", evaluator = LLM-judge + structural check
  - [ ] Парsekt-кеш датасетов
  - [ ] Тесты: каждая task загружается, evaluator на известных примерах даёт ожидаемые результаты
- **Exit:** `TaskRegistry.get("humaneval").sample(10)` возвращает 10 задач, evaluator работает

### M11 — Evaluation framework (2 дня)

- **Цель:** полноценный расчёт метрик для RQ1–RQ4
- **Зависимости:** M10
- **Задачи:**
  - [ ] `evaluation/metrics.py`: функции per-metric (quality, efficiency, time, human)
  - [ ] `evaluation/judges.py`: LLM-judge с rubric, pairwise-сравнения, self-consistency
  - [ ] `evaluation/ground_truth.py`: test-runners для programming, matcher для MMLU
  - [ ] `evaluation/tlx.py`: NASA-TLX опросник (6 шкал), агрегация в `raw_tlx_score`
  - [ ] Aggregator: post-run обновляет `runs.quality_score`
  - [ ] Тесты на известных примерах (правильный код → quality=1.0)
- **Exit:** после run'а заполнены все метрики в `runs`

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
  - [ ] Integration-тест: mini-grid (2 топологии × 2 задачи × 1 seed), без реальных LLM (FakeLLM)
- **Exit:** `atm grid --config exp1.yaml` запускает параллельно 4 runs, все пишутся в PG

### M13 — Analysis tooling (1 день)

- **Цель:** jupyter-шаблоны для анализа и графиков для диплома
- **Зависимости:** M12
- **Задачи:**
  - [ ] `analysis/loaders.py`: `load_experiment(exp_id)`, `load_llm_calls()`, `load_runs()` → pandas
  - [ ] `analysis/plots.py`:
    - quality vs cost (Pareto)
    - heatmap topology × task_type
    - phase-transition timelines для Adaptive
    - human cognitive load boxplots
  - [ ] Шаблонный notebook `notebooks/analysis_template.ipynb`
- **Exit:** после grid-а можно в одну ячейку сгенерировать сравнительный график

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

- Критичный путь: M0→M1→M2→M5→M6→M7→M8→M12→M13
- Параллелимо: M3/M4 после M1; M9 после M6; M10 после M5
- M8.7 (Oracle pipeline) зависит от результатов E1-pilot — может выполняться параллельно с M9/M10 после первых confirmed E1-runs
- Оценка по человеко-дням: ~30 дней на одного исполнителя (M8 расширен до 5–6 дней после перехода на L2), можно ужать до ~22 с распараллеливанием

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

- **Task mix в benchmark-датасете** — конкретные задачи analysis/creative. Собираем после M10 в отдельной итерации.
- **Протокол human studies** (сколько участников, какие задачи, IRB) — отдельно на этапе 2 диплома.
- **Adaptive router: rule-based vs LLM-based** — сравнить в ablation, выбрать по результатам.
- ~~Hierarchical — 2 vs 3 уровня~~ — **[Resolved: ровно 2 уровня, arch.md §7.6, §18/#5]**
- **LLM-judge: self-consistency + pairwise — как комбинировать** — pilot на M11.
- **Parquet row-group tuning** — профилирование на M13.
- **Streamlit vs Gradio** для human UI — посмотрим на M14.
- **Ray** для масштабного грида — включим если станет узким местом на M12.

## 12. Что дальше по этому плану

Сразу после утверждения:
1. **M0**: инициализируем репозиторий (`uv`, docker-compose, структура, миграция)
2. **M1**: пишем core types и state с reducer-тестами

Каждый milestone будем вести с чеклистом (task tracking), по завершении — короткий review против exit criteria.
