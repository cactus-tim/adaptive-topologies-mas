# Codebase Map
*Auto-generated. Last updated: 2026-04-23*

## Tech Stack
- **Language:** Python 3.11+
- **Package Manager:** uv
- **Orchestration:** LangGraph (multi-agent framework)
- **LLM SDK:** LangChain chat models (OpenAI/Anthropic/vLLM)
- **Database:** PostgreSQL 16 (async via SQLAlchemy 2.x + asyncpg, ORM with Alembic migrations)
- **Bulk Storage:** Apache Parquet (PyArrow) for experiment data
- **Config:** YAML + Pydantic + OmegaConf
- **Testing:** pytest + pytest-asyncio
- **Linting/Types:** ruff + mypy (strict)
- **Containerization:** Docker (code sandbox + dev environment)

## Project Structure
- `src/atm/` — main package (Adaptive Topologies MAS, imported as `atm`)
  - `core/` — base types, state, errors (Message, ToolCall, Phase, AgentState, GraphState)
  - `llm/` — LLMWrapper, providers (OpenAI/Anthropic/vLLM), budget tracking
  - `tools/` — Tool protocol, registry, global tools, Docker/subprocess sandbox
  - `agents/` — Agent base class, Planner/Researcher/Executor/Critic/Debater roles
  - `topology/` — Topology protocol, Star/Chain/Mesh/Debate/Hierarchical/Adaptive implementations
  - `phases/` — PhaseManager FSM, TopologyRouter, SwitchGuards, signals
  - `human/` — HumanGateway protocol, LLMSimulatedGateway, CLI gateway, Streamlit (M14+)
  - `storage/` — SQLAlchemy models, async session, ParquetWriter, checkpointer wrapper
  - `observability/` — ExperimentCallbackHandler (LangGraph async callbacks), tracer
  - `tasks/` — TaskSpec base, HumanEval/MMLU/Creative/Analysis implementations
  - `evaluation/` — LLM-as-judge, ground truth runners, metrics, NASA-TLX
  - `experiment/` — Config schemas (Pydantic), loader (OmegaConf), runner, grid executor, CLI (Typer)
  - `analysis/` — Loaders (read_experiment/llm_calls/runs), plots (Pareto/heatmaps/timelines)
- `tests/` — unit (smoke test), integration (fixtures organized by module)
- `conf/` — YAML configuration templates
  - `agents/` — agent set definitions (e.g., canonical_4.yaml)
  - `topology/` — topology configs (star.yaml, chain.yaml, etc.)
  - `task/` — task definitions (humaneval.yaml, mmlu.yaml, etc.)
  - `model/` — LLM provider configs (gpt4.yaml, claude.yaml, etc.)
  - `experiment/` — grid sweep definitions (topology/task/model combinations)
- `alembic/` — database migrations (async template, single initial migration)
- `notebooks/` — jupyter for post-experiment analysis
- `dev/` — documentation (PLAN.md, arch.md) and task tracking (done/active)

## Key Modules

### Core Types & State (`core/`)
- **Purpose:** Base data structures (Pydantic models + TypedDict) for message passing, errors, phase tracking; LangGraph-compatible reducers for merging agent state updates.
- **Exports:** `AgentRole`, `AgentState`, `AtmError`, `BudgetEvent`, `BudgetExceededError`, `GraphState`, `HumanContext`, `HumanResponse`, `HumanRole`, `LLMResponse`, `Message`, `MessageKind`, `Phase`, `PhaseError`, `PhaseTransition`, `RunResult`, `SharedState`, `TaskResult`, `TaskSpec`, `TokenUsage`, `ToolCall`, `ToolError`, `ToolResult`, `TopologyTransition`, `dedup_by_id_reducer`, `merge_agent_states`
- **Dependencies:** pydantic (validation), typing
- **Status:** M1 complete

### LLM Layer (`llm/`)
- **Purpose:** Unified wrapper over LangChain chat models with usage tracking, cost calculation, retry logic, prompt caching, and three-tier budget enforcement.
- **Exports (M2 planned):** `LLMWrapper`, `LLMResponse`, `BudgetTracker`, `FakeLLM`, provider classes (OpenAIProvider, AnthropicProvider, vLLMProvider)
- **Dependencies:** langchain-core, anthropic, openai, math (pricing)
- **Subdirs:** `providers/` — provider-specific implementations
- **Status:** M0 (skeleton), M2 not started

### Tools & Sandbox (`tools/`)
- **Purpose:** Tool registry protocol, global tools (calculator, search, file_read), isolated code execution via Docker or subprocess.
- **Exports (M4 planned):** `Tool` protocol, `ToolRegistry`, `CodeSandbox` protocol, `DockerSandbox`, `SubprocessSandbox`, global/local tool definitions
- **Dependencies:** docker (containers), pydantic, subprocess, asyncio
- **Subdirs:** `sandbox/` — sandbox implementations with hardening (resource limits, seccomp, tmpfs)
- **Status:** M0 (skeleton), M4 not started

### Agent Framework (`agents/`)
- **Purpose:** Base Agent class with LangGraph node signature (`step(state) -> state`), tool-calling loop, scratchpad policy C (windowing + summarization).
- **Exports (M5 planned):** `Agent` base, `Planner`, `Researcher`, `Executor`, `Critic`, `Debater` (with stance param)
- **Dependencies:** core.state, llm, tools, pydantic
- **Status:** M0 (skeleton), M5 not started

### Topology Framework (`topology/`)
- **Purpose:** Protocol-based topology definitions; each topology is a function returning a compiled LangGraph StateGraph; supports Star (coordinator-centric), Chain (linear with retry), Mesh (broadcast), Debate (pro/contra), Hierarchical (2-level subgraphs), Adaptive (meta-graph with phase/topology routers).
- **Exports (M6+ planned):** `Topology` protocol, `TopologyRegistry`, topology classes for each variant
- **Dependencies:** agents, core.state, langgraph, typing
- **Key Interfaces:** `topology.build(agents, cfg) -> CompiledStateGraph`
- **Status:** M0 (skeleton), M6 not started

### Phase Manager & Routers (`phases/`)
- **Purpose:** FSM for phase transitions (planning→execution→verification→done); rule-based and LLM-based routers; SwitchGuards prevent thrashing (min_dwell, cooldown, max_per_run, max_per_phase).
- **Exports (M8 planned):** `PhaseManager`, `RuleBasedPhaseRouter`, `LLMPhaseRouter`, `TopologyRouter` (rule/llm/oracle modes), `SwitchGuards`, signal emission helpers
- **Dependencies:** core.state, core.types, llm (for LLM router), pydantic
- **Status:** M0 (skeleton), M8 not started

### Human Gateway & HITL (`human/`)
- **Purpose:** Protocol for human interventions in agent workflows; implementations: LLM-simulated (for testing), CLI (for dev), Streamlit (for user studies M14+); handles timeouts, fallback policies, idempotency.
- **Exports (M9 planned):** `HumanGateway` protocol, `HumanContext`, `HumanResponse`, `LLMSimulatedGateway`, `CLIGateway`
- **Dependencies:** llm (for simulator), rich (CLI UI), async utilities
- **Status:** M0 (skeleton), M9 not started

### Storage & Observability (`storage/` + `observability/`)
- **Purpose:** SQLAlchemy async models (experiments, runs, phases, human_interactions, budget_events, topology_transitions), Parquet writer with async buffering, LangGraph callback handler, checkpointer wrapper (two separate async pools for isolation).
- **Exports (M3 planned):** `ExperimentCallbackHandler`, `ParquetWriter`, `Checkpointer`, SQLAlchemy session factory
- **Dependencies:** sqlalchemy, asyncpg, pyarrow, langgraph, pydantic
- **Status:** M0 (skeleton), M3 not started

### Tasks & Evaluation (`tasks/` + `evaluation/`)
- **Purpose:** Task definitions (TaskSpec protocol), concrete datasets (HumanEval, MMLU, Creative, Analysis), evaluators (LLM-as-judge, test runners), metrics aggregation, NASA-TLX survey.
- **Exports (M10-M11 planned):** `TaskSpec`, `TaskRegistry`, specific task loaders, evaluator functions, metric aggregators
- **Dependencies:** huggingface-hub (datasets), llm (judge), pydantic, pandas
- **Status:** M0 (skeleton), M10-M11 not started

### Experiment Runner & CLI (`experiment/`)
- **Purpose:** Config loading (OmegaConf → Pydantic validation), single-run executor, grid sweep with ProcessPoolExecutor, CLI entry point (Typer), dry-run budget estimator.
- **Exports (M12 planned):** `ExperimentConfig`, `RunConfig`, `ModelConfig`, `ConfigLoader`, `Runner`, `GridExecutor`, CLI commands (run, grid, estimate, status)
- **Dependencies:** omegaconf, pydantic, typer, asyncio, multiprocessing, click
- **Status:** M0 (skeleton), M12 not started

### Analysis & Plots (`analysis/`)
- **Purpose:** Data loaders for post-experiment analysis (load_experiment, load_llm_calls, load_runs returning pandas DataFrames), plot generators (Pareto, heatmaps, phase timelines, cognitive load boxplots).
- **Exports (M13 planned):** `load_experiment()`, `load_llm_calls()`, `load_runs()`, plot functions
- **Dependencies:** pandas, matplotlib, pyarrow (read parquet)
- **Status:** M0 (skeleton), M13 not started

## Patterns & Conventions

### State Management
- **LangGraph state:** TypedDict-based `GraphState` with shared fields + per-agent `AgentState` reducer for merging updates from parallel agents.
- **Scratchpad policy C:** Agents always write reasoning to `scratchpad`, but send to LLM only a sliding window (last K steps) + optional LLM-summarized older steps to stay within context limits.

### Error Handling
- **Budget cuts:** `BudgetExceededError` raised immediately when per-call/per-run/per-experiment limit exceeded; hard stop in runner.
- **Phase errors:** `PhaseError` for invalid state transitions; FSM prevents rollback (monotonic phases).
- **Tool errors:** `ToolError` wrapped from sandbox or API failures; retryable vs terminal categorized by agent logic.

### Configuration & Composition
- **YAML + OmegaConf:** Configs are YAML templates in `conf/`; `ConfigLoader` uses OmegaConf for variable interpolation and merges.
- **Pydantic validation:** All config values validated against schema (in `experiment/config.py`) before instantiation.
- **Registry pattern:** TopologyRegistry, ToolRegistry, TaskRegistry — dynamic lookup by name, simplifies CLI and config-driven runs.

### Dependency Injection
- **Factory functions:** `init_llm(provider, model, **opts)`, `init_sandbox(type, **opts)` — parametrized by config, injected into Agent/Topology constructors.
- **Protocol-based:** Agent doesn't know concrete Sandbox implementation; uses Protocol, allows swapping Docker↔Subprocess.

### Observability & Debugging
- **LangGraph callbacks:** `ExperimentCallbackHandler` hooks into on_llm_start/end, on_tool_start/end, on_chain_start/end → async write to Postgres + Parquet with buffering.
- **Event dispatch:** `dispatch_custom_event(type, **data)` for phase/topology transitions, signal emissions, guard activations.
- **Parquet + SQL:** Bulk data (LLM calls, messages, tool calls) in Parquet (fast analytics); metadata (run status, costs, timing) in Postgres (real-time queries, transactional integrity).

### Testing
- **FakeLLM:** Deterministic mock LLM — reads responses from YAML fixtures by (role, step_idx); used in unit tests to make runs reproducible.
- **Fixtures location:** `tests/fixtures/llm/<test_name>.yaml` — scripted LLM responses; isolated, no real API calls.
- **Smoke test:** Single integration test in M0 (test_atm_package_importable) — verifies package version.
- **Current unit tests (M1):** 138 tests covering errors hierarchy (20), types/enums (31), reducers with monoid invariants (24), state TypedDicts (24), public API (35), plus M0 smoke.

### Async Patterns
- **Agents & topologies:** All state updates are async; agents emit state updates; LangGraph compiles graph with async node functions.
- **Checkpointing:** Postgres checkpointer (separate async engine, autocommit=True) saves graph state at breaks (interrupts, phase transitions).
- **Semaphores:** Rate limiting on API calls (e.g., OpenAI rate limits) via asyncio.Semaphore; ProcessPoolExecutor for grid parallelism (separate from agent-level async).

## External Dependencies
- **langgraph:** Multi-agent orchestration graph primitives (nodes, edges, reducers, checkpointers).
- **langchain-core:** Chat model interface (`BaseChatModel`, `init_chat_model`), tool schema utilities.
- **sqlalchemy[asyncio]:** ORM with async engine (asyncpg driver), transaction control.
- **asyncpg:** PostgreSQL async client driver.
- **omegaconf:** Config composition (interpolation, merging, CLI override).
- **pydantic:** Data validation (all config schemas, message types).
- **pyarrow:** Parquet I/O for bulk experiment data.
- **pytest + pytest-asyncio:** Test framework with async fixture/node support.
- **ruff:** Fast Python linter/formatter.
- **mypy:** Static type checker (strict mode).
- **alembic:** Database migration tool (async template).
- **typer:** Modern CLI framework (used in M12 for experiment/grid commands).
- **anthropic, openai:** LLM provider SDKs (fallback/alternate to LangChain models in M2+).
- **huggingface-hub:** Dataset loading (HumanEval, MMLU in M10).
- **docker:** Container API for code sandbox (in M4).

## Test Setup
- **Framework:** pytest with pytest-asyncio (asyncio_mode="auto"), error-on-warning filter.
- **Structure:** `tests/unit/` (isolated, mocked), `tests/integration/` (with real DB in CI), `tests/fixtures/` (data: LLM responses, expected outputs).
- **Patterns:** FakeLLM fixtures (YAML-based scripted responses), in-memory Postgres for integration tests (future M3+), parametrized tests for topology variants.
- **Coverage target:** All core modules + critical paths (reducer correctness, budget enforcement, state transitions); lower coverage on M9+ until implementations exist.
