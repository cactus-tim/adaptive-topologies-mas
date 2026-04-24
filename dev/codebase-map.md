# Codebase Map
*Auto-generated. Last updated: 2026-04-24*

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
  - `llm/` — LLMWrapper, providers (OpenAI/Anthropic/vLLM), budget tracking, pricing, retry, fake LLM
  - `tools/` — Tool protocol, registry, global tools, Docker/subprocess sandbox (M4)
  - `agents/` — Agent base class, Planner/Researcher/Executor/Critic/Debater roles, scratchpad policy C (M5)
  - `topology/` — Topology protocol, Star/Chain/Mesh/Debate/Hierarchical/Adaptive (M6-M8)
  - `phases/` — PhaseManager FSM, TopologyRouter, SwitchGuards, signals (M8)
  - `human/` — HumanGateway protocol, LLMSimulatedGateway, CLI gateway (M9)
  - `storage/` — SQLAlchemy models, async session, ParquetWriter, checkpointer wrapper (M3 complete)
  - `observability/` — ExperimentCallbackHandler (LangGraph async callbacks), serializers (M3 complete)
  - `tasks/` — TaskSpec base, HumanEval/MMLU/Creative/Analysis implementations (M10)
  - `evaluation/` — LLM-as-judge, ground truth runners, metrics, NASA-TLX (M11)
  - `experiment/` — Config schemas (Pydantic), loader (OmegaConf), runner, grid executor, CLI (M12)
  - `analysis/` — Loaders, plots (M13)
- `tests/` — unit, integration, fixtures
  - `fixtures/llm/` — FakeLLM scripted response YAMLs (planner_simple, executor_code_run, determinism_seed, m5_*.yaml)
  - `fixtures/agents/` — Agent config fixtures (minimal_valid.yaml, debater_with_stance.yaml)
  - `unit/agents/` — 10 test files for Agent framework (config, tokens, basic, tool_loop, scratchpad_window, summarizer, budget_propagation, debater_stance, role_configs_load, executor_exit_criterion)
  - `unit/core/` — 150 tests for errors/types/reducers/state/public API
  - `unit/llm/` — 110 tests for pricing/budget/retry/providers/fake/wrapper
  - `integration/llm/` — 3 tests for LLM layer contract (end-to-end, budget exceed, replay round-trip)
- `conf/` — YAML configuration templates
  - `pricing.yaml` — per-1K-token prices for OpenAI/Anthropic/vLLM/fake models (version 1)
  - `agents/` — role-specific agent configs (planner.yaml, researcher.yaml, executor.yaml, critic.yaml, debater.yaml)
- `alembic/` — database migrations (async template, single initial migration)
- `dev/` — documentation (PLAN.md, arch.md) and task tracking (done/active)

## Key Modules

### Core Types & State (`core/`)
- **Purpose:** Base data structures (Pydantic models + TypedDict) for message passing, errors, phase tracking; LangGraph-compatible reducers for merging agent state updates.
- **Exports:** `AgentRole`, `AgentState`, `AtmError`, `BudgetEvent` (DB schema), `BudgetExceededError`, `GraphState`, `HumanContext`, `HumanResponse`, `HumanRole`, `LLMError`, `LLMResponse`, `Message`, `MessageKind`, `Phase`, `PhaseError`, `PhaseTransition`, `RunResult`, `SharedState`, `TaskResult`, `TaskSpec`, `TokenUsage`, `ToolCall`, `ToolError`, `ToolResult`, `TopologyTransition`, `dedup_by_id_reducer`, `merge_agent_states`
- **M2 additions:** `LLMError(AtmError)` with attrs provider/model/attempts; `LLMResponse.started_at: datetime` with `llm_calls` reducer sorted by started_at; `Message.to_lc()` / `Message.from_lc()` LangChain adapters (4-kind round-trip: user/assistant/system/tool).
- **Dependencies:** pydantic, typing, langchain-core (adapters)
- **Status:** M1 complete + M2 extensions.

### LLM Layer (`llm/`)
- **Purpose:** Unified wrapper over LangChain chat models with usage/cost/retry, three-tier budget enforcement, prompt caching support, and deterministic FakeLLM for tests.
- **Exports (from `atm.llm`):** `LLMWrapper`, `BudgetTracker`, `BudgetSignal` (runtime event), `BudgetLevel`, `FakeLLM`, `REPLAY_SCHEMA`, `Pricing`, `ModelPricing`, `RetryPolicy`, `with_retry`, `is_transient`, `build_openai`, `build_anthropic`, `build_vllm`, `inject_cache_control`, `DEFAULT_CACHE_TTL`.
- **Submodules:**
  - `pricing.py` — `ModelPricing` (frozen Pydantic, optional float fields), `Pricing.from_yaml` + `cost` (auto-detects OpenAI vs Anthropic cache convention) + `estimate` pre-call helper.
  - `budget.py` — `BudgetLevel` StrEnum, `BudgetSignal` frozen Pydantic (runtime warn/exceed event — not to be confused with `core.types.BudgetEvent` which is the DB persistence schema), `BudgetTracker` (asyncio.Lock, three-tier check/record, warn+exceed callbacks, sync or async callable).
  - `retry.py` — `RetryPolicy` frozen dataclass, `is_transient` duck-types `.status_code` 429/≥500 in addition to `retry_on` tuple, `with_retry` with exponential backoff + full jitter, wraps exhaustion as `LLMError` preserving `__cause__`.
  - `wrapper.py` — `LLMWrapper.ainvoke` flow: convert Message→BaseMessage via `Message.to_lc`, pre-call budget estimate (tiktoken or heuristic), optional `inject_cache_control` for Anthropic, `with_retry` wrapping `_llm.ainvoke`, parse OpenAI or Anthropic usage_metadata, compute cost via Pricing, record budget, build LLMResponse with started_at + latency_ms. `astream` raises NotImplementedError. Accepts `llm=` injection for tests. **M5 addition:** `model_id` property exposes the underlying model identifier.
  - `fake.py` — `FakeLLM(mode=scripted|replay|echo)`: scripted reads YAML fixtures by `(agent_id, step_idx)` with role fallback and asyncio.Lock; replay reads pyarrow Table with `REPLAY_SCHEMA` and maps `call_id → LLMResponse.id`; echo mirrors last user message content. `latency_ms=0` constant ensures bit-identical determinism.
  - `providers/openai.py` — `build_openai` strips `openai:` prefix, `api_key="EMPTY"` default.
  - `providers/anthropic.py` — `build_anthropic` + `inject_cache_control` (deep-copies last message, handles multi-modal `content=list[dict]`, always returns NEW list — never mutates input).
  - `providers/vllm.py` — `build_vllm` → `ChatOpenAI(base_url=..., api_key="EMPTY")` per vLLM OpenAI-compatible endpoint convention.
- **Dependencies:** langchain-core, langchain-openai, langchain-anthropic, anthropic, openai, tiktoken, pyyaml, pyarrow, asyncio.
- **Test coverage:** 110 unit tests + 3 integration tests, all green.
- **Status:** M2 complete + M5 extensions.

### Tools & Sandbox (`tools/`)
- **Purpose:** Tool protocol, registry, 12 concrete implementations (4 global, 8 local), sandbox abstraction with dev (subprocess) and prod (Docker) modes.
- **Key interfaces:** `Tool` (runtime_checkable Protocol), `ToolSchema` (frozen Pydantic, describes I/O schema + name), `ToolRegistry` (ainvoke_by_name with latency tracking via time.monotonic), `CodeSandbox` (Protocol with IS_ISOLATED ClassVar), `ExecResult` (frozen: stdout/stderr/exit_code/timed_out/duration_ms/oom_killed), `SandboxConfig` (tmpfs mount limits).
- **Factory:** `build_default_registry(workspace, corpus_dir, sandbox, prod_mode=False)` — registers all 12 tools; fail-closed (raises ToolError if prod_mode=True and sandbox.IS_ISOLATED=False).
- **Exports:** `Tool`, `ToolRegistry`, `ToolSchema`, `CodeSandbox`, `ExecResult`, `SandboxConfig`, `SubprocessSandbox`, `DockerSandbox`, all 12 tool classes, `build_default_registry`, `resolve_and_validate_url`, `with_tool_retry`.
- **Submodules:** base.py, _safety.py, _retry.py, defaults.py, sandbox/ subdir (base.py, subprocess_sandbox.py, docker_sandbox.py), global_/ subdir (calculator.py, file_read.py, search.py, url_fetch.py), local_/ subdir (code_run.py, test_run.py, file_write.py, diff.py, todo_write.py, plan_update.py, lint.py, semantic_search.py).
- **Sandbox hardening (DockerSandbox):** seccomp profile via JSON string (deny-list for 24 syscalls), cap_drop=["ALL"], read_only=True, network_mode="none", user="1000:1000", tmpfs /work + /tmp (noexec, nosuid).
- **Dependencies:** docker>=7.1, ddgs>=9.13, httpx>=0.27, numpy>=1.26; faiss-cpu optional.
- **Test coverage:** 121 unit tests + 13 non-docker integration tests; 29 docker/network tests gated via markers.
- **Status:** M4 complete.

### Agent Framework (`agents/`) — M5 complete
- **Purpose:** LangGraph-compatible agent base class with scratchpad policy C (append-only journal, windowed view, optional LLM-based summarization), tool-loop integration, and role-based subclasses.
- **Exports (from `atm.agents`):** `Agent`, `AgentConfig`, `load_agent_config`, `Planner`, `Researcher`, `Executor`, `Critic`, `Debater`.
- **Submodules:**
  - `config.py` — `AgentConfig` (frozen Pydantic v2 model with id/model_id/system_prompt/tools/window_size/max_loop_depth/params), `load_agent_config(path: Path) -> AgentConfig` YAML loader.
  - `_tokens.py` — `estimate_prompt_tokens(messages: list[Message], model_id: str) -> int`; tiktoken for `openai:*` providers, `len(text)//4 + 4` heuristic fallback, `lru_cache(maxsize=8)` on encoder.
  - `base.py` — `Agent` class with `__init__(cfg, llm, tools, summarizer_llm=None)`, `async step(state: AgentState) -> dict` (LangGraph node signature, returns delta AgentState), scratchpad policy C implementation (_build_prompt windowing, _maybe_summarize), tool-loop (parse tool_calls, invoke via ToolRegistry, accumulate results), `_StepOutcome` and `AgentView` frozen dataclasses for internal structuring.
  - `planner.py` — `Planner(Agent)` subclass (empty, inherits all behavior).
  - `researcher.py` — `Researcher(Agent)` subclass (empty, inherits all behavior).
  - `executor.py` — `Executor(Agent)` subclass (empty, inherits all behavior).
  - `critic.py` — `Critic(Agent)` subclass (empty, inherits all behavior).
  - `debater.py` — `Debater(Agent)` subclass with `__init__` override: substitutes `{{stance}}` placeholder in system_prompt via `cfg.model_copy(update=...)`, raises `ValueError` on unknown stance.
- **Key patterns:**
  - **Scratchpad policy C:** Always append reasoning + tool_calls + observations to scratchpad; build prompt from (1) full old scratchpad truncated to 2000 chars intra-loop + (2) optional LLM-summarized older steps, (3) recent window (window_size = event count, not steps). _build_prompt docstring documents semantics.
  - **Agent.step(state) signature:** LangGraph node compatible; returns delta dict (only new scratchpad_events + absolute counters) to avoid duplication through reducer merge_agent_states.
  - **Tool-loop:** Parses tool_calls from LLM response, invokes via ToolRegistry.ainvoke_by_name, accumulates ToolResult observations, re-prompts until stop or max_loop_depth exceeded.
  - **_StepOutcome dataclass:** Encapsulates response, scratchpad_events, tool_calls, tool_results from a single step.
  - **Debater stance substitution:** One-time substitution at __init__ via model_copy; prompt immutable after initialization.
  - **ToolRegistry-based wiring:** Agents accept ToolRegistry (not list[Tool]); dispatch and error-handling encapsulated in registry.ainvoke_by_name.
- **Config locations:** `conf/agents/{planner,researcher,executor,critic,debater}.yaml` — role-specific window_size, tools list, optional stance param for Debater.
- **Dependencies:** pydantic, langchain-core, asyncio, tiktoken (optional).
- **Test coverage:** 10 unit test files (91 tests total) covering config loading, token estimation, agent basic I/O, tool-loop C1 exit criterion, scratchpad windowing, summarizer integration, budget propagation, debater stance substitution, role config loading, executor exit strategies.
- **Test fixtures:** `tests/fixtures/agents/{minimal_valid,debater_with_stance}.yaml`, `tests/fixtures/llm/m5_{agent_basic,agent_tool_loop,agent_tool_loop_two_iters,agent_scratchpad,agent_summarizer_primary,agent_summarizer_secondary,executor_exit}.yaml`.
- **Status:** M5 complete.

### Topology Framework (`topology/`)
- **Exports (M6+ planned):** `Topology` protocol, `TopologyRegistry`, Star/Chain/Mesh/Debate/Hierarchical/Adaptive implementations.
- **Key Interface:** `topology.build(agents, cfg) -> CompiledStateGraph`
- **Status:** M0 skeleton, M6 not started.

### Phase Manager & Routers (`phases/`)
- **Exports (M8 planned):** `PhaseManager`, `RuleBasedPhaseRouter`, `LLMPhaseRouter`, `TopologyRouter` (rule/llm/oracle modes), `SwitchGuards`.
- **Status:** M0 skeleton, M8 not started.

### Human Gateway & HITL (`human/`)
- **Exports (M9 planned):** `HumanGateway` protocol, `LLMSimulatedGateway`, `CLIGateway`.
- **Status:** M0 skeleton, M9 not started.

### Storage Layer (`storage/`) — M3 complete

Persistence layer for experiments.

- `models.py` — 6 SQLAlchemy 2.x models: `Experiment` (root), `Run`, `Phase` (exported as `PhaseRow` from `__init__` to avoid collision with `atm.core.types.Phase` StrEnum), `HumanInteraction`, `BudgetEvent`, `TopologyTransition`. + `FinishReason` StrEnum. Schema matches arch.md §3.4.
- `session.py` — business DB: `create_engine`, `create_session_factory` (expire_on_commit=False), `session_scope` CM.
- `schemas.py` — 6 pyarrow schemas. Timestamps are `pa.timestamp('us', tz='UTC')` (tz-correctness).
- `parquet_writer.py` — buffered async writer with per-stream `asyncio.Lock`; buffer_rows + buffer_seconds auto-flush. `write_scratchpad` validates `agent_id` against `^[A-Za-z0-9_-]{1,64}$` (path-traversal guard).
- `checkpointer.py` — two-pool pattern per arch.md §11.3: `checkpointer_scope` (primary `@asynccontextmanager`), `build_checkpointer` (low-level). Uses psycopg AsyncConnectionPool with `autocommit=True, prepare_threshold=0`.
- **Exports (from `atm.storage`):** `Base`, `BudgetEvent`, `Experiment`, `FinishReason`, `HumanInteraction`, `LLM_CALL_SCHEMA`, `MESSAGE_SCHEMA`, `PHASE_SCHEMA`, `ParquetWriter`, `PhaseRow`, `Run`, `SCRATCHPAD_SCHEMA`, `TOOL_CALL_SCHEMA`, `TOPOLOGY_TRANSITION_SCHEMA`, `TopologyTransition`, `build_checkpointer`, `checkpointer_scope`, `create_engine`, `create_session_factory`, `session_scope`.

**Reproducibility fields — M5+ ownership TODO**
- `Run.models_by_role_json` — filled by **Runner** at INSERT time (M5+)
- `Run.model_version_snapshot` — filled by **LLMWrapper** on first successful call (M5+)
- `Run.sandbox_image_digest` — filled by **DockerSandbox** at init (M5+)

### Observability Layer (`observability/`) — M3 complete

LangGraph callback handler + serializers.

- `callbacks.py` — `ExperimentCallbackHandler(AsyncCallbackHandler)`. Implements arch.md §10.3 4 flush invariants: (a) root `on_chain_end` → close, (b) phase/topology transition → flush STRICTLY BEFORE PG INSERT, (c) root `on_chain_error` → close, (d) buffer overflow auto-flush at writer layer. Atomic `UPDATE ... RETURNING` for `runs.budget_spent_usd` and `experiments.total_cost_usd`. Budget warn/exceed one-shot events. `on_tool_start` captures `tool_name`, `agent_id` (from metadata), and `args_json`; `on_tool_end/error` preserve these fields.
- `serializers.py` — 6 pure functions converting pydantic domain types → parquet rows matching storage/schemas.py. `_dumps` helper: json.dumps with sort_keys=True, ensure_ascii=False, separators=(',', ':').
- **Exports (from `atm.observability`):** `ExperimentCallbackHandler`, `llm_response_to_row`, `message_to_row`, `phase_transition_to_row`, `scratchpad_entry_to_row`, `tool_call_to_row`, `topology_transition_to_row`.

### Migrations (`alembic/`) — M3 complete

- `versions/0001_initial_business_schema.py` — creates 6 business tables (not checkpoint tables; those are owned by `AsyncPostgresSaver.setup()`). Chained off `bc5f66dd0897` placeholder. Downgrade in reverse FK order.

### Tasks & Evaluation (`tasks/`, `evaluation/`)
- **Exports (M10-M11 planned):** `TaskSpec`, `TaskRegistry`, evaluators, NASA-TLX aggregator.
- **Status:** M0 skeleton, not started.

### Experiment Runner & CLI (`experiment/`)
- **Exports (M12 planned):** Pydantic schemas, `ConfigLoader`, `Runner`, `GridExecutor`, CLI commands.
- **Status:** M0 skeleton, M12 not started.

### Analysis & Plots (`analysis/`)
- **Exports (M13 planned):** `load_experiment()`, `load_llm_calls()`, `load_runs()`, plot functions.
- **Status:** M0 skeleton, M13 not started.

## Patterns & Conventions

### State Management
- **LangGraph state:** TypedDict-based `GraphState` with shared fields + per-agent `AgentState` reducer.
- **Scratchpad policy C (M5):** Append-only journal with windowed view; always write reasoning + tool_calls + observations; prompt builder handles multi-step summarization and recent event filtering via window_size (event count semantic).

### Agent Node Signature (M5)
- **Agent.step(state: AgentState) -> dict:** LangGraph node signature; returns delta dict (new scratchpad_events + absolute counters) to integrate seamlessly with `merge_agent_states` reducer.

### Error Handling
- **Budget cuts:** `BudgetExceededError` raised on three-tier limit crossings — hard stop.
- **LLM errors:** `LLMError(AtmError)` wraps retry exhaustion with `__cause__` preserved; duck-typing for transient classification.
- **Phase errors:** `PhaseError` on invalid state transitions; FSM enforces monotonic phases.
- **Tool errors:** `ToolError` wrapped from sandbox/API failures.

### Configuration & Composition
- **YAML + OmegaConf + Pydantic:** YAML in `conf/`, composed via OmegaConf, validated by Pydantic schemas.
- **Pricing config:** `conf/pricing.yaml` (version 1) with per-1K-token rates; OpenAI uses `cached_input_per_1k`, Anthropic splits `cache_read_per_1k`/`cache_write_per_1k`.
- **Agent configs (M5):** `conf/agents/*.yaml` define role-specific window_size, tools list, optional stance param.
- **Registry pattern:** Dynamic lookup by name for topologies/tools/tasks/agents.

### Dependency Injection
- **Factory functions:** `build_openai/anthropic/vllm` accept optional kwargs; `LLMWrapper(..., llm=fake)` allows test injection.
- **Protocol-based:** Agent / Sandbox / Topology / HumanGateway — Protocols with multiple implementations.
- **ToolRegistry-based agent wiring (M5):** Agents accept `ToolRegistry` instance; dispatch and error handling encapsulated in registry.

### Observability & Debugging
- **LangGraph callbacks:** `ExperimentCallbackHandler` (M3) hooks on_llm_*/on_tool_*/on_chain_* → async buffered write to Postgres + Parquet.
- **Event dispatch:** `dispatch_custom_event` for phase/topology transitions, signal emissions, budget warnings (`BudgetSignal`).
- **Parquet + SQL split:** Bulk data (LLM calls, messages, tool calls) in Parquet; metadata (runs, costs, phases) in Postgres.

### Testing
- **FakeLLM:** Deterministic mock, scripted fixtures keyed by `(agent_id, step_idx)` with role fallback; supports scripted/echo/replay.
- **Fixtures location:** `tests/fixtures/llm/<name>.yaml` — scripted LLM responses; `tests/fixtures/agents/<name>.yaml` — agent config fixtures.
- **M5 test coverage:** 10 test files (91 tests) covering Agent config loading, token estimation, basic agent I/O, tool-loop C1 exit criterion, scratchpad windowing, summarizer integration, budget propagation, Debater stance substitution, role configs, executor exit strategies. All passing.
- **Key test categories (M2-M5 cumulative):**
  - Core errors hierarchy including `LLMError` (M1).
  - Types/enums (Message, ToolCall, LLMResponse.started_at).
  - Reducers with monoid invariants + sort_by="started_at" for llm_calls (M2).
  - LLM pricing (OpenAI no-cache, cache hit, Anthropic cache_read+cache_write, estimate) (M2).
  - Budget tracker (per-call/per-run/per-experiment cutoffs, asyncio.gather concurrency, callback events) (M2).
  - Retry (success, 429/500 retry, non-transient skip, exhaustion with __cause__, jitter bounds) (M2).
  - Providers (build_openai/anthropic/vllm, inject_cache_control immutability) (M2).
  - FakeLLM (scripted step advancement, tool_calls, echo, replay call_id→id, determinism) (M2).
  - LLMWrapper (usage parsing, retry integration, budget-exceed-before-call, tool_calls passthrough) (M2).
  - Integration: end-to-end scripted fake + LLMWrapper, budget-exceed guard path, replay round-trip (M2).
  - Agent config, token estimation, step() I/O, tool-loop, scratchpad window, summarizer, budget propagation (M5).
  - Debater stance substitution, all role configs loading, executor exit criterion (M5).

### Async Patterns
- **Budget tracker:** `asyncio.Lock` for thread-safe three-tier checks.
- **Retry backoff:** `asyncio.sleep` + exponential backoff + full jitter.
- **FakeLLM:** `asyncio.Lock` around per-agent step counter for concurrent ainvoke safety.
- **Agent summarizer:** Separate `summarizer_llm: LLMWrapper | None` instance with budget/retry integration; agent_id suffixed "_summarizer" for fixture routing.

## External Dependencies

### Runtime (M2 additions)
- **langchain, langchain-core** (0.3.x): Chat model interface.
- **langchain-openai** (0.3.x): ChatOpenAI.
- **langchain-anthropic** (0.3.x): ChatAnthropic + cache control.
- **tiktoken** (≥0.8): OpenAI token counting (Anthropic uses heuristic fallback).
- **pyyaml** (≥6): Fixture + pricing + agent config parsing.
- **pyarrow** (≥16): Replay Table schema + bulk experiment data.

### M4 additions (Tools & Sandbox)
- **docker** (≥7.1): Container runtime for DockerSandbox.
- **ddgs** (≥9.13): DuckDuckGo search backend.
- **httpx** (≥0.27): Async HTTP client for UrlFetchTool.
- **numpy** (≥1.26): TF-IDF vectorization for SemanticSearchTool.
- **faiss-cpu** (≥1.11): Optional extra for semantic search acceleration.

### Existing (M0-M1)
- **langgraph:** Multi-agent orchestration primitives.
- **sqlalchemy[asyncio] + asyncpg:** Async ORM.
- **omegaconf:** Config composition.
- **pydantic:** Data validation.
- **pytest + pytest-asyncio:** Test framework.
- **ruff + mypy:** Lint + type checking.
- **alembic:** DB migrations.
- **typer:** CLI framework (M12).
- **huggingface-hub:** Dataset loading (M10).

### Dev
- **types-PyYAML:** Mypy stubs for PyYAML.

## Test Setup
- **Framework:** pytest + pytest-asyncio (asyncio_mode="auto").
- **Structure:** `tests/unit/{core,llm,tools,agents}/`, `tests/integration/{llm,tools}/`, `tests/fixtures/{llm,tools/corpus,agents}/`.
- **Total tests:** ~488 (91 agents + 121 tools + 13 non-docker integration tools + 263 legacy unit/integration core+llm).
- **M4 test markers:** `@pytest.mark.docker` and `@pytest.mark.network` gated via ATM_ENABLE_DOCKER_TESTS and ATM_ENABLE_NETWORK_TESTS env vars; 29 such tests auto-skipped otherwise.
- **M5 test markers:** None (all agent tests run by default).
- **Mocking approach:** FakeLLM with YAML fixtures for unit + integration; FakeDDGS for search tool testing; mock docker for sandbox unit tests.
- **Coverage target:** All core modules + critical paths; M4 tools + M5 agents fully tested; lower coverage on M6+ until implementations exist.
