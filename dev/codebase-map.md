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
  - `agents/` — Agent base class, Planner/Researcher/Executor/Critic/Debater roles (M5)
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
  - `fixtures/llm/` — FakeLLM scripted response YAMLs (planner_simple, executor_code_run, determinism_seed)
  - `unit/core/` — 150 tests for errors/types/reducers/state/public API
  - `unit/llm/` — 110 tests for pricing/budget/retry/providers/fake/wrapper
  - `integration/llm/` — 3 tests for LLM layer contract (end-to-end, budget exceed, replay round-trip)
- `conf/` — YAML configuration templates
  - `pricing.yaml` — per-1K-token prices for OpenAI/Anthropic/vLLM/fake models (version 1)
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
  - `wrapper.py` — `LLMWrapper.ainvoke` flow: convert Message→BaseMessage via `Message.to_lc`, pre-call budget estimate (tiktoken or heuristic), optional `inject_cache_control` for Anthropic, `with_retry` wrapping `_llm.ainvoke`, parse OpenAI or Anthropic usage_metadata, compute cost via Pricing, record budget, build LLMResponse with started_at + latency_ms. `astream` raises NotImplementedError. Accepts `llm=` injection for tests.
  - `fake.py` — `FakeLLM(mode=scripted|replay|echo)`: scripted reads YAML fixtures by `(agent_id, step_idx)` with role fallback and asyncio.Lock; replay reads pyarrow Table with `REPLAY_SCHEMA` and maps `call_id → LLMResponse.id`; echo mirrors last user message content. `latency_ms=0` constant ensures bit-identical determinism.
  - `providers/openai.py` — `build_openai` strips `openai:` prefix, `api_key="EMPTY"` default.
  - `providers/anthropic.py` — `build_anthropic` + `inject_cache_control` (deep-copies last message, handles multi-modal `content=list[dict]`, always returns NEW list — never mutates input).
  - `providers/vllm.py` — `build_vllm` → `ChatOpenAI(base_url=..., api_key="EMPTY")` per vLLM OpenAI-compatible endpoint convention.
- **Dependencies:** langchain-core, langchain-openai, langchain-anthropic, anthropic, openai, tiktoken, pyyaml, pyarrow, asyncio.
- **Test coverage:** 110 unit tests + 3 integration tests, all green.
- **Status:** M2 complete.

### Tools & Sandbox (`tools/`)
- **Purpose:** Tool protocol, registry, 12 concrete implementations (4 global, 8 local), sandbox abstraction with dev (subprocess) and prod (Docker) modes.
- **Key interfaces:** `Tool` (runtime_checkable Protocol), `ToolSchema` (frozen Pydantic, describes I/O schema + name), `ToolRegistry` (ainvoke_by_name with latency tracking via time.monotonic), `CodeSandbox` (Protocol with IS_ISOLATED ClassVar), `ExecResult` (frozen: stdout/stderr/exit_code/timed_out/duration_ms/oom_killed), `SandboxConfig` (tmpfs mount limits).
- **Factory:** `build_default_registry(workspace, corpus_dir, sandbox, prod_mode=False)` — registers all 12 tools; fail-closed (raises ToolError if prod_mode=True and sandbox.IS_ISOLATED=False).
- **Exports:** `Tool`, `ToolRegistry`, `ToolSchema`, `CodeSandbox`, `ExecResult`, `SandboxConfig`, `SubprocessSandbox`, `DockerSandbox`, all 12 tool classes, `build_default_registry`, `resolve_and_validate_url`, `with_tool_retry`.
- **Submodules:**
  - `base.py` — `Tool` Protocol (name, schema, ainvoke), `ToolSchema` (frozen: name, description, parameters, returns), `ToolRegistry`.
  - `_safety.py` — `resolve_and_validate_url` (SSRF guard: scheme allowlist {http,https}, socket.getaddrinfo, IP-check for private/loopback/link_local/reserved/multicast).
  - `_retry.py` — `with_tool_retry` (wraps `with_retry` from llm/, translates LLMError → ToolError).
  - `defaults.py` — `build_default_registry` factory (lazy imports to prevent circular deps).
  - `sandbox/base.py` — `CodeSandbox` Protocol, `ExecResult` frozen, `SandboxConfig` (timeouts, tmpfs /work + /tmp size/mode).
  - `sandbox/subprocess_sandbox.py` — dev-mode sandbox (IS_ISOLATED=False, asyncio.create_subprocess_exec, tempdir cleanup, timeout kill).
  - `sandbox/docker_sandbox.py` — prod sandbox (IS_ISOLATED=True, hardened: seccomp profile via JSON string, cap_drop=["ALL"], read_only=True, network_mode="none", user="1000:1000", tmpfs /work + /tmp with noexec/nosuid).
  - `global_/calculator.py` — `CalculatorTool` (safe AST-walker, no eval/exec, whitelisted BinOp/UnaryOp/Constant + math.* functions).
  - `global_/file_read.py` — `FileReadTool` (workspace-scoped read, path traversal + symlink-outside-workspace protection, max_bytes=1M default).
  - `global_/search.py` — `DuckDuckGoSearchTool` (ddgs via asyncio.to_thread, maps href→url, body→snippet, with_tool_retry support).
  - `global_/url_fetch.py` — `UrlFetchTool` (httpx async GET, SSRF-safe via resolve_and_validate_url, follow_redirects=False, aiter_bytes size limit, with_tool_retry).
  - `local_/code_run.py` — `CodeRunTool` (delegates to CodeSandbox.execute, wraps ExecResult).
  - `local_/test_run.py` — `TestRunTool` (sandbox-backed unittest runner, 2>&1 stderr merge, parses last stdout line as summary).
  - `local_/file_write.py` — `FileWriteTool` (atomic write via tempfile + os.replace, workspace-scoped, path traversal guard, optional create_parents).
  - `local_/diff.py` — `DiffTool` (difflib.unified_diff, output {"diff": str}).
  - `local_/todo_write.py` — `TodoWriteTool` (output {"state_update": {"shared": {"todos": […]}}}, NOT signals).
  - `local_/plan_update.py` — `PlanUpdateTool` (output {"state_update": {"shared": {"plan": …}}}, NOT signals).
  - `local_/lint.py` — `LintTool` (ruff via asyncio.create_subprocess_exec --output-format=json, optional pylint with graceful not-installed fallback).
  - `local_/semantic_search.py` — `SemanticSearchTool` (numpy TF-IDF, IDF=log((1+N)/(1+df))+1, L2-normalized cosine similarity, empty query → ok=False).
- **Sandbox hardening (DockerSandbox):**
  - seccomp profile: `conf/sandbox/seccomp.json` (generated from moby upstream via `scripts/derive_seccomp.py`; deny-list approach for 24 syscalls).
  - cap_drop=["ALL"], read_only=True (filesystem), network_mode="none", user="1000:1000" (unprivileged).
  - tmpfs /work (size=64m, noexec, nosuid) and /tmp (size=64m, noexec, nosuid) configured in `conf/sandbox/defaults.yaml`.
  - Seccomp profile passed as JSON-string to docker-py (not file path — API requirement).
- **Dependencies:** docker>=7.1, ddgs>=9.13, httpx>=0.27, numpy>=1.26; faiss-cpu available as optional extra.
- **Test coverage:** 121 unit tests (tools/) + 13 non-docker integration tests, all passing. 29 docker/network tests gated via @pytest.mark.docker / @pytest.mark.network (ATM_ENABLE_DOCKER_TESTS, ATM_ENABLE_NETWORK_TESTS env vars). Total: 134 pass + 29 skipped.
- **Status:** M4 complete (Steps 1–14).


### Agent Framework (`agents/`)
- **Exports (M5 planned):** `Agent` base with LangGraph node signature, `Planner`, `Researcher`, `Executor`, `Critic`, `Debater` with stance parameter.
- **Status:** M0 skeleton, M5 not started.

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
- `schemas.py` — 6 pyarrow schemas. Timestamps are `pa.timestamp('us', tz='UTC')` (minor correction from arch.md §3.5 for tz-correctness).
- `parquet_writer.py` — buffered async writer with per-stream `asyncio.Lock`; buffer_rows + buffer_seconds auto-flush (invariant (d) of arch.md §10.3). `write_scratchpad` validates `agent_id` against `^[A-Za-z0-9_-]{1,64}$` (path-traversal guard).
- `checkpointer.py` — two-pool pattern per arch.md §11.3: `checkpointer_scope` (primary `@asynccontextmanager`), `build_checkpointer` (low-level). Uses psycopg AsyncConnectionPool with `autocommit=True, prepare_threshold=0` (langgraph #2755).
- **Exports (from `atm.storage`):** `Base`, `BudgetEvent`, `Experiment`, `FinishReason`, `HumanInteraction`, `LLM_CALL_SCHEMA`, `MESSAGE_SCHEMA`, `PHASE_SCHEMA`, `ParquetWriter`, `PhaseRow`, `Run`, `SCRATCHPAD_SCHEMA`, `TOOL_CALL_SCHEMA`, `TOPOLOGY_TRANSITION_SCHEMA`, `TopologyTransition`, `build_checkpointer`, `checkpointer_scope`, `create_engine`, `create_session_factory`, `session_scope`.

**Reproducibility fields — M5+ ownership TODO**
- `Run.models_by_role_json` — filled by **Runner** at INSERT time (M5+)
- `Run.model_version_snapshot` — filled by **LLMWrapper** on first successful call (M5+)
- `Run.sandbox_image_digest` — filled by **DockerSandbox** at init (M5+)

### Observability Layer (`observability/`) — M3 complete

LangGraph callback handler + serializers.

- `callbacks.py` — `ExperimentCallbackHandler(AsyncCallbackHandler)`. Implements arch.md §10.3 4 flush invariants: (a) root `on_chain_end` → close, (b) phase/topology transition → flush STRICTLY BEFORE PG INSERT, (c) root `on_chain_error` → close, (d) buffer overflow auto-flush at writer layer. Atomic `UPDATE ... RETURNING` for `runs.budget_spent_usd` and `experiments.total_cost_usd` (arch.md §4.2, §18/#4). Budget warn/exceed one-shot events. `on_tool_start` captures `tool_name` (from `serialized['name']`), `agent_id` (from `metadata['agent_id']`) and `args_json`; `on_tool_end/error` preserve these fields; `result_json` serialized via shared `_dumps` helper.
- `serializers.py` — 6 pure functions (no I/O) converting pydantic domain types → parquet rows matching storage/schemas.py. `_dumps` helper: json.dumps with sort_keys=True, ensure_ascii=False, separators=(',', ':').
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
- **Scratchpad policy C:** Always write reasoning; send to LLM only a sliding window + optional LLM-summarized older steps.

### Error Handling
- **Budget cuts:** `BudgetExceededError` raised on three-tier limit crossings — hard stop.
- **LLM errors:** `LLMError(AtmError)` wraps retry exhaustion with `__cause__` preserved; duck-typing for transient classification.
- **Phase errors:** `PhaseError` on invalid state transitions; FSM enforces monotonic phases.
- **Tool errors:** `ToolError` wrapped from sandbox/API failures.

### Configuration & Composition
- **YAML + OmegaConf + Pydantic:** YAML in `conf/`, composed via OmegaConf, validated by Pydantic schemas.
- **Pricing config:** `conf/pricing.yaml` (version 1) with per-1K-token rates; OpenAI uses `cached_input_per_1k`, Anthropic splits `cache_read_per_1k`/`cache_write_per_1k`.
- **Registry pattern:** Dynamic lookup by name for topologies/tools/tasks.

### Dependency Injection
- **Factory functions:** `build_openai/anthropic/vllm` accept optional kwargs; `LLMWrapper(..., llm=fake)` allows test injection.
- **Protocol-based:** Agent / Sandbox / Topology / HumanGateway — Protocols with multiple implementations.

### Observability & Debugging
- **LangGraph callbacks:** `ExperimentCallbackHandler` (M3) hooks on_llm_*/on_tool_*/on_chain_* → async buffered write to Postgres + Parquet.
- **Event dispatch:** `dispatch_custom_event` for phase/topology transitions, signal emissions, budget warnings (`BudgetSignal`).
- **Parquet + SQL split:** Bulk data (LLM calls, messages, tool calls) in Parquet; metadata (runs, costs, phases) in Postgres.

### Testing
- **FakeLLM:** Deterministic mock, scripted fixtures keyed by `(agent_id, step_idx)` with role fallback; supports scripted/echo/replay.
- **Fixtures location:** `tests/fixtures/llm/<name>.yaml` — scripted LLM responses.
- **M2 test coverage:** 260 unit tests + 3 integration tests (all green). Key categories:
  - Core errors hierarchy including `LLMError`.
  - Types/enums (Message, ToolCall, LLMResponse.started_at, all Pydantic models).
  - Reducers with monoid invariants + sort_by="started_at" for llm_calls.
  - LLM pricing (OpenAI no-cache, OpenAI cache hit, Anthropic cache_read+cache_write, estimate).
  - Budget tracker (per-call/per-run/per-experiment cutoffs, asyncio.gather concurrency, callback events).
  - Retry (success, 429/500 retry, non-transient skip, exhaustion with __cause__, jitter bounds).
  - Providers (build_openai/anthropic/vllm, `inject_cache_control` immutability + multi-modal).
  - FakeLLM (scripted step advancement, tool_calls, echo, replay call_id→id, determinism, astream raises).
  - LLMWrapper (OpenAI + Anthropic usage parsing, retry integration, budget-exceed-before-call, tool_calls passthrough, astream stub).
  - Integration: end-to-end scripted fake + LLMWrapper, budget-exceed guard path, replay round-trip.

### Async Patterns
- **Budget tracker:** `asyncio.Lock` for thread-safe three-tier checks.
- **Retry backoff:** `asyncio.sleep` + exponential backoff + full jitter.
- **FakeLLM:** `asyncio.Lock` around per-agent step counter for concurrent ainvoke safety.

## External Dependencies

### Runtime (M2 additions)
- **langchain, langchain-core** (0.3.x): Chat model interface.
- **langchain-openai** (0.3.x): ChatOpenAI.
- **langchain-anthropic** (0.3.x): ChatAnthropic + cache control.
- **tiktoken** (≥0.8): OpenAI token counting (Anthropic uses heuristic fallback).
- **pyyaml** (≥6): Fixture + pricing config parsing.
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
- **Structure:** `tests/unit/{core,llm,tools}/`, `tests/integration/{llm,tools}/`, `tests/fixtures/{llm,tools/corpus}/`.
- **Total tests:** 397 (121 unit tools + 13 non-docker integration tools + 263 legacy unit/integration core+llm).
- **M4 test markers:** `@pytest.mark.docker` and `@pytest.mark.network` gated via ATM_ENABLE_DOCKER_TESTS and ATM_ENABLE_NETWORK_TESTS env vars; 29 such tests auto-skipped otherwise.
- **Mocking approach:** FakeLLM with YAML fixtures for unit + integration; FakeDDGS for search tool testing; mock docker for sandbox unit tests.
- **Coverage target:** All core modules + critical paths; M4 tools fully tested; lower coverage on M5+ until implementations exist.
