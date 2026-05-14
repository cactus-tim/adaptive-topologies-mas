# Codebase Map
*Auto-generated. Last updated: 2026-05-13 (M11-resync)*

## Tech Stack
- **Language:** Python 3.11+
- **Package Manager:** uv
- **Orchestration:** LangGraph (multi-agent framework)
- **LLM SDK:** LangChain chat models (OpenAI/Anthropic/Cerebras/vLLM)
- **Database:** PostgreSQL 16 (async via SQLAlchemy 2.x + asyncpg, ORM with Alembic migrations)
- **Bulk Storage:** Apache Parquet (PyArrow) for experiment data
- **Config:** YAML + Pydantic + OmegaConf
- **Testing:** pytest + pytest-asyncio
- **Linting/Types:** ruff + mypy (strict)
- **Containerization:** Docker (code sandbox + dev environment)
- **Logging:** structlog (optional bootstrap in atm/__init__.py, controlled via ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1)

## Project Structure
- `src/atm/` — main package (Adaptive Topologies MAS, imported as `atm`)
  - `core/` — base types, state, errors, reproducibility helpers (Message, ToolCall, Phase, AgentState, GraphState, seed_all)
  - `llm/` — LLMWrapper, providers (OpenAI/Anthropic/Cerebras/vLLM), budget tracking, pricing, retry, fake LLM, `factory.build_llm(model_id, ...)` provider-prefix router (supports `fake:scripted/echo/replay`)
  - `tools/` — Tool protocol, registry with role-based policy support, global tools, Docker/subprocess sandbox (M4)
  - `agents/` — Agent base class, Planner/Researcher/Executor/Critic/Debater roles, scratchpad policy C (M5)
  - `topology/` — Topology protocol + Registry, 5 implementations: Star + Chain (M6); Mesh + Debate + Hierarchical (M7 complete)
  - `phases/` — PhaseManager FSM, TopologyRouter, SwitchGuards, signals (M8)
  - `human/` — HumanGateway protocol, LLMSimulatedGateway, CLI gateway (M9)
  - `storage/` — SQLAlchemy models, async session, ParquetWriter, checkpointer wrapper (M3 complete)
  - `observability/` — ExperimentCallbackHandler (LangGraph async callbacks), serializers, structlog processors (M3 complete; M11-resync: filter_secrets)
  - `tasks/` — TaskSpec base, HumanEval/GSM8K/CommonGen/DABench loaders + evaluators (M10-resync)
  - `evaluation/` — LLM-as-judge, ground truth runners (dispatches via EVALUATORS.get), metrics, NASA-TLX persistence (M11 complete; M11-resync: _DEPS table dispatch pattern)
  - `experiment/` — Pydantic config schemas + OmegaConf loader, single-run runner (`run_one`), Typer CLI (`atm run`), inline evaluator (M6 complete; grid + sweep in M12)
  - `analysis/` — Loaders, plots (M13)
- `tests/` — unit, integration, fixtures
  - `fixtures/llm/` — FakeLLM scripted response YAMLs (planner_simple, executor_code_run, determinism_seed, m5_*.yaml, m7_*.yaml, m11_gsm8k_e2e_executor.yaml)
  - `fixtures/agents/` — Agent config fixtures (minimal_valid.yaml, debater_with_stance.yaml, debater_contra.yaml)
  - `unit/agents/` — 10 test files for Agent framework (config, tokens, basic, tool_loop, scratchpad_window, summarizer, budget_propagation, debater_stance, role_configs_load, executor_exit_criterion)
  - `unit/core/` — 150 tests for errors/types/reducers/state/public API, plus seed reproducibility (M11-resync)
  - `unit/llm/` — 110 tests for pricing/budget/retry/providers/fake/wrapper, plus model_version_snapshot (M11-resync)
  - `unit/topology/` — 5 test files (base, star, chain, mesh, debate, hierarchical) covering routing/stopping/registration/invariants; 6 files total with conftest.py
  - `integration/llm/` — 3 tests for LLM layer contract (end-to-end, budget exceed, replay round-trip)
  - `integration/topology/` — 4 test files (mesh_e2e, debate_e2e, hierarchical_e2e, all_topologies_sanity) covering e2e flows with FakeLLM
  - `integration/evaluation/` — quality aggregation + GSM8K e2e test (M11-resync: replaced MMLU with GSM8K)
- `conf/` — YAML configuration templates
  - `pricing.yaml` — per-1K-token prices for OpenAI/Anthropic/Cerebras/vLLM/fake models (version 1)
  - `agents/` — role-specific agent configs (planner.yaml, researcher.yaml, executor.yaml, critic.yaml, debater.yaml)
  - `topology/` — topology configs (star.yaml, chain.yaml, mesh.yaml, debate.yaml, hierarchical.yaml)
  - `tools_policy.yaml` — role-based tool assignment policy (M11-resync)
- `alembic/` — database migrations (async template, single initial migration)
- `.github/workflows/` — CI/CD pipelines (M11-resync: ci.yml with lint/unit/integration matrix)
- `dev/` — documentation (PLAN.md, arch.md) and task tracking (done/active)

## M11-resync (2026-05-13) — Key Updates

### Merged Upstream
- **feat/m10 merge:** GSM8K/CommonGen/DABench task mix added; MMLU removed from evaluators.
- **arch/PLAN.md pulled from feat/m9:** Added gate definitions (G2, G6, G7, G8, G10).

### New Modules & Exports
- **`core/seed.py::seed_all(seed)`** — Seeding helper for `random`, `numpy` (optional), `torch` (optional); integrated into `runner.run_one` at head.
- **`observability/log_processors.py::filter_secrets`** — structlog processor with regex `(?i).*(api_key|token|password|secret).*` and cyclic-dict guard.
- **`evaluation/tlx.py::persist_tlx(session, interaction_id, raw_score)`** — Idempotent NASA-TLX score writer to database.
- **`tools/base.py::ToolRegistry.tools_for(role: str) -> list[Tool]`** — Role-based tool lookup with lazy YAML policy loading.
- **`conf/tools_policy.yaml`** — Role→tools mapping; currently-registered tools only (no stubs).
- **`.github/workflows/ci.yml`** — Lint (Python 3.11/3.12 matrix), unit tests, integration tests with postgres:16; astral-sh/setup-uv@v3 pinned with lockfile caching.

### LLMWrapper & Reproducibility
- **`LLMWrapper.last_model_version: str | None`** — Captures Anthropic `response_metadata['model']`, OpenAI `response_metadata['system_fingerprint']` with fallback.
- **`runner.run_one` updates:**
  - `seed_all(cfg.seed)` at top.
  - `model_version_snapshot` UPDATE in finally block (collects non-None entries from all role wrappers).
  - `sandbox_image_digest` UPDATE after sandbox init (truncates to 80 chars).

### Sandbox & Docker
- **`DockerSandbox.image_digest: str | None`** — Captured from `client.images.get(image_map["python"]).id` in __init__, gracefully handles unavailable daemon.
- **`SubprocessSandbox.image_digest: ClassVar[str | None] = None`** — Protocol compliance.
- **`CodeSandbox` Protocol:** Added `image_digest` property.

### Evaluation & Ground Truth
- **`evaluation/ground_truth.py` rewrite:** Now dispatches via `EVALUATORS.get(spec.evaluator_key)` using literal `_DEPS` table (closes G8). Fully type-checkable.
- **Test fixtures:** `tests/fixtures/llm/m11_gsm8k_e2e_executor.yaml` (executor emits "The answer is 42.").
- **E2e test rewrite:** `test_aggregator_e2e.py` replaced MMLU with GSM8K (`TaskSpec(type="reasoning", evaluator_key="gsm8k_numeric")`).
- **Deleted:** `tests/fixtures/llm/m11_mmlu_e2e_executor.yaml`.

### Structlog Bootstrap
- **`atm/__init__.py`:** Bootstrap structlog with `filter_secrets` processor on import (gated by `ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1` env var); idempotent, safe for double-import.

### Test Count
- Unit tests: **1233 total** (gates open; sg docker).
- Unit tests only: **1150** (without network/docker markers).
- All gate open + GSM8K e2e verified passing.

## Key Modules

### Core Types & State (`core/`)
- **Purpose:** Base data structures (Pydantic models + TypedDict) for message passing, errors, phase tracking; LangGraph-compatible reducers for merging agent state updates; reproducibility seeding.
- **Exports:** `AgentRole`, `AgentState`, `AtmError`, `BudgetEvent` (DB schema), `BudgetExceededError`, `GraphState`, `HumanContext`, `HumanResponse`, `HumanRole`, `LLMError`, `LLMResponse`, `Message`, `MessageKind`, `Phase`, `PhaseError`, `PhaseTransition`, `RunResult`, `SharedState`, `TaskResult`, `TaskSpec`, `TokenUsage`, `ToolCall`, `ToolError`, `ToolResult`, `TopologyTransition`, `dedup_by_id_reducer`, `merge_agent_states`, `seed_all`.
- **M2 additions:** `LLMError(AtmError)` with attrs provider/model/attempts; `LLMResponse.started_at: datetime` with `llm_calls` reducer sorted by started_at; `Message.to_lc()` / `Message.from_lc()` LangChain adapters (4-kind round-trip: user/assistant/system/tool).
- **M11-resync additions:** `seed_all(seed: int)` seeding `random`, `numpy`, `torch` with ImportError fallback.
- **Dependencies:** pydantic, typing, langchain-core (adapters)
- **Status:** M1 complete + M2 + M11-resync extensions.

### LLM Layer (`llm/`)
- **Purpose:** Unified wrapper over LangChain chat models with usage/cost/retry, three-tier budget enforcement, prompt caching support, deterministic FakeLLM, and model version capture for reproducibility.
- **Exports (from `atm.llm`):** `LLMWrapper`, `BudgetTracker`, `BudgetSignal` (runtime event), `BudgetLevel`, `FakeLLM`, `REPLAY_SCHEMA`, `Pricing`, `ModelPricing`, `RetryPolicy`, `with_retry`, `is_transient`, `build_openai`, `build_anthropic`, `build_cerebras`, `build_vllm`, `inject_cache_control`, `DEFAULT_CACHE_TTL`.
- **M5 additions:** `model_id` property exposes underlying model identifier.
- **M11-resync additions:** `LLMWrapper.last_model_version: str | None` captures Anthropic `response_metadata['model']` and OpenAI `response_metadata['system_fingerprint']` with fallback.
- **Dependencies:** langchain-core, langchain-openai, langchain-anthropic, anthropic, openai, tiktoken, pyyaml, pyarrow, asyncio.
- **Test coverage:** 110 unit tests + 3 integration tests + M11-resync snapshot tests.
- **Status:** M2 complete + M5 + M11-resync extensions.

### Tools & Sandbox (`tools/`)
- **Purpose:** Tool protocol, registry with role-based policy support, 12 concrete implementations (4 global, 8 local), sandbox abstraction with dev (subprocess) and prod (Docker) modes.
- **Key interfaces:** `Tool` (runtime_checkable Protocol), `ToolSchema` (frozen Pydantic), `ToolRegistry` (ainvoke_by_name, tools_for(role)), `CodeSandbox` (Protocol with IS_ISOLATED ClassVar and image_digest), `ExecResult`, `SandboxConfig`.
- **M11-resync additions:** `ToolRegistry.tools_for(role: str) -> list[Tool]` with lazy YAML policy loading; `image_digest` property added to `CodeSandbox` Protocol and implementations.
- **Exports:** `Tool`, `ToolRegistry`, `ToolSchema`, `CodeSandbox`, `ExecResult`, `SandboxConfig`, `SubprocessSandbox`, `DockerSandbox`, all 12 tool classes, `build_default_registry`, `resolve_and_validate_url`, `with_tool_retry`.
- **Dependencies:** docker>=7.1, ddgs>=9.13, httpx>=0.27, numpy>=1.26; faiss-cpu optional.
- **Test coverage:** 121 unit tests + 13 non-docker integration tests; 29 docker/network tests gated via markers.
- **Status:** M4 complete + M11-resync extensions.

### Agent Framework (`agents/`) — M5 complete
- **Purpose:** LangGraph-compatible agent base class with scratchpad policy C (append-only journal, windowed view, optional LLM-based summarization), tool-loop integration, and role-based subclasses.
- **Exports (from `atm.agents`):** `Agent`, `AgentConfig`, `load_agent_config`, `Planner`, `Researcher`, `Executor`, `Critic`, `Debater`.
- **Key patterns:** Scratchpad policy C; Agent.step(state) LangGraph signature; ToolRegistry-based wiring.
- **Config locations:** `conf/agents/{planner,researcher,executor,critic,debater}.yaml`.
- **Dependencies:** pydantic, langchain-core, asyncio, tiktoken (optional).
- **Test coverage:** 10 unit test files (91 tests total).
- **Status:** M5 complete.

### Topology Framework (`topology/`) — M7 complete
- **Purpose:** Topology protocol + Registry with 5 registered implementations covering different multi-agent coordination patterns.
- **Implementations:** Star, Chain, Mesh, Debate, Hierarchical.
- **Key Interface:** `Topology.build(agents, cfg) -> CompiledStateGraph`.
- **Config locations:** `conf/topology/{star,chain,mesh,debate,hierarchical}.yaml`.
- **Test coverage:** 6 unit test files + 4 integration e2e tests; 11 sanity tests.
- **Status:** M7 complete (5 topologies registered, comprehensive test coverage).

### Phase Manager & Routers (`phases/`) — M8 complete
- **Purpose:** Monotonic Phase FSM (planning → execution → verification → done) with rule-based and LLM-based routers.
- **Exports (from `atm.phases`):** `RuleBasedPhaseRouter`, `LLMPhaseRouter`, `PhaseLimits`, `PhaseGuard`, `PhaseRouter`.
- **Test coverage:** 23 unit tests covering all guards, iter-cap logic, and fallback scenarios.
- **Status:** M8 complete (PhaseRouter Protocol, async routers, JSON validation, monotonicity checks).

### Storage Layer (`storage/`) — M3 complete
- **Purpose:** Persistence layer for experiments with 6 SQLAlchemy 2.x models and Parquet bulk writers.
- **Reproducibility fields:** `Run.models_by_role_json`, `Run.model_version_snapshot` (M11-resync), `Run.sandbox_image_digest` (M11-resync).
- **Exports (from `atm.storage`):** All 6 models, schemas, ParquetWriter, checkpointer utilities, reproducibility snapshots.
- **Status:** M3 complete + M11-resync snapshot fields.

### Observability Layer (`observability/`) — M3 complete + M11-resync
- **Purpose:** LangGraph callback handler + serializers + structlog processors.
- **M11-resync additions:** `log_processors.py` with `filter_secrets` processor (regex-based secret key detection, cyclic-dict guard via `id()`).
- **Exports (from `atm.observability`):** All serializers + `filter_secrets`.
- **Status:** M3 complete + M11-resync structlog integration.

### Evaluation Framework (`evaluation/`) — M11 complete + M11-resync
- **Purpose:** Post-hoc quality scoring, LLM-as-judge protocols, pure metric functions, NASA-TLX human-load model.
- **M11-resync key change:** `ground_truth.py` rewrite uses literal `_DEPS` table + `EVALUATORS.get(spec.evaluator_key)` dispatch (closes G8); fully type-checkable.
- **New:** `persist_tlx(session, interaction_id, raw_score)` idempotent UPDATE writer.
- **Exports (from `atm.evaluation`):** All judges, metrics, aggregators, `persist_tlx`, quality computation.
- **Integration tests:** GSM8K e2e (replaced MMLU).
- **Status:** M11 complete + M11-resync dispatch pattern.

### Tasks & Evaluation (`tasks/`) — M10-resync complete
- **Purpose:** Benchmark task infrastructure with 4 loaders (HumanEval/GSM8K/CommonGen/DABench) and 4 evaluators.
- **M10-resync:** Removed MMLU, added GSM8K/CommonGen/DABench.
- **Test coverage:** 86 unit tests; all GSM8K/CommonGen/DABench evaluated.
- **Status:** M10-resync complete (MMLU removed, GSM8K primary reasoning task).

### Experiment Runner & CLI (`experiment/`)
- **Purpose:** Single-run orchestrator with seed, model snapshot, and sandbox digest capture.
- **M11-resync key changes:** (a) `seed_all(cfg.seed)` at top of `run_one`; (b) `model_version_snapshot` UPDATE in finally block; (c) `sandbox_image_digest` UPDATE after sandbox init.
- **Status:** M0 skeleton + M11-resync seed/snapshot/digest wiring.

### Analysis Layer (`analysis/`) — M8.7
- **Purpose:** Oracle labels pipeline — produces `OracleTable` consumed by `OracleTopologyRouter` (M8.3) and LOO-plot analysis (M13 G9).
- **Exports (from `atm.analysis`):** `OracleTable` (Pydantic v2 frozen dataclass; fields: `by_task_type`, `by_task_id`, `default_topology` aliased `_default`), `build_leave_one_out_oracle` (async, reads `Run` rows from Postgres via SQLAlchemy 2.x session_factory), `build_loo_from_rows` (pure two-pass LOO aggregation helper, testable without DB), `load_oracle_table` (sync JSON/YAML loader, deserializes via `OracleTable.from_json_dict`).
- **Status:** M8.7 complete (oracle.py implemented; unit tests passing).

## Patterns & Conventions

### State Management
- **LangGraph state:** TypedDict-based `GraphState` with shared fields + per-agent `AgentState` reducer.
- **Scratchpad policy C (M5):** Append-only journal with windowed view; always write reasoning + tool_calls + observations.

### Reproducibility (M11-resync)
- **Seed:** `seed_all(cfg.seed)` at head of `run_one` before topology execution.
- **Model version:** `LLMWrapper.last_model_version` captures Anthropic/OpenAI fingerprints; persisted in `runs.model_version_snapshot` (finally block).
- **Sandbox digest:** `DockerSandbox.image_digest` captures Python image SHA; persisted in `runs.sandbox_image_digest` (80-char truncated).

### Configuration & Composition
- **YAML + OmegaConf + Pydantic:** YAML in `conf/`, composed via OmegaConf, validated by Pydantic schemas.
- **Role-based tool policy (M11-resync):** `conf/tools_policy.yaml` defines `global` list + `per_role` mapping; loaded lazily by `ToolRegistry.tools_for(role)`.

### Testing
- **FakeLLM:** Deterministic mock, scripted fixtures keyed by `(agent_id, step_idx)` with role fallback; supports scripted/echo/replay.
- **Test count:** 1233 unit (all gates open) / 1150 unit-only (no docker/network markers).
- **M11-resync fixtures:** `m11_gsm8k_e2e_executor.yaml` for GSM8K evaluation e2e test.

## External Dependencies

### Core (M0-M3)
- langchain, langchain-core, langchain-openai, langchain-anthropic, anthropic, openai, tiktoken, pyyaml, pyarrow, sqlalchemy[asyncio], asyncpg, omegaconf, pydantic, pytest, ruff, mypy, alembic, typer, huggingface-hub.

### M4-M7 additions
- docker>=7.1, ddgs>=9.13, httpx>=0.27, numpy>=1.26, faiss-cpu (optional), langgraph>=0.3.

### M11-resync
- structlog (optional; bootstrap gated by env var).

## Constraints & Notes
- **No schema migrations for M11-resync:** All columns (raw_tlx_score, model_version_snapshot, sandbox_image_digest) pre-exist.
- **Docker graceful fallback:** All docker-dependent code handles `(DockerException, OSError)` → None.
- **`torch` optional:** `seed_all` gracefully skips torch if not installed.
- **structlog optional:** Bootstrap can be disabled via `ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1`.
- **CI Python matrix:** `['3.11', '3.12']` for lint/unit; integration on 3.12 with postgres:16 service.
