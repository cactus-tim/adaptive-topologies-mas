# Codebase Map
*Auto-generated. Last updated: 2026-05-14 (post-m8.7, pre-m12)*

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
  - `human/` — M9 + M9.1 + M9.2 complete: HumanGateway protocol, LLMSimulatedGateway, CLI gateway; all 5 topologies + adaptive.py accept `human_cfg`, `human_gateway_llm`, `role_router` kwargs in `.build()`; RoleRouter protocol (FixedRoleRouter, RuleBasedRoleRouter, LLMRoleRouter, HumanRoleRouter); node_factory, timeout, runner helpers, prompts; integration tests per-topology + per-topology-human unit tests; cognitive_load_proxy metric
  - `storage/` — SQLAlchemy models, async session, ParquetWriter, checkpointer wrapper (M3 complete); wall_time_s populated by run; Alembic 0003 (cognitive_load_proxy column)
  - `observability/` — ExperimentCallbackHandler (LangGraph async callbacks), serializers, structlog processors (M3 complete; M11-resync: filter_secrets)
  - `tasks/` — TaskSpec base, HumanEval/GSM8K/CommonGen/DABench loaders + evaluators (M10-resync)
  - `evaluation/` — LLM-as-judge, ground truth runners (dispatches via EVALUATORS.get), metrics, NASA-TLX persistence (M11 complete; M11-resync: _DEPS table dispatch pattern); human_sim_cognitive_load_proxy metric (M9.2)
  - `experiment/` — Pydantic config schemas (ExperimentConfig.human: HumanCfg | None; HumanCfg with role_router/role_table/role_router_model) + OmegaConf loader, single-run runner (`run_one` with HITL wiring: _build_role_router, _build_human_gateway_llm, dynamic_human_role SELECT, cognitive_load_proxy post-run write), Typer CLI (`atm run`); wall_time_s and cognitive_load_proxy capture (M6 + M9-M10 merge + M11-merge)
  - `analysis/` — Loaders, plots (M13); Oracle pipeline: OracleTable, build_leave_one_out_oracle, build_loo_from_rows, load_oracle_table (M8.7)
- `tests/` — unit, integration, fixtures
  - `fixtures/llm/` — FakeLLM scripted response YAMLs (planner_simple, executor_code_run, determinism_seed, m5_*.yaml, m7_*.yaml, m11_gsm8k_e2e_executor.yaml)
  - `fixtures/agents/` — Agent config fixtures (minimal_valid.yaml, debater_with_stance.yaml, debater_contra.yaml)
  - `unit/agents/` — 10 test files for Agent framework (config, tokens, basic, tool_loop, scratchpad_window, summarizer, budget_propagation, debater_stance, role_configs_load, executor_exit_criterion)
  - `unit/core/` — 150 tests for errors/types/reducers/state/public API, plus seed reproducibility (M11-resync)
  - `unit/llm/` — 110 tests for pricing/budget/retry/providers/fake/wrapper, plus model_version_snapshot (M11-resync)
  - `unit/topology/` — 6 test files (base, star, chain, mesh, debate, hierarchical) + conftest.py; extended coverage for human-aware topologies (M9.1)
  - `unit/topology/` — human-aware tests: test_star_human.py, test_chain_human.py, test_mesh_human.py, test_debate_human.py, test_hierarchical_human.py, test_adaptive_human.py (M9.1)
  - `integration/llm/` — 3 tests for LLM layer contract (end-to-end, budget exceed, replay round-trip)
  - `integration/topology/` — 4 test files (mesh_e2e, debate_e2e, hierarchical_e2e, all_topologies_sanity) covering e2e flows with FakeLLM
  - `integration/human/` — HITL e2e tests: test_hitl_star.py, test_hitl_chain.py, test_hitl_mesh.py, test_hitl_debate.py, test_hitl_hierarchical.py (M9)
  - `integration/evaluation/` — quality aggregation + GSM8K e2e test (M11-resync: replaced MMLU with GSM8K)
  - `integration/analysis/` — test_oracle_integration.py (M8.7)
- `conf/` — YAML configuration templates
  - `pricing.yaml` — per-1K-token prices for OpenAI/Anthropic/Cerebras/vLLM/fake models (version 1)
  - `agents/` — role-specific agent configs (planner.yaml, researcher.yaml, executor.yaml, critic.yaml, debater.yaml)
  - `topology/` — topology configs (star.yaml, chain.yaml, mesh.yaml, debate.yaml, hierarchical.yaml)
  - `tools_policy.yaml` — role-based tool assignment policy (M11-resync)
  - `oracle/` — oracle config templates (type_level_manual.yaml for M8.7)
- `alembic/` — database migrations (async template; head = 0003_runs_cognitive_load_proxy)
- `.github/workflows/` — CI/CD pipelines (M11-resync: ci.yml with lint/unit/integration matrix)
- `dev/` — documentation (PLAN.md, arch.md) and task tracking (done/ is current, active/ empty post-m10-merge-m8)

## Recent Updates (2026-05-14)

### M9 + M9.1 + M9.2 Complete: HITL & Adaptive Role Router
- **`src/atm/human/`** fully populated: HumanGateway protocol, LLMSimulatedGateway, CLI gateway.
- **All 5 topologies + adaptive.py:** Now accept `human_cfg`, `human_gateway_llm`, `role_router` kwargs in `.build()`.
- **RoleRouter protocol:** `FixedRoleRouter`, `RuleBasedRoleRouter`, `LLMRoleRouter`, `HumanRoleRouter` (role_router.py).
- **Helpers:** `_node_factory.py`, `_timeout.py`, `prompts.py`, `runner.py` (wired into experiment runner).
- **Default role table:** planning→Coordinator, execution→Peer, verification→Reviewer.
- **Cognitive load:** `Run.cognitive_load_proxy` column (Alembic 0003); `evaluation/metrics.py::human_sim_cognitive_load_proxy`.
- **Test coverage:** `tests/integration/human/test_hitl_*.py` (5 e2e tests, one per topology) + `tests/unit/topology/test_*_human.py` (6 unit tests).

### M8.7: Oracle Pipeline
- **`src/atm/analysis/oracle.py`:** `OracleTable` (frozen dataclass), `build_leave_one_out_oracle`, `build_loo_from_rows`, `load_oracle_table`.
- **Config:** `conf/oracle/type_level_manual.yaml`.
- **Integration test:** `tests/integration/analysis/test_oracle_integration.py`.
- **Status:** Complete.

### m10-merge-m8: Experiment Runner Wired for HITL
- **`experiment/runner.py`:** `_build_role_router`, `_build_human_gateway_llm` (inline); `dynamic_human_role` SELECT from human_interactions; `cognitive_load_proxy` post-run write.
- **`experiment/config.py`:** `ExperimentConfig.human: HumanCfg | None`; `HumanCfg` with `role_router`, `role_table`, `role_router_model` fields.
- **`runs.wall_time_s`:** Now populated by `_update_run_success/_failed` via `time.monotonic()` snapshot at run start.
- **Status:** Complete, merged to main.

### 2026-05-14 Hygiene Pass
- **`runs.wall_time_s`:** Wired via `time.monotonic()` snapshot; populated in `_update_run_success` and `_update_run_failed`.
- **Env-var unification:** `ATM_ENABLE_PG_TESTS` is now canonical (was split between `ATM_INTEGRATION_PG` and `ATM_ENABLE_PG_TESTS`). Updated in `tests/integration/conftest.py` and 5 test files.
- **Task tracking:** `dev/active/` is now empty; m10-merge-m8 moved to `dev/done/`.
- **Status:** Hygiene pass complete.

## Key Modules

### Core Types & State (`core/`)
- **Purpose:** Base data structures (Pydantic models + TypedDict) for message passing, errors, phase tracking; LangGraph-compatible reducers for merging agent state updates; reproducibility seeding.
- **Exports:** `AgentRole`, `AgentState`, `AtmError`, `BudgetEvent` (DB schema), `BudgetExceededError`, `GraphState`, `HumanContext`, `HumanResponse`, `HumanRole`, `LLMError`, `LLMResponse`, `Message`, `MessageKind`, `Phase`, `PhaseError`, `PhaseTransition`, `RunResult`, `SharedState`, `TaskResult`, `TaskSpec`, `TokenUsage`, `ToolCall`, `ToolError`, `ToolResult`, `TopologyTransition`, `dedup_by_id_reducer`, `merge_agent_states`, `seed_all`.
- **Status:** M1 complete + M2 + M11-resync extensions.

### LLM Layer (`llm/`)
- **Purpose:** Unified wrapper over LangChain chat models with usage/cost/retry, three-tier budget enforcement, prompt caching support, deterministic FakeLLM, and model version capture for reproducibility.
- **Exports (from `atm.llm`):** `LLMWrapper`, `BudgetTracker`, `BudgetSignal`, `BudgetLevel`, `FakeLLM`, `REPLAY_SCHEMA`, `Pricing`, `ModelPricing`, `RetryPolicy`, `with_retry`, `is_transient`, `build_openai`, `build_anthropic`, `build_cerebras`, `build_vllm`, `inject_cache_control`, `DEFAULT_CACHE_TTL`.
- **M11-resync:** `LLMWrapper.last_model_version: str | None` captures Anthropic/OpenAI fingerprints.
- **Test coverage:** 110 unit tests + 3 integration tests.
- **Status:** M2 complete + M5 + M11-resync extensions.

### Tools & Sandbox (`tools/`)
- **Purpose:** Tool protocol, registry with role-based policy support, 12 concrete implementations (4 global, 8 local), sandbox abstraction with dev (subprocess) and prod (Docker) modes.
- **M11-resync:** `ToolRegistry.tools_for(role: str)` with lazy YAML policy loading; `CodeSandbox` Protocol with `image_digest`.
- **Exports:** `Tool`, `ToolRegistry`, `ToolSchema`, `CodeSandbox`, `ExecResult`, `SandboxConfig`, `SubprocessSandbox`, `DockerSandbox`, all 12 tool classes, `build_default_registry`.
- **Test coverage:** 121 unit tests + 13 non-docker integration tests.
- **Status:** M4 complete + M11-resync extensions.

### Agent Framework (`agents/`) — M5 complete
- **Purpose:** LangGraph-compatible agent base class with scratchpad policy C (append-only journal, windowed view, optional LLM-based summarization), tool-loop integration, and role-based subclasses.
- **Exports:** `Agent`, `AgentConfig`, `load_agent_config`, `Planner`, `Researcher`, `Executor`, `Critic`, `Debater`.
- **Test coverage:** 10 unit test files (91 tests total).
- **Status:** M5 complete.

### Topology Framework (`topology/`) — M7 complete + M9.1
- **Purpose:** Topology protocol + Registry with 5 registered implementations covering different multi-agent coordination patterns; all accept HITL wiring.
- **Implementations:** Star, Chain, Mesh, Debate, Hierarchical (all M9.1 HITL-aware).
- **Key Interface:** `Topology.build(agents, cfg, human_cfg=None, human_gateway_llm=None, role_router=None) -> CompiledStateGraph`.
- **Test coverage:** 6 base unit test files + 6 human-aware unit test files + 4 integration e2e tests.
- **Status:** M7 complete + M9.1 HITL extensions.

### Phase Manager & Routers (`phases/`) — M8 complete
- **Purpose:** Monotonic Phase FSM (planning → execution → verification → done) with rule-based and LLM-based routers.
- **Exports:** `RuleBasedPhaseRouter`, `LLMPhaseRouter`, `PhaseLimits`, `PhaseGuard`, `PhaseRouter`.
- **Test coverage:** 23 unit tests.
- **Status:** M8 complete.

### Human Interaction Framework (`human/`) — M9 + M9.1 + M9.2 complete
- **Purpose:** HITL orchestration with role-aware human delegation, adaptive routing, cognitive load modeling.
- **Components:** HumanGateway protocol (LLMSimulatedGateway, CLI gateway); RoleRouter protocol (FixedRoleRouter, RuleBasedRoleRouter, LLMRoleRouter, HumanRoleRouter); node_factory, timeout helpers, prompts, runner wiring.
- **Integration:** All 5 topologies + adaptive.py wire HITL via `.build(human_cfg, human_gateway_llm, role_router)`.
- **Exports (from `atm.human`):** `HumanGateway`, `LLMSimulatedGateway`, `CLIGateway`, `RoleRouter`, `FixedRoleRouter`, `RuleBasedRoleRouter`, `LLMRoleRouter`, `HumanRoleRouter`.
- **Test coverage:** `tests/integration/human/test_hitl_*.py` (5 topologies) + `tests/unit/topology/test_*_human.py` (6 unit).
- **Metrics:** `human_sim_cognitive_load_proxy` (M9.2).
- **Status:** M9 + M9.1 + M9.2 complete.

### Storage Layer (`storage/`) — M3 complete + M9.2 + M11
- **Purpose:** Persistence layer for experiments with 6 SQLAlchemy 2.x models and Parquet bulk writers.
- **New columns:** `Run.cognitive_load_proxy` (M9.2), `Run.model_version_snapshot` (M11), `Run.sandbox_image_digest` (M11), `Run.wall_time_s` (M11 + 2026-05-14 hygiene).
- **Alembic:** Head = 0003_runs_cognitive_load_proxy.
- **Status:** M3 complete + M9.2 + M11 + hygiene extensions.

### Observability Layer (`observability/`) — M3 complete + M11-resync
- **Purpose:** LangGraph callback handler + serializers + structlog processors.
- **M11-resync:** `log_processors.py` with `filter_secrets` processor.
- **Status:** M3 complete + M11-resync structlog integration.

### Evaluation Framework (`evaluation/`) — M11 complete + M9.2
- **Purpose:** Post-hoc quality scoring, LLM-as-judge protocols, pure metric functions, NASA-TLX human-load model.
- **M11-resync:** `ground_truth.py` rewrite uses literal `_DEPS` table + `EVALUATORS.get(spec.evaluator_key)` dispatch.
- **M9.2:** `human_sim_cognitive_load_proxy` metric.
- **Status:** M11 complete + M9.2 extensions.

### Tasks & Evaluation (`tasks/`) — M10-resync complete
- **Purpose:** Benchmark task infrastructure with 4 loaders (HumanEval/GSM8K/CommonGen/DABench) and 4 evaluators.
- **Status:** M10-resync complete (MMLU removed, GSM8K primary).

### Experiment Runner & CLI (`experiment/`) — M6 + M9-M10 merge
- **Purpose:** Single-run orchestrator with seed, model snapshot, sandbox digest, and HITL wiring.
- **M11-resync:** Seed, model_version_snapshot, sandbox_image_digest capture.
- **m10-merge-m8:** HITL wiring (_build_role_router, _build_human_gateway_llm, dynamic_human_role, cognitive_load_proxy).
- **2026-05-14 hygiene:** wall_time_s populated via time.monotonic() at run start.
- **Status:** M6 + M9-M10 HITL merge + M11 reproducibility + hygiene complete.

### Analysis Layer (`analysis/`) — M8.7 complete
- **Purpose:** Oracle labels pipeline — produces `OracleTable` consumed by `OracleTopologyRouter` (M8.3) and LOO-plot analysis (M13 G9).
- **Exports:** `OracleTable`, `build_leave_one_out_oracle`, `build_loo_from_rows`, `load_oracle_table`.
- **Config:** `conf/oracle/type_level_manual.yaml`.
- **Test coverage:** `tests/integration/analysis/test_oracle_integration.py`.
- **Status:** M8.7 complete.

## Patterns & Conventions

### State Management
- **LangGraph state:** TypedDict-based `GraphState` with shared fields + per-agent `AgentState` reducer.
- **Scratchpad policy C (M5):** Append-only journal with windowed view; always write reasoning + tool_calls + observations.

### Reproducibility (M11-resync + 2026-05-14 hygiene)
- **Seed:** `seed_all(cfg.seed)` at head of `run_one` before topology execution.
- **Model version:** `LLMWrapper.last_model_version` captures Anthropic/OpenAI fingerprints; persisted in `runs.model_version_snapshot`.
- **Sandbox digest:** `DockerSandbox.image_digest` captures Python image SHA; persisted in `runs.sandbox_image_digest`.
- **Wall time:** `time.monotonic()` snapshot at run start; persisted in `runs.wall_time_s`.

### Configuration & Composition
- **YAML + OmegaConf + Pydantic:** YAML in `conf/`, composed via OmegaConf, validated by Pydantic schemas.
- **Role-based tool policy (M11-resync):** `conf/tools_policy.yaml` defines `global` list + `per_role` mapping; loaded lazily by `ToolRegistry.tools_for(role)`.
- **HITL config (M9-M10):** `ExperimentConfig.human: HumanCfg | None` with `role_router`, `role_table`, `role_router_model` fields.

### Testing
- **FakeLLM:** Deterministic mock, scripted fixtures keyed by `(agent_id, step_idx)` with role fallback.
- **Test count:** 1495 unit + 47 non-PG/non-Docker integration tests (all green).
- **Status markers:** `@pytest.mark.docker`, `@pytest.mark.network`, `@pytest.mark.pg` for conditional execution.

## External Dependencies

### Core (M0-M3)
- langchain, langchain-core, langchain-openai, langchain-anthropic, anthropic, openai, tiktoken, pyyaml, pyarrow, sqlalchemy[asyncio], asyncpg, omegaconf, pydantic, pytest, ruff, mypy, alembic, typer, huggingface-hub.

### M4-M7 additions
- docker>=7.1, ddgs>=9.13, httpx>=0.27, numpy>=1.26, faiss-cpu (optional), langgraph>=0.3.

### M11-resync
- structlog (optional; bootstrap gated by env var).

## Constraints & Notes
- **PG integration tests:** Controlled via `ATM_ENABLE_PG_TESTS` env var (unified; was split between `ATM_INTEGRATION_PG` and `ATM_ENABLE_PG_TESTS`).
- **Docker graceful fallback:** All docker-dependent code handles `(DockerException, OSError)` → None.
- **`torch` optional:** `seed_all` gracefully skips torch if not installed.
- **structlog optional:** Bootstrap can be disabled via `ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1`.
- **CI Python matrix:** `['3.11', '3.12']` for lint/unit; integration on 3.12 with postgres:16 service.
- **Alembic head:** 0003_runs_cognitive_load_proxy (post-M11 + M9.2).
- **Task tracking:** `dev/active/` empty post-m10-merge-m8; all milestones M0-M11 + M8.7 + M9.1 + M9.2 complete.

