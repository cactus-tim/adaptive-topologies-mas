# Codebase Map
*Auto-generated. Last updated: 2026-05-13*

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

## Project Structure
- `src/atm/` — main package (Adaptive Topologies MAS, imported as `atm`)
  - `core/` — base types, state, errors (Message, ToolCall, Phase, AgentState, GraphState)
  - `llm/` — LLMWrapper, providers (OpenAI/Anthropic/Cerebras/vLLM), budget tracking, pricing, retry, fake LLM, `factory.build_llm(model_id, ...)` provider-prefix router (supports `fake:scripted/echo/replay`)
  - `tools/` — Tool protocol, registry, global tools, Docker/subprocess sandbox (M4)
  - `agents/` — Agent base class, Planner/Researcher/Executor/Critic/Debater roles, scratchpad policy C (M5)
  - `topology/` — Topology protocol + Registry, 5 implementations: Star + Chain (M6); Mesh + Debate + Hierarchical (M7 complete)
  - `phases/` — PhaseManager FSM, TopologyRouter, SwitchGuards, signals (M8)
  - `human/` — HumanGateway protocol, LLMSimulatedGateway, CLI gateway, HumanRoleRouter + 3 strategies (M9 + M9.2)
  - `storage/` — SQLAlchemy models, async session, ParquetWriter, checkpointer wrapper (M3 complete)
  - `observability/` — ExperimentCallbackHandler (LangGraph async callbacks), serializers (M3 complete)
  - `tasks/` — TaskSpec base, HumanEval/MMLU/Creative/Analysis implementations (M10)
  - `evaluation/` — `human_sim_cognitive_load_proxy` metric (M9.2); LLM-as-judge, ground-truth runners (M11 planned)
  - `experiment/` — Pydantic config schemas + OmegaConf loader, single-run runner (`run_one`), Typer CLI (`atm run`), inline evaluator (M6 complete; grid + sweep in M12)
  - `analysis/` — Loaders, plots (M13)
- `tests/` — unit, integration, fixtures
  - `fixtures/llm/` — FakeLLM scripted response YAMLs (planner_simple, executor_code_run, determinism_seed, m5_*.yaml, m7_*.yaml)
  - `fixtures/agents/` — Agent config fixtures (minimal_valid.yaml, debater_with_stance.yaml, debater_contra.yaml)
  - `unit/agents/` — 10 test files for Agent framework (config, tokens, basic, tool_loop, scratchpad_window, summarizer, budget_propagation, debater_stance, role_configs_load, executor_exit_criterion)
  - `unit/core/` — 150 tests for errors/types/reducers/state/public API
  - `unit/llm/` — 110 tests for pricing/budget/retry/providers/fake/wrapper
  - `unit/topology/` — 5 test files (base, star, chain, mesh, debate, hierarchical) covering routing/stopping/registration/invariants; 6 files total with conftest.py
  - `integration/llm/` — 3 tests for LLM layer contract (end-to-end, budget exceed, replay round-trip)
  - `integration/topology/` — 4 test files (mesh_e2e, debate_e2e, hierarchical_e2e, all_topologies_sanity) covering e2e flows with FakeLLM
- `conf/` — YAML configuration templates
  - `pricing.yaml` — per-1K-token prices for OpenAI/Anthropic/Cerebras/vLLM/fake models (version 1)
  - `agents/` — role-specific agent configs (planner.yaml, researcher.yaml, executor.yaml, critic.yaml, debater.yaml)
  - `topology/` — topology configs (star.yaml, chain.yaml, mesh.yaml, debate.yaml, hierarchical.yaml)
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
- **Exports (from `atm.llm`):** `LLMWrapper`, `BudgetTracker`, `BudgetSignal` (runtime event), `BudgetLevel`, `FakeLLM`, `REPLAY_SCHEMA`, `Pricing`, `ModelPricing`, `RetryPolicy`, `with_retry`, `is_transient`, `build_openai`, `build_anthropic`, `build_cerebras`, `build_vllm`, `inject_cache_control`, `DEFAULT_CACHE_TTL`.
- **Submodules:**
  - `pricing.py` — `ModelPricing` (frozen Pydantic, optional float fields), `Pricing.from_yaml` + `cost` (auto-detects OpenAI vs Anthropic cache convention) + `estimate` pre-call helper.
  - `budget.py` — `BudgetLevel` StrEnum, `BudgetSignal` frozen Pydantic (runtime warn/exceed event — not to be confused with `core.types.BudgetEvent` which is the DB persistence schema), `BudgetTracker` (asyncio.Lock, three-tier check/record, warn+exceed callbacks, sync or async callable).
  - `retry.py` — `RetryPolicy` frozen dataclass, `is_transient` duck-types `.status_code` 429/≥500 in addition to `retry_on` tuple, `with_retry` with exponential backoff + full jitter, wraps exhaustion as `LLMError` preserving `__cause__`.
  - `wrapper.py` — `LLMWrapper.ainvoke` flow: convert Message→BaseMessage via `Message.to_lc`, pre-call budget estimate (tiktoken or heuristic), optional `inject_cache_control` for Anthropic, `with_retry` wrapping `_llm.ainvoke`, parse OpenAI or Anthropic usage_metadata, compute cost via Pricing, record budget, build LLMResponse with started_at + latency_ms. `astream` raises NotImplementedError. Accepts `llm=` injection for tests. **M5 addition:** `model_id` property exposes the underlying model identifier.
  - `fake.py` — `FakeLLM(mode=scripted|replay|echo)`: scripted reads YAML fixtures by `(agent_id, step_idx)` with role fallback and asyncio.Lock; replay reads pyarrow Table with `REPLAY_SCHEMA` and maps `call_id → LLMResponse.id`; echo mirrors last user message content. `latency_ms=0` constant ensures bit-identical determinism.
  - `providers/openai.py` — `build_openai` strips `openai:` prefix, `api_key="EMPTY"` default.
  - `providers/anthropic.py` — `build_anthropic` + `inject_cache_control` (deep-copies last message, handles multi-modal `content=list[dict]`, always returns NEW list — never mutates input).
  - `providers/cerebras.py` — `build_cerebras` strips `cerebras:` prefix, `api_key="EMPTY"` default; active models: `llama3.1-8b` (deprecation 2026-05-27), `gpt-oss-120b`; requires `langchain-cerebras>=0.5,<0.6`.
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
- **Test fixtures:** `tests/fixtures/agents/{minimal_valid,debater_with_stance,debater_contra}.yaml`, `tests/fixtures/llm/m5_{agent_basic,agent_tool_loop,agent_tool_loop_two_iters,agent_scratchpad,agent_summarizer_primary,agent_summarizer_secondary,executor_exit}.yaml`.
- **Status:** M5 complete.

### Topology Framework (`topology/`) — M7 complete
- **Purpose:** Topology protocol + Registry with 5 registered implementations covering different multi-agent coordination patterns.
- **Topology Registry:** `TopologyRegistry` with `@register(name)` decorator; all 5 topologies registered via guarded imports in `__init__.py`.
- **Implementations:**
  - **Star (M6):** Central hub + spokes; coordinator agent routes to workers sequentially.
  - **Chain (M6):** Linear pipeline; each agent's output feeds to the next.
  - **Mesh (M7):** Broadcast-bus graph with round-robin + priority dispatcher; post-process node (mesh_broadcast) consolidates outputs, vote tally + consensus voting (threshold-based); stopping precedence: consensus → max_rounds → loop.
  - **Debate (M7):** Parallel fan-out (planner → debater_pro + debater_contra in same super-step); judge loop validates & decides; judge_postprocess extracts winner/approved; stopping precedence: judge-decides → max_rounds → loop.
  - **Hierarchical (M7):** Strict 2-level invariant with top coordinator + 2 compiled subgraphs (teams); top_coord and sub_coord are rule-based closures (not agents); final_answer = JSON-concat of team results; stopping precedence: all-finalized → max_rounds → loop.
- **Key Interface:** `Topology.build(agents, cfg) -> CompiledStateGraph`; schemas must overlap (M6 M7 pattern).
- **Config locations:** `conf/topology/{star,chain,mesh,debate,hierarchical}.yaml`
- **Test coverage:**
  - **Unit tests (6 files):** test_base.py, test_star.py, test_chain.py, test_mesh.py (22 tests), test_debate.py (17 tests), test_hierarchical.py (13 tests)
  - **Integration e2e tests (4 files):** test_mesh_e2e.py (2 e2e), test_debate_e2e.py (2 e2e), test_hierarchical_e2e.py (1 e2e), test_all_topologies_sanity.py (11 sanity tests: 1 registry + 5 smoke + 5 precedence)
- **LLM fixtures:** `tests/fixtures/llm/m7_{mesh_consensus,mesh_max_rounds,debate_judge_decides,debate_max_rounds,hierarchical_finalize}.yaml`
- **Exports (from `atm.topology`):** `Topology`, `TopologyRegistry`, `TopologyConfig`, `Star`, `Chain`, `Mesh`, `Debate`, `Hierarchical`.
- **Dependencies:** pydantic, langchain-core, langgraph (≥0.3 for CompiledStateGraph + parallel fan-out + subgraph compile).
- **Status:** M7 complete (5 topologies registered, 11 primary tests + sanity coverage).

### Phase Manager & Routers (`phases/`) — M8.1 + M8.2 complete
- **Purpose:** Monotonic Phase FSM (planning → execution → verification → done) with rule-based and LLM-based routers; JSON parse + monotonicity validation + fallback on LLM errors.
- **Exports (from `atm.phases`):** `RuleBasedPhaseRouter`, `LLMPhaseRouter`, `PhaseLimits`, `PhaseGuard`, `PhaseRouter`.
- **Submodules:**
  - `manager.py` — `PhaseGuard` (Callable type alias `Callable[[GraphState], bool]`), `PhaseLimits` (frozen Pydantic, per-phase iter caps), `PhaseRouter` (Protocol, runtime_checkable, `async decide(GraphState) -> PhaseDecision`), `RuleBasedPhaseRouter` (async: 4 builtin signal guards + custom override + iter-cap timeout, decided_by='rule'), `LLMPhaseRouter` (async ainvoke → JSON parse → monotonicity validation → fallback to rule on any error, decided_by='llm_router' on success, WARNING on fallback).
- **Core type additions (M8, in `atm.core.types`):** `TopologyDecision` (frozen Pydantic, router_cost_usd>=0 validator), `PhaseDecision` (frozen Pydantic, monotonic next_phase).
- **Test coverage:** 23 unit tests in `tests/unit/phases/test_manager.py` — 4 signal guards (ready_for_execution, ready_for_verification, critic_approved, iter_caps), iter-cap advance, terminal DONE, custom guard override, PhaseLimits frozen, LLMPhaseRouter happy-path + 4 fallback scenarios (JSON parse error, non-monotonic phase, network failure, invalid JSON), all using AsyncMock.
- **Status:** M8.1 + M8.2 complete (PhaseRouter Protocol, RuleBasedPhaseRouter, LLMPhaseRouter with async/await, JSON validation + monotonicity checks, comprehensive fallback coverage).

### `src/atm/human/` — HITL gateways (M9)

- `gateway.py` — `HumanGateway` Protocol (runtime_checkable) + re-exports of `HumanContext`/`HumanResponse`/`HumanRole` from `atm.core.types`; `TimeoutPolicy` Literal alias.
- `llm_simulated.py` — `LLMSimulatedGateway`: implements Protocol via `LLMWrapper`; role-aware prompts via `build_role_prompt`; single JSON-retry on bad LLM output; in-process idempotency cache `dict[(run_id, request_id), HumanResponse]` + `asyncio.Lock`; `source="llm_sim"` always.
- `cli_gateway.py` — `CLIGateway`: renders context to stdout, reads decision via `asyncio.to_thread(input, ...)`; no external dependencies; idempotency cache; `source="human"`; invalid action → `allowed_actions[0]` fallback.
- `_timeout.py` — `request_with_timeout(gateway, ctx, *, request_id, timeout_s, policy, llm_fallback_gateway=None) -> HumanResponse`: three policies (`fail` / `llm_fallback` / `skip`); `timeout_s=None` passes through with no deadline; `policy="llm_fallback"` + `llm_fallback_gateway=None` → `ValueError` at call time; fallback response always has `source="fallback"`.
- `runner.py` — `run_with_human(graph, initial_state, *, thread_id, gateway, ...)`: forward-compatible interrupt/resume orchestrator for M14+; detects `__interrupt__` key in ainvoke result; in-process idempotency cache on `(thread_id, request_id)`; `MaxInteractionsExceededError` guard (default `max_interactions=10`); >1 interrupt in single result → `RuntimeError`.
- `prompts.py` — `ROLE_SYSTEM_PROMPTS: dict[HumanRole, str]` for 5 roles (Coordinator/Reviewer/Judge/Peer/Monitor); `build_role_prompt(role, ctx) -> tuple[str, str]` renders (system_prompt, user_prompt); `allowed_actions` mentioned in user prompt.

**Exports (from `atm.human`):** `HumanGateway`, `HumanContext`, `HumanResponse`, `HumanRole`, `LLMSimulatedGateway`, `CLIGateway`, `request_with_timeout`, `run_with_human`, `MaxInteractionsExceededError`, `build_role_prompt`, `ROLE_SYSTEM_PROMPTS`, `build_human_node_factory`, `HumanRoleRouter`, `FixedRoleRouter`, `RuleBasedRoleRouter`, `LLMRoleRouter` (M9.2).

**M9.2 — `role_router.py`:**
- `HumanRoleRouter` — `@runtime_checkable` Protocol; `async decide(phase, state) -> HumanRole`; structural subtyping.
- `FixedRoleRouter(role)` — constant; back-compat default when `role_router="fixed"`.
- `RuleBasedRoleRouter(table, fallback)` — phase→HumanRole table; `DEFAULT_ROLE_TABLE = {planning:COORDINATOR, execution:PEER, verification:REVIEWER, done:REVIEWER}`; `from_yaml(path)` loader; ValueError on unknown phase/role string.
- `LLMRoleRouter(llm, fallback_router)` — LLM-judge via `_RoleRouterDecision` Pydantic model; `agent_id="role_router"`; falls back to rule-router on any failure (malformed JSON, unknown role, LLM exception, validation error).

**Persistence:** emitted via `adispatch_custom_event("human_request"|"human_response", ...)` from the `human_reviewer` node in Chain topology; handled by `ExperimentCallbackHandler` (`callbacks.py`) which writes to `human_interactions` table. Single-writer invariant (arch.md §10.2); idempotency by PG UNIQUE constraint `uq_human_interactions_run_request` on `(run_id, request_id)` (migration `0002_human_interactions_idempotency.py`). `on_custom_event` handles two new event names: `human_request` (INSERT ON CONFLICT DO NOTHING) and `human_response` (UPDATE WHERE response_json IS NULL).

**Topology integration (M9.1):** All 6 topologies now support HITL. Each inserts a HITL node when `human_cfg.enabled=True`; without it, the graph is byte-for-byte identical to pre-M9.1 behaviour.

| Topology | HITL node | Key design | `HumanCfg.extra` keys |
|---|---|---|---|
| Chain | `human_reviewer` | after `executor`, before `critic`; reference implementation from M9 | — |
| Star | `human_reviewer` | after `critic_postprocess`; `_route_from_coord` reads + clears `signals["human_phase_override"]` (`"advance"/"stay"/"finalize"`) | — |
| Mesh | `human_peer` | added to `agent_order`; skipped until `activation_round` (default 2); `mesh_postprocess` sets `signals["consensus_pending"]` on split-vote | `activation_round: int` |
| Debate | `human_judge` / `judge_combined` | mode `"human"`: replaces LLM judge; mode `"both"`: sequential composite (LLM → human → aggregate, human > critic); mode `"critic"` (default): no-op | `judge: "critic" \| "human" \| "both"` |
| Hierarchical | `human_top_reviewer` / `human_sub_reviewer` | `scope="top"` (default): fires once after top coordinator; `scope="sub_team"`: fires inside each compiled subgraph (WARNING: CLIGateway not supported in subgraph, deferred to M9.2) | `scope: "top" \| "sub_team"` |
| Adaptive | `human_advisor` | advisory mode (default): writes hint to `signals["human_advisor_hint"]`, routing unchanged; override mode: replaces `TopologyDecision`, writes `TopologyTransition.decided_by="human_override"`, SwitchGuards respected | `human_can_override_router: bool` |

**`HumanCfg.extra` (M9.1):** Optional `dict[str, Any] | None = None` field added to `HumanCfg`. Per-topology keys documented in the table above; typos pass silently (caller responsibility, plan GAP-2).

**`runs.human_role` (M9.1):** Now written by `Runner._insert_run` as `cfg.human.role.value` when `human_cfg.enabled=True`, `None` otherwise. Field was declared in `storage/models.py` since M3 but not written until M9.1.

**`decided_by` Literal (M9.1):** Both `TopologyDecision.decided_by` and `TopologyTransition.decided_by` in `src/atm/core/types.py` now include `"human_override"`. No Alembic migration needed (DB column is `String(24)`, 14-char value fits).

**M9.2 role routing across topologies:** all 6 HITL topologies (`Chain`/`Star`/`Mesh`/`Debate`/`Hierarchical`/`Adaptive`) accept `role_router: HumanRoleRouter | None = None` kwarg on `build(...)`. When `None` (or `HumanCfg.role_router=="fixed"`), behaviour is byte-identical to pre-M9.2 (uses `human_cfg.role`). When set, the HITL node closure calls `await router.decide(Phase(shared.get("phase", default)), shared)` and uses the result as `HumanContext.role`. Debate captures router in **outer `build()` scope** (used by both `judge_pre` and `judge_combined` via `_both_judge_postprocess` closure); Hierarchical threads router through `_build_subgraph_with_human` → `_build_human_sub_reviewer_node`. SharedState TypedDict NOT modified — role persisted via existing callback to `human_interactions.role`.

**M9.2 runner finalization (`experiment/runner.py`):**
- `_build_role_router(human_cfg, llm_factory) -> HumanRoleRouter | None`: returns `None` for `fixed` (back-compat short-circuit), `RuleBasedRoleRouter` for `rule`, `LLMRoleRouter(fallback=rule)` for `llm`, `ValueError` for unknown.
- After successful run: SQL `SELECT role FROM human_interactions WHERE run_id=:rid ORDER BY requested_at DESC LIMIT 1` → `runs.human_role`; `await human_sim_cognitive_load_proxy(session, run_id)` → `runs.cognitive_load_proxy`. Both wrapped in `try/except` — failure → NULL, run still completes. `_update_run_failed` path (budget-exceeded) unchanged: NULL acceptable.

**Cross-topology acceptance test:** `tests/integration/human/test_hitl_cross_topology.py` — 6 parametrized scenarios (one per topology), PG-gated via `ATM_ENABLE_PG_TESTS=1`. Each scenario asserts `human_interactions count >= 1` and `runs.human_role = 'reviewer'`.

**M9.2 integration test:** `tests/integration/human/test_role_router_star_e2e.py` (`@pytest.mark.requires_postgres`) — Rule router across 3 phases → ≥2 distinct roles in `human_interactions`, `runs.human_role == roles[-1]`, `runs.cognitive_load_proxy > 0`.

**Config:** `conf/human/llm_simulated.yaml`, `conf/human/cli.yaml`, `conf/human/role_table.yaml` (M9.2 phase→role) define `HumanCfg` parameters. `HumanCfg` M9.2 fields: `role_router: Literal["fixed","rule","llm"]="fixed"`, `role_table: dict[str,str] | None = None`, `role_router_model: str | None = None`.

**Status:** M9 complete, M9.1 complete (all 6 topologies HITL-enabled), M9.2 complete (Adaptive Role Router + cognitive_load_proxy).

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
- `versions/0002_human_interactions_idempotency.py` (M9) — adds UNIQUE constraint `uq_human_interactions_run_request` on `(run_id, request_id)`.
- `versions/0003_runs_cognitive_load_proxy.py` (M9.2) — adds nullable `cognitive_load_proxy: Float` column to `runs`.

### Tasks & Evaluation (`tasks/`, `evaluation/`)
- **`evaluation/metrics.py` (M9.2):** `human_sim_cognitive_load_proxy(session, run_id) -> float` computes `alpha*count + beta*mean(ctx_len_bytes) + gamma*mean(latency_s)` from `human_interactions`; `_load_weights(path)` reads `conf/evaluation/cognitive_load.yaml` with safe defaults fallback.
- **Exports (M10-M11 planned):** `TaskSpec`, `TaskRegistry`, evaluators, NASA-TLX aggregator.
- **Status:** M0 skeleton; M9.2 metric complete; M10-M11 evaluators not started.

### Experiment Runner & CLI (`experiment/`)
- **Exports (M12 planned):** Pydantic schemas, `ConfigLoader`, `Runner`, `GridExecutor`, CLI commands.
- **M9.2 enhancements:** `_build_role_router(human_cfg, llm_factory)` factory; `run_one()` post-success step queries last `human_interactions.role` and computes `cognitive_load_proxy`, writes both into `runs` via extended `_update_run_success(human_role, cognitive_load_proxy)` signature (both kwargs default `None` for back-compat).
- **Status:** M0 skeleton, M6 basic runner complete, M9.2 role routing + metrics finalization complete; M12 grid + sweep not started.

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
- **Topology configs (M6-M7):** `conf/topology/*.yaml` define topology-specific parameters (max_iterations, extra consensus/debate/hierarchy params).
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
- **M5-M7 test coverage:** 10 agent test files (91 tests) + 6 unit topology test files (67+ tests) + 4 integration topology test files (15+ sanity/e2e tests), all passing.
- **Key test categories (M2-M7 cumulative):**
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
  - Topology routing, stopping precedence, consensus voting (mesh), parallel fan-out (debate), 2-level hierarchy invariant (hierarchical) (M7).

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
- **langchain-cerebras** (0.5.x): ChatCerebras; pinned `<0.6` to keep `langchain-core<0.4`.
- **tiktoken** (≥0.8): OpenAI token counting (Anthropic uses heuristic fallback).
- **pyyaml** (≥6): Fixture + pricing + agent config parsing.
- **pyarrow** (≥16): Replay Table schema + bulk experiment data.

### M4 additions (Tools & Sandbox)
- **docker** (≥7.1): Container runtime for DockerSandbox.
- **ddgs** (≥9.13): DuckDuckGo search backend.
- **httpx** (≥0.27): Async HTTP client for UrlFetchTool.
- **numpy** (≥1.26): TF-IDF vectorization for SemanticSearchTool.
- **faiss-cpu** (≥1.11): Optional extra for semantic search acceleration.

### M6-M7 additions (Topologies)
- **langgraph** (≥0.3): StateGraph, parallel fan-out, compiled subgraphs, CompiledStateGraph.

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
- **Structure:** `tests/unit/{core,llm,tools,agents,topology}/`, `tests/integration/{llm,tools,topology}/`, `tests/fixtures/{llm,agents,tools/corpus}/`.
- **Total tests:** ~580+ (91 agents + 121 tools + 13 non-docker integration tools + 67+ unit topology + 15+ integration topology + 263 legacy unit/integration core+llm).
- **M4 test markers:** `@pytest.mark.docker` and `@pytest.mark.network` gated via ATM_ENABLE_DOCKER_TESTS and ATM_ENABLE_NETWORK_TESTS env vars; 29 such tests auto-skipped otherwise.
- **M5 test markers:** None (all agent tests run by default).
- **M7 test markers:** None (all topology tests run by default).
- **Mocking approach:** FakeLLM with YAML fixtures for unit + integration; FakeDDGS for search tool testing; mock docker for sandbox unit tests.
- **Coverage target:** All core modules + critical paths; M4 tools + M5 agents + M7 topologies fully tested; lower coverage on M8+ until implementations exist.
