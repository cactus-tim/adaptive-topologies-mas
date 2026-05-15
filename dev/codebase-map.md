# Codebase Map
*Auto-generated. Last updated: 2026-05-15 (post-m13-analysis-tooling)*

## Tech Stack
- **Language:** Python 3.11+
- **Package Manager:** uv
- **Orchestration:** LangGraph (multi-agent framework)
- **LLM SDK:** LangChain chat models (OpenAI/Anthropic/Cerebras/vLLM)
- **Database:** PostgreSQL 16 (async via SQLAlchemy 2.x + asyncpg, ORM with Alembic migrations)
- **Bulk Storage:** Apache Parquet (PyArrow) for experiment data
- **Config:** YAML + Pydantic + OmegaConf + grid sweep (M12)
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
  - `storage/` — SQLAlchemy models (6 models including `Run` with replay tracking), async session, ParquetWriter, checkpointer wrapper (M3 complete + M12); wall_time_s, cognitive_load_proxy, replay_of FK, host, process_pid columns; Alembic 0004 (replay and PID tracking)
  - `observability/` — ExperimentCallbackHandler (LangGraph async callbacks), serializers, structlog processors (M3 complete; M11-resync: filter_secrets)
  - `tasks/` — TaskSpec base, HumanEval/GSM8K/CommonGen/DABench loaders + evaluators (M10-resync)
  - `evaluation/` — LLM-as-judge, ground truth runners (dispatches via EVALUATORS.get), metrics, NASA-TLX persistence (M11 complete; M11-resync: _DEPS table dispatch pattern); human_sim_cognitive_load_proxy metric (M9.2)
  - `experiment/` — Pydantic config schemas + grid/estimate (M12), OmegaConf loader split into `loader.py`, single-run runner (`run_one` with HITL wiring), Typer CLI (`atm run`); wall_time_s and cognitive_load_proxy capture
  - `analysis/` — Full M13 surface: `loaders.py` (load_experiment, load_runs, load_llm_calls, load_llm_calls_for_experiment, load_topology_transitions, load_phases, load_human_interactions), `metrics.py` (7 RQ2 derived-metric helpers), `plots.py` (9 RQ1/RQ2/RQ4 plot functions), `oracle.py` (OracleTable, build_leave_one_out_oracle, build_loo_from_rows, load_oracle_table, plot_oracle_vs_router — M8.7 + M13)
- `tests/` — unit, integration, fixtures
  - `unit/experiment/` — test_config.py, test_smoke_yaml_loads.py, test_config_grid.py (M12), test_loader_grid_expansion.py (M12)
  - `integration/storage/` — test_smoke_run.py + test_migration_0004.py (M12)
- `alembic/` — database migrations (async template; head = 0004_m12_runs_replay_and_pid)
- `.github/workflows/` — CI/CD pipelines (M11-resync: ci.yml with lint/unit/integration matrix)
- `dev/` — documentation (PLAN.md, arch.md) and task tracking

## Recent Updates (2026-05-15)

### M13: Analysis Tooling (m13 — complete)
- **`src/atm/analysis/loaders.py`** (new): 7 async/sync loaders — `load_experiment`, `load_runs` (PG-backed); `load_llm_calls`, `load_llm_calls_for_experiment` (Parquet); `load_topology_transitions`, `load_phases`, `load_human_interactions` (PG + Parquet dual-source). JSONB fields decoded to dict; `raw_tlx_score` cast to float with NaN for empty strings.
- **`src/atm/analysis/metrics.py`** (new): 7 derived-metric helpers — `compute_hurt_rate`, `compute_guard_override_rate`, `compute_router_cost_share`, `compute_time_per_topology`, `compute_topology_switch_counts`, `compute_oracle_gap_manual`, `compute_oracle_gap_loo`. Pure functions, no side effects.
- **`src/atm/analysis/plots.py`** (new): 9 plot functions — RQ1: `plot_pareto`, `plot_topology_task_heatmap`, `plot_phase_timeline`; RQ2/G11: `plot_transition_timeline_quality`, `plot_guard_override_rate`, `plot_router_cost_share`, `plot_time_per_topology`, `plot_oracle_gap_loo`; RQ4: `plot_cognitive_load_boxplot`. matplotlib Agg backend forced at module top; seaborn imported lazily inside function bodies.
- **`src/atm/analysis/oracle.py`** (extended M8.7): Added `plot_oracle_vs_router` via `TYPE_CHECKING` pattern to keep oracle.py import-light; regression test `test_oracle_module_does_not_import_matplotlib` guards this.
- **`src/atm/analysis/__init__.py`** (updated): 29 public symbols in `__all__` covering all loaders, metrics, oracle, and plot functions.
- **`scripts/gen_analysis_notebook.py`** (new): Reproducible notebook generator; all 8 RQ2 symbols verified in generated cells.
- **`notebooks/analysis_template.ipynb`** (new): Committed template notebook (6 sections: setup, RQ1, RQ2/G11, RQ3, RQ4, export); passes `nbformat.validate()`.
- **Tests:** `test_loaders.py`, `test_metrics.py`, `test_plots.py`, `test_oracle_plot.py`, `test_oracle_no_matplotlib_import.py`, `test_notebook_generator.py`.

### M12: Config Schema & Grid Sweep (m12-config-schema — done)
- **`src/atm/experiment/config.py`:** Added `GridCfg` (dotpath-keyed sweep spec + parallelism + seeds + fail_fast), `EstimateCfg` (heuristic_tokens_per_call, calls_per_iter, use_historical), `_resolve_dotpath` validator (restricts sweep keys to typed scalar fields; handles both `typing.Union` AND Python 3.10+ `types.UnionType`), and `grid`/`estimate` fields to `ExperimentConfig`.
- **`src/atm/experiment/loader.py`** (new module): Extracted OmegaConf loader logic (`load_config`, `_merge_includes`); `load_config` re-exported from config.py for back-compat; new public `load_grid_configs(path, overrides)` expands `grid:` block via cartesian product over sweep dimensions × seeds. `_merge_includes` validates include paths stay within base_dir via `.resolve()` + `is_relative_to()` (path-traversal guard).
- **`src/atm/experiment/__init__.py`:** Added `GridCfg`, `EstimateCfg`, `load_grid_configs` to public exports.
- **`src/atm/storage/models.py`:** `Run` model gained `replay_of` (UUID nullable self-FK, ondelete=SET NULL, constraint `fk_runs_replay_of_runs`), `host` (String(64) nullable), `process_pid` (Integer nullable); added composite index `runs_exp_status_idx(exp_id, status)`.
- **`alembic/versions/0004_m12_runs_replay_and_pid.py`** (new migration): Upgrade adds 3 columns + self-FK + composite index; downgrade reverses cleanly (drop index → drop FK → drop columns).
- **Tests:** `test_config_grid.py` (11 cases: valid/invalid dotpaths, X | None handling, empty sweep, parallelism=0, no grid block); `test_loader_grid_expansion.py` (10 cases: 2×2, 2×2×2 sweeps, parallelism preservation, path-traversal); `test_migration_0004.py` (upgrade/downgrade round-trip via subprocess, introspection via information_schema + pg_constraint).
- **Status:** Complete.

### M9 + M9.1 + M9.2 Complete: HITL & Adaptive Role Router
- HumanGateway protocol, LLMSimulatedGateway, CLI gateway; all 5 topologies + adaptive accept HITL kwargs.
- RoleRouter protocol (Fixed/Rule/LLM/HumanRoleRouter); default role table: planning→Coordinator, execution→Peer, verification→Reviewer.
- Cognitive load proxy column (Alembic 0003) + metric in evaluation.

### M8.7: Oracle Pipeline
- `OracleTable`, `build_leave_one_out_oracle`, `build_loo_from_rows`, `load_oracle_table`; config template `conf/oracle/type_level_manual.yaml`.

### 2026-05-14 Pre-M12 Hygiene Pass
- `runs.wall_time_s` populated via `time.monotonic()`.
- Env-var unification: `ATM_ENABLE_PG_TESTS` is canonical.

## Key Modules

### Storage Layer (`storage/`) — M3 complete + M12
- **Run model columns:** `id`, `exp_id`, `topology`, `task_id`, `agent_set`, `human_role`, `seed`, `model`, `models_by_role_json` (JSONB), `model_version_snapshot` (JSONB), `sandbox_image_digest`, `status`, `finish_reason`, `budget_spent_usd`, `quality_score`, `cognitive_load_proxy`, `wall_time_s`, `iterations`, `started_at`, `finished_at`, `error`, `replay_of` (self-FK), `host`, `process_pid`.
- **Indices:** `runs_exp_id_idx(exp_id)`, `runs_topology_idx(topology)`, `runs_task_id_idx(task_id)`, `runs_status_idx(status)`, `runs_started_idx(started_at)`, `runs_exp_status_idx(exp_id, status)` (M12).
- **Alembic:** Head = `0004_m12_runs_replay_and_pid`.

### Analysis Layer (`analysis/`) — M13 complete
- **Loaders:** `load_experiment(dsn, exp_id)` → `Experiment`; `load_runs(dsn, exp_id)` → `DataFrame`; `load_llm_calls(parquet_dir, run_id)` → `DataFrame`; `load_llm_calls_for_experiment(parquet_dir, run_ids)` → `DataFrame`; `load_topology_transitions(dsn, parquet_dir, run_id)` → `DataFrame`; `load_phases(dsn, parquet_dir, run_id)` → `DataFrame`; `load_human_interactions(dsn, run_id)` → `DataFrame`.
- **Metrics (7):** `compute_hurt_rate`, `compute_guard_override_rate`, `compute_router_cost_share`, `compute_time_per_topology`, `compute_topology_switch_counts`, `compute_oracle_gap_manual`, `compute_oracle_gap_loo`.
- **Plots (9 + oracle):** `plot_pareto`, `plot_topology_task_heatmap`, `plot_phase_timeline`, `plot_transition_timeline_quality`, `plot_guard_override_rate`, `plot_router_cost_share`, `plot_time_per_topology`, `plot_oracle_gap_loo`, `plot_cognitive_load_boxplot`; `plot_oracle_vs_router` in `oracle.py`.
- **Oracle:** `OracleTable`, `build_leave_one_out_oracle`, `build_loo_from_rows`, `load_oracle_table`; config `conf/oracle/type_level_manual.yaml`.
- **Notebook:** `scripts/gen_analysis_notebook.py` generates `notebooks/analysis_template.ipynb` (6 sections; idempotent).

### Experiment Runner & Config (`experiment/`) — M12 partial (config-schema done)
- **GridCfg (M12):** `sweep: dict[str, list[scalar]]` (dotpath-validated) + `parallelism: int` + `seeds: list[int]` + `fail_fast: bool`.
- **EstimateCfg (M12):** `heuristic_tokens_per_call`, `calls_per_iter`, `use_historical`.
- **Loader (M12):** `load_config(path, overrides)` + new `load_grid_configs(path, overrides)` for cartesian expansion (moved into `loader.py`; re-exported from `config.py`).
- **CLI:** Typer `atm run` command. **TODO (other M12 blocks in flight):** `atm grid` (m12-grid-runner), `atm resume/replay/reconcile` (m12-resume-replay), `atm estimate/status` (m12-estimate-status-cli).

### LLM Layer (`llm/`)
- LLMWrapper (usage/cost/retry/budget); FakeLLM (scripted/echo/replay modes); `factory.build_llm(model_id, ...)`.
- M11-resync: `LLMWrapper.last_model_version` captures fingerprints.

### Topology Framework (`topology/`)
- 5 topologies + adaptive, all HITL-aware.

### Phase Manager & Routers (`phases/`)
- Monotonic Phase FSM, RuleBasedPhaseRouter + LLMPhaseRouter, TopologyRouter, SwitchGuards.

## Patterns & Conventions

### Reproducibility
- `seed_all(cfg.seed)` at head of `run_one`.
- `runs.model_version_snapshot`, `runs.sandbox_image_digest`, `runs.wall_time_s` captured per-run.

### Configuration (M12)
- YAML + OmegaConf + Pydantic. Grid sweep via `GridCfg.sweep: dict[str, list[scalar]]` with dotpath validators; `load_grid_configs` expands via cartesian product over sweep dimensions × `seeds`.

### Testing
- FakeLLM deterministic mock; scripted fixtures.
- 1513+ unit tests, ~47 integration tests; PG tests gated by `ATM_ENABLE_PG_TESTS=1`.

## Constraints & Notes
- **Alembic head:** `0004_m12_runs_replay_and_pid`.
- **Task tracking:** `dev/done/m12-config-schema`. In flight (parallel orchestrators on `feat/m12`): `m12-grid-runner`, `m12-resume-replay`, `m12-estimate-status-cli`. All depend on `m12-config-schema`. After they merge, `m12-grid-integration` will be the final block.
- **Grid sweep:** Sweep values restricted to scalars (str/int/float/bool); dotpath validation via `_resolve_dotpath` (handles `X | None` natively); self-FK `replay_of` on `runs.id` (ondelete=SET NULL).
- **Back-compat:** `from atm.experiment.config import load_config` re-exported from `loader.py`.
