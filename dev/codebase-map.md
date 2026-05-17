# Codebase Map
*Auto-generated. Last updated: 2026-05-17 (post-m14-streamlit-hitl)*

## Tech Stack
- **Language:** Python 3.11+
- **Package Manager:** uv
- **Orchestration:** LangGraph (multi-agent framework)
- **LLM SDK:** LangChain chat models (OpenAI/Anthropic/Cerebras/vLLM)
- **Database:** PostgreSQL 16 (async via SQLAlchemy 2.x + asyncpg, ORM with Alembic migrations)
- **Bulk Storage:** Apache Parquet (PyArrow) for experiment data
- **Config:** YAML + Pydantic + OmegaConf + grid sweep (M12)
- **UI:** Streamlit 1.36+ with nest_asyncio for async HITL (M14)
- **Testing:** pytest + pytest-asyncio
- **Linting/Types:** ruff + mypy (strict)
- **Containerization:** Docker (code sandbox + dev environment + Streamlit UI service)
- **Logging:** structlog (optional bootstrap in atm/__init__.py, controlled via ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1)

## Project Structure
- `src/atm/` — main package (Adaptive Topologies MAS, imported as `atm`)
  - `core/` — base types, state, errors, reproducibility helpers
  - `llm/` — LLMWrapper, providers (OpenAI/Anthropic/Cerebras/vLLM), budget, pricing, retry, FakeLLM, `factory.build_llm(model_id, ...)`
  - `tools/` — Tool protocol, registry (role-based policy), global tools, Docker/subprocess sandbox (M4)
  - `agents/` — Agent base + Planner/Researcher/Executor/Critic/Debater, scratchpad policy C (M5)
  - `topology/` — Topology protocol + Registry; 5 statics (Star/Chain/Mesh/Debate/Hierarchical) + adaptive. M14: all 6 accept `human_gateway`/`human_gateway_llm` kwargs (D7 canonical)
  - `phases/` — PhaseManager FSM, TopologyRouter, SwitchGuards, signals (M8)
  - `human/` — HumanGateway protocol; LLMSimulatedGateway, CLIGateway, RoleRouter (M9/9.1/9.2). M14: StreamlitHumanGateway + PG-backed `_queue.HumanRequestQueue`; `_node_factory.build_human_node_factory` accepts `fallback_llm`
  - `ui/` (M14 new) — Streamlit app (`app.py` with `nest_asyncio.apply()`), shared-secret auth (`auth.py` — hmac.compare_digest), NASA-TLX form (`tlx.py`, 6 scales — parity with `evaluation/tlx.py`), participant views (`views.py`: login/queue/response/proctor), async helpers (`_state.py: _run_sync`)
  - `storage/` — SQLAlchemy ORM (8 models incl. Run with replay tracking; M14 additions: StudySession, HumanRequestQueue, HumanInteraction.study_session_id FK); async session, ParquetWriter, checkpointer; alembic head = 0005
  - `observability/` — ExperimentCallbackHandler (async LangGraph callbacks), structlog. M14: `_handle_human_response` writes `tlx_scores`, `raw_tlx_score`, `study_session_id` in INSERT and UPDATE branches
  - `tasks/` — TaskSpec base + HumanEval/GSM8K/CommonGen/DABench (with `stage_workspace_for` from fix-dabench-task)
  - `evaluation/` — LLM-as-judge, ground truth runners (dispatch via `EVALUATORS.get`), metrics, NASA-TLX (`NasaTLX.raw_score`), `human_sim_cognitive_load_proxy`
  - `experiment/` — Pydantic config + grid/estimate (M12), OmegaConf loader (`loader.py`), `runner.run_one`/resume/replay, Typer CLI (`atm run`/`grid`/`resume`/`replay`/`estimate`/`status`). M14: `_build_human_gateway(human_cfg, llm_factory) -> (gateway, fallback_llm)` factory + wiring at both run sites
  - `analysis/` — `loaders.py` (7 loaders incl. `load_topology_transitions`, `load_human_interactions`), `metrics.py` (7 RQ2 helpers), `plots.py` (9 plots), `oracle.py` (build_leave_one_out_oracle, plot_oracle_vs_router) — M8.7 + M13
- `tests/` — unit + integration + fixtures
  - `unit/ui/` (M14 new) — `test_auth.py`, `test_tlx_form.py`
  - `unit/topology/` — M14 new: `test_topology_streamlit_dispatch.py` (6 topologies × `human_gateway` kwargs), `test_topology_fallback_no_llm.py` (8 fallback sites × no-`._llm` gateways)
  - `unit/human/` — `test_queue.py` (FakeSession queue tests), `test_streamlit_gateway.py` (FakeQueue gateway tests)
  - `unit/observability/` (M14 new) — `test_human_response_tlx.py` (INSERT + UPDATE branches × 17 edge cases)
  - `unit/experiment/` — `test_config_human.py` (incl. M14 fields), `test_runner_human_gateway.py` (M14: `_build_human_gateway` 4 scenarios), grid tests
  - `unit/storage/` — `test_models.py` (+44 M14 tests for StudySession/HumanRequestQueue)
  - `unit/tasks/` — humaneval/gsm8k/commongen/dabench
  - `integration/storage/` — `test_smoke_run.py`, `test_migration_0004.py`, `test_migration_0005.py` (M14: upgrade/downgrade round-trip + FK introspection)
  - `integration/human/` — M9.1 topology HITL tests; M14 new: `test_streamlit_e2e.py` (Chain + StreamlitGateway via PG queue → callback → DB), `test_queue_pg.py` (live PG CRUD); `conftest.py` with `truncate_m14_tables` fixture
- `alembic/` — migrations (head = `0005_m14_streamlit_hitl`)
- `.github/workflows/` — CI (lint/unit/integration matrix; PG service for integration job)
- `docker-compose.yml` — M14: ui service under `profiles: [ui]` (build from `Dockerfile.ui`, port 8501, read-only mounts)
- `Dockerfile.ui` (M14 new) — python:3.11-slim + uv + `uv sync --extra ui --frozen --no-dev`; streamlit on :8501
- `dev/` — PLAN.md, arch.md, audit-plan-gaps.md, `active/` (in-progress tasks), `done/` (completed tasks incl. m14)

## Recent Updates (2026-05-17)

### M14: Streamlit HITL UI, NASA-TLX, Proctoring Protocol (completed 2026-05-17)

8 implementation waves + 2 review rounds + lint pass. PR #17.

**New packages/modules:**
- `src/atm/ui/` — 6 modules (app, auth, _state, tlx, views, __init__) for live-participant HITL via Streamlit
- `src/atm/human/_queue.py` — PG-backed async queue (enqueue ON CONFLICT DO NOTHING; claim atomic UPDATE … RETURNING; submit_response WHERE response_json IS NULL; wait_for_response 1s polling; cancel)
- `src/atm/human/streamlit_gateway.py` — HumanGateway impl bridging runner ↔ UI through the queue; in-memory idempotency cache by (run_id, request_id); timeout returns `HumanResponse(timed_out=True, source='timeout')` for topology fallback chain to handle `llm_fallback` policy

**Extended modules:**
- `src/atm/storage/models.py` — StudySession (id/participant_id/status/consent_given/timestamps/proctor_notes/meta_json), HumanRequestQueue (id/run_id FK/request_id/context_json/response_json/status/claimed_by/timestamps; UNIQUE(run_id, request_id)), HumanInteraction.study_session_id (UUID FK → study_sessions.id, ON DELETE SET NULL)
- `src/atm/experiment/config.py` — HumanCfg.gateway Literal extended to include `"streamlit"`; new M14 fields (`queue_dsn`, `participant_id`, `study_session_id`, `shared_secret`, `fallback_llm_model`); back-compat default `gateway = "llm_simulated"`
- `src/atm/observability/callbacks.py` — `_handle_human_response` writes `tlx_scores` (JSONB), `raw_tlx_score` (float via NasaTLX), `study_session_id` (UUID parsed from `response_json["payload"]`) in BOTH INSERT and UPDATE branches (try/except guards on TLX score computation and UUID parse)
- `src/atm/experiment/runner.py` — `_build_human_gateway(human_cfg, llm_factory)` factory returning (HumanGateway | None, LLMWrapper | None). Lazy imports break circular dep with config.py. Called at run_one (~1253) and resume/replay (~2227); both sites pass `human_gateway` kwarg to topology builders alongside legacy `human_gateway_llm`
- `src/atm/human/_node_factory.py` — `build_human_node_factory` accepts `fallback_llm: Any = None` keyword; closure resolves fallback via `fallback_llm if not None else getattr(gateway, "_llm", None)` and guards `LLMSimulatedGateway(llm=_fb_llm) if _fb_llm is not None else None`

**Topology dispatch (6 files):** D7 canonical convention
- `chain.py` / `star.py` / `hierarchical.py`: new top-of-`build()` `gateway_llm = kwargs.get("human_gateway_llm")` (unconditional) + `human_gateway = kwargs.get("human_gateway")`; passes `fallback_llm=gateway_llm` to factory
- `mesh.py`: REASSIGNS existing `human_gateway` local (no rename); inline `human_peer_node` captures `gateway_llm` unconditionally
- `debate.py`: `kwargs.get("human_gateway") or kwargs.get("gateway")` (legacy back-compat for unit tests); `_gateway_llm` hoisted outside `if gateway is None:` block; `_build_human_judge_node` accepts `fallback_llm`; `_both_judge_postprocess` captures `_gateway_llm`
- `adaptive.py`: REUSES existing `_gateway` local; `llm_wrapper` hoisted outside `if _hitl_enabled:` block; `human_advisor_node` closure captures it

**8 fallback sites threaded** (A-H per plan): 5 factory parameters (`_node_factory.build_human_node_factory`, `chain._build_human_reviewer_node`, `debate._build_human_judge_node`, `hierarchical._build_human_top_reviewer_node`, `hierarchical._build_human_sub_reviewer_node`) + 3 closure captures (mesh `human_peer_node`, debate `_both_judge_postprocess`, adaptive `human_advisor_node`). Every fallback construction guarded with `if _fb_llm is not None else None` — pre-built Streamlit gateways without `._llm` cannot trigger AttributeError on `llm_fallback` timeout.

**Infrastructure:**
- `pyproject.toml` — `[project.optional-dependencies] ui = ["streamlit>=1.36,<2", "nest_asyncio>=1.5,<2"]`; mypy overrides for `streamlit.*` and `nest_asyncio`
- `docker-compose.yml` — ui service (profiles: [ui]; depends_on postgres healthy; read-only mounts `./src`, `./conf`, `./alembic`; port 8501; env: ATM_PG_DSN, ATM_UI_SECRET, ATM_UI_PORT)
- `Dockerfile.ui` — python:3.11-slim + uv + `uv sync --extra ui --frozen --no-dev`; CMD streamlit on :8501
- `.env.example` — placeholders ATM_PG_DSN, ATM_UI_SECRET, ATM_UI_PORT
- `alembic/versions/0005_m14_streamlit_hitl.py` — down_revision=0004; idempotent upgrade/downgrade

**Tests** (all green, 2044 unit + 9 PG-integration):
- unit ui (25), unit topology dispatch (16) + fallback (16), unit human queue (21) + streamlit_gateway (16), unit observability TLX (17), unit storage M14 (+44), unit experiment runner_human_gateway (4)
- integration: migration_0005 round-trip, queue_pg lifecycle, streamlit_e2e (full path: queue → gateway → callback → DB row with tlx_scores/raw_tlx_score/study_session_id)

**Docs (RU):** `dev/done/m14/README.md`, `dev/done/m14/proctor-protocol.md`, `dev/done/m14/participant-consent.md`

**Architectural decisions (D1-D7):**
- D1: PG-queue (not LangGraph `interrupt()`/`Command(resume)`) — preserves inline `gateway.request(...)` contract from M9.1
- D2: Polling 1s (TODO M14.1: LISTEN/NOTIFY)
- D3: TLX per-decision (not post-task)
- D4: Default `gateway = "llm_simulated"` — zero regression for E1–E4
- D5: `study_session_id` propagated via `HumanResponse.payload`
- D6: Streamlit as `[ui]` optional — `import atm.ui` works without the extra (lazy imports)
- D7: Canonical kwargs `human_gateway` / `human_gateway_llm`; locals `human_gateway` / `gateway_llm`

### fix-dabench-task: DABench Workspace Staging (2026-05-16)
- `tasks/dabench.py::stage_workspace_for` with path-traversal guards
- `experiment/runner.py::_augment_task_input_with_metadata` + `_pre_stage_workspace`
- `conf/agents/executor.yaml` extended

### M13: Analysis Tooling (complete)
- 7 loaders + 7 metrics + 9 plots + oracle visualization
- `scripts/gen_analysis_notebook.py` + `notebooks/analysis_template.ipynb`

### M12: Config Schema & Grid Sweep (complete)
- GridCfg/EstimateCfg, dotpath-validated sweep, cartesian expansion
- Run.replay_of, Run.host, Run.process_pid; composite index runs_exp_status_idx
- Alembic 0004; `atm grid` / `atm resume` / `atm replay` / `atm estimate` / `atm status`

### M9 + M9.1 + M9.2: HITL & Adaptive Role Router (complete)
- HumanGateway protocol; LLMSimulatedGateway/CLIGateway; topology HITL integration
- RoleRouter (Fixed/Rule/LLM/Human); cognitive_load_proxy column (alembic 0003) + metric

### M8.7: Oracle Pipeline
- OracleTable, build_leave_one_out_oracle, build_loo_from_rows, load_oracle_table

## Key Modules

### UI Layer (`ui/`) — M14
- `nest_asyncio.apply()` MUST be first after imports in `app.py`
- Pages: login → queue → response (with TLX inline) → proctor panel
- Auth: `hmac.compare_digest` of `HumanCfg.shared_secret` or `ATM_UI_SECRET` env
- TLX: 6 sliders → `raw_score` (performance scale inverted, matches `evaluation/tlx.py`)
- Response bundle: `{action, rationale, tlx_scores, payload: {study_session_id}}` → `queue.submit_response`

### HITL Queue (`human/_queue.py`, `human/streamlit_gateway.py`) — M14
- Queue: enqueue (idempotent), fetch_pending, claim (atomic), submit_response (idempotent), wait_for_response (polling), cancel
- StreamlitHumanGateway: bridges runner ↔ UI; idempotency cache; `payload['study_session_id']` backfill

### Storage Layer (`storage/`) — M3 + M12 + M14
- Run: 25 columns
- StudySession, HumanRequestQueue, HumanInteraction.study_session_id (M14)
- Alembic head = `0005_m14_streamlit_hitl`

### Experiment Runner & Config (`experiment/`) — M12 + M14 + fix-dabench-task
- `_build_human_gateway` factory (M14)
- GridCfg (M12); fail_fast; parallelism; seeds
- DABench workspace staging (fix-dabench-task)

### Topology Framework (`topology/`) — M6/M7 + M14
- 5 statics + adaptive; D7 canonical kwargs dispatch; 8 fallback sites with None guards

### Analysis Layer (`analysis/`) — M13
- Loaders, metrics, plots, oracle

## Patterns & Conventions

### M14 HITL & Streamlit
- Async/await throughout; `_run_sync(coro)` bridge for Streamlit; `nest_asyncio.apply()` at app startup
- Shared-secret auth: `hmac.compare_digest`; empty/None secret = dev mode disabled
- NASA-TLX: per-decision granularity; `raw_tlx_score` persisted alongside `tlx_scores` JSONB
- Inter-process HITL: single PG DSN shared between runner and UI processes; 1s polling
- D7: `human_gateway` kwarg key everywhere (debate also accepts legacy `gateway`); `human_gateway_llm` for fallback LLM

### Reproducibility
- `seed_all(cfg.seed)` at head of `run_one`; model_version_snapshot / sandbox_image_digest / wall_time_s captured

### Configuration (M12)
- YAML + OmegaConf + Pydantic; `GridCfg.sweep: dict[str, list[scalar]]`; `load_grid_configs` cartesian expansion

### Testing
- FakeLLM deterministic; scripted fixtures
- ~2050 unit tests, ~60 integration tests; PG tests gated by `ATM_ENABLE_PG_TESTS=1`

## Constraints & Notes
- **Alembic head:** `0005_m14_streamlit_hitl`
- **Streamlit:** `[ui]` optional install (`pip install -e ".[ui]"` / `uv sync --extra ui`)
- **Docker compose:** `--profile ui` enables ui service; `Dockerfile.ui` builds streamlit container
- **HITL gating:** `StreamlitHumanGateway` requires `human.queue_dsn` when `gateway="streamlit"`
- **Back-compat:** debate.py accepts both `kwargs["human_gateway"]` (canonical) and `kwargs["gateway"]` (legacy); all other topologies use `human_gateway` exclusively; default `HumanCfg.gateway = "llm_simulated"` keeps E1–E4 byte-identical
- **Fallback LLM threading:** every fallback construction guarded by `if _fb_llm is not None else None` — pre-built Streamlit gateways without `._llm` cannot AttributeError on `llm_fallback` timeout
- **mypy strict:** ui/ package fully typed; streamlit.* / nest_asyncio have `ignore_missing_imports` overrides
