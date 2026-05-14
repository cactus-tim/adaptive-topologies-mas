"""Experiment runner — run_one(cfg) orchestrator.

Full lifecycle:
    1. _ensure_experiment — INSERT experiments ON CONFLICT DO NOTHING RETURNING id
    2. _insert_run        — INSERT runs with status="running"
    3. Build checkpointer via checkpointer_scope(pg_dsn)
    4. Build LLMWrapper per role from cfg.model.by_role (+ default fallback)
    5. Build agents via _build_agents(cfg, llms)
    6. Build topology via TopologyRegistry.get(cfg.topology.name).build(agents, topology_cfg)
    7. Build ParquetWriter + ExperimentCallbackHandler
    8. Build initial_state with all 14 SharedState keys
    9. ainvoke(initial_state, config={callbacks, configurable:{thread_id}})
    10. parquet_writer.close()  — ALWAYS BEFORE _update_run_*
    11. compute_quality(spec, answer, sandbox, judge_llm, run_seed) via aggregator
    12. _update_run_success or _update_run_failed
    13. engine.dispose() in finally

Flush-before-update invariant (arch.md §10.3):
    parquet_writer.close() MUST be called BEFORE _update_run_success/_update_run_failed
    to ensure all buffered observability data is flushed to disk before the run record
    is marked complete.

Exception handling:
    - BudgetExceededError → status="budget_exceeded", flush + update, return RunResult
    - Any other Exception  → status="failed", flush + update, re-raise
"""

from __future__ import annotations

import subprocess
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import sqlalchemy as sa
import structlog
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from atm.core.errors import BudgetExceededError
from atm.core.seed import seed_all
from atm.core.types import HumanRole, Phase
from atm.evaluation.aggregator import compute_quality
from atm.evaluation.metrics import human_sim_cognitive_load_proxy
from atm.experiment.config import ExperimentConfig, HumanCfg
from atm.human.role_router import (
    DEFAULT_ROLE_TABLE,
    HumanRoleRouter,
    LLMRoleRouter,
    RuleBasedRoleRouter,
)
from atm.llm.budget import BudgetLevel, BudgetTracker
from atm.llm.factory import build_llm
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.checkpointer import checkpointer_scope
from atm.storage.models import Base, Experiment, FinishReason, Run
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_engine, create_session_factory, session_scope
from atm.tasks import resolve_spec
from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox
from atm.topology.base import TopologyConfig, TopologyRegistry

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# _build_role_router — factory that constructs the appropriate HumanRoleRouter
# ---------------------------------------------------------------------------


def _build_role_router(
    human_cfg: HumanCfg | None,
    llm_factory: Callable[[], Any] | None = None,
) -> HumanRoleRouter | None:
    """Build a HumanRoleRouter from HumanCfg.role_router strategy.

    Returns None for role_router="fixed" (or when human_cfg is None) as the
    back-compat short-circuit signal — topologies skip dynamic role lookup
    and use human_cfg.role directly, preserving pre-m9.2 behaviour byte-for-byte.
    """
    if human_cfg is None or human_cfg.role_router == "fixed":
        return None

    strategy = human_cfg.role_router

    if strategy == "rule":
        if human_cfg.role_table is not None:
            table: dict[Phase, HumanRole] = {
                Phase(k): HumanRole(v) for k, v in human_cfg.role_table.items()
            }
        else:
            table = dict(DEFAULT_ROLE_TABLE)
        fallback: HumanRole = human_cfg.role
        return RuleBasedRoleRouter(table=table, fallback=fallback)

    if strategy == "llm":
        if human_cfg.role_table is not None:
            rule_table: dict[Phase, HumanRole] = {
                Phase(k): HumanRole(v) for k, v in human_cfg.role_table.items()
            }
        else:
            rule_table = dict(DEFAULT_ROLE_TABLE)
        rule_router = RuleBasedRoleRouter(table=rule_table, fallback=human_cfg.role)
        llm = llm_factory() if llm_factory is not None else None
        return LLMRoleRouter(llm=llm, fallback=rule_router)

    raise ValueError(f"unknown role_router: {strategy!r}")


# ---------------------------------------------------------------------------
# RunResult — public contract for run_one() return value
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    """Result of a single experiment run.

    Fields:
        run_id:       UUID of the run row in PostgreSQL.
        exp_id:       UUID of the experiment row in PostgreSQL.
        status:       Terminal status string — one of FinishReason.value or "failed".
        metrics:      Dict with at minimum: quality_score (float), cost_usd (float),
                      iters (int).
        final_answer: The final answer string extracted from the graph state.
    """

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    exp_id: UUID
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    final_answer: str = Field(default="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_pricing() -> Pricing:
    """Load Pricing from conf/pricing.yaml, with fallback to empty pricing table."""
    candidates = [
        Path("conf/pricing.yaml"),
        Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return Pricing.from_yaml(candidate)
    # Fallback: empty pricing table (zero costs for all models)
    return Pricing(version=1, models={})


def _get_git_sha() -> str | None:
    """Return the current HEAD git SHA (7-char short), or None on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


async def _ensure_experiment(
    session_factory: async_sessionmaker[AsyncSession],
    cfg: ExperimentConfig,
) -> UUID:
    """INSERT experiment ON CONFLICT DO NOTHING RETURNING id, with SELECT fallback.

    Uses PostgreSQL INSERT ... ON CONFLICT DO NOTHING RETURNING id to handle
    concurrent runners safely (TOCTOU-free).

    Args:
        session_factory: Async session factory.
        cfg:             Experiment configuration.

    Returns:
        UUID of the experiment row.
    """
    git_sha = _get_git_sha()

    async with session_factory() as session:
        # Try INSERT ... ON CONFLICT DO NOTHING RETURNING id
        stmt = (
            sa.dialects.postgresql.insert(Experiment)
            .values(
                id=uuid.uuid4(),
                name=cfg.name,
                config_snapshot={
                    "name": cfg.name,
                    "topology": cfg.topology.name,
                    "task_name": cfg.task.name,
                    "seed": cfg.seed,
                },
                git_sha=git_sha,
                started_at=sa.func.now(),
                status="running",
            )
            .on_conflict_do_nothing(index_elements=["name"])
            .returning(Experiment.id)
        )

        result = await session.execute(stmt)
        row = result.fetchone()

        if row is not None:
            exp_id: UUID = row[0]
        else:
            # Conflict — SELECT the existing row
            select_result = await session.execute(
                sa.select(Experiment.id).where(Experiment.name == cfg.name)
            )
            exp_id = select_result.scalar_one()

        await session.commit()
        return exp_id


async def _insert_run(
    session_factory: async_sessionmaker[AsyncSession],
    exp_id: UUID,
    cfg: ExperimentConfig,
) -> UUID:
    """INSERT a new run row with status="running".

    Args:
        session_factory: Async session factory.
        exp_id:          UUID of the parent experiment.
        cfg:             Experiment configuration.

    Returns:
        UUID of the new run row.
    """
    run_id = uuid.uuid4()

    human_role_value: str | None = (
        cfg.human.role.value if cfg.human is not None and cfg.human.enabled else None
    )

    async with session_scope(session_factory) as session:
        run = Run(
            id=run_id,
            exp_id=exp_id,
            topology=cfg.topology.name,
            task_id=cfg.task.name,
            agent_set=cfg.agents.set,
            seed=cfg.seed,
            model=cfg.model.default,
            models_by_role_json=dict(cfg.model.by_role),
            model_version_snapshot={},
            status="running",
            budget_spent_usd=Decimal("0"),
            started_at=datetime.now(UTC),
            human_role=human_role_value,
        )
        session.add(run)

    return run_id


async def _update_run_success(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: UUID,
    exp_id: UUID,
    quality_score: float | None,
    budget_spent_usd: float,
    iterations: int,
    human_role: str | None = None,
    cognitive_load_proxy: float | None = None,
    wall_time_s: float | None = None,
) -> None:
    """Update run row to completed status.

    M9.2 fields: ``human_role`` overrides the static cfg.human.role when a
    dynamic role was selected by the RoleRouter during the run.
    ``cognitive_load_proxy`` is the NASA-TLX proxy aggregated post-run from
    ``human_interactions``. Both default to None for back-compat with non-HITL runs.
    ``wall_time_s`` is elapsed wall-clock seconds from run start to finish.
    """
    finished_at = datetime.now(UTC)
    values: dict[str, Any] = {
        "status": "completed",
        "finish_reason": FinishReason.SUCCESS.value,
        "quality_score": quality_score,
        "budget_spent_usd": Decimal(str(budget_spent_usd)),
        "iterations": iterations,
        "finished_at": finished_at,
        "cognitive_load_proxy": cognitive_load_proxy,
        "wall_time_s": wall_time_s,
    }
    if human_role is not None:
        values["human_role"] = human_role
    async with session_scope(session_factory) as session:
        await session.execute(sa.update(Run).where(Run.id == run_id).values(**values))


async def _update_run_failed(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: UUID,
    status: str,
    finish_reason: str,
    quality_score: float | None,
    budget_spent_usd: float,
    iterations: int,
    error_text: str | None = None,
    human_role: str | None = None,
    cognitive_load_proxy: float | None = None,
    wall_time_s: float | None = None,
) -> None:
    """Update run row to failed/budget_exceeded status (M9.2-aware)."""
    values: dict[str, Any] = {
        "status": status,
        "finish_reason": finish_reason,
        "quality_score": quality_score,
        "budget_spent_usd": Decimal(str(budget_spent_usd)),
        "iterations": iterations,
        "finished_at": datetime.now(UTC),
        "error": error_text,
        "cognitive_load_proxy": cognitive_load_proxy,
        "wall_time_s": wall_time_s,
    }
    if human_role is not None:
        values["human_role"] = human_role
    async with session_scope(session_factory) as session:
        await session.execute(sa.update(Run).where(Run.id == run_id).values(**values))


def _build_initial_state(cfg: ExperimentConfig, run_id: UUID) -> dict[str, Any]:
    """Build the initial GraphState with all 14 SharedState keys populated.

    Required SharedState keys (arch.md §3.2):
        task_id, task_input, phase, iteration, iter_total, active_topology,
        final_answer, signals, phase_started_at_iter, topology_started_at_iter,
        topology_history, topology_switch_count, phase_history, human_requests,
        human_responses, broadcast_bus

    Args:
        cfg:    Experiment configuration.
        run_id: UUID of the current run.

    Returns:
        A fully populated GraphState dict.
    """
    return {
        "shared": {
            "run_id": run_id,
            "task_id": cfg.task.name,
            "task_input": cfg.task.input,
            "phase": Phase.PLANNING,
            "iteration": 0,
            "iter_total": 0,
            # For adaptive meta-graph, leave active_topology unset so the
            # TopologyRouter picks a real sub-topology (e.g. "linear") on the
            # first tick instead of recursing into the meta-graph itself.
            "active_topology": None if cfg.topology.name == "adaptive" else cfg.topology.name,
            "final_answer": "",
            "signals": {},
            "phase_started_at_iter": 0,
            "topology_started_at_iter": 0,
            "topology_history": [],
            "topology_switch_count": 0,
            "phase_history": [],
            "human_requests": [],
            "human_responses": [],
            "broadcast_bus": [],
        },
        "agents": {},
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


def _build_llm_wrappers(
    cfg: ExperimentConfig,
    budget: BudgetTracker,
    pricing: Pricing,
) -> dict[str, LLMWrapper]:
    """Build one LLMWrapper per role from cfg.model.by_role + default fallback.

    For "fake:scripted" providers, fixture paths are resolved from
    ``cfg.model.fake_fixtures`` (a dict mapping role → path string). If no
    fixture is configured for a scripted role, falls back to FakeLLM(mode="echo")
    with a warning log.

    For "fake:echo" providers, injects FakeLLM(mode="echo").
    For real providers, init_chat_model is used (via build_llm / LLMWrapper).

    Returns:
        Dict mapping role name to LLMWrapper.
    """
    roles = ["planner", "executor", "critic", "researcher"]
    wrappers: dict[str, LLMWrapper] = {}

    for role in roles:
        model_id = cfg.model.get_model_for(role)
        provider = model_id.split(":", 1)[0] if ":" in model_id else model_id
        bare_model = model_id.split(":", 1)[1] if ":" in model_id else model_id

        if provider == "fake" and bare_model == "scripted":
            # Resolve fixture path from cfg.model.fake_fixtures if available
            fixture_str = cfg.model.fake_fixtures.get(role)
            if fixture_str is None:
                logger.warning(
                    "fake:scripted model requested but no fixture configured; "
                    "falling back to FakeLLM(mode='echo')",
                    role=role,
                    hint="Set model.fake_fixtures.<role>=<path> in experiment config",
                )
            fixture_path = Path(fixture_str) if fixture_str else None
            wrappers[role] = build_llm(
                model_id=model_id,
                pricing=pricing,
                budget=budget,
                fixture_path=fixture_path,
            )
        else:
            wrappers[role] = build_llm(
                model_id=model_id,
                pricing=pricing,
                budget=budget,
            )

    return wrappers


def _build_agents(
    cfg: ExperimentConfig,
    llms: dict[str, LLMWrapper],
    conf_dir: Path | None = None,
    tool_registry: Any = None,
) -> dict[str, Any]:
    """Build Agent instances for each role in the agent set.

    Reads agent configs from conf/agents/<role>.yaml.
    For M6, the 3 active agents are Planner, Executor, Critic.
    Researcher is instantiated but unreferenced by Star/Chain graphs.

    Dict keys are the role names (e.g. "planner", "executor", "critic") so that
    topology nodes can look up agents by role directly (e.g. agents["critic"]).
    The agent_id attribute on each Agent instance is also set to the role name.

    The "critic" role is instantiated as a ``Critic`` subclass so that
    Agent.step() emits MessageKind.DECISION (required by _critic_postprocess).
    All other roles use the base ``Agent`` class.

    Args:
        cfg:      Experiment configuration.
        llms:     Dict of role → LLMWrapper.
        conf_dir: Root directory for conf/ files (defaults to project root).

    Returns:
        Dict mapping role name to Agent instance.
    """
    from atm.agents.base import Agent
    from atm.agents.config import load_agent_config
    from atm.agents.critic import Critic
    from atm.tools.base import ToolRegistry

    if conf_dir is None:
        # Try to find conf dir relative to this file or cwd
        here = Path(__file__).parent
        for candidate in [here.parent.parent.parent / "conf", Path("conf")]:
            if candidate.exists():
                conf_dir = candidate
                break
        if conf_dir is None:
            conf_dir = Path("conf")

    agents_conf_dir = conf_dir / "agents"
    roles = ["planner", "executor", "critic", "researcher"]
    agents: dict[str, Any] = {}

    for role in roles:
        yaml_path = agents_conf_dir / f"{role}.yaml"
        if not yaml_path.exists():
            logger.warning("agent config not found, skipping", role=role, path=str(yaml_path))
            continue

        try:
            agent_cfg = load_agent_config(yaml_path)
        except Exception as e:
            logger.warning("failed to load agent config", role=role, error=str(e))
            continue

        llm = llms.get(role) or llms.get("planner")
        if llm is None:
            logger.warning("no LLM wrapper for role, skipping agent", role=role)
            continue

        tools = tool_registry if tool_registry is not None else ToolRegistry()
        # Use role name as both the dict key and the agent_id so topology nodes
        # (e.g. state["agents"]["critic"]) can find the agent by role directly.
        # BUG-3 fix: use Critic subclass for the critic role so step() emits DECISION.
        agent_cls: type = Critic if role == "critic" else Agent
        agents[role] = agent_cls(
            agent_id=role,
            cfg=agent_cfg,
            llm=llm,
            tools=tools,
        )

    return agents


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


async def run_one(cfg: ExperimentConfig) -> RunResult:
    """Execute a full experiment run lifecycle.

    See module docstring for complete lifecycle description.

    Args:
        cfg: Fully-validated ExperimentConfig.

    Returns:
        RunResult with status, metrics, and final_answer.

    Raises:
        Exception: Re-raises any non-BudgetExceededError exceptions after
                   flushing parquet and updating the run row to status="failed".
    """
    pg_dsn = cfg.observability.pg_dsn
    parquet_root = Path(cfg.observability.parquet_dir)

    # Create engine and session factory
    engine: AsyncEngine = create_engine(pg_dsn)
    session_factory = create_session_factory(engine)

    # Ensure schema exists (idempotent)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logger.warning("schema creation failed (may already exist)", error=str(e))

    exp_id: UUID | None = None
    run_id: UUID | None = None
    parquet_writer: ParquetWriter | None = None
    judge_llm: LLMWrapper | None = None
    quality_score: float | None = 0.0
    budget_spent: float = 0.0
    iterations: int = 0
    final_answer: str = ""
    llms: dict[str, LLMWrapper] | None = None  # populated in try; read in finally
    sandbox: Any | None = None  # populated in try; read in finally for digest capture

    # Seed all RNGs for reproducibility before any stochastic work.
    seed_all(cfg.seed)

    # Wall-clock start — captured once and reused across all terminal branches
    # (success / budget_exceeded / failed) so runs.wall_time_s reflects total
    # elapsed time including DB setup, topology build, evaluation, and finalize.
    _run_started_monotonic: float = time.monotonic()

    def _elapsed_s() -> float:
        return max(0.0, time.monotonic() - _run_started_monotonic)

    try:
        # Step 1: ensure experiment row exists
        exp_id = await _ensure_experiment(session_factory, cfg)

        # Step 2: insert run row
        run_id = await _insert_run(session_factory, exp_id, cfg)

        log = logger.bind(run_id=str(run_id), exp_id=str(exp_id))
        log.info("run started", topology=cfg.topology.name, task=cfg.task.name)

        # Step 3: build budget tracker and pricing
        budget = BudgetTracker(
            per_call_usd=cfg.budget.per_call_usd,
            per_run_usd=cfg.budget.per_run_usd,
            per_experiment_usd=cfg.budget.per_experiment_usd,
        )
        pricing = _load_pricing()

        # Build judge LLM wrapper (shares the run's BudgetTracker).
        # If construction fails (e.g. missing OPENAI_API_KEY when default judge_model
        # is "openai:gpt-4o" but the run uses fake providers), silently fall back to
        # judge_llm=None — the aggregator and ground_truth dispatch handle this and
        # judge-required evaluators will surface a clear error at evaluation time.
        try:
            judge_llm = build_llm(
                model_id=cfg.evaluation.judge_model,
                pricing=pricing,
                budget=budget,
            )
        except Exception as judge_build_exc:
            logger.warning(
                "judge LLM construction failed — proceeding without judge",
                judge_model=cfg.evaluation.judge_model,
                error=str(judge_build_exc)[:200],
            )
            judge_llm = None

        # Step 4: build LLM wrappers per role (assigned to outer-scope var for finally block)
        llms = _build_llm_wrappers(cfg, budget, pricing)

        # Step 5: build agents — share a single tool registry pre-populated with the
        # M4 default toolset so cfg.tools entries (code_run, file_*, calculator, etc.)
        # resolve at agent dispatch time instead of emitting "tool name not in registry"
        # warnings. Workspace and corpus dirs live under the parquet root scoped to
        # this run; SubprocessSandbox is the dev default (Docker required for prod).
        from atm.tools.defaults import build_default_registry
        from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox as _Sandbox

        tools_workspace = parquet_root / "workspace" / str(run_id)
        tools_workspace.mkdir(parents=True, exist_ok=True)
        tools_corpus = tools_workspace / "_corpus"
        tools_corpus.mkdir(exist_ok=True)
        try:
            tool_registry = build_default_registry(
                workspace=tools_workspace,
                corpus_dir=tools_corpus,
                sandbox=_Sandbox(),
            )
        except Exception:
            logger.warning(
                "build_default_registry failed; falling back to empty registry", exc_info=True
            )
            tool_registry = None
        agents = _build_agents(cfg, llms, tool_registry=tool_registry)

        # Step 7: build parquet writer and callback
        parquet_writer = ParquetWriter(
            root=parquet_root,
            run_id=run_id,
            exp_id=exp_id,
        )

        callback = ExperimentCallbackHandler(
            run_id=run_id,
            exp_id=exp_id,
            session_factory=session_factory,
            parquet_writer=parquet_writer,
            budget_warn_threshold=Decimal(str(cfg.budget.per_run_usd * 0.8)),
            budget_exceed_threshold=Decimal(str(cfg.budget.per_run_usd)),
        )

        # Step 8: build initial state
        initial_state = _build_initial_state(cfg, run_id)

        # Step 6: build topology
        topology_cfg = TopologyConfig(
            name=cfg.topology.name,
            max_iterations=cfg.topology.max_iterations,
            extra=cfg.topology.extra,
        )

        # Step 9: run graph
        final_state: dict[str, Any]

        # HITL wiring (M9/M9.1/M9.2) — built only when cfg.human.enabled.
        # Topology builders that pre-date M9.1 ignore unknown kwargs; the spread
        # is deferred via conditional dict so legacy build(agents, topology_cfg,
        # checkpointer=...) keeps working byte-identical when HITL is off.
        human_gateway_llm: LLMWrapper | None = None
        if cfg.human is not None and cfg.human.enabled and cfg.human.gateway == "llm_simulated":
            human_model_id = cfg.human.model or cfg.model.default
            # For fake:scripted, look up the fixture under fake_fixtures["human"]
            # so the simulator returns canned JSON instead of falling back to echo
            # mode (which would emit the prompt verbatim and crash the gateway).
            human_fixture_str = (
                cfg.model.fake_fixtures.get("human")
                if human_model_id.startswith("fake:scripted")
                else None
            )
            human_gateway_llm = build_llm(
                model_id=human_model_id,
                pricing=pricing,
                budget=budget,
                fixture_path=Path(human_fixture_str) if human_fixture_str else None,
            )

        def _role_router_llm_factory() -> LLMWrapper:
            role_model_id = (
                cfg.human.role_router_model if cfg.human is not None else None
            ) or cfg.model.default
            return build_llm(
                model_id=role_model_id,
                pricing=pricing,
                budget=budget,
            )

        role_router = _build_role_router(cfg.human, llm_factory=_role_router_llm_factory)

        async with checkpointer_scope(pg_dsn) as checkpointer:
            # BUG-4 fix: TopologyRegistry.get() returns the CLASS, not an instance.
            # Instantiate the class before calling build() so that self is bound.
            topology_cls = TopologyRegistry.get(cfg.topology.name)
            topology_instance = topology_cls()
            compiled_graph = topology_instance.build(
                agents,
                topology_cfg,
                checkpointer=checkpointer,
                human_cfg=cfg.human,
                human_gateway_llm=human_gateway_llm,
                role_router=role_router,
            )

            # Adaptive meta-graph runs many super-steps per task tick (4 nodes
            # per loop iteration x max_iterations of subgraph dispatch).
            # Default LangGraph recursion_limit=25 is too low.
            recursion_limit = max(100, (cfg.topology.max_iterations or 30) * 4 + 20)
            final_state = await compiled_graph.ainvoke(
                initial_state,
                config={
                    "callbacks": [callback],
                    "configurable": {"thread_id": str(run_id)},
                    "recursion_limit": recursion_limit,
                },
            )

        # Extract metrics
        shared_final: dict[str, Any] = final_state.get("shared") or {}
        final_answer = str(shared_final.get("final_answer") or "")
        iterations = int(shared_final.get("iter_total") or 0)

        # Get actual budget spent from tracker
        budget_spent = budget.totals.get(BudgetLevel.RUN, 0.0)

        # Step 11: evaluate
        sandbox = SubprocessSandbox()  # assigned to outer-scope var for finally digest capture
        spec = resolve_spec(cfg.task)
        if spec is None:
            logger.debug(
                "resolve_spec returned None (inline-prompt path); setting quality_score=0.0",
                task=cfg.task.name,
            )
            quality_score = 0.0
        else:
            try:
                quality_score, _details = await compute_quality(
                    spec,
                    final_answer,
                    sandbox=sandbox,
                    judge_llm=judge_llm,
                    run_seed=cfg.seed,
                )
            except Exception:
                logger.warning("compute_quality raised unexpectedly; setting quality_score=None")
                quality_score = None

        # Step 10: FLUSH PARQUET BEFORE UPDATE (invariant)
        await parquet_writer.close()

        # Step 12a: resolve dynamic human_role + cognitive_load_proxy (M9.2 RQ4).
        # Both ops wrapped in try/except — DB or metric failure must NEVER prevent
        # _update_run_success from completing. Budget-failed/exception branches
        # skip this block; both fields remain NULL there (acceptable per arch).
        dynamic_human_role: str | None = None
        dynamic_cog_proxy: float | None = None

        try:
            async with session_scope(session_factory) as _hi_session:
                last_role_result = await _hi_session.execute(
                    sa.text(
                        "SELECT role FROM human_interactions"
                        " WHERE run_id = :rid"
                        " ORDER BY requested_at DESC, id DESC LIMIT 1"
                    ).bindparams(rid=run_id)
                )
                last_role_row = last_role_result.fetchone()
                if last_role_row is not None:
                    dynamic_human_role = str(last_role_row[0])
        except Exception:
            logger.warning(
                "failed to query last human_interactions.role; human_role left as-is",
                run_id=str(run_id),
                exc_info=True,
            )

        try:
            async with session_scope(session_factory) as _cog_session:
                dynamic_cog_proxy = await human_sim_cognitive_load_proxy(_cog_session, run_id)
        except Exception:
            logger.warning(
                "failed to compute cognitive_load_proxy; leaving NULL",
                run_id=str(run_id),
                exc_info=True,
            )

        # Step 12: update run to completed
        await _update_run_success(
            session_factory,
            run_id,
            exp_id,
            quality_score=quality_score,
            budget_spent_usd=budget_spent,
            iterations=iterations,
            human_role=dynamic_human_role,
            cognitive_load_proxy=dynamic_cog_proxy,
            wall_time_s=_elapsed_s(),
        )

        log.info(
            "run completed",
            quality_score=quality_score,
            budget_spent_usd=budget_spent,
            iterations=iterations,
        )

        return RunResult(
            run_id=run_id,
            exp_id=exp_id,
            status="completed",
            metrics={
                "quality_score": quality_score,
                "cost_usd": budget_spent,
                "iters": iterations,
            },
            final_answer=final_answer,
        )

    except BudgetExceededError as exc:
        log = logger.bind(
            run_id=str(run_id) if run_id else "?", exp_id=str(exp_id) if exp_id else "?"
        )
        log.warning("budget exceeded", error=str(exc))

        _budget_exc_spec = resolve_spec(cfg.task)
        if _budget_exc_spec is None or not final_answer or judge_llm is None:
            quality_score = 0.0
        else:
            try:
                _sandbox = SubprocessSandbox()
                quality_score, _details = await compute_quality(
                    _budget_exc_spec,
                    final_answer,
                    sandbox=_sandbox,
                    judge_llm=judge_llm,
                    run_seed=cfg.seed,
                )
            except Exception:
                logger.warning(
                    "compute_quality raised unexpectedly on budget exceeded; "
                    "setting quality_score=None"
                )
                quality_score = None

        # FLUSH PARQUET BEFORE UPDATE (invariant — even on budget exceeded)
        if parquet_writer is not None:
            try:
                await parquet_writer.close()
            except Exception:
                log.error("parquet close failed after budget exceeded")

        if run_id is not None and exp_id is not None:
            try:
                await _update_run_failed(
                    session_factory,
                    run_id,
                    status="budget_exceeded",
                    finish_reason=FinishReason.BUDGET_EXCEEDED.value,
                    quality_score=quality_score,
                    budget_spent_usd=budget_spent,
                    iterations=iterations,
                    wall_time_s=_elapsed_s(),
                )
            except Exception:
                log.error("run update failed after budget exceeded")

        return RunResult(
            run_id=run_id or uuid.uuid4(),
            exp_id=exp_id or uuid.uuid4(),
            status="budget_exceeded",
            metrics={
                "quality_score": quality_score,
                "cost_usd": budget_spent,
                "iters": iterations,
            },
            final_answer=final_answer,
        )

    except Exception as exc:
        log = logger.bind(
            run_id=str(run_id) if run_id else "?", exp_id=str(exp_id) if exp_id else "?"
        )
        log.error("run failed", error=str(exc))

        _exc_spec = resolve_spec(cfg.task)
        if _exc_spec is None or not final_answer or judge_llm is None:
            quality_score = 0.0
        else:
            try:
                _sandbox = SubprocessSandbox()
                quality_score, _details = await compute_quality(
                    _exc_spec,
                    final_answer,
                    sandbox=_sandbox,
                    judge_llm=judge_llm,
                    run_seed=cfg.seed,
                )
            except Exception:
                logger.warning(
                    "compute_quality raised unexpectedly on run failure; setting quality_score=None"
                )
                quality_score = None
        error_text = traceback.format_exc()

        # FLUSH PARQUET BEFORE UPDATE (invariant — even on failure)
        if parquet_writer is not None:
            try:
                await parquet_writer.close()
            except Exception:
                log.error("parquet close failed after run failure")

        if run_id is not None and exp_id is not None:
            try:
                await _update_run_failed(
                    session_factory,
                    run_id,
                    status="failed",
                    finish_reason=FinishReason.ERROR.value,
                    quality_score=quality_score,
                    budget_spent_usd=budget_spent,
                    iterations=iterations,
                    error_text=error_text[:2000],
                    wall_time_s=_elapsed_s(),
                )
            except Exception:
                log.error("run update failed after run failure")

        raise

    finally:
        # Persist actual model versions reported by providers (G2 reproducibility).
        # Runs in finally so even failed runs record provider fingerprints.
        if run_id is not None and llms:
            try:
                snap = {
                    role: w.last_model_version
                    for role, w in llms.items()
                    if getattr(w, "last_model_version", None) is not None
                }
                if snap:
                    async with session_factory() as _snap_session:
                        await _snap_session.execute(
                            sa.update(Run)
                            .where(Run.id == run_id)
                            .values(model_version_snapshot=snap)
                        )
                        await _snap_session.commit()
            except Exception as _snap_exc:
                logger.warning(
                    "model_version_snapshot UPDATE failed",
                    error=str(_snap_exc)[:200],
                )

        # Persist sandbox image digest (G2 reproducibility).
        # Only fires when sandbox has a non-None image_digest (i.e. DockerSandbox).
        if run_id is not None and sandbox is not None:
            _sandbox_digest = getattr(sandbox, "image_digest", None)
            if _sandbox_digest is not None:
                try:
                    async with session_factory() as _dig_session:
                        await _dig_session.execute(
                            sa.update(Run)
                            .where(Run.id == run_id)
                            .values(sandbox_image_digest=_sandbox_digest[:80])
                        )
                        await _dig_session.commit()
                except Exception as _dig_exc:
                    logger.warning(
                        "sandbox_image_digest UPDATE failed",
                        error=str(_dig_exc)[:200],
                    )

        await engine.dispose()
