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
import traceback
import uuid
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
from atm.core.types import Phase
from atm.evaluation.aggregator import compute_quality
from atm.experiment.config import ExperimentConfig
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
) -> None:
    """Update run row to completed status."""
    async with session_scope(session_factory) as session:
        await session.execute(
            sa.update(Run)
            .where(Run.id == run_id)
            .values(
                status="completed",
                finish_reason=FinishReason.SUCCESS.value,
                quality_score=quality_score,
                budget_spent_usd=Decimal(str(budget_spent_usd)),
                iterations=iterations,
                finished_at=datetime.now(UTC),
            )
        )


async def _update_run_failed(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: UUID,
    status: str,
    finish_reason: str,
    quality_score: float | None,
    budget_spent_usd: float,
    iterations: int,
    error_text: str | None = None,
) -> None:
    """Update run row to failed/budget_exceeded status."""
    async with session_scope(session_factory) as session:
        await session.execute(
            sa.update(Run)
            .where(Run.id == run_id)
            .values(
                status=status,
                finish_reason=finish_reason,
                quality_score=quality_score,
                budget_spent_usd=Decimal(str(budget_spent_usd)),
                iterations=iterations,
                finished_at=datetime.now(UTC),
                error=error_text,
            )
        )


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

        tools = ToolRegistry()
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

        # Step 5: build agents
        agents = _build_agents(cfg, llms)

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

        async with checkpointer_scope(pg_dsn) as checkpointer:
            # BUG-4 fix: TopologyRegistry.get() returns the CLASS, not an instance.
            # Instantiate the class before calling build() so that self is bound.
            topology_cls = TopologyRegistry.get(cfg.topology.name)
            topology_instance = topology_cls()
            compiled_graph = topology_instance.build(
                agents,
                topology_cfg,
                checkpointer=checkpointer,
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

        # Step 12: update run to completed
        await _update_run_success(
            session_factory,
            run_id,
            exp_id,
            quality_score=quality_score,
            budget_spent_usd=budget_spent,
            iterations=iterations,
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
