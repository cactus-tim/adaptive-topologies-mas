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

Flush-before-update invariant:
    parquet_writer.close() MUST be called BEFORE _update_run_success/_update_run_failed
    to ensure all buffered observability data is flushed to disk before the run record
    is marked complete.

Exception handling:
    - BudgetExceededError → status="budget_exceeded", flush + update, return RunResult
    - Any other Exception  → status="failed", flush + update, re-raise
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
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
from atm.experiment.finalize import maybe_finalize_answer
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

try:
    from atm.tasks.dabench import stage_workspace_for
except ImportError:  # pragma: no cover
    stage_workspace_for = None  # type: ignore[assignment]
from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox
from atm.topology.base import TopologyConfig, TopologyRegistry

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


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


def _load_pricing() -> Pricing:
    """Load Pricing from conf/pricing.yaml, with fallback to empty pricing table."""
    candidates = [
        Path("conf/pricing.yaml"),
        Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return Pricing.from_yaml(candidate)
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

    full_snapshot = cfg.model_dump(mode="json")

    async with session_factory() as session:
        stmt = (
            sa.dialects.postgresql.insert(Experiment)
            .values(
                id=uuid.uuid4(),
                name=cfg.name,
                config_snapshot=full_snapshot,
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

    host = socket.gethostname()
    process_pid = os.getpid()

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
            host=host,
            process_pid=process_pid,
        )
        session.add(run)

    return run_id


async def _register_worker_identity(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: UUID,
    *,
    host: str | None = None,
    pid: int | None = None,
) -> None:
    """Persist host/process_pid for the worker that owns ``run_id`` (M12).

    Run UPDATE immediately after ``_insert_run`` so M12 reconcile (m12-resume-replay)
    can detect crashed workers by matching ``runs.process_pid`` against the live
    process list on ``runs.host``. Failure to UPDATE is logged but never raises —
    the run itself proceeds.

    Args:
        session_factory: Async session factory.
        run_id:          UUID of the run row to update.
        host:            Hostname (defaults to ``socket.gethostname()``).
        pid:             Process ID (defaults to ``os.getpid()``).
    """
    effective_host = host if host is not None else socket.gethostname()
    effective_pid = pid if pid is not None else os.getpid()

    if len(effective_host) > 64:
        effective_host = effective_host[:64]

    try:
        async with session_scope(session_factory) as session:
            await session.execute(
                sa.update(Run)
                .where(Run.id == run_id)
                .values(host=effective_host, process_pid=effective_pid)
            )
    except Exception as exc:
        logger.warning(
            "register_worker_identity UPDATE failed; leaving host/pid NULL",
            run_id=str(run_id),
            error=str(exc)[:200],
        )


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


def _augment_task_input_with_metadata(
    base: str,
    metadata: dict[str, Any] | None,
    spec_id: str | None = None,
) -> str:
    """Append a [Task metadata] block to *base* when metadata warrants it.

    Augmentation fires only when all three conditions hold:
    1. ``metadata`` is non-None and non-empty.
    2. ``spec_id`` is either None or starts with ``"dabench/"`` — defensive
       narrowing prevents future task families from accidentally triggering this
       path by adding a ``format`` key to their metadata.
    3. At least one of ``metadata["format"]`` or ``metadata["file_name"]`` is
       present and truthy.

    Lines appended (only when the corresponding key is present and truthy):
        Dataset file:   {file_name}
        Constraints:    {constraints}
        Answer format (must appear verbatim in your final reply): {format}

    Args:
        base:     The original task input string.
        metadata: Metadata dict from ``TaskSpec.metadata``, or None.
        spec_id:  The ``TaskSpec.id`` (e.g. ``"dabench/titanic/0"``), or None.

    Returns:
        The augmented string, or *base* unchanged if augmentation does not apply.
    """
    if not metadata:
        return base
    if spec_id is not None and not spec_id.startswith("dabench/"):
        return base
    if not (metadata.get("format") or metadata.get("file_name")):
        return base

    lines: list[str] = ["", "", "[Task metadata]"]
    file_name = metadata.get("file_name")
    constraints = metadata.get("constraints")
    fmt = metadata.get("format")
    if file_name:
        lines.append(f"Dataset file: {file_name}")
    if constraints:
        lines.append(f"Constraints: {constraints}")
    if fmt:
        lines.append(f"Answer format (must appear verbatim in your final reply): {fmt}")
    return base + "\n".join(lines)


def _build_initial_state(
    cfg: ExperimentConfig,
    run_id: UUID,
    *,
    spec: Any = None,
) -> dict[str, Any]:
    """Build the initial GraphState with all 14 SharedState keys populated.

    Required SharedState keys:
        task_id, task_input, phase, iteration, iter_total, active_topology,
        final_answer, signals, phase_started_at_iter, topology_started_at_iter,
        topology_history, topology_switch_count, phase_history, human_requests,
        human_responses, broadcast_bus

    ``resolve_spec`` is always attempted so that ``TaskSpec.metadata`` (e.g.
    DABench ``format`` / ``constraints`` / ``file_name``) can be appended to
    ``task_input``.  If resolution fails the function falls back gracefully.

    Args:
        cfg:    Experiment configuration.
        run_id: UUID of the current run.
        spec:   Optional pre-resolved ``TaskSpec``.  When provided, ``resolve_spec``
                is NOT called again — the caller is responsible for resolving the
                spec once and passing it here to avoid duplicate registry lookups.
                When omitted (e.g. in unit tests that call this function directly),
                ``resolve_spec`` is invoked internally.

    Returns:
        A fully populated GraphState dict.
    """
    _spec = spec
    if _spec is None:
        try:
            _spec = resolve_spec(cfg.task)
        except Exception:
            logger.warning(
                "resolve_spec failed",
                task_name=cfg.task.name,
            )

    task_input: str = cfg.task.input or (_spec.input if _spec else "") or ""
    task_input = _augment_task_input_with_metadata(
        task_input,
        _spec.metadata if _spec else None,
        spec_id=_spec.id if _spec else None,
    )

    return {
        "shared": {
            "run_id": run_id,
            "task_id": cfg.task.name,
            "task_input": task_input,
            "phase": Phase.PLANNING,
            "iteration": 0,
            "iter_total": 0,
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


def _pre_stage_workspace(
    cfg: ExperimentConfig,
    run_id: UUID,
    workspace_path: Path,
    *,
    spec: Any = None,
) -> list[Path]:
    """Stage task-specific files into *workspace_path* before the graph runs.

    Resolves the task spec via :func:`resolve_spec` (unless *spec* is already
    provided by the caller) and delegates to :func:`stage_workspace_for` from
    ``atm.tasks.dabench``.  For non-DABench task families, or when the spec
    cannot be resolved, the function returns an empty list without raising.

    Args:
        cfg:            Experiment configuration (used to resolve the task spec
                        when *spec* is not supplied).
        run_id:         UUID of the current run (for log context only).
        workspace_path: The per-run workspace directory that has already been
                        created by the caller (``tools_workspace``).
        spec:           Optional pre-resolved ``TaskSpec``.  When provided,
                        :func:`resolve_spec` is NOT called again, keeping the
                        total call count at 1 (required by the evaluation-wiring
                        test contract).  When omitted (e.g. in the resume path),
                        ``resolve_spec`` is invoked internally.

    Returns:
        List of :class:`~pathlib.Path` objects for every file staged into
        *workspace_path*, or ``[]`` on any failure.
    """
    if stage_workspace_for is None:  # pragma: no cover
        return []
    try:
        _spec = spec if spec is not None else resolve_spec(cfg.task)
        if _spec is None:
            return []
        return stage_workspace_for(_spec, workspace_path)
    except Exception:
        logger.warning(
            "pre_stage_workspace failed; continuing without staged files",
            run_id=str(run_id),
            exc_info=True,
        )
        return []


def _build_role_wrapper(
    role: str,
    model_id: str,
    *,
    pricing: Pricing,
    budget: BudgetTracker,
    fake_fixtures: dict[str, str],
    provider_opts: dict[str, Any],
    replay_sources: dict[str, Path] | None,
    shared_replay_llm: Any,
) -> LLMWrapper:
    """Build a single LLMWrapper for one role (or the summarizer) via dispatch.

    Encapsulates the provider dispatch shared by every entry in the role set so
    the summarizer can reuse the exact same logic:

      - ``fake:scripted`` → fixture from ``fake_fixtures[role]`` (or
        ``FakeLLM(mode="echo")`` fallback with a warning when none configured)
      - ``fake:replay``   → ``shared_replay_llm`` if provided, else the Parquet
        path from ``replay_sources[role]``
      - real provider     → ``build_llm`` with merged ``provider_opts``

    Args:
        role:             Role key (used for fixture/opts/replay lookups and logs).
        model_id:         Provider-qualified model id for this role.
        pricing:          Shared pricing table.
        budget:           Shared BudgetTracker.
        fake_fixtures:    ``cfg.model.fake_fixtures`` mapping role → fixture path.
        provider_opts:    ``cfg.model.provider_opts`` for the real-provider branch.
        replay_sources:   Optional ``{role: parquet_path}`` for ``fake:replay``.
        shared_replay_llm: Optional shared replay FakeLLM (deterministic replay).

    Returns:
        A configured LLMWrapper.
    """
    provider = model_id.split(":", 1)[0] if ":" in model_id else model_id
    bare_model = model_id.split(":", 1)[1] if ":" in model_id else model_id

    if provider == "fake" and bare_model == "scripted":
        fixture_str = fake_fixtures.get(role)
        if fixture_str is None:
            logger.warning(
                "fake:scripted model requested but no fixture configured; "
                "falling back to FakeLLM(mode='echo')",
                role=role,
                hint="Set model.fake_fixtures.<role>=<path> in experiment config",
            )
        fixture_path = Path(fixture_str) if fixture_str else None
        return build_llm(
            model_id=model_id,
            pricing=pricing,
            budget=budget,
            fixture_path=fixture_path,
        )
    if provider == "fake" and bare_model == "replay":
        if shared_replay_llm is not None:
            return LLMWrapper(
                model_id=model_id,
                pricing=pricing,
                budget=budget,
                llm=shared_replay_llm,
            )
        replay_path = replay_sources.get(role) if replay_sources else None
        if replay_path is None:
            raise ValueError(
                f"fake:replay requested for role={role!r} but no replay_source "
                f"provided. Pass replay_sources={{'{role}': <parquet_path>}} "
                f"or shared_replay_llm to _build_llm_wrappers / replay_one()."
            )
        return build_llm(
            model_id=model_id,
            pricing=pricing,
            budget=budget,
            replay_source=replay_path,
        )
    opts = _resolve_provider_opts(provider_opts, provider, role)
    return build_llm(
        model_id=model_id,
        pricing=pricing,
        budget=budget,
        cfg=opts or None,
    )


def _build_llm_wrappers(
    cfg: ExperimentConfig,
    budget: BudgetTracker,
    pricing: Pricing,
    *,
    replay_sources: dict[str, Path] | None = None,
    shared_replay_llm: Any = None,
) -> dict[str, LLMWrapper]:
    """Build one LLMWrapper per role from cfg.model.by_role + default fallback.

    For "fake:scripted" providers, fixture paths are resolved from
    ``cfg.model.fake_fixtures`` (a dict mapping role → path string). If no
    fixture is configured for a scripted role, falls back to FakeLLM(mode="echo")
    with a warning log.

    For "fake:echo" providers, injects FakeLLM(mode="echo").
    For "fake:replay" providers (m12-resume-replay), ``replay_sources`` must
    contain a Parquet path per role pointing at the original ``llm_calls.parquet``.
    For real providers, init_chat_model is used (via build_llm / LLMWrapper).

    Args:
        cfg:            Experiment configuration.
        budget:         Shared BudgetTracker.
        pricing:        Pricing table.
        replay_sources: Optional ``{role: parquet_path}`` for ``fake:replay``.
                        When the configured model_id for a role is ``fake:replay``,
                        the matching entry is forwarded to ``build_llm`` as
                        ``replay_source=...``.

    Returns:
        Dict mapping role name to LLMWrapper.
    """
    roles = ["planner", "executor", "critic", "researcher"]
    wrappers: dict[str, LLMWrapper] = {}

    for role in roles:
        wrappers[role] = _build_role_wrapper(
            role,
            cfg.model.get_model_for(role),
            pricing=pricing,
            budget=budget,
            fake_fixtures=cfg.model.fake_fixtures,
            provider_opts=cfg.model.provider_opts,
            replay_sources=replay_sources,
            shared_replay_llm=shared_replay_llm,
        )

    # Summarizer wrapper for the window-with-summary scratchpad policy. Built
    # through the SAME provider dispatch as the roles so fake/offline configs
    # stay fully offline (no API key / no network at construction). Construction
    # is best-effort, mirroring the judge-LLM resilience in run_one: a real
    # provider summarizer without credentials — common when model.summarizer is
    # left at its default in a fake config — must not abort the run; agents then
    # fall back to window-only scratchpad memory.
    try:
        wrappers["summarizer"] = _build_role_wrapper(
            "summarizer",
            cfg.model.summarizer,
            pricing=pricing,
            budget=budget,
            fake_fixtures=cfg.model.fake_fixtures,
            provider_opts=cfg.model.provider_opts,
            replay_sources=replay_sources,
            shared_replay_llm=shared_replay_llm,
        )
    except Exception as exc:
        logger.warning(
            "summarizer LLM construction failed — proceeding without scratchpad "
            "summarization (window-only fallback)",
            summarizer_model=cfg.model.summarizer,
            error=str(exc)[:200],
        )

    return wrappers


def _resolve_provider_opts(
    provider_opts: dict[str, Any], provider: str, role: str
) -> dict[str, Any]:
    """Merge provider-level + role-specific opts from ``cfg.model.provider_opts``.

    Two flat keys are recognised per provider:
      - ``<provider>``           — default opts applied to every role on the provider
      - ``<provider>_<role>``    — overrides applied on top of the default

    Example::

        provider_opts:
          cerebras: {reasoning_effort: low}
          cerebras_executor: {reasoning_effort: medium}

    Returns ``{}`` when no opts are configured. Non-dict values under either
    key are silently ignored (defensive against typo'd yaml).
    """
    merged: dict[str, Any] = {}
    base = provider_opts.get(provider)
    if isinstance(base, dict):
        merged.update(base)
    override = provider_opts.get(f"{provider}_{role}")
    if isinstance(override, dict):
        merged.update(override)
    return merged


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
    summarizer_llm = llms.get("summarizer")

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
        agent_cls: type = Critic if role == "critic" else Agent
        agents[role] = agent_cls(
            agent_id=role,
            cfg=agent_cfg,
            llm=llm,
            tools=tools,
            summarizer_llm=summarizer_llm,
        )

    topo_name = getattr(cfg.topology, "name", "")
    raw_extra = getattr(cfg.topology, "extra", None)
    if raw_extra is not None and hasattr(raw_extra, "model_dump"):
        hier_extra = raw_extra.hierarchical.model_dump(exclude_none=True)
        debate_extra = raw_extra.debate.model_dump(exclude_none=True)
    else:
        flat = dict(raw_extra) if raw_extra else {}
        hier_extra = flat.get("hierarchical", flat)
        debate_extra = flat.get("debate", flat)
    extra_workers: list[tuple[str, str]] = []

    needs_hier = topo_name in ("hierarchical", "adaptive")
    needs_debate = topo_name in ("debate", "adaptive")

    if needs_hier:
        sub_teams = list(hier_extra.get("sub_teams") or [])
        if len(sub_teams) < 2:
            sub_teams = [
                {"team_id": "team_a", "workers": ["executor_a1", "executor_a2"]},
                {"team_id": "team_b", "workers": ["executor_b1", "executor_b2"]},
            ]
        for team in sub_teams:
            for worker_id in team.get("workers") or []:
                extra_workers.append((str(worker_id), "executor"))
    if needs_debate:
        extra_workers.append((str(debate_extra.get("debater_pro_id") or "debater_pro"), "executor"))
        extra_workers.append(
            (str(debate_extra.get("debater_contra_id") or "debater_contra"), "executor")
        )
        extra_workers.append((str(debate_extra.get("judge_id") or "judge"), "critic"))

    for worker_id, base_role in extra_workers:
        if worker_id in agents:
            continue
        base_yaml = agents_conf_dir / f"{base_role}.yaml"
        if not base_yaml.exists():
            logger.warning(
                "topology extra worker base config missing",
                worker_id=worker_id,
                base_role=base_role,
                path=str(base_yaml),
            )
            continue
        try:
            agent_cfg = load_agent_config(base_yaml)
        except Exception as e:
            logger.warning(
                "topology extra worker config load failed",
                worker_id=worker_id,
                error=str(e),
            )
            continue
        if topo_name in ("debate", "adaptive") and worker_id in (
            str(debate_extra.get("debater_pro_id") or "debater_pro"),
            str(debate_extra.get("debater_contra_id") or "debater_contra"),
        ):
            debate_override = (
                "[DEBATE ROLE — HARD RULES, READ FIRST]\n"
                "You are a DEBATER who is ALSO the executor. Your job has TWO "
                "outputs every single turn, in this strict order:\n"
                "\n"
                "STEP 1 — PRODUCE THE ANSWER ARTIFACT.\n"
                "  • Code task (HumanEval, anything mentioning `def `, "
                "`function`, `class`, etc.):\n"
                "      Call the `file_write` tool to save your FULL candidate "
                "solution to `solution.py`. Use `overwrite=True` so you can "
                "refine across rounds. DO THIS BEFORE WRITING ANY DRAFT TEXT.\n"
                "  • Non-code task (math, sentence generation, classification, "
                "data analysis):\n"
                "      Your DRAFT message MUST begin with this exact marker, "
                "on its own line, before any other text:\n"
                "          ###ANSWER###\n"
                "          <your concrete final answer here — only the answer, "
                "no preamble>\n"
                "          ###END###\n"
                "      Only AFTER the closing marker may you write your "
                "argument / reasoning.\n"
                "\n"
                "STEP 2 — DEFEND YOUR ANSWER (in DRAFT).\n"
                "  Now you may argue your stance, attack the opponent's "
                "weaknesses, cite evidence. But the answer above is the "
                "AUTHORITATIVE submission — extractors and the judge will "
                "look at the file (code tasks) or the ###ANSWER### block "
                "(non-code) and ignore the rest of your argument.\n"
                "\n"
                "If you skip STEP 1, you have NO answer to defend and you "
                "CANNOT WIN, regardless of how persuasive your argument is. "
                "The judge picks the winning side's artifact verbatim — "
                "missing artifact = automatic loss.\n"
                "\n"
                "[END DEBATE ROLE RULES]\n\n"
            )
            patched_prompt = debate_override + (agent_cfg.system_prompt or "")
            agent_cfg = agent_cfg.model_copy(update={"system_prompt": patched_prompt})

        if topo_name in ("debate", "adaptive") and worker_id == str(
            debate_extra.get("judge_id") or "judge"
        ):
            judge_override = (
                "[DEBATE JUDGE — HARD RULES, READ FIRST]\n"
                "You judge a debate between two debaters (pro / contra). "
                "Output a DECISION message containing APPROVE or REJECT.\n"
                "\n"
                "WHERE TO LOOK FOR THE ANSWER (in priority order):\n"
                "  1. CODE TASKS (the task mentions `def `, `function`, "
                "a programming problem like HumanEval): inspect the "
                "`solution.py` file via the `file_read` tool. The winning "
                "side's `solution.py` is the authoritative answer.\n"
                "  2. NON-CODE TASKS (math word problems / sentence "
                "generation / data-analysis / anything else): DO NOT call "
                "`file_read`. Instead, read each debater's last DRAFT and "
                "find the block delimited by `###ANSWER###` and "
                "`###END###`. The text between those markers is each "
                "side's submission. Compare the two and judge correctness.\n"
                "\n"
                "VERDICT RULES:\n"
                "  • APPROVE whenever at least ONE side has a plausibly "
                "correct answer (a working `solution.py` for code, or a "
                "well-formed `###ANSWER###` block for non-code). Set "
                '`winner` to that side: "pro" or "contra".\n'
                "  • REJECT only when BOTH sides are clearly wrong, both "
                "drafts are missing the required artifact (no file for "
                "code tasks, no `###ANSWER###` block for non-code), or the "
                "answers are obviously malformed. Refusing to decide is "
                "EXPENSIVE — every REJECT triggers another debate round.\n"
                "\n"
                "DECISION FORMAT (last line of your reply, exact tokens):\n"
                "  APPROVE  — followed by `winner: pro` or `winner: contra`\n"
                "  REJECT   — only when both sides are unrecoverably wrong\n"
                "\n"
                "[END DEBATE JUDGE RULES]\n\n"
            )
            patched_prompt = judge_override + (agent_cfg.system_prompt or "")
            agent_cfg = agent_cfg.model_copy(update={"system_prompt": patched_prompt})

        llm = llms.get(worker_id) or llms.get(base_role) or llms.get("planner")
        if llm is None:
            logger.warning(
                "no LLM wrapper for topology extra worker",
                worker_id=worker_id,
            )
            continue
        tools = tool_registry if tool_registry is not None else ToolRegistry()
        agent_cls_extra: type = Critic if base_role == "critic" else Agent
        agents[worker_id] = agent_cls_extra(
            agent_id=worker_id,
            cfg=agent_cfg,
            llm=llm,
            tools=tools,
            summarizer_llm=summarizer_llm,
        )

    return agents


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

    engine: AsyncEngine = create_engine(pg_dsn)
    session_factory = create_session_factory(engine)

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
    llms: dict[str, LLMWrapper] | None = None
    sandbox: Any | None = None

    seed_all(cfg.seed)

    _run_started_monotonic: float = time.monotonic()

    def _elapsed_s() -> float:
        return max(0.0, time.monotonic() - _run_started_monotonic)

    try:
        exp_id = await _ensure_experiment(session_factory, cfg)

        run_id = await _insert_run(session_factory, exp_id, cfg)

        await _register_worker_identity(session_factory, run_id)

        log = logger.bind(run_id=str(run_id), exp_id=str(exp_id))
        log.info("run started", topology=cfg.topology.name, task=cfg.task.name)

        budget = BudgetTracker(
            per_call_usd=cfg.budget.per_call_usd,
            per_run_usd=cfg.budget.per_run_usd,
            per_experiment_usd=cfg.budget.per_experiment_usd,
        )
        pricing = _load_pricing()

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

        llms = _build_llm_wrappers(cfg, budget, pricing)

        from atm.tools.defaults import build_default_registry
        from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox as _Sandbox

        _run_spec = resolve_spec(cfg.task)

        tools_workspace = parquet_root / "workspace" / str(run_id)
        tools_workspace.mkdir(parents=True, exist_ok=True)
        _pre_stage_workspace(cfg, run_id, tools_workspace, spec=_run_spec)
        tools_corpus = tools_workspace / "_corpus"
        tools_corpus.mkdir(exist_ok=True)
        try:
            tool_registry = build_default_registry(
                workspace=tools_workspace,
                corpus_dir=tools_corpus,
                sandbox=_Sandbox(workspace=tools_workspace),
            )
        except Exception:
            logger.warning(
                "build_default_registry failed; falling back to empty registry", exc_info=True
            )
            tool_registry = None
        agents = _build_agents(cfg, llms, tool_registry=tool_registry)

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

        initial_state = _build_initial_state(cfg, run_id, spec=_run_spec)

        topology_cfg = TopologyConfig(
            name=cfg.topology.name,
            max_iterations=cfg.topology.max_iterations,
            extra=cfg.topology.extra.model_dump(exclude_none=True),
        )

        final_state: dict[str, Any]

        human_gateway_llm: LLMWrapper | None = None
        if cfg.human is not None and cfg.human.enabled and cfg.human.gateway == "llm_simulated":
            human_model_id = cfg.human.model or cfg.model.default
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
            topology_cls = TopologyRegistry.get(cfg.topology.name)
            topology_instance = topology_cls()
            topology_router_llm: LLMWrapper | None = None
            if cfg.topology.name == "adaptive":
                router_model_id = cfg.model.router or cfg.model.default
                topology_router_llm = build_llm(
                    model_id=router_model_id,
                    pricing=pricing,
                    budget=budget,
                )

            compiled_graph = topology_instance.build(
                agents,
                topology_cfg,
                checkpointer=checkpointer,
                human_cfg=cfg.human,
                human_gateway_llm=human_gateway_llm,
                role_router=role_router,
                topology_router_llm=topology_router_llm,
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

        shared_final: dict[str, Any] = final_state.get("shared") or {}
        final_answer = str(shared_final.get("final_answer") or "")
        iterations = int(shared_final.get("iter_total") or 0)

        _finalize_llm = llms.get("executor") or llms.get("default")
        if _finalize_llm is not None:
            final_answer = await maybe_finalize_answer(
                llm=_finalize_llm,
                task_spec=_run_spec,
                final_state=final_state,
                current_answer=final_answer,
            )

        budget_spent = budget.totals.get(BudgetLevel.RUN, 0.0)

        sandbox = SubprocessSandbox()
        spec = _run_spec
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

        _run_dir = parquet_root / "experiments" / str(exp_id) / "runs" / str(run_id)
        try:
            _write_replay_parquet(_run_dir, list(final_state.get("llm_calls") or []))
        except Exception:
            logger.warning("_write_replay_parquet failed — replay may not work", exc_info=True)

        await parquet_writer.close()

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


def _load_cfg_from_snapshot(snapshot: dict[str, Any]) -> ExperimentConfig:
    """Rehydrate an :class:`ExperimentConfig` from ``experiments.config_snapshot``.

    The m12-resume-replay block widens ``config_snapshot`` to the full
    ``cfg.model_dump(mode="json")`` payload, so this is just a thin wrapper
    around ``ExperimentConfig.model_validate``. We keep it as a named helper
    so that older sparse snapshots (4-field shape pre-m12) can be detected
    and rejected with a clear error message in one place.

    Args:
        snapshot: The JSONB dict pulled from ``experiments.config_snapshot``.

    Returns:
        A fully-validated :class:`ExperimentConfig`.

    Raises:
        ValueError: If the snapshot is missing required keys (typical of the
                    legacy 4-field shape from before m12-resume-replay).
    """
    required = {"name", "task", "model", "agents", "topology", "budget", "observability"}
    missing = required - snapshot.keys()
    if missing:
        raise ValueError(
            f"experiments.config_snapshot is missing keys {sorted(missing)} — "
            "this row was written by a pre-m12 runner. Resume/replay require a "
            "full cfg.model_dump() snapshot. Re-run the original experiment "
            "after the m12-resume-replay deployment, or restore the YAML and "
            "pass it explicitly."
        )
    return ExperimentConfig.model_validate(snapshot)


async def _fetch_run_row(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: UUID,
) -> dict[str, Any]:
    """SELECT a single runs row and return its key fields as a dict.

    Returns: dict with keys {id, exp_id, seed, model, models_by_role_json,
    model_version_snapshot, status, host, process_pid, replay_of}.

    Raises:
        LookupError: if no row matches ``run_id``.
    """
    async with session_scope(session_factory) as session:
        result = await session.execute(
            sa.select(
                Run.id,
                Run.exp_id,
                Run.seed,
                Run.model,
                Run.models_by_role_json,
                Run.model_version_snapshot,
                Run.status,
                Run.host,
                Run.process_pid,
                Run.replay_of,
            ).where(Run.id == run_id)
        )
        row = result.fetchone()
    if row is None:
        raise LookupError(f"run_id={run_id} not found in runs table")
    return {
        "id": row[0],
        "exp_id": row[1],
        "seed": row[2],
        "model": row[3],
        "models_by_role_json": row[4],
        "model_version_snapshot": row[5],
        "status": row[6],
        "host": row[7],
        "process_pid": row[8],
        "replay_of": row[9],
    }


async def _fetch_experiment_snapshot(
    session_factory: async_sessionmaker[AsyncSession],
    exp_id: UUID,
) -> dict[str, Any]:
    """SELECT experiments.config_snapshot for the given exp_id."""
    async with session_scope(session_factory) as session:
        result = await session.execute(
            sa.select(Experiment.config_snapshot).where(Experiment.id == exp_id)
        )
        row = result.fetchone()
    if row is None:
        raise LookupError(f"exp_id={exp_id} not found in experiments table")
    snapshot = row[0]
    if not isinstance(snapshot, dict):
        raise ValueError(
            f"experiments.config_snapshot for exp_id={exp_id} is not a dict "
            f"(got {type(snapshot).__name__})"
        )
    return snapshot


async def _set_run_status_running(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: UUID,
) -> None:
    """Reset a run row to status='running' (resume preamble).

    Also updates host/process_pid so a subsequent reconcile sees the *new*
    resuming worker, not the stale original. Idempotent.
    """
    async with session_scope(session_factory) as session:
        await session.execute(
            sa.update(Run)
            .where(Run.id == run_id)
            .values(
                status="running",
                finish_reason=None,
                host=socket.gethostname(),
                process_pid=os.getpid(),
            )
        )


async def resume_one(
    run_id: UUID,
    cfg: ExperimentConfig | None = None,
    *,
    force: bool = False,
) -> RunResult:
    """Resume a previously-interrupted run from the last LangGraph checkpoint.

    Reads the ``runs`` row identified by ``run_id``, rehydrates the
    :class:`ExperimentConfig` from ``experiments.config_snapshot``, resets the
    run status to 'running', and re-invokes the topology graph with
    ``thread_id=str(run_id)`` so LangGraph's PG checkpointer continues from
    the latest checkpoint.

    Args:
        run_id: UUID of the run row to resume.
        cfg:    Optional pre-built ExperimentConfig (skips snapshot load — used
                by tests). When None (default), config is rehydrated from
                ``experiments.config_snapshot``.
        force:  When True, skip the live-pid liveness check on the existing
                ``runs.process_pid`` and proceed regardless. Use when the
                original worker is stuck but not dead.

    Returns:
        :class:`RunResult` with the resumed run's terminal status + metrics.
        The returned ``run_id`` equals the input ``run_id`` — resume does NOT
        allocate a new run row.

    Raises:
        LookupError:       If ``run_id`` does not exist.
        RuntimeError:      If the run is still alive (pid_alive=True) and
                           ``force=False``.
        Exception:         Re-raises any execution failure after persisting
                           ``status='failed'``.
    """
    from atm.experiment.reconcile import _pid_alive

    bootstrap_engine = create_engine_for_dsn_discovery(cfg)
    if bootstrap_engine is None:
        raise RuntimeError(
            "resume_one: cannot determine pg_dsn — pass cfg explicitly or ensure ATM_PG_DSN is set."
        )
    engine, session_factory = bootstrap_engine

    try:
        run_row = await _fetch_run_row(session_factory, run_id)
        exp_id: UUID = run_row["exp_id"]

        if cfg is None:
            snapshot = await _fetch_experiment_snapshot(session_factory, exp_id)
            cfg = _load_cfg_from_snapshot(snapshot)

        if not force:
            stored_host = run_row["host"]
            stored_pid = run_row["process_pid"]
            same_host = stored_host == socket.gethostname()
            if same_host and stored_pid is not None and _pid_alive(int(stored_pid)):
                raise RuntimeError(
                    f"resume_one: run {run_id} still has a live process "
                    f"(pid={stored_pid}). Pass force=True to override."
                )

        await _set_run_status_running(session_factory, run_id)

        return await _execute_existing_run(
            cfg=cfg,
            run_id=run_id,
            exp_id=exp_id,
            engine=engine,
            session_factory=session_factory,
            replay_sources=None,
        )
    finally:
        await engine.dispose()


async def replay_one(
    original_run_id: UUID,
    mode: Literal["deterministic", "semantic"] = "deterministic",
    *,
    cfg_override: ExperimentConfig | None = None,
) -> RunResult:
    """Deterministically (or semantically) replay an existing run.

    Deterministic mode:
      - Locates ``data/experiments/{exp_id}/runs/{original_run_id}/llm_calls.parquet``.
      - Overrides ``cfg.model.default = "fake:replay"`` and points each role
        at the original Parquet via ``replay_sources``.
      - Verifies ``runs.model_version_snapshot`` of the original is non-empty
        and matches the new run's intended models (loose check — exact
        per-role parity is enforced only for non-fake models).
      - INSERTs a new run row with ``replay_of=<original_run_id>``.

    Semantic mode:
      - Uses the real LLM with the same model + seed.
      - Tolerance is checked downstream by the aggregator — this function
        only plumbs the flag.

    Args:
        original_run_id: UUID of the run to replay.
        mode:            "deterministic" (default) or "semantic".
        cfg_override:    Optional override for tests to inject a cfg directly
                         instead of loading from the snapshot.

    Returns:
        :class:`RunResult` for the new replay run.

    Raises:
        LookupError:    If ``original_run_id`` is unknown.
        FileNotFoundError: If deterministic mode and the original parquet is
                           missing.
        ValueError:     If deterministic mode and the model_version_snapshot
                        of the original is incompatible (caller can switch
                        to semantic mode).
    """
    bootstrap_engine = create_engine_for_dsn_discovery(cfg_override)
    if bootstrap_engine is None:
        raise RuntimeError(
            "replay_one: cannot determine pg_dsn — pass cfg_override explicitly or set ATM_PG_DSN."
        )
    engine, session_factory = bootstrap_engine

    try:
        run_row = await _fetch_run_row(session_factory, original_run_id)
        exp_id: UUID = run_row["exp_id"]

        if cfg_override is None:
            snapshot = await _fetch_experiment_snapshot(session_factory, exp_id)
            cfg = _load_cfg_from_snapshot(snapshot)
        else:
            cfg = cfg_override

        if mode == "deterministic":
            parquet_root = Path(cfg.observability.parquet_dir)
            llm_calls_path = (
                parquet_root
                / "experiments"
                / str(exp_id)
                / "runs"
                / str(original_run_id)
                / "llm_calls.parquet"
            )
            if not llm_calls_path.exists():
                raise FileNotFoundError(
                    f"replay_one: original llm_calls.parquet not found at "
                    f"{llm_calls_path}. Deterministic replay requires the "
                    f"original run's parquet bundle."
                )

            orig_versions = run_row["model_version_snapshot"] or {}
            if orig_versions:
                logger.info(
                    "replay_one: deterministic mode overrides original model versions",
                    original_versions=orig_versions,
                )

            import pyarrow.parquet as pq

            from atm.llm.fake import FakeLLM

            shared_table = pq.read_table(str(llm_calls_path))  # type: ignore[no-untyped-call]
            shared_fake: Any = FakeLLM(mode="replay", replay_table=shared_table)
            replay_sources = None

            new_model_cfg = cfg.model.model_copy(update={"default": "fake:replay"})
            cfg = cfg.model_copy(update={"model": new_model_cfg})

        elif mode == "semantic":
            replay_sources = None
            shared_fake = None
        else:
            raise ValueError(f"replay_one: unknown mode {mode!r}")

        new_run_id = await _insert_replay_run(
            session_factory, exp_id, cfg, replay_of=original_run_id
        )

        return await _execute_existing_run(
            cfg=cfg,
            run_id=new_run_id,
            exp_id=exp_id,
            engine=engine,
            session_factory=session_factory,
            replay_sources=replay_sources,
            shared_replay_llm=shared_fake,
        )
    finally:
        await engine.dispose()


def _write_replay_parquet(
    run_dir: Path,
    llm_responses: list[Any],
) -> None:
    """Write llm_calls from graph state to ``llm_calls.parquet`` using REPLAY_SCHEMA.

    This ensures ``replay_one(mode='deterministic')`` can read back the calls even
    when FakeLLM is used (FakeLLM does not fire LangChain ``on_llm_end`` callbacks,
    so the observability parquet writer never writes the file).  Calling this after
    graph ``ainvoke`` completes writes all ``LLMResponse`` objects from the graph
    state into a parquet file that ``FakeLLM(mode='replay')`` can consume.

    Args:
        run_dir: Resolved path  ``{parquet_root}/experiments/{exp_id}/runs/{run_id}/``.
        llm_responses: List of :class:`~atm.core.types.LLMResponse` objects collected
            in the graph state under the ``llm_calls`` key.
    """
    if not llm_responses:
        return

    import json as _json

    import pyarrow as _pa
    import pyarrow.parquet as _pq

    from atm.llm.fake import REPLAY_SCHEMA

    rows: list[dict[str, Any]] = []
    for resp in llm_responses:
        tool_calls_json_val: str | None = None
        if resp.tool_calls:
            raw_tcs = [
                {
                    "id": str(tc.id),
                    "name": tc.tool_name,
                    "args": tc.args,
                    "issued_by": getattr(tc, "issued_by", ""),
                }
                for tc in resp.tool_calls
            ]
            tool_calls_json_val = _json.dumps(raw_tcs)

        rows.append(
            {
                "call_id": str(resp.id),
                "model": resp.model,
                "content": resp.text or "",
                "usage_input": resp.usage.prompt_tokens,
                "usage_output": resp.usage.completion_tokens,
                "usage_total": resp.usage.total_tokens,
                "usage_cached": resp.usage.cached_input_tokens,
                "cost_usd": resp.cost_usd,
                "latency_ms": resp.latency_ms,
                "finish_reason": resp.finish_reason,
                "started_at": resp.started_at.isoformat(),
                "tool_calls_json": tool_calls_json_val,
            }
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "llm_calls.parquet"
    table = _pa.Table.from_pylist(rows, schema=REPLAY_SCHEMA)
    writer = _pq.ParquetWriter(out_path, REPLAY_SCHEMA)  # type: ignore[no-untyped-call]
    writer.write_table(table)  # type: ignore[no-untyped-call]
    writer.close()  # type: ignore[no-untyped-call]


def create_engine_for_dsn_discovery(
    cfg: ExperimentConfig | None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]] | None:
    """Helper: build (engine, session_factory) from cfg.observability.pg_dsn.

    Falls back to the ``ATM_PG_DSN`` environment variable when ``cfg`` is None
    (used by CLI commands such as ``atm replay`` and ``atm resume`` that do not
    receive a config object before they can open the DB connection).

    Returns None when both ``cfg`` is None and ``ATM_PG_DSN`` is unset.
    """
    import os

    if cfg is not None:
        dsn: str = cfg.observability.pg_dsn
    else:
        dsn_env = os.environ.get("ATM_PG_DSN")
        if dsn_env is None:
            return None
        dsn = dsn_env
    engine: AsyncEngine = create_engine(dsn)
    session_factory = create_session_factory(engine)
    return engine, session_factory


async def _insert_replay_run(
    session_factory: async_sessionmaker[AsyncSession],
    exp_id: UUID,
    cfg: ExperimentConfig,
    *,
    replay_of: UUID,
) -> UUID:
    """INSERT a new replay-run row. Enforces exp_id parity with original."""
    run_id = uuid.uuid4()

    human_role_value: str | None = (
        cfg.human.role.value if cfg.human is not None and cfg.human.enabled else None
    )

    async with session_scope(session_factory) as session:
        parent_check = await session.execute(sa.select(Run.exp_id).where(Run.id == replay_of))
        parent_row = parent_check.fetchone()
        if parent_row is None:
            raise LookupError(f"replay_of={replay_of} not found in runs table")
        if parent_row[0] != exp_id:
            raise ValueError(
                f"replay_one: parent run {replay_of} belongs to exp_id "
                f"{parent_row[0]} but new run targets exp_id {exp_id}. "
                "Replay runs must live in the same experiment as the original."
            )

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
            host=socket.gethostname(),
            process_pid=os.getpid(),
            replay_of=replay_of,
        )
        session.add(run)

    return run_id


async def _execute_existing_run(
    *,
    cfg: ExperimentConfig,
    run_id: UUID,
    exp_id: UUID,
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    replay_sources: dict[str, Path] | None,
    shared_replay_llm: Any = None,
) -> RunResult:
    """Inner execution path shared by resume_one / replay_one.

    This is a stripped-down mirror of ``run_one`` that:
      - skips experiment/run row creation (caller owns those);
      - accepts a pre-allocated ``run_id`` + ``exp_id``;
      - forwards ``replay_sources`` to ``_build_llm_wrappers``;
      - delegates lifecycle terminals to ``_update_run_success`` /
        ``_update_run_failed`` against the *given* run_id.

    The shared LangGraph ``thread_id=str(run_id)`` semantics mean that when
    the same run_id is re-invoked against the PG checkpointer, the graph
    continues from the last persisted state (resume_one); when a NEW run_id
    is invoked, the graph starts fresh (replay_one).
    """
    pg_dsn = cfg.observability.pg_dsn
    parquet_root = Path(cfg.observability.parquet_dir)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logger.warning("schema creation failed (may already exist)", error=str(e))

    parquet_writer: ParquetWriter | None = None
    judge_llm: LLMWrapper | None = None
    quality_score: float | None = 0.0
    budget_spent: float = 0.0
    iterations: int = 0
    final_answer: str = ""
    llms: dict[str, LLMWrapper] | None = None
    sandbox: Any | None = None

    seed_all(cfg.seed)

    _run_started_monotonic: float = time.monotonic()

    def _elapsed_s() -> float:
        return max(0.0, time.monotonic() - _run_started_monotonic)

    log = logger.bind(run_id=str(run_id), exp_id=str(exp_id))
    log.info(
        "execute_existing_run",
        topology=cfg.topology.name,
        task=cfg.task.name,
        replay=replay_sources is not None,
    )

    try:
        budget = BudgetTracker(
            per_call_usd=cfg.budget.per_call_usd,
            per_run_usd=cfg.budget.per_run_usd,
            per_experiment_usd=cfg.budget.per_experiment_usd,
        )
        pricing = _load_pricing()

        try:
            judge_llm = build_llm(
                model_id=cfg.evaluation.judge_model,
                pricing=pricing,
                budget=budget,
            )
        except Exception:
            judge_llm = None

        llms = _build_llm_wrappers(
            cfg,
            budget,
            pricing,
            replay_sources=replay_sources,
            shared_replay_llm=shared_replay_llm,
        )

        from atm.tools.defaults import build_default_registry
        from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox as _Sandbox

        try:
            _resume_spec = resolve_spec(cfg.task)
        except Exception:
            _resume_spec = None

        tools_workspace = parquet_root / "workspace" / str(run_id)
        tools_workspace.mkdir(parents=True, exist_ok=True)
        _pre_stage_workspace(cfg, run_id, tools_workspace, spec=_resume_spec)
        tools_corpus = tools_workspace / "_corpus"
        tools_corpus.mkdir(exist_ok=True)
        try:
            tool_registry = build_default_registry(
                workspace=tools_workspace,
                corpus_dir=tools_corpus,
                sandbox=_Sandbox(workspace=tools_workspace),
            )
        except Exception:
            tool_registry = None
        agents = _build_agents(cfg, llms, tool_registry=tool_registry)

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

        initial_state = _build_initial_state(cfg, run_id, spec=_resume_spec)
        topology_cfg = TopologyConfig(
            name=cfg.topology.name,
            max_iterations=cfg.topology.max_iterations,
            extra=cfg.topology.extra.model_dump(exclude_none=True),
        )

        human_gateway_llm: LLMWrapper | None = None
        if cfg.human is not None and cfg.human.enabled and cfg.human.gateway == "llm_simulated":
            human_model_id = cfg.human.model or cfg.model.default
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
            topology_cls = TopologyRegistry.get(cfg.topology.name)
            topology_instance = topology_cls()
            topology_router_llm: LLMWrapper | None = None
            if cfg.topology.name == "adaptive":
                router_model_id = cfg.model.router or cfg.model.default
                topology_router_llm = build_llm(
                    model_id=router_model_id,
                    pricing=pricing,
                    budget=budget,
                )

            compiled_graph = topology_instance.build(
                agents,
                topology_cfg,
                checkpointer=checkpointer,
                human_cfg=cfg.human,
                human_gateway_llm=human_gateway_llm,
                role_router=role_router,
                topology_router_llm=topology_router_llm,
            )

            recursion_limit = max(100, (cfg.topology.max_iterations or 30) * 4 + 20)
            final_state: dict[str, Any] = await compiled_graph.ainvoke(
                initial_state,
                config={
                    "callbacks": [callback],
                    "configurable": {"thread_id": str(run_id)},
                    "recursion_limit": recursion_limit,
                },
            )

        shared_final: dict[str, Any] = final_state.get("shared") or {}
        final_answer = str(shared_final.get("final_answer") or "")
        iterations = int(shared_final.get("iter_total") or 0)
        budget_spent = budget.totals.get(BudgetLevel.RUN, 0.0)

        sandbox = SubprocessSandbox()
        spec = resolve_spec(cfg.task)

        _finalize_llm = llms.get("executor") or llms.get("default")
        if _finalize_llm is not None and spec is not None:
            final_answer = await maybe_finalize_answer(
                llm=_finalize_llm,
                task_spec=spec,
                final_state=final_state,
                current_answer=final_answer,
            )

        if spec is None:
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
                quality_score = None

        _run_dir = parquet_root / "experiments" / str(exp_id) / "runs" / str(run_id)
        try:
            _write_replay_parquet(_run_dir, list(final_state.get("llm_calls") or []))
        except Exception:
            log.warning("_write_replay_parquet failed — replay may not work", exc_info=True)

        await parquet_writer.close()

        await _update_run_success(
            session_factory,
            run_id,
            exp_id,
            quality_score=quality_score,
            budget_spent_usd=budget_spent,
            iterations=iterations,
            wall_time_s=_elapsed_s(),
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

    except BudgetExceededError:
        if parquet_writer is not None:
            with contextlib.suppress(Exception):
                await parquet_writer.close()
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
            run_id=run_id,
            exp_id=exp_id,
            status="budget_exceeded",
            metrics={
                "quality_score": quality_score,
                "cost_usd": budget_spent,
                "iters": iterations,
            },
            final_answer=final_answer,
        )

    except Exception:
        error_text = traceback.format_exc()
        if parquet_writer is not None:
            with contextlib.suppress(Exception):
                await parquet_writer.close()
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
