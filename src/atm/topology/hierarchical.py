"""HierarchicalTopology — two-level coordinator + compiled subgraphs.

Architecture (arch.md §7.6):
  - Exactly 2 levels: Top-Coordinator → SubCoord_A/SubCoord_B → Workers.
  - top_coord is a rule-based closure (NOT an Agent, NOT in agents dict).
  - sub_coord_a / sub_coord_b are rule-based closures (NOT Agents, NOT in agents dict).
  - 2 subgraphs compiled as StateGraph(GraphState).compile() and embedded in parent.
  - Workers (4): executor_a1, executor_a2, executor_b1, executor_b2 — Agent nodes.
  - final_answer = json.dumps({"team_a": ..., "team_b": ...}, ensure_ascii=False)
  - ValueError raised if cfg.extra["sub_teams"][i] contains "sub_teams" key (3rd level).

Stopping precedence (arch.md §7.1):
  1. max_iter (global): iter_total >= cfg.max_iterations → END
  2. topology_success: top_coord_finalize signal → END
  3. topology_max: max_rounds reached → END
  4. continue: loop back

Nodes (top-level):
  top_coord, team_a (subgraph), team_b (subgraph), hierarchical_finalize

Nodes (each subgraph):
  sub_coord_X, worker_X1, worker_X2

TopologyConfig.extra defaults:
  max_rounds: 4
  final_answer_strategy: "json_concat"
  sub_teams: [{team_id: "team_a", workers: [...]}, {team_id: "team_b", workers: [...]}]
  finalize_signal: "top_coord_finalize"

HITL (M9.1):
  scope="top" (default): human_top_reviewer node inserted between top_coord
    (finalize decision) and hierarchical_finalize (i.e., after both teams have
    produced drafts the human reviews before finalise is committed).
    Specifically: after_team_b → human_top_reviewer → hierarchical_finalize (when
    finalize condition is met).

  scope="sub_team": human_sub_reviewer inserted inside each compiled subgraph,
    AFTER sub_coord but BEFORE the first worker.  Uses LLMSimulatedGateway only
    (no interrupt() API — see WARNING in _build_subgraph_with_human).

  Back-compat: without human_cfg.enabled the graph is byte-for-byte identical
    to the pre-M9.1 Hierarchical graph.
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState
from atm.core.types import HumanContext, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop

if TYPE_CHECKING:
    from atm.experiment.config import HumanCfg
    from atm.human.gateway import HumanGateway

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for HITL — module-level so tests can patch them
# ---------------------------------------------------------------------------

try:
    from atm.human.llm_simulated import LLMSimulatedGateway
except ImportError:  # pragma: no cover
    LLMSimulatedGateway = None  # type: ignore[assignment,misc]

try:
    from atm.human.cli_gateway import CLIGateway
except ImportError:  # pragma: no cover
    CLIGateway = None  # type: ignore[assignment,misc]

try:
    from atm.human._timeout import request_with_timeout
except ImportError:  # pragma: no cover
    request_with_timeout = None  # type: ignore[assignment]

try:
    from langchain_core.callbacks.manager import adispatch_custom_event
except ImportError:  # pragma: no cover
    adispatch_custom_event = None  # type: ignore[assignment]

HumanRoleRouter: Any
try:
    from atm.human.role_router import HumanRoleRouter as HumanRoleRouter
except ImportError:  # pragma: no cover
    HumanRoleRouter = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ROUNDS = 4
_DEFAULT_FINALIZE_SIGNAL = "top_coord_finalize"
_ROUTE_TEAM_A = "team_a"
_ROUTE_TEAM_B = "team_b"
_ROUTE_FINALIZE = "hierarchical_finalize"
_ROUTE_END = "__end__"


# ---------------------------------------------------------------------------
# Helper: extract last DRAFT content from a specific agent's outbox
# ---------------------------------------------------------------------------


def _extract_draft(state: dict[str, Any], agent_id: str) -> str | None:
    """Extract the last DRAFT message content from the given agent's outbox.

    Args:
        state: GraphState-like dict.
        agent_id: Agent identifier to look up in state["agents"].

    Returns:
        The content string of the last DRAFT message, or None if not found.
    """
    agents: dict[str, Any] = dict(state.get("agents") or {})
    agent_state: dict[str, Any] = dict(agents.get(agent_id) or {})
    outbox: list[Any] = list(agent_state.get("outbox") or [])

    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DRAFT or str(kind) == "draft":
            content = getattr(msg, "content", None)
            if content:
                return str(content)
    return None


# ---------------------------------------------------------------------------
# HITL node builder — top-scope
# ---------------------------------------------------------------------------


def _build_human_top_reviewer_node(
    human_cfg: HumanCfg,
    gateway: HumanGateway,
    *,
    role_router: Any = None,
) -> Any:
    """Build and return async human_top_reviewer node for scope='top'.

    Dispatches human_request / human_response events (same canonical
    format as Chain reference-implementation) and updates state:
      approve  → shared["human_approved"] = True
      reject   → append Message + shared["needs_rerun"] = True
      abstain/timeout → no-op

    Args:
        human_cfg: HumanCfg instance with role/timeout settings.
        gateway:   Pre-constructed HumanGateway instance.
        role_router: Optional HumanRoleRouter; when not None, overrides human_cfg.role
            dynamically via ``await role_router.decide(phase, state)``.

    Returns:
        Async callable compatible with LangGraph node signature.
    """
    from atm.core.types import Message

    _role_router = role_router

    async def human_top_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        shared: dict[str, Any] = dict(deepcopy(state.get("shared", {})))
        run_id = shared.get("run_id")
        if run_id is None:
            raise RuntimeError(
                "human_top_reviewer requires state['shared']['run_id']. "
                "Ensure Runner._build_initial_state sets run_id."
            )

        import uuid as _uuid

        if not isinstance(run_id, _uuid.UUID):
            run_id = _uuid.UUID(str(run_id))

        iter_total: int = int(shared.get("iter_total", 0))

        # Build question from last available team draft or generic
        signals: dict[str, Any] = dict(shared.get("signals") or {})
        team_a_draft = signals.get("team_a_draft")
        team_b_draft = signals.get("team_b_draft")
        if team_a_draft and team_b_draft:
            question = (
                f"Team A draft: {str(team_a_draft)[:200]}\n"
                f"Team B draft: {str(team_b_draft)[:200]}\n"
                "Please review and decide to approve, reject, or abstain."
            )
        else:
            question = (
                "Please review the hierarchical team outputs and approve, reject, or abstain."
            )

        # Resolve active role (dynamic via role_router or static from cfg)
        from atm.core.types import Phase

        if _role_router is not None:
            _raw_phase = shared.get("phase", "execution")
            _phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            active_role = await _role_router.decide(_phase, shared)
        else:
            active_role = human_cfg.role

        ctx = HumanContext(
            run_id=run_id,
            role=active_role,
            question=question,
            recent_messages=(),
            allowed_actions=("approve", "reject", "abstain"),
        )

        request_id: str = f"hierarchical:top:{iter_total}:reviewer"

        # Dispatch human_request
        try:
            if adispatch_custom_event is not None:
                await adispatch_custom_event(
                    "human_request",
                    {
                        "run_id": run_id,
                        "request_id": request_id,
                        "role": str(
                            active_role.value
                            if hasattr(active_role, "value")
                            else active_role
                        ),
                        "context_json": ctx.model_dump(mode="json"),
                        "requested_at": datetime.now(UTC),
                    },
                )
        except Exception:
            logger.debug("adispatch human_request skipped (no callback ctx)", exc_info=True)

        # Call gateway (with timeout wrapper if configured)
        timeout_s_val: float | None = getattr(human_cfg, "timeout_s", None)
        policy: str = getattr(human_cfg, "timeout_policy", "skip")

        _t0 = time.monotonic()

        if request_with_timeout is not None and timeout_s_val is not None:
            _fallback_gateway: Any = None
            if policy == "llm_fallback" and LLMSimulatedGateway is not None:
                _fb_llm: Any = getattr(gateway, "_llm", None)
                _fallback_gateway = LLMSimulatedGateway(llm=_fb_llm)
            response = await request_with_timeout(
                gateway,
                ctx,
                request_id=request_id,
                timeout_s=timeout_s_val,
                policy=policy,  # type: ignore[arg-type]
                llm_fallback_gateway=_fallback_gateway,
            )
        else:
            response = await gateway.request(ctx, request_id=request_id)

        latency_s = time.monotonic() - _t0

        # Dispatch human_response
        try:
            if adispatch_custom_event is not None:
                await adispatch_custom_event(
                    "human_response",
                    {
                        "run_id": run_id,
                        "request_id": request_id,
                        "answered_at": datetime.now(UTC),
                        "response_json": response.model_dump(mode="json"),
                        "source": response.source,
                        "timed_out": response.timed_out,
                        "latency_s": latency_s,
                    },
                )
        except Exception:
            logger.debug("adispatch human_response skipped (no callback ctx)", exc_info=True)

        # Update state based on action
        action: str = response.action
        delta: dict[str, Any] = {}

        if action == "approve":
            shared["human_approved"] = True
            delta["shared"] = shared
        elif action == "reject":
            comment: str = response.comment or "Human reviewer rejected this iteration."
            rejection_msg = Message(
                sender="human_top_reviewer",
                kind=MessageKind.CRITIQUE,
                content=f"[Human Top Reviewer Rejection] {comment}",
                payload={"human_rejected": True, "comment": comment},
            )
            shared["needs_rerun"] = True
            delta["shared"] = shared
            delta["messages"] = [rejection_msg]

        # abstain / timeout → no-op
        return delta

    return human_top_reviewer


# ---------------------------------------------------------------------------
# HITL node builder — sub_team scope (inside subgraph, no interrupt())
# ---------------------------------------------------------------------------


def _build_human_sub_reviewer_node(
    team_id: str,
    human_cfg: HumanCfg,
    gateway: HumanGateway,
    *,
    role_router: Any = None,
) -> Any:
    """Build and return async human_sub_reviewer node for scope='sub_team'.

    WARNING: CLIGateway is NOT supported in sub_team scope.
    The interactive interrupt() API is incompatible with LangGraph compiled
    subgraphs + checkpointer (flagged risk arch.md §M9.1 exit criterion,
    lines 560-561).  Only LLMSimulatedGateway (synchronous) is supported.
    CLIGateway callers MUST use scope='top' or wait for M9.2.

    This node uses adispatch_custom_event to write human_interactions rows
    from inside the subgraph context.  LangGraph 0.3+ propagates RunnableConfig
    (including callbacks) into compiled subgraph ainvoke, so the callback
    handler receives these events correctly.

    Args:
        team_id:   Team identifier (e.g. "team_a", "team_b") for request_id.
        human_cfg: HumanCfg with role/timeout settings.
        gateway:   Pre-constructed HumanGateway instance (LLMSimulatedGateway).
        role_router: Optional HumanRoleRouter; when not None, overrides human_cfg.role
            dynamically via ``await role_router.decide(phase, state)``.

    Returns:
        Async callable compatible with LangGraph node signature.
    """
    _team_id = team_id
    _role_router = role_router

    async def human_sub_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        shared: dict[str, Any] = dict(deepcopy(state.get("shared", {})))
        run_id = shared.get("run_id")
        if run_id is None:
            raise RuntimeError(
                f"human_sub_reviewer ({_team_id}) requires state['shared']['run_id'] to be set."
            )

        import uuid as _uuid

        if not isinstance(run_id, _uuid.UUID):
            run_id = _uuid.UUID(str(run_id))

        iter_total: int = int(shared.get("iter_total", 0))

        question = f"Please review sub-team {_team_id} activity and approve, reject, or abstain."

        # Resolve active role (dynamic via role_router or static from cfg)
        from atm.core.types import Phase

        if _role_router is not None:
            _raw_phase = shared.get("phase", "execution")
            _phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            active_role = await _role_router.decide(_phase, shared)
        else:
            active_role = human_cfg.role

        ctx = HumanContext(
            run_id=run_id,
            role=active_role,
            question=question,
            recent_messages=(),
            allowed_actions=("approve", "reject", "abstain"),
        )

        request_id: str = f"hierarchical:{_team_id}:{iter_total}:reviewer"

        # Dispatch human_request from within subgraph context.
        # adispatch_custom_event propagates through LangGraph's callback chain
        # into the parent graph's ExperimentCallbackHandler (verified by
        # test_hitl_hierarchical.py sub_team scenario assertion).
        try:
            if adispatch_custom_event is not None:
                await adispatch_custom_event(
                    "human_request",
                    {
                        "run_id": run_id,
                        "request_id": request_id,
                        "role": str(
                            active_role.value
                            if hasattr(active_role, "value")
                            else active_role
                        ),
                        "context_json": ctx.model_dump(mode="json"),
                        "requested_at": datetime.now(UTC),
                    },
                )
        except Exception:
            logger.debug(
                "adispatch human_request skipped in subgraph %s (no callback ctx)",
                _team_id,
                exc_info=True,
            )

        # Gateway call — LLMSimulatedGateway only in sub_team scope
        timeout_s_val: float | None = getattr(human_cfg, "timeout_s", None)
        policy: str = getattr(human_cfg, "timeout_policy", "skip")

        _t0 = time.monotonic()

        if request_with_timeout is not None and timeout_s_val is not None:
            _fallback_gateway: Any = None
            if policy == "llm_fallback" and LLMSimulatedGateway is not None:
                _fb_llm: Any = getattr(gateway, "_llm", None)
                _fallback_gateway = LLMSimulatedGateway(llm=_fb_llm)
            response = await request_with_timeout(
                gateway,
                ctx,
                request_id=request_id,
                timeout_s=timeout_s_val,
                policy=policy,  # type: ignore[arg-type]
                llm_fallback_gateway=_fallback_gateway,
            )
        else:
            response = await gateway.request(ctx, request_id=request_id)

        latency_s = time.monotonic() - _t0

        # Dispatch human_response from within subgraph
        try:
            if adispatch_custom_event is not None:
                await adispatch_custom_event(
                    "human_response",
                    {
                        "run_id": run_id,
                        "request_id": request_id,
                        "answered_at": datetime.now(UTC),
                        "response_json": response.model_dump(mode="json"),
                        "source": response.source,
                        "timed_out": response.timed_out,
                        "latency_s": latency_s,
                    },
                )
        except Exception:
            logger.debug(
                "adispatch human_response skipped in subgraph %s (no callback ctx)",
                _team_id,
                exc_info=True,
            )

        # sub_team scope: no state modification — purely observational reviewer
        # (approve/reject/abstain all produce no-op state delta in sub_team mode)
        return {}

    return human_sub_reviewer


# ---------------------------------------------------------------------------
# HierarchicalTopology
# ---------------------------------------------------------------------------


@TopologyRegistry.register("hierarchical")
class HierarchicalTopology:
    """Two-level hierarchical topology with compiled subgraphs.

    Graph structure (top-level):
      START → top_coord → conditional routing →
        team_a (subgraph node) → top_coord (loop)
        team_b (subgraph node) → top_coord (loop)
        hierarchical_finalize → END

    Each subgraph (team_a, team_b):
      START → sub_coord_X → worker_X1 → worker_X2 → END

    With HITL enabled (human_cfg.enabled=True):
      scope="top" (default):
        after_team_b node routes to human_top_reviewer before hierarchical_finalize
        when the finalize condition is met.  Otherwise loops back to team_a.

      scope="sub_team":
        human_sub_reviewer node inserted after sub_coord_X in each subgraph.
        WARNING: CLIGateway not supported in sub_team scope (interrupt()
        incompatible with subgraph). Use LLMSimulatedGateway only.

    The top_coord node:
      - Increments iter_total.
      - On first iteration (iter_total == 1): sets signal to activate both sub-teams.
      - On subsequent iterations: checks if both team drafts exist → finalize.
      - If max_rounds reached → routes to END.

    top_coord and sub_coord closures are NOT Agents — they are rule-based.
    They are NOT in the agents dict passed to build().

    Only workers are in the agents dict.
    """

    name = "hierarchical"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a CompiledStateGraph.

        Args:
            agents: Dict mapping agent_id → Agent instance.
                    Expected keys: worker agent ids from sub_teams config.
                    top_coord and sub_coord are NOT in agents dict (MC-4).
            cfg:    TopologyConfig with max_iterations and extra fields.
            **kwargs: Optional; checkpointer=... is forwarded to graph.compile().
                      human_cfg=HumanCfg enables HITL nodes.
                      human_gateway_llm=LLMWrapper used for LLMSimulatedGateway.

        Returns:
            CompiledStateGraph ready for ainvoke.

        Raises:
            ValueError: If any sub_team config contains a nested "sub_teams" key
                        (strict 2-level invariant — MC-1).
        """
        checkpointer = kwargs.get("checkpointer")

        # Extract configuration
        extra = cfg.extra or {}
        max_rounds: int = int(extra.get("max_rounds", _DEFAULT_MAX_ROUNDS))
        final_answer_strategy: str = str(extra.get("final_answer_strategy", "json_concat"))
        finalize_signal: str = str(extra.get("finalize_signal", _DEFAULT_FINALIZE_SIGNAL))

        sub_teams: list[dict[str, Any]] = list(extra.get("sub_teams") or [])
        if len(sub_teams) < 2:
            # Default to two teams if not configured
            sub_teams = [
                {"team_id": "team_a", "workers": ["executor_a1", "executor_a2"]},
                {"team_id": "team_b", "workers": ["executor_b1", "executor_b2"]},
            ]

        # ----------------------------------------------------------------
        # Strict 2-level invariant validation (MC-1)
        # ----------------------------------------------------------------
        for i, team_cfg in enumerate(sub_teams):
            if "sub_teams" in team_cfg:
                raise ValueError(
                    f"HierarchicalTopology: sub_teams[{i}] contains a nested 'sub_teams' key. "
                    f"Only 2 levels are allowed (arch.md §7.6). "
                    f"3-level nesting is not supported in this implementation."
                )

        team_a_cfg = sub_teams[0]
        team_b_cfg = sub_teams[1]
        team_a_id: str = str(team_a_cfg.get("team_id", "team_a"))
        team_b_id: str = str(team_b_cfg.get("team_id", "team_b"))
        workers_a: list[str] = list(team_a_cfg.get("workers") or [])
        workers_b: list[str] = list(team_b_cfg.get("workers") or [])

        # ----------------------------------------------------------------
        # HITL config extraction
        # ----------------------------------------------------------------
        human_cfg: HumanCfg | None = kwargs.get("human_cfg")
        role_router: Any = kwargs.get("role_router")
        hitl_enabled = human_cfg is not None and human_cfg.enabled
        hitl_scope: str = "top"  # default scope
        if hitl_enabled and human_cfg is not None:
            human_extra = human_cfg.extra or {}
            hitl_scope = str(human_extra.get("scope", "top"))

        # ----------------------------------------------------------------
        # Build gateway (shared for both top + sub_team scopes)
        # ----------------------------------------------------------------
        gateway_instance: Any = None
        if hitl_enabled and human_cfg is not None:
            gateway_llm: Any = kwargs.get("human_gateway_llm")
            if human_cfg.gateway == "cli":
                gateway_instance = CLIGateway() if CLIGateway is not None else None
            else:
                gateway_instance = (
                    LLMSimulatedGateway(llm=gateway_llm)
                    if LLMSimulatedGateway is not None
                    else None
                )

            if gateway_instance is None:  # pragma: no cover
                raise ImportError(
                    f"Gateway class for '{human_cfg.gateway}' could not be imported. "
                    "Ensure atm.human is installed."
                )

        # ----------------------------------------------------------------
        # Build team subgraphs
        # ----------------------------------------------------------------
        if hitl_enabled and hitl_scope == "sub_team" and human_cfg is not None:
            # sub_team scope: insert human reviewer inside each subgraph
            team_a_subgraph = self._build_subgraph_with_human(
                team_a_id, workers_a, agents, human_cfg, gateway_instance,
                role_router=role_router,
            )
            team_b_subgraph = self._build_subgraph_with_human(
                team_b_id, workers_b, agents, human_cfg, gateway_instance,
                role_router=role_router,
            )
        else:
            team_a_subgraph = self._build_subgraph(team_a_id, workers_a, agents)
            team_b_subgraph = self._build_subgraph(team_b_id, workers_b, agents)

        # ----------------------------------------------------------------
        # Build top_coord node (rule-based closure)
        # ----------------------------------------------------------------

        async def top_coord_node(state: GraphState) -> dict[str, Any]:
            """Rule-based top coordinator: increments counters, signals sub-team activation."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            # Increment global counters
            new_iter_total = int(shared.get("iter_total") or 0) + 1
            new_iteration = int(shared.get("iteration") or 0) + 1
            shared["iter_total"] = new_iter_total
            shared["iteration"] = new_iteration

            # First iteration: signal that both sub-teams should activate
            if new_iter_total == 1:
                signals["activate_team_a"] = True
                signals["activate_team_b"] = True

            shared["signals"] = signals
            return {"shared": shared}

        # ----------------------------------------------------------------
        # Build routing function for top_coord
        # ----------------------------------------------------------------

        def _route_from_top_coord(state: GraphState) -> str:
            """Routing function — decides next step after top_coord."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            iter_total: int = int(shared.get("iter_total") or 0)

            # Check global stop and topology_success through unified _should_stop path
            # topology_success is True when finalize_signal is set (MC-4 §7.1)
            topology_success = bool(signals.get(finalize_signal))
            stop, _reason = _should_stop(
                dict(state),
                cfg,
                topology_success=topology_success,
                topology_max_reached=False,
            )
            if stop:
                return _ROUTE_FINALIZE

            # First iteration: activate both teams (route to team_a first)
            activate_a = bool(signals.get("activate_team_a"))
            activate_b = bool(signals.get("activate_team_b"))

            if activate_a:
                return _ROUTE_TEAM_A
            if activate_b:
                return _ROUTE_TEAM_B

            # Check if both drafts present → finalize
            team_a_draft = signals.get("team_a_draft")
            team_b_draft = signals.get("team_b_draft")
            if team_a_draft and team_b_draft:
                return _ROUTE_FINALIZE

            # Max rounds exceeded → finalize
            if iter_total > max_rounds:
                return _ROUTE_FINALIZE

            # Default: route to team_a to continue
            return _ROUTE_TEAM_A

        # ----------------------------------------------------------------
        # Build sub_coord node for team_a (rule-based closure)
        # ----------------------------------------------------------------

        _team_a_workers = workers_a
        _team_b_workers = workers_b

        async def top_coord_after_team_a(state: GraphState) -> dict[str, Any]:
            """Post-team_a coordinator: collects team_a draft, activates team_b."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            # Collect last DRAFT from team_a workers
            team_a_draft: str | None = None
            for worker_id in _team_a_workers:
                draft = _extract_draft(dict(state), worker_id)
                if draft:
                    team_a_draft = draft

            if team_a_draft:
                signals["team_a_draft"] = team_a_draft

            # Clear activate_team_a, activate team_b next
            signals.pop("activate_team_a", None)
            signals["activate_team_b"] = True

            # Increment counters
            new_iter_total = int(shared.get("iter_total") or 0) + 1
            new_iteration = int(shared.get("iteration") or 0) + 1
            shared["iter_total"] = new_iter_total
            shared["iteration"] = new_iteration
            shared["signals"] = signals
            return {"shared": shared}

        async def top_coord_after_team_b(state: GraphState) -> dict[str, Any]:
            """Post-team_b coordinator: collects team_b draft, checks finalize condition."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            # Collect last DRAFT from team_b workers
            team_b_draft: str | None = None
            for worker_id in _team_b_workers:
                draft = _extract_draft(dict(state), worker_id)
                if draft:
                    team_b_draft = draft

            if team_b_draft:
                signals["team_b_draft"] = team_b_draft

            # Clear activate_team_b
            signals.pop("activate_team_b", None)

            # If both team drafts present → set finalize signal
            team_a_draft = signals.get("team_a_draft")
            if team_a_draft and team_b_draft:
                signals[finalize_signal] = True
                logger.info("top_coord: both team drafts present, setting %s=True", finalize_signal)

            # Increment counters
            new_iter_total = int(shared.get("iter_total") or 0) + 1
            new_iteration = int(shared.get("iteration") or 0) + 1
            shared["iter_total"] = new_iter_total
            shared["iteration"] = new_iteration
            shared["signals"] = signals
            return {"shared": shared}

        def _route_from_after_team_a(state: GraphState) -> str:
            """Route after team_a processing."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            # Always prefer going to team_b if activate_team_b is set
            if signals.get("activate_team_b"):
                return _ROUTE_TEAM_B

            team_a_draft = signals.get("team_a_draft")
            team_b_draft = signals.get("team_b_draft")
            if team_a_draft and team_b_draft:
                return _ROUTE_FINALIZE

            return _ROUTE_TEAM_B

        def _route_from_after_team_b(state: GraphState) -> str:
            """Route after team_b processing.

            With HITL scope='top': when finalize condition met, route to
            human_top_reviewer first (before hierarchical_finalize).
            Without HITL or on continue path: routes to hierarchical_finalize
            or team_a as before.
            """
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            iter_total: int = int(shared.get("iter_total") or 0)

            # Unified stop check: max_iter > topology_success > topology_max (§7.1)
            topology_success = bool(signals.get(finalize_signal))
            topology_max_reached = iter_total > max_rounds
            stop, _reason = _should_stop(
                dict(state),
                cfg,
                topology_success=topology_success,
                topology_max_reached=topology_max_reached,
            )
            if stop:
                if hitl_enabled and hitl_scope == "top":
                    return "human_top_reviewer"
                return _ROUTE_FINALIZE

            return _ROUTE_TEAM_A

        # ----------------------------------------------------------------
        # Build hierarchical_finalize node
        # ----------------------------------------------------------------

        async def hierarchical_finalize_node(state: GraphState) -> dict[str, Any]:
            """Aggregate team drafts into JSON-concat final_answer (MC-6)."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            team_a_draft = signals.get("team_a_draft")
            team_b_draft = signals.get("team_b_draft")

            # Fallback: scan agent outboxes directly
            if not team_a_draft:
                for worker_id in _team_a_workers:
                    draft = _extract_draft(dict(state), worker_id)
                    if draft:
                        team_a_draft = draft
                        break

            if not team_b_draft:
                for worker_id in _team_b_workers:
                    draft = _extract_draft(dict(state), worker_id)
                    if draft:
                        team_b_draft = draft
                        break

            # Build JSON-concat final answer
            if final_answer_strategy == "json_concat":
                final_answer = json.dumps(
                    {
                        team_a_id: team_a_draft or "<incomplete>",
                        team_b_id: team_b_draft or "<incomplete>",
                    },
                    ensure_ascii=False,
                )
            else:
                # Fallback: simple concat
                parts = []
                if team_a_draft:
                    parts.append(f"[{team_a_id}] {team_a_draft}")
                if team_b_draft:
                    parts.append(f"[{team_b_id}] {team_b_draft}")
                final_answer = " | ".join(parts) if parts else "<incomplete>"

            shared["final_answer"] = final_answer
            shared["signals"] = signals
            return {"shared": shared}

        # ----------------------------------------------------------------
        # Assemble the top-level graph
        # ----------------------------------------------------------------

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        # Add nodes
        graph.add_node("top_coord", top_coord_node)
        graph.add_node("after_team_a", top_coord_after_team_a)
        graph.add_node("after_team_b", top_coord_after_team_b)

        # Add subgraph nodes — subgraphs embedded as nodes (LangGraph pattern)
        graph.add_node(_ROUTE_TEAM_A, team_a_subgraph)
        graph.add_node(_ROUTE_TEAM_B, team_b_subgraph)

        graph.add_node(_ROUTE_FINALIZE, hierarchical_finalize_node)

        # Entry point
        graph.add_edge(START, "top_coord")

        # top_coord → conditional routing
        graph.add_conditional_edges(
            "top_coord",
            _route_from_top_coord,
            {
                _ROUTE_TEAM_A: _ROUTE_TEAM_A,
                _ROUTE_TEAM_B: _ROUTE_TEAM_B,
                _ROUTE_FINALIZE: _ROUTE_FINALIZE,
                _ROUTE_END: END,
            },
        )

        # team_a → after_team_a → conditional routing
        graph.add_edge(_ROUTE_TEAM_A, "after_team_a")
        graph.add_conditional_edges(
            "after_team_a",
            _route_from_after_team_a,
            {
                _ROUTE_TEAM_B: _ROUTE_TEAM_B,
                _ROUTE_FINALIZE: _ROUTE_FINALIZE,
                _ROUTE_END: END,
            },
        )

        # team_b → after_team_b → conditional routing (HITL scope=top: may route to reviewer)
        graph.add_edge(_ROUTE_TEAM_B, "after_team_b")

        if hitl_enabled and hitl_scope == "top" and human_cfg is not None:
            # Insert human_top_reviewer as an intermediate step before finalize
            node_fn = _build_human_top_reviewer_node(human_cfg, gateway_instance, role_router=role_router)
            graph.add_node("human_top_reviewer", node_fn)

            # after_team_b → conditional (finalize condition → human_top_reviewer | loop)
            graph.add_conditional_edges(
                "after_team_b",
                _route_from_after_team_b,
                {
                    _ROUTE_TEAM_A: _ROUTE_TEAM_A,
                    _ROUTE_FINALIZE: _ROUTE_FINALIZE,
                    _ROUTE_END: END,
                    "human_top_reviewer": "human_top_reviewer",
                },
            )
            # human_top_reviewer → hierarchical_finalize (always proceed after review)
            graph.add_edge("human_top_reviewer", _ROUTE_FINALIZE)
        else:
            graph.add_conditional_edges(
                "after_team_b",
                _route_from_after_team_b,
                {
                    _ROUTE_TEAM_A: _ROUTE_TEAM_A,
                    _ROUTE_FINALIZE: _ROUTE_FINALIZE,
                    _ROUTE_END: END,
                },
            )

        # finalize → END
        graph.add_edge(_ROUTE_FINALIZE, END)

        return graph.compile(checkpointer=checkpointer)

    def _build_subgraph(
        self,
        team_id: str,
        worker_ids: list[str],
        agents: dict[str, Any],
    ) -> Any:
        """Build a compiled subgraph for a sub-team.

        The subgraph has a rule-based sub_coord node followed by worker nodes in order.

        Args:
            team_id: Identifier for this team (e.g., "team_a").
            worker_ids: Ordered list of worker agent_ids for this team.
            agents: The agents dict passed to build().

        Returns:
            A compiled subgraph (CompiledStateGraph).
        """
        _worker_ids = list(worker_ids)
        _team_id = team_id

        # Sub-coordinator: rule-based closure (NOT in agents dict)
        async def sub_coord_node(state: GraphState) -> dict[str, Any]:
            """Rule-based sub-coordinator: initialises sub-team iteration."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            # Mark this sub-team as active
            signals[f"{_team_id}_active"] = True
            shared["signals"] = signals
            return {"shared": shared}

        # Build subgraph
        sub_graph: StateGraph[GraphState] = StateGraph(GraphState)

        # Add sub_coord node (rule-based, NOT in agents dict)
        sub_coord_name = f"sub_coord_{_team_id}"
        sub_graph.add_node(sub_coord_name, sub_coord_node)
        sub_graph.add_edge(START, sub_coord_name)

        # Add worker nodes in order
        if not _worker_ids:
            # No workers — just sub_coord → END
            sub_graph.add_edge(sub_coord_name, END)
        else:
            # Chain workers sequentially
            prev_node = sub_coord_name
            for worker_id in _worker_ids:
                worker_agent = agents.get(worker_id)

                if worker_agent is None:
                    logger.warning(
                        "_build_subgraph: worker %r not found in agents dict; using no-op node.",
                        worker_id,
                    )

                    async def _noop(state: GraphState, _wid: str = worker_id) -> dict[str, Any]:
                        return {
                            "agents": {_wid: {"agent_id": _wid, "outbox": []}},
                        }

                    sub_graph.add_node(worker_id, _noop)
                else:
                    _agent = worker_agent

                    async def _worker_node(
                        state: GraphState,
                        _a: Any = _agent,
                        _wid: str = worker_id,
                    ) -> dict[str, Any]:
                        result: dict[str, Any] = await _a.step(state)
                        return result

                    sub_graph.add_node(worker_id, _worker_node)

                sub_graph.add_edge(prev_node, worker_id)
                prev_node = worker_id

            # Last worker → END
            sub_graph.add_edge(prev_node, END)

        return sub_graph.compile()

    def _build_subgraph_with_human(
        self,
        team_id: str,
        worker_ids: list[str],
        agents: dict[str, Any],
        human_cfg: HumanCfg,
        gateway: Any,
        *,
        role_router: Any = None,
    ) -> Any:
        """Build a compiled subgraph with human_sub_reviewer inserted after sub_coord.

        WARNING: CLIGateway is NOT supported in sub_team scope.
        The interactive interrupt() API is incompatible with LangGraph compiled
        subgraphs + checkpointer (flagged risk arch.md §M9.1).  Only
        LLMSimulatedGateway (synchronous) is supported here.
        CLIGateway callers MUST use scope='top' or wait for M9.2.

        The human_sub_reviewer node uses adispatch_custom_event to write
        human_interactions rows.  LangGraph 0.3+ propagates RunnableConfig
        (including callbacks) into compiled subgraph ainvoke, so the
        ExperimentCallbackHandler receives these events from inside the subgraph.

        Args:
            team_id: Identifier for this team (e.g., "team_a").
            worker_ids: Ordered list of worker agent_ids for this team.
            agents: The agents dict passed to build().
            human_cfg: HumanCfg with role/timeout settings.
            gateway: Pre-constructed HumanGateway instance.
            role_router: Optional HumanRoleRouter; when not None, overrides human_cfg.role
                dynamically via ``await role_router.decide(phase, state)``.

        Returns:
            A compiled subgraph (CompiledStateGraph) with human node inserted.
        """
        _worker_ids = list(worker_ids)
        _team_id = team_id

        async def sub_coord_node(state: GraphState) -> dict[str, Any]:
            """Rule-based sub-coordinator: initialises sub-team iteration."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            signals[f"{_team_id}_active"] = True
            shared["signals"] = signals
            return {"shared": shared}

        sub_graph: StateGraph[GraphState] = StateGraph(GraphState)

        sub_coord_name = f"sub_coord_{_team_id}"
        sub_graph.add_node(sub_coord_name, sub_coord_node)
        sub_graph.add_edge(START, sub_coord_name)

        # Insert human_sub_reviewer immediately after sub_coord
        human_node_name = f"human_sub_reviewer_{_team_id}"
        human_node_fn = _build_human_sub_reviewer_node(_team_id, human_cfg, gateway, role_router=role_router)
        sub_graph.add_node(human_node_name, human_node_fn)
        sub_graph.add_edge(sub_coord_name, human_node_name)

        # Add worker nodes in order (chained after human reviewer)
        if not _worker_ids:
            sub_graph.add_edge(human_node_name, END)
        else:
            prev_node = human_node_name
            for worker_id in _worker_ids:
                worker_agent = agents.get(worker_id)

                if worker_agent is None:
                    logger.warning(
                        "_build_subgraph_with_human: worker %r not found in agents dict; "
                        "using no-op node.",
                        worker_id,
                    )

                    async def _noop(state: GraphState, _wid: str = worker_id) -> dict[str, Any]:
                        return {
                            "agents": {_wid: {"agent_id": _wid, "outbox": []}},
                        }

                    sub_graph.add_node(worker_id, _noop)
                else:
                    _agent = worker_agent

                    async def _worker_node(
                        state: GraphState,
                        _a: Any = _agent,
                        _wid: str = worker_id,
                    ) -> dict[str, Any]:
                        result: dict[str, Any] = await _a.step(state)
                        return result

                    sub_graph.add_node(worker_id, _worker_node)

                sub_graph.add_edge(prev_node, worker_id)
                prev_node = worker_id

            sub_graph.add_edge(prev_node, END)

        return sub_graph.compile()
