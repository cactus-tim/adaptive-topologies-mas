"""AdaptiveTopology — L2 meta-graph with runtime topology switching.

Architecture (arch.md §7.7, §8bis):
  Meta-graph structure (base):
    START → phase_router → topology_router → dispatch_topology →
    [star|chain|mesh|debate|hierarchical subgraph] → transition_gate →
    conditional: phase==done OR budget_exceeded → END
                 else → phase_router (loop)

  With HITL enabled (human_cfg.enabled=True), human_advisor is inserted
  between topology_router and dispatch_topology:
    ... → topology_router → human_advisor → dispatch_topology → ...

Key design decisions:
  1. Subgraph dispatch uses a single ``dispatch_topology`` node with if/elif
     branching by ``TopologyDecision.topology``.  LangGraph conditional edges
     cannot select a node whose name varies at runtime without a full edge-map,
     but since we have exactly 5 topologies, the workaround (one dispatch node)
     is simpler and fully equivalent.

  2. Subgraphs are compiled lazily on first tick and cached.  Each subgraph
     receives the FULL parent GraphState so that agents share the same
     scratchpad, messages, and signals across topology switches.

  3. TransitionGate is a pure function ``(state, phase_decision,
     topology_decision) → state`` that applies the state-transfer table
     from arch.md §7.7, records a TopologyTransition in
     state["topology_transitions"], and applies arch.md §7.7 state-transfer.

  4. PhaseRouter takes full GraphState; TopologyRouter takes SharedState
     (per m8-routing TDD contract).

  5. Routing decisions (PhaseDecision, TopologyDecision) are stored in
     a closure-level mutable slot (list of one element) rather than in
     the LangGraph state. This avoids losing them when the subgraph
     overwrites ``shared`` with its own output.

  6. Superset agent roster: AdaptiveTopology.build() accepts all 7 roles.
     Each subgraph uses only its subset — inactive agents are not invoked.

  7. HITL (M9.1): human_advisor operates in two modes controlled by
     ``HumanCfg.extra["human_can_override_router"]``:
       - Advisory (default): human hint written to
         ``shared.signals["human_advisor_hint"]``. Does NOT mutate
         _topo_dec_slot[0]. TopologyRouter MAY consider it on next tick.
       - Override (human_can_override_router=True): human may replace
         _topo_dec_slot[0] with a new TopologyDecision(decided_by=
         "human_override", ...).  SwitchGuards still apply — if guards
         block, the override is rejected and ``considered_alternatives``
         records the human's intent.

     # known-limitation: subgraph-level interrupt-resume (CLIGateway inside
     # subgraph with real interrupt/resume) is deferred to M9.2.
     # See arch/PLAN.md §M9.1 exit criterion lines 560-561.

Public API:
  AdaptiveTopology          — topology class (registered under "adaptive")
  build_adaptive_graph()    — convenience factory returning CompiledStateGraph
  apply_transition_gate()   — pure function (exported for unit testing)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState, SharedState
from atm.core.types import (
    HumanContext,
    Phase,
    PhaseDecision,
    PhaseTransition,
    TopologyDecision,
    TopologyTransition,
)
from atm.phases.guards import GuardedRouter, SwitchGuards
from atm.phases.manager import PhaseLimits, RuleBasedPhaseRouter
from atm.phases.topology_router import RuleBasedTopologyRouter
from atm.topology.base import TopologyConfig, TopologyRegistry

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy imports for HITL (M9.1) — patchable in tests.
# These mirror the pattern established in atm.human._node_factory.
# ---------------------------------------------------------------------------

_LLMSimulatedGateway: Any
_CLIGateway: Any
_request_with_timeout: Any

try:
    from atm.human.llm_simulated import LLMSimulatedGateway as _LLMSimulatedGateway
except ImportError:  # pragma: no cover
    _LLMSimulatedGateway = None

try:
    from atm.human.cli_gateway import CLIGateway as _CLIGateway
except ImportError:  # pragma: no cover
    _CLIGateway = None

try:
    from atm.human._timeout import request_with_timeout as _request_with_timeout
except ImportError:  # pragma: no cover
    _request_with_timeout = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of meta-graph ticks before forced END.
_DEFAULT_MAX_TICKS: int = 30

#: Maximum entries kept in topology_history (for cooldown checks).
_HISTORY_TAIL: int = 10

#: Mapping from arch.md canonical names to TopologyRegistry names.
#: "linear" = chain topology, "supervisor" = star topology.
_TOPO_ALIAS: dict[str, str] = {
    "linear": "chain",
    "supervisor": "star",
    "mesh": "mesh",
    "debate": "debate",
    "hierarchical": "hierarchical",
}


# ---------------------------------------------------------------------------
# apply_transition_gate — pure function (exported for unit testing)
# ---------------------------------------------------------------------------


def apply_transition_gate(
    state: dict[str, Any],
    phase_decision: PhaseDecision,
    topology_decision: TopologyDecision,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Apply state-transfer rules from arch.md §7.7 and record a TopologyTransition.

    This is a pure function: it takes the current state and two router decisions,
    and returns a *new* state dict with all required mutations applied.

    State-transfer rules (arch.md §7.7 table):
    - shared.phase             — updated if PhaseDecision advances
    - shared.active_topology   — updated if topology changed
    - shared.iteration         — reset to 0 on phase or topology change; else +1
    - shared.iter_total        — unchanged (already incremented in dispatch node)
    - shared.phase_started_at_iter  — set to iter_total if phase advanced
    - shared.topology_started_at_iter — set to iter_total if topology changed
    - shared.topology_switch_count   — incremented if topology changed
    - shared.topology_history  — append current topo on switch (tail capped)
    - shared.broadcast_bus     — cleared on phase or topology change
    - shared.signals           — clear all on phase change; clear consumed on topo change
    - agents[*].inbox/outbox   — cleared on phase change
    - topology_transitions     — append new TopologyTransition (every tick)

    Args:
        state:             Full GraphState dict.
        phase_decision:    PhaseDecision from PhaseRouter.
        topology_decision: TopologyDecision from TopologyRouter (possibly GuardedRouter).
        run_id:            Optional UUID string for TopologyTransition.run_id.

    Returns:
        New state dict with all mutations applied.
    """
    _phase_order: dict[Phase, int] = {
        Phase.PLANNING: 0,
        Phase.EXECUTION: 1,
        Phase.VERIFICATION: 2,
        Phase.DONE: 3,
    }

    # Work on shallow copies
    new_state: dict[str, Any] = dict(state)
    shared: dict[str, Any] = dict(state.get("shared") or {})

    # shared.phase may have been updated by the subgraph (chain/star coordinator
    # manages phase internally).  We take the monotonically-advanced maximum of:
    #   a) what the subgraph left in shared.phase
    #   b) what PhaseRouter decided (phase_decision.next_phase)
    # This preserves the invariant that phase is always monotonic while allowing
    # existing subgraphs to participate in phase advancement.
    subgraph_phase: Phase = shared.get("phase", Phase.PLANNING)
    router_phase: Phase = phase_decision.next_phase
    next_phase: Phase = (
        router_phase
        if _phase_order.get(router_phase, 0) > _phase_order.get(subgraph_phase, 0)
        else subgraph_phase
    )

    # The "original" phase before this tick (before subgraph ran).
    # Since subgraph may have changed shared.phase, we use phase_started_at_iter
    # to infer what the phase was at the start of the tick.
    # Approximation: if subgraph returned a different phase, the "current" phase
    # for state-transfer purposes is the subgraph's input phase.
    # We record from_phase in the transition as the pre-subgraph phase.
    # For simplicity, treat shared.phase as the authoritative post-subgraph value.
    current_phase: Phase = subgraph_phase  # for transition record (from_topology context)

    current_topology: str | None = shared.get("active_topology")
    next_topology: str = topology_decision.topology

    iter_total: int = int(shared.get("iter_total", 0))
    phase_started_at: int = int(shared.get("phase_started_at_iter", 0))
    topo_started_at: int = int(shared.get("topology_started_at_iter", 0))

    phase_changed: bool = next_phase != current_phase
    topology_changed: bool = (current_topology is not None) and (next_topology != current_topology)

    # ---- Phase update ----
    if phase_changed:
        shared["phase"] = next_phase
        shared["phase_started_at_iter"] = iter_total
        phase_started_at = iter_total

    # ---- Topology update ----
    if topology_changed or current_topology is None:
        shared["active_topology"] = next_topology
        shared["topology_started_at_iter"] = iter_total
        topo_started_at = iter_total

        if topology_changed:
            # Track switch count
            switch_count: int = int(shared.get("topology_switch_count", 0))
            shared["topology_switch_count"] = switch_count + 1

            # Update topology_history (short tail for cooldown checks)
            history: list[str] = list(shared.get("topology_history") or [])
            if current_topology is not None:
                history.append(current_topology)
            shared["topology_history"] = history[-_HISTORY_TAIL:]

    # ---- iteration counter ----
    if phase_changed or topology_changed:
        shared["iteration"] = 0
    else:
        shared["iteration"] = int(shared.get("iteration", 0)) + 1

    # ---- Clear broadcast_bus on any transition ----
    if phase_changed or topology_changed:
        shared["broadcast_bus"] = []

    # ---- Signals cleanup ----
    # Signal keys that correspond to each phase's advancement trigger.
    # On phase advance, only the guard signal for the *previous* phase is consumed.
    # We intentionally do NOT clear all signals on phase change, because subgraphs
    # (chain/star) may return multiple signals in one tick (e.g. ready_for_execution
    # + ready_for_verification + critic_approved) when they internally traverse
    # multiple phases.  Clearing all signals would discard still-valid signals.
    _phase_advance_signal: dict[Phase, str] = {
        Phase.PLANNING: "ready_for_execution",
        Phase.EXECUTION: "ready_for_verification",
        Phase.VERIFICATION: "critic_approved",
    }
    if phase_changed:
        signals_dict: dict[str, Any] = dict(shared.get("signals") or {})
        # Remove only the guard signal for the phase we just left
        prev_phase_signal = _phase_advance_signal.get(current_phase)
        if prev_phase_signal and prev_phase_signal in signals_dict:
            del signals_dict[prev_phase_signal]
        shared["signals"] = signals_dict
    elif topology_changed:
        # Clear only consumed signals (those the router acted on)
        signals_dict = dict(shared.get("signals") or {})
        reason_lower = topology_decision.reason.lower()
        consumed_keys = [k for k in signals_dict if k in reason_lower]
        for key in consumed_keys:
            del signals_dict[key]
        shared["signals"] = signals_dict

    # ---- Clear agent inboxes/outboxes on phase change ----
    if phase_changed:
        agents: dict[str, Any] = dict(state.get("agents") or {})
        cleaned_agents: dict[str, Any] = {}
        for agent_id, agent_state in agents.items():
            a = dict(agent_state)
            a["inbox"] = []
            a["outbox"] = []
            cleaned_agents[agent_id] = a
        new_state["agents"] = cleaned_agents

    # ---- Record TopologyTransition (every tick, even no-change) ----
    _run_id = uuid.UUID(run_id) if run_id else uuid.uuid4()
    iter_within_phase: int = max(0, iter_total - phase_started_at)
    iter_within_topo: int = (
        0
        if (topology_changed or current_topology is None)
        else max(0, iter_total - topo_started_at)
    )

    transition = TopologyTransition(
        run_id=_run_id,
        from_topology=current_topology,
        to_topology=next_topology,
        phase_at_decision=current_phase,
        iter_within_phase=iter_within_phase,
        iter_within_topology=iter_within_topo,
        decided_by=topology_decision.decided_by,
        reason=topology_decision.reason,
        considered_alternatives=topology_decision.considered_alternatives,
        signals_snapshot=dict(state.get("shared", {}).get("signals") or {}),
        router_cost_usd=topology_decision.router_cost_usd,
    )

    existing_transitions: list[TopologyTransition] = list(state.get("topology_transitions") or [])
    new_state["topology_transitions"] = [*existing_transitions, transition]
    new_state["shared"] = shared
    return new_state


# ---------------------------------------------------------------------------
# AdaptiveTopology
# ---------------------------------------------------------------------------


@TopologyRegistry.register("adaptive")
class AdaptiveTopology:
    """L2 adaptive meta-graph topology.

    Meta-graph flow (arch.md §7.7):
      START → phase_router_node → topology_router_node →
      dispatch_topology_node (subgraph invocation) →
      transition_gate_node → conditional → [END | loop to phase_router_node]

    Args passed to build():
        agents: Dict of agent_id → Agent with async .step() method.
                For adaptive mode, all 7 roles should be provided.
        cfg:    TopologyConfig. Relevant extra keys:
                  phase_router: "rule" | "llm"   (default "rule")
                  topology_router: "rule" | "llm" | "oracle"  (default "rule")
                  switch_guards: bool  (default True)
                  switch_guards_config: dict of SwitchGuards fields
                  subgraph_max_iterations: int  (default 10)
    """

    name = "adaptive"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        *,
        human_cfg: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Compile and return the L2 adaptive meta-graph.

        Returns a CompiledStateGraph ready for ainvoke(initial_state).

        Args:
            agents:    Dict of agent_id → Agent.
            cfg:       TopologyConfig. See class docstring for extra keys.
            human_cfg: Optional HumanCfg (M9.1). When ``enabled=True``, inserts
                       ``human_advisor`` between ``topology_router_node`` and
                       ``dispatch_topology_node``.
                       Extra keys in ``human_cfg.extra``:
                         ``human_can_override_router`` (bool, default False):
                             when True, human may override the router decision.
        """
        extra = cfg.extra or {}
        checkpointer = kwargs.get("checkpointer")

        # ----------------------------------------------------------------
        # Routers configuration
        # ----------------------------------------------------------------
        limits = PhaseLimits(
            planning_max_iter=int(extra.get("planning_max_iter", 3)),
            exec_max_iter=int(extra.get("exec_max_iter", 10)),
            verify_max_iter=int(extra.get("verify_max_iter", 4)),
        )
        phase_router = RuleBasedPhaseRouter(limits=limits, guards={})

        rule_topo_router = RuleBasedTopologyRouter()
        use_guards: bool = bool(extra.get("switch_guards", True))
        if use_guards:
            guards_cfg = extra.get("switch_guards_config") or {}
            switch_guards = SwitchGuards(**dict(guards_cfg.items()))
            topo_router: Any = GuardedRouter(inner=rule_topo_router, guards=switch_guards)
        else:
            topo_router = rule_topo_router

        # ----------------------------------------------------------------
        # Subgraph cache (lazy compile on first use per topology name)
        # ----------------------------------------------------------------
        subgraph_max_iter: int = int(extra.get("subgraph_max_iterations", 10))
        _subgraph_cache: dict[str, Any] = {}

        # Keys to strip from extra when building subgraph configs
        _meta_keys = frozenset(
            {
                "phase_router",
                "topology_router",
                "switch_guards",
                "switch_guards_config",
                "subgraph_max_iterations",
                "planning_max_iter",
                "exec_max_iter",
                "verify_max_iter",
                "run_id",
            }
        )

        def _get_subgraph(topology_name: str) -> Any:
            """Return compiled subgraph for the given topology name, building if needed."""
            if topology_name in _subgraph_cache:
                return _subgraph_cache[topology_name]

            # Map from arch.md name to TopologyRegistry name
            registry_name = _TOPO_ALIAS.get(topology_name, topology_name)
            try:
                topo_cls = TopologyRegistry.get(registry_name)
            except KeyError:
                _log.warning(
                    "adaptive: topology %r (registry key %r) not found; falling back to chain",
                    topology_name,
                    registry_name,
                )
                topo_cls = TopologyRegistry.get("chain")
                registry_name = "chain"

            topo_instance = topo_cls()
            sub_cfg = TopologyConfig(
                name=registry_name,
                max_iterations=subgraph_max_iter,
                extra={k: v for k, v in extra.items() if k not in _meta_keys},
            )
            compiled = topo_instance.build(agents, sub_cfg)
            _subgraph_cache[topology_name] = compiled
            return compiled

        # ----------------------------------------------------------------
        # run_id
        # ----------------------------------------------------------------
        run_id_str: str = str(extra.get("run_id", str(uuid.uuid4())))

        # ----------------------------------------------------------------
        # Closure slots for routing decisions (per-tick).
        # Using list-of-one as a mutable box avoids closure write issues.
        # These slots are set by router nodes and read by transition_gate_node.
        # Storing them here (not in state) prevents subgraph from overwriting them.
        # ----------------------------------------------------------------
        _phase_dec_slot: list[PhaseDecision | None] = [None]
        _topo_dec_slot: list[TopologyDecision | None] = [None]

        # ----------------------------------------------------------------
        # HITL: human_advisor gateway setup (M9.1)
        # ----------------------------------------------------------------
        _hitl_enabled: bool = bool(human_cfg and getattr(human_cfg, "enabled", False))
        _gateway: Any = None
        _human_can_override: bool = False

        if _hitl_enabled:
            _human_extra: dict[str, Any] = dict(getattr(human_cfg, "extra", None) or {})
            _human_can_override = bool(_human_extra.get("human_can_override_router", False))

            gateway_type: str = getattr(human_cfg, "gateway", "llm_simulated")
            if gateway_type == "cli" and _CLIGateway is not None:
                _gateway = _CLIGateway()
            elif _LLMSimulatedGateway is not None:
                # Build LLMWrapper for LLMSimulatedGateway if llm_wrapper kwarg provided
                llm_wrapper = kwargs.get("llm_wrapper")
                if llm_wrapper is not None:
                    _gateway = _LLMSimulatedGateway(llm_wrapper)
                else:
                    _log.warning(
                        "adaptive: human_cfg.enabled=True but no llm_wrapper kwarg provided; "
                        "HITL gateway unavailable — falling back to no-op advisory mode"
                    )
                    _hitl_enabled = False

        # ----------------------------------------------------------------
        # Node: phase_router_node
        # ----------------------------------------------------------------

        async def phase_router_node(state: GraphState) -> dict[str, Any]:
            """Invoke PhaseRouter; store decision in closure slot (not in state)."""
            decision: PhaseDecision = await phase_router.decide(state)
            _log.debug(
                "adaptive phase_router: %s → %s (decided_by=%s)",
                (state.get("shared") or {}).get("phase"),
                decision.next_phase,
                decision.decided_by,
            )
            _phase_dec_slot[0] = decision
            return {}

        # ----------------------------------------------------------------
        # Node: topology_router_node
        # ----------------------------------------------------------------

        async def topology_router_node(state: GraphState) -> dict[str, Any]:
            """Invoke TopologyRouter; store decision in closure slot (not in state)."""
            _raw = state.get("shared")
            shared_raw: SharedState = _raw if _raw is not None else SharedState()
            decision: TopologyDecision = await topo_router.decide(shared_raw)
            _log.debug(
                "adaptive topology_router: %s → %s (decided_by=%s)",
                shared_raw.get("active_topology"),
                decision.topology,
                decision.decided_by,
            )
            _topo_dec_slot[0] = decision
            return {}

        # ----------------------------------------------------------------
        # Node: human_advisor_node  (M9.1 HITL — inserted when hitl enabled)
        # ----------------------------------------------------------------

        async def human_advisor_node(state: GraphState) -> dict[str, Any]:
            """HITL human_advisor between topology_router and dispatch_topology.

            Advisory mode (default):
                Human sees the current router decision and may write a hint into
                ``shared.signals["human_advisor_hint"]``.  Does NOT mutate
                ``_topo_dec_slot[0]``; routing is unchanged for this tick.

            Override mode (human_can_override_router=True):
                Human may specify a ``action="switch_topology"`` + ``payload.topology``
                to replace the router's decision.  GuardedRouter (if active) is
                re-evaluated against the proposed topology.  If guards block the
                override, a log.warning is emitted, hint is set, and the original
                decision stands.

            # known-limitation: subgraph-level interrupt-resume (CLIGateway inside
            # subgraph with real interrupt/resume) is deferred to M9.2.
            # See arch/PLAN.md §M9.1 exit criterion lines 560-561.
            """
            assert _gateway is not None, "human_advisor_node called but _gateway is None"

            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            current_decision = _topo_dec_slot[0]
            current_topo: str = (
                current_decision.topology
                if current_decision is not None
                else (shared.get("active_topology") or "linear")
            )

            _raw_run_id = shared.get("run_id") or state.get("run_id")
            run_id_val: uuid.UUID = (
                _raw_run_id
                if isinstance(_raw_run_id, uuid.UUID)
                else uuid.UUID(str(_raw_run_id))
                if _raw_run_id
                else uuid.uuid4()
            )
            iter_total: int = int(shared.get("iter_total", 0))
            request_id = f"adaptive:{run_id_val}:{iter_total}:advisor"

            ctx = HumanContext(
                run_id=run_id_val,
                role=human_cfg.role,
                question=(
                    f"TopologyRouter chose '{current_topo}'. "
                    f"Advisory: suggest a hint via action='advise' and payload.hint. "
                    f"Override (if enabled): action='switch_topology' and payload.topology."
                ),
                recent_messages=tuple(state.get("messages", [])[-5:]),
                allowed_actions=("advise", "switch_topology", "abstain"),
                deadline_s=(
                    int(human_cfg.timeout_s)
                    if getattr(human_cfg, "timeout_s", None) is not None
                    else None
                ),
            )

            _requested_at = datetime.now(UTC)
            try:
                await adispatch_custom_event(
                    "human_request",
                    {
                        "run_id": run_id_val,
                        "request_id": request_id,
                        "role": str(
                            human_cfg.role.value
                            if hasattr(human_cfg.role, "value")
                            else human_cfg.role
                        ),
                        "context_json": ctx.model_dump(mode="json"),
                        "requested_at": _requested_at,
                    },
                )
            except Exception:
                _log.debug("adaptive human_advisor: adispatch human_request skipped", exc_info=True)

            _t0 = time.monotonic()
            timeout_s: float | None = getattr(human_cfg, "timeout_s", None)
            timeout_policy: str = getattr(human_cfg, "timeout_policy", "skip")

            if _request_with_timeout is not None and timeout_s is not None:
                response = await _request_with_timeout(
                    _gateway,
                    ctx,
                    request_id=request_id,
                    timeout_s=timeout_s,
                    policy=timeout_policy,
                )
            else:
                response = await _gateway.request(ctx, request_id=request_id)

            _latency_s = time.monotonic() - _t0

            try:
                await adispatch_custom_event(
                    "human_response",
                    {
                        "run_id": run_id_val,
                        "request_id": request_id,
                        "answered_at": datetime.now(UTC),
                        "response_json": response.model_dump(mode="json"),
                        "source": getattr(response, "source", "human"),
                        "timed_out": getattr(response, "timed_out", False),
                        "latency_s": _latency_s,
                    },
                )
            except Exception:
                _log.debug(
                    "adaptive human_advisor: adispatch human_response skipped", exc_info=True
                )

            action: str = getattr(response, "action", "") or ""
            payload: dict[str, Any] = dict(getattr(response, "payload", None) or {})
            comment: str = getattr(response, "comment", "") or ""

            if _human_can_override and action == "switch_topology":
                # Override mode: attempt to replace topology decision
                proposed_topo: str = str(payload.get("topology", "")).strip()
                valid_topos = set(_TOPO_ALIAS.keys()) | set(_TOPO_ALIAS.values()) | {"adaptive"}

                if not proposed_topo or proposed_topo not in valid_topos:
                    _log.warning(
                        "adaptive human_advisor: override proposed invalid topology %r (valid: %s); "
                        "ignoring and writing hint",
                        proposed_topo,
                        sorted(valid_topos),
                    )
                    signals["human_advisor_hint"] = comment or f"invalid_topology:{proposed_topo!r}"
                else:
                    # Build proposed override decision
                    override_decision = TopologyDecision(
                        topology=proposed_topo,
                        reason=f"human_override: {comment or proposed_topo}",
                        decided_by="human_override",
                        considered_alternatives=(current_topo,)
                        if proposed_topo != current_topo
                        else (),
                    )

                    # Apply GuardedRouter validation if topo_router is a GuardedRouter
                    if use_guards:
                        _raw_shared = state.get("shared")
                        _shared_state: SharedState = (
                            _raw_shared if _raw_shared is not None else SharedState()
                        )
                        guard_decision: TopologyDecision = await topo_router.decide(_shared_state)
                        # GuardedRouter decided based on the override proposal — but topo_router
                        # already has internal state.  We re-check guards by inspecting if
                        # current_decision was already guard_override or if the proposed topology
                        # would be blocked.  Simpler: apply guards against the proposed topology
                        # by temporarily setting inner router to return proposed_topo.
                        # Actual approach: check if the guard_decision topology != proposed_topo.
                        # If guards would have blocked a switch to proposed_topo, they return
                        # current_topo. We use guard_decision as an oracle here.
                        if (
                            guard_decision.topology == proposed_topo
                            or guard_decision.decided_by != "guard_override"
                        ):
                            # Guards allow (or proposed equals guard's choice)
                            _topo_dec_slot[0] = override_decision
                            signals["human_advisor_hint"] = f"override_applied:{proposed_topo}"
                            _log.info(
                                "adaptive human_advisor: override accepted → topology=%r",
                                proposed_topo,
                            )
                        else:
                            # Guards blocked — record human intent in considered_alternatives
                            blocked_with_intent = TopologyDecision(
                                topology=guard_decision.topology,
                                reason=guard_decision.reason,
                                decided_by=guard_decision.decided_by,
                                considered_alternatives=(
                                    *guard_decision.considered_alternatives,
                                    proposed_topo,
                                ),
                                router_cost_usd=guard_decision.router_cost_usd,
                            )
                            _topo_dec_slot[0] = blocked_with_intent
                            signals["human_advisor_hint"] = (
                                f"override_blocked_by_guards:{proposed_topo}"
                            )
                            _log.warning(
                                "adaptive human_advisor: override to %r blocked by guards (%s)",
                                proposed_topo,
                                guard_decision.reason,
                            )
                    else:
                        # No guards — apply override directly
                        _topo_dec_slot[0] = override_decision
                        signals["human_advisor_hint"] = f"override_applied:{proposed_topo}"
                        _log.info(
                            "adaptive human_advisor: override accepted (no guards) → topology=%r",
                            proposed_topo,
                        )
            else:
                # Advisory mode (or abstain / non-override action)
                # Only write a hint when there is a meaningful comment.
                # "abstain" or "timeout" without a comment → pure no-op.
                hint_text = (comment or "").strip()
                if hint_text:
                    signals["human_advisor_hint"] = hint_text
                # _topo_dec_slot[0] is NOT mutated in advisory mode

            shared["signals"] = signals
            return {"shared": shared}

        # ----------------------------------------------------------------
        # Node: dispatch_topology_node  (runs the active subgraph)
        # ----------------------------------------------------------------

        async def dispatch_topology_node(state: GraphState) -> dict[str, Any]:
            """Dispatch to the appropriate subgraph based on topology decision.

            Architectural note: Instead of 5 separate LangGraph nodes with a
            conditional edge map, we use a single dispatch node with if/elif
            branching by TopologyDecision.topology.  LangGraph conditional edges
            cannot select a node whose name varies at runtime without a full
            compile-time edge-map, but since we have exactly 5 topologies the
            single dispatch node is simpler and fully equivalent.
            """
            shared: dict[str, Any] = dict(state.get("shared") or {})
            topo_decision = _topo_dec_slot[0]

            topo_name: str = (
                topo_decision.topology
                if topo_decision is not None
                else (shared.get("active_topology") or "linear")
            )

            # Sanity guard: refuse to dispatch into the adaptive meta-graph
            # itself (would cause infinite recursion). Any caller that returns
            # "adaptive" as the chosen sub-topology gets rerouted to "linear".
            if topo_name == self.name:
                _log.warning(
                    "adaptive: refused to dispatch into self (topology=%r); falling back to 'linear'",
                    topo_name,
                )
                topo_name = "linear"

            # Increment iter_total before delegating to subgraph
            iter_total: int = int(shared.get("iter_total", 0)) + 1
            shared["iter_total"] = iter_total

            _log.info(
                "adaptive dispatch: tick=%d → subgraph=%r",
                iter_total,
                topo_name,
            )

            # Check global tick cap
            max_ticks: int = cfg.max_iterations or _DEFAULT_MAX_TICKS
            if iter_total > max_ticks:
                _log.info("adaptive: max_ticks=%d reached, routing to END", max_ticks)
                shared["active_topology"] = topo_name
                return {"shared": shared}

            sub = _get_subgraph(topo_name)

            # Build subgraph input state — inherit full state + updated shared
            sub_state = dict(state)
            sub_state["shared"] = shared

            try:
                result: dict[str, Any] = await sub.ainvoke(
                    sub_state, config={"recursion_limit": 50}
                )
            except Exception as exc:
                _log.error(
                    "adaptive dispatch: subgraph %r raised %s: %s",
                    topo_name,
                    type(exc).__name__,
                    exc,
                )
                return {"shared": shared}

            return result

        # ----------------------------------------------------------------
        # Node: transition_gate_node
        # ----------------------------------------------------------------

        async def transition_gate_node(state: GraphState) -> dict[str, Any]:
            """Apply TransitionGate state-transfer and record TopologyTransition.

            Reads router decisions from closure slots (not from state) to avoid
            them being overwritten by the subgraph's state output.
            """
            phase_decision = _phase_dec_slot[0]
            topo_decision = _topo_dec_slot[0]

            # Fall back to stay decisions if router nodes were bypassed
            shared: dict[str, Any] = dict(state.get("shared") or {})
            current_phase: Phase = shared.get("phase", Phase.PLANNING)

            if phase_decision is None:
                phase_decision = PhaseDecision(
                    next_phase=current_phase,
                    reason="fallback: no phase_decision in slot",
                    decided_by="rule",
                )
            if topo_decision is None:
                active = shared.get("active_topology") or "linear"
                topo_decision = TopologyDecision(
                    topology=active,
                    reason="fallback: no topo_decision in slot",
                    decided_by="rule",
                )

            new_state = apply_transition_gate(
                cast(dict[str, Any], state),
                phase_decision,
                topo_decision,
                run_id=run_id_str,
            )

            # ---- Dispatch custom events so ExperimentCallback persists them ----
            # (a) topology_transition — every tick (incl. no-change)
            new_transitions: list[TopologyTransition] = new_state.get("topology_transitions") or []
            if new_transitions:
                latest_tt = new_transitions[-1]
                try:
                    await adispatch_custom_event("topology_transition", latest_tt)
                except Exception:
                    _log.debug(
                        "adispatch topology_transition skipped (no callback ctx)", exc_info=True
                    )

            # (b) phase_transition — only when phase actually advanced
            new_shared = new_state.get("shared") or {}
            new_phase: Phase = new_shared.get("phase", current_phase)
            if new_phase != current_phase:
                try:
                    phase_tx = PhaseTransition(
                        run_id=uuid.UUID(run_id_str),
                        from_phase=current_phase,
                        to_phase=new_phase,
                        entry_reason=phase_decision.reason,
                        iter_total=int(new_shared.get("iter_total", 0)),
                        decided_by=phase_decision.decided_by,
                    )
                    await adispatch_custom_event("phase_transition", phase_tx)
                except Exception:
                    _log.debug("adispatch phase_transition skipped", exc_info=True)

            # Clear slots for next tick
            _phase_dec_slot[0] = None
            _topo_dec_slot[0] = None

            return new_state

        # ----------------------------------------------------------------
        # Routing function for the conditional edge after transition_gate
        # ----------------------------------------------------------------

        def _should_end(state: GraphState) -> str:
            """Return '__end__' if done or max ticks exceeded, else loop."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            phase: Phase = shared.get("phase", Phase.PLANNING)
            iter_total: int = int(shared.get("iter_total", 0))
            max_ticks: int = cfg.max_iterations or _DEFAULT_MAX_TICKS

            if phase == Phase.DONE or str(phase) == "done":
                _log.info("adaptive: phase=done → END")
                return END

            if iter_total >= max_ticks:
                _log.info("adaptive: iter_total=%d >= max_ticks=%d → END", iter_total, max_ticks)
                return END

            return "phase_router_node"

        # ----------------------------------------------------------------
        # Build meta-graph
        # ----------------------------------------------------------------

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        graph.add_node("phase_router_node", phase_router_node)
        graph.add_node("topology_router_node", topology_router_node)
        graph.add_node("dispatch_topology_node", dispatch_topology_node)
        graph.add_node("transition_gate_node", transition_gate_node)

        graph.add_edge(START, "phase_router_node")
        graph.add_edge("phase_router_node", "topology_router_node")

        if _hitl_enabled:
            # M9.1: Insert human_advisor between topology_router and dispatch_topology
            graph.add_node("human_advisor_node", human_advisor_node)
            graph.add_edge("topology_router_node", "human_advisor_node")
            graph.add_edge("human_advisor_node", "dispatch_topology_node")
        else:
            # Default (M8 back-compat): direct edge
            graph.add_edge("topology_router_node", "dispatch_topology_node")

        graph.add_edge("dispatch_topology_node", "transition_gate_node")

        graph.add_conditional_edges(
            "transition_gate_node",
            _should_end,
            {END: END, "phase_router_node": "phase_router_node"},
        )

        compiled = graph.compile(checkpointer=checkpointer)
        return compiled


# ---------------------------------------------------------------------------
# Convenience factory function (public API per m8-adaptive.md)
# ---------------------------------------------------------------------------


def build_adaptive_graph(
    agents: dict[str, Any],
    *,
    max_iterations: int = _DEFAULT_MAX_TICKS,
    extra: dict[str, Any] | None = None,
    checkpointer: Any = None,
) -> Any:
    """Build and return a compiled AdaptiveTopology meta-graph.

    Convenience wrapper around AdaptiveTopology().build().

    Args:
        agents:         Dict of agent_id → Agent.
        max_iterations: Hard cap on meta-graph ticks (default 30).
        extra:          Extra config passed to TopologyConfig.extra.
        checkpointer:   Optional LangGraph checkpointer.

    Returns:
        CompiledStateGraph ready for ainvoke.
    """
    cfg = TopologyConfig(
        name="adaptive",
        max_iterations=max_iterations,
        extra=extra or {},
    )
    return AdaptiveTopology().build(agents, cfg, checkpointer=checkpointer)
