"""AdaptiveTopology — L2 meta-graph with runtime topology switching."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
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
from atm.phases.guards import (
    GuardedRouter,
    SwitchGuards,
    _violates_cooldown,
    _violates_max_per_phase,
    _violates_max_per_run,
    _violates_min_dwell,
)
from atm.phases.manager import PhaseLimits, RuleBasedPhaseRouter
from atm.phases.topology_router import (
    LLMTopologyRouter,
    OracleTopologyRouter,
    RuleBasedTopologyRouter,
)
from atm.topology.base import TopologyConfig, TopologyRegistry, get_topology_extras

_log = logging.getLogger(__name__)

_LLMSimulatedGateway: Any
_CLIGateway: Any
_request_with_timeout: Any
_HumanRoleRouter: Any

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

try:
    from atm.human.role_router import HumanRoleRouter as _HumanRoleRouter
except ImportError:  # pragma: no cover
    _HumanRoleRouter = None

_DEFAULT_MAX_TICKS: int = 30
_HISTORY_TAIL: int = 10
_TOPO_ALIAS: dict[str, str] = {
    "linear": "chain",
    "supervisor": "star",
    "mesh": "mesh",
    "debate": "debate",
    "hierarchical": "hierarchical",
}


def apply_transition_gate(
    state: dict[str, Any],
    phase_decision: PhaseDecision,
    topology_decision: TopologyDecision,
    *,
    run_id: str | None = None,
    pre_subgraph_phase: Phase | None = None,
) -> dict[str, Any]:
    """Apply state-transfer rules and record a TopologyTransition; returns new state."""
    _phase_order: dict[Phase, int] = {
        Phase.PLANNING: 0,
        Phase.EXECUTION: 1,
        Phase.VERIFICATION: 2,
        Phase.DONE: 3,
    }

    new_state: dict[str, Any] = dict(state)
    shared: dict[str, Any] = dict(state.get("shared") or {})

    subgraph_phase: Phase = shared.get("phase", Phase.PLANNING)
    router_phase: Phase = phase_decision.next_phase
    next_phase: Phase = (
        router_phase
        if _phase_order.get(router_phase, 0) > _phase_order.get(subgraph_phase, 0)
        else subgraph_phase
    )

    current_phase: Phase = pre_subgraph_phase if pre_subgraph_phase is not None else subgraph_phase

    current_topology: str | None = shared.get("active_topology")
    next_topology: str = topology_decision.topology

    iter_total: int = int(shared.get("iter_total", 0))
    phase_started_at: int = int(shared.get("phase_started_at_iter", 0))
    topo_started_at: int = int(shared.get("topology_started_at_iter", 0))

    phase_changed: bool = next_phase != current_phase
    topology_changed: bool = (current_topology is not None) and (next_topology != current_topology)

    if phase_changed:
        shared["phase"] = next_phase
        shared["phase_started_at_iter"] = iter_total
        phase_started_at = iter_total
        _signals_for_phase_reset: dict[str, Any] = dict(shared.get("signals") or {})
        _signals_for_phase_reset["phase_switch_count"] = 0
        shared["signals"] = _signals_for_phase_reset

    if topology_changed or current_topology is None:
        shared["active_topology"] = next_topology
        shared["topology_started_at_iter"] = iter_total
        topo_started_at = iter_total

        if topology_changed:
            switch_count: int = int(shared.get("topology_switch_count", 0))
            shared["topology_switch_count"] = switch_count + 1

            _signals_with_phase_count: dict[str, Any] = dict(shared.get("signals") or {})
            _prev_phase_count: int = int(_signals_with_phase_count.get("phase_switch_count", 0))
            _signals_with_phase_count["phase_switch_count"] = _prev_phase_count + 1
            shared["signals"] = _signals_with_phase_count

            history: list[str] = list(shared.get("topology_history") or [])
            if current_topology is not None:
                history.append(current_topology)
            shared["topology_history"] = history[-_HISTORY_TAIL:]

    if phase_changed or topology_changed:
        shared["iteration"] = 0
    else:
        shared["iteration"] = int(shared.get("iteration", 0)) + 1

    if phase_changed or topology_changed:
        shared["broadcast_bus"] = []

    _phase_advance_signal: dict[Phase, str] = {
        Phase.PLANNING: "ready_for_execution",
        Phase.EXECUTION: "ready_for_verification",
        Phase.VERIFICATION: "critic_approved",
    }
    if phase_changed:
        signals_dict: dict[str, Any] = dict(shared.get("signals") or {})
        prev_phase_signal = _phase_advance_signal.get(current_phase)
        if prev_phase_signal and prev_phase_signal in signals_dict:
            del signals_dict[prev_phase_signal]
        shared["signals"] = signals_dict
    elif topology_changed:
        signals_dict = dict(shared.get("signals") or {})
        reason_lower = topology_decision.reason.lower()
        consumed_keys = [k for k in signals_dict if k in reason_lower]
        for key in consumed_keys:
            del signals_dict[key]
        shared["signals"] = signals_dict

    if phase_changed:
        agents: dict[str, Any] = dict(state.get("agents") or {})
        cleaned_agents: dict[str, Any] = {}
        for agent_id, agent_state in agents.items():
            a = dict(agent_state)
            a["inbox"] = []
            a["outbox"] = []
            cleaned_agents[agent_id] = a
        new_state["agents"] = cleaned_agents

    _hint_signals: dict[str, Any] = dict(shared.get("signals") or {})
    if "human_advisor_hint" in _hint_signals:
        del _hint_signals["human_advisor_hint"]
        shared["signals"] = _hint_signals

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


@TopologyRegistry.register("adaptive")
class AdaptiveTopology:
    """L2 adaptive meta-graph: phase_router → topo_router → dispatch_subgraph → transition_gate."""

    name = "adaptive"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        *,
        human_cfg: Any = None,
        role_router: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Compile and return the L2 adaptive meta-graph."""
        extras = get_topology_extras(cfg, "adaptive")
        checkpointer = kwargs.get("checkpointer")

        limits = PhaseLimits(
            planning_max_iter=int(extras.get("planning_max_iter", 3)),
            exec_max_iter=int(extras.get("exec_max_iter", 10)),
            verify_max_iter=int(extras.get("verify_max_iter", 4)),
        )
        phase_router = RuleBasedPhaseRouter(limits=limits, guards={})

        rule_topo_router = RuleBasedTopologyRouter()

        topo_router_mode: str = str(extras.get("topology_router", "rule")).lower()
        inner_topo_router: Any
        if topo_router_mode == "llm":
            router_llm: Any = kwargs.get("topology_router_llm")
            if router_llm is None:
                _log.warning(
                    "adaptive: topology_router='llm' but no topology_router_llm "
                    "kwarg provided; falling back to rule"
                )
                inner_topo_router = rule_topo_router
            else:
                inner_topo_router = LLMTopologyRouter(
                    llm=router_llm, rule_fallback=rule_topo_router
                )
        elif topo_router_mode == "oracle":
            oracle_path_str: str = str(
                extras.get("oracle_table_path") or "data/oracle/e1_leave_one_out.json"
            )
            oracle_path = Path(oracle_path_str)
            if not oracle_path.exists():
                _log.warning(
                    "adaptive: topology_router='oracle' but oracle table %s missing; "
                    "falling back to rule",
                    oracle_path,
                )
                inner_topo_router = rule_topo_router
            else:
                try:
                    inner_topo_router = OracleTopologyRouter(oracle_path)
                except Exception as exc:  # pragma: no cover — defensive
                    _log.warning(
                        "adaptive: OracleTopologyRouter init failed (%s); falling back to rule",
                        exc,
                    )
                    inner_topo_router = rule_topo_router
        else:
            inner_topo_router = rule_topo_router

        use_guards: bool = bool(extras.get("switch_guards", True))
        if use_guards:
            guards_cfg = extras.get("switch_guards_config") or {}
            switch_guards = SwitchGuards(**dict(guards_cfg.items()))
            topo_router: Any = GuardedRouter(inner=inner_topo_router, guards=switch_guards)
        else:
            topo_router = inner_topo_router

        subgraph_max_iter: int = int(extras.get("subgraph_max_iterations", 10))
        _subgraph_cache: dict[str, Any] = {}

        def _get_subgraph(topology_name: str) -> Any:
            """Return compiled subgraph for the given topology name, building if needed."""
            if topology_name in _subgraph_cache:
                return _subgraph_cache[topology_name]

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
                extra=dict(cfg.extra),
            )
            compiled = topo_instance.build(agents, sub_cfg)
            _subgraph_cache[topology_name] = compiled
            return compiled

        run_id_str: str = str(extras.get("run_id", str(uuid.uuid4())))

        _phase_dec_slot: list[PhaseDecision | None] = [None]
        _topo_dec_slot: list[TopologyDecision | None] = [None]
        _pre_subgraph_phase_slot: list[Phase | None] = [None]

        _hitl_enabled: bool = bool(human_cfg and getattr(human_cfg, "enabled", False))
        _gateway: Any = None
        _human_can_override: bool = False
        _role_router: Any = role_router

        if _hitl_enabled:
            _human_extra: dict[str, Any] = dict(getattr(human_cfg, "extra", None) or {})
            _human_can_override = bool(_human_extra.get("human_can_override_router", False))

            gateway_type: str = getattr(human_cfg, "gateway", "llm_simulated")
            if gateway_type == "cli" and _CLIGateway is not None:
                _gateway = _CLIGateway()
            elif _LLMSimulatedGateway is not None:
                llm_wrapper = kwargs.get("human_gateway_llm")
                if llm_wrapper is not None:
                    _gateway = _LLMSimulatedGateway(llm_wrapper)
                else:
                    _log.warning(
                        "adaptive: human_cfg.enabled=True but no human_gateway_llm kwarg provided; "
                        "HITL gateway unavailable — falling back to no-op advisory mode"
                    )
                    _hitl_enabled = False

        async def phase_router_node(state: GraphState) -> dict[str, Any]:
            """Invoke PhaseRouter; capture pre-dispatch phase and decision in slots."""
            _shared_now: dict[str, Any] = dict(state.get("shared") or {})
            _raw_phase = _shared_now.get("phase", Phase.PLANNING)
            _pre_phase: Phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            _pre_subgraph_phase_slot[0] = _pre_phase

            decision: PhaseDecision = await phase_router.decide(state)
            _log.debug(
                "adaptive phase_router: %s → %s (decided_by=%s)",
                _shared_now.get("phase"),
                decision.next_phase,
                decision.decided_by,
            )
            _phase_dec_slot[0] = decision
            return {}

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

        async def human_advisor_node(state: GraphState) -> dict[str, Any]:
            """HITL advisor: advisory or override mode for topology routing."""
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

            if _role_router is not None:
                _raw_phase = shared.get("phase", "planning")
                _phase_val: Phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
                active_role = await _role_router.decide(_phase_val, shared)
            else:
                active_role = human_cfg.role

            ctx = HumanContext(
                run_id=run_id_val,
                role=active_role,
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
                            active_role.value if hasattr(active_role, "value") else active_role
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
                _fallback_gateway: Any = None
                if timeout_policy == "llm_fallback" and _LLMSimulatedGateway is not None:
                    _fb_llm: Any = getattr(_gateway, "_llm", None)
                    _fallback_gateway = _LLMSimulatedGateway(llm=_fb_llm)
                response = await _request_with_timeout(
                    _gateway,
                    ctx,
                    request_id=request_id,
                    timeout_s=timeout_s,
                    policy=timeout_policy,
                    llm_fallback_gateway=_fallback_gateway,
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
                    override_decision = TopologyDecision(
                        topology=proposed_topo,
                        reason=f"human_override: {comment or proposed_topo}",
                        decided_by="human_override",
                        considered_alternatives=(current_topo,)
                        if proposed_topo != current_topo
                        else (),
                    )

                    if use_guards:
                        _raw_shared = state.get("shared")
                        _shared_state: SharedState = (
                            _raw_shared if _raw_shared is not None else SharedState()
                        )
                        _guards_cfg: SwitchGuards = (
                            topo_router._guards
                            if isinstance(topo_router, GuardedRouter)
                            else SwitchGuards()
                        )
                        _applied: list[str] = []
                        if proposed_topo != current_topo:
                            if _violates_min_dwell(_shared_state, _guards_cfg):
                                _applied.append("min_dwell")
                            if _violates_cooldown(_shared_state, proposed_topo, _guards_cfg):
                                _applied.append("cooldown")
                            if _violates_max_per_run(_shared_state, _guards_cfg):
                                _applied.append("max_per_run")
                            if _violates_max_per_phase(_shared_state, _guards_cfg):
                                _applied.append("max_per_phase")

                        if not _applied:
                            _topo_dec_slot[0] = override_decision
                            signals["human_advisor_hint"] = f"override_applied:{proposed_topo}"
                            _log.info(
                                "adaptive human_advisor: override accepted → topology=%r",
                                proposed_topo,
                            )
                        else:
                            _current_dec = _topo_dec_slot[0]
                            blocked_with_intent = TopologyDecision(
                                topology=current_topo,
                                reason=f"guards={_applied}: keep '{current_topo}'",
                                decided_by="guard_override",
                                considered_alternatives=(
                                    *(_current_dec.considered_alternatives if _current_dec else ()),
                                    proposed_topo,
                                ),
                            )
                            _topo_dec_slot[0] = blocked_with_intent
                            signals["human_advisor_hint"] = (
                                f"override_blocked_by_guards:{proposed_topo}"
                            )
                            _log.warning(
                                "adaptive human_advisor: override to %r blocked by guards %s",
                                proposed_topo,
                                _applied,
                            )
                    else:
                        _topo_dec_slot[0] = override_decision
                        signals["human_advisor_hint"] = f"override_applied:{proposed_topo}"
                        _log.info(
                            "adaptive human_advisor: override accepted (no guards) → topology=%r",
                            proposed_topo,
                        )
            else:
                hint_text = (comment or "").strip()
                if hint_text:
                    signals["human_advisor_hint"] = hint_text

            shared["signals"] = signals
            return {"shared": shared}

        async def dispatch_topology_node(state: GraphState) -> dict[str, Any]:
            """Dispatch to and invoke the active sub-topology subgraph."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            topo_decision = _topo_dec_slot[0]

            topo_name: str = (
                topo_decision.topology
                if topo_decision is not None
                else (shared.get("active_topology") or "linear")
            )

            if topo_name == self.name:
                _log.warning(
                    "adaptive: refused to dispatch into self (topology=%r); falling back to 'linear'",
                    topo_name,
                )
                topo_name = "linear"

            meta_ticks: int = int(shared.get("meta_ticks", 0)) + 1
            shared["meta_ticks"] = meta_ticks

            iter_total: int = int(shared.get("iter_total", 0))

            _log.info(
                "adaptive dispatch: meta_tick=%d iter_total=%d → subgraph=%r",
                meta_ticks,
                iter_total,
                topo_name,
            )

            if meta_ticks > _DEFAULT_MAX_TICKS:
                _log.info(
                    "adaptive: meta_ticks=%d > %d safety cap, routing to END",
                    meta_ticks,
                    _DEFAULT_MAX_TICKS,
                )
                shared["active_topology"] = topo_name
                return {"shared": shared}

            max_iter: int = cfg.max_iterations or _DEFAULT_MAX_TICKS
            if iter_total >= max_iter:
                _log.info(
                    "adaptive: iter_total=%d >= max_iterations=%d, routing to END",
                    iter_total,
                    max_iter,
                )
                shared["active_topology"] = topo_name
                return {"shared": shared}

            sub = _get_subgraph(topo_name)

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

        async def transition_gate_node(state: GraphState) -> dict[str, Any]:
            """Apply TransitionGate state-transfer and record TopologyTransition."""
            phase_decision = _phase_dec_slot[0]
            topo_decision = _topo_dec_slot[0]

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
                pre_subgraph_phase=_pre_subgraph_phase_slot[0],
            )

            new_transitions: list[TopologyTransition] = new_state.get("topology_transitions") or []
            if new_transitions:
                latest_tt = new_transitions[-1]
                try:
                    await adispatch_custom_event("topology_transition", latest_tt)
                except Exception:
                    _log.debug(
                        "adispatch topology_transition skipped (no callback ctx)", exc_info=True
                    )

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

            _phase_dec_slot[0] = None
            _topo_dec_slot[0] = None
            _pre_subgraph_phase_slot[0] = None

            return new_state

        def _should_end(state: GraphState) -> str:
            """Return END or 'phase_router_node' based on stopping conditions."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            phase: Phase = shared.get("phase", Phase.PLANNING)
            iter_total: int = int(shared.get("iter_total", 0))
            meta_ticks: int = int(shared.get("meta_ticks", 0))
            max_iter: int = cfg.max_iterations or _DEFAULT_MAX_TICKS

            if phase == Phase.DONE or str(phase) == "done":
                _log.info("adaptive: phase=done → END")
                return END

            if iter_total >= max_iter:
                _log.info(
                    "adaptive: iter_total=%d >= max_iterations=%d → END",
                    iter_total,
                    max_iter,
                )
                return END

            if meta_ticks >= _DEFAULT_MAX_TICKS:
                _log.info(
                    "adaptive: meta_ticks=%d >= safety cap %d → END",
                    meta_ticks,
                    _DEFAULT_MAX_TICKS,
                )
                return END

            return "phase_router_node"

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        graph.add_node("phase_router_node", phase_router_node)
        graph.add_node("topology_router_node", topology_router_node)
        graph.add_node("dispatch_topology_node", dispatch_topology_node)
        graph.add_node("transition_gate_node", transition_gate_node)

        graph.add_edge(START, "phase_router_node")
        graph.add_edge("phase_router_node", "topology_router_node")

        if _hitl_enabled:
            graph.add_node("human_advisor_node", human_advisor_node)
            graph.add_edge("topology_router_node", "human_advisor_node")
            graph.add_edge("human_advisor_node", "dispatch_topology_node")
        else:
            graph.add_edge("topology_router_node", "dispatch_topology_node")

        graph.add_edge("dispatch_topology_node", "transition_gate_node")

        graph.add_conditional_edges(
            "transition_gate_node",
            _should_end,
            {END: END, "phase_router_node": "phase_router_node"},
        )

        compiled = graph.compile(checkpointer=checkpointer)
        return compiled


def build_adaptive_graph(
    agents: dict[str, Any],
    *,
    max_iterations: int = _DEFAULT_MAX_TICKS,
    extra: dict[str, Any] | None = None,
    checkpointer: Any = None,
) -> Any:
    """Build and return a compiled AdaptiveTopology meta-graph."""
    cfg = TopologyConfig(
        name="adaptive",
        max_iterations=max_iterations,
        extra=extra or {},
    )
    return AdaptiveTopology().build(agents, cfg, checkpointer=checkpointer)
