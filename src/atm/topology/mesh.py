"""MeshTopology — broadcast-bus graph with dispatcher, voting, and consensus."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState
from atm.core.types import MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop, get_topology_extras
from atm.topology.star import _extract_final_answer

LLMSimulatedGateway: Any
CLIGateway: Any
request_with_timeout: Any

try:
    from atm.human.llm_simulated import LLMSimulatedGateway as LLMSimulatedGateway
except ImportError:  # pragma: no cover
    LLMSimulatedGateway = None

try:
    from atm.human.cli_gateway import CLIGateway as CLIGateway
except ImportError:  # pragma: no cover
    CLIGateway = None

try:
    from atm.human._timeout import request_with_timeout as request_with_timeout
except ImportError:  # pragma: no cover
    request_with_timeout = None

HumanRoleRouter: Any
try:
    from atm.human.role_router import HumanRoleRouter as HumanRoleRouter
except ImportError:  # pragma: no cover
    HumanRoleRouter = None

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ROUNDS = 12
_DEFAULT_CONSENSUS_THRESHOLD = 3
_DEFAULT_BROADCAST_BUS_CAP = 200
_DEFAULT_ACTIVATION_POLICY = "round_robin"
_DEFAULT_AGENT_ORDER = ["planner", "researcher", "executor", "critic"]

_DEFAULT_HUMAN_ACTIVATION_ROUND = 2
_HUMAN_PEER_ID = "human_peer"

_ROUTE_AGENT_PREFIX = "agent:"
_ROUTE_DISPATCHER = "dispatcher"
_ROUTE_END = "__end__"


@TopologyRegistry.register("mesh")
class MeshTopology:
    """Broadcast-bus topology: dispatcher → agent → broadcast → consensus vote → loop/END."""

    name = "mesh"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a CompiledStateGraph."""
        checkpointer = kwargs.get("checkpointer")
        extras = get_topology_extras(cfg, "mesh")

        max_rounds: int = int(extras.get("max_rounds", _DEFAULT_MAX_ROUNDS))
        consensus_threshold: int = int(
            extras.get("consensus_threshold", _DEFAULT_CONSENSUS_THRESHOLD)
        )
        activation_policy: str = str(extras.get("activation_policy", _DEFAULT_ACTIVATION_POLICY))
        agent_order: list[str] = list(extras.get("agent_order", _DEFAULT_AGENT_ORDER))
        broadcast_bus_cap: int = int(extras.get("broadcast_bus_cap", _DEFAULT_BROADCAST_BUS_CAP))

        human_cfg: Any = kwargs.get("human_cfg")
        human_enabled: bool = human_cfg is not None and bool(getattr(human_cfg, "enabled", False))
        role_router: Any = kwargs.get("role_router")

        human_extra: dict[str, Any] = {}
        if human_enabled and human_cfg is not None:
            human_extra = dict(getattr(human_cfg, "extra", None) or {})

        activation_round: int = int(
            human_extra.get("activation_round", _DEFAULT_HUMAN_ACTIVATION_ROUND)
        )
        on_consensus_pending: bool = bool(human_extra.get("on_consensus_pending", False))

        human_gateway: Any = None
        if human_enabled and human_cfg is not None:
            gateway_llm: Any = kwargs.get("human_gateway_llm")
            gateway_kind: str = getattr(human_cfg, "gateway", "llm_simulated")
            if gateway_kind == "cli":
                human_gateway = CLIGateway() if CLIGateway is not None else None
            else:
                human_gateway = (
                    LLMSimulatedGateway(llm=gateway_llm)
                    if LLMSimulatedGateway is not None
                    else None
                )
            if human_gateway is None:  # pragma: no cover
                raise ImportError(
                    f"Gateway class for '{gateway_kind}' could not be imported. "
                    "Ensure atm.human is installed."
                )

        effective_agent_order: list[str] = list(agent_order)
        if human_enabled:
            effective_agent_order.append(_HUMAN_PEER_ID)

        async def dispatcher_node(state: GraphState) -> dict[str, Any]:
            """Select next agent via round-robin or priority; skip human_peer before activation_round."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            rr_index: int = int(signals.get("_mesh_rr_index", 0))
            dispatch_round: int = int(signals.get("_mesh_dispatch_round", 0))

            consensus_pending_set: bool = bool(signals.get("consensus_pending", False))

            attempts = 0
            max_attempts = len(effective_agent_order) + 1
            next_agent: str = effective_agent_order[0]
            while attempts < max_attempts:
                if activation_policy == "priority":
                    candidate = _pick_priority_agent(state, effective_agent_order, rr_index)
                else:
                    candidate = effective_agent_order[rr_index % len(effective_agent_order)]

                skip = False
                if (
                    candidate == _HUMAN_PEER_ID
                    and human_enabled
                    and (
                        dispatch_round < activation_round
                        or (on_consensus_pending and not consensus_pending_set)
                    )
                ):
                    skip = True

                rr_index = (rr_index + 1) % len(effective_agent_order)

                if not skip:
                    next_agent = candidate
                    break
                attempts += 1
            else:
                for aid in effective_agent_order:
                    if aid != _HUMAN_PEER_ID:
                        next_agent = aid
                        break

            new_dispatch_round = dispatch_round + 1

            signals["_mesh_rr_index"] = rr_index
            signals["_mesh_dispatch_round"] = new_dispatch_round
            signals["_mesh_active_agent"] = next_agent

            shared["signals"] = signals
            return {"shared": shared}

        def _route_from_dispatcher(state: GraphState) -> str:
            """Return the agent node name to activate."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            active_agent: str = str(signals.get("_mesh_active_agent", effective_agent_order[0]))
            return active_agent

        async def _wrap_agent(agent_id: str, state: GraphState) -> dict[str, Any]:
            """Call agent.step() if the agent exists, else return empty delta."""
            agent = agents.get(agent_id)
            if agent is None:
                logger.warning("mesh: agent %r not found in agents dict", agent_id)
                return {}
            result: dict[str, Any] = await agent.step(state)
            return result

        def _make_agent_node(agent_id: str) -> Any:
            async def node(state: GraphState) -> dict[str, Any]:
                return await _wrap_agent(agent_id, state)

            node.__name__ = agent_id
            return node

        planner_node = _make_agent_node("planner")
        researcher_node = _make_agent_node("researcher")
        executor_node = _make_agent_node("executor")
        critic_node = _make_agent_node("critic")

        _agent_nodes: dict[str, Any] = {
            "planner": planner_node,
            "researcher": researcher_node,
            "executor": executor_node,
            "critic": critic_node,
        }

        if human_enabled and human_cfg is not None and human_gateway is not None:
            import time
            from copy import deepcopy
            from datetime import UTC, datetime

            from langchain_core.callbacks.manager import adispatch_custom_event

            _hcfg = human_cfg
            _hgw = human_gateway
            _role_router = role_router

            async def human_peer_node(state: GraphState) -> dict[str, Any]:
                """HITL peer node: request a vote and append DECISION to broadcast_bus."""
                import uuid as _uuid_mod

                shared: dict[str, Any] = dict(deepcopy(state.get("shared") or {}))
                signals: dict[str, Any] = dict(shared.get("signals") or {})

                dispatch_round_now: int = int(signals.get("_mesh_dispatch_round", 0))

                _raw_run_id = shared.get("run_id") or (
                    state.get("run_id") if hasattr(state, "get") else None
                )
                run_id: _uuid_mod.UUID = (
                    _raw_run_id
                    if isinstance(_raw_run_id, _uuid_mod.UUID)
                    else _uuid_mod.UUID(str(_raw_run_id))
                    if _raw_run_id
                    else _uuid_mod.uuid4()
                )

                iter_total_now: int = int(shared.get("iter_total", 0))
                request_id = f"mesh:{run_id}:{iter_total_now}:{dispatch_round_now}:peer"

                from atm.core.types import HumanContext, Message, MessageKind, Phase

                bus: list[Any] = list(shared.get("broadcast_bus") or [])
                question = "Please vote for the best answer. Reply with your choice as 'vote_for'."
                for msg in reversed(bus):
                    if getattr(msg, "kind", None) == MessageKind.DRAFT:
                        question = getattr(msg, "content", question) or question
                        break

                if _role_router is not None:
                    _raw_phase = shared.get("phase", "execution")
                    _phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
                    active_role = await _role_router.decide(_phase, shared)
                else:
                    active_role = _hcfg.role

                ctx = HumanContext(
                    run_id=run_id,
                    role=active_role,
                    question=question,
                    recent_messages=(),
                    allowed_actions=("vote",),
                    deadline_s=int(_hcfg.timeout_s) if _hcfg.timeout_s is not None else None,
                )

                _requested_at = datetime.now(UTC)
                try:
                    await adispatch_custom_event(
                        "human_request",
                        {
                            "run_id": run_id,
                            "request_id": request_id,
                            "role": str(
                                active_role.value if hasattr(active_role, "value") else active_role
                            ),
                            "context_json": ctx.model_dump(mode="json"),
                            "requested_at": _requested_at,
                        },
                    )
                except Exception:
                    logger.debug(
                        "mesh.human_peer: adispatch human_request skipped (no callback ctx)",
                        exc_info=True,
                    )

                _t0 = time.monotonic()
                timeout_s_val: float | None = getattr(_hcfg, "timeout_s", None)
                policy: str = getattr(_hcfg, "timeout_policy", "skip")

                if request_with_timeout is not None and timeout_s_val is not None:
                    _fallback_gateway: Any = None
                    if policy == "llm_fallback" and LLMSimulatedGateway is not None:
                        _fb_llm: Any = getattr(_hgw, "_llm", None)
                        _fallback_gateway = LLMSimulatedGateway(llm=_fb_llm)
                    response = await request_with_timeout(
                        _hgw,
                        ctx,
                        request_id=request_id,
                        timeout_s=timeout_s_val,
                        policy=policy,
                        llm_fallback_gateway=_fallback_gateway,
                    )
                else:
                    response = await _hgw.request(ctx, request_id=request_id)

                _latency_s = time.monotonic() - _t0

                try:
                    await adispatch_custom_event(
                        "human_response",
                        {
                            "run_id": run_id,
                            "request_id": request_id,
                            "answered_at": datetime.now(UTC),
                            "response_json": response.model_dump(mode="json"),
                            "source": getattr(response, "source", "human"),
                            "timed_out": getattr(response, "timed_out", False),
                            "latency_s": _latency_s,
                        },
                    )
                except Exception:
                    logger.debug(
                        "mesh.human_peer: adispatch human_response skipped (no callback ctx)",
                        exc_info=True,
                    )

                action: str = getattr(response, "action", "") or ""
                comment: str = getattr(response, "comment", "") or ""

                resp_payload: dict[str, Any] = dict(getattr(response, "payload", {}) or {})
                vote_for: str = str(resp_payload.get("vote_for") or comment.strip() or action)

                vote_msg = Message(
                    sender=_HUMAN_PEER_ID,
                    kind=MessageKind.DECISION,
                    content=f"Human peer votes for: {vote_for}",
                    payload={"vote_for": vote_for},
                )

                new_bus = [*list(bus), vote_msg]
                if len(new_bus) > broadcast_bus_cap:
                    new_bus = new_bus[-broadcast_bus_cap:]
                shared["broadcast_bus"] = new_bus

                logger.debug(
                    "mesh.human_peer: voted vote_for=%r at dispatch_round=%d",
                    vote_for,
                    dispatch_round_now,
                )

                return {"shared": shared}

            _agent_nodes[_HUMAN_PEER_ID] = human_peer_node

        async def mesh_broadcast_node(state: GraphState) -> dict[str, Any]:
            """Flush active agent's outbox to broadcast_bus with cap."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            active_agent: str = str(signals.get("_mesh_active_agent", ""))

            agents_state: dict[str, Any] = dict(state.get("agents") or {})
            agent_s: dict[str, Any] = dict(agents_state.get(active_agent) or {})
            outbox: list[Any] = list(agent_s.get("outbox") or [])

            existing_bus: list[Any] = list(shared.get("broadcast_bus") or [])
            new_bus = existing_bus + outbox
            if len(new_bus) > broadcast_bus_cap:
                new_bus = new_bus[-broadcast_bus_cap:]

            shared["broadcast_bus"] = new_bus
            return {"shared": shared}

        async def mesh_postprocess_node(state: GraphState) -> dict[str, Any]:
            """Tally DECISION votes on broadcast_bus; set consensus_reached when threshold met."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            bus: list[Any] = list(shared.get("broadcast_bus") or [])
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            old_iter: int = int(shared.get("iter_total") or 0)
            shared["iter_total"] = old_iter + 1

            vote_counts: dict[str, int] = {}
            for msg in bus:
                kind = getattr(msg, "kind", None)
                if kind != MessageKind.DECISION and str(kind) != "decision":
                    continue
                payload: dict[str, Any] = getattr(msg, "payload", {}) or {}
                vote_for = payload.get("vote_for")
                if not isinstance(vote_for, str):
                    continue
                vote_counts[vote_for] = vote_counts.get(vote_for, 0) + 1

            winner: str | None = None
            for candidate, count in vote_counts.items():
                if count >= consensus_threshold:
                    winner = candidate
                    break

            if winner is not None:
                signals["consensus_reached"] = True
                signals["consensus_winner"] = winner
                shared["final_answer"] = winner
            else:
                signals["consensus_reached"] = False
                signals.pop("consensus_winner", None)

            if human_enabled:
                total_votes = sum(vote_counts.values())
                if total_votes > 0 and winner is None:
                    signals["consensus_pending"] = True
                else:
                    signals["consensus_pending"] = False

            if not shared.get("final_answer"):
                provisional = _extract_final_answer(state)
                if provisional and provisional != "<incomplete>":
                    shared["final_answer"] = provisional

            shared["signals"] = signals
            return {"shared": shared}

        def _route_from_postprocess(state: GraphState) -> str:
            """Return dispatcher or END based on stopping conditions."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            dispatch_round: int = int(signals.get("_mesh_dispatch_round", 0))
            consensus_reached: bool = bool(signals.get("consensus_reached", False))
            max_rounds_reached: bool = dispatch_round >= max_rounds

            stop, _reason = _should_stop(
                {"shared": shared},
                cfg,
                topology_success=consensus_reached,
                topology_max_reached=max_rounds_reached,
            )
            if stop:
                return _ROUTE_END

            return _ROUTE_DISPATCHER

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        graph.add_node("dispatcher", dispatcher_node)
        graph.add_node("mesh_broadcast", mesh_broadcast_node)
        graph.add_node("mesh_postprocess", mesh_postprocess_node)

        for agent_id, node_fn in _agent_nodes.items():
            graph.add_node(agent_id, node_fn)

        graph.add_edge(START, "dispatcher")

        dispatcher_routing: dict[str, str] = {
            agent_id: agent_id for agent_id in effective_agent_order
        }
        graph.add_conditional_edges(
            "dispatcher",
            _route_from_dispatcher,
            dispatcher_routing,  # type: ignore[arg-type]
        )

        for agent_id in effective_agent_order:
            graph.add_edge(agent_id, "mesh_broadcast")

        graph.add_edge("mesh_broadcast", "mesh_postprocess")
        graph.add_conditional_edges(
            "mesh_postprocess",
            _route_from_postprocess,
            {
                _ROUTE_DISPATCHER: "dispatcher",
                _ROUTE_END: END,
            },
        )

        return graph.compile(checkpointer=checkpointer)


def _pick_priority_agent(
    state: GraphState,
    agent_order: list[str],
    rr_index: int,
) -> str:
    """Return 'critic' when DRAFT exists on broadcast_bus, else round-robin fallback."""
    shared: dict[str, Any] = dict(state.get("shared") or {})
    bus: list[Any] = list(shared.get("broadcast_bus") or [])

    for msg in bus:
        kind = getattr(msg, "kind", None)
        if (kind == MessageKind.DRAFT or str(kind) == "draft") and "critic" in agent_order:
            return "critic"

    return agent_order[rr_index % len(agent_order)]
