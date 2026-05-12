"""MeshTopology — broadcast-bus graph with dispatcher, voting, and consensus.

Architecture (arch.md §7.3):
  Agents communicate via a shared broadcast bus. A dispatcher node selects
  the next agent to activate using either round-robin or priority policy.
  After each agent run, mesh_broadcast reads the agent's outbox and appends
  messages to shared.broadcast_bus (capped at broadcast_bus_cap).
  mesh_postprocess tallies DECISION votes and checks for consensus.

Graph structure (without HITL):
  START → dispatcher → agent_node → mesh_broadcast → mesh_postprocess
          ↑                                              |
          |_____________loop (no consensus)______________|
          |____________END  (consensus or max_rounds)____|

Graph structure (with HITL, human_cfg.enabled=True):
  human_peer is inserted into agent_order and participates in round-robin.
  human_peer is skipped by the dispatcher until activation_round is reached
  (default 2). human_peer votes via shared.broadcast_bus DECISION messages.

Nodes:
  dispatcher, planner, researcher, executor, critic,
  mesh_broadcast, mesh_postprocess
  (+ human_peer when human_cfg.enabled=True)

TopologyConfig.extra keys:
  max_rounds           — max dispatch rounds before forced END (default 6)
  consensus_threshold  — votes needed to reach consensus (default 3)
  activation_policy    — "round_robin" (default) or "priority"
  agent_order          — list of agent_ids for round-robin ordering
  broadcast_bus_cap    — hard cap on broadcast_bus list length (default 200, MC-5)

HumanCfg.extra keys (when human_cfg.enabled=True):
  activation_round     — first dispatch_round when human_peer participates (default 2)
  on_consensus_pending — if True, human_peer also fires when consensus_pending signal
                         is raised (default False)

Stopping precedence (arch.md §7.1):
  budget → (raised upstream by LLMWrapper as BudgetExceededError)
  max_iter (global) → topology_success (consensus) → topology_max (max_rounds) → continue

Vote payload format (MC-7 / task 2.2 test_consensus_vote_payload_str_format):
  MessageKind.DECISION with payload={"vote_for": str}
  Non-string vote_for values are silently ignored during tally.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState
from atm.core.types import MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop

# ---------------------------------------------------------------------------
# Lazy imports for HITL — patchable in tests
# ---------------------------------------------------------------------------
# These are set to None if the respective module is unavailable at import time.
# Tests patch "atm.topology.mesh.<Name>" to inject stubs.
# Typed as Any to allow both the class/function and None without mypy complaints.

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ROUNDS = 6
_DEFAULT_CONSENSUS_THRESHOLD = 3
_DEFAULT_BROADCAST_BUS_CAP = 200
_DEFAULT_ACTIVATION_POLICY = "round_robin"
_DEFAULT_AGENT_ORDER = ["planner", "researcher", "executor", "critic"]

# HITL constants
_DEFAULT_HUMAN_ACTIVATION_ROUND = 2
_HUMAN_PEER_ID = "human_peer"

# Route sentinels
_ROUTE_AGENT_PREFIX = "agent:"  # sentinel prefix for dispatcher → agent routing
_ROUTE_DISPATCHER = "dispatcher"
_ROUTE_END = "__end__"


# ---------------------------------------------------------------------------
# MeshTopology
# ---------------------------------------------------------------------------


@TopologyRegistry.register("mesh")
class MeshTopology:
    """Broadcast-bus topology with dispatcher and consensus voting.

    Graph:
      START → dispatcher → (agent_node) → mesh_broadcast → mesh_postprocess
              ↑                                                    |
              |_____________ loop (no consensus, rounds left) _____|
                             END (consensus reached OR max_rounds hit OR global max_iter)

    Activation policies:
      round_robin — cycles through agent_order list; state counter tracks position.
      priority    — routes to "critic" if any broadcast_bus message has
                    kind==DRAFT, otherwise falls back to round-robin position.

    Vote tally in mesh_postprocess:
      Counts MessageKind.DECISION messages in broadcast_bus with string payload["vote_for"].
      First vote_for value reaching consensus_threshold wins.
      Sets shared.signals["consensus_reached"] = True and shared.final_answer = winner.
    """

    name = "mesh"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a CompiledStateGraph.

        Args:
            agents: Dict mapping agent_id → Agent instance. Expected keys:
                    planner, researcher, executor, critic.
            cfg:    TopologyConfig with max_iterations and extra mesh params.
            **kwargs: Optional; checkpointer=... forwarded to graph.compile().
                      human_cfg=HumanCfg enables HITL (human_peer in agent_order).
                      human_gateway_llm=LLMWrapper forwarded to LLMSimulatedGateway.

        Returns:
            CompiledStateGraph ready for ainvoke.
        """
        checkpointer = kwargs.get("checkpointer")
        extra = cfg.extra or {}

        max_rounds: int = int(extra.get("max_rounds", _DEFAULT_MAX_ROUNDS))
        consensus_threshold: int = int(
            extra.get("consensus_threshold", _DEFAULT_CONSENSUS_THRESHOLD)
        )
        activation_policy: str = str(extra.get("activation_policy", _DEFAULT_ACTIVATION_POLICY))
        agent_order: list[str] = list(extra.get("agent_order", _DEFAULT_AGENT_ORDER))
        broadcast_bus_cap: int = int(extra.get("broadcast_bus_cap", _DEFAULT_BROADCAST_BUS_CAP))

        # ----------------------------------------------------------------
        # HITL configuration — human_peer participates in round-robin
        # ----------------------------------------------------------------
        human_cfg: Any = kwargs.get("human_cfg")
        human_enabled: bool = human_cfg is not None and bool(getattr(human_cfg, "enabled", False))

        # Per-topology HITL extra config
        human_extra: dict[str, Any] = {}
        if human_enabled and human_cfg is not None:
            human_extra = dict(getattr(human_cfg, "extra", None) or {})

        activation_round: int = int(
            human_extra.get("activation_round", _DEFAULT_HUMAN_ACTIVATION_ROUND)
        )
        on_consensus_pending: bool = bool(human_extra.get("on_consensus_pending", False))

        # Build human_peer gateway when HITL is enabled
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

        # Extend agent_order with human_peer when HITL is enabled
        effective_agent_order: list[str] = list(agent_order)
        if human_enabled:
            effective_agent_order.append(_HUMAN_PEER_ID)

        # ----------------------------------------------------------------
        # dispatcher node — selects next agent to activate
        # ----------------------------------------------------------------

        async def dispatcher_node(state: GraphState) -> dict[str, Any]:
            """Increment round counter and record active_agent_id in signals.

            When human_peer is in effective_agent_order, it is skipped if the
            current dispatch_round is < activation_round. The round-robin index
            continues advancing through human_peer's slot (so ordering is stable)
            but the dispatcher is called recursively until a non-skipped agent
            is selected. This is done via a loop rather than recursion to avoid
            deep LangGraph node re-invocation.
            """
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            # Advance round-robin index
            rr_index: int = int(signals.get("_mesh_rr_index", 0))
            dispatch_round: int = int(signals.get("_mesh_dispatch_round", 0))

            # Determine next agent — may skip human_peer before activation_round
            # and skip human_peer if consensus_pending is not set (when on_consensus_pending=True)
            consensus_pending_set: bool = bool(signals.get("consensus_pending", False))

            # Loop to find the next non-skipped agent
            attempts = 0
            max_attempts = len(effective_agent_order) + 1
            next_agent: str = effective_agent_order[0]
            while attempts < max_attempts:
                if activation_policy == "priority":
                    candidate = _pick_priority_agent(state, effective_agent_order, rr_index)
                else:
                    candidate = effective_agent_order[rr_index % len(effective_agent_order)]

                # Determine if this candidate should be skipped
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
                # All candidates were skipped (e.g. only human_peer in order, not activated)
                # Fall back to first non-human agent
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

        # ----------------------------------------------------------------
        # _route_from_dispatcher — conditional edge after dispatcher
        # ----------------------------------------------------------------

        def _route_from_dispatcher(state: GraphState) -> str:
            """Return the agent node name to activate."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            active_agent: str = str(signals.get("_mesh_active_agent", effective_agent_order[0]))
            return active_agent

        # ----------------------------------------------------------------
        # Agent node wrappers
        # ----------------------------------------------------------------

        async def _wrap_agent(agent_id: str, state: GraphState) -> dict[str, Any]:
            """Call agent.step() if the agent exists, else return empty delta."""
            agent = agents.get(agent_id)
            if agent is None:
                logger.warning("mesh: agent %r not found in agents dict", agent_id)
                return {}
            result: dict[str, Any] = await agent.step(state)
            return result

        # Build individual node closures (must capture agent_id by value)
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

        # ----------------------------------------------------------------
        # human_peer node — HITL voter in the mesh
        # ----------------------------------------------------------------

        if human_enabled and human_cfg is not None and human_gateway is not None:
            import time
            from copy import deepcopy
            from datetime import UTC, datetime

            from langchain_core.callbacks.manager import adispatch_custom_event

            _hcfg = human_cfg
            _hgw = human_gateway

            async def human_peer_node(state: GraphState) -> dict[str, Any]:
                """HITL peer node — requests a vote from the human gateway.

                The human votes by returning an action payload that is converted
                to a DECISION message with payload={"vote_for": action} and
                appended to the broadcast_bus via shared state.

                request_id: "mesh:{run_id}:{dispatch_round}:peer"
                """
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

                request_id = f"mesh:{run_id}:{dispatch_round_now}:peer"

                # Build HumanContext — extract question from bus or use default
                from atm.core.types import HumanContext, Message, MessageKind

                bus: list[Any] = list(shared.get("broadcast_bus") or [])
                question = "Please vote for the best answer. Reply with your choice as 'vote_for'."
                for msg in reversed(bus):
                    if getattr(msg, "kind", None) == MessageKind.DRAFT:
                        question = getattr(msg, "content", question) or question
                        break

                ctx = HumanContext(
                    run_id=run_id,
                    role=_hcfg.role,
                    question=question,
                    recent_messages=(),
                    allowed_actions=("vote",),
                    deadline_s=int(_hcfg.timeout_s) if _hcfg.timeout_s is not None else None,
                )

                # Dispatch human_request event
                _requested_at = datetime.now(UTC)
                try:
                    await adispatch_custom_event(
                        "human_request",
                        {
                            "run_id": run_id,
                            "request_id": request_id,
                            "role": str(
                                _hcfg.role.value if hasattr(_hcfg.role, "value") else _hcfg.role
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

                # Call gateway
                _t0 = time.monotonic()
                timeout_s_val: float | None = getattr(_hcfg, "timeout_s", None)
                policy: str = getattr(_hcfg, "timeout_policy", "skip")

                if request_with_timeout is not None and timeout_s_val is not None:
                    response = await request_with_timeout(
                        _hgw,
                        ctx,
                        request_id=request_id,
                        timeout_s=timeout_s_val,
                        policy=policy,
                    )
                else:
                    response = await _hgw.request(ctx, request_id=request_id)

                _latency_s = time.monotonic() - _t0

                # Dispatch human_response event
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

                # Convert response action to a DECISION vote on the bus
                action: str = getattr(response, "action", "") or ""
                # The comment or payload may contain the vote_for value
                comment: str = getattr(response, "comment", "") or ""

                # Extract vote_for from response: prefer payload["vote_for"],
                # then comment, then action itself
                resp_payload: dict[str, Any] = dict(getattr(response, "payload", {}) or {})
                vote_for: str = str(resp_payload.get("vote_for") or comment.strip() or action)

                vote_msg = Message(
                    sender=_HUMAN_PEER_ID,
                    kind=MessageKind.DECISION,
                    content=f"Human peer votes for: {vote_for}",
                    payload={"vote_for": vote_for},
                )

                # Append to broadcast_bus
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

        # ----------------------------------------------------------------
        # _route_from_agent — after agent, go to mesh_broadcast
        # (always, regardless of which agent ran)
        # ----------------------------------------------------------------

        # We cannot use a single conditional edge from multiple sources to
        # mesh_broadcast — instead we add direct edges from each agent node.

        # ----------------------------------------------------------------
        # mesh_broadcast node factory (reads active_agent from signals)
        # ----------------------------------------------------------------

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

        # ----------------------------------------------------------------
        # mesh_postprocess — vote tally and consensus check
        # ----------------------------------------------------------------

        async def mesh_postprocess_node(state: GraphState) -> dict[str, Any]:
            """Count DECISION votes in broadcast_bus; check consensus threshold.

            Vote format: MessageKind.DECISION with payload={"vote_for": str}.
            Non-string vote_for values are silently ignored.

            Sets shared.signals["consensus_reached"] = True and
            shared.final_answer = winner_value when threshold met.
            Also increments shared.iter_total (one iteration = one dispatch cycle).

            When HITL is enabled: additionally sets signals["consensus_pending"] = True
            when ≥1 vote exists but threshold is NOT yet reached. This is additive —
            consensus_reached is set/cleared independently. consensus_pending is
            consumed by the dispatcher to optionally activate human_peer early.
            """
            shared: dict[str, Any] = dict(state.get("shared") or {})
            bus: list[Any] = list(shared.get("broadcast_bus") or [])
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            # Increment global iter_total (each postprocess = 1 iteration)
            old_iter: int = int(shared.get("iter_total") or 0)
            shared["iter_total"] = old_iter + 1

            # Tally votes
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

            # Check consensus
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

            # Additive: set consensus_pending when HITL is enabled and there are
            # votes but threshold not reached (split-vote condition).
            if human_enabled:
                total_votes = sum(vote_counts.values())
                if total_votes > 0 and winner is None:
                    signals["consensus_pending"] = True
                else:
                    signals["consensus_pending"] = False

            shared["signals"] = signals
            return {"shared": shared}

        # ----------------------------------------------------------------
        # _route_from_postprocess — conditional edge after mesh_postprocess
        # ----------------------------------------------------------------

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

        # ----------------------------------------------------------------
        # Assemble graph
        # ----------------------------------------------------------------

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        # Add nodes
        graph.add_node("dispatcher", dispatcher_node)
        graph.add_node("mesh_broadcast", mesh_broadcast_node)
        graph.add_node("mesh_postprocess", mesh_postprocess_node)

        for agent_id, node_fn in _agent_nodes.items():
            graph.add_node(agent_id, node_fn)

        # Entry point
        graph.add_edge(START, "dispatcher")

        # Dispatcher → agent (conditional)
        # Include human_peer in routing map when HITL enabled
        dispatcher_routing: dict[str, str] = {
            agent_id: agent_id for agent_id in effective_agent_order
        }
        graph.add_conditional_edges(
            "dispatcher",
            _route_from_dispatcher,
            dispatcher_routing,  # type: ignore[arg-type]
        )

        # Each agent → mesh_broadcast (includes human_peer when enabled)
        for agent_id in effective_agent_order:
            graph.add_edge(agent_id, "mesh_broadcast")

        # mesh_broadcast → mesh_postprocess
        graph.add_edge("mesh_broadcast", "mesh_postprocess")

        # mesh_postprocess → dispatcher or END
        graph.add_conditional_edges(
            "mesh_postprocess",
            _route_from_postprocess,
            {
                _ROUTE_DISPATCHER: "dispatcher",
                _ROUTE_END: END,
            },
        )

        return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_priority_agent(
    state: GraphState,
    agent_order: list[str],
    rr_index: int,
) -> str:
    """Priority policy: route to 'critic' if any DRAFT exists on broadcast_bus.

    Falls back to round-robin position when no DRAFT is found.

    Args:
        state:       Current GraphState.
        agent_order: Ordered list of agent IDs for fallback round-robin.
        rr_index:    Current round-robin pointer (for fallback).

    Returns:
        agent_id string.
    """
    shared: dict[str, Any] = dict(state.get("shared") or {})
    bus: list[Any] = list(shared.get("broadcast_bus") or [])

    for msg in bus:
        kind = getattr(msg, "kind", None)
        if (kind == MessageKind.DRAFT or str(kind) == "draft") and "critic" in agent_order:
            # Route to critic as priority reviewer
            return "critic"

    # Fallback: round-robin
    return agent_order[rr_index % len(agent_order)]
