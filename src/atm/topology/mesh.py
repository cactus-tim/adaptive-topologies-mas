"""MeshTopology — broadcast-bus graph with dispatcher, voting, and consensus.

Architecture (arch.md §7.3):
  Agents communicate via a shared broadcast bus. A dispatcher node selects
  the next agent to activate using either round-robin or priority policy.
  After each agent run, mesh_broadcast reads the agent's outbox and appends
  messages to shared.broadcast_bus (capped at broadcast_bus_cap).
  mesh_postprocess tallies DECISION votes and checks for consensus.

Graph structure:
  START → dispatcher → agent_node → mesh_broadcast → mesh_postprocess
          ↑                                              |
          |_____________loop (no consensus)______________|
          |____________END  (consensus or max_rounds)____|

Nodes:
  dispatcher, planner, researcher, executor, critic,
  mesh_broadcast, mesh_postprocess

TopologyConfig.extra keys:
  max_rounds           — max dispatch rounds before forced END (default 6)
  consensus_threshold  — votes needed to reach consensus (default 3)
  activation_policy    — "round_robin" (default) or "priority"
  agent_order          — list of agent_ids for round-robin ordering
  broadcast_bus_cap    — hard cap on broadcast_bus list length (default 200, MC-5)

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ROUNDS = 6
_DEFAULT_CONSENSUS_THRESHOLD = 3
_DEFAULT_BROADCAST_BUS_CAP = 200
_DEFAULT_ACTIVATION_POLICY = "round_robin"
_DEFAULT_AGENT_ORDER = ["planner", "researcher", "executor", "critic"]

# Route sentinels
_ROUTE_AGENT_PREFIX = "agent:"  # sentinel prefix for dispatcher → agent routing
_ROUTE_DISPATCHER = "dispatcher"
_ROUTE_END = "__end__"


# ---------------------------------------------------------------------------
# mesh_broadcast — post-process node (MC-5: bounded broadcast_bus)
# ---------------------------------------------------------------------------


async def mesh_broadcast(state: GraphState, *, _active_agent_id: str) -> dict[str, Any]:
    """Read the active agent's outbox and append to shared.broadcast_bus with cap.

    This is the ONLY node that writes to broadcast_bus. Agent nodes write
    exclusively to their own outbox.

    Args:
        state:             Current GraphState.
        _active_agent_id:  Agent whose outbox should be flushed to bus.

    Returns:
        State delta updating shared.broadcast_bus.
    """
    agents: dict[str, Any] = dict(state.get("agents") or {})
    agent_state: dict[str, Any] = dict(agents.get(_active_agent_id) or {})
    outbox: list[Any] = list(agent_state.get("outbox") or [])

    shared: dict[str, Any] = dict(state.get("shared") or {})
    existing_bus: list[Any] = list(shared.get("broadcast_bus") or [])

    # Append new messages and apply cap (MC-5)
    new_bus = existing_bus + outbox
    cap: int = _DEFAULT_BROADCAST_BUS_CAP
    if len(new_bus) > cap:
        new_bus = new_bus[-cap:]

    shared["broadcast_bus"] = new_bus
    return {"shared": shared}


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
        # dispatcher node — selects next agent to activate
        # ----------------------------------------------------------------

        async def dispatcher_node(state: GraphState) -> dict[str, Any]:
            """Increment round counter and record active_agent_id in signals."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            # Advance round-robin index
            rr_index: int = int(signals.get("_mesh_rr_index", 0))
            dispatch_round: int = int(signals.get("_mesh_dispatch_round", 0))

            # Determine next agent
            if activation_policy == "priority":
                next_agent = _pick_priority_agent(state, agent_order, rr_index)
            else:
                next_agent = agent_order[rr_index % len(agent_order)]

            # Advance round-robin pointer
            new_rr_index = (rr_index + 1) % len(agent_order)
            # Increment dispatch round when we wrap around (full cycle) or always?
            # We count dispatch_round as total calls to dispatcher.
            new_dispatch_round = dispatch_round + 1

            signals["_mesh_rr_index"] = new_rr_index
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
            active_agent: str = str(signals.get("_mesh_active_agent", agent_order[0]))
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
        graph.add_conditional_edges(
            "dispatcher",
            _route_from_dispatcher,
            {agent_id: agent_id for agent_id in agent_order},
        )

        # Each agent → mesh_broadcast
        for agent_id in agent_order:
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
