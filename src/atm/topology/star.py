"""StarTopology — coordinator-centered LangGraph with phase-advance semantics.

Architecture (arch.md §7.2):
  - Graph: coordinator → {planner|executor|critic} → coordinator → ... → END
  - Entry point: "coordinator"
  - Coordinator is a rule-based (not LLM) async node.
  - Phase-advance semantics:
      planning → execution → verification → done
  - Phase-advance triggers:
      planning:     signals.ready_for_execution OR iter_within_phase >= planning_max_iter
      execution:    signals.ready_for_verification OR iter_within_phase >= exec_max_iter
      verification: signals.critic_approved == True OR iter_within_phase >= verify_max_iter → done
      verification: otherwise → critic (loop)
  - Global guard: _should_stop(iter_total >= max_iterations) → END

Nodes:
  coordinator, planner, executor, critic, critic_postprocess

Edges:
  START → coordinator
  coordinator → conditional via _route_from_coord
  planner → coordinator
  executor → coordinator
  critic → critic_postprocess → coordinator

TopologyConfig.extra defaults:
  planning_max_iter: 2
  exec_max_iter: 5
  verify_max_iter: 3
"""

from __future__ import annotations

import logging
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState
from atm.core.types import MessageKind, Phase
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — default phase caps
# ---------------------------------------------------------------------------

_DEFAULT_PLANNING_MAX_ITER = 2
_DEFAULT_EXEC_MAX_ITER = 5
_DEFAULT_VERIFY_MAX_ITER = 3

# Route sentinels used in conditional edge map
_ROUTE_PLANNER = "planner"
_ROUTE_EXECUTOR = "executor"
_ROUTE_CRITIC = "critic"
_ROUTE_END = "__end__"


# ---------------------------------------------------------------------------
# _critic_postprocess adapter node
# ---------------------------------------------------------------------------


async def _critic_postprocess(state: GraphState) -> dict[str, Any]:
    """Parse the last DECISION-kind message from critic.outbox.

    Sets shared.signals["critic_approved"]:
      - True  if payload["approved"] == True
      - False if payload["approved"] == False OR message is malformed

    Malformed = no MessageKind.DECISION messages OR missing payload["approved"].
    Malformed is treated as rejected with a WARNING log (no exception raised).

    Args:
        state: GraphState dict.

    Returns:
        Delta dict updating shared.signals["critic_approved"].
    """
    agents: dict[str, Any] = dict(state.get("agents") or {})
    critic_state: dict[str, Any] = dict(agents.get("critic") or {})
    outbox: list[Any] = list(critic_state.get("outbox") or [])

    # Find last DECISION-kind message
    decision_msg = None
    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DECISION or str(kind) == "decision":
            decision_msg = msg
            break

    approved: bool = False

    if decision_msg is None:
        logger.warning(
            "critic_postprocess: no DECISION message found in critic outbox; "
            "treating as rejected"
        )
    else:
        payload: dict[str, Any] = getattr(decision_msg, "payload", {}) or {}
        if "approved" not in payload:
            logger.warning(
                "critic_postprocess: DECISION message has no 'approved' key in payload; "
                "treating as rejected"
            )
            approved = False
        else:
            try:
                approved = bool(payload["approved"])
            except (TypeError, ValueError):
                logger.warning(
                    "critic_postprocess: could not parse payload['approved'] as bool; "
                    "treating as rejected"
                )
                approved = False

    # Return delta: update shared.signals["critic_approved"]
    existing_shared: dict[str, Any] = dict(state.get("shared") or {})
    existing_signals: dict[str, Any] = dict(existing_shared.get("signals") or {})
    existing_signals["critic_approved"] = approved
    existing_shared["signals"] = existing_signals

    return {"shared": existing_shared}


# ---------------------------------------------------------------------------
# StarTopology
# ---------------------------------------------------------------------------


@TopologyRegistry.register("star")
class StarTopology:
    """Coordinator-centered LangGraph topology.

    Graph structure:
      START → coordinator → conditional routing →
        planner → coordinator (loop)
        executor → coordinator (loop)
        critic → critic_postprocess → coordinator (loop)
        END

    The coordinator node:
      1. Increments iter_total and iteration.
      2. Computes iter_within_phase = iter_total - phase_started_at_iter.
      3. Applies phase-advance rules (plan → exec → verify → done).
      4. On verification→done: populates shared.final_answer from last Executor DRAFT.
      5. Updates shared with new counters and phase_history.

    _route_from_coord() reads the updated state and returns the next node name.
    """

    name = "star"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a CompiledStateGraph.

        Args:
            agents: Dict mapping agent_id → Agent instance with async .step() method.
            cfg:    TopologyConfig holding max_iterations and extra phase caps.
            **kwargs: Optional; checkpointer=... is forwarded to graph.compile().

        Returns:
            CompiledStateGraph ready for ainvoke.
        """
        checkpointer = kwargs.get("checkpointer")

        # Extract phase caps from extra dict with defaults
        extra = cfg.extra or {}
        planning_max_iter: int = int(extra.get("planning_max_iter", _DEFAULT_PLANNING_MAX_ITER))
        exec_max_iter: int = int(extra.get("exec_max_iter", _DEFAULT_EXEC_MAX_ITER))
        verify_max_iter: int = int(extra.get("verify_max_iter", _DEFAULT_VERIFY_MAX_ITER))

        # ----------------------------------------------------------------
        # Build coordinator node (closure captures cfg + phase caps)
        # ----------------------------------------------------------------

        async def coordinator_node(state: GraphState) -> dict[str, Any]:
            """Rule-based coordinator: increments counters + advances phase."""
            shared: dict[str, Any] = dict(state.get("shared") or {})

            # Increment global counters
            old_iter_total: int = int(shared.get("iter_total") or 0)
            new_iter_total = old_iter_total + 1
            new_iteration = int(shared.get("iteration") or 0) + 1
            shared["iter_total"] = new_iter_total
            shared["iteration"] = new_iteration

            # Current phase and phase tracking
            phase_started_at: int = int(shared.get("phase_started_at_iter") or 0)
            iter_within_phase = new_iter_total - phase_started_at
            current_phase = shared.get("phase") or Phase.PLANNING
            current_phase_str = str(current_phase)

            signals: dict[str, Any] = dict(shared.get("signals") or {})
            phase_history: list[Any] = list(shared.get("phase_history") or [])

            # ---- Phase-advance logic ----

            if current_phase_str == "planning":
                # Advance to execution if signal or iter cap reached
                if signals.get("ready_for_execution") or iter_within_phase >= planning_max_iter:
                    phase_history = list(phase_history)
                    phase_history.append(Phase.PLANNING)
                    shared["phase"] = Phase.EXECUTION
                    shared["phase_started_at_iter"] = new_iter_total
                    shared["phase_history"] = phase_history

            elif current_phase_str == "execution":
                # Advance to verification if signal or iter cap reached
                if signals.get("ready_for_verification") or iter_within_phase >= exec_max_iter:
                    phase_history = list(phase_history)
                    phase_history.append(Phase.EXECUTION)
                    shared["phase"] = Phase.VERIFICATION
                    shared["phase_started_at_iter"] = new_iter_total
                    shared["phase_history"] = phase_history

            elif current_phase_str == "verification":
                critic_approved = signals.get("critic_approved")
                if critic_approved is True or iter_within_phase >= verify_max_iter:
                    # Advance to done
                    phase_history = list(phase_history)
                    phase_history.append(Phase.VERIFICATION)
                    shared["phase"] = Phase.DONE
                    shared["phase_history"] = phase_history

                    # Populate final_answer from last Executor DRAFT message
                    final_answer = _extract_final_answer(state)
                    shared["final_answer"] = final_answer

            shared["signals"] = signals
            return {"shared": shared}

        # ----------------------------------------------------------------
        # Build _route_from_coord (closure captures cfg)
        # ----------------------------------------------------------------

        def _route_from_coord(state: GraphState) -> str:
            """Routing function — returns next node name after coordinator."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            current_phase = shared.get("phase") or Phase.PLANNING

            # Global stop guard
            stop, _reason = _should_stop(
                cast(dict[str, Any], state),
                cfg,
                topology_success=(str(current_phase) == "done"),
                topology_max_reached=False,
            )
            if stop:
                return _ROUTE_END

            # Phase-based routing
            phase_str = str(current_phase)

            if phase_str == "planning":
                return _ROUTE_PLANNER

            if phase_str == "execution":
                return _ROUTE_EXECUTOR

            if phase_str == "verification":
                return _ROUTE_CRITIC

            if phase_str == "done":
                return _ROUTE_END

            # Fallback: END (should not happen)
            logger.warning("star coordinator: unknown phase %r, routing to END", current_phase)
            return _ROUTE_END

        # ----------------------------------------------------------------
        # Agent node wrappers
        # ----------------------------------------------------------------

        planner_agent = agents.get("planner")
        executor_agent = agents.get("executor")
        critic_agent = agents.get("critic")

        async def planner_node(state: GraphState) -> dict[str, Any]:
            if planner_agent is None:
                return {}
            result: dict[str, Any] = await planner_agent.step(state)
            return result

        async def executor_node(state: GraphState) -> dict[str, Any]:
            if executor_agent is None:
                return {}
            result: dict[str, Any] = await executor_agent.step(state)
            return result

        async def critic_node(state: GraphState) -> dict[str, Any]:
            if critic_agent is None:
                return {}
            result: dict[str, Any] = await critic_agent.step(state)
            return result

        # ----------------------------------------------------------------
        # Assemble the graph
        # ----------------------------------------------------------------

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        # Add nodes
        graph.add_node("coordinator", coordinator_node)
        graph.add_node("planner", planner_node)
        graph.add_node("executor", executor_node)
        graph.add_node("critic", critic_node)
        graph.add_node("critic_postprocess", _critic_postprocess)

        # Entry point
        graph.add_edge(START, "coordinator")

        # Coordinator → conditional routing
        graph.add_conditional_edges(
            "coordinator",
            _route_from_coord,
            {
                _ROUTE_PLANNER: "planner",
                _ROUTE_EXECUTOR: "executor",
                _ROUTE_CRITIC: "critic",
                _ROUTE_END: END,
            },
        )

        # Agent → coordinator edges
        graph.add_edge("planner", "coordinator")
        graph.add_edge("executor", "coordinator")

        # Critic pipeline: critic → critic_postprocess → coordinator
        graph.add_edge("critic", "critic_postprocess")
        graph.add_edge("critic_postprocess", "coordinator")

        return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helper: extract final_answer from state
# ---------------------------------------------------------------------------


def _extract_final_answer(state: GraphState) -> str:
    """Extract final_answer from the last Executor DRAFT message in state.

    Searches:
      1. state["agents"]["executor"]["outbox"] for last DRAFT message
      2. state["messages"] for last DRAFT message from executor
      3. Fallback: "<incomplete>"

    Args:
        state: Current GraphState dict.

    Returns:
        The extracted content string or "<incomplete>".
    """
    # Strategy 1: check executor outbox
    agents: dict[str, Any] = dict(state.get("agents") or {})
    executor_state: dict[str, Any] = dict(agents.get("executor") or {})
    outbox: list[Any] = list(executor_state.get("outbox") or [])

    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DRAFT or str(kind) == "draft":
            content = getattr(msg, "content", None)
            if content:
                return str(content)

    # Strategy 2: search global messages
    messages: list[Any] = list(state.get("messages") or [])
    for msg in reversed(messages):
        sender = getattr(msg, "sender", "")
        kind = getattr(msg, "kind", None)
        if (sender == "executor") and (kind == MessageKind.DRAFT or str(kind) == "draft"):
            content = getattr(msg, "content", None)
            if content:
                return str(content)

    return "<incomplete>"
