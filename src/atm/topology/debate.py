"""DebateTopology — parallel Debater fan-out with Critic-as-judge loop.

Architecture (arch.md §7.5):
  Graph:
    START → planner
    planner → debater_pro   (parallel fan-out — same super-step)
    planner → debater_contra
    debater_pro   → judge
    debater_contra → judge   (fan-in — both must complete before judge runs)
    judge → judge_postprocess
    judge_postprocess → conditional:
      approved OR max_rounds → END
      else                   → debate_round_start → debater_pro + debater_contra (loop)

Parallel fan-out semantics:
  LangGraph automatically creates a concurrent super-step when multiple edges
  leave the same node. Because the debater count is fixed at 2, Send() API is
  not needed. Each debater writes only to its own agents[agent_id] outbox to
  avoid shared-write collisions.

Stopping precedence (arch.md §7.1):
  1. max_iter (global)   — iter_total >= cfg.max_iterations → END
  2. topology_success    — judge approved=True             → END
  3. topology_max        — debate_round >= max_rounds      → END
  4. continue            — loop back to debaters

TopologyConfig.extra defaults:
  max_rounds: 4          — maximum debate rounds before forced END
  debater_pro_id:   required in agents dict (any key)
  debater_contra_id: required in agents dict (any key, must differ from pro)
  judge_id:         required in agents dict

Nodes:
  planner, debater_pro, debater_contra,
  judge, judge_postprocess, debate_round_start

Invariants:
  - build() raises ValueError if debater_pro_id == debater_contra_id
  - judge_postprocess treats malformed DECISION as approved=False
  - Debaters write only to agents[their_id] outbox (not shared)
  - Message.id uniqueness guaranteed by uuid4 default_factory

Registration:
  @TopologyRegistry.register("debate") — side-effect on import
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

_DEFAULT_MAX_ROUNDS: int = 4

# Route sentinels
_ROUTE_LOOP = "debate_round_start"
_ROUTE_END = "__end__"


# ---------------------------------------------------------------------------
# judge_postprocess — adapter node
# ---------------------------------------------------------------------------


async def _judge_postprocess(
    state: GraphState,
    *,
    judge_id: str,
    debater_pro_id: str,
    debater_contra_id: str,
    cfg: TopologyConfig,
) -> dict[str, Any]:
    """Parse the last DECISION message from judge's outbox.

    Actions:
      1. Find last DECISION-kind message in agents[judge_id]["outbox"].
      2. Parse payload["approved"] (bool) and payload["winner"] ("pro"|"contra").
         Malformed → approved=False + WARNING.
      3. If approved=True:
           - Set shared.signals["judge_decided"] = True
           - Set shared.signals["debate_winner"] = winner ("pro" or "contra")
           - Extract final_answer from winning debater's DRAFT message.
      4. Increment shared["debate_round"] by 1.

    Args:
        state: GraphState dict.
        judge_id: Agent id of the judge.
        debater_pro_id: Agent id of pro debater.
        debater_contra_id: Agent id of contra debater.
        cfg: TopologyConfig (unused here, passed for consistency).

    Returns:
        State delta dict.
    """
    agents: dict[str, Any] = dict(state.get("agents") or {})
    judge_state: dict[str, Any] = dict(agents.get(judge_id) or {})
    outbox: list[Any] = list(judge_state.get("outbox") or [])

    # Find last DECISION message
    decision_msg = None
    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DECISION or str(kind) == "decision":
            decision_msg = msg
            break

    approved: bool = False
    winner: str = ""

    if decision_msg is None:
        logger.warning(
            "judge_postprocess: no DECISION message in judge outbox; treating as rejected"
        )
    else:
        payload: dict[str, Any] = getattr(decision_msg, "payload", {}) or {}
        if "approved" not in payload:
            logger.warning(
                "judge_postprocess: DECISION message missing 'approved' key; "
                "treating as rejected"
            )
            approved = False
        else:
            try:
                approved = bool(payload["approved"])
            except (TypeError, ValueError):
                logger.warning(
                    "judge_postprocess: could not parse payload['approved'] as bool; "
                    "treating as rejected"
                )
                approved = False

        if approved:
            raw_winner = payload.get("winner", "pro")
            winner = str(raw_winner) if raw_winner else "pro"

    # Build shared delta
    existing_shared: dict[str, Any] = dict(state.get("shared") or {})
    existing_signals: dict[str, Any] = dict(existing_shared.get("signals") or {})

    # Increment debate round counter
    debate_round: int = int(existing_shared.get("debate_round") or 0) + 1
    existing_shared["debate_round"] = debate_round

    if approved:
        existing_signals["judge_decided"] = True
        existing_signals["debate_winner"] = winner

        # Extract final_answer from winning debater's DRAFT
        winner_id = debater_pro_id if winner == "pro" else debater_contra_id
        final_answer = _extract_draft(agents, winner_id)
        existing_shared["final_answer"] = final_answer

    existing_shared["signals"] = existing_signals
    return {"shared": existing_shared}


# ---------------------------------------------------------------------------
# _route_from_judge — conditional edge router
# ---------------------------------------------------------------------------


def _route_from_judge(
    state: GraphState,
    cfg: TopologyConfig,
    max_rounds: int,
) -> str:
    """Route after judge_postprocess: END or loop back to debaters.

    Stopping precedence (arch.md §7.1):
      1. max_iter  — iter_total >= cfg.max_iterations → END
      2. approved  — judge_decided=True               → END  (topology_success)
      3. max_rounds — debate_round >= max_rounds       → END  (topology_max)
      4. continue  — loop

    Returns:
        _ROUTE_END or _ROUTE_LOOP.
    """
    shared: dict[str, Any] = dict(state.get("shared") or {})
    signals: dict[str, Any] = dict(shared.get("signals") or {})

    judge_decided: bool = bool(signals.get("judge_decided", False))
    debate_round: int = int(shared.get("debate_round") or 0)
    topology_max_reached: bool = debate_round >= max_rounds

    stop, _reason = _should_stop(
        dict(state),
        cfg,
        topology_success=judge_decided,
        topology_max_reached=topology_max_reached,
    )

    if stop:
        return _ROUTE_END

    return _ROUTE_LOOP


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_draft(agents: dict[str, Any], agent_id: str) -> str:
    """Extract last DRAFT message content from agent's outbox.

    Args:
        agents: agents dict from GraphState.
        agent_id: Agent id to look up.

    Returns:
        Content string or "<incomplete>".
    """
    agent_state: dict[str, Any] = dict(agents.get(agent_id) or {})
    outbox: list[Any] = list(agent_state.get("outbox") or [])

    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DRAFT or str(kind) == "draft":
            content = getattr(msg, "content", None)
            if content:
                return str(content)

    return "<incomplete>"


# ---------------------------------------------------------------------------
# DebateTopology
# ---------------------------------------------------------------------------


@TopologyRegistry.register("debate")
class DebateTopology:
    """Parallel Debater fan-out topology with Critic-as-judge loop.

    Graph structure:
      START → planner → [debater_pro, debater_contra] (parallel)
      debater_pro, debater_contra → judge → judge_postprocess
      judge_postprocess → conditional:
        approved / max_rounds → END
        else                  → debate_round_start → [debater_pro, debater_contra]

    Invariants:
      - debater_pro_id != debater_contra_id (ValueError otherwise)
      - judge_postprocess treats malformed DECISION as rejected
      - Debaters write only to their own agents outbox

    TopologyConfig.extra:
      max_rounds:       int  (default 4)
      debater_pro_id:   str  (required — key in agents dict)
      debater_contra_id: str (required — key in agents dict)
      judge_id:         str  (required — key in agents dict)
    """

    name = "debate"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a CompiledStateGraph.

        Args:
            agents: Dict mapping agent_id → Agent instance with async .step() method.
                    Must contain debater_pro_id, debater_contra_id, and judge_id keys.
            cfg:    TopologyConfig with max_iterations and extra.max_rounds.
            **kwargs: Optional; checkpointer=... forwarded to graph.compile().

        Returns:
            CompiledStateGraph ready for ainvoke.

        Raises:
            ValueError: If debater_pro_id == debater_contra_id.
        """
        checkpointer = kwargs.get("checkpointer")
        extra: dict[str, Any] = cfg.extra or {}

        max_rounds: int = int(extra.get("max_rounds", _DEFAULT_MAX_ROUNDS))
        debater_pro_id: str = str(extra.get("debater_pro_id", "debater_pro"))
        debater_contra_id: str = str(extra.get("debater_contra_id", "debater_contra"))
        judge_id: str = str(extra.get("judge_id", "judge"))

        # Invariant: pro and contra must be different agents
        if debater_pro_id == debater_contra_id:
            raise ValueError(
                f"debater_pro_id and debater_contra_id must be different; "
                f"both are {debater_pro_id!r}"
            )

        # Resolve agent instances
        planner_agent = agents.get("planner")
        debater_pro_agent = agents.get(debater_pro_id)
        debater_contra_agent = agents.get(debater_contra_id)
        judge_agent = agents.get(judge_id)

        # ----------------------------------------------------------------
        # Node: planner — opening statement
        # ----------------------------------------------------------------

        async def planner_node(state: GraphState) -> dict[str, Any]:
            """Planner node: produces opening statement for the debate."""
            if planner_agent is None:
                return {}
            result: dict[str, Any] = await planner_agent.step(state)
            # Increment iter_total on each planner call
            shared: dict[str, Any] = dict((result.get("shared") or state.get("shared")) or {})
            shared["iter_total"] = int(shared.get("iter_total") or 0) + 1
            result["shared"] = shared
            return result

        # ----------------------------------------------------------------
        # Node: debater_pro — argues for "pro" stance
        # ----------------------------------------------------------------

        async def debater_pro_node(state: GraphState) -> dict[str, Any]:
            """Debater pro node: writes only to its own outbox."""
            if debater_pro_agent is None:
                return {}
            result: dict[str, Any] = await debater_pro_agent.step(state)
            return result

        # ----------------------------------------------------------------
        # Node: debater_contra — argues for "contra" stance
        # ----------------------------------------------------------------

        async def debater_contra_node(state: GraphState) -> dict[str, Any]:
            """Debater contra node: writes only to its own outbox."""
            if debater_contra_agent is None:
                return {}
            result: dict[str, Any] = await debater_contra_agent.step(state)
            return result

        # ----------------------------------------------------------------
        # Node: judge — Critic-as-judge, evaluates debate
        # ----------------------------------------------------------------

        async def judge_node(state: GraphState) -> dict[str, Any]:
            """Judge node: evaluates arguments from both debaters."""
            if judge_agent is None:
                return {}
            result: dict[str, Any] = await judge_agent.step(state)
            return result

        # ----------------------------------------------------------------
        # Node: judge_postprocess — parses judge decision, emits signals
        # ----------------------------------------------------------------

        async def judge_postprocess_node(state: GraphState) -> dict[str, Any]:
            """Judge postprocess: parse DECISION, emit signals, set final_answer."""
            return await _judge_postprocess(
                state,
                judge_id=judge_id,
                debater_pro_id=debater_pro_id,
                debater_contra_id=debater_contra_id,
                cfg=cfg,
            )

        # ----------------------------------------------------------------
        # Node: debate_round_start — no-op entry for loop continuation
        # ----------------------------------------------------------------

        async def debate_round_start_node(state: GraphState) -> dict[str, Any]:
            """No-op node that serves as loop re-entry point."""
            return {}

        # ----------------------------------------------------------------
        # Routing function
        # ----------------------------------------------------------------

        def _route(state: GraphState) -> str:
            return _route_from_judge(state, cfg, max_rounds)

        # ----------------------------------------------------------------
        # Assemble the graph
        # ----------------------------------------------------------------

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        # Add nodes
        graph.add_node("planner", planner_node)
        graph.add_node("debater_pro", debater_pro_node)
        graph.add_node("debater_contra", debater_contra_node)
        graph.add_node("judge", judge_node)
        graph.add_node("judge_postprocess", judge_postprocess_node)
        graph.add_node("debate_round_start", debate_round_start_node)

        # Entry point
        graph.add_edge(START, "planner")

        # Parallel fan-out: planner → both debaters (same super-step)
        graph.add_edge("planner", "debater_pro")
        graph.add_edge("planner", "debater_contra")

        # Fan-in: both debaters → judge
        graph.add_edge("debater_pro", "judge")
        graph.add_edge("debater_contra", "judge")

        # Judge pipeline
        graph.add_edge("judge", "judge_postprocess")

        # Conditional routing from postprocess
        graph.add_conditional_edges(
            "judge_postprocess",
            _route,
            {
                _ROUTE_END: END,
                _ROUTE_LOOP: "debate_round_start",
            },
        )

        # Loop: round_start → both debaters again
        graph.add_edge("debate_round_start", "debater_pro")
        graph.add_edge("debate_round_start", "debater_contra")

        return graph.compile(checkpointer=checkpointer)
