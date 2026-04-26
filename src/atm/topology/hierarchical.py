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
"""

from __future__ import annotations

import json
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
             (or conditional routing within sub)

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
        # Build team subgraphs
        # ----------------------------------------------------------------
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

            # Check finalize signal — route to finalize node (NOT END) so final_answer is set
            if signals.get(finalize_signal):
                return _ROUTE_FINALIZE

            # Check global stop — if max_iterations exceeded, route to finalize to aggregate
            stop, _reason = _should_stop(
                dict(state),
                cfg,
                topology_success=False,
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
            """Route after team_b processing."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            iter_total: int = int(shared.get("iter_total") or 0)

            # If finalize signal is set → go to finalize node (sets final_answer)
            if signals.get(finalize_signal):
                return _ROUTE_FINALIZE

            # Max rounds exceeded → finalize
            if iter_total > max_rounds:
                return _ROUTE_FINALIZE

            # Global stop → finalize for graceful exit
            stop, _reason = _should_stop(
                dict(state),
                cfg,
                topology_success=False,
                topology_max_reached=False,
            )
            if stop:
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
            signals[finalize_signal] = True
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
        # If compiled subgraph direct embedding works, use it. Otherwise wrap.
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

        # team_b → after_team_b → conditional routing
        graph.add_edge(_ROUTE_TEAM_B, "after_team_b")
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
