"""ChainTopology — linear Planner → Executor → Critic pipeline with retry-loop.

Architecture (arch.md §6):
  Graph:  START → planner → executor → critic → critic_postprocess
          → conditional edge (→ END | → executor)

Chain does NOT use phases like Star. In M6, shared.phase is pinned to
"execution" for the entire run.

Stopping precedence (arch.md §7.1):
  1. max_iter (global):  iter_total >= cfg.max_iterations → END
  2. topology_success:   critic_approved=True            → END
  3. topology_max:       (unused in Chain for M6)
  4. continue:           → executor (retry loop)

NOTE: Steps 2 and 3 (star/chain) must NOT edit topology/__init__.py.
Registration is done via @TopologyRegistry.register("chain") side-effect.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from atm.core.types import MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# END sentinel — matches LangGraph's END constant value ("__end__").
# Using a module-level constant avoids a hard import of langgraph.graph
# in functions that are unit-tested independently.
# ---------------------------------------------------------------------------

CHAIN_END: str = "__end__"

# ---------------------------------------------------------------------------
# StateGraph — module-level reference for testability.
# Tests patch "atm.topology.chain.StateGraph" to inject a mock graph factory.
# At runtime, build() resolves the real StateGraph via _import_state_graph().
# ---------------------------------------------------------------------------
#
# We use a private import function to avoid a module-level try/except that
# confuses mypy. The module attribute is set to the resolved class at module
# load time (if langgraph.graph is available) or left as a placeholder for
# mocking. Since pyproject.toml doesn't include langgraph as a direct dep
# yet, the module-level import is guarded and the None case is handled in build().

_StateGraphCls: type | None  # forward declaration for type checker


def _import_state_graph() -> type | None:
    """Import StateGraph from langgraph.graph if available.

    Returns:
        The StateGraph class, or None if langgraph.graph is not installed.
    """
    try:
        from langgraph.graph import StateGraph as _StateGraph  # type: ignore[import-untyped]

        return _StateGraph  # type: ignore[no-any-return]
    except ImportError:  # pragma: no cover
        return None


_StateGraphCls = _import_state_graph()

# Module-level alias for test patching: patch("atm.topology.chain.StateGraph")
StateGraph: type | None = _StateGraphCls


# ---------------------------------------------------------------------------
# _critic_postprocess — adapter node
# ---------------------------------------------------------------------------


async def _critic_postprocess(state: dict[str, Any]) -> dict[str, Any]:
    """Parse the last DECISION message from critic.outbox and update shared state.

    Actions:
      1. Read last DECISION-kind message from state["agents"]["critic"]["outbox"].
      2. Parse payload["approved"] (bool). Malformed → approved=False + warning.
      3. Write shared.signals["critic_approved"].
      4. If approved=True:
           - Find last DRAFT message in state["agents"]["executor"]["outbox"].
           - Set shared["final_answer"] = msg.payload["draft"] or msg.content.
           - Fallback to "<incomplete>" if no DRAFT found.
      5. Pin shared["phase"] = "execution" (Chain M6 invariant).

    Returns:
        A state delta dict (only "shared" key) to be merged by LangGraph.
    """
    shared = dict(deepcopy(state.get("shared", {})))
    agents = state.get("agents", {})

    # Initialize signals dict if missing
    signals: dict[str, Any] = dict(shared.get("signals", {}))

    # --- Step 1 & 2: Find last DECISION message in critic outbox ---
    critic_state = agents.get("critic", {})
    critic_outbox: list[Any] = list(critic_state.get("outbox", []))

    approved = False
    for msg in reversed(critic_outbox):
        if getattr(msg, "kind", None) == MessageKind.DECISION:
            try:
                approved = bool(msg.payload["approved"])
            except (KeyError, TypeError, AttributeError):
                logger.warning(
                    "ChainTopology: malformed DECISION message — missing payload['approved']; "
                    "treating as approved=False. message=%s",
                    str(msg),
                )
                approved = False
            break

    # --- Step 3: Write signals ---
    signals["critic_approved"] = approved

    # --- Step 4: Populate final_answer if approved ---
    final_answer: str | None = None
    if approved:
        executor_state = agents.get("executor", {})
        executor_outbox: list[Any] = list(executor_state.get("outbox", []))

        draft_text: str | None = None
        for msg in reversed(executor_outbox):
            if getattr(msg, "kind", None) == MessageKind.DRAFT:
                # Prefer payload["draft"], fallback to msg.content
                raw_payload = getattr(msg, "payload", {}) or {}
                draft_text = raw_payload.get("draft") or getattr(msg, "content", None)
                break

        final_answer = draft_text if draft_text is not None else "<incomplete>"

    # --- Step 5: Pin phase to "execution" ---
    shared["phase"] = "execution"
    shared["signals"] = signals
    shared["final_answer"] = final_answer

    return {"shared": shared}


# ---------------------------------------------------------------------------
# _route_from_critic — conditional edge router
# ---------------------------------------------------------------------------


def _route_from_critic(state: dict[str, Any], cfg: TopologyConfig) -> str:
    """Decide whether to go to END or retry via executor.

    Side effects:
      - Increments state["shared"]["iter_total"] by 1.
      - Increments state["shared"]["iteration"] by 1.

    Routing logic (after counter increment):
      1. Run _should_stop:
           - If stop (max_iter or topology_success) → CHAIN_END.
      2. If critic_approved in signals → CHAIN_END.
      3. Otherwise → "executor" (retry loop; planner does NOT re-run).

    Returns:
        CHAIN_END ("__end__") or "executor".
    """
    shared = state["shared"]

    # Increment counters BEFORE evaluating stop conditions
    shared["iter_total"] = shared.get("iter_total", 0) + 1
    shared["iteration"] = shared.get("iteration", 0) + 1

    signals: dict[str, Any] = shared.get("signals", {})
    critic_approved: bool = bool(signals.get("critic_approved", False))

    # Evaluate stopping conditions (max_iter priority over topology_success)
    stop, _reason = _should_stop(
        state,
        cfg,
        topology_success=critic_approved,
        topology_max_reached=False,
    )

    if stop:
        return CHAIN_END

    return "executor"


# ---------------------------------------------------------------------------
# ChainTopology — main class
# ---------------------------------------------------------------------------


@TopologyRegistry.register("chain")
class ChainTopology:
    """Linear topology: Planner → Executor → Critic → (retry | END).

    Graph structure:
      START → planner → executor → critic → critic_postprocess
      critic_postprocess → conditional: approved/max → END | → executor

    M6 constraints:
      - shared.phase pinned to "execution" (no phase transitions).
      - max_iterations from TopologyConfig.max_iterations.
      - TopologyConfig.extra is empty for Chain in M6.
    """

    name: str = "chain"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Build and compile the Chain LangGraph.

        Args:
            agents:      Dict mapping agent_id → Agent instance.
                         Expected keys: "planner", "executor", "critic".
            cfg:         TopologyConfig with max_iterations.
            **kwargs:    Passed to compile() (e.g. checkpointer=...).

        Returns:
            A compiled LangGraph CompiledStateGraph.

        Raises:
            ImportError: If langgraph.graph is not installed.
            KeyError: If "planner", "executor", or "critic" missing from agents.
        """
        # Resolve StateGraph — may be mocked in tests
        sg_cls = StateGraph
        if sg_cls is None:  # pragma: no cover
            raise ImportError(
                "langgraph.graph is required for ChainTopology.build(). "
                "Install: uv add langgraph"
            )

        planner = agents["planner"]
        executor = agents["executor"]
        critic = agents["critic"]

        # Build the graph — GraphState is passed as state schema
        from atm.core.state import GraphState

        graph: Any = sg_cls(GraphState)

        # --- Node: planner ---
        graph.add_node("planner", planner.step)

        # --- Node: executor ---
        graph.add_node("executor", executor.step)

        # --- Node: critic ---
        graph.add_node("critic", critic.step)

        # --- Node: critic_postprocess ---
        graph.add_node("critic_postprocess", _critic_postprocess)

        # --- Edges: linear chain ---
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "critic")
        graph.add_edge("critic", "critic_postprocess")

        # --- Conditional edge: postprocess → (END | executor) ---
        def _route(state: dict[str, Any]) -> str:
            return _route_from_critic(state, cfg)

        graph.add_conditional_edges(
            "critic_postprocess",
            _route,
            {
                "executor": "executor",
                CHAIN_END: CHAIN_END,
            },
        )

        # --- Entry point ---
        graph.set_entry_point("planner")

        # --- Compile ---
        checkpointer = kwargs.get("checkpointer")
        return graph.compile(checkpointer=checkpointer)
