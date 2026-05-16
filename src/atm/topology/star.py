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

With HITL enabled (human_cfg.enabled=True, default role=reviewer):
  After critic_postprocess, a human_reviewer node is inserted before routing.
  Reviewer mode (default): human reviews critic output; approve → continue, reject → loop back.

  Coordinator-override mode (extra["override_coordinator"]=True):
    DEFERRED to M9.2 — only reviewer mode is wired up in M9.1.
    ``extra.override_coordinator`` and ``signals["human_phase_override"]`` reader in
    _route_from_coord exist as scaffolding for M9.2.  No built-in code path currently
    writes ``human_phase_override`` from inside the topology.

  _route_from_coord FIRST checks signals["human_phase_override"] as scaffolding for M9.2;
  # known-limitation: no M9.1 code path writes this signal — it will only become
  # functional when the coordinator-override custom apply_decision is wired in M9.2.
  Falls through to normal phase-based routing if signal is absent/None.

Nodes (HITL disabled — back-compat):
  coordinator, planner, executor, critic, critic_postprocess

Nodes (HITL enabled):
  coordinator, planner, executor, critic, critic_postprocess, human_reviewer

Edges:
  START → coordinator
  coordinator → conditional via _route_from_coord
  planner → coordinator
  executor → coordinator
  critic → critic_postprocess → [human_reviewer →] coordinator

TopologyConfig.extra defaults (under namespaced extras.star):
  planning_max_iter: 2   — mirrors ``_DEFAULT_PLANNING_MAX_ITER``
  exec_max_iter: 5       — mirrors ``_DEFAULT_EXEC_MAX_ITER``
  verify_max_iter: 3     — mirrors ``_DEFAULT_VERIFY_MAX_ITER``

  Legacy flat keys (e.g. ``extra: {planning_max_iter: 2}``) are auto-remapped
  to ``extras.star.*`` with a ``DeprecationWarning`` by the bw-compat validator
  in ``atm.experiment.config.TopologyCfg``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState
from atm.core.types import MessageKind, Phase
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop, get_topology_extras

if TYPE_CHECKING:
    from atm.experiment.config import HumanCfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy imports for HITL — patchable in tests
# ---------------------------------------------------------------------------

LLMSimulatedGateway: Any
CLIGateway: Any
build_human_node_factory: Any

try:
    from atm.human.llm_simulated import LLMSimulatedGateway as LLMSimulatedGateway
except ImportError:  # pragma: no cover
    LLMSimulatedGateway = None

try:
    from atm.human.cli_gateway import CLIGateway as CLIGateway
except ImportError:  # pragma: no cover
    CLIGateway = None

try:
    from atm.human._node_factory import build_human_node_factory as build_human_node_factory
except ImportError:  # pragma: no cover
    build_human_node_factory = None

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
            "critic_postprocess: no DECISION message found in critic outbox; treating as rejected"
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
                      human_cfg: HumanCfg | None — HITL configuration (M9.1).
                      human_gateway_llm: LLM wrapper for LLMSimulatedGateway.

        Returns:
            CompiledStateGraph ready for ainvoke.
        """
        checkpointer = kwargs.get("checkpointer")

        # Extract phase caps from per-topology extras bucket with defaults
        extras = get_topology_extras(cfg, "star")
        planning_max_iter: int = int(extras.get("planning_max_iter", _DEFAULT_PLANNING_MAX_ITER))
        exec_max_iter: int = int(extras.get("exec_max_iter", _DEFAULT_EXEC_MAX_ITER))
        verify_max_iter: int = int(extras.get("verify_max_iter", _DEFAULT_VERIFY_MAX_ITER))

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

            # Always populate final_answer with whatever the executor has
            # produced so far. The coordinator runs every tick, so this keeps
            # shared.final_answer current right up to the moment _should_stop
            # cuts the run off (global iter cap, budget exceeded). Without
            # this, hitting the cap before the verification→done branch
            # leaves the field empty and the evaluator scores 0 even on
            # otherwise-correct work. The verification→done branch above is
            # still authoritative for "approved" runs.
            if not shared.get("final_answer"):
                _provisional = _extract_final_answer(state)
                if _provisional and _provisional != "<incomplete>":
                    shared["final_answer"] = _provisional

            shared["signals"] = signals
            return {"shared": shared}

        # ----------------------------------------------------------------
        # Build _route_from_coord (closure captures cfg)
        # ----------------------------------------------------------------

        def _route_from_coord(state: GraphState) -> str:
            """Routing function — returns next node name after coordinator.

            FIRST checks signals["human_phase_override"] (set by human_coordinator mode).
            If set to one of {"advance","stay","finalize"}, forces the routing decision:
              - "advance" → next phase (executor/critic based on current phase + 1)
              - "stay"    → keep current phase (same routing as normal)
              - "finalize" → END
            After reading, clears: shared.signals["human_phase_override"] = None
            (mutates state in-place so the next coordinator tick sees it cleared).
            Falls through to normal phase-based routing if unset/None.
            """
            # Access shared directly (mutable reference) to allow signal clearing
            shared_raw: dict[str, Any] = cast(dict[str, Any], state).get("shared") or {}
            signals_raw: dict[str, Any] = shared_raw.get("signals") or {}
            current_phase = shared_raw.get("phase") or Phase.PLANNING

            # ---- Human phase override (coordinator mode scaffolding) ----
            # known-limitation (M9.2 deferral): coordinator-override mode is not
            # implemented in M9.1.  No built-in code path writes this signal.
            # The read+clear logic below is scaffolding for M9.2 where a custom
            # apply_decision callback will populate human_phase_override.
            override = signals_raw.get("human_phase_override")
            if override is not None and override in ("advance", "stay", "finalize"):
                # Clear the signal in the original signals dict (consumed)
                signals_raw["human_phase_override"] = None

                if override == "finalize":
                    return _ROUTE_END

                if override == "advance":
                    phase_str = str(current_phase)
                    if phase_str == "planning":
                        return _ROUTE_EXECUTOR
                    if phase_str == "execution":
                        return _ROUTE_CRITIC
                    # verification or done → end
                    return _ROUTE_END

                # override == "stay" → fall through to normal routing below

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
        # HITL configuration — extract from kwargs
        # ----------------------------------------------------------------

        human_cfg: HumanCfg | None = kwargs.get("human_cfg")
        hitl_enabled = human_cfg is not None and human_cfg.enabled

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

        # Critic pipeline: critic → critic_postprocess → [human_reviewer →] coordinator
        graph.add_edge("critic", "critic_postprocess")

        if hitl_enabled:
            assert human_cfg is not None  # narrowing for mypy
            # known-limitation (M9.2 deferral): extra.override_coordinator is read here
            # as scaffolding but coordinator-override mode is not implemented in M9.1.
            # When M9.2 is implemented, pass a custom apply_decision to
            # build_human_node_factory that maps action → signals["human_phase_override"].
            # _override_coordinator = bool((human_cfg.extra or {}).get("override_coordinator"))

            # Build gateway
            gateway_llm: Any = kwargs.get("human_gateway_llm")
            if human_cfg.gateway == "cli":
                gateway_instance: Any = CLIGateway() if CLIGateway is not None else None
            else:
                # default: llm_simulated
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

            # Build human_reviewer node using build_human_node_factory (module-level import)
            def _star_question_extractor(state: dict[str, Any]) -> str:
                """Extract question from critic's latest DECISION message."""
                agents_s: dict[str, Any] = state.get("agents", {})
                critic_outbox: list[Any] = list((agents_s.get("critic") or {}).get("outbox", []))
                question = "Please review the critic's evaluation and decide to approve or reject."
                for msg in reversed(critic_outbox):
                    if getattr(msg, "kind", None) == MessageKind.DECISION:
                        q = getattr(msg, "content", None) or question
                        return str(q)
                return question

            role_router: Any = kwargs.get("role_router")
            human_reviewer_node = build_human_node_factory(
                topology_name="star",
                human_cfg=human_cfg,
                gateway=gateway_instance,
                request_id_template="star:{run_id}:{iter_total}:reviewer",
                question_extractor=_star_question_extractor,
                role_router=role_router,
            )

            graph.add_node("human_reviewer", human_reviewer_node)
            graph.add_edge("critic_postprocess", "human_reviewer")
            graph.add_edge("human_reviewer", "coordinator")
        else:
            # Default path (back-compat): critic_postprocess → coordinator
            graph.add_edge("critic_postprocess", "coordinator")

        return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helper: extract final_answer from state
# ---------------------------------------------------------------------------


def _extract_final_answer(state: GraphState) -> str:
    """Extract final_answer from the executor's output in state.

    Tool-using executors typically emit short DRAFT messages ("Solution
    written to solution.py") while the actual artifact lives inside a
    ``file_write`` tool call's ``args["content"]``. Strategy order
    therefore prefers concrete file artifacts over chatty DRAFT text:

      1. state["agents"]["executor"]["tool_calls"] — last ``file_write``
         (preferring solution.py / main.py, then any ``.py`` path).
      2. state["agents"]["executor"]["outbox"] for last DRAFT message
         (text-only executors, no tool use).
      3. state["messages"] for last DRAFT message from executor.
      4. Any non-python file artifact (last resort).
      5. Fallback: "<incomplete>".
    """
    agents: dict[str, Any] = dict(state.get("agents") or {})
    executor_state: dict[str, Any] = dict(agents.get("executor") or {})

    # Strategy 1: prefer a written file artifact (most accurate for tool-using
    # executors). Walk tool_calls in REVERSE so the most recent write wins;
    # within ties, prefer paths ending in solution.py / main.py / .py.
    # IMPORTANT: only consider writes whose ``ToolResult.ok`` is True. The
    # ``file_write`` tool defaults ``overwrite=False``, so a model that calls
    # it twice on the same path gets the SECOND call rejected — taking the
    # latest call blindly would return the rejected (often degenerate, e.g.
    # "# test") payload while the file on disk still holds the first write.
    tool_calls: list[Any] = list(executor_state.get("tool_calls") or [])
    tool_results: list[Any] = list(executor_state.get("tool_results") or [])
    ok_call_ids: set[Any] = {
        getattr(r, "call_id", None) for r in tool_results if getattr(r, "ok", False)
    }
    py_solution: str | None = None
    py_any: str | None = None
    any_file: str | None = None
    for tc in reversed(tool_calls):
        if getattr(tc, "tool_name", None) != "file_write":
            continue
        # Skip writes that the tool layer rejected (e.g. overwrite=False).
        # If tool_results is empty (e.g. older runs without result tracking)
        # treat absence as success to preserve back-compat.
        if tool_results and getattr(tc, "id", None) not in ok_call_ids:
            continue
        args = getattr(tc, "args", None) or {}
        content = args.get("content") if isinstance(args, dict) else None
        path = args.get("path") if isinstance(args, dict) else None
        if not content:
            continue
        if any_file is None:
            any_file = str(content)
        if isinstance(path, str) and path.endswith(".py"):
            if py_any is None:
                py_any = str(content)
            if py_solution is None and (path.endswith("solution.py") or path.endswith("main.py")):
                py_solution = str(content)
                break  # best-quality match — stop early
    if py_solution is not None:
        return py_solution
    if py_any is not None:
        return py_any

    # Strategy 2: check executor outbox for DRAFT
    outbox: list[Any] = list(executor_state.get("outbox") or [])
    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DRAFT or str(kind) == "draft":
            content = getattr(msg, "content", None)
            if content:
                return str(content)

    # Strategy 3: search global messages
    messages: list[Any] = list(state.get("messages") or [])
    for msg in reversed(messages):
        sender = getattr(msg, "sender", "")
        kind = getattr(msg, "kind", None)
        if (sender == "executor") and (kind == MessageKind.DRAFT or str(kind) == "draft"):
            content = getattr(msg, "content", None)
            if content:
                return str(content)

    # Strategy 4: any non-python file artifact (last resort)
    if any_file is not None:
        return any_file

    return "<incomplete>"
