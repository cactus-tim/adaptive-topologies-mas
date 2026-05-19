"""StarTopology — coordinator-centered LangGraph with phase-advance semantics."""

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

_DEFAULT_PLANNING_MAX_ITER = 2
_DEFAULT_EXEC_MAX_ITER = 5
_DEFAULT_VERIFY_MAX_ITER = 3

_ROUTE_PLANNER = "planner"
_ROUTE_EXECUTOR = "executor"
_ROUTE_CRITIC = "critic"
_ROUTE_END = "__end__"


async def _critic_postprocess(state: GraphState) -> dict[str, Any]:
    """Parse critic's last DECISION and update signals['critic_approved']."""
    agents: dict[str, Any] = dict(state.get("agents") or {})
    critic_state: dict[str, Any] = dict(agents.get("critic") or {})
    outbox: list[Any] = list(critic_state.get("outbox") or [])

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

    existing_shared: dict[str, Any] = dict(state.get("shared") or {})
    existing_signals: dict[str, Any] = dict(existing_shared.get("signals") or {})
    existing_signals["critic_approved"] = approved
    existing_shared["signals"] = existing_signals

    return {"shared": existing_shared}


@TopologyRegistry.register("star")
class StarTopology:
    """Coordinator-centered topology: START → coordinator → {planner|executor|critic} → loop."""

    name = "star"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Build and compile the Star LangGraph."""
        checkpointer = kwargs.get("checkpointer")

        extras = get_topology_extras(cfg, "star")
        planning_max_iter: int = int(extras.get("planning_max_iter", _DEFAULT_PLANNING_MAX_ITER))
        exec_max_iter: int = int(extras.get("exec_max_iter", _DEFAULT_EXEC_MAX_ITER))
        verify_max_iter: int = int(extras.get("verify_max_iter", _DEFAULT_VERIFY_MAX_ITER))

        async def coordinator_node(state: GraphState) -> dict[str, Any]:
            """Rule-based coordinator: increments counters + advances phase."""
            shared: dict[str, Any] = dict(state.get("shared") or {})

            old_iter_total: int = int(shared.get("iter_total") or 0)
            new_iter_total = old_iter_total + 1
            new_iteration = int(shared.get("iteration") or 0) + 1
            shared["iter_total"] = new_iter_total
            shared["iteration"] = new_iteration

            phase_started_at: int = int(shared.get("phase_started_at_iter") or 0)
            iter_within_phase = new_iter_total - phase_started_at
            current_phase = shared.get("phase") or Phase.PLANNING
            current_phase_str = str(current_phase)

            signals: dict[str, Any] = dict(shared.get("signals") or {})
            phase_history: list[Any] = list(shared.get("phase_history") or [])

            if current_phase_str == "planning":
                if signals.get("ready_for_execution") or iter_within_phase >= planning_max_iter:
                    phase_history = list(phase_history)
                    phase_history.append(Phase.PLANNING)
                    shared["phase"] = Phase.EXECUTION
                    shared["phase_started_at_iter"] = new_iter_total
                    shared["phase_history"] = phase_history

            elif current_phase_str == "execution":
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

                    final_answer = _extract_final_answer(state)
                    shared["final_answer"] = final_answer

            if not shared.get("final_answer"):
                _provisional = _extract_final_answer(state)
                if _provisional and _provisional != "<incomplete>":
                    shared["final_answer"] = _provisional

            shared["signals"] = signals
            return {"shared": shared}

        def _route_from_coord(state: GraphState) -> str:
            """Return next node name; checks human_phase_override signal first."""
            shared_raw: dict[str, Any] = cast(dict[str, Any], state).get("shared") or {}
            signals_raw: dict[str, Any] = shared_raw.get("signals") or {}
            current_phase = shared_raw.get("phase") or Phase.PLANNING

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
                    return _ROUTE_END

            stop, _reason = _should_stop(
                cast(dict[str, Any], state),
                cfg,
                topology_success=(str(current_phase) == "done"),
                topology_max_reached=False,
            )
            if stop:
                return _ROUTE_END

            phase_str = str(current_phase)

            if phase_str == "planning":
                return _ROUTE_PLANNER

            if phase_str == "execution":
                return _ROUTE_EXECUTOR

            if phase_str == "verification":
                return _ROUTE_CRITIC

            if phase_str == "done":
                return _ROUTE_END

            logger.warning("star coordinator: unknown phase %r, routing to END", current_phase)
            return _ROUTE_END

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

        human_cfg: HumanCfg | None = kwargs.get("human_cfg")
        hitl_enabled = human_cfg is not None and human_cfg.enabled

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        graph.add_node("coordinator", coordinator_node)
        graph.add_node("planner", planner_node)
        graph.add_node("executor", executor_node)
        graph.add_node("critic", critic_node)
        graph.add_node("critic_postprocess", _critic_postprocess)

        graph.add_edge(START, "coordinator")

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

        graph.add_edge("planner", "coordinator")
        graph.add_edge("executor", "coordinator")

        graph.add_edge("critic", "critic_postprocess")

        if hitl_enabled:
            assert human_cfg is not None  # narrowing for mypy

            gateway_llm: Any = kwargs.get("human_gateway_llm")
            if human_cfg.gateway == "cli":
                gateway_instance: Any = CLIGateway() if CLIGateway is not None else None
            else:
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
            graph.add_edge("critic_postprocess", "coordinator")

        return graph.compile(checkpointer=checkpointer)


_CODE_TASKS: set[str] = {"humaneval"}


def _extract_final_answer(state: GraphState) -> str:
    """Extract final_answer from executor output; code tasks prefer file artifact, others DRAFT."""
    agents: dict[str, Any] = dict(state.get("agents") or {})
    executor_state: dict[str, Any] = dict(agents.get("executor") or {})

    shared: dict[str, Any] = dict(state.get("shared") or {})
    task_id: str = str(shared.get("task_id") or "").lower()
    is_code_task: bool = task_id in _CODE_TASKS

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
                break
    file_artifact: str | None = py_solution or py_any

    draft: str | None = None
    outbox: list[Any] = list(executor_state.get("outbox") or [])
    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DRAFT or str(kind) == "draft":
            content = getattr(msg, "content", None)
            if content:
                draft = str(content)
                break
    if draft is None:
        messages: list[Any] = list(state.get("messages") or [])
        for msg in reversed(messages):
            sender = getattr(msg, "sender", "")
            kind = getattr(msg, "kind", None)
            if (sender == "executor") and (kind == MessageKind.DRAFT or str(kind) == "draft"):
                content = getattr(msg, "content", None)
                if content:
                    draft = str(content)
                    break

    if is_code_task:
        primary, secondary = file_artifact, draft
    else:
        primary, secondary = draft, file_artifact

    return primary or secondary or any_file or "<incomplete>"
