"""DebateTopology — parallel Debater fan-out with Critic-as-judge loop."""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState
from atm.core.types import HumanContext, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop, get_topology_extras

if TYPE_CHECKING:
    from atm.experiment.config import HumanCfg
    from atm.human.gateway import HumanGateway

logger = logging.getLogger(__name__)

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

_DEFAULT_MAX_ROUNDS: int = 4

_ROUTE_LOOP = "debate_round_start"
_ROUTE_END = "__end__"


async def _judge_postprocess(
    state: GraphState,
    *,
    judge_id: str,
    debater_pro_id: str,
    debater_contra_id: str,
    cfg: TopologyConfig,
) -> dict[str, Any]:
    """Parse last DECISION from judge's outbox; update signals, final_answer, debate_round."""
    agents: dict[str, Any] = dict(state.get("agents") or {})
    judge_state: dict[str, Any] = dict(agents.get(judge_id) or {})
    outbox: list[Any] = list(judge_state.get("outbox") or [])

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
                "judge_postprocess: DECISION message missing 'approved' key; treating as rejected"
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

    existing_shared: dict[str, Any] = dict(state.get("shared") or {})
    existing_signals: dict[str, Any] = dict(existing_shared.get("signals") or {})

    debate_round: int = int(existing_shared.get("debate_round") or 0) + 1
    existing_shared["debate_round"] = debate_round

    iter_total: int = int(existing_shared.get("iter_total") or 0) + 1
    existing_shared["iter_total"] = iter_total

    task_id: str = str(existing_shared.get("task_id") or "")

    if approved:
        existing_signals["judge_decided"] = True
        existing_signals["debate_winner"] = winner

        winner_id = debater_pro_id if winner == "pro" else debater_contra_id
        final_answer = _extract_winner_artifact(agents, winner_id, task_id=task_id)
        existing_shared["final_answer"] = final_answer
    else:
        existing_signals["judge_decided"] = False

        pro_answer = _extract_winner_artifact(agents, debater_pro_id, task_id=task_id)
        contra_answer = _extract_winner_artifact(agents, debater_contra_id, task_id=task_id)
        pro_len = len(pro_answer) if pro_answer != "<incomplete>" else 0
        contra_len = len(contra_answer) if contra_answer != "<incomplete>" else 0
        if pro_len == 0 and contra_len == 0:
            fallback = "<incomplete>"
        elif contra_len > pro_len:
            fallback = contra_answer
            existing_signals["debate_winner"] = "contra"
        else:
            fallback = pro_answer
            existing_signals["debate_winner"] = "pro"
        if not existing_shared.get("final_answer"):
            existing_shared["final_answer"] = fallback

    existing_shared["signals"] = existing_signals
    return {"shared": existing_shared}


def _route_from_judge(
    state: GraphState,
    cfg: TopologyConfig,
    max_rounds: int,
) -> str:
    """Return _ROUTE_END or _ROUTE_LOOP following stopping precedence."""
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


def _extract_draft(agents: dict[str, Any], agent_id: str) -> str:
    """Return last DRAFT message content from agent's outbox, or "<incomplete>"."""
    agent_state: dict[str, Any] = dict(agents.get(agent_id) or {})
    outbox: list[Any] = list(agent_state.get("outbox") or [])

    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DRAFT or str(kind) == "draft":
            content = getattr(msg, "content", None)
            if content:
                return str(content)

    return "<incomplete>"


_CODE_TASKS: set[str] = {"humaneval"}


def _extract_winner_artifact(
    agents: dict[str, Any],
    agent_id: str,
    task_id: str = "",
) -> str:
    """Return winner debater's primary artifact; task-aware (code→file_write, else DRAFT)."""
    agent_state: dict[str, Any] = dict(agents.get(agent_id) or {})
    is_code_task: bool = task_id.lower() in _CODE_TASKS

    tool_calls: list[Any] = list(agent_state.get("tool_calls") or [])
    tool_results: list[Any] = list(agent_state.get("tool_results") or [])
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
    raw_draft = _extract_draft(agents, agent_id)
    if raw_draft and raw_draft != "<incomplete>":
        import re as _re

        m = _re.search(r"###ANSWER###\s*\n(.*?)\n\s*###END###", raw_draft, _re.DOTALL)
        if m:
            extracted = m.group(1).strip()
            draft = extracted or raw_draft
        else:
            draft = raw_draft

    if is_code_task:
        primary, secondary = file_artifact, draft
    else:
        primary, secondary = draft, file_artifact

    return primary or secondary or any_file or "<incomplete>"


def _build_human_judge_node(
    human_cfg: HumanCfg,
    gateway: HumanGateway,
    *,
    judge_id: str,
    debater_pro_id: str,
    debater_contra_id: str,
    role_router: Any = None,
) -> Any:
    """Build async HITL node that synthesizes a DECISION message into agents[judge_id]["outbox"]."""
    _role_router = role_router

    async def human_judge_node(state: GraphState) -> dict[str, Any]:
        """HITL judge node — calls gateway and writes synthesized DECISION to judge outbox."""
        import uuid as _uuid_mod

        from atm.core.types import Message

        shared: dict[str, Any] = dict(deepcopy(state.get("shared") or {}))
        agents: dict[str, Any] = dict(state.get("agents") or {})

        _raw_run_id = shared.get("run_id") or state.get("run_id")
        run_id: _uuid_mod.UUID = (
            _raw_run_id
            if isinstance(_raw_run_id, _uuid_mod.UUID)
            else _uuid_mod.UUID(str(_raw_run_id))
            if _raw_run_id
            else _uuid_mod.uuid4()
        )
        iter_total: int = int(shared.get("iter_total", 0))
        request_id: str = f"debate:{iter_total}:judge"

        pro_draft = _extract_draft(agents, debater_pro_id)
        contra_draft = _extract_draft(agents, debater_contra_id)
        question = (
            f"Judge this debate. Pro argument: {pro_draft[:200]}. "
            f"Contra argument: {contra_draft[:200]}. "
            "Do you approve? Action: 'approve' to approve pro side, 'reject' otherwise."
        )

        from atm.core.types import Phase

        if _role_router is not None:
            _raw_phase = shared.get("phase", "execution")
            _phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            active_role = await _role_router.decide(_phase, shared)
        else:
            active_role = human_cfg.role

        ctx = HumanContext(
            run_id=run_id,
            role=active_role,
            question=question,
            recent_messages=tuple(state.get("messages", [])[-5:]),
            allowed_actions=("approve", "reject"),
            deadline_s=int(human_cfg.timeout_s) if human_cfg.timeout_s is not None else None,
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
                "debate.human_judge_node: adispatch human_request skipped (no callback ctx)",
                exc_info=True,
            )

        _t0 = time.monotonic()
        timeout_s_val: float | None = getattr(human_cfg, "timeout_s", None)
        policy: str = getattr(human_cfg, "timeout_policy", "skip")

        if request_with_timeout is not None and timeout_s_val is not None:
            _fallback_gateway: Any = None
            if policy == "llm_fallback" and LLMSimulatedGateway is not None:
                _fb_llm: Any = getattr(gateway, "_llm", None)
                _fallback_gateway = LLMSimulatedGateway(llm=_fb_llm)
            response = await request_with_timeout(
                gateway,
                ctx,
                request_id=request_id,
                timeout_s=timeout_s_val,
                policy=policy,
                llm_fallback_gateway=_fallback_gateway,
            )
        else:
            response = await gateway.request(ctx, request_id=request_id)

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
                "debate.human_judge_node: adispatch human_response skipped (no callback ctx)",
                exc_info=True,
            )

        action: str = getattr(response, "action", "") or ""
        comment: str = getattr(response, "comment", None) or ""

        if action == "approve":
            _payload = getattr(response, "payload", {}) or {}
            human_winner: str = str(_payload.get("winner", "pro"))
            approved: bool = True
        else:
            human_winner = ""
            approved = False

        decision_msg = Message(
            sender="human_judge",
            kind=MessageKind.DECISION,
            content="APPROVE" if approved else "REJECT",
            payload={
                "approved": approved,
                "winner": human_winner,
                "comment": comment,
                "source": "human",
            },
        )

        judge_agent_state: dict[str, Any] = dict(agents.get(judge_id) or {})
        judge_outbox: list[Any] = list(judge_agent_state.get("outbox") or [])
        judge_outbox.append(decision_msg)
        judge_agent_state["outbox"] = judge_outbox

        new_agents: dict[str, Any] = dict(agents)
        new_agents[judge_id] = judge_agent_state

        logger.debug(
            "debate.human_judge_node: action=%r approved=%r iter_total=%d",
            action,
            approved,
            iter_total,
        )

        return {"agents": new_agents}

    return human_judge_node


@TopologyRegistry.register("debate")
class DebateTopology:
    """Parallel Debater fan-out topology with Critic-as-judge loop (modes: critic/human/both)."""

    name = "debate"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a CompiledStateGraph; raises ValueError if pro==contra agent."""
        checkpointer = kwargs.get("checkpointer")
        human_cfg: HumanCfg | None = kwargs.get("human_cfg")
        role_router: Any = kwargs.get("role_router")

        gateway: HumanGateway | None = kwargs.get("gateway")
        if gateway is None and human_cfg is not None and getattr(human_cfg, "enabled", False):
            _gateway_llm: Any = kwargs.get("human_gateway_llm")
            if getattr(human_cfg, "gateway", "llm_simulated") == "cli":
                gateway = CLIGateway() if CLIGateway is not None else None
            else:
                gateway = (
                    LLMSimulatedGateway(llm=_gateway_llm)
                    if LLMSimulatedGateway is not None
                    else None
                )

        extras: dict[str, Any] = get_topology_extras(cfg, "debate")

        max_rounds: int = int(extras.get("max_rounds", _DEFAULT_MAX_ROUNDS))
        debater_pro_id: str = str(extras.get("debater_pro_id", "debater_pro"))
        debater_contra_id: str = str(extras.get("debater_contra_id", "debater_contra"))
        judge_id: str = str(extras.get("judge_id", "judge"))

        judge_mode: str = "critic"
        if human_cfg is not None and getattr(human_cfg, "enabled", False):
            human_extra: dict[str, Any] = getattr(human_cfg, "extra", None) or {}
            judge_mode = str(human_extra.get("judge", "critic"))

        if debater_pro_id == debater_contra_id:
            raise ValueError(
                f"debater_pro_id and debater_contra_id must be different; "
                f"both are {debater_pro_id!r}"
            )

        planner_agent = agents.get("planner")
        debater_pro_agent = agents.get(debater_pro_id)
        debater_contra_agent = agents.get(debater_contra_id)
        judge_agent = agents.get(judge_id)

        async def planner_node(state: GraphState) -> dict[str, Any]:
            """Planner node: opening statement for the debate."""
            if planner_agent is None:
                return {}
            result: dict[str, Any] = await planner_agent.step(state)
            shared: dict[str, Any] = dict((result.get("shared") or state.get("shared")) or {})
            shared["iter_total"] = int(shared.get("iter_total") or 0) + 1
            result["shared"] = shared
            return result

        async def debater_pro_node(state: GraphState) -> dict[str, Any]:
            """Debater pro node."""
            if debater_pro_agent is None:
                return {}
            result: dict[str, Any] = await debater_pro_agent.step(state)
            return result

        async def debater_contra_node(state: GraphState) -> dict[str, Any]:
            """Debater contra node."""
            if debater_contra_agent is None:
                return {}
            result: dict[str, Any] = await debater_contra_agent.step(state)
            return result

        async def debate_round_start_node(state: GraphState) -> dict[str, Any]:
            """No-op node that serves as loop re-entry point."""
            return {}

        def _route(state: GraphState) -> str:
            return _route_from_judge(state, cfg, max_rounds)

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        graph.add_node("planner", planner_node)
        graph.add_node("debater_pro", debater_pro_node)
        graph.add_node("debater_contra", debater_contra_node)
        graph.add_node("debate_round_start", debate_round_start_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "debater_pro")
        graph.add_edge("planner", "debater_contra")
        graph.add_edge("debate_round_start", "debater_pro")
        graph.add_edge("debate_round_start", "debater_contra")

        if judge_mode == "critic":

            async def judge_node(state: GraphState) -> dict[str, Any]:
                """Judge node."""
                if judge_agent is None:
                    return {}
                result: dict[str, Any] = await judge_agent.step(state)
                return result

            async def judge_postprocess_node(state: GraphState) -> dict[str, Any]:
                """Judge postprocess node."""
                return await _judge_postprocess(
                    state,
                    judge_id=judge_id,
                    debater_pro_id=debater_pro_id,
                    debater_contra_id=debater_contra_id,
                    cfg=cfg,
                )

            graph.add_node("judge", judge_node)
            graph.add_node("judge_postprocess", judge_postprocess_node)
            graph.add_edge("debater_pro", "judge")
            graph.add_edge("debater_contra", "judge")
            graph.add_edge("judge", "judge_postprocess")
            graph.add_conditional_edges(
                "judge_postprocess",
                _route,
                {
                    _ROUTE_END: END,
                    _ROUTE_LOOP: "debate_round_start",
                },
            )

        elif judge_mode == "human":
            assert human_cfg is not None and gateway is not None, (
                "judge mode 'human' requires human_cfg and gateway to be provided to build()"
            )

            human_judge = _build_human_judge_node(
                human_cfg,
                gateway,
                judge_id=judge_id,
                debater_pro_id=debater_pro_id,
                debater_contra_id=debater_contra_id,
                role_router=role_router,
            )

            async def judge_postprocess_node_human(state: GraphState) -> dict[str, Any]:
                """Judge postprocess node."""
                return await _judge_postprocess(
                    state,
                    judge_id=judge_id,
                    debater_pro_id=debater_pro_id,
                    debater_contra_id=debater_contra_id,
                    cfg=cfg,
                )

            graph.add_node("human_judge", human_judge)
            graph.add_node("judge_postprocess", judge_postprocess_node_human)
            graph.add_edge("debater_pro", "human_judge")
            graph.add_edge("debater_contra", "human_judge")
            graph.add_edge("human_judge", "judge_postprocess")
            graph.add_conditional_edges(
                "judge_postprocess",
                _route,
                {
                    _ROUTE_END: END,
                    _ROUTE_LOOP: "debate_round_start",
                },
            )

        elif judge_mode == "both":
            assert human_cfg is not None and gateway is not None, (
                "judge mode 'both' requires human_cfg and gateway to be provided to build()"
            )

            async def _both_judge_postprocess(state: GraphState) -> dict[str, Any]:
                """Sequential LLM+HITL judge: run critic, call gateway, human-override wins."""
                import uuid as _uuid_mod

                from atm.core.types import Message

                agents_state: dict[str, Any] = dict(state.get("agents") or {})
                shared_state: dict[str, Any] = dict(deepcopy(state.get("shared") or {}))

                llm_judge_result: dict[str, Any] = {}
                if judge_agent is not None:
                    llm_judge_result = await judge_agent.step(state)

                if llm_judge_result.get("agents"):
                    agents_state = dict(agents_state)
                    agents_state.update(llm_judge_result["agents"])

                judge_agent_state: dict[str, Any] = dict(agents_state.get(judge_id) or {})
                llm_outbox: list[Any] = list(judge_agent_state.get("outbox") or [])

                critic_approved: bool = False
                critic_winner: str = ""
                for msg in reversed(llm_outbox):
                    kind = getattr(msg, "kind", None)
                    if kind == MessageKind.DECISION or str(kind) == "decision":
                        _payload: dict[str, Any] = getattr(msg, "payload", {}) or {}
                        try:
                            critic_approved = bool(_payload.get("approved", False))
                        except (TypeError, ValueError):
                            critic_approved = False
                        if critic_approved:
                            critic_winner = str(_payload.get("winner", "pro"))
                        break

                _raw_run_id = shared_state.get("run_id") or state.get("run_id")
                run_id: _uuid_mod.UUID = (
                    _raw_run_id
                    if isinstance(_raw_run_id, _uuid_mod.UUID)
                    else _uuid_mod.UUID(str(_raw_run_id))
                    if _raw_run_id
                    else _uuid_mod.uuid4()
                )
                iter_total_val: int = int(shared_state.get("iter_total", 0))
                h_request_id: str = f"debate:{iter_total_val}:judge_combined"

                pro_draft = _extract_draft(agents_state, debater_pro_id)
                contra_draft = _extract_draft(agents_state, debater_contra_id)
                critic_verdict = "APPROVE" if critic_approved else "REJECT"
                question = (
                    f"[Debate Judge Review] Critic verdict: {critic_verdict}. "
                    f"Pro: {pro_draft[:150]}. Contra: {contra_draft[:150]}. "
                    "Override? 'approve' to finalize, 'reject' to continue debate."
                )

                from atm.core.types import Phase

                if role_router is not None:
                    _raw_phase = shared_state.get("phase", "execution")
                    _phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
                    active_role_both = await role_router.decide(_phase, shared_state)
                else:
                    active_role_both = human_cfg.role

                ctx = HumanContext(
                    run_id=run_id,
                    role=active_role_both,
                    question=question,
                    recent_messages=tuple(state.get("messages", [])[-5:]),
                    allowed_actions=("approve", "reject"),
                    deadline_s=int(human_cfg.timeout_s)
                    if human_cfg.timeout_s is not None
                    else None,
                )

                _requested_at = datetime.now(UTC)
                try:
                    await adispatch_custom_event(
                        "human_request",
                        {
                            "run_id": run_id,
                            "request_id": h_request_id,
                            "role": str(
                                active_role_both.value
                                if hasattr(active_role_both, "value")
                                else active_role_both
                            ),
                            "context_json": ctx.model_dump(mode="json"),
                            "requested_at": _requested_at,
                        },
                    )
                except Exception:
                    logger.debug(
                        "debate._both_judge_postprocess: adispatch human_request skipped",
                        exc_info=True,
                    )

                _t0 = time.monotonic()
                timeout_s_val: float | None = getattr(human_cfg, "timeout_s", None)
                h_policy: str = getattr(human_cfg, "timeout_policy", "skip")

                if request_with_timeout is not None and timeout_s_val is not None:
                    _h_fallback_gateway: Any = None
                    if h_policy == "llm_fallback" and LLMSimulatedGateway is not None:
                        _h_fb_llm: Any = getattr(gateway, "_llm", None)
                        _h_fallback_gateway = LLMSimulatedGateway(llm=_h_fb_llm)
                    h_response = await request_with_timeout(
                        gateway,
                        ctx,
                        request_id=h_request_id,
                        timeout_s=timeout_s_val,
                        policy=h_policy,
                        llm_fallback_gateway=_h_fallback_gateway,
                    )
                else:
                    h_response = await gateway.request(ctx, request_id=h_request_id)

                _latency_s = time.monotonic() - _t0

                try:
                    await adispatch_custom_event(
                        "human_response",
                        {
                            "run_id": run_id,
                            "request_id": h_request_id,
                            "answered_at": datetime.now(UTC),
                            "response_json": h_response.model_dump(mode="json"),
                            "source": getattr(h_response, "source", "human"),
                            "timed_out": getattr(h_response, "timed_out", False),
                            "latency_s": _latency_s,
                        },
                    )
                except Exception:
                    logger.debug(
                        "debate._both_judge_postprocess: adispatch human_response skipped",
                        exc_info=True,
                    )

                h_action: str = getattr(h_response, "action", "") or ""
                human_approved: bool = h_action == "approve"
                _h_payload: dict[str, Any] = getattr(h_response, "payload", {}) or {}
                human_winner: str = str(_h_payload.get("winner", "pro")) if human_approved else ""

                final_approved: bool = human_approved
                final_winner: str = human_winner if human_approved else critic_winner

                if human_approved != critic_approved:
                    logger.debug(
                        "debate._both_judge_postprocess: human overrides critic "
                        "(human=%r, critic=%r) → final=%r",
                        human_approved,
                        critic_approved,
                        final_approved,
                    )

                h_comment: str = getattr(h_response, "comment", None) or ""
                final_decision = Message(
                    sender="judge_combined",
                    kind=MessageKind.DECISION,
                    content="APPROVE" if final_approved else "REJECT",
                    payload={
                        "approved": final_approved,
                        "winner": final_winner,
                        "comment": h_comment,
                        "source": "both",
                        "critic_approved": critic_approved,
                        "human_approved": human_approved,
                    },
                )

                final_judge_state: dict[str, Any] = dict(agents_state.get(judge_id) or {})
                final_judge_outbox: list[Any] = list(final_judge_state.get("outbox") or [])
                final_judge_outbox.append(final_decision)
                final_judge_state["outbox"] = final_judge_outbox
                agents_state[judge_id] = final_judge_state

                synthetic_state = dict(state)
                synthetic_state["agents"] = agents_state
                synthetic_state["shared"] = shared_state

                postprocess_delta = await _judge_postprocess(
                    synthetic_state,  # type: ignore[arg-type]
                    judge_id=judge_id,
                    debater_pro_id=debater_pro_id,
                    debater_contra_id=debater_contra_id,
                    cfg=cfg,
                )

                return {
                    "agents": agents_state,
                    "shared": postprocess_delta["shared"],
                }

            graph.add_node("judge_combined", _both_judge_postprocess)
            graph.add_edge("debater_pro", "judge_combined")
            graph.add_edge("debater_contra", "judge_combined")
            graph.add_conditional_edges(
                "judge_combined",
                _route,
                {
                    _ROUTE_END: END,
                    _ROUTE_LOOP: "debate_round_start",
                },
            )

        else:
            logger.warning(
                "DebateTopology.build: unknown judge mode %r; falling back to 'critic'",
                judge_mode,
            )

            async def judge_node_fallback(state: GraphState) -> dict[str, Any]:
                """Judge node (fallback)."""
                if judge_agent is None:
                    return {}
                result: dict[str, Any] = await judge_agent.step(state)
                return result

            async def judge_postprocess_node_fallback(state: GraphState) -> dict[str, Any]:
                """Judge postprocess node (fallback)."""
                return await _judge_postprocess(
                    state,
                    judge_id=judge_id,
                    debater_pro_id=debater_pro_id,
                    debater_contra_id=debater_contra_id,
                    cfg=cfg,
                )

            graph.add_node("judge", judge_node_fallback)
            graph.add_node("judge_postprocess", judge_postprocess_node_fallback)

            graph.add_edge("debater_pro", "judge")
            graph.add_edge("debater_contra", "judge")
            graph.add_edge("judge", "judge_postprocess")

            graph.add_conditional_edges(
                "judge_postprocess",
                _route,
                {
                    _ROUTE_END: END,
                    _ROUTE_LOOP: "debate_round_start",
                },
            )

        return graph.compile(checkpointer=checkpointer)
