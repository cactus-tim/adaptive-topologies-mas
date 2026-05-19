"""ChainTopology — linear Planner → Executor → Critic pipeline with retry-loop."""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import adispatch_custom_event

from atm.core.types import HumanContext, Message, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop

if TYPE_CHECKING:
    from atm.experiment.config import HumanCfg
    from atm.human.gateway import HumanGateway
    from atm.human.role_router import HumanRoleRouter

logger = logging.getLogger(__name__)

CHAIN_END: str = "__end__"

_StateGraphCls: type | None


def _import_state_graph() -> type | None:
    """Return StateGraph from langgraph.graph, or None if not installed."""
    try:
        from langgraph.graph import StateGraph as _StateGraph

        return _StateGraph
    except ImportError:  # pragma: no cover
        return None


_StateGraphCls = _import_state_graph()

StateGraph: type | None = _StateGraphCls

try:
    from atm.human.llm_simulated import LLMSimulatedGateway
except ImportError:  # pragma: no cover
    LLMSimulatedGateway = None  # type: ignore[assignment,misc]

try:
    from atm.human.cli_gateway import CLIGateway
except ImportError:  # pragma: no cover
    CLIGateway = None  # type: ignore[assignment,misc]

try:
    from atm.human._timeout import request_with_timeout
except ImportError:  # pragma: no cover
    request_with_timeout = None  # type: ignore[assignment]


def _build_human_reviewer_node(
    human_cfg: HumanCfg,
    gateway: HumanGateway,
    role_router: HumanRoleRouter | None = None,
) -> Any:
    """Build and return the async human_reviewer node function."""

    async def human_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        shared: dict[str, Any] = dict(deepcopy(state.get("shared", {})))

        run_id = shared.get("run_id")
        if run_id is None:
            raise RuntimeError(
                "human_reviewer node requires state['shared']['run_id'] to be set. "
                "Ensure Runner._build_initial_state populates run_id (M9 Step 3.1)."
            )

        iter_total: int = int(shared.get("iter_total", 0))

        agents: dict[str, Any] = state.get("agents", {})
        critic_outbox: list[Any] = list((agents.get("critic") or {}).get("outbox", []))
        question: str = "Please review the latest output and decide to approve, reject, or abstain."
        for msg in reversed(critic_outbox):
            if getattr(msg, "kind", None) == MessageKind.DECISION:
                question = getattr(msg, "content", question) or question
                break

        recent_msgs_raw: list[Any] = state.get("messages", [])
        recent_messages: tuple[Message, ...] = tuple(
            m for m in recent_msgs_raw if isinstance(m, Message)
        )

        import uuid as _uuid

        if not isinstance(run_id, _uuid.UUID):
            run_id = _uuid.UUID(str(run_id))

        from atm.core.types import Phase as _Phase

        if role_router is not None:
            _raw_phase = shared.get("phase", "execution")
            _active_phase: _Phase = (
                _Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            )
            active_role = await role_router.decide(_active_phase, shared)
        else:
            active_role = human_cfg.role

        ctx = HumanContext(
            run_id=run_id,
            role=active_role,
            question=question,
            recent_messages=recent_messages,
            allowed_actions=("approve", "reject", "abstain"),
        )

        request_id: str = f"chain:{iter_total}:reviewer"

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
                    "requested_at": datetime.now(UTC),
                },
            )
        except Exception:
            logger.debug("adispatch human_request skipped (no callback ctx)", exc_info=True)

        timeout_s_val: float | None = getattr(human_cfg, "timeout_s", None)
        policy: str = getattr(human_cfg, "timeout_policy", "skip")

        _t0 = time.monotonic()

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
                policy=policy,  # type: ignore[arg-type]
                llm_fallback_gateway=_fallback_gateway,
            )
        else:
            response = await gateway.request(ctx, request_id=request_id)

        latency_s = time.monotonic() - _t0

        try:
            await adispatch_custom_event(
                "human_response",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "answered_at": datetime.now(UTC),
                    "response_json": response.model_dump(mode="json"),
                    "source": response.source,
                    "timed_out": response.timed_out,
                    "latency_s": latency_s,
                },
            )
        except Exception:
            logger.debug("adispatch human_response skipped (no callback ctx)", exc_info=True)

        action: str = response.action
        delta: dict[str, Any] = {}

        if action == "approve":
            shared["human_approved"] = True
            delta["shared"] = shared

        elif action == "reject":
            comment: str = response.comment or "Human reviewer rejected this iteration."
            rejection_msg = Message(
                sender="human_reviewer",
                kind=MessageKind.CRITIQUE,
                content=f"[Human Reviewer Rejection] {comment}",
                payload={"human_rejected": True, "comment": comment},
            )
            shared["needs_rerun"] = True
            delta["shared"] = shared
            delta["messages"] = [rejection_msg]

        return delta

    return human_reviewer


async def _critic_postprocess(state: dict[str, Any]) -> dict[str, Any]:
    """Parse critic's last DECISION, update signals/final_answer, pin phase='execution'."""
    shared = dict(deepcopy(state.get("shared", {})))
    agents = state.get("agents", {})

    signals: dict[str, Any] = dict(shared.get("signals", {}))

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

    signals["critic_approved"] = approved

    executor_state = agents.get("executor", {})

    executor_tool_calls: list[Any] = list(executor_state.get("tool_calls", []))
    executor_tool_results: list[Any] = list(executor_state.get("tool_results", []))
    ok_call_ids: set[Any] = {
        getattr(r, "call_id", None) for r in executor_tool_results if getattr(r, "ok", False)
    }
    file_artifact: str | None = None
    for tc in reversed(executor_tool_calls):
        if getattr(tc, "tool_name", None) != "file_write":
            continue
        if executor_tool_results and getattr(tc, "id", None) not in ok_call_ids:
            continue
        args = getattr(tc, "args", None) or {}
        content = args.get("content") if isinstance(args, dict) else None
        path = args.get("path") if isinstance(args, dict) else None
        if not content:
            continue
        if isinstance(path, str) and path.endswith(".py"):
            file_artifact = str(content)
            break
        if file_artifact is None:
            file_artifact = str(content)

    executor_outbox: list[Any] = list(executor_state.get("outbox", []))
    draft_text: str | None = None
    for msg in reversed(executor_outbox):
        if getattr(msg, "kind", None) == MessageKind.DRAFT:
            raw_payload = getattr(msg, "payload", {}) or {}
            draft_text = raw_payload.get("draft") or getattr(msg, "content", None)
            break

    task_id = str(shared.get("task_id") or "").lower()
    _code_tasks = {"humaneval"}
    if task_id in _code_tasks:
        final_answer = file_artifact or draft_text or "<incomplete>"
    else:
        final_answer = draft_text or file_artifact or "<incomplete>"

    shared["phase"] = "execution"
    shared["signals"] = signals
    shared["final_answer"] = final_answer

    return {"shared": shared}


def _route_from_critic(state: dict[str, Any], cfg: TopologyConfig) -> str:
    """Return CHAIN_END or 'executor'; increments iter_total/iteration first."""
    shared = state["shared"]

    shared["iter_total"] = shared.get("iter_total", 0) + 1
    shared["iteration"] = shared.get("iteration", 0) + 1

    signals: dict[str, Any] = shared.get("signals", {})
    critic_approved: bool = bool(signals.get("critic_approved", False))

    stop, _reason = _should_stop(
        state,
        cfg,
        topology_success=critic_approved,
        topology_max_reached=False,
    )

    if stop:
        return CHAIN_END

    return "executor"


@TopologyRegistry.register("chain")
class ChainTopology:
    """Linear topology: START → planner → executor → critic → (retry | END)."""

    name: str = "chain"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Build and compile the Chain LangGraph."""
        sg_cls = StateGraph
        if sg_cls is None:  # pragma: no cover
            raise ImportError(
                "langgraph.graph is required for ChainTopology.build(). Install: uv add langgraph"
            )

        planner = agents["planner"]
        executor = agents["executor"]
        critic = agents["critic"]

        from atm.core.state import GraphState

        graph: Any = sg_cls(GraphState)

        graph.add_node("planner", planner.step)
        graph.add_node("executor", executor.step)
        graph.add_node("critic", critic.step)
        graph.add_node("critic_postprocess", _critic_postprocess)

        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "critic")
        graph.add_edge("critic", "critic_postprocess")

        human_cfg: HumanCfg | None = kwargs.get("human_cfg")

        def _route(state: dict[str, Any]) -> str:
            return _route_from_critic(state, cfg)

        if human_cfg is not None and human_cfg.enabled:
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

            role_router = kwargs.get("role_router")
            node_fn = _build_human_reviewer_node(
                human_cfg, gateway_instance, role_router=role_router
            )
            graph.add_node("human_reviewer", node_fn)

            graph.add_edge("critic_postprocess", "human_reviewer")
            graph.add_conditional_edges(
                "human_reviewer",
                _route,
                {
                    "executor": "executor",
                    CHAIN_END: CHAIN_END,
                },
            )
        else:
            graph.add_conditional_edges(
                "critic_postprocess",
                _route,
                {
                    "executor": "executor",
                    CHAIN_END: CHAIN_END,
                },
            )

        graph.set_entry_point("planner")

        checkpointer = kwargs.get("checkpointer")
        return graph.compile(checkpointer=checkpointer)
