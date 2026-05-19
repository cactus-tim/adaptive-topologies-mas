"""HierarchicalTopology — two-level coordinator + compiled subgraphs."""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState
from atm.core.types import HumanContext, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop, get_topology_extras

if TYPE_CHECKING:
    from atm.experiment.config import HumanCfg
    from atm.human.gateway import HumanGateway

logger = logging.getLogger(__name__)

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

try:
    from langchain_core.callbacks.manager import adispatch_custom_event
except ImportError:  # pragma: no cover
    adispatch_custom_event = None  # type: ignore[assignment]

HumanRoleRouter: Any
try:
    from atm.human.role_router import HumanRoleRouter as HumanRoleRouter
except ImportError:  # pragma: no cover
    HumanRoleRouter = None

_DEFAULT_MAX_ROUNDS = 4
_DEFAULT_FINALIZE_SIGNAL = "top_coord_finalize"
_ROUTE_TEAM_A = "team_a"
_ROUTE_TEAM_B = "team_b"
_ROUTE_FINALIZE = "hierarchical_finalize"
_ROUTE_END = "__end__"


def _extract_draft(state: dict[str, Any], agent_id: str) -> str | None:
    """Return last DRAFT message content from the given agent's outbox, or None."""
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


def _extract_hierarchical_artifact(
    agents: dict[str, Any],
    sub_team_worker_ids: list[str],
) -> str | None:
    """Return best file_write artifact from any sub-team worker, or None."""
    py_solution: str | None = None
    py_any: str | None = None
    any_file: str | None = None

    for worker_id in sub_team_worker_ids:
        worker_state: dict[str, Any] = dict(agents.get(worker_id) or {})
        tool_calls: list[Any] = list(worker_state.get("tool_calls") or [])
        tool_results: list[Any] = list(worker_state.get("tool_results") or [])
        ok_call_ids: set[Any] = {
            getattr(r, "call_id", None) for r in tool_results if getattr(r, "ok", False)
        }

        for tc in reversed(tool_calls):
            if getattr(tc, "tool_name", None) != "file_write":
                continue
            if tool_results and getattr(tc, "id", None) not in ok_call_ids:
                continue
            args = getattr(tc, "args", None) or {}
            if not isinstance(args, dict):
                continue
            content = args.get("content")
            path = args.get("path")
            if not content:
                continue
            if any_file is None:
                any_file = str(content)
            if isinstance(path, str) and path.endswith(".py"):
                if py_any is None:
                    py_any = str(content)
                if py_solution is None and (
                    path.endswith("solution.py") or path.endswith("main.py")
                ):
                    py_solution = str(content)
                    break

        if py_solution is not None:
            break

    if py_solution is not None:
        return py_solution
    if py_any is not None:
        return py_any
    if any_file is not None:
        return any_file
    return None


def _build_human_top_reviewer_node(
    human_cfg: HumanCfg,
    gateway: HumanGateway,
    *,
    role_router: Any = None,
) -> Any:
    """Build async human_top_reviewer node for scope='top'."""
    from atm.core.types import Message

    _role_router = role_router

    async def human_top_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        shared: dict[str, Any] = dict(deepcopy(state.get("shared", {})))
        run_id = shared.get("run_id")
        if run_id is None:
            raise RuntimeError(
                "human_top_reviewer requires state['shared']['run_id']. "
                "Ensure Runner._build_initial_state sets run_id."
            )

        import uuid as _uuid

        if not isinstance(run_id, _uuid.UUID):
            run_id = _uuid.UUID(str(run_id))

        iter_total: int = int(shared.get("iter_total", 0))

        signals: dict[str, Any] = dict(shared.get("signals") or {})
        team_a_draft = signals.get("team_a_draft")
        team_b_draft = signals.get("team_b_draft")
        if team_a_draft and team_b_draft:
            question = (
                f"Team A draft: {str(team_a_draft)[:200]}\n"
                f"Team B draft: {str(team_b_draft)[:200]}\n"
                "Please review and decide to approve, reject, or abstain."
            )
        else:
            question = (
                "Please review the hierarchical team outputs and approve, reject, or abstain."
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
            recent_messages=(),
            allowed_actions=("approve", "reject", "abstain"),
        )

        request_id: str = f"hierarchical:top:{iter_total}:reviewer"

        try:
            if adispatch_custom_event is not None:
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
            if adispatch_custom_event is not None:
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
                sender="human_top_reviewer",
                kind=MessageKind.CRITIQUE,
                content=f"[Human Top Reviewer Rejection] {comment}",
                payload={"human_rejected": True, "comment": comment},
            )
            shared["needs_rerun"] = True
            delta["shared"] = shared
            delta["messages"] = [rejection_msg]

        return delta

    return human_top_reviewer


def _build_human_sub_reviewer_node(
    team_id: str,
    human_cfg: HumanCfg,
    gateway: HumanGateway,
    *,
    role_router: Any = None,
) -> Any:
    """Build async human_sub_reviewer node for scope='sub_team'.

    WARNING: CLIGateway is NOT supported in sub_team scope — the interactive
    interrupt() API is incompatible with compiled subgraphs + checkpointer.
    Only LLMSimulatedGateway is supported; CLIGateway callers must use scope='top'.
    """
    _team_id = team_id
    _role_router = role_router

    async def human_sub_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        shared: dict[str, Any] = dict(deepcopy(state.get("shared", {})))
        run_id = shared.get("run_id")
        if run_id is None:
            raise RuntimeError(
                f"human_sub_reviewer ({_team_id}) requires state['shared']['run_id'] to be set."
            )

        import uuid as _uuid

        if not isinstance(run_id, _uuid.UUID):
            run_id = _uuid.UUID(str(run_id))

        iter_total: int = int(shared.get("iter_total", 0))

        question = f"Please review sub-team {_team_id} activity and approve, reject, or abstain."

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
            recent_messages=(),
            allowed_actions=("approve", "reject", "abstain"),
        )

        request_id: str = f"hierarchical:{_team_id}:{iter_total}:reviewer"

        try:
            if adispatch_custom_event is not None:
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
            logger.debug(
                "adispatch human_request skipped in subgraph %s (no callback ctx)",
                _team_id,
                exc_info=True,
            )

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
            if adispatch_custom_event is not None:
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
            logger.debug(
                "adispatch human_response skipped in subgraph %s (no callback ctx)",
                _team_id,
                exc_info=True,
            )

        return {}

    return human_sub_reviewer


@TopologyRegistry.register("hierarchical")
class HierarchicalTopology:
    """Two-level hierarchical topology: top_coord → [team_a, team_b] subgraphs → finalize."""

    name = "hierarchical"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a CompiledStateGraph; raises ValueError on nested sub_teams."""
        checkpointer = kwargs.get("checkpointer")

        extras = get_topology_extras(cfg, "hierarchical")
        max_rounds: int = int(extras.get("max_rounds", _DEFAULT_MAX_ROUNDS))
        final_answer_strategy: str = str(extras.get("final_answer_strategy", "json_concat"))
        finalize_signal: str = str(extras.get("finalize_signal", _DEFAULT_FINALIZE_SIGNAL))

        sub_teams: list[dict[str, Any]] = list(extras.get("sub_teams") or [])
        if len(sub_teams) < 2:
            sub_teams = [
                {"team_id": "team_a", "workers": ["executor_a1", "executor_a2"]},
                {"team_id": "team_b", "workers": ["executor_b1", "executor_b2"]},
            ]

        for i, team_cfg in enumerate(sub_teams):
            if "sub_teams" in team_cfg:
                raise ValueError(
                    f"HierarchicalTopology: sub_teams[{i}] contains a nested 'sub_teams' key. "
                    f"Only 2 levels are allowed. "
                    f"3-level nesting is not supported in this implementation."
                )

        team_a_cfg = sub_teams[0]
        team_b_cfg = sub_teams[1]
        team_a_id: str = str(team_a_cfg.get("team_id", "team_a"))
        team_b_id: str = str(team_b_cfg.get("team_id", "team_b"))
        workers_a: list[str] = list(team_a_cfg.get("workers") or [])
        workers_b: list[str] = list(team_b_cfg.get("workers") or [])

        human_cfg: HumanCfg | None = kwargs.get("human_cfg")
        role_router: Any = kwargs.get("role_router")
        hitl_enabled = human_cfg is not None and human_cfg.enabled
        hitl_scope: str = "top"
        if hitl_enabled and human_cfg is not None:
            human_extra = human_cfg.extra or {}
            hitl_scope = str(human_extra.get("scope", "top"))

        gateway_instance: Any = None
        if hitl_enabled and human_cfg is not None:
            gateway_llm: Any = kwargs.get("human_gateway_llm")
            if human_cfg.gateway == "cli":
                gateway_instance = CLIGateway() if CLIGateway is not None else None
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

        if hitl_enabled and hitl_scope == "sub_team" and human_cfg is not None:
            team_a_subgraph = self._build_subgraph_with_human(
                team_a_id,
                workers_a,
                agents,
                human_cfg,
                gateway_instance,
                role_router=role_router,
            )
            team_b_subgraph = self._build_subgraph_with_human(
                team_b_id,
                workers_b,
                agents,
                human_cfg,
                gateway_instance,
                role_router=role_router,
            )
        else:
            team_a_subgraph = self._build_subgraph(team_a_id, workers_a, agents)
            team_b_subgraph = self._build_subgraph(team_b_id, workers_b, agents)

        async def top_coord_node(state: GraphState) -> dict[str, Any]:
            """Rule-based top coordinator."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            new_iter_total = int(shared.get("iter_total") or 0) + 1
            new_iteration = int(shared.get("iteration") or 0) + 1
            shared["iter_total"] = new_iter_total
            shared["iteration"] = new_iteration

            if new_iter_total == 1:
                signals["activate_team_a"] = True
                signals["activate_team_b"] = True

            shared["signals"] = signals
            return {"shared": shared}

        def _route_from_top_coord(state: GraphState) -> str:
            """Route after top_coord."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            iter_total: int = int(shared.get("iter_total") or 0)

            topology_success = bool(signals.get(finalize_signal))
            stop, _reason = _should_stop(
                dict(state),
                cfg,
                topology_success=topology_success,
                topology_max_reached=False,
            )
            if stop:
                return _ROUTE_FINALIZE

            activate_a = bool(signals.get("activate_team_a"))
            activate_b = bool(signals.get("activate_team_b"))

            if activate_a:
                return _ROUTE_TEAM_A
            if activate_b:
                return _ROUTE_TEAM_B

            team_a_draft = signals.get("team_a_draft")
            team_b_draft = signals.get("team_b_draft")
            if team_a_draft and team_b_draft:
                return _ROUTE_FINALIZE

            if iter_total > max_rounds:
                return _ROUTE_FINALIZE

            return _ROUTE_TEAM_A

        _team_a_workers = workers_a
        _team_b_workers = workers_b

        async def top_coord_after_team_a(state: GraphState) -> dict[str, Any]:
            """Post-team_a coordinator: collects team_a draft, activates team_b."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            team_a_draft: str | None = None
            for worker_id in _team_a_workers:
                draft = _extract_draft(dict(state), worker_id)
                if draft:
                    team_a_draft = draft

            if team_a_draft:
                signals["team_a_draft"] = team_a_draft

            signals.pop("activate_team_a", None)
            signals["activate_team_b"] = True

            new_iter_total = int(shared.get("iter_total") or 0) + 1
            new_iteration = int(shared.get("iteration") or 0) + 1
            shared["iter_total"] = new_iter_total
            shared["iteration"] = new_iteration
            shared["signals"] = signals
            return {"shared": shared}

        async def top_coord_after_team_b(state: GraphState) -> dict[str, Any]:
            """Post-team_b coordinator: collects draft, checks finalize condition."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            team_b_draft: str | None = None
            for worker_id in _team_b_workers:
                draft = _extract_draft(dict(state), worker_id)
                if draft:
                    team_b_draft = draft

            if team_b_draft:
                signals["team_b_draft"] = team_b_draft

            signals.pop("activate_team_b", None)

            team_a_draft = signals.get("team_a_draft")
            if team_a_draft and team_b_draft:
                signals[finalize_signal] = True
                logger.info("top_coord: both team drafts present, setting %s=True", finalize_signal)

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

            topology_success = bool(signals.get(finalize_signal))
            topology_max_reached = iter_total > max_rounds
            stop, _reason = _should_stop(
                dict(state),
                cfg,
                topology_success=topology_success,
                topology_max_reached=topology_max_reached,
            )
            if stop:
                if hitl_enabled and hitl_scope == "top":
                    return "human_top_reviewer"
                return _ROUTE_FINALIZE

            return _ROUTE_TEAM_A

        async def hierarchical_finalize_node(state: GraphState) -> dict[str, Any]:
            """Aggregate team drafts into final_answer."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            team_a_draft = signals.get("team_a_draft")
            team_b_draft = signals.get("team_b_draft")

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

            task_id: str = str(shared.get("task_id") or "").lower()
            is_code_task: bool = task_id in {"humaneval"}
            if is_code_task:
                agents_state: dict[str, Any] = dict(state.get("agents") or {})
                all_workers: list[str] = list(_team_a_workers) + list(_team_b_workers)
                artifact = _extract_hierarchical_artifact(agents_state, all_workers)
                if artifact:
                    shared["final_answer"] = artifact
                    shared["signals"] = signals
                    return {"shared": shared}

            if final_answer_strategy == "json_concat":
                final_answer = json.dumps(
                    {
                        team_a_id: team_a_draft or "<incomplete>",
                        team_b_id: team_b_draft or "<incomplete>",
                    },
                    ensure_ascii=False,
                )
            else:
                parts = []
                if team_a_draft:
                    parts.append(f"[{team_a_id}] {team_a_draft}")
                if team_b_draft:
                    parts.append(f"[{team_b_id}] {team_b_draft}")
                final_answer = " | ".join(parts) if parts else "<incomplete>"

            shared["final_answer"] = final_answer
            shared["signals"] = signals
            return {"shared": shared}

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        graph.add_node("top_coord", top_coord_node)
        graph.add_node("after_team_a", top_coord_after_team_a)
        graph.add_node("after_team_b", top_coord_after_team_b)

        graph.add_node(_ROUTE_TEAM_A, team_a_subgraph)
        graph.add_node(_ROUTE_TEAM_B, team_b_subgraph)

        graph.add_node(_ROUTE_FINALIZE, hierarchical_finalize_node)

        graph.add_edge(START, "top_coord")
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

        graph.add_edge(_ROUTE_TEAM_B, "after_team_b")

        if hitl_enabled and hitl_scope == "top" and human_cfg is not None:
            node_fn = _build_human_top_reviewer_node(
                human_cfg, gateway_instance, role_router=role_router
            )
            graph.add_node("human_top_reviewer", node_fn)

            graph.add_conditional_edges(
                "after_team_b",
                _route_from_after_team_b,
                {
                    _ROUTE_TEAM_A: _ROUTE_TEAM_A,
                    _ROUTE_FINALIZE: _ROUTE_FINALIZE,
                    _ROUTE_END: END,
                    "human_top_reviewer": "human_top_reviewer",
                },
            )
            graph.add_edge("human_top_reviewer", _ROUTE_FINALIZE)
        else:
            graph.add_conditional_edges(
                "after_team_b",
                _route_from_after_team_b,
                {
                    _ROUTE_TEAM_A: _ROUTE_TEAM_A,
                    _ROUTE_FINALIZE: _ROUTE_FINALIZE,
                    _ROUTE_END: END,
                },
            )

        graph.add_edge(_ROUTE_FINALIZE, END)

        return graph.compile(checkpointer=checkpointer)

    def _build_subgraph(
        self,
        team_id: str,
        worker_ids: list[str],
        agents: dict[str, Any],
    ) -> Any:
        """Build a compiled subgraph for a sub-team: sub_coord → workers in order."""
        _worker_ids = list(worker_ids)
        _team_id = team_id

        async def sub_coord_node(state: GraphState) -> dict[str, Any]:
            """Rule-based sub-coordinator."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})

            signals[f"{_team_id}_active"] = True
            shared["signals"] = signals
            return {"shared": shared}

        sub_graph: StateGraph[GraphState] = StateGraph(GraphState)

        sub_coord_name = f"sub_coord_{_team_id}"
        sub_graph.add_node(sub_coord_name, sub_coord_node)
        sub_graph.add_edge(START, sub_coord_name)

        if not _worker_ids:
            sub_graph.add_edge(sub_coord_name, END)
        else:
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

            sub_graph.add_edge(prev_node, END)

        return sub_graph.compile()

    def _build_subgraph_with_human(
        self,
        team_id: str,
        worker_ids: list[str],
        agents: dict[str, Any],
        human_cfg: HumanCfg,
        gateway: Any,
        *,
        role_router: Any = None,
    ) -> Any:
        """Build a compiled subgraph with human_sub_reviewer inserted after sub_coord.

        WARNING: CLIGateway is NOT supported in sub_team scope — the interactive
        interrupt() API is incompatible with compiled subgraphs + checkpointer.
        Only LLMSimulatedGateway is supported; CLIGateway callers must use scope='top'.
        """
        _worker_ids = list(worker_ids)
        _team_id = team_id

        async def sub_coord_node(state: GraphState) -> dict[str, Any]:
            """Rule-based sub-coordinator: initialises sub-team iteration."""
            shared: dict[str, Any] = dict(state.get("shared") or {})
            signals: dict[str, Any] = dict(shared.get("signals") or {})
            signals[f"{_team_id}_active"] = True
            shared["signals"] = signals
            return {"shared": shared}

        sub_graph: StateGraph[GraphState] = StateGraph(GraphState)

        sub_coord_name = f"sub_coord_{_team_id}"
        sub_graph.add_node(sub_coord_name, sub_coord_node)
        sub_graph.add_edge(START, sub_coord_name)

        human_node_name = f"human_sub_reviewer_{_team_id}"
        human_node_fn = _build_human_sub_reviewer_node(
            _team_id, human_cfg, gateway, role_router=role_router
        )
        sub_graph.add_node(human_node_name, human_node_fn)
        sub_graph.add_edge(sub_coord_name, human_node_name)

        if not _worker_ids:
            sub_graph.add_edge(human_node_name, END)
        else:
            prev_node = human_node_name
            for worker_id in _worker_ids:
                worker_agent = agents.get(worker_id)

                if worker_agent is None:
                    logger.warning(
                        "_build_subgraph_with_human: worker %r not found in agents dict; "
                        "using no-op node.",
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

            sub_graph.add_edge(prev_node, END)

        return sub_graph.compile()
