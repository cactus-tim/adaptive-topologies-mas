"""build_human_node_factory — DRY helper for building HITL LangGraph nodes."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import adispatch_custom_event

if TYPE_CHECKING:
    from atm.human.role_router import HumanRoleRouter

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

__all__ = ["build_human_node_factory"]


def _default_apply_decision(
    response: Any,
    state: dict[str, Any],
    shared: dict[str, Any],
) -> None:
    """Apply the human decision to shared state.

    Approve → ``shared["human_approved"] = True``.
    Reject → append CRITIQUE Message, set ``shared["needs_rerun"] = True``.
    """
    action: str = getattr(response, "action", "") or ""

    if action == "approve":
        shared["human_approved"] = True
        logger.debug("_default_apply_decision: approved")
    else:
        from atm.core.types import Message, MessageKind

        comment = getattr(response, "comment", None) or f"Human rejected with action={action!r}"
        rejection_msg = Message(
            sender="human_reviewer",
            kind=MessageKind.CRITIQUE,
            content=comment,
            payload={"action": action},
        )
        agents: dict[str, Any] = state.get("agents", {})
        reviewer_state = dict(agents.get("human_reviewer", {}))
        outbox: list[Any] = list(reviewer_state.get("outbox", []))
        outbox.append(rejection_msg)
        reviewer_state["outbox"] = outbox
        shared["needs_rerun"] = True
        logger.debug("_default_apply_decision: rejected — needs_rerun=True")


def build_human_node_factory(
    topology_name: str,
    human_cfg: Any,
    gateway: Any,
    *,
    request_id_template: str,
    question_extractor: Callable[[dict[str, Any]], str],
    apply_decision: Callable[[Any, dict[str, Any], dict[str, Any]], None] | None = None,
    role_router: HumanRoleRouter | None = None,
) -> Any:
    """Build an async LangGraph-compatible HITL node closure.

    Args:
        topology_name:        Caller topology name (used in logging and request_id).
        human_cfg:            HumanCfg with ``role``, ``timeout_s``, ``timeout_policy``.
        gateway:              HumanGateway-compatible object (or None if disabled).
        request_id_template:  str.format template; vars: ``{topology_name}``, ``{run_id}``, ``{iter_total}``.
        question_extractor:   ``(state) -> str`` callable.
        apply_decision:       Optional ``(response, state, shared) -> None``; defaults to _default_apply_decision.
        role_router:          Optional HumanRoleRouter; falls back to ``human_cfg.role`` if None.

    Returns:
        Async node callable ``async def _human_node(state) -> dict``.
    """
    _apply = apply_decision if apply_decision is not None else _default_apply_decision

    async def _human_node(state: dict[str, Any]) -> dict[str, Any]:
        """HITL node — invokes the human gateway and updates shared state."""
        shared = dict(deepcopy(state.get("shared", {})))

        import uuid as _uuid_mod

        _raw_run_id = shared.get("run_id") or state.get("run_id")
        run_id: _uuid_mod.UUID = (
            _raw_run_id
            if isinstance(_raw_run_id, _uuid_mod.UUID)
            else _uuid_mod.UUID(str(_raw_run_id))
            if _raw_run_id
            else _uuid_mod.uuid4()
        )
        iter_total: int = int(shared.get("iter_total", 0))

        request_id = request_id_template.format(
            topology_name=topology_name,
            run_id=run_id,
            iter_total=iter_total,
        )

        question = question_extractor(state)

        from atm.core.types import Phase

        if role_router is not None:
            _raw_phase = shared.get("phase", "execution")
            active_phase: Phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            active_role = await role_router.decide(active_phase, shared)
        else:
            active_role = human_cfg.role

        from atm.core.types import HumanContext

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
                "%s._human_node: adispatch human_request skipped (no callback ctx)",
                topology_name,
                exc_info=True,
            )

        _t0 = time.monotonic()
        timeout_s: float | None = getattr(human_cfg, "timeout_s", None)
        timeout_policy: str = getattr(human_cfg, "timeout_policy", "skip")

        if request_with_timeout is not None and timeout_s is not None:
            _fallback_gateway: Any = None
            if timeout_policy == "llm_fallback" and LLMSimulatedGateway is not None:
                _fb_llm: Any = getattr(gateway, "_llm", None)
                _fallback_gateway = LLMSimulatedGateway(llm=_fb_llm)
            response = await request_with_timeout(
                gateway,
                ctx,
                request_id=request_id,
                timeout_s=timeout_s,
                policy=timeout_policy,
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
                "%s._human_node: adispatch human_response skipped (no callback ctx)",
                topology_name,
                exc_info=True,
            )

        _apply(response, state, shared)

        logger.debug(
            "%s._human_node: action=%r iter_total=%d",
            topology_name,
            getattr(response, "action", None),
            iter_total,
        )

        return {"shared": shared}

    return _human_node
