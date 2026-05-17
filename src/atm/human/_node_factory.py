"""build_human_node_factory — DRY helper for building HITL LangGraph nodes (M9.1 Step 1.2).

This module extracts the canonical HITL-node pattern established in the Chain
topology reference implementation (M9) and makes it reusable for Star/Mesh.

Usage pattern (topology build methods)::

    from atm.human import build_human_node_factory

    human_node = build_human_node_factory(
        topology_name="star",
        human_cfg=human_cfg,
        gateway=gateway,
        request_id_template="star:{run_id}:{iter_total}:reviewer",
        question_extractor=lambda state: "Should the executor retry?",
    )
    graph.add_node("human_reviewer", human_node)

Design decisions
----------------
- Module-level **lazy try/except imports** for ``LLMSimulatedGateway``,
  ``CLIGateway``, and ``request_with_timeout`` mirror the Chain pattern: the
  module-level names are set to ``None`` on import failure, enabling tests to
  patch them without installing the full gateway stack.
- ``apply_decision`` default implements approve → ``shared["human_approved"]=True``;
  reject → append Message + ``needs_rerun=True``. Custom callbacks override this.
- Return type is ``Any`` to avoid a hard dependency on LangGraph's node
  signature type (not a direct dep in pyproject.toml).
- Topology-specific semantics (judge outbox write, subgraph scope, router
  override) are NOT covered here — use inline closures for those.
- MeshTopology deliberately rolls its own ``human_peer_node`` inline closure
  because broadcast_bus voting semantics and round-aware activation are tightly
  coupled to the mesh dispatcher and cannot be expressed via the default
  ``apply_decision`` without a new overload.  See mesh.py module docstring.
"""

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

# ---------------------------------------------------------------------------
# Module-level lazy imports — patchable in tests
# ---------------------------------------------------------------------------
# These are set to None if the respective module is unavailable at import time.
# Tests patch "atm.human._node_factory.<Name>" to inject stubs without
# needing the full gateway stack installed.
#
# Typed as Any to allow both the class and None without mypy complaints.
# The try/except pragma: no cover blocks are for ImportError paths that only
# trigger when the package is not installed (which does not happen in tests).

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


# ---------------------------------------------------------------------------
# Default apply_decision callback
# ---------------------------------------------------------------------------


def _default_apply_decision(
    response: Any,
    state: dict[str, Any],
    shared: dict[str, Any],
) -> None:
    """Apply the human decision to shared state.

    Approve (action == "approve"):
        Sets ``shared["human_approved"] = True``.

    Reject (any other action):
        Appends a CRITIQUE Message to ``state["agents"]`` outbox (if agents
        are present) and sets ``shared["needs_rerun"] = True``.

    Parameters
    ----------
    response:
        A :class:`~atm.core.types.HumanResponse` instance.
    state:
        The full LangGraph state dict (for accessing agents).
    shared:
        The mutable ``shared`` dict extracted from state (already deep-copied
        by the node closure before this callback is called).
    """
    action: str = getattr(response, "action", "") or ""

    if action == "approve":
        shared["human_approved"] = True
        logger.debug("_default_apply_decision: approved")
    else:
        # Reject — write a CRITIQUE Message and flag needs_rerun
        from atm.core.types import Message, MessageKind

        comment = getattr(response, "comment", None) or f"Human rejected with action={action!r}"
        rejection_msg = Message(
            sender="human_reviewer",
            kind=MessageKind.CRITIQUE,
            content=comment,
            payload={"action": action},
        )
        # Append to a generic "human_reviewer" outbox in agents state
        agents: dict[str, Any] = state.get("agents", {})
        reviewer_state = dict(agents.get("human_reviewer", {}))
        outbox: list[Any] = list(reviewer_state.get("outbox", []))
        outbox.append(rejection_msg)
        reviewer_state["outbox"] = outbox
        # NOTE: we do NOT mutate state["agents"] here — the node closure
        # returns a delta dict; the topology is responsible for merging it.
        shared["needs_rerun"] = True
        logger.debug("_default_apply_decision: rejected — needs_rerun=True")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_human_node_factory(
    topology_name: str,
    human_cfg: Any,  # HumanCfg — typed as Any to avoid circular import at module level
    gateway: Any,  # HumanGateway — typed as Any
    *,
    request_id_template: str,
    question_extractor: Callable[[dict[str, Any]], str],
    apply_decision: Callable[[Any, dict[str, Any], dict[str, Any]], None] | None = None,
    role_router: HumanRoleRouter | None = None,
    fallback_llm: Any = None,
) -> Any:
    """Build an async LangGraph-compatible node closure for HITL decisions.

    The returned node:
    1. Extracts ``run_id`` and ``iter_total`` from ``state["shared"]``.
    2. Formats ``request_id`` from ``request_id_template`` using ``{run_id}``
       and ``{iter_total}`` (and optionally ``{topology_name}``).
    3. Dispatches ``human_request`` custom event via ``adispatch_custom_event``.
    4. Calls the gateway via ``request_with_timeout`` (if available) or direct
       ``gateway.request()``.
    5. Dispatches ``human_response`` custom event.
    6. Calls ``apply_decision(response, state, shared)`` to update shared state.
    7. Returns a state delta dict with the updated ``shared`` key.

    Parameters
    ----------
    topology_name:
        Name of the calling topology (e.g. ``"star"``, ``"mesh"``).  Used for
        logging and as a ``{topology_name}`` format variable in ``request_id_template``.
    human_cfg:
        A :class:`~atm.experiment.config.HumanCfg` instance (or duck-typed
        equivalent with ``role``, ``timeout_s``, ``timeout_policy`` attributes).
    gateway:
        A :class:`~atm.human.gateway.HumanGateway`-compatible object.  May be
        ``None`` if ``human_cfg.enabled`` is False (node will no-op).
    request_id_template:
        A str.format template for the idempotency key.  Available variables:
        ``{topology_name}``, ``{run_id}``, ``{iter_total}``.
        Example: ``"star:{run_id}:{iter_total}:reviewer"``.
    question_extractor:
        A callable ``(state) -> str`` that extracts the decision question from
        the current graph state.
    apply_decision:
        Optional callback ``(response, state, shared) -> None`` that applies
        the human decision to ``shared``.  Defaults to ``_default_apply_decision``
        (approve → ``human_approved=True``; reject → ``needs_rerun=True``).
    role_router:
        Optional :class:`~atm.human.role_router.HumanRoleRouter` instance.
        When ``None`` (default), ``human_cfg.role`` is used for every interaction
        (byte-identical to pre-m9.2 behaviour).  When provided, the router's
        ``decide(phase, shared)`` is awaited to determine the active role for
        each node invocation.
    fallback_llm:
        Optional LLM wrapper to pass to ``LLMSimulatedGateway`` when building
        the fallback gateway for ``timeout_policy="llm_fallback"``.  When
        ``None`` (default), the factory attempts to read ``gateway._llm`` for
        back-compat.  This parameter exists specifically so topologies that use
        a ``StreamlitHumanGateway`` (which has no ``._llm``) can still thread
        the fallback LLM through without an ``AttributeError``.
        Guard: ``LLMSimulatedGateway(llm=_fb_llm) if _fb_llm is not None else None``.

    Returns
    -------
    Any
        An async callable ``async def _human_node(state) -> dict[str, Any]``
        compatible with LangGraph's node signature.
    """
    _apply = apply_decision if apply_decision is not None else _default_apply_decision

    async def _human_node(state: dict[str, Any]) -> dict[str, Any]:
        """HITL node — invokes the human gateway and updates shared state."""
        shared = dict(deepcopy(state.get("shared", {})))

        # Extract identifiers for event dispatching and request_id formatting
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

        # Extract question from state
        question = question_extractor(state)

        # Determine active role: use role_router if provided, else fall back to human_cfg.role
        from atm.core.types import Phase

        if role_router is not None:
            _raw_phase = shared.get("phase", "execution")
            active_phase: Phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            active_role = await role_router.decide(active_phase, shared)
        else:
            active_role = human_cfg.role

        # Build HumanContext
        from atm.core.types import HumanContext

        ctx = HumanContext(
            run_id=run_id,
            role=active_role,
            question=question,
            recent_messages=tuple(state.get("messages", [])[-5:]),
            allowed_actions=("approve", "reject"),
            deadline_s=int(human_cfg.timeout_s) if human_cfg.timeout_s is not None else None,
        )

        # Dispatch human_request event
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

        # Call gateway
        _t0 = time.monotonic()
        timeout_s: float | None = getattr(human_cfg, "timeout_s", None)
        timeout_policy: str = getattr(human_cfg, "timeout_policy", "skip")

        if request_with_timeout is not None and timeout_s is not None:
            _fallback_gateway: Any = None
            if timeout_policy == "llm_fallback" and LLMSimulatedGateway is not None:
                _fb_llm: Any = (
                    fallback_llm if fallback_llm is not None else getattr(gateway, "_llm", None)
                )
                _fallback_gateway = (
                    LLMSimulatedGateway(llm=_fb_llm) if _fb_llm is not None else None
                )
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

        # Dispatch human_response event
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

        # Apply decision — mutates shared in place
        _apply(response, state, shared)

        logger.debug(
            "%s._human_node: action=%r iter_total=%d",
            topology_name,
            getattr(response, "action", None),
            iter_total,
        )

        return {"shared": shared}

    return _human_node
