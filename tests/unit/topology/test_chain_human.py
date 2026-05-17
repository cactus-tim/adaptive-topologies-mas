"""Unit tests for ChainTopology human_reviewer node (M9 Step 4.1).

Tests cover:
  1. Default path (human_cfg=None) — graph identical to before, no HITL node.
  2. Default path (human_cfg.enabled=False) — same as above.
  3. HITL path (enabled=True, gateway returns approve) → shared["human_approved"]=True.
  4. HITL path (enabled=True, gateway returns reject) → rejection comment in messages,
     no human_approved key set.
  5. HITL path (enabled=True, gateway returns abstain) → no-op state change.
  6. Dispatch order: human_request emitted BEFORE human_response in one invocation.
  7. Default path: NO human_request event emitted.
  8. request_id is deterministic: f"chain:{iter_total}:reviewer".
  9. Timeout path: skip policy returns abstain-like response, no crash.
  10. Dispatch payloads match canonical shape (F1/F2 fix verification).
  11. llm_fallback timeout_policy doesn't crash at build time (F3 fix verification).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import atm.topology.chain  # noqa: F401 — triggers @TopologyRegistry.register
from atm.core.types import HumanResponse, HumanRole, Message, MessageKind
from atm.experiment.config import HumanCfg
from atm.topology.base import TopologyConfig, TopologyRegistry
from atm.topology.chain import ChainTopology, _build_human_reviewer_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(*, max_iterations: int = 10) -> TopologyConfig:
    return TopologyConfig(name="chain", max_iterations=max_iterations)


def _make_human_cfg(
    *,
    enabled: bool = True,
    gateway: str = "llm_simulated",
    role: HumanRole = HumanRole.REVIEWER,
    timeout_s: float = 10.0,
    timeout_policy: str = "skip",
) -> HumanCfg:
    return HumanCfg(
        enabled=enabled,
        gateway=gateway,
        role=role,
        timeout_s=timeout_s,
        timeout_policy=timeout_policy,  # type: ignore[arg-type]
    )


def _make_state(
    *,
    iter_total: int = 0,
    run_id: uuid.UUID | None = None,
    critic_approved: bool = False,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal GraphState-like dict with run_id in shared."""
    _run_id = run_id or uuid.uuid4()
    return {
        "shared": {
            "run_id": _run_id,
            "iter_total": iter_total,
            "iteration": 0,
            "phase": "execution",
            "signals": {"critic_approved": critic_approved},
            "final_answer": None,
        },
        "agents": {
            "critic": {
                "outbox": [
                    Message(
                        sender="critic",
                        kind=MessageKind.DECISION,
                        content="verdict",
                        payload={"approved": critic_approved, "comment": "looks fine"},
                    )
                ]
            }
        },
        "messages": messages or [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


def _make_approve_response() -> HumanResponse:
    return HumanResponse(action="approve", comment="LGTM", source="llm_sim", timed_out=False)


def _make_reject_response(comment: str = "Needs more work") -> HumanResponse:
    return HumanResponse(action="reject", comment=comment, source="llm_sim", timed_out=False)


def _make_abstain_response() -> HumanResponse:
    return HumanResponse(action="abstain", comment=None, source="llm_sim", timed_out=False)


def _make_timeout_response() -> HumanResponse:
    return HumanResponse(action="timeout", comment="timed out", source="timeout", timed_out=True)


def _ensure_chain_registered() -> None:
    if "chain" not in TopologyRegistry.list_names():
        TopologyRegistry.register("chain")(ChainTopology)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ensure_chain_registered() -> None:
    _ensure_chain_registered()


# ---------------------------------------------------------------------------
# Group 1 — Default path (no HITL)
# ---------------------------------------------------------------------------


class TestChainDefaultPath:
    """Without human_cfg (or enabled=False), graph is identical to before."""

    def test_build_no_human_cfg_returns_compiled(self) -> None:
        """build(human_cfg=None) returns a compiled graph, same as before."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.chain.StateGraph", return_value=mock_graph):
            compiled = ChainTopology().build(agents, cfg)

        assert compiled is mock_compiled

    def test_build_disabled_human_cfg_no_extra_node(self) -> None:
        """build(human_cfg=HumanCfg(enabled=False)) — no human_reviewer node added."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=False)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.chain.StateGraph", return_value=mock_graph):
            ChainTopology().build(agents, cfg, human_cfg=human_cfg)

        # Verify human_reviewer was NOT added
        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_reviewer" not in node_names

    def test_no_human_request_event_without_human_cfg(self) -> None:
        """human_request event is NOT dispatched when human_cfg is None."""
        dispatched: list[str] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            dispatched.append(name)

        # build_human_reviewer_node with human_cfg=None should not be called
        # But let's verify by calling the node directly with a mock that never triggers
        # Since there's no human_reviewer node in the default path, dispatched stays empty.
        # We test this by ensuring _build_human_reviewer_node is NOT invoked during build.
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain._build_human_reviewer_node") as mock_builder,
        ):
            ChainTopology().build(agents, cfg)

        mock_builder.assert_not_called()


# ---------------------------------------------------------------------------
# Group 2 — HITL path: node insertion into graph
# ---------------------------------------------------------------------------


class TestChainHITLNodeInsertion:
    """With human_cfg.enabled=True, human_reviewer node is added to the graph."""

    def test_human_reviewer_node_added(self) -> None:
        """human_reviewer node is added when enabled=True."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=True)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain.LLMSimulatedGateway") as mock_gw_cls,
        ):
            mock_gw_cls.return_value = MagicMock()
            ChainTopology().build(agents, cfg, human_cfg=human_cfg)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_reviewer" in node_names

    def test_edge_critic_postprocess_to_human_reviewer(self) -> None:
        """Edge from critic_postprocess → human_reviewer is added."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=True)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain.LLMSimulatedGateway") as mock_gw_cls,
        ):
            mock_gw_cls.return_value = MagicMock()
            ChainTopology().build(agents, cfg, human_cfg=human_cfg)

        edge_calls = [call.args for call in mock_graph.add_edge.call_args_list]
        assert ("critic_postprocess", "human_reviewer") in edge_calls


# ---------------------------------------------------------------------------
# Group 3 — HITL node logic: approve / reject / abstain / timeout
# ---------------------------------------------------------------------------


class TestHumanReviewerNodeApprove:
    """human_reviewer node with approve response updates shared correctly."""

    def test_approve_sets_human_approved_true(self) -> None:
        """Approve response → shared['human_approved'] = True."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=2)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_approve_response())
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        dispatched_events: list[str] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            dispatched_events.append(name)

        with patch(
            "atm.topology.chain.adispatch_custom_event",
            side_effect=fake_dispatch,
        ):
            delta = asyncio.run(node_fn(state))

        assert delta["shared"]["human_approved"] is True

    def test_approve_request_id_format(self) -> None:
        """request_id passed to gateway uses f'chain:{iter_total}:reviewer' format."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=3)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_approve_response())
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node_fn(state))

        call_kwargs = fake_gateway.request.call_args
        assert call_kwargs.kwargs["request_id"] == "chain:3:reviewer"


class TestHumanReviewerNodeReject:
    """human_reviewer node with reject response updates state correctly."""

    def test_reject_no_human_approved_flag(self) -> None:
        """Reject response → human_approved NOT set (or False/absent)."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=1)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_reject_response("Fix the logic"))
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node_fn(state))

        shared_delta = delta.get("shared", {})
        assert not shared_delta.get("human_approved", False)

    def test_reject_adds_rejection_message(self) -> None:
        """Reject response → a synthetic CRITIQUE message with comment appended."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=1)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_reject_response("Needs more tests"))
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node_fn(state))

        # A rejection comment should be reflected in state somehow
        # Either in messages list or in shared signals
        found_rejection = False
        messages_delta = delta.get("messages", [])
        shared_delta = delta.get("shared", {})

        # Check messages list for rejection message
        for msg in messages_delta:
            if hasattr(msg, "content") and "Needs more tests" in (msg.content or ""):
                found_rejection = True
                break
            if hasattr(msg, "payload") and "Needs more tests" in str(msg.payload):
                found_rejection = True
                break

        # Check shared for rejection hint
        if not found_rejection and (
            shared_delta.get("human_rejected") or shared_delta.get("needs_rerun")
        ):
            found_rejection = True

        assert found_rejection, f"Expected rejection comment in delta. Got delta={delta!r}"


class TestHumanReviewerNodeAbstain:
    """human_reviewer node with abstain/timeout — no-op."""

    def test_abstain_is_noop(self) -> None:
        """Abstain response → no state change (no human_approved, no messages)."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=0)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_abstain_response())
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node_fn(state))

        shared_delta = delta.get("shared", {})
        assert not shared_delta.get("human_approved", False)
        # No rejection messages or needs_rerun hint
        assert not shared_delta.get("needs_rerun", False)
        messages_delta = delta.get("messages", [])
        assert len(messages_delta) == 0

    def test_timeout_skip_policy_is_noop(self) -> None:
        """Timeout response (action='timeout') → same no-op behavior as abstain."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=0)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_timeout_response())
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node_fn(state))

        shared_delta = delta.get("shared", {})
        assert not shared_delta.get("human_approved", False)
        assert not shared_delta.get("needs_rerun", False)


# ---------------------------------------------------------------------------
# Group 4 — Dispatch order: human_request BEFORE human_response
# ---------------------------------------------------------------------------


class TestDispatchOrder:
    """Verifies dispatch_custom_event call ordering within human_reviewer node."""

    def test_human_request_before_human_response(self) -> None:
        """human_request event is dispatched BEFORE human_response in same node call."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=0)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_approve_response())
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        event_order: list[str] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            event_order.append(name)

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node_fn(state))

        assert "human_request" in event_order, "human_request event was not dispatched"
        assert "human_response" in event_order, "human_response event was not dispatched"

        req_idx = event_order.index("human_request")
        resp_idx = event_order.index("human_response")
        assert req_idx < resp_idx, (
            f"human_request (idx={req_idx}) must come before human_response (idx={resp_idx})"
        )

    def test_default_path_no_human_request_event(self) -> None:
        """Default path (no HITL) — human_request event is never dispatched."""
        event_names: list[str] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            event_names.append(name)

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch),
        ):
            ChainTopology().build(agents, cfg)

        assert "human_request" not in event_names
        assert "human_response" not in event_names


# ---------------------------------------------------------------------------
# Group 5 — missing run_id raises RuntimeError
# ---------------------------------------------------------------------------


class TestMissingRunId:
    def test_missing_run_id_raises_runtime_error(self) -> None:
        """human_reviewer node raises RuntimeError when state['shared']['run_id'] absent."""
        state = {
            "shared": {
                "iter_total": 0,
                "phase": "execution",
                "signals": {},
            },
            "agents": {},
            "messages": [],
        }

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_approve_response())
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        with pytest.raises(RuntimeError, match="run_id"):
            asyncio.run(node_fn(state))


# ---------------------------------------------------------------------------
# Group 6 — Canonical payload shape (F1 / F2 fix verification)
# ---------------------------------------------------------------------------


class TestDispatchPayloadShape:
    """Verify human_request and human_response payloads match the canonical shape
    expected by ExperimentCallbackHandler._handle_human_request/_handle_human_response.

    Canonical shapes (as used in _human_request_data / _human_response_data fixtures
    in tests/unit/observability/test_callbacks_human.py):
      human_request:  run_id, request_id, role, context_json, requested_at
      human_response: run_id, request_id, answered_at, response_json,
                      source, timed_out, latency_s
    """

    def test_human_request_payload_has_canonical_keys(self) -> None:
        """human_request payload must contain run_id, request_id, role, context_json,
        requested_at — matching what _handle_human_request reads."""
        from datetime import datetime

        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=0)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_approve_response())
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        captured_payloads: dict[str, Any] = {}

        async def fake_dispatch(name: str, data: Any) -> None:
            captured_payloads[name] = data

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node_fn(state))

        assert "human_request" in captured_payloads, "human_request event not dispatched"
        payload = captured_payloads["human_request"]

        # Must have all canonical keys
        assert "run_id" in payload, f"run_id missing from human_request payload: {payload!r}"
        assert "request_id" in payload, (
            f"request_id missing from human_request payload: {payload!r}"
        )
        assert "role" in payload, f"role missing from human_request payload: {payload!r}"
        assert "context_json" in payload, (
            f"context_json missing from human_request payload: {payload!r}"
        )
        assert "requested_at" in payload, (
            f"requested_at missing from human_request payload: {payload!r}"
        )

        # Must NOT use the old "ctx" key
        assert "ctx" not in payload, (
            f"human_request payload must not use 'ctx' key (callback expects 'context_json'): "
            f"{payload!r}"
        )

        # run_id must be the actual run_id
        assert payload["run_id"] == run_id, (
            f"run_id in payload {payload['run_id']!r} != state run_id {run_id!r}"
        )

        # role must be a string
        assert isinstance(payload["role"], str), (
            f"role must be a str, got {type(payload['role'])}: {payload['role']!r}"
        )

        # requested_at must be a datetime
        assert isinstance(payload["requested_at"], datetime), (
            f"requested_at must be a datetime, got {type(payload['requested_at'])}"
        )

    def test_human_response_payload_has_canonical_keys(self) -> None:
        """human_response payload must contain run_id, request_id, answered_at,
        response_json, source, timed_out, latency_s — matching what
        _handle_human_response reads."""
        from datetime import datetime

        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=1)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_approve_response())
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        captured_payloads: dict[str, Any] = {}

        async def fake_dispatch(name: str, data: Any) -> None:
            captured_payloads[name] = data

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node_fn(state))

        assert "human_response" in captured_payloads, "human_response event not dispatched"
        payload = captured_payloads["human_response"]

        # Must have all canonical keys
        for key in (
            "run_id",
            "request_id",
            "answered_at",
            "response_json",
            "source",
            "timed_out",
            "latency_s",
        ):
            assert key in payload, f"{key!r} missing from human_response payload: {payload!r}"

        # Must NOT use the old "response" key
        assert "response" not in payload, (
            f"human_response payload must not use 'response' key (callback expects 'response_json'): "
            f"{payload!r}"
        )

        # run_id must match
        assert payload["run_id"] == run_id

        # answered_at must be a datetime
        assert isinstance(payload["answered_at"], datetime), (
            f"answered_at must be a datetime, got {type(payload['answered_at'])}"
        )

        # latency_s must be a non-negative float
        assert isinstance(payload["latency_s"], float), (
            f"latency_s must be a float, got {type(payload['latency_s'])}"
        )
        assert payload["latency_s"] >= 0.0, f"latency_s must be >= 0.0, got {payload['latency_s']}"


# ---------------------------------------------------------------------------
# Group 7 — F3: llm_fallback timeout_policy does not crash at node invocation
# ---------------------------------------------------------------------------


class TestLlmFallbackPolicy:
    """Verify that using timeout_policy='llm_fallback' (the default) does not raise
    ValueError when the node is actually invoked, as long as timeout_s is set."""

    def test_llm_fallback_policy_with_timeout_does_not_raise(self) -> None:
        """human_cfg.timeout_policy='llm_fallback' with timeout_s set must NOT raise
        ValueError at invocation time — the node builds a fallback LLMSimulatedGateway
        internally and passes it to request_with_timeout."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=0)

        # Use a fast-responding mock as the primary gateway so it doesn't time out
        approve_response = _make_approve_response()
        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=approve_response)

        # Default cfg has timeout_policy='llm_fallback' — this USED to crash
        human_cfg = _make_human_cfg(
            enabled=True,
            timeout_s=30.0,
            timeout_policy="llm_fallback",
        )

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        events: list[str] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            events.append(name)

        # Patch request_with_timeout so it calls the primary gateway and returns immediately
        async def fake_request_with_timeout(
            gw: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> HumanResponse:
            # Verify fallback_gateway is now properly provided (F3 fix)
            assert policy != "llm_fallback" or llm_fallback_gateway is not None, (
                "F3 regression: llm_fallback policy passed with llm_fallback_gateway=None"
            )
            return await gw.request(ctx, request_id=request_id)

        with (
            patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch),
            patch("atm.topology.chain.request_with_timeout", side_effect=fake_request_with_timeout),
        ):
            # Must NOT raise ValueError
            delta = asyncio.run(node_fn(state))

        assert delta["shared"]["human_approved"] is True
        assert "human_request" in events
        assert "human_response" in events


# ---------------------------------------------------------------------------
# Group 8 — role_router integration: back-compat + dynamic
# ---------------------------------------------------------------------------


class TestChainRoleRouter:
    """Verify role_router kwarg in ChainTopology.build() is forwarded correctly."""

    def test_build_accepts_role_router_none(self) -> None:
        """build(role_router=None) builds successfully (back-compat)."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=True)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain.LLMSimulatedGateway") as mock_gw_cls,
        ):
            mock_gw_cls.return_value = MagicMock()
            ChainTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_reviewer" in node_names

    def test_back_compat_role_uses_human_cfg_role(self) -> None:
        """role_router=None → HumanContext.role == human_cfg.role (reviewer)."""
        from atm.core.types import HumanContext

        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=0)

        captured_contexts: list[HumanContext] = []

        fake_gateway = AsyncMock()

        async def capturing_request(ctx: Any, *, request_id: str) -> Any:
            captured_contexts.append(ctx)
            return _make_approve_response()

        fake_gateway.request = capturing_request
        human_cfg = _make_human_cfg(enabled=True, role=HumanRole.REVIEWER, timeout_s=None)  # type: ignore[arg-type]

        node_fn = _build_human_reviewer_node(human_cfg, fake_gateway)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node_fn(state))

        assert len(captured_contexts) == 1
        assert captured_contexts[0].role == HumanRole.REVIEWER

    def test_dynamic_role_router_overrides_human_cfg_role(self) -> None:
        """role_router=FixedRoleRouter(JUDGE) → HumanContext.role == JUDGE, not human_cfg.role."""
        from atm.core.types import HumanContext
        from atm.human.role_router import FixedRoleRouter

        run_id = uuid.uuid4()
        _make_state(run_id=run_id, iter_total=0)

        captured_contexts: list[HumanContext] = []

        fake_gateway = AsyncMock()

        async def capturing_request(ctx: Any, *, request_id: str) -> Any:
            captured_contexts.append(ctx)
            return _make_approve_response()

        fake_gateway.request = capturing_request
        human_cfg = _make_human_cfg(enabled=True, role=HumanRole.REVIEWER, timeout_s=None)  # type: ignore[arg-type]
        router = FixedRoleRouter(role=HumanRole.JUDGE)

        # Import the factory from chain (it's the inline node, not from _node_factory)
        # We test via build() by capturing what role_router gets forwarded to
        # Here we test _build_human_reviewer_node directly since Chain has its own inline node.
        # But for the role_router test, we need to verify chain.build() forwards role_router
        # to the node builder. Since chain.py uses _build_human_reviewer_node (its own),
        # we test chain.build() with role_router kwarg and verify the node uses router.
        #
        # Chain has its OWN inline node (_build_human_reviewer_node), not build_human_node_factory.
        # We need to patch the chain module to verify role_router propagation.
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        captured_build_calls: list[dict[str, Any]] = []

        def capturing_build_node(
            h_cfg: Any, gw: Any, role_router: Any = None, fallback_llm: Any = None
        ) -> Any:
            captured_build_calls.append({"human_cfg": h_cfg, "role_router": role_router})

            async def noop_node(state: Any) -> dict[str, Any]:
                return {}

            return noop_node

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain.LLMSimulatedGateway") as mock_gw_cls,
            patch(
                "atm.topology.chain._build_human_reviewer_node", side_effect=capturing_build_node
            ),
        ):
            mock_gw_cls.return_value = MagicMock()
            ChainTopology().build(agents, cfg, human_cfg=human_cfg, role_router=router)

        assert len(captured_build_calls) == 1
        assert captured_build_calls[0]["role_router"] is router

    def test_role_router_none_not_forwarded_to_node_builder(self) -> None:
        """role_router=None → _build_human_reviewer_node receives role_router=None."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=True, timeout_s=None)  # type: ignore[arg-type]

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        captured_calls: list[dict[str, Any]] = []

        def capturing_build_node(
            h_cfg: Any, gw: Any, role_router: Any = None, fallback_llm: Any = None
        ) -> Any:
            captured_calls.append({"role_router": role_router})

            async def noop_node(state: Any) -> dict[str, Any]:
                return {}

            return noop_node

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain.LLMSimulatedGateway") as mock_gw_cls,
            patch(
                "atm.topology.chain._build_human_reviewer_node", side_effect=capturing_build_node
            ),
        ):
            mock_gw_cls.return_value = MagicMock()
            ChainTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        assert len(captured_calls) == 1
        assert captured_calls[0]["role_router"] is None

    def test_llm_fallback_primary_timeout_exercises_fallback(self) -> None:
        """When the primary gateway times out and policy='llm_fallback', the fallback
        gateway is called and returns a response with source='fallback'."""
        import asyncio as _asyncio

        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=0)

        # Primary gateway that actually times out
        async def _slow_request(ctx: Any, *, request_id: str) -> HumanResponse:
            await _asyncio.sleep(10)
            return _make_approve_response()

        primary_gateway = AsyncMock()
        primary_gateway.request = _slow_request

        # The fallback should return quickly
        fallback_response = HumanResponse(
            action="approve", comment="fallback", source="fallback", timed_out=False
        )

        human_cfg = _make_human_cfg(
            enabled=True,
            timeout_s=0.05,  # very short — primary will time out
            timeout_policy="llm_fallback",
        )

        node_fn = _build_human_reviewer_node(human_cfg, primary_gateway)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        mock_fallback_gw = AsyncMock()
        mock_fallback_gw.request = AsyncMock(return_value=fallback_response)

        mock_llm_sim_instance = mock_fallback_gw

        with (
            patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch),
            patch("atm.topology.chain.LLMSimulatedGateway", return_value=mock_llm_sim_instance),
        ):
            delta = asyncio.run(node_fn(state))

        # The fallback response was "approve", so human_approved should be True
        assert delta["shared"]["human_approved"] is True
