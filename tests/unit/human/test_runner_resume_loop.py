"""Unit tests for atm.human.runner.run_with_human (M9 Step 5.1).

Test scenarios:
1. Passthrough (no interrupt)      — graph returns immediately, no gateway call.
2. Single interrupt/resume         — graph interrupts once; gateway resolves; final state OK.
3. Multi interrupt/resume          — graph interrupts twice; each resolved in turn.
4. Max-interactions guard          — exceeding max_interactions raises MaxInteractionsExceededError.
5. Idempotency (same request_id)   — crashed/replayed graph re-interrupts with the same
                                     request_id; gateway is called only once.
6. >1 interrupts simultaneously    — raises RuntimeError.
7. Interrupt but no gateway        — raises ValueError.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from atm.core.types import HumanContext, HumanResponse, HumanRole
from atm.human.runner import MaxInteractionsExceededError, run_with_human

# ---------------------------------------------------------------------------
# Helpers: Fake graph stubs
# ---------------------------------------------------------------------------


class _FakeNoInterruptGraph:
    """Graph that returns a fixed final state without any interrupt."""

    def __init__(self, final_state: dict[str, Any]) -> None:
        self._final_state = final_state

    async def ainvoke(self, state: Any, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        # The initial_state is echoed back (merged) — return just the final state.
        return dict(self._final_state)


class _FakeSingleInterruptGraph:
    """Graph that interrupts once with a fixed payload, then returns final state on resume."""

    def __init__(
        self,
        interrupt_payload: dict[str, Any],
        final_state: dict[str, Any],
    ) -> None:
        self._interrupt_payload = interrupt_payload
        self._final_state = final_state
        self._call_count = 0

    async def ainvoke(self, state: Any, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            # First call: return state with __interrupt__
            return _make_interrupt_result(self._interrupt_payload)
        # Second call (resume): return final state
        return dict(self._final_state)


class _FakeMultiInterruptGraph:
    """Graph that interrupts N times (with distinct payloads), then returns final state."""

    def __init__(
        self,
        interrupt_payloads: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> None:
        self._payloads = list(interrupt_payloads)
        self._final_state = final_state
        self._call_count = 0

    async def ainvoke(self, state: Any, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self._call_count += 1
        idx = self._call_count - 1
        if idx < len(self._payloads):
            return _make_interrupt_result(self._payloads[idx])
        return dict(self._final_state)


class _FakeReplayInterruptGraph:
    """Simulates a crash-recovery scenario.

    First two calls both emit an interrupt with the SAME request_id (as if
    the graph replays from checkpoint), the third call returns final state.
    """

    def __init__(
        self,
        interrupt_payload: dict[str, Any],
        final_state: dict[str, Any],
    ) -> None:
        self._interrupt_payload = interrupt_payload
        self._final_state = final_state
        self._call_count = 0

    async def ainvoke(self, state: Any, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self._call_count += 1
        if self._call_count <= 2:
            # Both first and second calls emit the same interrupt (replayed)
            return _make_interrupt_result(self._interrupt_payload)
        return dict(self._final_state)


class _FakeMultiSimultaneousInterruptGraph:
    """Graph that returns two interrupts in a single result (pathological)."""

    async def ainvoke(self, state: Any, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "value": "initial",
            "__interrupt__": [
                _make_interrupt_obj({"request_id": "r1", "ctx": _ctx_dict()}),
                _make_interrupt_obj({"request_id": "r2", "ctx": _ctx_dict()}),
            ],
        }


# ---------------------------------------------------------------------------
# Helpers: interrupt object / payload factories
# ---------------------------------------------------------------------------


class _FakeInterrupt:
    """Mimics langgraph.types.Interrupt(value=...)."""

    def __init__(self, value: Any) -> None:
        self.value = value


def _make_interrupt_obj(payload: dict[str, Any]) -> _FakeInterrupt:
    return _FakeInterrupt(value=payload)


def _make_interrupt_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": "interrupted",
        "__interrupt__": [_make_interrupt_obj(payload)],
    }


def _ctx_dict(run_id: uuid.UUID | None = None) -> dict[str, Any]:
    """Return a HumanContext serialised as a JSON-compatible dict."""
    run_id = run_id or uuid.uuid4()
    ctx = HumanContext(
        run_id=run_id,
        role=HumanRole.REVIEWER,
        question="Please review.",
        recent_messages=(),
        allowed_actions=("approve", "reject", "abstain"),
    )
    return ctx.model_dump(mode="json")


def _make_gateway(action: str = "approve") -> AsyncMock:
    """Return a mock HumanGateway whose request() returns a fixed approve response."""
    gw = AsyncMock()
    gw.request = AsyncMock(
        return_value=HumanResponse(action=action, source="llm_sim", timed_out=False)
    )
    return gw


# ---------------------------------------------------------------------------
# 1. Passthrough — no interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passthrough_no_interrupt_returns_final_state() -> None:
    """Graph with no interrupt → run_with_human returns the final state unchanged."""
    final = {"shared": {"answer": "42"}, "messages": []}
    graph = _FakeNoInterruptGraph(final)
    gw = _make_gateway()

    result = await run_with_human(
        graph,
        {"shared": {}},
        thread_id="thread-pass",
        gateway=gw,
    )

    assert result == final
    gw.request.assert_not_called()


@pytest.mark.asyncio
async def test_passthrough_no_gateway_no_interrupt() -> None:
    """Graph with no interrupt → run_with_human works even without gateway=None."""
    final = {"ok": True}
    graph = _FakeNoInterruptGraph(final)

    result = await run_with_human(
        graph,
        {"ok": False},
        thread_id="thread-no-gw",
        gateway=None,
    )

    assert result == final


# ---------------------------------------------------------------------------
# 2. Single interrupt/resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_interrupt_resolves_via_gateway() -> None:
    """Graph interrupts once; gateway returns approve; final state returned."""
    run_id = uuid.uuid4()
    payload = {"request_id": "req-001", "ctx": _ctx_dict(run_id)}
    final = {"shared": {"human_approved": True}}
    graph = _FakeSingleInterruptGraph(payload, final)
    gw = _make_gateway("approve")

    result = await run_with_human(
        graph,
        {"shared": {"run_id": str(run_id)}},
        thread_id="thread-single",
        gateway=gw,
    )

    assert result == final
    gw.request.assert_called_once()
    call_kwargs = gw.request.call_args
    # request_id should be passed through
    assert call_kwargs.kwargs["request_id"] == "req-001"


@pytest.mark.asyncio
async def test_single_interrupt_ctx_is_validated() -> None:
    """The HumanContext in the interrupt payload is reconstructed via model_validate."""
    run_id = uuid.uuid4()
    ctx_data = _ctx_dict(run_id)
    payload = {"request_id": "req-ctx", "ctx": ctx_data}
    final = {"done": True}
    graph = _FakeSingleInterruptGraph(payload, final)

    received_ctx: list[HumanContext] = []

    async def capture_request(ctx: HumanContext, *, request_id: str) -> HumanResponse:
        received_ctx.append(ctx)
        return HumanResponse(action="approve", source="llm_sim")

    gw = AsyncMock()
    gw.request = capture_request

    await run_with_human(
        graph,
        {},
        thread_id="thread-ctx",
        gateway=gw,
    )

    assert len(received_ctx) == 1
    assert isinstance(received_ctx[0], HumanContext)
    assert received_ctx[0].run_id == run_id


# ---------------------------------------------------------------------------
# 3. Multi interrupt/resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_interrupt_all_resolved() -> None:
    """Graph interrupts twice; both are resolved; final state returned."""
    run_id = uuid.uuid4()
    payloads = [
        {"request_id": "req-A", "ctx": _ctx_dict(run_id)},
        {"request_id": "req-B", "ctx": _ctx_dict(run_id)},
    ]
    final = {"iterations": 2}
    graph = _FakeMultiInterruptGraph(payloads, final)
    gw = _make_gateway("abstain")

    result = await run_with_human(
        graph,
        {},
        thread_id="thread-multi",
        gateway=gw,
        max_interactions=5,
    )

    assert result == final
    assert gw.request.call_count == 2


# ---------------------------------------------------------------------------
# 4. Max-interactions guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_interactions_exceeded_raises_error() -> None:
    """Exceeding max_interactions raises MaxInteractionsExceededError."""
    run_id = uuid.uuid4()
    # Graph interrupts infinitely (different request_ids to bypass idempotency cache)
    payloads = [
        {"request_id": f"req-{i}", "ctx": _ctx_dict(run_id)}
        for i in range(20)
    ]
    final = {"done": True}
    # _FakeMultiInterruptGraph will produce `len(payloads)` interrupts,
    # but max_interactions=3 should stop us before we reach final state.
    graph = _FakeMultiInterruptGraph(payloads, final)
    gw = _make_gateway("approve")

    with pytest.raises(MaxInteractionsExceededError) as exc_info:
        await run_with_human(
            graph,
            {},
            thread_id="thread-max",
            gateway=gw,
            max_interactions=3,
        )

    err = exc_info.value
    assert err.max_interactions == 3
    assert "thread-max" in str(err)


@pytest.mark.asyncio
async def test_max_interactions_zero_raises_on_first_interrupt() -> None:
    """max_interactions=0 should raise on the very first interrupt."""
    run_id = uuid.uuid4()
    payload = {"request_id": "req-zero", "ctx": _ctx_dict(run_id)}
    final = {"done": True}
    graph = _FakeSingleInterruptGraph(payload, final)
    gw = _make_gateway()

    with pytest.raises(MaxInteractionsExceededError):
        await run_with_human(
            graph,
            {},
            thread_id="thread-zero",
            gateway=gw,
            max_interactions=0,
        )


# ---------------------------------------------------------------------------
# 5. Idempotency — same request_id on replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_same_request_id_not_double_called() -> None:
    """If a replay emits the same interrupt twice, gateway is called only once."""
    run_id = uuid.uuid4()
    # Same payload / request_id on both interrupts (simulated crash-recovery)
    payload = {"request_id": "req-idem", "ctx": _ctx_dict(run_id)}
    final = {"idempotent": True}
    graph = _FakeReplayInterruptGraph(payload, final)
    gw = _make_gateway("approve")

    result = await run_with_human(
        graph,
        {},
        thread_id="thread-idem",
        gateway=gw,
        max_interactions=5,
    )

    assert result == final
    # Gateway must have been called exactly once despite two interrupts with the same request_id
    gw.request.assert_called_once()


# ---------------------------------------------------------------------------
# 6. >1 simultaneous interrupts → RuntimeError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_simultaneous_interrupts_raises_runtime_error() -> None:
    """If a single invocation returns >1 Interrupt, RuntimeError is raised."""
    graph = _FakeMultiSimultaneousInterruptGraph()
    gw = _make_gateway()

    with pytest.raises(RuntimeError, match="2 simultaneous interrupts"):
        await run_with_human(
            graph,
            {},
            thread_id="thread-multi-sim",
            gateway=gw,
        )


# ---------------------------------------------------------------------------
# 7. Interrupt with no gateway → ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_without_gateway_raises_value_error() -> None:
    """If the graph interrupts but gateway=None, ValueError is raised."""
    run_id = uuid.uuid4()
    payload = {"request_id": "req-no-gw", "ctx": _ctx_dict(run_id)}
    final = {"done": True}
    graph = _FakeSingleInterruptGraph(payload, final)

    with pytest.raises(ValueError, match="gateway=None"):
        await run_with_human(
            graph,
            {},
            thread_id="thread-no-gw",
            gateway=None,
        )
