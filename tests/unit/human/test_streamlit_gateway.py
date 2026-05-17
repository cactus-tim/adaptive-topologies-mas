"""Unit tests for atm.human.streamlit_gateway.StreamlitHumanGateway.

All tests use a FakeHumanRequestQueue — no live Postgres required.

Coverage:
  - success path: enqueue called, wait_for_response returns dict → HumanResponse(source='human')
  - timeout path: wait_for_response returns None → HumanResponse(timed_out=True, source='timeout')
  - idempotency: second call with same (run_id, request_id) returns cached response without
    re-enqueueing
  - study_session_id propagation: payload includes study_session_id from gateway config when
    UI omits it
  - study_session_id from payload: UI-supplied study_session_id is preserved in response payload
  - tlx_scores propagation: tlx_scores dict from payload is copied to HumanResponse.tlx_scores
  - context_json passed to enqueue contains model_dump of HumanContext
  - action / comment / payload fields parsed correctly from response dict
"""

from __future__ import annotations

import uuid
from typing import Any

from atm.core.types import HumanContext, HumanResponse, HumanRole
from atm.experiment.config import HumanCfg
from atm.human._queue import HumanRequestQueue
from atm.human.streamlit_gateway import StreamlitHumanGateway

# ---------------------------------------------------------------------------
# Fake queue — subclass with overridden async methods
# ---------------------------------------------------------------------------


class FakeQueue(HumanRequestQueue):
    """Fake HumanRequestQueue that avoids all DB I/O.

    Caller configures ``_next_response`` before calling ``wait_for_response``:
      - dict → returned as the response payload
      - None → simulates timeout

    ``enqueue_calls`` records every call to ``enqueue``.
    """

    def __init__(self, next_response: dict[str, Any] | None = None) -> None:
        # Do NOT call super().__init__() — no session factory needed.
        self._next_response = next_response
        self.enqueue_calls: list[tuple[uuid.UUID, str, dict[str, Any]]] = []
        self.wait_calls: list[tuple[uuid.UUID, str, float]] = []

    async def enqueue(
        self,
        run_id: uuid.UUID,
        request_id: str,
        context_json: dict[str, Any],
    ) -> bool:
        self.enqueue_calls.append((run_id, request_id, context_json))
        return True

    async def wait_for_response(
        self,
        run_id: uuid.UUID,
        request_id: str,
        *,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        self.wait_calls.append((run_id, request_id, timeout_s))
        return self._next_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    timeout_s: float = 30.0,
    study_session_id: str | None = None,
    participant_id: str | None = None,
) -> HumanCfg:
    return HumanCfg(
        gateway="streamlit",
        timeout_s=timeout_s,
        study_session_id=study_session_id,
        participant_id=participant_id,
    )


def _make_ctx(run_id: uuid.UUID | None = None) -> HumanContext:
    return HumanContext(
        run_id=run_id or uuid.uuid4(),
        role=HumanRole.REVIEWER,
        question="Should we proceed?",
        recent_messages=(),
        allowed_actions=("approve", "reject"),
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestSuccessPath:
    async def test_returns_human_response_on_success(self) -> None:
        """When wait_for_response returns a dict, response has source='human'."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg()
        fake_q = FakeQueue(
            next_response={
                "action": "approve",
                "comment": "Looks good",
                "payload": {},
                "study_session_id": "sess-123",
            }
        )
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        resp = await gw.request(ctx, request_id="req-1")

        assert isinstance(resp, HumanResponse)
        assert resp.source == "human"
        assert resp.timed_out is False
        assert resp.action == "approve"
        assert resp.comment == "Looks good"

    async def test_enqueue_called_with_context_json(self) -> None:
        """enqueue must be called once with ctx.model_dump(mode='json') as context_json."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg()
        fake_q = FakeQueue(next_response={"action": "reject", "study_session_id": "s1"})
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        await gw.request(ctx, request_id="req-2")

        assert len(fake_q.enqueue_calls) == 1
        enq_run_id, enq_req_id, enq_ctx = fake_q.enqueue_calls[0]
        assert enq_run_id == run_id
        assert enq_req_id == "req-2"
        # context_json should be a dict with at least run_id
        assert "run_id" in enq_ctx

    async def test_wait_called_with_correct_timeout(self) -> None:
        """wait_for_response is called with timeout_s from HumanCfg."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg(timeout_s=120.0)
        fake_q = FakeQueue(next_response={"action": "approve", "study_session_id": "s2"})
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        await gw.request(ctx, request_id="req-3")

        assert len(fake_q.wait_calls) == 1
        _run_id, _req_id, timeout = fake_q.wait_calls[0]
        assert timeout == 120.0

    async def test_tlx_scores_copied_from_payload(self) -> None:
        """tlx_scores in response dict are propagated to HumanResponse.tlx_scores."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg()
        tlx = {
            "mental": 60,
            "physical": 30,
            "temporal": 50,
            "effort": 70,
            "performance": 40,
            "frustration": 20,
        }
        fake_q = FakeQueue(
            next_response={
                "action": "approve",
                "tlx_scores": tlx,
                "study_session_id": "s3",
            }
        )
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        resp = await gw.request(ctx, request_id="req-4")

        assert resp.tlx_scores == tlx


# ---------------------------------------------------------------------------
# Timeout path
# ---------------------------------------------------------------------------


class TestTimeoutPath:
    async def test_timeout_returns_timed_out_response(self) -> None:
        """When wait_for_response returns None, response has timed_out=True and source='timeout'."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg(timeout_s=5.0)
        fake_q = FakeQueue(next_response=None)
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        resp = await gw.request(ctx, request_id="req-timeout")

        assert resp.timed_out is True
        assert resp.source == "timeout"
        assert resp.action == "timeout"

    async def test_timeout_response_is_cached(self) -> None:
        """A timed-out response is still cached for idempotency."""
        ctx = _make_ctx()
        cfg = _make_cfg(timeout_s=1.0)
        fake_q = FakeQueue(next_response=None)
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        resp1 = await gw.request(ctx, request_id="req-t2")
        resp2 = await gw.request(ctx, request_id="req-t2")

        # enqueue should only be called once
        assert len(fake_q.enqueue_calls) == 1
        assert resp1 is resp2


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    async def test_second_call_returns_cached_response(self) -> None:
        """Second call with same (run_id, request_id) returns cached response."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg()
        fake_q = FakeQueue(next_response={"action": "approve", "study_session_id": "s4"})
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        resp1 = await gw.request(ctx, request_id="req-idem")
        resp2 = await gw.request(ctx, request_id="req-idem")

        # Same object returned from cache
        assert resp1 is resp2

    async def test_second_call_does_not_re_enqueue(self) -> None:
        """Second call must NOT call enqueue again."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg()
        fake_q = FakeQueue(next_response={"action": "reject", "study_session_id": "s5"})
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        await gw.request(ctx, request_id="req-idem2")
        await gw.request(ctx, request_id="req-idem2")

        assert len(fake_q.enqueue_calls) == 1

    async def test_different_request_ids_are_separate_entries(self) -> None:
        """Different request_ids for same run_id are independent (no false cache hit)."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg()
        fake_q = FakeQueue(next_response={"action": "approve", "study_session_id": "s6"})
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        await gw.request(ctx, request_id="req-A")
        await gw.request(ctx, request_id="req-B")

        assert len(fake_q.enqueue_calls) == 2


# ---------------------------------------------------------------------------
# study_session_id propagation
# ---------------------------------------------------------------------------


class TestStudySessionId:
    async def test_session_id_from_cfg_used_when_payload_omits_it(self) -> None:
        """When UI response has no study_session_id, gateway fills it from cfg."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg(study_session_id="cfg-session-42")
        # payload deliberately omits study_session_id
        fake_q = FakeQueue(next_response={"action": "approve"})
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg, study_session_id="cfg-session-42")

        resp = await gw.request(ctx, request_id="req-sess-1")

        assert resp.payload.get("study_session_id") == "cfg-session-42"

    async def test_session_id_from_payload_is_preserved(self) -> None:
        """When UI includes study_session_id in response, it is preserved."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg(study_session_id="cfg-session-99")
        fake_q = FakeQueue(
            next_response={
                "action": "reject",
                "study_session_id": "ui-session-7",
            }
        )
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg, study_session_id="cfg-session-99")

        resp = await gw.request(ctx, request_id="req-sess-2")

        assert resp.payload.get("study_session_id") == "ui-session-7"

    async def test_no_session_id_in_cfg_and_payload(self) -> None:
        """When neither cfg nor payload has study_session_id, field is absent or None."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg(study_session_id=None)
        fake_q = FakeQueue(next_response={"action": "approve"})
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)

        resp = await gw.request(ctx, request_id="req-sess-3")

        # No study_session_id in payload (or it's None)
        assert resp.payload.get("study_session_id") is None

    async def test_study_session_id_from_constructor_kwarg(self) -> None:
        """study_session_id constructor kwarg overrides cfg when explicitly passed."""
        run_id = uuid.uuid4()
        ctx = _make_ctx(run_id=run_id)
        cfg = _make_cfg(study_session_id="cfg-session-from-cfg")
        fake_q = FakeQueue(next_response={"action": "approve"})
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg, study_session_id="override-session")

        resp = await gw.request(ctx, request_id="req-sess-4")

        assert resp.payload.get("study_session_id") == "override-session"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_instantiation_with_minimal_args(self) -> None:
        """StreamlitHumanGateway can be constructed with queue and human_cfg only."""
        cfg = _make_cfg()
        fake_q = FakeQueue()
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)
        assert gw is not None

    def test_instantiation_with_all_args(self) -> None:
        """All optional constructor kwargs are accepted."""
        cfg = _make_cfg(study_session_id="s-init", participant_id="p-init")
        fake_q = FakeQueue()
        gw = StreamlitHumanGateway(
            fake_q,
            human_cfg=cfg,
            study_session_id="s-init",
            participant_id="p-init",
        )
        assert gw is not None


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_human_gateway_protocol(self) -> None:
        """StreamlitHumanGateway satisfies the HumanGateway Protocol (structural check)."""
        from atm.human.gateway import HumanGateway

        cfg = _make_cfg()
        fake_q = FakeQueue()
        gw = StreamlitHumanGateway(fake_q, human_cfg=cfg)
        assert isinstance(gw, HumanGateway)
