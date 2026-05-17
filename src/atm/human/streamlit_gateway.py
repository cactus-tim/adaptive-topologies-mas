"""StreamlitHumanGateway — Streamlit HITL gateway for user-study sessions (M14 §3.1).

Implements the HumanGateway Protocol by persisting requests to the Postgres
``human_request_queue`` table and polling for responses placed by the Streamlit
UI.

Flow (one async request):
  1. Check in-process idempotency cache; return cached response if found.
  2. Enqueue request via ``HumanRequestQueue.enqueue(...)`` — idempotent at DB
     level (ON CONFLICT DO NOTHING).
  3. ``await queue.wait_for_response(...)`` — polls every 1 second.
  4. On ``None`` (timeout): return ``HumanResponse(timed_out=True, source='timeout',
     action='timeout')``.
  5. On response dict: parse action/comment/payload/tlx_scores; ensure
     ``payload["study_session_id"]`` is set (falls back to ``self._study_session_id``
     when the UI omits it).

Idempotency:
  Same ``(run_id, request_id)`` pair is served from an in-process
  ``dict[tuple[UUID, str], HumanResponse]`` cache on the second call — no
  re-enqueue, no extra DB poll.

Timeout policy:
  The gateway simply returns a ``timed_out=True`` response; choosing a fallback
  action is the responsibility of the topology (consistent with M9 design).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from atm.core.types import HumanContext, HumanResponse
from atm.experiment.config import HumanCfg
from atm.human._queue import HumanRequestQueue

__all__ = ["StreamlitHumanGateway"]

logger = logging.getLogger(__name__)


class StreamlitHumanGateway:
    """Bridges the runner to the Streamlit HITL UI via a Postgres request queue.

    Satisfies the :class:`~atm.human.gateway.HumanGateway` Protocol via
    structural subtyping (no explicit inheritance required).

    Parameters
    ----------
    queue:
        Configured :class:`~atm.human._queue.HumanRequestQueue` instance.
    human_cfg:
        ``HumanCfg`` from experiment config; supplies ``timeout_s``,
        ``study_session_id``, and ``participant_id``.
    study_session_id:
        Explicit override for the study session ID; when provided, takes
        precedence over ``human_cfg.study_session_id``.  Populated by the
        runner from its session-management logic.
    participant_id:
        Identifier of the human participant in the study session.  Stored
        as metadata; not used for routing.
    """

    def __init__(
        self,
        queue: HumanRequestQueue,
        *,
        human_cfg: HumanCfg,
        study_session_id: str | None = None,
        participant_id: str | None = None,
    ) -> None:
        self._queue = queue
        self._cfg = human_cfg
        # Explicit kwarg takes precedence over cfg field
        self._study_session_id: str | None = (
            study_session_id if study_session_id is not None else human_cfg.study_session_id
        )
        self._participant_id: str | None = (
            participant_id if participant_id is not None else human_cfg.participant_id
        )
        # In-process idempotency cache: (run_id, request_id) → HumanResponse
        self._cache: dict[tuple[UUID, str], HumanResponse] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # HumanGateway Protocol
    # ------------------------------------------------------------------

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        """Enqueue a request and wait for the Streamlit UI to respond.

        Parameters
        ----------
        ctx:
            Full context describing the decision point (run_id, role, question,
            recent messages, allowed actions, optional deadline).
        request_id:
            Caller-assigned stable identifier for this interaction.  Used to
            enforce idempotency at both application and database level.

        Returns
        -------
        HumanResponse
            The human participant's decision, or a ``timed_out=True`` response
            if ``human_cfg.timeout_s`` expires before the UI submits an answer.
        """
        cache_key = (ctx.run_id, request_id)

        # --- In-process idempotency check ---
        async with self._lock:
            if cache_key in self._cache:
                logger.debug(
                    "StreamlitHumanGateway: cache hit for run_id=%s request_id=%s",
                    ctx.run_id,
                    request_id,
                )
                return self._cache[cache_key]

        # --- Enqueue (idempotent at DB level) ---
        context_json: dict[str, Any] = ctx.model_dump(mode="json")
        await self._queue.enqueue(
            run_id=ctx.run_id,
            request_id=request_id,
            context_json=context_json,
        )

        logger.debug(
            "StreamlitHumanGateway: waiting for response run_id=%s request_id=%s timeout_s=%s",
            ctx.run_id,
            request_id,
            self._cfg.timeout_s,
        )

        # --- Poll for response ---
        timeout_s: float = self._cfg.timeout_s if self._cfg.timeout_s is not None else 900.0
        raw: dict[str, Any] | None = await self._queue.wait_for_response(
            run_id=ctx.run_id,
            request_id=request_id,
            timeout_s=timeout_s,
        )

        # --- Build HumanResponse ---
        response: HumanResponse
        if raw is None:
            # Timeout — let topology decide fallback action
            response = HumanResponse(
                action="timeout",
                timed_out=True,
                source="timeout",
                comment=None,
                payload={},
            )
            logger.warning(
                "StreamlitHumanGateway: timed out waiting for run_id=%s request_id=%s",
                ctx.run_id,
                request_id,
            )
        else:
            response = self._parse_response(raw)
            logger.debug(
                "StreamlitHumanGateway: received action=%r for run_id=%s request_id=%s",
                response.action,
                ctx.run_id,
                request_id,
            )

        # --- Cache and return ---
        async with self._lock:
            self._cache[cache_key] = response

        return self._cache[cache_key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> HumanResponse:
        """Parse a response dict submitted by the Streamlit UI into a HumanResponse.

        Parameters
        ----------
        raw:
            Dict stored by the UI in ``human_request_queue.response_json``.

        Returns
        -------
        HumanResponse
            With ``source='human'``, ``timed_out=False``.
        """
        action: str = str(raw.get("action", "abstain")).strip()
        comment: str | None = raw.get("comment") or None
        if isinstance(comment, str):
            comment = comment.strip() or None

        payload: dict[str, Any] = raw.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        # Propagate study_session_id: prefer UI-supplied, fall back to gateway cfg
        ui_session_id: str | None = raw.get("study_session_id") or None
        effective_session_id: str | None = (
            ui_session_id if ui_session_id is not None else self._study_session_id
        )
        if effective_session_id is not None:
            payload = {**payload, "study_session_id": effective_session_id}

        # NASA-TLX scores (optional)
        tlx_raw = raw.get("tlx_scores")
        tlx_scores: dict[str, int] | None = None
        if isinstance(tlx_raw, dict):
            tlx_scores = {str(k): int(v) for k, v in tlx_raw.items()}

        return HumanResponse(
            action=action,
            comment=comment,
            payload=payload,
            source="human",
            timed_out=False,
            tlx_scores=tlx_scores,
        )
