"""LLMSimulatedGateway — LLM-driven implementation of HumanGateway (M9 Step 2.1).

This gateway uses an LLMWrapper to simulate human responses for testing and
offline experiments.  It satisfies the HumanGateway Protocol and enforces
in-process idempotency via a cache keyed on (run_id, request_id).

Idempotency layers
------------------
1. **In-process**: ``dict[(run_id, request_id), HumanResponse]`` + asyncio.Lock.
   A second call with the same key returns the cached response immediately,
   without issuing another LLM request.

2. **Database**: enforced externally by ``ExperimentCallbackHandler`` via
   ``INSERT ... ON CONFLICT ON CONSTRAINT uq_human_interactions_run_request DO NOTHING``.

Retry on invalid JSON
---------------------
If the LLM returns text that cannot be parsed as a valid HumanResponse JSON
object, the gateway retries **once** with a stricter prompt that explicitly
instructs the model to output only JSON.  If the second attempt also fails,
a ``ValueError`` is raised with a descriptive message.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from atm.core.types import HumanContext, HumanResponse
from atm.human.prompts import build_role_prompt
from atm.llm.wrapper import LLMWrapper

__all__ = ["LLMSimulatedGateway"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strict JSON retry prompt suffix
# ---------------------------------------------------------------------------

_STRICT_JSON_SUFFIX = (
    "\n\n"
    "IMPORTANT: Your previous response could not be parsed as JSON. "
    "You MUST respond with a single valid JSON object and NOTHING else. "
    "Do not include markdown fences, prose, or explanations. "
    'Example: {"action": "approve", "payload": {}, "comment": "LGTM"}'
)


class LLMSimulatedGateway:
    """Simulates a human reviewer/judge/etc. using an LLM.

    Satisfies the :class:`~atm.human.gateway.HumanGateway` Protocol via
    structural subtyping (no explicit inheritance required).

    Args:
        llm: An :class:`~atm.llm.wrapper.LLMWrapper` instance.  In tests,
             pass one wrapping a ``FakeLLM(mode='scripted')``.
    """

    def __init__(self, llm: LLMWrapper) -> None:
        self._llm = llm
        # Cache: (run_id, request_id) → HumanResponse
        self._cache: dict[tuple[UUID, str], HumanResponse] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # HumanGateway.request implementation
    # ------------------------------------------------------------------

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        """Simulate a human decision via LLM.

        Idempotency: if ``(ctx.run_id, request_id)`` was already resolved in
        this process, the cached :class:`HumanResponse` is returned immediately.

        Parameters
        ----------
        ctx:
            Full decision context including role, question, allowed actions,
            recent messages, and run_id.
        request_id:
            Stable caller-assigned identifier.  Re-used across LangGraph
            retries to prevent duplicate LLM calls.

        Returns
        -------
        HumanResponse
            Parsed response with ``source='llm_sim'`` and ``timed_out=False``.
        """
        cache_key = (ctx.run_id, request_id)

        # --- In-process idempotency check ---
        async with self._lock:
            if cache_key in self._cache:
                logger.debug(
                    "LLMSimulatedGateway: cache hit for run_id=%s request_id=%s",
                    ctx.run_id,
                    request_id,
                )
                return self._cache[cache_key]

        # --- Build prompts ---
        system_prompt, user_prompt = build_role_prompt(ctx.role, ctx)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # --- First LLM call ---
        llm_response = await self._llm.ainvoke(
            messages,
            agent_id="llm_simulator",
        )
        raw_text = llm_response.text or ""

        # --- Parse; retry once on invalid JSON ---
        parsed = _try_parse_response(raw_text)
        if parsed is None:
            logger.warning(
                "LLMSimulatedGateway: first response was not valid JSON; retrying "
                "(run_id=%s, request_id=%s)",
                ctx.run_id,
                request_id,
            )
            retry_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt + _STRICT_JSON_SUFFIX),
            ]
            llm_response = await self._llm.ainvoke(
                retry_messages,
                agent_id="llm_simulator",
            )
            raw_text = llm_response.text or ""
            parsed = _try_parse_response(raw_text)

        if parsed is None:
            raise ValueError(
                f"LLMSimulatedGateway: LLM returned invalid JSON after retry. "
                f"Last response text: {raw_text!r}"
            )

        # --- Build HumanResponse ---
        human_response = HumanResponse(
            action=parsed.get("action", "abstain"),
            comment=parsed.get("comment"),
            payload=parsed.get("payload") or {},
            source="llm_sim",
            timed_out=False,
        )

        # --- Cache and return ---
        async with self._lock:
            self._cache[cache_key] = human_response

        logger.debug(
            "LLMSimulatedGateway: resolved action=%r for run_id=%s request_id=%s",
            human_response.action,
            ctx.run_id,
            request_id,
        )
        return human_response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_parse_response(text: str) -> dict[str, Any] | None:
    """Attempt to parse ``text`` as a JSON object.

    Returns the parsed dict if successful, or ``None`` on any parse failure.
    Also returns ``None`` if the parsed value is not a dict (e.g. a JSON array
    or primitive).
    """
    text = text.strip()
    if not text:
        return None
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    return result
