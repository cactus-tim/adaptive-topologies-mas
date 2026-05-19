"""LLMSimulatedGateway — LLM-driven implementation of HumanGateway."""

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

_STRICT_JSON_SUFFIX = (
    "\n\n"
    "IMPORTANT: Your previous response could not be parsed as JSON. "
    "You MUST respond with a single valid JSON object and NOTHING else. "
    "Do not include markdown fences, prose, or explanations. "
    'Example: {"action": "approve", "payload": {}, "comment": "LGTM"}'
)


class LLMSimulatedGateway:
    """Simulates a human reviewer using an LLM; idempotent on (run_id, request_id)."""

    def __init__(self, llm: LLMWrapper) -> None:
        self._llm = llm
        self._cache: dict[tuple[UUID, str], HumanResponse] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        """Simulate a human decision via LLM; returns cached response on repeat calls."""
        cache_key = (ctx.run_id, request_id)

        async with self._lock:
            if cache_key in self._cache:
                logger.debug(
                    "LLMSimulatedGateway: cache hit for run_id=%s request_id=%s",
                    ctx.run_id,
                    request_id,
                )
                return self._cache[cache_key]

        system_prompt, user_prompt = build_role_prompt(ctx.role, ctx)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        llm_response = await self._llm.ainvoke(
            messages,
            agent_id="llm_simulator",
        )
        raw_text = llm_response.text or ""

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

        human_response = HumanResponse(
            action=parsed.get("action", "abstain"),
            comment=parsed.get("comment"),
            payload=parsed.get("payload") or {},
            source="llm_sim",
            timed_out=False,
        )

        async with self._lock:
            self._cache[cache_key] = human_response

        logger.debug(
            "LLMSimulatedGateway: resolved action=%r for run_id=%s request_id=%s",
            human_response.action,
            ctx.run_id,
            request_id,
        )
        return human_response


def _try_parse_response(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as a JSON dict; returns None on failure."""
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
