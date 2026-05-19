"""Shared LLM-judge helper (_invoke_judge) for task evaluators."""

from __future__ import annotations

import json
import re
from typing import Any

from atm.tasks.base import LLMLike

__all__ = ["_invoke_judge"]


async def _invoke_judge(
    llm_like: LLMLike,
    *,
    prompt: str,
    agent_id: str,
) -> tuple[float, str, str | None]:
    """Invoke the LLM judge; return (score in [0,1], reasoning, error_or_None)."""
    try:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt, "kind": "request"}]
        response = await llm_like.ainvoke(messages, agent_id=agent_id)
        text: str = response.text or ""

        parsed = _parse_judge_json(text)
        if parsed is None:
            return (0.0, "", f"Could not parse JSON from judge response: {text!r}")

        raw_score, reasoning = parsed
        score = _normalise_score(raw_score)
        return (score, reasoning, None)

    except Exception as exc:  # intentional broad catch — never raise from judge
        return (0.0, "", str(exc))


def _parse_judge_json(text: str) -> tuple[float, str] | None:
    """Extract (score, reasoning) from text; tries direct JSON then regex fallback."""
    obj = _try_json_loads(text.strip())
    if obj is not None:
        return _extract_fields(obj)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        obj = _try_json_loads(match.group(0))
        if obj is not None:
            return _extract_fields(obj)

    return None


def _try_json_loads(s: str) -> dict[str, Any] | None:
    """Return parsed JSON dict or ``None`` on failure."""
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _extract_fields(obj: dict[str, Any]) -> tuple[float, str] | None:
    """Extract (score, reasoning) from a parsed dict; None if score missing/invalid."""
    try:
        raw_score = float(obj["score"])
    except (KeyError, TypeError, ValueError):
        return None

    reasoning: str = str(obj.get("reasoning", ""))
    return (raw_score, reasoning)


def _normalise_score(raw: float) -> float:
    """Normalise to [0.0, 1.0]; divides by 10 if raw > 1.0 (LLM 0-10 scale)."""
    if raw > 1.0:
        raw = raw / 10.0
    return max(0.0, min(1.0, raw))
