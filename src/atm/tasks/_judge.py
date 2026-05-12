"""Shared LLM-judge helper for task evaluators.

Provides ``_invoke_judge`` — a single implementation of the pattern:
    1. Build a message dict list from ``prompt``.
    2. Call ``llm_like.ainvoke(messages, agent_id=agent_id)``.
    3. Parse the LLM response as JSON ``{"score": <number>, "reasoning": "<str>"}``.
    4. Normalise score from [0..10] to [0..1] if needed.
    5. Return ``(score, reasoning, error_or_None)``.

Both ``CreativeJudgeEvaluator`` and ``AnalysisHybridEvaluator`` import this
helper instead of duplicating the parsing logic locally.

JSON parse robustness:
    - First tries ``json.loads`` on the full response text.
    - If that fails, uses ``re.search(r'\\{.*\\}', text, re.DOTALL)`` to extract
      the first JSON-like object from noisy LLM output (e.g. markdown wrappers).
    - If both fail, returns ``(0.0, "", "<error message>")``.

On any exception (network error, missing field, type error, etc.) the function
never raises — it always returns a ``(score, reasoning, error)`` triple where
``error is not None`` signals failure.
"""

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
    """Invoke the LLM judge and parse its JSON response.

    Args:
        llm_like:  Any object conforming to the ``LLMLike`` Protocol.
        prompt:    The full evaluation prompt to send to the judge.
        agent_id:  Identifier forwarded to ``ainvoke`` for tracing / fixture routing.

    Returns:
        A 3-tuple ``(score, reasoning, error)``:

        - ``score``     — normalised float in ``[0.0, 1.0]``.
        - ``reasoning`` — string explanation from the LLM (may be empty on error).
        - ``error``     — ``None`` on success; error message string on failure.
    """
    try:
        # Use dict-format messages — FakeLLM and LLMWrapper both accept list[dict[str,Any]]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt, "kind": "request"}
        ]
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_judge_json(text: str) -> tuple[float, str] | None:
    """Try to extract ``(score, reasoning)`` from ``text``.

    Attempts, in order:
    1. ``json.loads(text.strip())`` — clean JSON response.
    2. Regex ``re.search(r'\\{.*\\}', text, re.DOTALL)`` — JSON embedded in prose.

    Returns ``None`` if neither attempt succeeds or if required fields are missing.
    """
    # Attempt 1: direct JSON parse
    obj = _try_json_loads(text.strip())
    if obj is not None:
        return _extract_fields(obj)

    # Attempt 2: regex fallback — first {…} block in the text
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
    """Extract ``(score, reasoning)`` from a parsed JSON dict.

    Returns ``None`` if ``score`` field is missing or cannot be converted to float.
    ``reasoning`` defaults to ``""`` if not present.
    """
    try:
        raw_score = float(obj["score"])
    except (KeyError, TypeError, ValueError):
        return None

    reasoning: str = str(obj.get("reasoning", ""))
    return (raw_score, reasoning)


def _normalise_score(raw: float) -> float:
    """Normalise ``raw`` to ``[0.0, 1.0]``.

    - Values already in ``[0.0, 1.0]`` are returned as-is.
    - Values in ``(1.0, 10.0]`` are divided by 10 (0..10 scale from LLMs).
    - Values outside ``[0.0, 10.0]`` are clamped to ``[0.0, 1.0]`` after any
      normalisation (safety net for malformed scores).
    """
    if raw > 1.0:
        raw = raw / 10.0
    # Clamp to [0.0, 1.0] as a safety net
    return max(0.0, min(1.0, raw))
