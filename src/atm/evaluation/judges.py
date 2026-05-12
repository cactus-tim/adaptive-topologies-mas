"""LLM-as-judge protocols for post-hoc answer evaluation.

Three judge classes:

- ``RubricJudge``         — single rubric call; wraps ``_invoke_judge`` from M10.
- ``SelfConsistentJudge`` — N calls with shuffled rubric per call; returns mean score.
- ``PairwiseJudge``       — AB + BA swap test; declares winner only when both orderings agree.

Score scale note: ``_invoke_judge`` normalises LLM output (0..10) → [0.0, 1.0].
Arch §13.2 specifies 0..5 but M10 convention (0..10) is used throughout; see
context.md Decision note.

Position-bias mitigation follows arXiv 2406.07791: swap test (AB + BA), winner
declared only when both orderings agree; explicit ``_to_answer_relative`` helper
eliminates silent slot→answer translation bugs.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from atm.tasks._judge import _invoke_judge
from atm.tasks.base import LLMLike

__all__ = [
    "PairwiseJudge",
    "PairwiseResult",
    "RubricJudge",
    "SelfConsistentJudge",
]

# ---------------------------------------------------------------------------
# PairwiseResult
# ---------------------------------------------------------------------------


class PairwiseResult(BaseModel):
    """Outcome of a pairwise comparison with swap-consistency check."""

    model_config = ConfigDict(frozen=True)

    winner: Literal["a", "b", "tie"]
    swap_consistent: bool
    reason_ab: str
    reason_ba: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_answer_relative(
    slot_winner: Literal["A", "B", "tie"],
    *,
    swapped: bool,
) -> Literal["a", "b", "tie"]:
    """Translate a slot-relative winner to an answer-relative winner.

    In the AB call (not swapped): slot A = answer_a, slot B = answer_b.
    In the BA call (swapped):     slot A = answer_b, slot B = answer_a.

    Args:
        slot_winner: Which slot the judge preferred ("A", "B", or "tie").
        swapped:     True when the call was made in BA order.

    Returns:
        Answer-relative winner ("a", "b", or "tie").
    """
    if slot_winner == "tie":
        return "tie"
    if not swapped:
        return "a" if slot_winner == "A" else "b"
    # swapped: slot A holds answer_b, slot B holds answer_a
    return "b" if slot_winner == "A" else "a"


def _parse_pairwise_response(text: str) -> tuple[Literal["A", "B", "tie"], str]:
    """Extract winner and reasoning from a pairwise LLM response.

    Expects JSON ``{"winner": "A"/"B"/"tie", "reasoning": "..."}``.
    Falls back to regex on noisy output. Returns ``("tie", "")`` on parse failure.
    """
    obj = _try_json(text.strip())
    if obj is None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            obj = _try_json(match.group(0))

    if obj is not None:
        raw_winner = str(obj.get("winner", "")).upper().strip()
        reasoning = str(obj.get("reasoning", ""))
        if raw_winner in ("A", "B", "TIE"):
            winner: Literal["A", "B", "tie"] = (
                "tie" if raw_winner == "TIE" else raw_winner  # type: ignore[assignment]
            )
            return winner, reasoning

    return "tie", ""


def _try_json(s: str) -> dict[str, Any] | None:
    """Return parsed JSON dict or None."""
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _shuffle_rubric(rubric: str, seed: int) -> str:
    """Shuffle rubric bullet-lines with a fixed seed for self-consistency calls.

    If the rubric has no recognisable lines (single paragraph), returns as-is.
    """
    lines = rubric.splitlines()
    if len(lines) <= 1:
        return rubric
    rng = random.Random(seed)
    shuffled = lines[:]
    rng.shuffle(shuffled)
    return "\n".join(shuffled)


# ---------------------------------------------------------------------------
# RubricJudge
# ---------------------------------------------------------------------------


class RubricJudge:
    """Single-call LLM rubric judge.

    Uses ``_invoke_judge`` (M10) for prompt construction, invocation, JSON
    parsing, and normalisation. The score is in [0.0, 1.0].

    agent_id for tracing: ``rubric_judge``.
    """

    async def score(
        self,
        spec_input: str,
        rubric: str,
        answer: str,
        *,
        judge_llm: LLMLike,
    ) -> tuple[float, str, str | None]:
        """Evaluate ``answer`` against ``rubric`` for task ``spec_input``.

        Returns:
            ``(score, reasoning, error)`` — score in [0.0, 1.0]; error is None on success.
        """
        prompt = _build_rubric_prompt(spec_input, rubric, answer)
        return await _invoke_judge(judge_llm, prompt=prompt, agent_id="rubric_judge")


# ---------------------------------------------------------------------------
# SelfConsistentJudge
# ---------------------------------------------------------------------------


class SelfConsistentJudge:
    """Self-consistency rubric judge — N calls with per-call shuffled rubric.

    Each call uses a deterministic seed derived from ``(run_seed, i)`` so that
    results are reproducible.  Mean of scalar scores is returned.

    agent_id for tracing: ``self_consistency_judge``.
    """

    async def score(
        self,
        spec_input: str,
        rubric: str,
        answer: str,
        *,
        judge_llm: LLMLike,
        run_seed: int,
        n: int = 3,
    ) -> float:
        """Return mean rubric score across ``n`` independent judge calls.

        Args:
            spec_input: The task prompt / question shown to the agent.
            rubric:     Evaluation criteria (may contain bullet lines).
            answer:     The agent's answer to evaluate.
            judge_llm:  LLM-like object used for judge calls.
            run_seed:   Seed for this run; per-call seed = hash((run_seed, i)) & 0xFFFFFFFF.
            n:          Number of independent judge calls (default 3).

        Returns:
            Mean normalised score in [0.0, 1.0].  Returns 0.0 if all calls fail.
        """
        scores: list[float] = []
        for i in range(n):
            call_seed = hash((run_seed, i)) & 0xFFFFFFFF
            shuffled = _shuffle_rubric(rubric, call_seed)
            prompt = _build_rubric_prompt(spec_input, shuffled, answer)
            score, _reasoning, error = await _invoke_judge(
                judge_llm, prompt=prompt, agent_id="self_consistency_judge"
            )
            if error is None:
                scores.append(score)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# PairwiseJudge
# ---------------------------------------------------------------------------


class PairwiseJudge:
    """Pairwise answer comparison with position-bias swap test.

    Makes two calls:
    - AB call (agent_id ``pairwise_judge_ab``): prompt presents answer_a first.
    - BA call (agent_id ``pairwise_judge_ba``): prompt presents answer_b first (swapped).

    ``_to_answer_relative`` translates slot-relative results back to answer-relative.
    ``winner`` is declared only when both orderings agree (``swap_consistent=True``).
    If they disagree, ``winner="tie"`` and ``swap_consistent=False``.
    """

    async def compare(
        self,
        spec_input: str,
        rubric: str,
        answer_a: str,
        answer_b: str,
        *,
        judge_llm: LLMLike,
    ) -> PairwiseResult:
        """Compare ``answer_a`` vs ``answer_b`` with swap-consistency check.

        Args:
            spec_input: The task prompt / question.
            rubric:     Evaluation criteria.
            answer_a:   First candidate answer.
            answer_b:   Second candidate answer.
            judge_llm:  LLM-like object used for judge calls.

        Returns:
            :class:`PairwiseResult` with winner, swap_consistent, and per-ordering reasons.
        """
        # AB ordering: slot A = answer_a, slot B = answer_b
        prompt_ab = _build_pairwise_prompt(spec_input, rubric, answer_a, answer_b)
        response_ab = await judge_llm.ainvoke(
            [{"role": "user", "content": prompt_ab, "kind": "request"}],
            agent_id="pairwise_judge_ab",
        )
        slot_ab, reason_ab = _parse_pairwise_response(response_ab.text or "")
        winner_ab = _to_answer_relative(slot_ab, swapped=False)

        # BA ordering: slot A = answer_b, slot B = answer_a (swapped)
        prompt_ba = _build_pairwise_prompt(spec_input, rubric, answer_b, answer_a)
        response_ba = await judge_llm.ainvoke(
            [{"role": "user", "content": prompt_ba, "kind": "request"}],
            agent_id="pairwise_judge_ba",
        )
        slot_ba, reason_ba = _parse_pairwise_response(response_ba.text or "")
        winner_ba = _to_answer_relative(slot_ba, swapped=True)

        swap_consistent = winner_ab == winner_ba and winner_ab != "tie"
        winner: Literal["a", "b", "tie"] = winner_ab if swap_consistent else "tie"

        return PairwiseResult(
            winner=winner,
            swap_consistent=swap_consistent,
            reason_ab=reason_ab,
            reason_ba=reason_ba,
        )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_rubric_prompt(spec_input: str, rubric: str, answer: str) -> str:
    """Build a rubric evaluation prompt for ``_invoke_judge``."""
    return (
        f"You are an expert evaluator. Score the following answer on a scale of 0 to 10.\n\n"
        f"Task:\n{spec_input}\n\n"
        f"Evaluation rubric:\n{rubric}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f'Respond ONLY with JSON: {{"score": <integer 0-10>, "reasoning": "<brief explanation>"}}'
    )


def _build_pairwise_prompt(
    spec_input: str, rubric: str, slot_a: str, slot_b: str
) -> str:
    """Build a pairwise comparison prompt."""
    return (
        f"You are an expert evaluator. Compare two answers and pick the better one.\n\n"
        f"Task:\n{spec_input}\n\n"
        f"Evaluation rubric:\n{rubric}\n\n"
        f"Answer A:\n{slot_a}\n\n"
        f"Answer B:\n{slot_b}\n\n"
        f'Respond ONLY with JSON: {{"winner": "A" or "B" or "tie", "reasoning": "<brief explanation>"}}'
    )
