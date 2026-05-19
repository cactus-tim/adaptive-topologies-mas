"""LLM-as-judge protocols: RubricJudge, SelfConsistentJudge, PairwiseJudge."""

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


class PairwiseResult(BaseModel):
    """Outcome of a pairwise comparison with swap-consistency check."""

    model_config = ConfigDict(frozen=True)

    winner: Literal["a", "b", "tie"]
    swap_consistent: bool
    reason_ab: str
    reason_ba: str


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


class RubricJudge:
    """Single-call LLM rubric judge. Score in [0.0, 1.0]."""

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


class SelfConsistentJudge:
    """Self-consistency rubric judge — N calls with shuffled rubric, returns mean score."""

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
        """Return mean rubric score across ``n`` independent judge calls (0.0 if all fail)."""
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


class PairwiseJudge:
    """Pairwise comparison with AB + BA swap test; winner only when both orderings agree."""

    async def compare(
        self,
        spec_input: str,
        rubric: str,
        answer_a: str,
        answer_b: str,
        *,
        judge_llm: LLMLike,
    ) -> PairwiseResult:
        """Compare ``answer_a`` vs ``answer_b``. Returns PairwiseResult."""
        prompt_ab = _build_pairwise_prompt(spec_input, rubric, answer_a, answer_b)
        response_ab = await judge_llm.ainvoke(
            [{"role": "user", "content": prompt_ab, "kind": "request"}],
            agent_id="pairwise_judge_ab",
        )
        slot_ab, reason_ab = _parse_pairwise_response(response_ab.text or "")
        winner_ab = _to_answer_relative(slot_ab, swapped=False)

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


def _build_rubric_prompt(spec_input: str, rubric: str, answer: str) -> str:
    """Build a rubric evaluation prompt for ``_invoke_judge``."""
    return (
        f"You are an expert evaluator. Score the following answer on a scale of 0 to 10.\n\n"
        f"Task:\n{spec_input}\n\n"
        f"Evaluation rubric:\n{rubric}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f'Respond ONLY with JSON: {{"score": <integer 0-10>, "reasoning": "<brief explanation>"}}'
    )


def _build_pairwise_prompt(spec_input: str, rubric: str, slot_a: str, slot_b: str) -> str:
    """Build a pairwise comparison prompt."""
    return (
        f"You are an expert evaluator. Compare two answers and pick the better one.\n\n"
        f"Task:\n{spec_input}\n\n"
        f"Evaluation rubric:\n{rubric}\n\n"
        f"Answer A:\n{slot_a}\n\n"
        f"Answer B:\n{slot_b}\n\n"
        f'Respond ONLY with JSON: {{"winner": "A" or "B" or "tie", "reasoning": "<brief explanation>"}}'
    )
