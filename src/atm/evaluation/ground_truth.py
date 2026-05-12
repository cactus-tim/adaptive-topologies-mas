"""Ground-truth façade over M10 EVALUATORS.

Provides a single async entry-point ``score_ground_truth`` that dispatches to
the appropriate M10 evaluator based on ``spec.evaluator_key``.

Supported keys (flat if/elif dispatch — fully type-checkable):
    humaneval_pytest  — HumanEvalEvaluator  (requires sandbox)
    mmlu_exact_match  — MMLUEvaluator       (no extra deps)
    creative_judge    — CreativeJudgeEvaluator (requires judge_llm)
    analysis_hybrid   — AnalysisHybridEvaluator (requires judge_llm)

Raises:
    ValueError: if a required dependency (sandbox or judge_llm) is None.
    KeyError:   if spec.evaluator_key is not one of the four known keys.
"""

from __future__ import annotations

from typing import Any

from atm.tasks.base import EvalResult, LLMLike, TaskSpec


async def score_ground_truth(
    spec: TaskSpec,
    answer: str,
    *,
    sandbox: Any | None = None,
    judge_llm: LLMLike | None = None,
) -> EvalResult:
    """Evaluate ``answer`` against ``spec`` using the appropriate M10 evaluator.

    Args:
        spec:      The task specification; ``spec.evaluator_key`` selects the evaluator.
        answer:    The agent's answer string.
        sandbox:   A ``SubprocessSandbox`` instance — required for ``humaneval_pytest``.
        judge_llm: An ``LLMLike`` instance — required for ``creative_judge`` and
                   ``analysis_hybrid``.

    Returns:
        An ``EvalResult`` with score ∈ [0.0, 1.0].

    Raises:
        ValueError: if ``sandbox`` is None for ``humaneval_pytest``, or if
                    ``judge_llm`` is None for ``creative_judge`` / ``analysis_hybrid``.
        KeyError:   if ``spec.evaluator_key`` is not a known evaluator key.
    """
    key = spec.evaluator_key

    if key == "humaneval_pytest":
        if sandbox is None:
            raise ValueError(
                "score_ground_truth: sandbox is required for evaluator_key='humaneval_pytest'"
            )
        from atm.tasks.humaneval import HumanEvalEvaluator

        return await HumanEvalEvaluator(sandbox=sandbox).evaluate(spec, answer)

    if key == "mmlu_exact_match":
        from atm.tasks.mmlu import MMLUEvaluator

        return await MMLUEvaluator().evaluate(spec, answer)

    if key == "creative_judge":
        if judge_llm is None:
            raise ValueError(
                "score_ground_truth: judge_llm is required for evaluator_key='creative_judge'"
            )
        from atm.tasks.creative import CreativeJudgeEvaluator

        return await CreativeJudgeEvaluator(judge_llm=judge_llm).evaluate(spec, answer)

    if key == "analysis_hybrid":
        if judge_llm is None:
            raise ValueError(
                "score_ground_truth: judge_llm is required for evaluator_key='analysis_hybrid'"
            )
        from atm.tasks.analysis import AnalysisHybridEvaluator

        return await AnalysisHybridEvaluator(judge_llm=judge_llm).evaluate(spec, answer)

    raise KeyError(
        f"score_ground_truth: unknown evaluator_key={key!r}. "
        f"Known keys: 'humaneval_pytest', 'mmlu_exact_match', "
        f"'creative_judge', 'analysis_hybrid'"
    )
