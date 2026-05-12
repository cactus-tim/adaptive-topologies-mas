"""Ground-truth facade over M10 EVALUATORS.

Provides a single async entry-point ``score_ground_truth`` that dispatches to
the appropriate M10 evaluator based on ``spec.evaluator_key``.

Dispatch strategy: ``_DEPS`` table (NOT introspection).
The ``_DEPS`` dict maps each evaluator_key to the tuple of kwarg names required
to construct that evaluator.  When adding a new evaluator, add a row to ``_DEPS``
and ensure the kwarg is accepted by ``score_ground_truth``.

Supported keys (update ``_DEPS`` when registering new evaluators):
    humaneval_pytest          — HumanEvalEvaluator (requires sandbox)
    gsm8k_numeric             — GSM8KMatcher       (no extra deps)
    commongen_rouge_coverage  — CommonGenEvaluator  (no extra deps)
    dabench_numeric_exact     — DABenchEvaluator    (no extra deps)

Raises:
    ValueError: if a required dependency (sandbox or judge_llm) is None.
    KeyError:   if spec.evaluator_key is not in ``_DEPS``.
"""

from __future__ import annotations

from typing import Any

from atm.tasks import EVALUATORS, EvalResult, TaskSpec

# ---------------------------------------------------------------------------
# Dependency table
# ---------------------------------------------------------------------------

# Map evaluator_key -> tuple of required kwarg names.
# Update this when registering new evaluators.
_DEPS: dict[str, tuple[str, ...]] = {
    "humaneval_pytest": ("sandbox",),
    "gsm8k_numeric": (),
    "commongen_rouge_coverage": (),
    "dabench_numeric_exact": (),
}


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


async def score_ground_truth(
    spec: TaskSpec,
    answer: str,
    *,
    sandbox: Any | None = None,
    judge_llm: Any | None = None,
) -> EvalResult:
    """Dispatch to the M10 evaluator registered under spec.evaluator_key.

    Args:
        spec:      The task specification; ``spec.evaluator_key`` selects the evaluator.
        answer:    The agent's answer string.
        sandbox:   A ``SubprocessSandbox`` instance — required for ``humaneval_pytest``.
        judge_llm: An LLM-like object — reserved for future evaluators that require it.

    Returns:
        An ``EvalResult`` with score in [0.0, 1.0].

    Raises:
        KeyError:   if ``spec.evaluator_key`` is not a key in ``_DEPS``.
        ValueError: if a required dep (e.g. ``sandbox``) is ``None`` for this evaluator.
    """
    key = spec.evaluator_key

    if key not in _DEPS:
        try:
            available = ", ".join(EVALUATORS.names())
        except Exception:
            available = "<unavailable>"
        raise KeyError(
            f"score_ground_truth: unknown evaluator_key={key!r}. Known: [{available}]."
        )

    deps_required = _DEPS[key]
    kwargs: dict[str, Any] = {}
    available_kwargs = {"sandbox": sandbox, "judge_llm": judge_llm}

    for dep in deps_required:
        val = available_kwargs.get(dep)
        if val is None:
            raise ValueError(
                f"score_ground_truth: {dep!r} is required for evaluator_key={key!r}"
            )
        kwargs[dep] = val

    evaluator_cls = EVALUATORS.get(key)
    evaluator = evaluator_cls(**kwargs)
    return await evaluator.evaluate(spec, answer)
