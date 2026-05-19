"""Ground-truth facade — dispatches to the appropriate M10 evaluator via ``_DEPS`` table."""

from __future__ import annotations

from typing import Any

from atm.tasks import EVALUATORS, EvalResult, TaskSpec

_DEPS: dict[str, tuple[str, ...]] = {
    "humaneval_pytest": ("sandbox",),
    "gsm8k_numeric": (),
    "commongen_rouge_coverage": (),
    "dabench_numeric_exact": (),
}


async def score_ground_truth(
    spec: TaskSpec,
    answer: str,
    *,
    sandbox: Any | None = None,
    judge_llm: Any | None = None,
) -> EvalResult:
    """Dispatch to the M10 evaluator registered under ``spec.evaluator_key``.

    Raises:
        KeyError:   unknown ``evaluator_key``.
        ValueError: required dep (e.g. ``sandbox``) is None.
    """
    key = spec.evaluator_key

    if key not in _DEPS:
        try:
            available = ", ".join(EVALUATORS.names())
        except Exception:
            available = "<unavailable>"
        raise KeyError(f"score_ground_truth: unknown evaluator_key={key!r}. Known: [{available}].")

    deps_required = _DEPS[key]
    kwargs: dict[str, Any] = {}
    available_kwargs = {"sandbox": sandbox, "judge_llm": judge_llm}

    for dep in deps_required:
        val = available_kwargs.get(dep)
        if val is None:
            raise ValueError(f"score_ground_truth: {dep!r} is required for evaluator_key={key!r}")
        kwargs[dep] = val

    evaluator_cls = EVALUATORS.get(key)
    evaluator = evaluator_cls(**kwargs)
    result: EvalResult = await evaluator.evaluate(spec, answer)
    return result
