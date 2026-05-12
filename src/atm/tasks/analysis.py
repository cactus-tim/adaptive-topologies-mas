"""Analysis task loader and hybrid evaluator for ATM benchmark.

Public API:
- ``AnalysisLoader``          — loads analysis prompts from a YAML file
- ``AnalysisHybridEvaluator`` — hybrid evaluator combining structural checks + LLM judge

Hybrid scoring formula:
    final_score = 0.5 * structural_score + 0.5 * judge_score
    passed = final_score >= 0.6

Structural check types supported:
- "contains"   — ``value in answer``
- "regex"      — ``re.search(value, answer) is not None``
- "min_length" — ``len(answer) >= int(value)``

If the structural_checks list is empty, structural_score = 1.0 (all checks pass by default).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from atm.core.types import TaskSpec
from atm.tasks._judge import _invoke_judge
from atm.tasks.base import EVALUATORS, TASKS, EvalResult, LLMLike

__all__ = [
    "AnalysisHybridEvaluator",
    "AnalysisLoader",
]

_DEFAULT_YAML_PATH = Path("conf") / "tasks" / "analysis_prompts.yaml"
_MIN_PROMPTS = 5


# ---------------------------------------------------------------------------
# AnalysisLoader
# ---------------------------------------------------------------------------


@TASKS.register
class AnalysisLoader:
    """Loads analysis benchmark tasks from a YAML configuration file.

    The YAML file must have a top-level ``prompts`` key containing a list of
    entries with the following fields:
        - id               (str) — unique task identifier
        - prompt           (str) — question/instruction for the agent
        - data_csv         (str) — inline CSV data string
        - expected_pattern (str) — expected pattern or regex for reference
        - rubric           (str) — evaluation rubric passed to the judge
        - structural_checks (list[dict]) — list of structural check dicts

    Raises:
        ValueError: if the loaded YAML contains fewer than 5 prompts.
        FileNotFoundError: if the YAML file does not exist.
    """

    name: str = "analysis"

    def __init__(self, yaml_path: Path | None = None) -> None:
        self._yaml_path: Path = yaml_path if yaml_path is not None else _DEFAULT_YAML_PATH

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Parse the YAML file and return a list of TaskSpec instances.

        Args:
            cache_dir: Unused — analysis tasks are read directly from YAML.
                       Accepted for Protocol compatibility with TaskLoader.

        Returns:
            List of TaskSpec with type="analysis" and evaluator_key="analysis_hybrid".

        Raises:
            ValueError: if fewer than 5 prompts are present in the YAML.
            FileNotFoundError: if the YAML path does not exist.
        """
        with self._yaml_path.open(encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)

        raw_prompts: list[dict[str, Any]] = data.get("prompts", [])

        if len(raw_prompts) < _MIN_PROMPTS:
            raise ValueError(
                f"analysis: need at least {_MIN_PROMPTS} prompts, "
                f"got {len(raw_prompts)} in {self._yaml_path}"
            )

        specs: list[TaskSpec] = []
        for entry in raw_prompts:
            prompt: str = str(entry["prompt"])
            data_csv: str = str(entry.get("data_csv", ""))
            combined_input = f"{prompt}\n\nData:\n{data_csv}"

            structural_checks: list[dict[str, str]] = [
                dict(c) for c in entry.get("structural_checks", [])
            ]

            spec = TaskSpec(
                id=str(entry["id"]),
                type="analysis",
                input=combined_input,
                expected=str(entry.get("expected_pattern", "")),
                evaluator_key="analysis_hybrid",
                metadata={
                    "rubric": str(entry.get("rubric", "")),
                    "structural_checks": structural_checks,
                },
            )
            specs.append(spec)

        return specs


# ---------------------------------------------------------------------------
# AnalysisHybridEvaluator
# ---------------------------------------------------------------------------


@EVALUATORS.register
class AnalysisHybridEvaluator:
    """Hybrid evaluator that combines structural rule checks with an LLM judge.

    Scoring:
        structural_score = mean of per-check pass/fail (1.0 / 0.0)
        judge_score      = normalised LLM score in [0.0, 1.0]
        final_score      = 0.5 * structural_score + 0.5 * judge_score
        passed           = final_score >= 0.6

    Args:
        judge_llm: Any object conforming to the LLMLike Protocol.
    """

    name: str = "analysis_hybrid"

    def __init__(self, judge_llm: LLMLike) -> None:
        self._judge_llm = judge_llm

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def _run_structural_checks(
        self,
        checks: list[dict[str, Any]],
        answer: str,
    ) -> float:
        """Evaluate structural checks against the answer and return mean pass rate.

        Args:
            checks: List of check dicts, each with keys ``type`` and ``value``.
            answer: The agent's answer string to check.

        Returns:
            Float in [0.0, 1.0]: mean of per-check boolean results.
            Returns 1.0 if ``checks`` is empty (vacuous truth).

        Supported check types:
            - "contains"   — value substring is present in answer
            - "regex"      — re.search(value, answer) matches
            - "min_length" — len(answer) >= int(value)
        """
        if not checks:
            return 1.0

        results: list[float] = []
        for check in checks:
            check_type: str = str(check.get("type", ""))
            value: str = str(check.get("value", ""))

            if check_type == "contains":
                passed = value in answer
            elif check_type == "regex":
                passed = re.search(value, answer) is not None
            elif check_type == "min_length":
                passed = len(answer) >= int(value)
            else:
                # Unknown check type — treat as failed
                passed = False

            results.append(1.0 if passed else 0.0)

        return sum(results) / len(results)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
    ) -> EvalResult:
        """Evaluate ``answer`` against ``spec`` using structural + judge hybrid.

        Args:
            spec:   TaskSpec with metadata keys ``rubric`` and ``structural_checks``.
            answer: The agent's answer to evaluate.

        Returns:
            EvalResult with:
                score   = 0.5 * structural_score + 0.5 * judge_score
                passed  = score >= 0.6
                details = {"structural_score": ..., "judge_score": ..., "reasoning": ...}
                error   = None on success; error string if judge invocation failed
        """
        structural_checks: list[dict[str, Any]] = spec.metadata.get("structural_checks", [])
        rubric: str = str(spec.metadata.get("rubric", ""))

        struct_score = self._run_structural_checks(structural_checks, answer)

        judge_prompt = (
            f"Evaluate analysis quality.\n"
            f"Rubric: {rubric}\n"
            f"Question: {spec.input}\n"
            f"Answer: {answer}\n\n"
            f'Return JSON {{"score": 0..10, "reasoning": "..."}}'
        )

        judge_score, reasoning, error = await _invoke_judge(
            self._judge_llm,
            prompt=judge_prompt,
            agent_id="analysis_judge",
        )

        final_score = 0.5 * struct_score + 0.5 * judge_score
        passed = final_score >= 0.6

        return EvalResult(
            score=final_score,
            passed=passed,
            details={
                "structural_score": struct_score,
                "judge_score": judge_score,
                "reasoning": reasoning,
            },
            error=error,
        )
