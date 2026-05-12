"""Creative task loader and judge evaluator for ATM tasks.

Public API:
- ``CreativeLoader``         — loads creative writing prompts from a YAML file
- ``CreativeJudgeEvaluator`` — evaluates creative writing answers via an LLM judge

Both classes are auto-registered with the module-level singletons:
- ``TASKS``      via ``@TASKS.register``
- ``EVALUATORS`` via ``@EVALUATORS.register``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from atm.core.types import TaskSpec
from atm.tasks._judge import _invoke_judge
from atm.tasks.base import EVALUATORS, TASKS, EvalResult, LLMLike

__all__ = ["CreativeJudgeEvaluator", "CreativeLoader"]

# ---------------------------------------------------------------------------
# Default YAML path (relative to the project root)
# ---------------------------------------------------------------------------

_DEFAULT_YAML: Path = Path("conf") / "tasks" / "creative_prompts.yaml"

# ---------------------------------------------------------------------------
# CreativeLoader
# ---------------------------------------------------------------------------


@TASKS.register
class CreativeLoader:
    """Loads creative writing prompts from a YAML configuration file.

    The YAML file must have a top-level ``prompts`` list where each entry has:
    - ``id``     (str)  — unique identifier, e.g. "c1"
    - ``prompt`` (str)  — the creative writing prompt shown to the agent
    - ``rubric`` (str)  — evaluation criteria used by the judge

    Example YAML::

        version: 1
        prompts:
          - id: c1
            prompt: "Write a short story about ..."
            rubric: "Evaluate for ..."

    The loader enforces a minimum of 5 prompts and raises ``ValueError`` if
    the file contains fewer entries.
    """

    name: str = "creative"

    def __init__(self, yaml_path: Path | None = None) -> None:
        self._yaml_path: Path = yaml_path if yaml_path is not None else _DEFAULT_YAML

    def load(
        self,
        cache_dir: Path | None = None,  # required by TaskLoader Protocol; not used here
    ) -> list[TaskSpec]:
        """Parse the YAML file and return a list of :class:`~atm.core.types.TaskSpec`.

        Args:
            cache_dir: Unused by this loader (no network download or caching
                       needed for local YAML prompts). Present to satisfy the
                       ``TaskLoader`` Protocol.

        Returns:
            A list of ``TaskSpec`` instances, one per prompt entry.

        Raises:
            ValueError: If the YAML contains fewer than 5 prompts.
            FileNotFoundError: If the YAML file does not exist.
        """
        raw: dict[str, Any]
        with self._yaml_path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        prompts: list[dict[str, Any]] = raw.get("prompts", [])

        if len(prompts) < 5:
            raise ValueError("creative: need at least 5 prompts")

        specs: list[TaskSpec] = []
        for entry in prompts:
            spec = TaskSpec(
                id=str(entry["id"]),
                type="creative",
                input=str(entry["prompt"]),
                expected=None,
                evaluator_key="creative_judge",
                metadata={"rubric": str(entry["rubric"])},
            )
            specs.append(spec)

        return specs


# ---------------------------------------------------------------------------
# CreativeJudgeEvaluator
# ---------------------------------------------------------------------------


@EVALUATORS.register
class CreativeJudgeEvaluator:
    """Evaluates creative writing answers using an LLM judge.

    The evaluator constructs a rubric-guided prompt, forwards it to the judge
    LLM via :func:`~atm.tasks._judge._invoke_judge`, and normalises the score.

    Scoring:
        - LLM returns a JSON ``{"score": 0..10, "reasoning": "..."}`` object.
        - ``_invoke_judge`` normalises the raw score to ``[0.0, 1.0]``.
        - ``passed = score >= 0.6``

    Args:
        judge_llm: Any object satisfying the :class:`~atm.tasks.base.LLMLike`
                   Protocol (e.g. ``FakeLLM`` in tests, ``LLMWrapper`` in prod).
    """

    name: str = "creative_judge"

    def __init__(self, judge_llm: LLMLike) -> None:
        self._judge_llm: LLMLike = judge_llm

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against the creative writing ``spec``.

        Args:
            spec:      The :class:`~atm.core.types.TaskSpec` describing the task.
                       ``spec.metadata['rubric']`` provides evaluation criteria.
            answer:    The agent's creative writing response.
            artifacts: Unused. Present for ``Evaluator`` Protocol compatibility.

        Returns:
            An :class:`~atm.tasks.base.EvalResult` with:

            - ``score``   — normalised float in ``[0.0, 1.0]``.
            - ``passed``  — ``True`` if ``score >= 0.6``.
            - ``details`` — ``{"reasoning": <str>, "raw": <first 200 chars of answer>}``.
            - ``error``   — ``None`` on success; error message string on failure.
        """
        rubric: str = spec.metadata.get("rubric", "")
        prompt: str = (
            "You are evaluating creative writing.\n"
            f"Rubric: {rubric}\n"
            f"Prompt: {spec.input}\n"
            f"Answer: {answer}\n\n"
            'Return JSON: {"score": 0..10, "reasoning": "..."}'
        )

        score, reasoning, error = await _invoke_judge(
            self._judge_llm,
            prompt=prompt,
            agent_id="creative_judge",
        )

        passed: bool = score >= 0.6

        return EvalResult(
            score=score,
            passed=passed,
            details={
                "reasoning": reasoning,
                "raw": answer[:200],
            },
            error=error,
        )
