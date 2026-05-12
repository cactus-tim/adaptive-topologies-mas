"""HumanEval task loader and evaluator for the ATM tasks module.

Public API:
- ``_strip_code_fences(text: str) -> str``  — remove markdown code fences
- ``HumanEvalLoader``                        — TaskLoader for openai/openai_humaneval
- ``HumanEvalEvaluator``                     — Evaluator using subprocess code execution
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import datasets  # type: ignore[import-untyped]

from atm.core.types import TaskSpec
from atm.tasks._cache import is_cached, read_cache, write_cache
from atm.tasks.base import EVALUATORS, TASKS, EvalResult
from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox

__all__ = [
    "HumanEvalEvaluator",
    "HumanEvalLoader",
    "_strip_code_fences",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"^```(?:python)?\n(.*?)\n```$",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from ``text``.

    Handles:
    - ````python\\n...\\n``` `` — language-tagged fences
    - ````\\n...\\n``` `` — plain fences (no language tag)
    - Plain text — returned unchanged.
    - Unclosed fences (opening only) — returned unchanged.

    Args:
        text: Raw text that may contain a markdown code fence.

    Returns:
        The inner code content if a complete fence is found, otherwise ``text``.
    """
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1)
    return text


# ---------------------------------------------------------------------------
# HumanEvalLoader
# ---------------------------------------------------------------------------

_CACHE_KEY = "humaneval"
_HF_DATASET = "openai/openai_humaneval"


@TASKS.register
class HumanEvalLoader:
    """Task loader for the OpenAI HumanEval benchmark.

    Uses Parquet cache (``data/cache/tasks/humaneval.parquet``) to avoid
    repeated network calls.  Falls back to ``datasets.load_dataset`` on first
    call.
    """

    name: str = "humaneval"

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Load HumanEval tasks, hitting cache when available.

        Args:
            cache_dir: Override for the default Parquet cache directory.

        Returns:
            List of ``TaskSpec`` instances, one per HumanEval problem.
        """
        if is_cached(_CACHE_KEY, cache_dir):
            rows: list[dict[str, Any]] = read_cache(_CACHE_KEY, cache_dir)
        else:
            hf_dataset = datasets.load_dataset(_HF_DATASET)
            split = hf_dataset["test"]
            rows = [
                {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "canonical_solution": row["canonical_solution"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                }
                for row in split
            ]
            write_cache(_CACHE_KEY, rows, cache_dir=cache_dir, dataset_revision=None)

        return [_row_to_spec(row) for row in rows]


def _row_to_spec(row: dict[str, Any]) -> TaskSpec:
    """Convert a raw HumanEval row dict into a ``TaskSpec``."""
    return TaskSpec(
        id=row["task_id"],
        type="programming",
        input=row["prompt"],
        expected=row["canonical_solution"],
        evaluator_key="humaneval_pytest",
        metadata={
            "test": row["test"],
            "entry_point": row["entry_point"],
        },
    )


# ---------------------------------------------------------------------------
# HumanEvalEvaluator
# ---------------------------------------------------------------------------


@EVALUATORS.register
class HumanEvalEvaluator:
    """Evaluator for HumanEval tasks using subprocess code execution.

    Builds a Python script from the task prompt, the model's answer, the
    test harness, and the entry-point call.  Executes it in a
    ``SubprocessSandbox`` and determines pass/fail by exit code.
    """

    name: str = "humaneval_pytest"

    def __init__(self, sandbox: SubprocessSandbox | None = None) -> None:
        self._sandbox: SubprocessSandbox = sandbox if sandbox is not None else SubprocessSandbox()

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against the HumanEval ``spec``.

        Steps:
        1. Strip markdown code fences from ``answer``.
        2. Build executable payload: prompt + answer + test harness + check call.
        3. Execute in sandbox with a 10-second timeout.
        4. passed = exit_code == 0; score = 1.0 if passed else 0.0.

        Args:
            spec:      The ``TaskSpec`` from ``HumanEvalLoader``.
            answer:    Model-generated code (may be fenced or plain).
            artifacts: Unused. Present for ``Evaluator`` Protocol compatibility.

        Returns:
            An ``EvalResult`` with score, passed, details, and optional error.
        """
        clean_answer = _strip_code_fences(answer)

        test_code = spec.metadata["test"]
        entry_point = spec.metadata["entry_point"]

        # HumanEval prompt already contains the function signature.
        # Concatenate: prompt (signature+docstring) + answer (body) + test + check call.
        payload = f"{spec.input}{clean_answer}\n\n{test_code}\n\ncheck({entry_point})"

        exec_result = await self._sandbox.execute(
            "python",
            payload,
            files={},
            timeout=10,
        )

        passed = exec_result.exit_code == 0
        score = 1.0 if passed else 0.0
        details: dict[str, Any] = {
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "exit_code": exec_result.exit_code,
        }
        error: str | None = exec_result.stderr if not passed else None

        return EvalResult(
            score=score,
            passed=passed,
            details=details,
            error=error,
        )
