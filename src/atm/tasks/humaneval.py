"""HumanEval task loader and subprocess-execution evaluator."""

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

_CODE_FENCE_RE = re.compile(
    r"^```(?:python)?\n(.*?)\n```$",
    re.DOTALL,
)

_EMBEDDED_FENCE_RE = re.compile(
    r"```(?:python|py)?[ \t]*\n(.*?)\n```",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences; returns last embedded block or plain text unchanged."""
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1)
    embedded = list(_EMBEDDED_FENCE_RE.finditer(stripped))
    if embedded:
        return embedded[-1].group(1)
    return text


_CACHE_KEY = "humaneval"
_HF_DATASET = "openai/openai_humaneval"


@TASKS.register
class HumanEvalLoader:
    """Task loader for the OpenAI HumanEval benchmark (openai/openai_humaneval)."""

    name: str = "humaneval"

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Load HumanEval tasks; uses Parquet cache when available."""
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


@EVALUATORS.register
class HumanEvalEvaluator:
    """Evaluator for HumanEval: executes generated code in SubprocessSandbox."""

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
        """Evaluate ``answer`` against the HumanEval ``spec``."""
        clean_answer = _strip_code_fences(answer)

        test_code = spec.metadata["test"]
        entry_point = spec.metadata["entry_point"]

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
