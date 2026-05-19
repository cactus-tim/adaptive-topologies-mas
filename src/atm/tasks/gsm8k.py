"""GSM8K task loader and numeric-match evaluator."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import datasets  # type: ignore[import-untyped]

from atm.core.types import TaskSpec
from atm.tasks._cache import is_cached, read_cache, write_cache
from atm.tasks.base import EVALUATORS, TASKS, EvalResult

__all__ = [
    "GSM8KLoader",
    "GSM8KMatcher",
    "_extract_final_number",
]

_FINAL_NUMBER_RE = re.compile(r"####\s*(-?[0-9][\d,]*(?:\.\d+)?)")

_NUMERIC_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _extract_final_number(text: str) -> str | None:
    """Return the number after ``####`` (commas stripped), or None if not found."""
    match = _FINAL_NUMBER_RE.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", "")


_CACHE_KEY = "gsm8k"
_HF_DATASET = "openai/gsm8k"
_HF_CONFIG = "main"
_HF_SPLIT = "test"


@TASKS.register
class GSM8KLoader:
    """Task loader for the GSM8K math-reasoning benchmark (openai/gsm8k, test split)."""

    name: str = "gsm8k"

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Load GSM8K tasks; uses Parquet cache when available."""
        if is_cached(_CACHE_KEY, cache_dir):
            rows: list[dict[str, Any]] = read_cache(_CACHE_KEY, cache_dir)
        else:
            split = datasets.load_dataset(_HF_DATASET, _HF_CONFIG, split=_HF_SPLIT)
            rows = []
            for idx, row in enumerate(split):
                expected = _extract_final_number(row["answer"])
                if expected is None:
                    continue
                rows.append(
                    {
                        "idx": idx,
                        "question": row["question"],
                        "expected": expected,
                    }
                )
            write_cache(_CACHE_KEY, rows, cache_dir=cache_dir, dataset_revision=None)

        return [_row_to_spec(row) for row in rows]


def _row_to_spec(row: dict[str, Any]) -> TaskSpec:
    """Convert a cached GSM8K row dict into a ``TaskSpec``."""
    return TaskSpec(
        id=f"gsm8k/{row['idx']}",
        type="reasoning",
        input=row["question"],
        expected=row["expected"],
        evaluator_key="gsm8k_numeric",
    )


@EVALUATORS.register
class GSM8KMatcher:
    """Evaluator for GSM8K: last-numeric-token extraction + math.isclose(abs_tol=1e-6)."""

    name: str = "gsm8k_numeric"

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against a GSM8K ``spec``."""
        tokens = _NUMERIC_TOKEN_RE.findall(answer)
        if not tokens:
            return EvalResult(
                score=0.0,
                passed=False,
                error="no number found",
            )

        last_token = tokens[-1].replace(",", "")
        try:
            actual = float(last_token)
        except ValueError:
            return EvalResult(
                score=0.0,
                passed=False,
                error="no number found",
            )

        expected_val = float(spec.expected)  # type: ignore[arg-type]
        passed = math.isclose(actual, expected_val, abs_tol=1e-6)
        score = 1.0 if passed else 0.0

        return EvalResult(
            score=score,
            passed=passed,
            details={"actual": actual, "expected": expected_val},
        )
