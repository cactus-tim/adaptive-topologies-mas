"""GSM8K task loader and numeric-match evaluator for the ATM tasks module.

Public API:
- ``_extract_final_number(text: str) -> str | None``  — extract #### N from answer
- ``GSM8KLoader``                                      — TaskLoader for openai/gsm8k
- ``GSM8KMatcher``                                     — Evaluator: last-number regex + math.isclose
"""

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches "#### <number>" at the end of a GSM8K answer string.
# Group 1: the raw number string (may include commas, optional decimal part).
_FINAL_NUMBER_RE = re.compile(r"####\s*(-?[0-9][\d,]*(?:\.\d+)?)")

# Matches any numeric token (with optional sign, commas, decimals) in free text.
_NUMERIC_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _extract_final_number(text: str) -> str | None:
    """Extract the canonical answer from a GSM8K-style answer string.

    GSM8K answers encode the final numeric result after ``####``, for example::

        "She made 9 * 2 = 18 dollars.\\n#### 18"

    The extracted string has commas stripped so it can be passed to ``float()``.

    Args:
        text: Full answer text from a GSM8K row.

    Returns:
        The final number as a plain string (no commas), or ``None`` if not found.
    """
    match = _FINAL_NUMBER_RE.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", "")


# ---------------------------------------------------------------------------
# GSM8KLoader
# ---------------------------------------------------------------------------

_CACHE_KEY = "gsm8k"
_HF_DATASET = "openai/gsm8k"
_HF_CONFIG = "main"
_HF_SPLIT = "test"


@TASKS.register
class GSM8KLoader:
    """Task loader for the GSM8K math-reasoning benchmark.

    Loads from ``openai/gsm8k`` (HuggingFace) and caches to Parquet to avoid
    repeated network calls.  On cache miss, ``datasets.load_dataset`` is called;
    on hit, the local file is read directly.

    Each row is converted to a ``TaskSpec`` with:
    - ``type = "reasoning"``
    - ``evaluator_key = "gsm8k_numeric"``
    - ``id = f"gsm8k/{idx}"`` (zero-based index in dataset order)
    - ``expected``: numeric string extracted via ``_extract_final_number``
    """

    name: str = "gsm8k"

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Load GSM8K tasks, hitting Parquet cache when available.

        Args:
            cache_dir: Override for the default Parquet cache directory.

        Returns:
            List of ``TaskSpec`` instances, one per GSM8K problem.
        """
        if is_cached(_CACHE_KEY, cache_dir):
            rows: list[dict[str, Any]] = read_cache(_CACHE_KEY, cache_dir)
        else:
            split = datasets.load_dataset(_HF_DATASET, _HF_CONFIG, split=_HF_SPLIT)
            rows = []
            for idx, row in enumerate(split):
                expected = _extract_final_number(row["answer"])
                if expected is None:
                    # Skip malformed rows that have no "####" marker.
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


# ---------------------------------------------------------------------------
# GSM8KMatcher
# ---------------------------------------------------------------------------


@EVALUATORS.register
class GSM8KMatcher:
    """Evaluator for GSM8K tasks using last-numeric-token matching.

    Strategy:
    1. Find the last numeric token in the model's answer using a permissive regex.
    2. Strip commas and convert to ``float``.
    3. Compare to ``float(spec.expected)`` via ``math.isclose(abs_tol=1e-6)``.

    If no numeric token is found the result is ``passed=False`` with
    ``error="no number found"``.
    """

    name: str = "gsm8k_numeric"

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against a GSM8K ``spec``.

        Args:
            spec:      The ``TaskSpec`` from ``GSM8KLoader`` (``expected`` holds
                       the canonical numeric string).
            answer:    Free-text model answer — the last numeric token is extracted.
            artifacts: Unused. Present for ``Evaluator`` Protocol compatibility.

        Returns:
            ``EvalResult`` with score 1.0/passed=True on match, 0.0/False otherwise.
        """
        # Extract all numeric tokens from the answer, take the last one.
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
