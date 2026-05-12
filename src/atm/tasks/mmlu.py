"""MMLU-Pro task loader and evaluator for the ATM benchmark suite.

Implements:
- ``MMLULoader``    -- loads MMLU-Pro tasks from HuggingFace Datasets with Parquet cache
- ``MMLUEvaluator`` -- exact-match evaluator that extracts A-J letters via regex

Public API:
    MMLULoader(limit=500)   -- registered under name "mmlu"
    MMLUEvaluator()         -- registered under name "mmlu_exact_match"

Cache behaviour:
    On first ``load()``, fetches from HuggingFace with
    ``datasets.load_dataset("TIGER-Lab/MMLU-Pro", split=f"test[:{limit}]")``,
    writes to Parquet cache under key ``f"mmlu_limit{limit}"``.
    Subsequent calls read from Parquet; network is never called again.

Evaluator normalisation:
    Extraction priority:
    1. ``re.search(r'\\(\\s*([A-Ja-j])\\s*\\)', answer)`` — parenthesised "(A)"
    2. ``re.search(r'\\b([A-Ja-j])\\b', answer)``          — bare letter
    The matched character is uppercased before comparison with ``spec.expected``.
    If no letter is found, returns score=0.0 with an error field.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import datasets  # type: ignore[import-untyped]

from atm.core.types import TaskSpec
from atm.tasks._cache import is_cached, read_cache, write_cache
from atm.tasks.base import EVALUATORS, TASKS, EvalResult, Evaluator, TaskLoader

__all__ = ["MMLUEvaluator", "MMLULoader"]

# ---------------------------------------------------------------------------
# MMLULoader
# ---------------------------------------------------------------------------

_DATASET_NAME = "TIGER-Lab/MMLU-Pro"


@TASKS.register
class MMLULoader:
    """Load MMLU-Pro tasks from HuggingFace Datasets with Parquet caching.

    Args:
        limit: Maximum number of test rows to fetch from the dataset.
               Defaults to 500.  The cache key embeds ``limit`` so that
               different limits produce independent cache files.
    """

    name = "mmlu"

    def __init__(self, *, limit: int = 500) -> None:
        self._limit = limit

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Return MMLU-Pro tasks as a list of TaskSpec.

        On the first call, fetches from HuggingFace and writes a Parquet cache.
        On subsequent calls with the same ``cache_dir``, reads from cache without
        touching the network.

        Args:
            cache_dir: Directory for Parquet cache files.
                       Defaults to ``data/cache/tasks/``.

        Returns:
            List of TaskSpec instances, one per MMLU-Pro row.
        """
        cache_key = f"mmlu_limit{self._limit}"

        if is_cached(cache_key, cache_dir):
            cached_rows = read_cache(cache_key, cache_dir)
            return [self._row_to_spec(row) for row in cached_rows]

        # Fetch from HuggingFace
        split = f"test[:{self._limit}]"
        dataset = datasets.load_dataset(_DATASET_NAME, split=split)

        fetched_rows: list[dict[str, Any]] = list(dataset)
        write_cache(cache_key, fetched_rows, cache_dir=cache_dir)

        return [self._row_to_spec(row) for row in fetched_rows]

    @staticmethod
    def _row_to_spec(row: dict[str, Any]) -> TaskSpec:
        """Convert a single dataset row dict to a TaskSpec."""
        question_id: str = str(row["question_id"])
        question: str = str(row["question"])
        options: list[str] = list(row["options"])
        answer: str = str(row["answer"])
        answer_index: int = int(row["answer_index"])
        category: str = str(row["category"])

        # Format input as question + lettered option list
        option_lines = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )
        task_input = f"{question}\n\n{option_lines}"

        return TaskSpec(
            id=question_id,
            type="qa",
            input=task_input,
            expected=answer,
            evaluator_key="mmlu_exact_match",
            metadata={
                "category": category,
                "answer_index": answer_index,
                "options": options,
            },
        )


# ---------------------------------------------------------------------------
# MMLUEvaluator
# ---------------------------------------------------------------------------

# Regex to extract letter from parenthesised form: "(A)" or "( a )"
_PAREN_LETTER_RE = re.compile(r"\(\s*([A-Ja-j])\s*\)")
# Regex to extract bare letter as a whole word (word-boundary anchored), case-insensitive
_BARE_LETTER_RE = re.compile(r"\b([A-Ja-j])\b")


@EVALUATORS.register
class MMLUEvaluator:
    """Exact-match evaluator for MMLU-Pro multiple-choice tasks.

    Extracts the first A-J letter (case-insensitive) from the model's answer
    using a two-step regex strategy, then compares the uppercased result to
    ``spec.expected``.

    Extraction priority:
    1. Parenthesised form ``(A)`` — ``re.search(r'\\(\\s*([A-Ja-j])\\s*\\)', text)``
    2. Bare word-boundary letter — ``re.search(r'\\b([A-Ja-j])\\b', text)``

    Registered under name ``"mmlu_exact_match"``.
    """

    name = "mmlu_exact_match"

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate the model's ``answer`` against the expected choice letter.

        Args:
            spec:      TaskSpec whose ``expected`` field holds the correct letter (A-J).
            answer:    Raw model response string (may contain prose around the letter).
            artifacts: Unused. Present for ``Evaluator`` Protocol compatibility.

        Returns:
            EvalResult with:
            - ``score``: 1.0 if correct, 0.0 otherwise.
            - ``passed``: True if correct.
            - ``details``: {"normalised": extracted_letter, "raw": answer[:200]}
            - ``error``: "no letter found" if regex found nothing; None otherwise.
        """
        text = answer.strip()

        # Priority 1: parenthesised form "(A)" / "(a)"
        match = _PAREN_LETTER_RE.search(text)
        if match is None:
            # Priority 2: bare word-boundary letter
            match = _BARE_LETTER_RE.search(text)

        if match is None:
            return EvalResult(
                score=0.0,
                passed=False,
                details={"normalised": None, "raw": text[:200]},
                error="no letter found",
            )

        normalised: str = match.group(1).upper()
        passed: bool = normalised == spec.expected
        score: float = 1.0 if passed else 0.0

        return EvalResult(
            score=score,
            passed=passed,
            details={"normalised": normalised, "raw": text[:200]},
            error=None,
        )


# ---------------------------------------------------------------------------
# Protocol conformance assertions (checked at import time in --strict mypy)
# ---------------------------------------------------------------------------

def _assert_protocols() -> None:
    """Verify MMLULoader and MMLUEvaluator satisfy their Protocols at import time."""
    loader: TaskLoader = MMLULoader()
    evaluator: Evaluator = MMLUEvaluator()
    _ = (loader, evaluator)  # suppress unused-variable warnings


_assert_protocols()
