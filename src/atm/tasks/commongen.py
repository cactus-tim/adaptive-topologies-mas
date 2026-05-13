"""CommonGen task loader, in-house ROUGE-L, and concept-coverage evaluator.

Public API:
- ``_rouge_l(pred, ref)``         — in-house ROUGE-L F1 (no external dependency)
- ``CommonGenLoader``             — TaskLoader for allenai/common_gen
- ``CommonGenEvaluator``          — Evaluator: ROUGE-L + concept substring coverage

Design rationale — in-house ROUGE-L:
    We implement a 25-line LCS-based ROUGE-L instead of adding ``rouge-score`` (~5 MB
    with transitive ``absl-py`` + ``nltk`` punkt data).  CommonGen sentences are
    short (5-15 tokens); whitespace tokenisation is sufficient. Results are
    deterministic and require no NLTK data download.

Design rationale — substring concept coverage:
    Concepts in the allenai/common_gen dataset are root/base forms ("ski", "ride",
    "bake").  Substring matching (``concept.lower() in answer.lower()``) accepts
    morphological variants such as "skiing" for "ski" and "riding" for "ride".
    This matches the CommonGen-Lite evaluation baseline.  Strict word-boundary
    matching would unfairly penalise grammatically correct answers.

ROUGE-L formula:
    For token lists P (pred) and R (ref):
        lcs_len = |LCS(P, R)|
        precision = lcs_len / len(P)
        recall    = lcs_len / len(R)
        F1        = 2 * precision * recall / (precision + recall)
    Edge cases: empty pred or empty ref → F1 = 0.0.

Dataset split:
    We load ``allenai/common_gen`` split ``validation`` (NOT ``test``).
    The ``test`` split in this dataset has empty ``target`` fields —
    only ``validation`` contains reference sentences.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any

import datasets  # type: ignore[import-untyped]

from atm.core.types import TaskSpec
from atm.tasks._cache import is_cached, read_cache, write_cache
from atm.tasks.base import EVALUATORS, TASKS, EvalResult

__all__ = [
    "CommonGenEvaluator",
    "CommonGenLoader",
    "_rouge_l",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_KEY = "commongen"
_HF_DATASET = "allenai/common_gen"
_HF_SPLIT = "validation"

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# In-house ROUGE-L
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return text.lower().translate(_PUNCT_TABLE).split()


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the Longest Common Subsequence of ``a`` and ``b``.

    Standard DP algorithm: O(len(a) * len(b)) time and O(len(b)) space
    (single-row rolling array).
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


def _rouge_l(pred: str, ref: str) -> float:
    """Return the ROUGE-L F1 score between ``pred`` and ``ref``.

    Tokenisation: lowercase + strip ``string.punctuation`` + whitespace split.
    LCS computed via standard DP.

    Args:
        pred: Model-generated text.
        ref:  Reference text.

    Returns:
        F1 score in [0.0, 1.0].  Returns 0.0 if either input is empty
        (or all-punctuation) after tokenisation.
    """
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0.0:
        return 0.0

    f1 = 2.0 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# CommonGenLoader
# ---------------------------------------------------------------------------


@TASKS.register
class CommonGenLoader:
    """Task loader for the CommonGen creative-generation benchmark.

    Loads from ``allenai/common_gen`` (HuggingFace), split ``validation``
    (the ``test`` split has empty targets).  Rows are grouped by
    ``concept_set_idx`` so that multiple reference sentences for the same
    concept set are aggregated into a single ``TaskSpec``.

    Each TaskSpec:
    - ``type = "creative"``
    - ``evaluator_key = "commongen_rouge_coverage"``
    - ``id = f"commongen/{concept_set_idx}"``
    - ``input = "concepts: <c1>, <c2>, ..."``
    - ``expected`` = first reference (deterministic: min row-id order)
    - ``metadata["concepts"]`` = list of concept strings
    - ``metadata["references"]`` = all references for this concept set
    """

    name: str = "commongen"

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Load CommonGen tasks, hitting Parquet cache when available.

        Args:
            cache_dir: Override for the default Parquet cache directory.

        Returns:
            List of ``TaskSpec`` instances, one per concept set.
        """
        if is_cached(_CACHE_KEY, cache_dir):
            rows: list[dict[str, Any]] = read_cache(_CACHE_KEY, cache_dir)
            return [_row_to_spec(row) for row in rows]

        split = datasets.load_dataset(_HF_DATASET, split=_HF_SPLIT)

        # Group rows by concept_set_idx, preserving insertion order for determinism
        groups: dict[int, dict[str, Any]] = {}
        for row in split:
            idx: int = row["concept_set_idx"]
            if idx not in groups:
                groups[idx] = {
                    "idx": idx,
                    "concepts": list(row["concepts"]),
                    "references": [],
                }
            groups[idx]["references"].append(row["target"])

        # Build cache rows (one per concept_set_idx, in ascending idx order)
        cache_rows: list[dict[str, Any]] = []
        for idx in sorted(groups.keys()):
            group = groups[idx]
            cache_rows.append(
                {
                    "idx": group["idx"],
                    "concepts": group["concepts"],
                    "references": group["references"],
                    "input": "concepts: " + ", ".join(group["concepts"]),
                    "expected": group["references"][0],
                }
            )

        write_cache(_CACHE_KEY, cache_rows, cache_dir=cache_dir, dataset_revision=None)
        return [_row_to_spec(row) for row in cache_rows]


def _row_to_spec(row: dict[str, Any]) -> TaskSpec:
    """Convert a cached CommonGen row dict into a ``TaskSpec``."""
    return TaskSpec(
        id=f"commongen/{row['idx']}",
        type="creative",
        input=row["input"],
        expected=row["expected"],
        evaluator_key="commongen_rouge_coverage",
        metadata={
            "concepts": list(row["concepts"]),
            "references": list(row["references"]),
        },
    )


# ---------------------------------------------------------------------------
# CommonGenEvaluator
# ---------------------------------------------------------------------------


@EVALUATORS.register
class CommonGenEvaluator:
    """Evaluator for CommonGen using ROUGE-L + concept substring coverage.

    Scoring formula:
        score_rouge    = max(_rouge_l(answer, ref) for ref in references)
                         (0.0 if no references)
        score_coverage = # of concepts that appear as substrings in answer /
                         len(concepts)
                         (1.0 if concepts is empty — vacuously satisfied)
        final_score    = 0.5 * score_rouge + 0.5 * score_coverage
        passed         = final_score >= 0.5

    Concept matching uses substring (``concept.lower() in answer.lower()``)
    to accept morphological variants: "ski" matches "skiing", "ride" matches
    "riding".  See module docstring for rationale.
    """

    name: str = "commongen_rouge_coverage"

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against a CommonGen ``spec``.

        Args:
            spec:      TaskSpec from ``CommonGenLoader``; must have
                       ``metadata["concepts"]`` (list[str]) and
                       ``metadata["references"]`` (list[str]).
            answer:    Model-generated text.
            artifacts: Unused. Present for ``Evaluator`` Protocol compatibility.

        Returns:
            ``EvalResult`` with score = 0.5*ROUGE-L + 0.5*coverage.
        """
        meta: dict[str, Any] = spec.metadata or {}
        concepts: list[str] = list(meta.get("concepts", []))
        references: list[str] = list(meta.get("references", []))

        # ROUGE-L: max over all references
        score_rouge = 0.0
        if references:
            score_rouge = max(_rouge_l(answer, ref) for ref in references)

        # Concept substring coverage
        if not concepts:
            score_coverage = 1.0
        else:
            matched = sum(1 for c in concepts if c.lower() in answer.lower())
            score_coverage = matched / len(concepts)

        final_score = 0.5 * score_rouge + 0.5 * score_coverage
        # Clamp to [0, 1] as a safety net
        final_score = max(0.0, min(1.0, final_score))

        passed = final_score >= 0.5

        return EvalResult(
            score=final_score,
            passed=passed,
            details={"rouge_l": score_rouge, "coverage": score_coverage},
        )
