"""CommonGen task loader, in-house ROUGE-L, and concept-coverage evaluator."""

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

_CACHE_KEY = "commongen"
_HF_DATASET = "allenai/common_gen"
_HF_SPLIT = "validation"

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return text.lower().translate(_PUNCT_TABLE).split()


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Return LCS length via standard DP; O(len(a)*len(b)) time, O(len(b)) space."""
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
    """Return ROUGE-L F1 in [0.0, 1.0]; 0.0 if either input is empty after tokenisation."""
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


@TASKS.register
class CommonGenLoader:
    """Task loader for the CommonGen benchmark (allenai/common_gen, validation split)."""

    name: str = "commongen"

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Load CommonGen tasks; uses Parquet cache when available."""
        if is_cached(_CACHE_KEY, cache_dir):
            rows: list[dict[str, Any]] = read_cache(_CACHE_KEY, cache_dir)
            return [_row_to_spec(row) for row in rows]

        split = datasets.load_dataset(_HF_DATASET, split=_HF_SPLIT)

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


@EVALUATORS.register
class CommonGenEvaluator:
    """Evaluator for CommonGen: score = 0.5*ROUGE-L + 0.5*concept-coverage; passed ≥ 0.5."""

    name: str = "commongen_rouge_coverage"

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against a CommonGen ``spec``."""
        meta: dict[str, Any] = spec.metadata or {}
        concepts: list[str] = list(meta.get("concepts", []))
        references: list[str] = list(meta.get("references", []))

        score_rouge = 0.0
        if references:
            score_rouge = max(_rouge_l(answer, ref) for ref in references)

        if not concepts:
            score_coverage = 1.0
        else:
            matched = sum(1 for c in concepts if c.lower() in answer.lower())
            score_coverage = matched / len(concepts)

        final_score = max(0.0, min(1.0, 0.5 * score_rouge + 0.5 * score_coverage))

        passed = final_score >= 0.5

        return EvalResult(
            score=final_score,
            passed=passed,
            details={"rouge_l": score_rouge, "coverage": score_coverage},
        )
