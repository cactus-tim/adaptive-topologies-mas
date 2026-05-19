"""DABench task loader and numeric-exact evaluator (regex @name[value] template)."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import datasets  # type: ignore[import-untyped]
import structlog

from atm.core.types import TaskSpec
from atm.tasks._cache import is_cached, read_cache, write_cache
from atm.tasks.base import EVALUATORS, TASKS, EvalResult

__all__ = [
    "DABenchEvaluator",
    "DABenchLoader",
    "_load_curated_jsonl",
    "_serialise_common_answers",
    "stage_workspace_for",
]

_DABENCH_COMMIT_SHA = "6ad4a487a3968682cdcbb9ae24664e680f8981a6"
_DABENCH_QUESTIONS_URL = (
    f"https://raw.githubusercontent.com/InfiAgent/InfiAgent/"
    f"{_DABENCH_COMMIT_SHA}/examples/DA-Agent/data/da-dev-questions.jsonl"
)
_DABENCH_LABELS_URL = (
    f"https://raw.githubusercontent.com/InfiAgent/InfiAgent/"
    f"{_DABENCH_COMMIT_SHA}/examples/DA-Agent/data/da-dev-labels.jsonl"
)

_CURATED_FIXTURE = (
    Path(__file__).parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "tasks"
    / "dabench_curated.jsonl"
)

_CACHE_KEY = "dabench"

_DABENCH_TABLES_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/InfiAgent/InfiAgent/"
    "{sha}/examples/DA-Agent/data/da-dev-tables/{file_name}"
)

_DEFAULT_TABLES_CACHE: Path = Path("data") / "cache" / "dabench_tables"

_log = structlog.get_logger(__name__)


def stage_workspace_for(
    spec: TaskSpec,
    workspace_path: Path,
    cache_dir: Path | None = None,
) -> list[Path]:
    """Download and stage the CSV table referenced by a DABench ``spec``; returns [] on skip/fail."""
    if not spec.id.startswith("dabench/"):
        return []

    file_name: str = spec.metadata.get("file_name", "") or ""
    if not file_name:
        return []

    if any(c in file_name for c in ("/", "\\", "\x00")) or any(ord(c) < 32 for c in file_name):
        _log.warning(
            "rejecting file_name with path separator / null / control chars",
            file_name=file_name,
            spec_id=spec.id,
        )
        return []
    if Path(file_name).is_absolute():
        _log.warning(
            "rejecting absolute file_name",
            file_name=file_name,
            spec_id=spec.id,
        )
        return []
    if ".." in Path(file_name).parts:
        _log.warning(
            "rejecting file_name containing path traversal",
            file_name=file_name,
            spec_id=spec.id,
        )
        return []
    if len(file_name) > 255:
        _log.warning(
            "rejecting file_name exceeding maximum length",
            file_name=file_name[:40],
            spec_id=spec.id,
        )
        return []

    if os.getenv("ATM_DABENCH_OFFLINE", "0") == "1":
        return []

    resolved_cache_dir: Path = cache_dir if cache_dir is not None else _DEFAULT_TABLES_CACHE
    cached_file = resolved_cache_dir / file_name

    try:
        if not cached_file.exists():
            resolved_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = resolved_cache_dir / f"{file_name}.tmp"

            from urllib.parse import quote

            url = _DABENCH_TABLES_URL_TEMPLATE.format(
                sha=_DABENCH_COMMIT_SHA,
                file_name=quote(file_name, safe=""),
            )
            with urlopen(url, timeout=30) as response:
                data: bytes = response.read()

            tmp_path.write_bytes(data)
            os.replace(tmp_path, cached_file)

        workspace_path.mkdir(parents=True, exist_ok=True)
        dest = workspace_path / file_name
        shutil.copy2(cached_file, dest)
        return [dest]

    except (URLError, OSError, TimeoutError) as exc:
        _log.warning(
            "dabench stage_workspace_for failed",
            file_name=file_name,
            spec_id=spec.id,
            error=str(exc),
        )
        return []


_PAIR_RE = re.compile(r"@([A-Za-z_][\w]*)\[([^\]]+)\]")


def _serialise_common_answers(pairs: list[tuple[str, str]]) -> str:
    """Serialise ``(name, value)`` pairs into a sorted ``@name[value]`` string."""
    return " ".join(f"@{name}[{value}]" for name, value in sorted(pairs, key=lambda p: p[0]))


def _load_curated_jsonl() -> list[TaskSpec]:
    """Read the curated JSONL fixture; raises FileNotFoundError if missing."""
    specs: list[TaskSpec] = []
    with _CURATED_FIXTURE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            spec = TaskSpec(
                id=f"dabench/{row['id']}",
                type="decision",
                input=row["question"],
                expected=row["expected"],
                evaluator_key="dabench_numeric_exact",
                metadata={
                    "concepts": row.get("concepts", []),
                    "constraints": row.get("constraints", ""),
                    "format": row.get("format", ""),
                    "level": row.get("level", ""),
                    "file_name": row.get("file_name", ""),
                },
            )
            specs.append(spec)
    return specs


@TASKS.register
class DABenchLoader:
    """Task loader for InfiAgent-DABench; falls back to curated fixture on network failure."""

    name: str = "dabench"

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Load DABench tasks with Parquet cache, offline mode, and remote fallback."""
        if is_cached(_CACHE_KEY, cache_dir):
            rows: list[dict[str, Any]] = read_cache(_CACHE_KEY, cache_dir)
            return [_cached_row_to_spec(row) for row in rows]

        if os.getenv("ATM_DABENCH_OFFLINE", "0") == "1":
            return _load_curated_jsonl()

        try:
            q_ds = datasets.load_dataset(
                "json",
                data_files=_DABENCH_QUESTIONS_URL,
                split="train",
            )
            l_ds = datasets.load_dataset(
                "json",
                data_files=_DABENCH_LABELS_URL,
                split="train",
            )

            label_index: dict[Any, dict[str, Any]] = {}
            for lrow in l_ds:
                label_index[lrow["id"]] = lrow

            specs: list[TaskSpec] = []
            cache_rows: list[dict[str, Any]] = []

            for q in q_ds:
                qid = q["id"]
                label = label_index.get(qid)
                if label is None:
                    continue

                raw_pairs: list[list[str]] = label.get("common_answers", [])
                pairs: list[tuple[str, str]] = [(p[0], p[1]) for p in raw_pairs]
                expected = _serialise_common_answers(pairs)

                concepts: list[str] = list(q.get("concepts", []) or [])
                constraints: str = str(q.get("constraints", "") or "")
                fmt: str = str(q.get("format", "") or "")
                level: str = str(q.get("level", "") or "")
                file_name: str = str(q.get("file_name", "") or "")

                spec = TaskSpec(
                    id=f"dabench/{qid}",
                    type="decision",
                    input=str(q.get("question", "") or ""),
                    expected=expected,
                    evaluator_key="dabench_numeric_exact",
                    metadata={
                        "concepts": concepts,
                        "constraints": constraints,
                        "format": fmt,
                        "level": level,
                        "file_name": file_name,
                    },
                )
                specs.append(spec)

                cache_rows.append(
                    {
                        "id": f"dabench/{qid}",
                        "question": str(q.get("question", "") or ""),
                        "expected": expected,
                        "concepts": concepts,
                        "constraints": constraints,
                        "format": fmt,
                        "level": level,
                        "file_name": file_name,
                    }
                )

            if cache_rows:
                write_cache(
                    _CACHE_KEY,
                    cache_rows,
                    cache_dir=cache_dir,
                    dataset_revision=_DABENCH_COMMIT_SHA,
                )

            return specs

        except (OSError, ConnectionError, URLError, RuntimeError) as exc:
            _log.warning(
                "dabench remote unavailable, falling back to curated",
                error=str(exc),
            )
            return _load_curated_jsonl()


def _cached_row_to_spec(row: dict[str, Any]) -> TaskSpec:
    """Convert a cached DABench row dict into a ``TaskSpec``."""
    concepts = row.get("concepts")
    if concepts is None:
        concepts = []
    return TaskSpec(
        id=str(row["id"]),
        type="decision",
        input=str(row.get("question", "")),
        expected=str(row.get("expected", "")),
        evaluator_key="dabench_numeric_exact",
        metadata={
            "concepts": list(concepts),
            "constraints": str(row.get("constraints", "")),
            "format": str(row.get("format", "")),
            "level": str(row.get("level", "")),
            "file_name": str(row.get("file_name", "")),
        },
    )


@EVALUATORS.register
class DABenchEvaluator:
    """Evaluator for DABench: regex @name[value] extraction + numeric-exact (abs_tol=1e-2)."""

    name: str = "dabench_numeric_exact"

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against a DABench ``spec``."""
        expected_str = spec.expected or ""
        expected_pairs = _PAIR_RE.findall(expected_str)

        if not expected_pairs:
            return EvalResult(
                score=1.0,
                passed=True,
                details={"correct": 0, "total": 0, "per_pair": []},
            )

        total = len(expected_pairs)
        correct = 0
        per_pair: list[dict[str, Any]] = []

        for name, exp_val in expected_pairs:
            per_re = re.compile(r"@" + re.escape(name) + r"\[([^\]]+)\]")
            match = per_re.search(answer)

            if match is None:
                per_pair.append(
                    {"name": name, "expected": exp_val, "extracted": None, "correct": False}
                )
                continue

            extracted = match.group(1)

            pair_correct: bool
            try:
                pair_correct = math.isclose(float(extracted), float(exp_val), abs_tol=1e-2)
            except ValueError:
                pair_correct = extracted.strip().lower() == exp_val.strip().lower()

            if pair_correct:
                correct += 1

            per_pair.append(
                {
                    "name": name,
                    "expected": exp_val,
                    "extracted": extracted,
                    "correct": pair_correct,
                }
            )

        score = correct / total
        passed = score == 1.0

        return EvalResult(
            score=score,
            passed=passed,
            details={"correct": correct, "total": total, "per_pair": per_pair},
        )
