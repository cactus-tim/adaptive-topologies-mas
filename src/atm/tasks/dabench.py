"""DABench task loader and numeric-exact evaluator for the ATM tasks module.

Public API:
- ``_serialise_common_answers(pairs)``  — sort pairs by name, format as "@name[value] ..."
- ``_load_curated_jsonl()``             — read curated fallback JSONL from tests/fixtures
- ``DABenchLoader``                     — TaskLoader for InfiAgent-DABench (pinned SHA)
- ``DABenchEvaluator``                  — Evaluator: regex template + numeric-exact compare

Design rationale — arch/experiment_plan.md:12 ("numeric exact"):
    The DABench evaluator uses a regex template format ``@name[value]`` to encode
    expected numeric answers in ``TaskSpec.expected``.  For example:
        ``"@mean_fare[34.65] @survival_rate[0.63]"``
    The evaluator parses this string, extracts the expected values, searches for
    matching ``@name[...]`` templates in the model's answer, and compares numerically
    using ``math.isclose(abs_tol=1e-2)`` (corresponding to "Round to 2 decimals").

    LLM-judge is intentionally NOT used for DABench — per arch/experiment_plan.md §0.
    Numeric answers are self-contained and do not require semantic judgement.

Template format:
    ``@<name>[<value>]`` where:
    - ``name``  matches ``[A-Za-z_][\\w]*``
    - ``value`` is any non-``]`` string (numeric or categorical)
    Multiple pairs in ``expected`` are space-separated and sorted by ``name``
    for determinism across repeated dataset loads.

Categorical fallback (N2):
    If both ``expected_value`` and ``extracted_value`` cannot be parsed as float,
    comparison falls back to case-insensitive string equality:
        ``"YES"`` matches ``"yes"``, ``"No"`` matches ``"NO"``
    This handles non-numeric columns in DABench queries.

Offline mode:
    Set ``ATM_DABENCH_OFFLINE=1`` to skip remote fetch and use the curated fixture
    at ``tests/fixtures/tasks/dabench_curated.jsonl``.  CI always runs offline.
    On network failure, the loader falls back to curated automatically.

Pinned dataset:
    SHA: 6ad4a487a3968682cdcbb9ae24664e680f8981a6
    Source: https://github.com/InfiAgent/InfiAgent/tree/...
"""

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

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_DABENCH_COMMIT_SHA = "6ad4a487a3968682cdcbb9ae24664e680f8981a6"
_DABENCH_QUESTIONS_URL = (
    f"https://raw.githubusercontent.com/InfiAgent/InfiAgent/"
    f"{_DABENCH_COMMIT_SHA}/examples/DA-Agent/data/da-dev-questions.jsonl"
)
_DABENCH_LABELS_URL = (
    f"https://raw.githubusercontent.com/InfiAgent/InfiAgent/"
    f"{_DABENCH_COMMIT_SHA}/examples/DA-Agent/data/da-dev-labels.jsonl"
)

# Curated fallback: lives in tests/fixtures/ for offline CI determinism.
# Path resolves relative to this source file: src/atm/tasks/dabench.py
#   → src/atm/tasks/ → src/atm/ → src/ → project_root/ → tests/fixtures/tasks/
_CURATED_FIXTURE = (
    Path(__file__).parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "tasks"
    / "dabench_curated.jsonl"
)

_CACHE_KEY = "dabench"

# URL template for downloading a DABench CSV table from the pinned GitHub commit.
# Caller must format with sha=_DABENCH_COMMIT_SHA and file_name=<table filename>.
_DABENCH_TABLES_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/InfiAgent/InfiAgent/"
    "{sha}/examples/DA-Agent/data/da-dev-data/{file_name}"
)

# Default byte cache location for CSV tables.
# CWD-relative — resolves to <project_root>/data/cache/dabench_tables/ when
# the process is started from the project root (the normal invocation pattern).
_DEFAULT_TABLES_CACHE: Path = Path("data") / "cache" / "dabench_tables"

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Workspace staging
# ---------------------------------------------------------------------------


def stage_workspace_for(
    spec: TaskSpec,
    workspace_path: Path,
    cache_dir: Path | None = None,
) -> list[Path]:
    """Download and stage the CSV table referenced by a DABench ``spec``.

    For non-DABench specs or specs without a ``file_name`` metadata key this
    function is a no-op and returns an empty list.  This preserves back-compat
    with HumanEval, GSM8K, and CommonGen which do not carry ``file_name``.

    Download behaviour:
    1. If ``ATM_DABENCH_OFFLINE=1`` is set → return ``[]`` (mirrors
       ``DABenchLoader.load`` offline short-circuit).
    2. Cache hit (``cache_dir / file_name`` exists) → copy straight to workspace.
    3. Cache miss → fetch from ``_DABENCH_TABLES_URL_TEMPLATE``, write
       atomically via a ``.tmp`` intermediate, copy to workspace.
    4. On any ``(URLError, OSError, TimeoutError)`` → log a warning and return
       ``[]`` so the calling run path degrades gracefully.

    Args:
        spec:           The ``TaskSpec`` to stage tables for.
        workspace_path: Per-run tools workspace directory (must already exist
                        or be creatable).
        cache_dir:      Override for the default byte-cache directory.
                        Defaults to ``_DEFAULT_TABLES_CACHE``.

    Returns:
        A list containing the ``Path`` of the staged file inside
        ``workspace_path``, or ``[]`` if staging was skipped or failed.
    """
    # Guard 1: only handle dabench specs
    if not spec.id.startswith("dabench/"):
        return []

    # Guard 2: file_name must be present and non-empty
    file_name: str = spec.metadata.get("file_name", "") or ""
    if not file_name:
        return []

    # Guard 3: validate file_name against path traversal and injection
    # Whitelist: only allow safe filename characters (no slashes, dots-dot, etc.)
    if not re.match(r"^[A-Za-z0-9._-]+$", file_name):
        _log.warning(
            "rejecting file_name with disallowed characters",
            file_name=file_name,
            spec_id=spec.id,
        )
        return []
    # Reject absolute paths
    if Path(file_name).is_absolute():
        _log.warning(
            "rejecting absolute file_name",
            file_name=file_name,
            spec_id=spec.id,
        )
        return []
    # Reject path traversal sequences
    if ".." in Path(file_name).parts:
        _log.warning(
            "rejecting file_name containing path traversal",
            file_name=file_name,
            spec_id=spec.id,
        )
        return []
    # Reject filenames that are too long
    if len(file_name) > 255:
        _log.warning(
            "rejecting file_name exceeding maximum length",
            file_name=file_name[:40],
            spec_id=spec.id,
        )
        return []

    # Guard 4: offline mode short-circuit
    if os.getenv("ATM_DABENCH_OFFLINE", "0") == "1":
        return []

    resolved_cache_dir: Path = cache_dir if cache_dir is not None else _DEFAULT_TABLES_CACHE
    cached_file = resolved_cache_dir / file_name

    try:
        if not cached_file.exists():
            # Download to a sibling .tmp file first, then atomically rename
            resolved_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = resolved_cache_dir / f"{file_name}.tmp"

            url = _DABENCH_TABLES_URL_TEMPLATE.format(
                sha=_DABENCH_COMMIT_SHA,
                file_name=file_name,
            )
            with urlopen(url, timeout=30) as response:
                data: bytes = response.read()

            tmp_path.write_bytes(data)
            os.replace(tmp_path, cached_file)

        # Copy from cache to workspace
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


# ---------------------------------------------------------------------------
# Regex for parsing @name[value] pairs
# ---------------------------------------------------------------------------

_PAIR_RE = re.compile(r"@([A-Za-z_][\w]*)\[([^\]]+)\]")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialise_common_answers(pairs: list[tuple[str, str]]) -> str:
    """Serialise a list of (name, value) tuples into a sorted @name[value] string.

    Pairs are sorted by name (alphabetically) to ensure determinism across
    repeated dataset loads and different row orderings.

    Args:
        pairs: List of (metric_name, value_string) tuples from DABench labels.

    Returns:
        Space-joined string like ``"@a_metric[1.0] @b_metric[2.5]"``.
        Returns ``""`` for an empty list.

    Example::

        >>> _serialise_common_answers([("b", "2.5"), ("a", "1.0")])
        '@a[1.0] @b[2.5]'
    """
    return " ".join(f"@{name}[{value}]" for name, value in sorted(pairs, key=lambda p: p[0]))


def _load_curated_jsonl() -> list[TaskSpec]:
    """Read the curated JSONL fixture and return a list of TaskSpec.

    Each line in ``dabench_curated.jsonl`` is a JSON object in post-join form
    (i.e., already has the ``expected`` string in ``@name[value]`` format).

    Returns:
        List of ``TaskSpec`` instances from the fixture file.

    Raises:
        FileNotFoundError: if the fixture file is missing (should not happen
            in normal operation since it is tracked in the repository).
    """
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


# ---------------------------------------------------------------------------
# DABenchLoader
# ---------------------------------------------------------------------------


@TASKS.register
class DABenchLoader:
    """Task loader for the InfiAgent-DABench data-analysis benchmark.

    Loads from pinned GitHub raw JSONL (questions + labels), joins on ``id``,
    serialises ``common_answers`` as ``@name[value]`` pairs sorted by name.
    Falls back to curated fixture on network errors or when
    ``ATM_DABENCH_OFFLINE=1``.

    Each TaskSpec:
    - ``type = "decision"``
    - ``evaluator_key = "dabench_numeric_exact"``
    - ``id = f"dabench/{question_id}"``
    - ``expected`` = ``@name1[v1] @name2[v2] ...`` (sorted by name)
    - ``metadata`` = {concepts, constraints, format, level, file_name}
    """

    name: str = "dabench"

    def load(self, cache_dir: Path | None = None) -> list[TaskSpec]:
        """Load DABench tasks, with cache, offline mode, and network fallback.

        Load priority:
        1. Parquet cache hit → return cached specs immediately.
        2. ``ATM_DABENCH_OFFLINE=1`` env var → return curated JSONL.
        3. Remote fetch via ``datasets.load_dataset`` + join on id.
        4. On ``(OSError, ConnectionError, URLError, RuntimeError)`` →
           log warning and return curated JSONL.

        Args:
            cache_dir: Override for the default Parquet cache directory.

        Returns:
            List of ``TaskSpec`` instances.
        """
        # 1. Cache hit
        if is_cached(_CACHE_KEY, cache_dir):
            rows: list[dict[str, Any]] = read_cache(_CACHE_KEY, cache_dir)
            return [_cached_row_to_spec(row) for row in rows]

        # 2. Offline mode
        if os.getenv("ATM_DABENCH_OFFLINE", "0") == "1":
            return _load_curated_jsonl()

        # 3. Try remote fetch
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

            # Build label index keyed by id
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

                # common_answers: list of [name, value] pairs
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


# ---------------------------------------------------------------------------
# DABenchEvaluator
# ---------------------------------------------------------------------------


@EVALUATORS.register
class DABenchEvaluator:
    """Evaluator for DABench using regex template matching + numeric-exact comparison.

    Strategy:
    1. Parse expected ``@name[value]`` pairs via ``_PAIR_RE``.
    2. If no pairs found → vacuous: score=1.0, passed=True.
    3. For each ``(name, exp_val)`` pair:
       a. Build per-pair regex ``@<name>[<extracted>]`` and search in answer.
       b. If no match → pair wrong.
       c. Try ``math.isclose(float(extracted), float(exp_val), abs_tol=1e-2)``.
       d. On ``ValueError`` (non-numeric) → case-insensitive string compare (N2).
    4. ``score = correct / total``; ``passed = score == 1.0``.

    Tolerance:
        ``abs_tol=1e-2`` corresponds to "Round to 2 decimal places" typical in
        DABench constraints.  Dynamic tolerance from ``constraints`` is M11+.

    LLM-judge:
        NOT used — per arch/experiment_plan.md §0.
    """

    name: str = "dabench_numeric_exact"

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against a DABench ``spec``.

        Args:
            spec:      The ``TaskSpec`` from ``DABenchLoader``.  ``expected`` holds
                       the serialised ``@name[value]`` string.
            answer:    Free-text model answer containing ``@name[value]`` templates.
            artifacts: Unused.  Present for ``Evaluator`` Protocol compatibility.

        Returns:
            ``EvalResult`` with per-pair details.
        """
        expected_str = spec.expected or ""
        expected_pairs = _PAIR_RE.findall(expected_str)  # list of (name, value)

        # Vacuous: no expected pairs → trivially correct
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
            # Search for @name[...] in the answer
            per_re = re.compile(r"@" + re.escape(name) + r"\[([^\]]+)\]")
            match = per_re.search(answer)

            if match is None:
                per_pair.append(
                    {"name": name, "expected": exp_val, "extracted": None, "correct": False}
                )
                continue

            extracted = match.group(1)

            # Numeric comparison with abs_tol=1e-2
            pair_correct: bool
            try:
                pair_correct = math.isclose(float(extracted), float(exp_val), abs_tol=1e-2)
            except ValueError:
                # Categorical fallback: case-insensitive string comparison
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
